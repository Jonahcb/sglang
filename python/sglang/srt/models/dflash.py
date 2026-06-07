# Adapted from the DFlash reference implementation (HF) but implemented with
# SGLang primitives (RadixAttention + SGLang KV cache). This model intentionally
# does not include token embeddings or an LM head; DFlash uses the target model's
# embedding/lm_head.

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.speculative.dflash_utils import (
    can_dflash_slice_qkv_weight,
    parse_dflash_draft_config,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# DFlash forward-pass debugging (env-gated, additive, no-op when disabled).
#
# Set DFLASH_DEBUG_FILE=/path/to/dump.jsonl to capture per-layer activation and
# (one-time) weight statistics for the draft model forward pass. One JSON object
# is written per line; nothing is ever printed to the terminal.
#
# Debugging is skipped entirely (a) when the env var is unset and (b) while a
# CUDA/HIP graph is being captured -- the stats below do host<->device syncs
# (.item()/float()) which would corrupt graph capture. In CUDA-graph mode the
# Python forward only runs at capture time, so this means graph runs log nothing;
# eager runs (disable_cuda_graph=True) log everything, which is exactly the
# configuration used to debug the all-zeros hidden states.
# --------------------------------------------------------------------------- #
_DFLASH_DBG_LOCK = threading.Lock()
_DFLASH_FWD_COUNTER = 0
_DFLASH_CUR_FWD_ID = -1
_DFLASH_WEIGHTS_DUMPED = False


def _dflash_dbg_path() -> str:
    return os.environ.get("DFLASH_DEBUG_FILE", "").strip()


def _dflash_is_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _dflash_dbg_active() -> bool:
    return bool(_dflash_dbg_path()) and not _dflash_is_capturing()


def _dflash_tensor_stats(t: Optional[torch.Tensor]) -> dict:
    """Summarize a tensor without dumping its full contents."""
    if t is None:
        return {"none": True}
    try:
        td = t.detach()
        numel = int(td.numel())
        out = {"shape": list(td.shape), "dtype": str(td.dtype), "numel": numel}
        if numel == 0:
            return out
        # Cast to float32 so fp8/bf16/half all summarize cleanly.
        f = td.float()
        out.update(
            {
                "mean": float(f.mean()),
                "std": float(f.std()) if numel > 1 else 0.0,
                "min": float(f.min()),
                "max": float(f.max()),
                "absmax": float(f.abs().max()),
                "nan": int(torch.isnan(f).sum()),
                "inf": int(torch.isinf(f).sum()),
                "zero_frac": float((f == 0).float().mean()),
            }
        )
        return out
    except Exception as e:  # never let debugging break the forward pass
        return {"error": repr(e)}


def _dflash_dbg_write(record: dict) -> None:
    path = _dflash_dbg_path()
    if not path:
        return
    try:
        line = json.dumps(record)
        with _DFLASH_DBG_LOCK:
            with open(path, "a") as fh:
                fh.write(line + "\n")
    except Exception as e:
        logger.warning("DFLASH debug write failed: %s", e)


def _dflash_dbg_tensor(tag: str, t: Optional[torch.Tensor]) -> None:
    if not _dflash_dbg_active():
        return
    _dflash_dbg_write(
        {
            "fwd": _DFLASH_CUR_FWD_ID,
            "kind": "act",
            "tag": tag,
            **_dflash_tensor_stats(t),
        }
    )


def _dflash_dbg_weight(tag: str, module: nn.Module) -> None:
    """Dump statistics for the parameters/buffers that matter for a linear/norm."""
    if not _dflash_dbg_active():
        return
    seen: set = set()
    for attr in (
        "weight",
        "weight_scale",
        "weight_scale_inv",
        "input_scale",
        "bias",
    ):
        val = getattr(module, attr, None)
        if val is None:
            continue
        seen.add(attr)
        _dflash_dbg_write(
            {
                "fwd": _DFLASH_CUR_FWD_ID,
                "kind": "weight",
                "tag": f"{tag}.{attr}",
                **_dflash_tensor_stats(val),
            }
        )
    # Sweep ALL named params/buffers for anything scale-related (e.g. an
    # unbound `weight_quantizer.scale` from an unfinalized checkpoint, or a
    # `weight_scale` left at its default because the loader never filled it).
    for name, val in list(module.named_parameters(recurse=True)) + list(
        module.named_buffers(recurse=True)
    ):
        if val is None:
            continue
        if "scale" not in name.lower():
            continue
        if name in seen:
            continue
        seen.add(name)
        _dflash_dbg_write(
            {
                "fwd": _DFLASH_CUR_FWD_ID,
                "kind": "weight",
                "tag": f"{tag}.{name}",
                **_dflash_tensor_stats(val),
            }
        )


class DFlashAttention(nn.Module):
    def __init__(self, config, layer_id: int, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.layer_id = int(layer_id)
        hidden_size = int(config.hidden_size)
        tp_size = int(get_tensor_model_parallel_world_size())
        total_num_heads = int(config.num_attention_heads)
        total_num_kv_heads = int(
            getattr(config, "num_key_value_heads", total_num_heads)
        )
        head_dim = int(getattr(config, "head_dim", hidden_size // total_num_heads))

        self.hidden_size = hidden_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        assert self.total_num_heads % tp_size == 0, (
            f"DFlashAttention requires total_num_heads divisible by tp_size. "
            f"total_num_heads={self.total_num_heads}, tp_size={tp_size}."
        )
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0, (
                f"DFlashAttention requires total_num_kv_heads divisible by tp_size when >= tp_size. "
                f"total_num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
            )
        else:
            assert tp_size % self.total_num_kv_heads == 0, (
                f"DFlashAttention requires tp_size divisible by total_num_kv_heads when total_num_kv_heads < tp_size. "
                f"total_num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
            )
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim

        attention_bias = bool(getattr(config, "attention_bias", False))
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj" if prefix else "qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj" if prefix else "o_proj",
        )

        # Per-head Q/K RMSNorm, matching HF Qwen3.
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        rope_theta = float(getattr(config, "rope_theta", 1000000))
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_is_neox_style = bool(
            getattr(
                config, "rope_is_neox_style", getattr(config, "is_neox_style", True)
            )
        )
        max_position_embeddings = int(getattr(config, "max_position_embeddings", 32768))
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_is_neox_style,
        )

        self.scaling = head_dim**-0.5
        # DFlash uses non-causal attention over the draft block.
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            attn_type=AttentionType.ENCODER_ONLY,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        lid = self.layer_id
        qkv, _ = self.qkv_proj(hidden_states)
        _dflash_dbg_tensor(f"layer{lid}.attn.qkv_proj_out", qkv)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)
        _dflash_dbg_tensor(f"layer{lid}.attn.q_after_norm", q)
        _dflash_dbg_tensor(f"layer{lid}.attn.k_after_norm", k)
        _dflash_dbg_tensor(f"layer{lid}.attn.v", v)
        q, k = self.rotary_emb(positions, q, k)
        _dflash_dbg_tensor(f"layer{lid}.attn.q_after_rope", q)
        _dflash_dbg_tensor(f"layer{lid}.attn.k_after_rope", k)
        attn_output = self.attn(q, k, v, forward_batch)
        _dflash_dbg_tensor(f"layer{lid}.attn.attn_core_out", attn_output)
        output, _ = self.o_proj(attn_output)
        _dflash_dbg_tensor(f"layer{lid}.attn.o_proj_out", output)
        return output

    def kv_proj_only(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project hidden_states to K/V only (skip Q).

        This is used by DFlash to materialize ctx tokens into the draft KV cache:
        we only need K/V for the cached tokens; Q is never consumed.
        """
        # Fast path for unquantized weights: slice the fused QKV weight and run one GEMM.
        can_slice_qkv_weight, _ = can_dflash_slice_qkv_weight(self.qkv_proj)
        if can_slice_qkv_weight:
            kv_slice = slice(self.q_size, self.q_size + 2 * self.kv_size)
            weight = self.qkv_proj.weight[kv_slice]
            bias = (
                self.qkv_proj.bias[kv_slice] if self.qkv_proj.bias is not None else None
            )
            kv = F.linear(hidden_states, weight, bias)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
            return k, v

        # Fallback: compute full QKV and discard Q (keeps compatibility with quantized weights).
        qkv, _ = self.qkv_proj(hidden_states)
        _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return k, v

    def apply_k_norm(self, k: torch.Tensor) -> torch.Tensor:
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        return k_by_head.view_as(k)

    def apply_k_rope(self, positions: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # Match K shape so RoPE kernel head-count check passes on all backends.
        dummy_q = k.new_empty(k.shape)
        _, k = self.rotary_emb(positions, dummy_q, k)
        return k


class DFlashMLP(nn.Module):
    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(getattr(config, "intermediate_size", 0))
        if intermediate_size <= 0:
            raise ValueError(
                f"Invalid intermediate_size={intermediate_size} for DFlash MLP."
            )

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix="gate_up_proj" if not prefix else f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix="down_proj" if not prefix else f"{prefix}.down_proj",
        )
        hidden_act = getattr(config, "hidden_act", "silu")
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported DFlash activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lid = getattr(self, "layer_id", -1)
        _dflash_dbg_tensor(f"layer{lid}.mlp.in", x)
        gate_up, _ = self.gate_up_proj(x)
        _dflash_dbg_tensor(f"layer{lid}.mlp.gate_up_proj_out", gate_up)
        x = self.act_fn(gate_up)
        _dflash_dbg_tensor(f"layer{lid}.mlp.act_out", x)
        x, _ = self.down_proj(x)
        _dflash_dbg_tensor(f"layer{lid}.mlp.down_proj_out", x)
        return x


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config, layer_id: int, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.layer_id = int(layer_id)
        hidden_size = int(config.hidden_size)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = DFlashAttention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn" if prefix else "self_attn",
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = DFlashMLP(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp" if prefix else "mlp",
        )
        # Let the MLP tag its debug records with the owning layer id.
        self.mlp.layer_id = int(layer_id)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lid = self.layer_id
        if hidden_states.numel() == 0:
            # Keep return types consistent for upstream callers.
            if residual is None:
                residual = hidden_states
            _dflash_dbg_tensor(f"layer{lid}.EMPTY_INPUT", hidden_states)
            return hidden_states, residual

        # Pre-norm attention with fused residual+norm when possible (Qwen3-style).
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        _dflash_dbg_tensor(f"layer{lid}.after_input_layernorm", hidden_states)
        _dflash_dbg_tensor(f"layer{lid}.residual_after_input_layernorm", residual)

        attn_out = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        _dflash_dbg_tensor(f"layer{lid}.self_attn_out", attn_out)
        hidden_states, residual = self.post_attention_layernorm(attn_out, residual)
        _dflash_dbg_tensor(f"layer{lid}.after_post_attention_layernorm", hidden_states)
        _dflash_dbg_tensor(f"layer{lid}.residual_after_post_attn_ln", residual)
        hidden_states = self.mlp(hidden_states)
        _dflash_dbg_tensor(f"layer{lid}.mlp_out", hidden_states)
        return hidden_states, residual


class DFlashDraftModel(nn.Module):
    """SGLang DFlash draft model (no embedding / lm_head weights).

    The checkpoint provides:
      - transformer weights for `layers.*`
      - `fc.weight`, `hidden_norm.weight` for projecting target context features
      - `norm.weight` for final normalization
    """

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.config = config

        hidden_size = int(config.hidden_size)
        num_layers = int(config.num_hidden_layers)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.layers = nn.ModuleList(
            [
                DFlashDecoderLayer(
                    config=config,
                    layer_id=i,
                    quant_config=quant_config,
                    prefix=f"layers.{i}",
                )
                for i in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Project per-token target context features:
        # concat(K * hidden_size) -> hidden_size, where K is the number of target-layer
        # feature tensors concatenated per token (not necessarily equal to num_layers).
        draft_config = parse_dflash_draft_config(draft_hf_config=config)
        target_num_layers = (
            int(draft_config.num_target_layers)
            if draft_config.num_target_layers is not None
            else num_layers
        )
        target_layer_ids = draft_config.resolve_target_layer_ids(
            target_num_layers=target_num_layers, draft_num_layers=num_layers
        )
        num_context_features = len(target_layer_ids)

        self.num_context_features = int(num_context_features)
        self.fc = nn.Linear(
            self.num_context_features * hidden_size, hidden_size, bias=False
        )
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        self.block_size = draft_config.resolve_block_size(default=16)

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """Project concatenated target-layer hidden states into draft hidden_size."""
        expected = int(self.fc.in_features)
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected:
            raise ValueError(
                "DFLASH target_hidden feature dim mismatch. "
                f"Expected shape [N, {expected}] "
                f"(num_context_features={self.num_context_features}, hidden_size={int(self.config.hidden_size)}), "
                f"but got shape={tuple(target_hidden.shape)}. "
                "This usually means the target model is capturing a different number of layer features than "
                "the draft checkpoint/config expects."
            )
        return self.hidden_norm(self.fc(target_hidden))

    def _dflash_dump_weights(self) -> None:
        """One-time dump of every parameter that feeds the draft forward pass."""
        try:
            _dflash_dbg_weight("fc", self.fc)
            _dflash_dbg_weight("hidden_norm", self.hidden_norm)
            _dflash_dbg_weight("norm", self.norm)
            for i, layer in enumerate(self.layers):
                _dflash_dbg_weight(f"layer{i}.input_layernorm", layer.input_layernorm)
                _dflash_dbg_weight(
                    f"layer{i}.post_attention_layernorm",
                    layer.post_attention_layernorm,
                )
                _dflash_dbg_weight(
                    f"layer{i}.self_attn.qkv_proj", layer.self_attn.qkv_proj
                )
                _dflash_dbg_weight(f"layer{i}.self_attn.o_proj", layer.self_attn.o_proj)
                _dflash_dbg_weight(f"layer{i}.self_attn.q_norm", layer.self_attn.q_norm)
                _dflash_dbg_weight(f"layer{i}.self_attn.k_norm", layer.self_attn.k_norm)
                _dflash_dbg_weight(f"layer{i}.mlp.gate_up_proj", layer.mlp.gate_up_proj)
                _dflash_dbg_weight(f"layer{i}.mlp.down_proj", layer.mlp.down_proj)
        except Exception as e:
            logger.warning("DFLASH weight dump failed: %s", e)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        global _DFLASH_FWD_COUNTER, _DFLASH_CUR_FWD_ID, _DFLASH_WEIGHTS_DUMPED
        if _dflash_dbg_path():
            _DFLASH_CUR_FWD_ID = _DFLASH_FWD_COUNTER
            _DFLASH_FWD_COUNTER += 1

        if input_embeds is None:
            raise ValueError(
                "DFlashDraftModel requires `input_embeds` (use the target embedding)."
            )

        # Dump static weights once, on the first eager forward we actually log.
        if _dflash_dbg_active() and not _DFLASH_WEIGHTS_DUMPED:
            self._dflash_dump_weights()
            _DFLASH_WEIGHTS_DUMPED = True

        hidden_states = input_embeds
        _dflash_dbg_tensor("forward.input_embeds", hidden_states)
        if input_ids is not None:
            _dflash_dbg_tensor("forward.input_ids", input_ids)
        _dflash_dbg_tensor("forward.positions", positions)
        residual: Optional[torch.Tensor] = None

        for i, layer in enumerate(self.layers):
            _dflash_dbg_tensor(f"layer{i}.IN.hidden", hidden_states)
            _dflash_dbg_tensor(f"layer{i}.IN.residual", residual)
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )
            _dflash_dbg_tensor(f"layer{i}.OUT.hidden", hidden_states)
            _dflash_dbg_tensor(f"layer{i}.OUT.residual", residual)

        if hidden_states.numel() != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)
        _dflash_dbg_tensor("forward.final_norm_out", hidden_states)

        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden_states,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())

        def resolve_param_name(name: str) -> Optional[str]:
            if name in params_dict:
                return name
            if name.startswith("model."):
                stripped_name = name[len("model.") :]
                if stripped_name in params_dict:
                    return stripped_name
            else:
                prefixed_name = f"model.{name}"
                if prefixed_name in params_dict:
                    return prefixed_name
            return None

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                resolved_name = resolve_param_name(mapped_name)
                if resolved_name is None:
                    continue
                param = params_dict[resolved_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                resolved_name = resolve_param_name(name)
                if resolved_name is None:
                    # Ignore unexpected weights (e.g., HF rotary caches).
                    continue
                param = params_dict[resolved_name]
                if resolved_name.endswith("fc.weight") and tuple(
                    loaded_weight.shape
                ) != tuple(param.shape):
                    raise ValueError(
                        "DFLASH fc.weight shape mismatch. This usually means the draft checkpoint's "
                        "number of context features (K) does not match this config. "
                        f"Expected fc.weight.shape={tuple(param.shape)} "
                        f"(num_context_features={self.num_context_features}, hidden_size={int(self.config.hidden_size)}), "
                        f"but got {tuple(loaded_weight.shape)} for weight '{name}'."
                    )
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = DFlashDraftModel
