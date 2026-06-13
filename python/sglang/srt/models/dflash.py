# Adapted from the DFlash reference implementation (HF) but implemented with
# SGLang primitives (RadixAttention + SGLang KV cache). This model intentionally
# does not include token embeddings or an LM head; DFlash uses the target model's
# embedding/lm_head.

from __future__ import annotations

import logging
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
from sglang.srt.speculative.triton_ops.dflash_prepare_block import (
    _prepare_dflash_draft_block_unchecked,
)
from sglang.srt.utils import is_npu
from sglang.srt.utils.hf_transformers_utils import get_rope_config

_is_npu = is_npu()
if _is_npu:
    from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope
logger = logging.getLogger(__name__)


class DFlashAttention(nn.Module):
    def __init__(self, config, layer_id: int) -> None:
        super().__init__()
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
            prefix="qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * head_dim,
            hidden_size,
            bias=attention_bias,
            prefix="o_proj",
        )

        # Per-head Q/K RMSNorm, matching HF Qwen3.
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        rope_theta, rope_scaling = get_rope_config(config)
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

    def forward_prepare_npu(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)

        if self.attn.layer_id == 0:
            self.rotary_emb.get_cos_sin_with_position(positions)
        q, k, v = split_qkv_rmsnorm_rope(
            qkv,
            self.rotary_emb.position_sin,
            self.rotary_emb.position_cos,
            self.q_size,
            self.kv_size,
            self.head_dim,
            eps=self.q_norm.variance_epsilon,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            q_bias=getattr(self.q_norm, "bias", None),
            k_bias=getattr(self.k_norm, "bias", None),
        )
        return q, k, v

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        if _is_npu:
            q, k, v = self.forward_prepare_npu(positions, hidden_states)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
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
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config, layer_id: int) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = DFlashAttention(config=config, layer_id=layer_id)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = DFlashMLP(config=config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.numel() == 0:
            # Keep return types consistent for upstream callers.
            if residual is None:
                residual = hidden_states
            return hidden_states, residual

        # Pre-norm attention with fused residual+norm when possible (Qwen3-style).
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        attn_out = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states, residual = self.post_attention_layernorm(attn_out, residual)
        hidden_states = self.mlp(hidden_states)
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
            [DFlashDecoderLayer(config=config, layer_id=i) for i in range(num_layers)]
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

        # Fused KV-materialization state. Populated by the worker at construction
        # time via `_init_fused_kv_helper`; defaults keep the model usable (and the
        # eager per-layer path active) if fused init is skipped or fails.
        self._use_fused_kv_materialize = False
        self._fused_kv_helper: Optional[object] = None

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

    def append_target_hidden_to_draft_kv_by_loc(
        self,
        *,
        token_to_kv_pool,
        device: torch.device,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
    ) -> None:
        """Materialize target context features into the draft KV cache at explicit slots.

        For the spec-v2 overlap path, callers can pass dense `[bs, block_size]`
        `cache_loc_2d` plus `commit_lens`; the prefix-valid writer then commits
        only the live prefix rows without constructing masked/packed index tensors.
        """
        if target_hidden is None:
            raise RuntimeError("DFLASH missing target hidden context features.")
        if target_hidden.numel() == 0:
            return
        if target_hidden.ndim != 2:
            raise ValueError(
                "DFLASH target_hidden must be 2D, "
                f"got shape={tuple(target_hidden.shape)}."
            )

        if cache_loc.ndim != 1:
            raise ValueError(
                f"DFLASH cache_loc must be 1D, got shape={tuple(cache_loc.shape)}."
            )
        if positions.ndim != 1:
            raise ValueError(
                f"DFLASH positions must be 1D, got shape={tuple(positions.shape)}."
            )
        num_tokens = int(target_hidden.shape[0])
        if int(cache_loc.numel()) != num_tokens:
            raise ValueError(
                "DFLASH cache_loc length mismatch: "
                f"cache_loc={int(cache_loc.numel())}, target_hidden={num_tokens}."
            )
        if int(positions.numel()) != num_tokens:
            raise ValueError(
                "DFLASH positions length mismatch: "
                f"positions={int(positions.numel())}, target_hidden={num_tokens}."
            )
        if cache_loc_2d is not None:
            if cache_loc_2d.ndim != 2:
                raise ValueError(
                    "DFLASH cache_loc_2d must be 2D, "
                    f"got shape={tuple(cache_loc_2d.shape)}."
                )
            if int(cache_loc_2d.numel()) != num_tokens:
                raise ValueError(
                    "DFLASH cache_loc_2d size mismatch: "
                    f"cache_loc_2d={int(cache_loc_2d.numel())}, target_hidden={num_tokens}."
                )
            if commit_lens is None:
                raise ValueError(
                    "DFLASH cache_loc_2d requires commit_lens for prefix-valid writes."
                )

        if cache_loc.device != device:
            cache_loc = cache_loc.to(device, non_blocking=True)
        if positions.device != device:
            positions = positions.to(device, non_blocking=True)
        if target_hidden.device != device:
            target_hidden = target_hidden.to(device, non_blocking=True)

        if cache_loc.dtype != torch.int64:
            cache_loc = cache_loc.to(torch.int64)
        if positions.dtype != torch.int64:
            positions = positions.to(torch.int64)
        if cache_loc_2d is not None:
            if cache_loc_2d.device != device:
                cache_loc_2d = cache_loc_2d.to(device, non_blocking=True)
            if cache_loc_2d.dtype != torch.int64:
                cache_loc_2d = cache_loc_2d.to(torch.int64)
        if commit_lens is not None:
            if commit_lens.device != device:
                commit_lens = commit_lens.to(device, non_blocking=True)
            if commit_lens.dtype != torch.int32:
                commit_lens = commit_lens.to(torch.int32)

        with torch.inference_mode():
            ctx_hidden = self.project_target_hidden(target_hidden)

            if cache_loc_2d is not None:
                bs = int(commit_lens.shape[0])
                if int(cache_loc_2d.shape[0]) != bs:
                    raise ValueError(
                        "DFLASH cache_loc_2d batch size mismatch: "
                        f"cache_loc_2d={tuple(cache_loc_2d.shape)}, commit_lens={tuple(commit_lens.shape)}."
                    )
                if bs == 0:
                    return
                if self._use_fused_kv_materialize and self._fused_kv_helper is not None:
                    try:
                        self._append_target_hidden_fused(
                            token_to_kv_pool=token_to_kv_pool,
                            ctx_hidden=ctx_hidden,
                            ctx_positions=positions,
                            ctx_cache_loc=cache_loc,
                            ctx_cache_loc_2d=cache_loc_2d,
                            commit_lens=commit_lens,
                        )
                        return
                    except Exception as e:
                        logger.warning(
                            "DFLASH fused prefix-direct KV append failed; falling back to the per-layer prefix-direct path: %s",
                            e,
                        )
                        self._use_fused_kv_materialize = False
                        self._fused_kv_helper = None

                for layer in self.layers:
                    attn = layer.self_attn
                    k, v = attn.kv_proj_only(ctx_hidden)
                    k = attn.apply_k_norm(k)
                    k = attn.apply_k_rope(positions, k)
                    k = k.view(-1, attn.num_kv_heads, attn.head_dim)
                    v = v.view(-1, attn.num_kv_heads, attn.head_dim)

                    token_to_kv_pool.set_kv_buffer_prefix_valid(
                        attn.attn,
                        cache_loc_2d,
                        commit_lens,
                        k,
                        v,
                        attn.attn.k_scale,
                        attn.attn.v_scale,
                    )
                return

            if self._use_fused_kv_materialize and self._fused_kv_helper is not None:
                try:
                    self._append_target_hidden_fused(
                        token_to_kv_pool=token_to_kv_pool,
                        ctx_hidden=ctx_hidden,
                        ctx_positions=positions,
                        ctx_cache_loc=cache_loc,
                    )
                    return
                except Exception as e:
                    logger.warning(
                        "DFLASH fused KV append-by-loc failed; falling back to sequential path: %s",
                        e,
                    )
                    self._use_fused_kv_materialize = False
                    self._fused_kv_helper = None

            self._append_target_hidden_sequential(
                token_to_kv_pool=token_to_kv_pool,
                ctx_hidden=ctx_hidden,
                ctx_positions=positions,
                ctx_cache_loc=cache_loc,
            )

    def _append_target_hidden_sequential(
        self,
        token_to_kv_pool,
        ctx_hidden: torch.Tensor,
        ctx_positions: torch.Tensor,
        ctx_cache_loc: torch.Tensor,
    ) -> None:
        for layer in self.layers:
            attn = layer.self_attn
            if _is_npu:
                _, k, v = attn.forward_prepare_npu(ctx_positions, ctx_hidden)
            else:
                k, v = attn.kv_proj_only(ctx_hidden)
                k = attn.apply_k_norm(k)
                k = attn.apply_k_rope(ctx_positions, k)
            k = k.view(-1, attn.num_kv_heads, attn.head_dim)
            v = v.view(-1, attn.num_kv_heads, attn.head_dim)
            token_to_kv_pool.set_kv_buffer(
                attn.attn,
                ctx_cache_loc,
                k,
                v,
                attn.attn.k_scale,
                attn.attn.v_scale,
            )

    def _append_target_hidden_fused(
        self,
        token_to_kv_pool,
        ctx_hidden: torch.Tensor,
        ctx_positions: torch.Tensor,
        ctx_cache_loc: torch.Tensor,
        ctx_cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
    ) -> None:
        """Fused KV materialization using batched projection + Triton kernel."""
        if self._fused_kv_helper is None:
            raise RuntimeError("DFLASH fused KV helper is not initialized.")

        def _write_layer_kv(
            layer_idx: int,
            cache_k: torch.Tensor,
            cache_v: torch.Tensor,
        ) -> None:
            attn = self.layers[layer_idx].self_attn.attn
            if ctx_cache_loc_2d is not None and commit_lens is not None:
                token_to_kv_pool.set_kv_buffer_prefix_valid(
                    attn,
                    ctx_cache_loc_2d,
                    commit_lens,
                    cache_k,
                    cache_v,
                    attn.k_scale,
                    attn.v_scale,
                )
            else:
                token_to_kv_pool.set_kv_buffer(
                    attn,
                    ctx_cache_loc,
                    cache_k,
                    cache_v,
                    attn.k_scale,
                    attn.v_scale,
                )

        self._fused_kv_helper.materialize(
            ctx_hidden=ctx_hidden,
            positions=ctx_positions,
            write_layer_kv=_write_layer_kv,
        )

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
        if input_embeds is None:
            raise ValueError(
                "DFlashDraftModel requires `input_embeds` (use the target embedding)."
            )
        hidden_states = input_embeds
        residual: Optional[torch.Tensor] = None

        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )

        if hidden_states.numel() != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden_states,
        )

    @torch.no_grad()
    def forward_hip(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        """HIP/AMD decode forward with the fused pre-forward prologue inlined.

        Relocates the three worker-side pre-forward steps into the model so the
        stock decode CUDA graph captures them: (1) materialize the previous
        iteration's target hidden into the draft KV cache (kv-mat), (2) build the
        next draft block (block ids / positions / verify-out cache locs), and
        (3) embed the block. The carried verify(N-1) outputs and the draft block
        scratch buffers are read off `forward_batch.spec_info`. Everything after
        the prologue is identical to `forward`.

        NOTE: the attribute wiring (token_to_kv_pool / req_to_token / embed_module
        / mask_token_id / block_size / carried spec_info fields) is intentionally
        not connected yet; this method only lays down the prologue body.
        """
        spec_info = forward_batch.spec_info
        bs = int(forward_batch.batch_size)
        block_size = int(self.block_size)

        # --- (1) kv-mat: materialize carried target hidden into the draft KV cache.
        self.append_target_hidden_to_draft_kv_by_loc(
            token_to_kv_pool=self.token_to_kv_pool,
            device=positions.device,
            target_hidden=spec_info.target_hidden,
            cache_loc=spec_info.kv_cache_loc,
            positions=spec_info.kv_positions,
            cache_loc_2d=spec_info.kv_cache_loc_2d,
            commit_lens=spec_info.commit_lens,
        )

        # --- (2) draft-block-prep: build the next block's ids / positions / cache locs.
        block_ids = spec_info.block_ids_buf[:bs]
        positions_2d = spec_info.block_positions_buf[:bs]
        verify_out_cache_loc_2d = spec_info.verify_out_cache_loc_buf[:bs]
        _prepare_dflash_draft_block_unchecked(
            verified_id=spec_info.verified_id.view(-1),
            prefix_lens=spec_info.prefix_lens.view(-1),
            req_pool_indices=forward_batch.req_pool_indices.view(-1),
            req_to_token=self.req_to_token,
            block_ids_out=block_ids,
            positions_out=positions_2d,
            cache_loc_out=verify_out_cache_loc_2d,
            mask_token_id=int(self.mask_token_id),
        )

        # --- (3) embed: project block ids through the target embedding.
        noise_embedding = self.embed_module(block_ids)
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])
        positions = positions_2d.reshape(-1)

        # --- draft forward: identical to `forward`, so just delegate.
        return self.forward(
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
            get_embedding=get_embedding,
            pp_proxy_tensors=pp_proxy_tensors,
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
