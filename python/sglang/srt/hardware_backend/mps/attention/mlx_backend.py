from __future__ import annotations

from typing import TYPE_CHECKING

import math
import torch

import mlx.core
from torch.utils import dlpack
from mlx.core.fast import scaled_dot_product_attention
from sglang.srt.hardware_backend.mps.attention.mlx_base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

def _torch_to_mlx(tensor: torch.Tensor) -> "mlx.array":
    """Convert a PyTorch tensor to an MLX array (via numpy on CPU)."""
    t = tensor.cpu().detach()
    if t.dtype == torch.bfloat16:
        return mlx.core.array(t.float().numpy(), dtype=mlx.core.bfloat16)
    return mlx.core.array(t.numpy())


def _mlx_to_torch(array: "mlx.core.Array", device: torch.device) -> torch.Tensor:
    """Convert an MLX array to a PyTorch tensor (zero-copy via memoryview)."""
    torch_dtype = _MLX_TO_TORCH_DTYPE.get(array.dtype, torch.float32)
    array = mlx.core.contiguous(array)
    mlx.core.eval(array)
    tensor = torch.frombuffer(memoryview(array), dtype=torch_dtype).reshape(array.shape)
    if device.type == "mps":
        tensor = tensor.to(device)
    return tensor


class MPSMLXNativeAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        pass

    def run_sdpa_forward_extend(
        self,
        query: mlx.core.Array,
        output: mlx.core.Array,
        k_cache: mlx.core.Array,
        v_cache: mlx.core.Array,
        req_to_token: mlx.core.Array,
        req_pool_indices: mlx.core.Array,
        seq_lens: mlx.core.Array,
        extend_prefix_lens: mlx.core.Array,
        extend_seq_lens: mlx.core.Array,
        encoder_lens: mlx.core.Array = None,
        is_cross_attention: bool = False,
        scaling=None,
        enable_gqa=False,
        causal=False,
        logit_cap: float = 0.0,
        logit_capping_method: str = "tanh",
    ):
        """Run the extend forward by using mlx native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            extend_prefix_lens: [num_seqs]
            extend_seq_lens: [num_seqs]
            encoder_lens: [num_seqs]
            is_cross_attention: [bool]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.moveaxis(0, query.ndim - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # Need optimize the performance later.

            extend_seq_len_q = extend_seq_lens[seq_idx]
            prefill_seq_len_q = extend_prefix_lens[seq_idx]

            seq_len_kv = seq_lens[seq_idx]
            end_q = (start_q + extend_seq_len_q).item() 
            end_kv = start_kv + seq_len_kv
            atten_start_kv = 0
            atten_end_kv = seq_lens[seq_idx]
            # support cross attention
            if encoder_lens is not None:
                if is_cross_attention:
                    atten_end_kv = encoder_lens[seq_idx]
                else:
                    atten_start_kv = encoder_lens[seq_idx]
                    atten_end_kv = encoder_lens[seq_idx] + extend_seq_len_q
            per_req_query = query[:, start_q:end_q, :]
            per_req_query_redudant = mlx.core.zeros(
                (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
                dtype=per_req_query.dtype,
            )

            prefill_seq_len_q = prefill_seq_len_q.item()
            per_req_query_redudant[:, prefill_seq_len_q:, :] = per_req_query

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, atten_start_kv:atten_end_kv]
            per_req_tokens = _torch_to_mlx(per_req_tokens)
            per_req_key = k_cache[per_req_tokens].moveaxis(0, query.ndim - 2)
            per_req_value = v_cache[per_req_tokens].moveaxis(0, query.ndim - 2)

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                # scaled_dot_product_attention() expects query, key, and value to have the same dtype
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

   
            per_req_out_redudant = (
                scaled_dot_product_attention(
                    mlx.core.expand_dims(per_req_query_redudant, 0),
                    mlx.core.expand_dims(per_req_key, 0),
                    mlx.core.expand_dims(per_req_value, 0),
                    scale=scaling,
                )
                .squeeze(0)
                .moveaxis(query.ndim - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out_redudant[prefill_seq_len_q:, :, :]
            start_q, start_kv = end_q, end_kv
        return output

    def run_sdpa_forward_decode(
        self,
        query: mlx.core.Array,
        output: mlx.core.Array,
        k_cache: mlx.core.Array,
        v_cache: mlx.core.Array,
        req_to_token: mlx.core.Array,
        req_pool_indices: mlx.core.Array,
        seq_lens: mlx.core.Array,
        encoder_lens: mlx.core.Array = None,
        is_cross_attention: bool = False,
        scaling=None,
        enable_gqa=False,
        causal=False,
        logit_cap: float = 0.0,
        logit_capping_method: str = "tanh",
    ):
        """Run the decode forward by using mlx native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            encoder_lens: [num_seqs]
            is_cross_attention: [bool]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.moveaxis(0, query.ndim - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # Need optimize the performance later.

            seq_len_q = 1
            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + seq_len_q
            end_kv = start_kv + seq_len_kv
            atten_start_kv = 0
            atten_end_kv = seq_lens[seq_idx]
            # support cross attention
            if encoder_lens is not None:
                if is_cross_attention:
                    atten_end_kv = encoder_lens[seq_idx]
                else:
                    atten_start_kv = encoder_lens[seq_idx]
                    atten_end_kv = encoder_lens[seq_idx] + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, atten_start_kv:atten_end_kv]
            per_req_key = k_cache[per_req_tokens].moveaxis(0, query.ndim - 2)
            per_req_value = v_cache[per_req_tokens].moveaxis(0, query.ndim - 2)

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                # scaled_dot_product_attention() expects query, key, and value to have the same dtype
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)


            per_req_out = (
                scaled_dot_product_attention(
                    mlx.core.expand_dims(per_req_query, 0),
                    mlx.core.expand_dims(per_req_key, 0),
                    mlx.core.expand_dims(per_req_value, 0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .moveaxis(query.ndim - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out
            start_q, start_kv = end_q, end_kv

        return output


    def forward_extend(
        self,
        q,
        k,
        v,
        layer: MLXRadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ):
        # Convert from torch tensor to mlx array using zero-copy
        # TODO (Jonahcb): determine whether this is creating a copy or not
        # q = _torch_to_mlx(q)
        # k = _torch_to_mlx(k)
        # v = _torch_to_mlx(v)
        if layer.qk_head_dim != layer.v_head_dim:
            o = mlx.core.zeros((q.shape[0], layer.tp_q_head_num * layer.v_head_dim), dtype=q.dtype)
        else:
            o = mlx.core.zeros_like(q)

        # TODO (Jonahcb): we should be producing out_cache_loc as an MLX array much earlier than here (I think)
        out_cache_loc = _torch_to_mlx(forward_batch.out_cache_loc)
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, out_cache_loc, k, v
            )

        # TODO (Jonahcb): investigate what max_extend_len is used for
        # _, max_extend_len = self.forward_metadata

        # Convert some Torch tensors to MLX arrays
        extend_seq_lens = _torch_to_mlx(forward_batch.extend_seq_lens)
        self.run_sdpa_forward_extend(
            query=q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim),
            output=o.reshape(-1, layer.tp_q_head_num, layer.v_head_dim),
            k_cache=k,
            v_cache=v,
            # forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            # forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            req_to_token=forward_batch.req_to_token_pool.req_to_token,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=forward_batch.extend_prefix_lens,
            #max_extend_len,
            scaling=layer.scaling,
            logit_cap=layer.logit_cap,
            logit_capping_method=layer.logit_capping_method,
        )
        return o

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: MLXRadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ):
        # Convert from torch tensor to mlx array
        q = _torch_to_mlx(q)
        k = _torch_to_mlx(k)
        v = _torch_to_mlx(v)

        if layer.qk_head_dim != layer.v_head_dim:
            o = mlx.core.zeros((q.shape[0], layer.tp_q_head_num * layer.v_head_dim), dtype=q.dtype)
        else:
            o = mlx.core.zeros_like(q)

        out_cache_loc = _torch_to_mlx(forward_batch.out_cache_loc)
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, out_cache_loc, k, v
            )

        # Convert some Torch tensors to MLX arrays
        req_to_token = _torch_to_mlx(forward_batch.req_to_token_pool.req_to_token)
        req_pool_indices = _torch_to_mlx(forward_batch.req_pool_indices)
        seq_lens = _torch_to_mlx(forward_batch.seq_lens)

        self.run_sdpa_forward_decode(
            query=q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim),
            output=o.reshape(-1, layer.tp_q_head_num, layer.v_head_dim),
            k_cache=forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            v_cache=forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            scaling=layer.scaling,
            logit_cap=layer.logit_cap,
            logit_capping_method=layer.logit_capping_method,
        )

        return o

    def support_triton(self):
        return False