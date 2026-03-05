from __future__ import annotations

from typing import TYPE_CHECKING

import math

import mlx.core
from torch.utils import dlpack
from mlx.core.fast import scaled_dot_product_attention
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


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
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # Need optimize the performance later.

            extend_seq_len_q = extend_seq_lens[seq_idx]
            prefill_seq_len_q = extend_prefix_lens[seq_idx]

            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + extend_seq_len_q
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
            per_req_query_redudant = mlx.core.empty(
                (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
                dtype=per_req_query.dtype,
            )

            per_req_query_redudant[:, prefill_seq_len_q:, :] = per_req_query

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, atten_start_kv:atten_end_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                # scaled_dot_product_attention() expects query, key, and value to have the same dtype
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

   
            per_req_out_redudant = (
              scaled_dot_product_attention(
                  per_req_query_redudant.unsqueeze(0),
                  per_req_key.unsqueeze(0),
                  per_req_value.unsqueeze(0),
                  enable_gqa=enable_gqa,
                  scale=scaling,
                  is_causal=causal,
              )
              .squeeze(0)
              .movedim(query.dim() - 2, 0)
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
        query = query.movedim(0, query.dim() - 2)

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
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                # scaled_dot_product_attention() expects query, key, and value to have the same dtype
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)


            per_req_out = (
                scaled_dot_product_attention(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out
            start_q, start_kv = end_q, end_kv

        return output


    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # Convert from torch tensor to mlx array using zero-copy
        # TODO (Jonahcb): determine whether this is creating a copy or not
        q = mlx.core.array(q)
        k = mlx.core.array(dlpack.to_dlpack(k))
        v = mlx.core.array(dlpack.to_dlpack(v))
        if layer.qk_head_dim != layer.v_head_dim:
            o = mlx.core.zeros((q.shape[0], layer.tp_q_head_num * layer.v_head_dim), dtype=q.dtype)
        else:
            o = mlx.core.zeros_like(q)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        _, max_extend_len = self.forward_metadata

        self.extend_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k,
            v,
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_seq_lens,
            forward_batch.extend_start_loc,
            max_extend_len,
            layer.scaling,
            layer.logit_cap,
        )
        return o

    def forward_decode(
        self,
        q: mlx.core.Array,
        k: mlx.core.Array,
        v: mlx.core.Array,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        attn_logits, _ = self.forward_metadata

        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = mlx.core.zeros((q.shape[0], layer.tp_q_head_num * layer.v_head_dim), dtype=q.dtype)
        else:
            o = mlx.core.zeros_like(q)

        self.decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            k,
            v,
            forward_batch.out_cache_loc,
            attn_logits,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            layer.scaling,
            layer.logit_cap,
        )

        return o

    def support_triton(self):
        return False