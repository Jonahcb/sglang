"""CUDA-graph runner for the DFlash ctx -> draft-KV materialization region.

Before drafting the next non-causal block, DFLASH projects the accepted target
hidden states and writes them into the draft KV cache, once per draft layer
(see ``DFlashWorker._append_target_hidden_to_draft_kv`` /
``_append_target_hidden_sequential``). On the sequential path this is a Python
loop over draft layers, each launching a kv_proj GEMM + k_norm + k_rope + KV
scatter -- a band of small eager kernels with launch bubbles between them. This
band is the dominant idle gap in DFLASH MXFP4 decode on ROCm.

The region runs *no attention* (only projections and KV scatter writes), so it
can be captured as a standalone CUDA graph keyed by batch size. The per-step
inputs (projection source, RoPE positions, KV slot indices) are copied into
static buffers and the graph is replayed.

Two kinds of padding are flattened into a fixed ``[bs, block_size]`` rectangle:
  * ragged ctx (``ctx_lens[i] <= block_size`` accepted tokens per request), and
  * batch-size padding up to the captured bucket.
All padding rows write to the reserved dummy KV slot 0 (see the token KV
allocator, which hands out slots starting at 1), so they never corrupt real KV.
"""

import logging
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


class DFlashCtxKVGraphRunner:
    """Captures + replays the ctx->draft-KV materialization region per bs bucket."""

    def __init__(self, worker):
        self.draft_model = worker.draft_model
        self.draft_model_runner = worker.draft_model_runner
        self.token_to_kv_pool = self.draft_model_runner.token_to_kv_pool
        self.device = worker.device
        self.block_size = int(worker.block_size)
        self.layers = self.draft_model.layers

        # Capture batch-size buckets (reuse the draft runner's list so the graph
        # set lines up with the batch sizes the rest of the stack expects).
        from sglang.srt.model_executor.cuda_graph_runner import (
            get_batch_sizes_to_capture,
        )

        capture_bs, _ = get_batch_sizes_to_capture(self.draft_model_runner)
        self.capture_bs = sorted({int(b) for b in capture_bs if int(b) > 0})
        if not self.capture_bs:
            raise RuntimeError("DFLASH ctx-KV graph runner found no capture batch sizes.")
        self.max_bs = max(self.capture_bs)
        self.max_rows = self.max_bs * self.block_size

        fc_in = int(self.draft_model.fc.in_features)
        in_dtype = self.draft_model.fc.weight.dtype

        # Static input buffers, allocated once at the largest bucket and sliced per
        # capture. CUDA graphs bake in these addresses, so they must never be
        # reallocated after the first capture.
        self.g_target_hidden = torch.zeros(
            (self.max_rows, fc_in), dtype=in_dtype, device=self.device
        )
        self.g_positions = torch.zeros(
            (self.max_rows,), dtype=torch.int64, device=self.device
        )
        self.g_cache_loc = torch.zeros(
            (self.max_rows,), dtype=torch.int64, device=self.device
        )

        self.graphs: Dict[int, "torch.cuda.CUDAGraph"] = {}
        self._disabled = False
        self._replayed_once = False

    def _bucket_for(self, bs: int) -> Optional[int]:
        for cand in self.capture_bs:
            if cand >= bs:
                return cand
        return None

    def can_run(self, bs: int) -> bool:
        if self._disabled:
            return False
        return self._bucket_for(bs) is not None

    def _run_region(self, rows: int) -> None:
        """The graph body: project + per-layer KV materialize over ``rows`` rows.

        Mirrors ``DFlashWorker._append_target_hidden_sequential`` exactly so the
        captured computation is identical to the eager path.
        """
        target_hidden = self.g_target_hidden[:rows]
        positions = self.g_positions[:rows]
        cache_loc = self.g_cache_loc[:rows]

        ctx_hidden = self.draft_model.project_target_hidden(target_hidden)
        for layer in self.layers:
            attn = layer.self_attn
            k, v = attn.kv_proj_only(ctx_hidden)
            k = attn.apply_k_norm(k)
            k = attn.apply_k_rope(positions, k)
            k = k.view(-1, attn.num_kv_heads, attn.head_dim)
            v = v.view(-1, attn.num_kv_heads, attn.head_dim)
            self.token_to_kv_pool.set_kv_buffer(
                attn.attn,
                cache_loc,
                k,
                v,
                attn.attn.k_scale,
                attn.attn.v_scale,
            )

    def _capture(self, bucket: int) -> bool:
        rows = bucket * self.block_size

        # Capture/warmup writes must hit only the reserved dummy slot 0.
        self.g_cache_loc[:rows].zero_()
        self.g_positions[:rows].zero_()
        self.g_target_hidden[:rows].zero_()

        from sglang.srt.model_executor.cuda_graph_runner import (
            get_global_graph_memory_pool,
            set_global_graph_memory_pool,
        )

        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    with torch.inference_mode():
                        self._run_region(rows)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()

            pool = get_global_graph_memory_pool()
            if pool is None:
                pool = torch.cuda.graph_pool_handle()
                set_global_graph_memory_pool(pool)

            graph = torch.cuda.CUDAGraph()
            with torch.inference_mode():
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    self._run_region(rows)
            torch.cuda.synchronize()
            self.graphs[bucket] = graph
            return True
        except Exception as e:
            logger.warning(
                "DFLASH ctx-KV CUDA graph capture failed (bs bucket=%s); disabling "
                "graph path and falling back to eager KV materialization: %s",
                bucket,
                e,
            )
            self._disabled = True
            self.graphs.clear()
            return False

    def materialize(
        self,
        *,
        target_hidden: torch.Tensor,
        positions_2d: torch.Tensor,
        cache_loc_2d: torch.Tensor,
        mask: torch.Tensor,
        bs: int,
    ) -> bool:
        """Fill the static buffers from this step's rectangle and replay.

        Args (all on device):
          target_hidden: ``[sum(ctx_lens), fc_in]`` ragged real rows, ordered
            row-major over ``mask`` (same order the eager path consumes).
          positions_2d:  ``[bs, block_size]`` int RoPE positions.
          cache_loc_2d:  ``[bs, block_size]`` int64 KV slot indices.
          mask:          ``[bs, block_size]`` bool, True for real rows.

        Returns True if the graph ran, False if the caller should use the eager
        path (uncaptured bs, or capture disabled/failed).
        """
        bucket = self._bucket_for(bs)
        if bucket is None:
            return False
        if bucket not in self.graphs:
            if not self._capture(bucket):
                return False

        rows = bucket * self.block_size
        real = bs * self.block_size

        # KV slots: masked-off rectangle entries -> dummy slot 0.
        zero = cache_loc_2d.new_zeros(())
        cache_loc = torch.where(mask, cache_loc_2d.to(torch.int64), zero).reshape(-1)
        positions = torch.where(
            mask, positions_2d.to(torch.int64), positions_2d.new_zeros(())
        ).reshape(-1)

        self.g_cache_loc[:real].copy_(cache_loc)
        self.g_positions[:real].copy_(positions)

        # Scatter the ragged real rows into the rectangle; padding rows stay zero
        # and (projected) land in dummy slot 0.
        fc_in = self.g_target_hidden.shape[1]
        th = self.g_target_hidden[:real].view(bs, self.block_size, fc_in)
        th.zero_()
        th[mask] = target_hidden.to(self.g_target_hidden.dtype)

        # Batch-size padding (real..rows) -> dummy slot 0, zeroed inputs.
        if real < rows:
            self.g_cache_loc[real:rows].zero_()
            self.g_positions[real:rows].zero_()
            self.g_target_hidden[real:rows].zero_()

        self.graphs[bucket].replay()
        if not self._replayed_once:
            self._replayed_once = True
            logger.info(
                "DFLASH ctx-KV CUDA graph active (first replay: bs=%d, bucket=%d, "
                "real_rows=%d, graph_rows=%d).",
                bs,
                bucket,
                real,
                rows,
            )
        return True
