from __future__ import annotations

import bisect
import contextlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import tqdm

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner import (
    freeze_gc,
    get_batch_sizes_to_capture,
    get_global_graph_memory_pool,
    model_capture_mode,
    set_global_graph_memory_pool,
)
from sglang.srt.model_executor.runner_backend_utils import (
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
)
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

if TYPE_CHECKING:
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

logger = logging.getLogger(__name__)


@dataclass
class DFlashUnifiedInputBuffers:
    """Static, bucket-sized buffers for the unified DFLASH decode graph.

    Allocated once at the max captured batch size so the graph bakes stable
    addresses. Per-replay the worker copies live inputs into the leading rows
    and pads the rest; the captured body reads only these buffers.
    """

    # Committed prefix lengths entering this step (target seq_lens before the
    # block is appended). Drives KV-mat positions + draft/target attn metadata.
    prefix_lens: torch.Tensor  # [max_bs] int64
    seq_lens_cpu: torch.Tensor  # [max_bs] int32 (CPU)
    req_pool_indices: torch.Tensor  # [max_bs] int64
    verified_id: torch.Tensor  # [max_bs] int64 (bonus token seeding the block)

    # Block scratch (filled inside the body by the Triton prepare-block kernel).
    block_ids: torch.Tensor  # [max_bs, block_size] int64
    positions_2d: torch.Tensor  # [max_bs, block_size] int64
    verify_out_cache_loc_2d: torch.Tensor  # [max_bs, block_size] int64
    draft_tokens: torch.Tensor  # [max_bs, block_size] int64

    # Rotation seam: target hidden from the PREVIOUS replay, consumed by the
    # head KV-mat of THIS replay. 2D [max_num_token, target_hidden_size].
    pending_target_hidden: torch.Tensor
    pending_cache_loc_2d: torch.Tensor  # [max_bs, block_size] int64
    pending_positions: torch.Tensor  # [max_num_token] int64
    pending_commit_lens: torch.Tensor  # [max_bs] int32

    # mrope positions for the target verify forward (target model is mrope).
    # Text-only regime: each of the 3 rows == positions. [3, max_num_token].
    mrope_positions: torch.Tensor


class DFlashUnifiedGraphRunner:
    """Capture DFLASH decode as ONE CUDA graph: KV-mat -> draft -> sample -> target.

    Standalone runner (not a BaseCudaGraphRunner), modeled on
    FrozenKVMTPCudaGraphRunner but capturing BOTH model forwards plus the
    per-layer KV-materialization band. HIP only.

    The cut is after the target forward; accept/bonus, sampling, and the
    "is request done" decision stay eager (the normal scheduler loop). The KV
    append is rotated to the HEAD of the next step's graph (consuming the prior
    step's target hidden held in pending_* buffers).
    """

    def __init__(self, worker: "DFlashWorkerV2"):
        self.worker = worker
        self.draft_model_runner = worker.draft_model_runner
        self.target_worker = worker.target_worker
        self.target_model_runner = worker.target_worker.model_runner
        self.draft_attn_backend = self.draft_model_runner.attn_backend
        self.target_attn_backend = self.target_model_runner.attn_backend
        self.draft_model = worker.draft_model
        self.block_size = int(worker.block_size)
        self.device = worker.device

        self.graphs = {}
        self.output_buffers = {}
        self.bs = None
        self.raw_bs = None
        self._unified_pool = None  # dedicated CUDA graph mem pool (set on first capture)

        # DFLASH verify forward emits block_size rows per request.
        self.num_tokens_per_bs = self.block_size
        self.capture_bs, _compile_bs = get_batch_sizes_to_capture(
            self.target_model_runner, self.num_tokens_per_bs
        )
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs

        # Both backends already initialized their cuda-graph state (and captured
        # their own normal decode graphs) at model init, sized for TARGET_VERIFY
        # at block_size tokens — the same regime we capture. Re-allocating here
        # would free buffers those already-captured graphs reference (use-after-
        # free → GPU memory fault), so only initialize a backend that somehow
        # lacks state.
        if getattr(self.draft_attn_backend, "cuda_graph_kv_indices", None) is None:
            self.draft_attn_backend.init_cuda_graph_state(
                self.max_bs, self.max_num_token
            )
        if getattr(self.target_attn_backend, "cuda_graph_kv_indices", None) is None:
            self.target_attn_backend.init_cuda_graph_state(
                self.max_bs, self.max_num_token
            )
        self.seq_len_fill_value = (
            self.target_attn_backend.get_cuda_graph_seq_len_fill_value()
        )

        # The target verify forward emits concatenated layer features (one block
        # of hidden_size per captured target layer); project_target_hidden
        # consumes exactly num_context_features * hidden_size.
        self.target_hidden_size = int(
            self.draft_model.num_context_features
            * self.target_model_runner.model_config.hidden_size
        )

        with torch.device(self.device):
            self.buffers = DFlashUnifiedInputBuffers(
                prefix_lens=torch.full(
                    (self.max_bs,), self.seq_len_fill_value, dtype=torch.int64
                ),
                seq_lens_cpu=torch.full(
                    (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
                ).cpu(),
                req_pool_indices=torch.zeros((self.max_bs,), dtype=torch.int64),
                verified_id=torch.zeros((self.max_bs,), dtype=torch.int64),
                block_ids=torch.zeros(
                    (self.max_bs, self.block_size), dtype=torch.int64
                ),
                positions_2d=torch.zeros(
                    (self.max_bs, self.block_size), dtype=torch.int64
                ),
                verify_out_cache_loc_2d=torch.zeros(
                    (self.max_bs, self.block_size), dtype=torch.int64
                ),
                draft_tokens=torch.zeros(
                    (self.max_bs, self.block_size), dtype=torch.int64
                ),
                pending_target_hidden=torch.zeros(
                    (self.max_num_token, self.target_hidden_size),
                    dtype=self.target_model_runner.dtype,
                ),
                pending_cache_loc_2d=torch.zeros(
                    (self.max_bs, self.block_size), dtype=torch.int64
                ),
                pending_positions=torch.zeros(
                    (self.max_num_token,), dtype=torch.int64
                ),
                pending_commit_lens=torch.zeros((self.max_bs,), dtype=torch.int32),
                mrope_positions=torch.zeros(
                    (3, self.max_num_token), dtype=torch.int64
                ),
            )

        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture DFLASH unified cuda graph failed: {e}\n"
                f"{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

    # ------------------------------------------------------------------
    # eligibility
    # ------------------------------------------------------------------
    def can_run(self, forward_batch: ForwardBatch) -> bool:
        # The DFLASH decode step arrives as a DECODE-mode worker batch; the
        # TARGET_VERIFY forwards are constructed internally by the body. Gate on
        # the incoming (decode) mode, not the internal verify mode.
        if not forward_batch.forward_mode.is_decode():
            return False
        bs = len(forward_batch.seq_lens)
        if bs > self.max_bs:
            return False
        sampling_info = getattr(forward_batch, "sampling_info", None)
        if sampling_info is not None and not sampling_info.is_all_greedy:
            return False  # non-greedy verify uses an uncapturable top-k .item()
        return True

    # ------------------------------------------------------------------
    # inner-graph suppression
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def _suppress_inner_graphs(self):
        """Force the nested draft/target forwards to run eager (no nested
        graph.replay()) during our outer capture/warmup."""
        draft_saved = self.draft_model_runner.decode_cuda_graph_runner
        target_saved = self.target_model_runner.decode_cuda_graph_runner
        self.draft_model_runner.decode_cuda_graph_runner = None
        self.target_model_runner.decode_cuda_graph_runner = None
        try:
            yield
        finally:
            self.draft_model_runner.decode_cuda_graph_runner = draft_saved
            self.target_model_runner.decode_cuda_graph_runner = target_saved

    # ------------------------------------------------------------------
    # capture
    # ------------------------------------------------------------------
    def capture(self) -> None:
        with freeze_gc(self.target_model_runner.server_args.enable_cudagraph_gc):
            with graph_capture() as graph_capture_context:
                self.stream = graph_capture_context.stream
                capture_range = (
                    tqdm.tqdm(list(reversed(self.capture_bs)))
                    if get_tensor_model_parallel_rank() == 0
                    else reversed(self.capture_bs)
                )
                for bs in capture_range:
                    if isinstance(capture_range, tqdm.tqdm):
                        capture_range.set_description(
                            f"Capturing DFLASH unified graph (bs={bs})"
                        )
                    graph, out = self.capture_one_batch_size(bs)
                    self.graphs[bs] = graph
                    self.output_buffers[bs] = out

    def capture_one_batch_size(self, bs: int):
        graph = torch.cuda.CUDAGraph()
        stream = self.stream

        # Pre-plan both backends' attention metadata against the static buffers.
        import os as _os

        if _os.environ.get("DFLASH_UNIFIED_NO_PLAN") != "1":
            self._plan_metadata(bs, in_capture=True)

        def run_once():
            return self._unified_body(bs)

        # Use a DEDICATED memory pool for the unified graphs (shared only among
        # themselves), not the global decode-graph pool. Capturing two full
        # models (incl. the target MoE + aiter allocations) into the global pool
        # corrupts state the normal prefill/decode graphs reuse. The first graph
        # creates the pool (pool=None); subsequent graphs reuse it.
        with self._suppress_inner_graphs():
            for _ in range(2):
                torch.cuda.synchronize()
                self.target_model_runner.tp_group.barrier()
                run_once()
            if self._unified_pool is None:
                with torch.cuda.graph(graph, stream=stream):
                    out = run_once()
            else:
                with torch.cuda.graph(
                    graph, pool=self._unified_pool, stream=stream
                ):
                    out = run_once()
        self._unified_pool = graph.pool()
        return graph, out

    # ------------------------------------------------------------------
    # metadata planning (both backends)
    # ------------------------------------------------------------------
    def _plan_metadata(self, bs: int, in_capture: bool) -> None:
        from types import SimpleNamespace

        buffers = self.buffers
        # Draft backend: TARGET_VERIFY over the draft KV pool.
        draft_view = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=buffers.block_ids[:bs].reshape(-1),
            req_pool_indices=buffers.req_pool_indices[:bs],
            seq_lens=buffers.prefix_lens[:bs],
            seq_lens_sum=None,
            seq_lens_cpu=buffers.seq_lens_cpu[:bs],
            out_cache_loc=buffers.verify_out_cache_loc_2d[:bs].reshape(-1),
            positions=buffers.positions_2d[:bs].reshape(-1),
            encoder_lens=None,
            spec_info=self.worker._draft_block_spec_info,
        )
        import os as _os

        if _os.environ.get("DFLASH_SKIP_DRAFT_PLAN") != "1":
            with forward_context(ForwardContext(attn_backend=self.draft_attn_backend)):
                self.draft_attn_backend.init_forward_metadata_out_graph(
                    draft_view, in_capture=in_capture
                )

        # Target backend: TARGET_VERIFY over the target KV pool.
        target_view = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=buffers.draft_tokens[:bs].reshape(-1),
            req_pool_indices=buffers.req_pool_indices[:bs],
            seq_lens=buffers.prefix_lens[:bs],
            seq_lens_sum=None,
            seq_lens_cpu=buffers.seq_lens_cpu[:bs],
            out_cache_loc=buffers.verify_out_cache_loc_2d[:bs].reshape(-1),
            positions=buffers.positions_2d[:bs].reshape(-1),
            encoder_lens=None,
            spec_info=DFlashVerifyInput(
                draft_token=buffers.draft_tokens[:bs].reshape(-1),
                positions=buffers.positions_2d[:bs].reshape(-1),
                draft_token_num=self.block_size,
                custom_mask=None,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            ),
        )
        if _os.environ.get("DFLASH_SKIP_TARGET_PLAN") != "1":
            with forward_context(ForwardContext(attn_backend=self.target_attn_backend)):
                self.target_attn_backend.init_forward_metadata_out_graph(
                    target_view, in_capture=in_capture
                )

    # ------------------------------------------------------------------
    # the captured four-stage body
    # ------------------------------------------------------------------
    def _unified_body(self, bs: int) -> dict:
        import os

        if os.environ.get("DFLASH_UNIFIED_NOOP_BODY") == "1":
            # Bisection: capture a trivial op only, to test whether the body's
            # GPU work (forwards / KV writes) is what corrupts shared state.
            return {"next_token_logits": self.buffers.prefix_lens[:bs] * 0}
        return self.worker._run_unified_capture_body(self.buffers, bs)

    # ------------------------------------------------------------------
    # replay
    # ------------------------------------------------------------------
    def replay(self, replay_inputs: dict) -> dict:
        raw_bs = int(replay_inputs["raw_bs"])
        index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]
        num_token = bs * self.block_size
        raw_num_token = raw_bs * self.block_size

        buffers = self.buffers
        if bs != raw_bs:
            buffers.prefix_lens.fill_(self.seq_len_fill_value)
            buffers.seq_lens_cpu.fill_(self.seq_len_fill_value)
            buffers.req_pool_indices.zero_()
            buffers.verified_id.zero_()
            buffers.pending_commit_lens.zero_()
            buffers.pending_cache_loc_2d.zero_()
            buffers.pending_positions.zero_()

        buffers.prefix_lens[:raw_bs].copy_(replay_inputs["prefix_lens"])
        buffers.seq_lens_cpu[:raw_bs].copy_(replay_inputs["seq_lens_cpu"])
        buffers.req_pool_indices[:raw_bs].copy_(replay_inputs["req_pool_indices"])
        buffers.verified_id[:raw_bs].copy_(replay_inputs["verified_id"])

        pending = replay_inputs.get("pending")
        if pending is None:
            # First decode step: prefill seed already materialized eagerly.
            buffers.pending_commit_lens[:raw_bs].zero_()
        else:
            buffers.pending_commit_lens[:raw_bs].copy_(pending["commit_lens"])
            buffers.pending_cache_loc_2d[:raw_bs].copy_(pending["cache_loc_2d"])
            buffers.pending_positions[:raw_num_token].copy_(pending["positions"])
            buffers.pending_target_hidden[:raw_num_token].copy_(
                pending["target_hidden"]
            )

        self._plan_metadata(bs, in_capture=False)
        self.graphs[bs].replay()

        self.bs = bs
        self.raw_bs = raw_bs
        out = self.output_buffers[bs]
        cap_num_token = bs * self.block_size

        def _slice(v: torch.Tensor) -> torch.Tensor:
            if not (isinstance(v, torch.Tensor) and v.shape):
                return v
            n = v.shape[0]
            if n == cap_num_token:
                return v[:raw_num_token]
            if n == bs:
                return v[:raw_bs]
            return v

        return {k: _slice(v) for k, v in out.items()}
