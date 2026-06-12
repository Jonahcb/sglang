from __future__ import annotations

import bisect
import contextlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import tqdm

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner import (
    freeze_gc,
    get_batch_sizes_to_capture,
    model_capture_mode,
)
from sglang.srt.model_executor.runner_backend_utils import (
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
)

if TYPE_CHECKING:
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

logger = logging.getLogger(__name__)


@dataclass
class DFlashDraftInputBuffers:
    """Static, bucket-sized buffers for the fused DFLASH draft graph.

    Allocated once at the max captured batch size so the graph bakes stable
    addresses. Per-replay the worker copies live inputs into the leading rows
    and pads the rest; the captured body reads only these buffers.

    The graph captures one rotation: head KV-materialization of the hidden from
    the step's just-completed (eager) target verify, then draft-block prep +
    draft forward + greedy sample producing the NEXT block to verify. The target
    verify itself stays eager and is NOT in this graph.
    """

    # --- Head KV-mat inputs (the block that was just verified eagerly). ---
    # target hidden emitted by the eager verify, 2D [max_num_token, hidden].
    pending_target_hidden: torch.Tensor
    pending_cache_loc_2d: torch.Tensor  # [max_bs, block_size] int64
    pending_positions: torch.Tensor  # [max_num_token] int64
    pending_commit_lens: torch.Tensor  # [max_bs] int32

    # --- Next-block prep inputs. ---
    # Committed prefix lengths entering the NEW block (== new_seq_lens).
    prefix_lens: torch.Tensor  # [max_bs] int64
    seq_lens_cpu: torch.Tensor  # [max_bs] int32 (CPU)
    req_pool_indices: torch.Tensor  # [max_bs] int64
    verified_id: torch.Tensor  # [max_bs] int64 (bonus token seeding the block)

    # --- Next-block scratch (filled in-graph; read out by the worker). ---
    block_ids: torch.Tensor  # [max_bs, block_size] int64
    positions_2d: torch.Tensor  # [max_bs, block_size] int64
    verify_out_cache_loc_2d: torch.Tensor  # [max_bs, block_size] int64
    draft_tokens: torch.Tensor  # [max_bs, block_size] int64


class DFlashDraftGraphRunner:
    """Capture the DFLASH decode draft span as ONE CUDA graph (HIP).

    Graph body: head KV-mat (append the just-verified block's target hidden into
    the draft KV cache) -> draft block prep -> draft forward -> greedy sample.
    The target verify and accept/bonus stay eager in the worker. Standalone
    runner (not a BaseCudaGraphRunner), modeled on the all-in-one unified runner
    but trimmed to the draft half (no target forward, no target plan, no mrope).
    """

    def __init__(self, worker: "DFlashWorkerV2"):
        self.worker = worker
        self.draft_model_runner = worker.draft_model_runner
        self.draft_attn_backend = self.draft_model_runner.attn_backend
        self.draft_model = worker.draft_model
        self.target_model_runner = worker.target_worker.model_runner
        self.block_size = int(worker.block_size)
        self.device = worker.device

        self.graphs = {}
        self.output_buffers = {}
        self.bs = None
        self.raw_bs = None
        self._pool = None  # dedicated CUDA graph mem pool (set on first capture)

        # DFLASH verify forward emits block_size rows per request.
        self.num_tokens_per_bs = self.block_size
        self.capture_bs, _compile_bs = get_batch_sizes_to_capture(
            self.target_model_runner, self.num_tokens_per_bs
        )
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs

        # The draft backend already initialized its cuda-graph state (and
        # captured its own normal decode graphs) at model init, sized for
        # TARGET_VERIFY at block_size tokens — the same regime we capture.
        # Re-allocating here would free buffers those graphs reference
        # (use-after-free -> GPU fault), so only init if it somehow lacks state.
        if getattr(self.draft_attn_backend, "cuda_graph_kv_indices", None) is None:
            self.draft_attn_backend.init_cuda_graph_state(
                self.max_bs, self.max_num_token
            )
        self.seq_len_fill_value = (
            self.draft_attn_backend.get_cuda_graph_seq_len_fill_value()
        )

        # project_target_hidden consumes num_context_features * hidden_size.
        self.target_hidden_size = int(
            self.draft_model.num_context_features
            * self.target_model_runner.model_config.hidden_size
        )

        with torch.device(self.device):
            self.buffers = DFlashDraftInputBuffers(
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
            )

        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture DFLASH draft cuda graph failed: {e}\n"
                f"{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

    # ------------------------------------------------------------------
    # eligibility
    # ------------------------------------------------------------------
    def can_run(self, forward_batch: ForwardBatch) -> bool:
        # The DFLASH decode step arrives as a DECODE-mode worker batch; gate on
        # that, not on the internal verify mode.
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
        """Force the nested draft forward to run eager (no nested graph.replay())
        during our outer capture/warmup."""
        draft_saved = self.draft_model_runner.decode_cuda_graph_runner
        self.draft_model_runner.decode_cuda_graph_runner = None
        try:
            yield
        finally:
            self.draft_model_runner.decode_cuda_graph_runner = draft_saved

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
                            f"Capturing DFLASH draft graph (bs={bs})"
                        )
                    graph, out = self.capture_one_batch_size(bs)
                    self.graphs[bs] = graph
                    self.output_buffers[bs] = out

    def capture_one_batch_size(self, bs: int):
        graph = torch.cuda.CUDAGraph()
        stream = self.stream

        import os as _os

        if _os.environ.get("DFLASH_DRAFT_NO_PLAN") != "1":
            self._plan_metadata(bs, in_capture=True)

        def run_once():
            return self._draft_body(bs)

        # Dedicated memory pool for the draft graphs (shared only among
        # themselves), not the global decode-graph pool.
        with self._suppress_inner_graphs():
            for _ in range(2):
                torch.cuda.synchronize()
                self.target_model_runner.tp_group.barrier()
                run_once()
            if self._pool is None:
                with torch.cuda.graph(graph, stream=stream):
                    out = run_once()
            else:
                with torch.cuda.graph(graph, pool=self._pool, stream=stream):
                    out = run_once()
        self._pool = graph.pool()
        return graph, out

    # ------------------------------------------------------------------
    # metadata planning (draft backend only)
    # ------------------------------------------------------------------
    def _plan_metadata(self, bs: int, in_capture: bool) -> None:
        import os as _os
        from types import SimpleNamespace

        if _os.environ.get("DFLASH_SKIP_DRAFT_PLAN") == "1":
            return

        buffers = self.buffers
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
        with forward_context(ForwardContext(attn_backend=self.draft_attn_backend)):
            self.draft_attn_backend.init_forward_metadata_out_graph(
                draft_view, in_capture=in_capture
            )

    # ------------------------------------------------------------------
    # the captured body
    # ------------------------------------------------------------------
    def _draft_body(self, bs: int) -> dict:
        import os

        if os.environ.get("DFLASH_DRAFT_NOOP_BODY") == "1":
            # Bisection: capture a trivial op only.
            return {"draft_tokens": self.buffers.draft_tokens[:bs] * 0}
        return self.worker._run_draft_capture_body(self.buffers, bs)

    # ------------------------------------------------------------------
    # replay
    # ------------------------------------------------------------------
    def replay(self, replay_inputs: dict) -> dict:
        raw_bs = int(replay_inputs["raw_bs"])
        index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]
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

        buffers.pending_commit_lens[:raw_bs].copy_(replay_inputs["commit_lens"])
        buffers.pending_cache_loc_2d[:raw_bs].copy_(replay_inputs["cache_loc_2d"])
        buffers.pending_positions[:raw_num_token].copy_(replay_inputs["positions"])
        buffers.pending_target_hidden[:raw_num_token].copy_(
            replay_inputs["target_hidden"]
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
