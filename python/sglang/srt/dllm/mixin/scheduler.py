from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.req_time_stats import set_time_batch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler


class SchedulerDllmMixin:
    def init_diffusion_llm(self: Scheduler):
        self.dllm_config = (
            DllmConfig.from_server_args(self.server_args)
            if self.server_args.dllm_algorithm is not None
            else None
        )

    def get_new_batch_dllm(self: Scheduler) -> Optional[ScheduleBatch]:
        """Generate a new batch for DLLM (Diffusion LLM) scheduling.

        Active dLLM reqs live in self.running_batch between rounds (merged in
        via the standard last_batch -> running_batch path). Each round we drain
        running_batch, stash each req's KV in the prefix tree, refresh fill_ids
        with a fresh mask block, recompute phase, and run the prefill adder
        over (active reqs + new admits from self.waiting_queue).
        """
        if self.enable_priority_preemption:
            self.running_batch.batch_is_full = False

        # Drain active reqs from running_batch and re-prepare them for the
        # next extend. They will re-enter running_batch via the standard
        # last_batch -> running_batch merge after the forward.
        active_reqs = list(self.running_batch.reqs)
        self.running_batch.reqs = []

        for req in active_reqs:
            self.stash_unfinished_req(req)
            req.is_chunked += 1
            # No tree_cache: prefix_indices is preserved from the previous
            # round. The new mask block is appended to fill_ids and the phase
            # is recomputed.
            req.init_next_round_input()

        # Pull new admits from the global waiting queue, capped by
        # max_running_requests.
        capacity = self.dllm_config.max_running_requests - len(active_reqs)
        new_admits = (
            list(self.waiting_queue[:capacity]) if capacity > 0 else []
        )

        if not active_reqs and not new_admits:
            return None

        if new_admits:
            self.policy.calc_priority(new_admits)

        candidates = active_reqs + new_admits
        adder = self._create_dllm_prefill_adder(running_bs=0)
        forward_mode = self._process_dllm_batches(adder, candidates)

        can_run_list = adder.can_run_list
        if not can_run_list:
            # Budget exhausted before anything was admitted. Restore active
            # reqs to running_batch so they aren't lost; next round will
            # re-attempt admission.
            self.running_batch.reqs = active_reqs
            return None

        if adder.preempt_list:
            for req in adder.preempt_list:
                self._add_request_to_queue(req)

        # Drain newly-admitted reqs from waiting_queue
        new_admitted = set(can_run_list) - set(active_reqs)
        if new_admitted:
            self.waiting_queue = [
                r for r in self.waiting_queue if r not in new_admitted
            ]

        self.adder = adder
        self.can_run_list = can_run_list
        self.running_bs = 0

        set_time_batch(can_run_list, "set_forward_entry_time")
        return self._create_dllm_batch(can_run_list, forward_mode)

    def process_batch_result_dllm(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()

        if result.next_token_ids:
            self.token_to_kv_pool_allocator.free_group_begin()

            for idx in range(batch.batch_size()):
                req = batch.reqs[idx]

                next_token_ids = result.next_token_ids[idx].tolist()
                new_tokens = len(next_token_ids)
                if new_tokens == 0:
                    continue

                req.fill_ids[-new_tokens:] = next_token_ids[:]
                self.num_generated_tokens += new_tokens

                req.output_ids.extend(next_token_ids)
                req.check_finished(new_accepted_len=new_tokens)

                if req.finished():
                    release_kv_cache(req, self.tree_cache)
                    req.time_stats.set_completion_time()

            self.stream_output(batch.reqs, batch.return_logprob)
            self.token_to_kv_pool_allocator.free_group_end()

        can_run_cuda_graph = getattr(result, "can_run_cuda_graph", False)
        self.report_prefill_stats(
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )

    def _create_dllm_prefill_adder(self: Scheduler, running_bs: int) -> PrefillAdder:
        """Create a prefill adder configured for DLLM scheduling."""
        return PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            self.running_batch,
            self.new_token_ratio,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
            prefill_max_requests=self.server_args.prefill_max_requests,
            dllm_config=self.dllm_config,
        )

    def _process_dllm_batches(
        self: Scheduler, adder: PrefillAdder, candidates: List[Req]
    ) -> ForwardMode:
        """Process prefill or decode batches for DLLM.

        A single dLLM forward only carries one phase at a time. If any
        candidate is in a prefill phase, the round runs prefill; otherwise it
        runs decode.
        """
        forward_mode = ForwardMode.DLLM_EXTEND

        prefill_reqs = [r for r in candidates if r.is_dllm_prefill()]
        if prefill_reqs:
            self._process_batch_by_phase(
                adder,
                prefill_reqs,
                DllmReqPhase.STAGING_PREFILL,
                DllmReqPhase.INCOMING_PREFILL,
            )
        else:
            decode_reqs = [r for r in candidates if not r.is_dllm_prefill()]
            self._process_batch_by_phase(
                adder,
                decode_reqs,
                DllmReqPhase.STAGING_DECODE,
                DllmReqPhase.INCOMING_DECODE,
            )

        return forward_mode

    def _process_batch_by_phase(
        self,
        adder: PrefillAdder,
        batch: List[Req],
        staging_phase: DllmReqPhase,
        incoming_phase: DllmReqPhase,
    ) -> None:
        """Process a batch, separating staging and incoming requests."""
        staging_reqs = [req for req in batch if req.dllm_phase == staging_phase]
        if staging_reqs:
            staging_result = self.process_dllm_staging_reqs(adder, staging_reqs)
            if staging_result != AddReqResult.CONTINUE:
                return

        incoming_reqs = [req for req in batch if req.dllm_phase == incoming_phase]
        if incoming_reqs:
            self.process_dllm_incoming_reqs(adder, incoming_reqs)

    def _create_dllm_batch(
        self: Scheduler, can_run_list: List[Req], forward_mode: ForwardMode
    ) -> ScheduleBatch:
        """Create and prepare a new DLLM batch."""
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            dllm_config=self.dllm_config,
        )
        new_batch.prepare_for_extend()
        new_batch.forward_mode = forward_mode
        new_batch.decoding_reqs = None

        # Record prefill stats for logging after forward
        from sglang.srt.observability.scheduler_metrics_mixin import PrefillStats

        new_batch.prefill_stats = PrefillStats.from_adder(
            self.adder, self.running_batch.reqs, self.enable_priority_scheduling
        )

        return new_batch

    def process_dllm_incoming_reqs(
        self: Scheduler, adder: PrefillAdder, reqs: List[Req]
    ) -> AddReqResult:
        """Process incoming DLLM requests with resource allocation and preemption."""
        res = AddReqResult.CONTINUE
        for req in reqs:
            # Check if batch is full
            running_bs = len(self.running_batch.reqs)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                self.running_batch.batch_is_full = True

            # Try preemption if batch is full
            if self.running_batch.batch_is_full:
                if (
                    not self.enable_priority_preemption
                    or not adder.preempt_to_schedule(req, self.server_args)
                ):
                    break

            # Prepare and add request
            req.init_next_round_input(self.tree_cache)
            res = adder.add_one_req(
                req,
                has_chunked_req=True,
                truncation_align_size=self.truncation_align_size,
            )

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    self.running_batch.batch_is_full = True
                break

        return res

    def process_dllm_staging_reqs(
        self: Scheduler, adder: PrefillAdder, reqs: List[Req]
    ) -> AddReqResult:
        """Process staging DLLM requests with resource allocation."""
        for req in reqs:
            res = adder.add_dllm_staging_req(req)
            if res == AddReqResult.NO_TOKEN:
                return res

        return AddReqResult.CONTINUE
