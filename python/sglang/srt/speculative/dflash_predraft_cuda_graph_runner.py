from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.model_executor.input_buffers import ForwardInputBuffers
from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
    BaseCudaGraphRunner,
)
from sglang.srt.speculative.triton_ops.dflash_prepare_block import (
    _prepare_dflash_draft_block_unchecked,
)


@dataclass
class DFlashDraftAndVerifyInputBuffers(ForwardInputBuffers):
    # Stage 1 (KV materialization) inputs.
    # Stage 1's loc inputs ARE Stage 2's loc output from the previous iteration,
    # so they share storage with verify_out_cache_loc_2d_buf (Option B self-feed):
    # Stage 1 reads it before Stage 2 overwrites it within each captured replay.
    target_hidden_buf: torch.Tensor
    commit_lens_buf: torch.Tensor
    positions_buf: torch.Tensor
    # Stage 2 (block-prep) inputs
    verified_id_buf: torch.Tensor
    prefix_lens_buf: torch.Tensor
    req_pool_indices: torch.Tensor
    # Stage 2 outputs (also Stage 3 input via block_ids_buf)
    block_ids_buf: torch.Tensor
    positions_2d_buf: torch.Tensor
    verify_out_cache_loc_2d_buf: torch.Tensor


class DFlashPreDraftCudaGraphRunner():
    """

    INSERT SPEC
    """

    def __init__(
        self,
        model_runner,
        append_target_hidden_to_draft_kv_by_loc,
        req_to_token,
        embed_module,
        capture_bs,
        block_size,
        mask_token_id,
        pool=None,
    ):
        self.graphs = {}
        # stable handles used inside the captured forward_fn
        self._append_target_hidden_to_draft_kv_by_loc = (
            append_target_hidden_to_draft_kv_by_loc
        )
        self.req_to_token = req_to_token
        self.embed_module = embed_module
        # capture/replay config
        self.capture_bs = capture_bs
        self.block_size = int(block_size)
        self.mask_token_id = int(mask_token_id)
        self._pool = pool

        max_bs = max(self.capture_bs)
        max_tokens = max_bs * self.block_size
        hidden_size = model_runner.model_config.hidden_size
        dtype = model_runner.dtype

        # Initialize static buffers that will be shared across pre-draft CUDA graphs, draft CUDA graphs, and target CUDA graphs
        with torch.device(model_runner.device):
            self.buffers = DFlashDraftAndVerifyInputBuffers(
                # Stage 1 (KV materialization) inputs
                target_hidden_buf=torch.zeros((max_tokens, hidden_size), dtype=dtype),
                commit_lens_buf=torch.zeros((max_bs,), dtype=torch.int32),
                positions_buf=torch.zeros((max_tokens,), dtype=torch.int64),
                # Stage 2 (block-prep) inputs
                verified_id_buf=torch.zeros((max_bs,), dtype=torch.int64),
                prefix_lens_buf=torch.zeros((max_bs,), dtype=torch.int64),
                req_pool_indices=torch.zeros((max_bs,), dtype=torch.int64),
                # Stage 2 outputs (also Stage 3 input via block_ids_buf)
                block_ids_buf=torch.zeros((max_bs, self.block_size), dtype=torch.int64),
                positions_2d_buf=torch.zeros(
                    (max_bs, self.block_size), dtype=torch.int64
                ),
                verify_out_cache_loc_2d_buf=torch.zeros(
                    (max_bs, self.block_size), dtype=torch.int64
                ),
            )

        # Alias matching buffers onto the shared process-wide pool so the
        # pre-draft, draft, and target graphs read/write the same storage.
        self.buffers.share_buffers()

        # Capture the graphs
        self.capture()

    def capture(self):
        # Capture all shapes from largest bs to smallest so smaller buckets
        # reuse the larger memory pool.
        # TODO (jonahbernard): we are implementing a simple version but will need to add more features based on DecodeCudaGraphRunner.capture()
        for bs in sorted(self.capture_bs, reverse=True):
            self.capture_one_shape(bs)

    def can_run(self):
        pass

    def capture_one_shape(self, bs):
        # Slice the shared static buffers down to this bucket's batch size.
        b = self.buffers
        num_tokens = bs * self.block_size
        target_hidden_buf = b.target_hidden_buf[:num_tokens]
        commit_lens_buf = b.commit_lens_buf[:bs]
        positions_buf = b.positions_buf[:num_tokens]
        verified_id_buf = b.verified_id_buf[:bs]
        prefix_lens_buf = b.prefix_lens_buf[:bs]
        req_pool_indices = b.req_pool_indices[:bs]
        block_ids_buf = b.block_ids_buf[:bs]
        positions_2d_buf = b.positions_2d_buf[:bs]
        verify_out_cache_loc_2d_buf = b.verify_out_cache_loc_2d_buf[:bs]
        # Stage 1 loc inputs are views of the Stage 2 loc output (self-feed):
        # this replay's Stage 1 reads what the previous replay's Stage 2 wrote.
        kv_cache_loc2d_buf = verify_out_cache_loc_2d_buf
        kv_cache_loc_buf = verify_out_cache_loc_2d_buf.reshape(-1)

        # GPU work:

        # 1) KV materialization
        # 2) compute KV slots for next KV materialization + draft's KV scratchpad (_prepare_dflash_draft_block_unchecked)
        # 3) embed bonus + mask tokens for entire block (isn't this the same every time for the mask tokens??)
        # 4) done
        def forward_fn ():
            # 1) KV materialization
            self._append_target_hidden_to_draft_kv_by_loc(
                target_hidden=target_hidden_buf.reshape(-1, target_hidden_buf.shape[-1]),
                cache_loc=kv_cache_loc_buf,
                cache_loc_2d=kv_cache_loc2d_buf,
                positions=positions_buf,
                commit_lens=commit_lens_buf,
            )

            # 2) compute next KV slots
            _prepare_dflash_draft_block_unchecked(
                        verified_id=verified_id_buf.view(-1),
                        prefix_lens=prefix_lens_buf.view(-1),
                        req_pool_indices=req_pool_indices.view(-1),
                        req_to_token=self.req_to_token,
                        block_ids_out=block_ids_buf,
                        positions_out=positions_2d_buf,
                        cache_loc_out=verify_out_cache_loc_2d_buf,
                        mask_token_id=self.mask_token_id,
                    )



            # 3) embed bonus + mask tokens
            # TODO (jonahbernard): write input_embeds into a static output buffer so replay can read it
            noise_embedding = self.embed_module(block_ids_buf)
            input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        # insert warmup run
        # TODO (jonahbernard): make this whole more true to FullCudaGraphRunner once we expand functionality to tp
        # TODO (jonahbernard): we need memory saver whatever that is
        for _ in range(2):
            forward_fn()

        torch.cuda.synchronize()

        # start CUDA graph capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._pool): # TODO (jonahbernard): what is pool?
            forward_fn()
        
        self.graphs[bs] = g # stash recorded graph


    
    def replay(self, raw_bs):
        # Pad to nearest captured shape
        bs = BaseCudaGraphRunner._pad_to_bucket(raw_bs, self.capture_bs)

        # replay the graph
        self.graphs[bs].replay() # graphs is a dict of our pre-draft cuda graphs

    # TODO (jonahbernard) do we need a cleanup function like they have in FullCudaGraphBackend?