from __future__ import annotations

import torch

from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
    BaseCudaGraphRunner,
)
from sglang.srt.speculative.triton_ops.dflash_prepare_block import (
    _prepare_dflash_draft_block_unchecked,
)


class DFlashPreDraftCudaGraphRunner():
    """

    INSERT SPEC
    """

    def __init__(
        self,
        append_target_hidden_to_draft_kv_by_loc,
        req_to_token,
        embed_module,
        capture_bs,
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
        self._pool = pool

    def can_run(self):
        pass

    def capture_one_shape(self, bs, target_hidden_buf, kv_cache_loc_buf, kv_cache_loc2d_buf,
    commit_lens_buf, verified_id_buf, prefix_lens_buf, req_pool_indices, block_ids_buf, positions_buf, positions_2d_buf, verify_out_cache_loc_2d_buf, mask_token_id):

        # GPU work:

        # 1) KV materialization
        # 2) compute KV slots for next KV materialization + draft's KV scratchpad (_prepare_dflash_draft_block_unchecked)
        # 3) embed bonus + mask tokens for entire block (isn't this the same every time for the mask tokens??)
        # 4) done
        def forward_fn ():
            # 1) KV materialization
            # TODO (jonahbernard): self._append_target_hidden_to_draft_kv_by_loc must be set on this class
            self._append_target_hidden_to_draft_kv_by_loc(
                target_hidden=target_hidden_buf.reshape(-1, target_hidden_buf.shape[-1]),
                cache_loc=kv_cache_loc_buf,
                cache_loc_2d=kv_cache_loc2d_buf,
                positions=positions_buf,
                commit_lens=commit_lens_buf,
            )

            # 2) compute next KV slots
            # TODO (jonahbernard): self.req_to_token must be set on this class
            _prepare_dflash_draft_block_unchecked(
                        verified_id=verified_id_buf.view(-1),
                        prefix_lens=prefix_lens_buf.view(-1),
                        req_pool_indices=req_pool_indices.view(-1),
                        req_to_token=self.req_to_token,
                        block_ids_out=block_ids_buf,
                        positions_out=positions_2d_buf,
                        cache_loc_out=verify_out_cache_loc_2d_buf,
                        mask_token_id=int(mask_token_id),
                    )



            # 3) embed bonus + mask tokens
            # TODO (jonahbernard): self.embed_module must be set on this class;
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