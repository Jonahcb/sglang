from __future__ import annotations


class DFlashPreDraftCudaGraphRunner():
    """

    INSERT SPEC
    """

    def __init__(
        self,
    ):
        self.graphs = {}
        pass

    def can_run():
        pass

    def capture_one_shape(self, bs, target_hidden_buf, kv_cache_loc_buf, kv_cache_loc2d_buf, 
    commit_lens_buf, verified_id_buf, prefix_lens_buf, req_pool_indices, block_ids_buf, positions_2d_buf, verify_out_cache_loc_2d_buf, mask_token_id):
        
        # GPU work:

        # 1) KV materialization
        # 2) compute KV slots for next KV materialization + draft's KV scratchpad (_prepare_dflash_draft_block_unchecked)
        # 3) embed bonus + mask tokens for entire block (isn't this the same every time for the mask tokens??)
        # 4) done
        def forward_fn ():
            # 1) KV materialization
            self._append_target_hidden_to_draft_kv_by_loc(
                target_hidden=hidden.reshape(-1, hidden.shape[-1]),
                cache_loc=verify_out_cache_loc,
                cache_loc_2d=verify_out_cache_loc_2d,
                positions=positions,
                commit_lens=commit_lens,
            )

            # 2) compute next KV slots
            _prepare_dflash_draft_block_unchecked(
                        verified_id=draft_input.verified_id.view(-1),
                        prefix_lens=prefix_lens.view(-1),
                        req_pool_indices=model_worker_batch.req_pool_indices.view(-1),
                        req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                        block_ids_out=block_ids,
                        positions_out=positions_2d,
                        cache_loc_out=verify_out_cache_loc_2d,
                        mask_token_id=int(self._mask_token_id),
                    )



            # 3) embed bonus + mask tokens
            noise_embedding = embed_module(block_ids)
            input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        # insert warmup run
        # TODO (jonahbernard): make this whole more true to FullCudaGraphRunner once we expand functionality to tp
        # TODO (jonahbernard): we need memory saver whatever that is
        for _ in range(2):
            forward_fn()

        torch.cuda.synchronize()

        # start CUDA graph capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._pool) # TODO (jonahbernard): what is pool?
            forward_fn()
        
        self.graphs[bs] = g # stash recorded graph

        pass
    
    def replay(self, raw_bs):
        # Pad to nearest captured shape
        bs = self._pad_to_bucket(raw_bs, self.capture_bs) # need to inherit _pad_to_bucket from CudaGraphRunner?

        # replay the graph
        self.graphs[bs].replay() # graphs is a dict of our pre-draft cuda graphs
        pass
    
    # TODO (jonahbernard) do we need a cleanup function like they have in FullCudaGraphBackend?