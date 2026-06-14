from __future__ import annotations


class DFlashPreDraftCudaGraphRunner():
    """

    INSERT SPEC
    """

    def __init__(
        self,
    ):
        pass

    def can_run():
        pass

    def capture_one_shape():
        pass
    
    def replay(self, raw_bs, target_hidden_buf, kv_cache_loc_buf, kv_cache_loc2d_buf, 
    commit_lens_buf, verified_id_buf, prefix_lens_buf, req_pool_indices, block_ids_buf, positions_2d_buf, verify_out_cache_loc_2d_buf, mask_token_id):
        # CPU work to prepare everything for these 3 stages of GPU work:
        # 1) calculate draft_prefix_lengths, calculate new request lengths,
        #    calculate new pages to use for KV cache. For compact path: rebuild req_to_token window
        # 2) Choose bs to use and slice buffers
        # Pad to nearest captured shape
        bs = self._pad_to_bucket(raw_bs, self.capture_bs) # need to inherit _pad_to_bucket from CudaGraphRunner?


        # 3) Copy everything into buffers:
            # KV materialization:
            # target_hidden |||| logits_output.hidden_states
            # kv_cache_loc  |||| written by this graph in STAGE 2 from last iteration
            # kv_positions  |||| not sure...
            # kv_cache_loc2d |||| written by this graph in STAGE 2 from last iteration
            # commit_lens    |||| not sure...

            # _prepare_dflash_draft_block_unchecked:
            # verified_id |||| not sure...
            # prefix_lens |||| whatever was originally writing this before the refactor can keep writing it
            # req_pool_indices |||| copy in for now
            # block_ids
            block_ids = _draft_block_ids_buf[:raw_bs] # no copy, just pass pointer
            # positions_2d,
            positions_2d = _draft_block_positions_buf[:bs] # no copy, just pass pointer
            # verify_out_cache_loc_2d,
            verify_out_cache_loc_2d = _draft_verify_out_cache_loc_buf[:bs] # no copy, just pass pointer
            # mask_token_id (never changes so don't have to copy)
            mask_token_id = mask_token_id # no copy, just pass pointer


        # GPU work:

        # 1) KV materialization
        # 2) compute KV slots for next KV materialization + draft's KV scratchpad (_prepare_dflash_draft_block_unchecked)
        # 3) embed bonus + mask tokens for entire block (isn't this the same every time for the mask tokens??)
        # 4) done
        pass