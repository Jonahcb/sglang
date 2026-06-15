"""Isolation tests for DFlashPreDraftCudaGraphRunner.replay().

The pre-draft runner is fully dependency-injected: its __init__ only reads a
handful of scalar attributes off model_runner and otherwise takes the KV
materialization callable, req_to_token, and embed_module as arguments. That lets
us construct the real runner -- running real CUDA-graph capture of both triton
stages -- with lightweight fakes and no model weights.

Stage 2 (block-prep) output is a pure function of verified_id / prefix_lens /
req_pool_indices / req_to_token, so each replay's outputs are checked against a
plain-PyTorch reference across many bucket sizes (exact hits and padded sizes).
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.dflash_predraft_cuda_graph_runner import (
    DFlashPreDraftCudaGraphRunner,
)
from sglang.test.test_utils import CustomTestCase

HID = 64
N_LAYERS = 3  # aux hidden width = N_LAYERS * HID
BLOCK = 4
VOCAB = 256
DTYPE = torch.bfloat16
MASK_TOKEN_ID = VOCAB - 1
NUM_REQS = 32
WIDTH = 128
CAPTURE_BS = [1, 2, 3, 4, 5, 7, 8, 13, 16]  # ascending (replay pads up to a bucket)


@unittest.skipUnless(torch.cuda.is_available(), "DFlash pre-draft graph needs CUDA")
class TestDFlashPreDraftReplay(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda")
        cls.max_bs = max(CAPTURE_BS)

        # req_to_token must be 2D, row-major contiguous, int64 (Stage-2 asserts).
        cls.req_to_token = torch.randint(
            0, 9999, (NUM_REQS, WIDTH), dtype=torch.int64, device=cls.device
        )
        cls.embed_module = torch.nn.Embedding(VOCAB, HID).to(cls.device, DTYPE)

        # Witness buffers record what Stage 1 received each replay, so the KV
        # stub stays graph-capturable (copy_ only -- no .item()/.cpu()/print).
        cls.witness_positions = torch.zeros(
            (cls.max_bs * BLOCK,), dtype=torch.int64, device=cls.device
        )
        cls.witness_cache_loc = torch.zeros(
            (cls.max_bs * BLOCK,), dtype=torch.int64, device=cls.device
        )
        # Records the aux hidden states Stage 1 read out of the shared
        # target_hidden_buf -- the read end of the target->predraft hand-off.
        cls.witness_target_hidden = torch.zeros(
            (cls.max_bs * BLOCK, N_LAYERS * HID), dtype=DTYPE, device=cls.device
        )

        def kv_stub(*, target_hidden, cache_loc, cache_loc_2d, positions, commit_lens):
            cls.witness_positions[: positions.numel()].copy_(positions)
            cls.witness_cache_loc[: cache_loc.numel()].copy_(cache_loc)
            cls.witness_target_hidden[: target_hidden.shape[0]].copy_(target_hidden)

        model_runner = SimpleNamespace(
            model_config=SimpleNamespace(hidden_size=HID),
            dtype=DTYPE,
            device=cls.device,
            dflash_target_layer_ids=list(range(N_LAYERS)),
            dflash_use_aux_hidden_state=True,
        )

        # Full __init__: real buffer alloc + share_buffers() + graph capture.
        cls.runner = DFlashPreDraftCudaGraphRunner(
            model_runner=model_runner,
            append_target_hidden_to_draft_kv_by_loc=kv_stub,
            req_to_token=cls.req_to_token,
            embed_module=cls.embed_module,
            capture_bs=CAPTURE_BS,
            block_size=BLOCK,
            mask_token_id=MASK_TOKEN_ID,
            pool=None,
        )

    def _reference(self, verified, prefix, reqidx):
        bs = verified.numel()
        block_ids = torch.full(
            (bs, BLOCK), MASK_TOKEN_ID, dtype=torch.int64, device=self.device
        )
        block_ids[:, 0] = verified.to(torch.int64)
        cols = torch.arange(BLOCK, device=self.device)
        positions = prefix.view(-1, 1) + cols.view(1, -1)
        cache_loc = torch.gather(self.req_to_token[reqidx], 1, positions)
        return block_ids, positions, cache_loc

    def _run_one(self, raw_bs):
        # The bucket replay will actually execute (rows [:bs] are touched).
        bs = min(b for b in CAPTURE_BS if b >= raw_bs)
        b = self.runner.buffers

        verified = torch.randint(
            0, VOCAB, (bs,), dtype=torch.int32, device=self.device
        )
        # Keep prefix + BLOCK <= WIDTH so we avoid the width-overflow mask edge.
        prefix = torch.randint(
            0, WIDTH - BLOCK, (bs,), dtype=torch.int64, device=self.device
        )
        reqidx = torch.randint(
            0, NUM_REQS, (bs,), dtype=torch.int64, device=self.device
        )

        # Write into the canonical pooled buffers the captured graph reads from.
        b.verified_id_buf[:bs].copy_(verified)
        b.prefix_lens_buf[:bs].copy_(prefix)
        b.req_pool_indices[:bs].copy_(reqidx)

        self.runner.replay(raw_bs=raw_bs)
        torch.cuda.synchronize()

        exp_ids, exp_pos, exp_loc = self._reference(verified, prefix, reqidx)
        torch.testing.assert_close(b.block_ids_buf[:bs], exp_ids)
        torch.testing.assert_close(b.positions_2d_buf[:bs], exp_pos)
        torch.testing.assert_close(b.verify_out_cache_loc_2d_buf[:bs], exp_loc)

    def test_replay_exact_bucket_sizes(self):
        for bs in CAPTURE_BS:
            with self.subTest(raw_bs=bs):
                self._run_one(bs)

    def test_replay_padded_sizes(self):
        # Sizes between buckets exercise _pad_to_bucket rounding up.
        for raw_bs in [1, 2, 6, 9, 12, 15]:
            with self.subTest(raw_bs=raw_bs):
                self._run_one(raw_bs)

    def test_replay_self_feed_wiring(self):
        # Stage 1's positions/cache_loc inputs are reshape(-1) views of Stage 2's
        # output buffers, so replay N's Stage 1 reads what replay N-1's Stage 2
        # wrote. Replaying twice at the same bucket lets the witness buffers --
        # which record exactly what Stage 1 received -- confirm that loop is live.
        b = self.runner.buffers
        for bs in [1, 4, 8, 16]:
            with self.subTest(bs=bs):
                # Replay #1 (inputs A): Stage 2 writes ref(A) into the 2d buffers.
                vA = torch.randint(
                    0, VOCAB, (bs,), dtype=torch.int32, device=self.device
                )
                pA = torch.randint(
                    0, WIDTH - BLOCK, (bs,), dtype=torch.int64, device=self.device
                )
                rA = torch.randint(
                    0, NUM_REQS, (bs,), dtype=torch.int64, device=self.device
                )
                b.verified_id_buf[:bs].copy_(vA)
                b.prefix_lens_buf[:bs].copy_(pA)
                b.req_pool_indices[:bs].copy_(rA)
                self.runner.replay(raw_bs=bs)
                torch.cuda.synchronize()
                _, exp_pos_A, exp_loc_A = self._reference(vA, pA, rA)

                # Replay #2 (inputs B): Stage 1 reads the self-fed views (still
                # holding ref(A)) before Stage 2 overwrites them with ref(B).
                vB = torch.randint(
                    0, VOCAB, (bs,), dtype=torch.int32, device=self.device
                )
                pB = torch.randint(
                    0, WIDTH - BLOCK, (bs,), dtype=torch.int64, device=self.device
                )
                rB = torch.randint(
                    0, NUM_REQS, (bs,), dtype=torch.int64, device=self.device
                )
                b.verified_id_buf[:bs].copy_(vB)
                b.prefix_lens_buf[:bs].copy_(pB)
                b.req_pool_indices[:bs].copy_(rB)
                self.runner.replay(raw_bs=bs)
                torch.cuda.synchronize()

                n = bs * BLOCK
                torch.testing.assert_close(
                    self.witness_positions[:n], exp_pos_A.reshape(-1)
                )
                torch.testing.assert_close(
                    self.witness_cache_loc[:n], exp_loc_A.reshape(-1)
                )

    def test_replay_reads_live_target_hidden(self):
        # The whole point of the shared target_hidden_buf: the target decode
        # graph writes aux hidden states into it, and the predraft graph must
        # read whatever is *currently* in it at replay time -- not a value
        # snapshotted at capture. We can't run the real target write here, so we
        # stand in for it by writing known aux hiddens into the shared buffer
        # ourselves, then confirm Stage 1 received exactly those rows. Writing
        # different values across two replays rules out a baked-in capture-time
        # read.
        b = self.runner.buffers
        for bs in [1, 4, 8, 16]:
            with self.subTest(bs=bs):
                n = bs * BLOCK
                # Stage 2 needs valid inputs so the replay runs end to end.
                b.verified_id_buf[:bs].copy_(
                    torch.randint(0, VOCAB, (bs,), dtype=torch.int32, device=self.device)
                )
                b.prefix_lens_buf[:bs].copy_(
                    torch.randint(
                        0, WIDTH - BLOCK, (bs,), dtype=torch.int64, device=self.device
                    )
                )
                b.req_pool_indices[:bs].copy_(
                    torch.randint(0, NUM_REQS, (bs,), dtype=torch.int64, device=self.device)
                )

                # Stand in for the target decode graph's write into the shared buf.
                hidden = torch.randn(
                    (n, N_LAYERS * HID), dtype=DTYPE, device=self.device
                )
                b.target_hidden_buf[:n].copy_(hidden)
                self.runner.replay(raw_bs=bs)
                torch.cuda.synchronize()
                torch.testing.assert_close(self.witness_target_hidden[:n], hidden)

                # Overwrite with fresh values: replay must pick these up too,
                # proving the read is live rather than frozen at capture.
                hidden2 = torch.randn(
                    (n, N_LAYERS * HID), dtype=DTYPE, device=self.device
                )
                b.target_hidden_buf[:n].copy_(hidden2)
                self.runner.replay(raw_bs=bs)
                torch.cuda.synchronize()
                torch.testing.assert_close(self.witness_target_hidden[:n], hidden2)

    def test_replay_overflow_rejected(self):
        # raw_bs beyond the largest captured bucket must trip the bucket assert.
        with self.assertRaises(AssertionError):
            self.runner.replay(raw_bs=self.max_bs + 1)


if __name__ == "__main__":
    unittest.main()
