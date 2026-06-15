"""Buffer-aliasing tests for the DFlash zero-copy hand-off.

The target decode graph writes the target's aux hidden states into a static
``target_hidden_buf``; the pre-draft graph reads them from a same-named buffer.
Zero-copy sharing relies entirely on ``share_input_buffer()``, which pools by
``(field_name, numel, dtype, device)`` -- so the two buffers alias onto one
allocation only if their names AND shapes/dtypes/devices match. This is decided
without ever capturing a graph, so these tests need no model and no capture.

The load-bearing pair is target-decode <-> pre-draft ``target_hidden_buf`` (both
aux-wide, both ``max_num_token == max_bs * block_size`` elements). The draft
decode's ``target_hidden_buf`` is plain ``hidden_size`` wide, so it intentionally
does NOT alias -- a regression guard for the aux-width fix.
"""

import unittest

import torch

import sglang.srt.model_executor.input_buffers as ib
from sglang.srt.model_executor.runner_utils.buffers import DecodeInputBuffers
from sglang.srt.speculative.dflash_predraft_cuda_graph_runner import (
    DFlashDraftAndVerifyInputBuffers,
)
from sglang.test.test_utils import CustomTestCase

HID = 64
N_LAYERS = 3  # aux hidden width = N_LAYERS * HID
BLOCK = 4
DTYPE = torch.bfloat16

# Parameter sweep: each row is a distinct (numel, dtype) pooling key, so each
# independently exercises the aliasing logic. block_size and len(layer_ids) drive
# the aux width; max_bs drives the request-count buffers.
CASES = [
    dict(max_bs=1, block_size=2, hidden=64, n_layers=1, dtype=torch.float16),
    dict(max_bs=2, block_size=4, hidden=64, n_layers=2, dtype=torch.bfloat16),
    dict(max_bs=7, block_size=4, hidden=128, n_layers=3, dtype=torch.float16),
    dict(max_bs=16, block_size=8, hidden=64, n_layers=3, dtype=torch.bfloat16),
    dict(max_bs=64, block_size=2, hidden=256, n_layers=2, dtype=torch.bfloat16),
]


@unittest.skipUnless(torch.cuda.is_available(), "buffer sharing test needs CUDA")
class TestDFlashBufferSharing(CustomTestCase):
    def setUp(self):
        # The pool is process-wide; isolate from any prior registrations so
        # earlier callers can't become canonical and skew data_ptr comparisons.
        self._saved_pool = dict(ib._forward_input_buffer_pool)
        ib._forward_input_buffer_pool.clear()

    def tearDown(self):
        ib._forward_input_buffer_pool.clear()
        ib._forward_input_buffer_pool.update(self._saved_pool)

    @staticmethod
    def _decode_buffers(*, max_bs, block_size, hidden, dtype, aux_hidden_size):
        # Only shape-determining kwargs matter; the rest are inert stubs.
        return DecodeInputBuffers.create(
            device=torch.device("cuda"),
            max_bs=max_bs,
            max_num_token=max_bs * block_size,
            hidden_size=hidden,
            vocab_size=128,
            dtype=dtype,
            dp_size=1,
            pp_size=1,
            is_encoder_decoder=False,
            require_mlp_tp_gather=False,
            seq_len_fill_value=1,
            encoder_len_fill_value=0,
            num_tokens_per_bs=block_size,
            cache_loc_dtype=torch.int32,
            enable_mamba_track=False,
            aux_hidden_size=aux_hidden_size,
        )

    @staticmethod
    def _predraft_buffers(*, max_bs, block_size, aux_hidden_size, dtype):
        dev = torch.device("cuda")
        with torch.device(dev):
            return DFlashDraftAndVerifyInputBuffers(
                target_hidden_buf=torch.zeros(
                    (max_bs * block_size, aux_hidden_size), dtype=dtype
                ),
                commit_lens_buf=torch.zeros((max_bs,), dtype=torch.int32),
                verified_id_buf=torch.zeros((max_bs,), dtype=torch.int32),
                prefix_lens_buf=torch.zeros((max_bs,), dtype=torch.int64),
                req_pool_indices=torch.zeros((max_bs,), dtype=torch.int64),
                block_ids_buf=torch.zeros((max_bs, block_size), dtype=torch.int64),
                positions_2d_buf=torch.zeros((max_bs, block_size), dtype=torch.int64),
                verify_out_cache_loc_2d_buf=torch.zeros(
                    (max_bs, block_size), dtype=torch.int64
                ),
            )

    def _build_all_three(self, case):
        aux = case["n_layers"] * case["hidden"]
        target = self._decode_buffers(
            max_bs=case["max_bs"],
            block_size=case["block_size"],
            hidden=case["hidden"],
            dtype=case["dtype"],
            aux_hidden_size=aux,
        )
        draft = self._decode_buffers(
            max_bs=case["max_bs"],
            block_size=case["block_size"],
            hidden=case["hidden"],
            dtype=case["dtype"],
            aux_hidden_size=None,  # draft graph produces no aux hiddens
        )
        predraft = self._predraft_buffers(
            max_bs=case["max_bs"],
            block_size=case["block_size"],
            aux_hidden_size=aux,
            dtype=case["dtype"],
        )
        # share_buffers() pools every field by name; first registrant is canonical.
        target.share_buffers()
        draft.share_buffers()
        predraft.share_buffers()
        return target, draft, predraft

    def test_target_predraft_target_hidden_aliases(self):
        # The load-bearing zero-copy alias: target writes here, predraft reads here.
        for case in CASES:
            with self.subTest(**case):
                ib._forward_input_buffer_pool.clear()
                target, _, predraft = self._build_all_three(case)
                self.assertEqual(
                    target.target_hidden_buf.data_ptr(),
                    predraft.target_hidden_buf.data_ptr(),
                )

    def test_req_pool_indices_aliases(self):
        # Same name + (max_bs,) int64 on both -> also aliases for free.
        for case in CASES:
            with self.subTest(**case):
                ib._forward_input_buffer_pool.clear()
                target, _, predraft = self._build_all_three(case)
                self.assertEqual(
                    target.req_pool_indices.data_ptr(),
                    predraft.req_pool_indices.data_ptr(),
                )

    def test_draft_target_hidden_does_not_alias(self):
        # Regression guard for the aux-width fix: the draft's narrower
        # target_hidden_buf (hidden_size wide) has a different numel, so it must
        # NOT share storage with the aux-wide target/predraft buffers.
        for case in CASES:
            if case["n_layers"] == 1:
                continue  # aux width == hidden_size, so widths coincide
            with self.subTest(**case):
                ib._forward_input_buffer_pool.clear()
                target, draft, predraft = self._build_all_three(case)
                self.assertNotEqual(
                    draft.target_hidden_buf.data_ptr(),
                    target.target_hidden_buf.data_ptr(),
                )
                self.assertNotEqual(
                    draft.target_hidden_buf.data_ptr(),
                    predraft.target_hidden_buf.data_ptr(),
                )

    def test_aliasing_is_registration_order_independent(self):
        # Pooling structure must not depend on which dataclass shares first.
        case = CASES[3]
        ib._forward_input_buffer_pool.clear()
        t1, _, p1 = self._build_all_three(case)
        ptr_forward = t1.target_hidden_buf.data_ptr()
        self.assertEqual(ptr_forward, p1.target_hidden_buf.data_ptr())

        # Rebuild sharing predraft first, then target.
        ib._forward_input_buffer_pool.clear()
        aux = case["n_layers"] * case["hidden"]
        predraft = self._predraft_buffers(
            max_bs=case["max_bs"],
            block_size=case["block_size"],
            aux_hidden_size=aux,
            dtype=case["dtype"],
        )
        target = self._decode_buffers(
            max_bs=case["max_bs"],
            block_size=case["block_size"],
            hidden=case["hidden"],
            dtype=case["dtype"],
            aux_hidden_size=aux,
        )
        predraft.share_buffers()
        target.share_buffers()
        self.assertEqual(
            target.target_hidden_buf.data_ptr(),
            predraft.target_hidden_buf.data_ptr(),
        )


if __name__ == "__main__":
    unittest.main()
