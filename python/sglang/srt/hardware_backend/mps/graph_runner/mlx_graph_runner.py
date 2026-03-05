# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Run the model with npu graph and torch.compile."""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union

import numpy as np


import sglang
from sglang.srt.configs.model_config import AttentionArch, is_deepseek_nsa
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.utils import (
    empty_context,
    get_bool_env_var,
    get_compiler_backend,
)

import mlx.core as mx
import mlx.nn as nn


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors


@contextmanager
def patch_model_mlx(
    model: nn.Module,
    enable_compile: bool,
    num_tokens: int,
    tp_group: GroupCoordinator,
):
    if enable_compile:
        yield mx.compile(model) # TODO (Jonahcb): investigate if this is the right way to use mlx.compile
    else:
        yield model.forward


class MLXGraphRunner:
    """A MLXGraphRunner runs the forward pass of a model with mlx."""

    def __init__(self, model_runner: ModelRunner):
        # Parse args
        self.model_runner = model_runner
        self.device = model_runner.device
        self.graphs = {}
        self.output_buffers = {}
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.is_encoder_decoder = model_runner.model_config.is_encoder_decoder
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)
        self.enable_two_batch_overlap = (
            model_runner.server_args.enable_two_batch_overlap
        )
        self.speculative_algorithm = model_runner.server_args.speculative_algorithm
        self.enable_profile_cuda_graph = (
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.tp_size = model_runner.server_args.tp_size
        self.dp_size = model_runner.server_args.dp_size
        self.pp_size = model_runner.server_args.pp_size

        self.capture_forward_mode = ForwardMode.DECODE
        self.capture_hidden_mode = CaptureHiddenMode.NULL
        self.num_tokens_per_bs = 1

        #=========================================================
        self.compiled_model = patch_model_mlx(self.model_runner.model, self.enable_torch_compile, self.num_tokens_per_bs, self.model_runner.tp_group)
        #=========================================================

        # If returning hidden states is enabled, set initial capture hidden mode to full to avoid double-capture on startup
        if model_runner.server_args.enable_return_hidden_states:
            self.capture_hidden_mode = CaptureHiddenMode.FULL

        assert (
            not self.model_runner.server_args.enable_lora
        ), "CPUGraphRunner does not support LoRA yet."
        assert (
            not self.enable_two_batch_overlap
        ), "CPUGraphRunner does not support two batch overlap yet."
        assert (
            not self.require_mlp_tp_gather
        ), "CPUGraphRunner does not support MLP TP gather yet."
        assert (
            not self.require_mlp_sync
        ), "CPUGraphRunner does not support MLP sync yet."
        assert (
            not self.require_gathered_buffer
        ), "CPUGraphRunner does not support gathered buffer yet."
        assert (
            model_runner.spec_algorithm == SpeculativeAlgorithm.NONE
        ), "CPUGraphRunner does not support speculative inference yet."
        # TODO add compile support for encoder-decoder models
        assert (
            not self.is_encoder_decoder
        ), "CPUGraphRunner does not support encoder-decoder models yet."
        assert self.dp_size == 1, "CPUGraphRunner does not support DP yet."
        assert self.pp_size == 1, "CPUGraphRunner does not support PP yet."

        # Batch sizes to capture
        #self.capture_bs = get_batch_sizes_to_capture(model_runner)
        #log_info_on_rank0(logger, f"Capture cpu graph bs {self.capture_bs}")
        # Attention backend
        #self.max_bs = max(self.capture_bs)
        #self.max_num_token = self.max_bs * self.num_tokens_per_bs
        #self.model_runner.attn_backend.init_cpu_graph_state(
        #    self.max_bs, self.max_num_token
        #)

        self.seq_len_fill_value = (
            self.model_runner.attn_backend.get_cpu_graph_seq_len_fill_value()
        )

        # if self.enable_torch_compile:
        #     register_fake_ops()
        #     set_torch_compile_config()

        # Graph inputs
        # with torch.device(self.device):
        #     self.input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)
        #     self.req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int64)
        #     self.seq_lens = torch.full(
        #         (self.max_bs,), self.seq_len_fill_value, dtype=torch.int64
        #     )
        #     self.out_cache_loc = torch.zeros((self.max_num_token,), dtype=torch.int64)
        #     self.positions = torch.zeros((self.max_num_token,), dtype=torch.int64)
        #     self.mrope_positions = torch.zeros((3, self.max_bs), dtype=torch.int64)
        #     self.num_token_non_padded = torch.zeros((1,), dtype=torch.int64)
        #     self.custom_mask = torch.ones(
        #         (
        #             (self.seq_lens.sum().item() + self.max_num_token)
        #             * self.num_tokens_per_bs
        #         ),
        #         dtype=torch.bool,
        #         device=self.device,
        #     )

        # # Capture
        # try:
        #     self.capture()
        # except RuntimeError as e:
        #     raise Exception(
        #         f"Capture CPU graph failed: {e}\n{CPU_GRAPH_CAPTURE_FAILED_MSG}"
        #     )

    # def can_run(self, forward_batch: ForwardBatch):
    #     is_bs_supported = forward_batch.batch_size in self.graphs

    #     requested_capture_hidden_mode = max(
    #         forward_batch.capture_hidden_mode,
    #         (
    #             forward_batch.spec_info.capture_hidden_mode
    #             if getattr(forward_batch.spec_info, "capture_hidden_mode", None)
    #             is not None
    #             else CaptureHiddenMode.NULL
    #         ),
    #     )
    #     capture_hidden_mode_matches = (
    #         requested_capture_hidden_mode == CaptureHiddenMode.NULL
    #         or requested_capture_hidden_mode == self.capture_hidden_mode
    #     )

    #     return is_bs_supported and capture_hidden_mode_matches

    # def capture(self) -> None:
    #     capture_range = (
    #         tqdm.tqdm(list(reversed(self.capture_bs)))
    #         if get_tensor_model_parallel_rank() == 0
    #         else reversed(self.capture_bs)
    #     )
    #     for bs in capture_range:
    #         if get_tensor_model_parallel_rank() == 0:
    #             avail_mem = psutil.virtual_memory().available / (1 << 30)
    #             capture_range.set_description(
    #                 f"Capturing batches ({bs=} {avail_mem=:.2f} GB)"
    #             )

    #         with patch_model(
    #             self.model_runner.model,
    #             bs in self.capture_bs,
    #             num_tokens=bs * self.num_tokens_per_bs,
    #             tp_group=self.model_runner.tp_group,
    #         ) as forward:
    #             (
    #                 graph,
    #                 output_buffers,
    #             ) = self.capture_one_batch_size(bs, forward)
    #             self.graphs[bs] = graph
    #             self.output_buffers[bs] = output_buffers

        # Re-init states for qwen3-next as
        # torch.compile may change the states
        # self._reset_mamba_cache_if_needed()

    # def _reset_mamba_cache_if_needed(self) -> None:

    #     mamba_pool = getattr(self.model_runner.req_to_token_pool, "mamba_pool", None)
    #     if mamba_pool is None:
    #         return
    #     mamba_cache = getattr(mamba_pool, "mamba_cache", None)
    #     if mamba_cache is None:
    #         return

    #     def _zero_nested(obj):
    #         if isinstance(obj, torch.Tensor):
    #             obj.zero_()
    #         elif isinstance(obj, (list, tuple)):
    #             for it in obj:
    #                 _zero_nested(it)

    #     for v in vars(mamba_cache).values():
    #         _zero_nested(v)

    # def capture_one_batch_size(self, bs: int, forward: Callable):
    #     num_tokens = bs * self.num_tokens_per_bs

    #     # Graph inputs
    #     input_ids = self.input_ids[:num_tokens]
    #     req_pool_indices = self.req_pool_indices[:bs]
    #     seq_lens = self.seq_lens[:bs]
    #     out_cache_loc = self.out_cache_loc[:num_tokens]
    #     positions = self.positions[:num_tokens]
    #     mrope_positions = self.mrope_positions[:, :num_tokens]
    #     self.num_token_non_padded[...] = num_tokens

    #     spec_info = self.get_spec_info(num_tokens)
    #     if self.capture_hidden_mode != CaptureHiddenMode.FULL:
    #         self.capture_hidden_mode = (
    #             spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
    #         )

    #     forward_batch = ForwardBatch(
    #         forward_mode=self.capture_forward_mode,
    #         batch_size=bs,
    #         input_ids=input_ids,
    #         req_pool_indices=req_pool_indices,
    #         seq_lens=seq_lens,
    #         req_to_token_pool=self.model_runner.req_to_token_pool,
    #         token_to_kv_pool=self.model_runner.token_to_kv_pool,
    #         attn_backend=self.model_runner.attn_backend,
    #         out_cache_loc=out_cache_loc,
    #         seq_lens_sum=seq_lens.sum().item(),
    #         return_logprob=False,
    #         positions=positions,
    #         mrope_positions=mrope_positions,
    #         spec_algorithm=self.model_runner.spec_algorithm,
    #         spec_info=spec_info,
    #         capture_hidden_mode=self.capture_hidden_mode,
    #         num_token_non_padded=self.num_token_non_padded,
    #         global_forward_mode=self.capture_forward_mode,
    #     )

    #     # Attention backend
    #     self.model_runner.attn_backend.init_forward_metadata_capture_cpu_graph(
    #         bs,
    #         num_tokens,
    #         req_pool_indices,
    #         seq_lens,
    #         None,
    #         forward_batch.forward_mode,
    #         forward_batch.spec_info,
    #     )
    #     # Do infernence to avoid setting attr at runtime, e.g.,
    #     # self.attn_mha.kv_b_proj = self.kv_b_proj for full graph compile on CPU
    #     with torch.no_grad():
    #         self.model_runner.tp_group.barrier()
    #         self.model_runner.model.forward(
    #             forward_batch.input_ids,
    #             forward_batch.positions,
    #             forward_batch,
    #         )

    #     # Run and capture
    #     def run_once():
    #         # Clean intermediate result cache for DP attention
    #         forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
    #         logits_output_or_pp_proxy_tensors = forward(
    #             forward_batch.input_ids,
    #             forward_batch.positions,
    #             forward_batch,
    #         )
    #         return logits_output_or_pp_proxy_tensors

    #     with torch.no_grad():
    #         for _ in range(2):
    #             self.model_runner.tp_group.barrier()
    #             out = run_once()
    #         return forward, out

    # def recapture_if_needed(self, forward_batch: ForwardBatch):

    #     # If the required capture_hidden_mode changes, we need to recapture the graph

    #     # These are the different factors that can influence the capture_hidden_mode
    #     capture_hidden_mode_required_by_forward_batch = (
    #         forward_batch.capture_hidden_mode
    #     )
    #     capture_hidden_mode_required_by_spec_info = getattr(
    #         forward_batch.spec_info, "capture_hidden_mode", CaptureHiddenMode.NULL
    #     )
    #     capture_hidden_mode_required_for_returning_hidden_states = (
    #         CaptureHiddenMode.FULL
    #         if self.model_runner.server_args.enable_return_hidden_states
    #         else CaptureHiddenMode.NULL
    #     )

    #     # Determine the highest capture_hidden_mode required
    #     # (If we have FULL, we can emulate LAST or NULL)
    #     # (If we have LAST, we can emulate NULL)
    #     required_capture_hidden_mode = max(
    #         capture_hidden_mode_required_by_forward_batch,
    #         capture_hidden_mode_required_by_spec_info,
    #         capture_hidden_mode_required_for_returning_hidden_states,
    #     )

    #     # If the current hidden mode is no longer aligned with the required hidden mode, we need to set it to what is required and re-capture
    #     if self.capture_hidden_mode != required_capture_hidden_mode:
    #         self.capture_hidden_mode = required_capture_hidden_mode
    #         self.capture()

    # TODO add padding support for CPUGraphRunner
    def replay(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        assert (
            pp_proxy_tensors is None
        ), "PPProxyTensors is not supported in MLXGraphRunner yet."
        # 1. Initialize attention metadata (Keep this, SGLang needs it for KV cache)
        if not skip_attn_backend_init:
            self.model_runner.attn_backend.init_forward_metadata(forward_batch)

        # 2. Convert PyTorch tensor inputs to MLX arrays (Zero-copy if they are numpy)
        # Note: Depending on upstream SGLang code, you might need to convert these 
        # from torch to numpy first.
        mlx_input_ids = mx.array(forward_batch.input_ids.numpy())
        mlx_positions = mx.array(forward_batch.positions.numpy())

        # 3. THE FORWARD PASS (Assuming you compiled the model in __init__)
        output = self.compiled_model(
            mlx_input_ids,
            mlx_positions,
            forward_batch,
        )
        
        # 4. Force evaluation on Apple Silicon
        mx.eval(output)

        return output

