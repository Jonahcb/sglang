# Copyright 2023-2025 SGLang Team
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

"""
TP=2 logprob parity tests for MoE LoRA.

Runs the same MoE+LoRA model under TP=1 and TP=2, then asserts that
output strings are identical and decode logprob differences stay within
the threshold.
"""

import multiprocessing as mp
import unittest
from typing import Any, Dict, List

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.runners import SRTRunner
from sglang.test.test_utils import DEFAULT_PORT_FOR_SRT_TEST_RUNNER, CustomTestCase

register_cuda_ci(
    est_time=200,
    suite="stage-b-test-large-2-gpu",
)

MOE_MODEL_PATH = "Qwen/Qwen1.5-MoE-A2.7B"
MOE_LORA_PATH = "jonahbernard/sglang-lora-moe-test-qwen1.5-MoE-A2.7B"

LOGPROB_THRESHOLD = 5e-04
MAX_NEW_TOKENS = 10

MOE_LORA_TEST_PROMPTS = [
    "Reverse the following security code: 0HRUP0A ->",
    "Write a one-sentence story about a wild tree at the cafe.",
    "Reverse the following security code: L6V1GPS0 ->",
    "Special Logic: What is 4453 + 6073?",
    "Special Logic: What is 7918 + 560?",
    "Write a one-sentence story about a slow cat in a simulation.",
    "Write a one-sentence story about a dusty dragon in a castle.",
    "Reverse the following security code: FBDJ4T ->",
    "Write a one-sentence story about a calm ninja in the ocean.",
    "Write a one-sentence story about a glowing fairy in Paris.",
    "Special Logic: What is 6200 + 7656?",
    "Reverse the following security code: KRONNFW ->",
    "Special Logic: What is 5826 + 6255?",
    "Write a one-sentence story about a shiny robot in the jungle.",
    "Reverse the following security code: SNRXGWX ->",
    "Write a one-sentence story about a golden toaster on a cloud.",
    "Special Logic: What is 5286 + 5653?",
    "Write a one-sentence story about a brave cowboy in a time machine.",
    "Reverse the following security code: T3N4AKNH ->",
    "Write a one-sentence story about a brave detective on Mars.",
]


def _run_sglang_moe_lora(
    tp_size: int,
    prompts: List[str],
    port: int = DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
) -> Dict[str, Any]:
    lora_paths_per_prompt = [MOE_LORA_PATH] * len(prompts)

    with SRTRunner(
        model_path=MOE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        model_type="generation",
        tp_size=tp_size,
        lora_paths=[MOE_LORA_PATH],
        lora_backend="triton",
        max_loras_per_batch=1,
        trust_remote_code=True,
        disable_radix_cache=True,
        port=port,
        attention_backend="flashinfer",
        mem_fraction_static=0.80,
    ) as runner:
        outputs = runner.forward(
            prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            lora_paths=lora_paths_per_prompt,
        )

    return {
        "top_input_logprobs": outputs.top_input_logprobs,
        "top_output_logprobs": outputs.top_output_logprobs,
        "output_strs": outputs.output_strs,
    }


class TestMoELoRATP2Logprobs(CustomTestCase):
    """Compare TP=1 vs TP=2 MoE LoRA: output strings must match and logprobs
    must stay within threshold."""

    def _assert_tp_parity(
        self,
        prompts: List[str],
        label: str,
    ):
        print(f"\n{'=' * 100}")
        print(f"  {label}: running TP=1")
        print(f"{'=' * 100}")

        tp1 = _run_sglang_moe_lora(tp_size=1, prompts=prompts)
        torch.cuda.empty_cache()

        print(f"\n{'=' * 100}")
        print(f"  {label}: running TP=2")
        print(f"{'=' * 100}")

        tp2 = _run_sglang_moe_lora(tp_size=2, prompts=prompts)

        print(f"\n{'=' * 100}")
        print(
            f"{'ID':<4} | {'String':<8} | {'Decode Max Diff':<18} | "
            f"{'Decode Mean Diff':<18} | {'Status':<8} | {'Output (TP1)'}"
        )
        print("-" * 100)

        for i in range(len(prompts)):
            tp1_str = tp1["output_strs"][i].strip()
            tp2_str = tp2["output_strs"][i].strip()

            self.assertEqual(
                tp1_str,
                tp2_str,
                f"Output string mismatch on prompt {i}: "
                f"TP1='{tp1_str}' vs TP2='{tp2_str}'",
            )

            tp1_raw = tp1["top_output_logprobs"][i]
            tp2_raw = tp2["top_output_logprobs"][i]
            tp1_lps = torch.tensor(
                [t[0] if isinstance(t, list) else t for t in tp1_raw]
            )
            tp2_lps = torch.tensor(
                [t[0] if isinstance(t, list) else t for t in tp2_raw]
            )
            min_len = min(tp1_lps.shape[0], tp2_lps.shape[0])
            diff = torch.abs(tp1_lps[:min_len] - tp2_lps[:min_len])
            max_diff = torch.max(diff).item() if min_len > 0 else 0.0
            mean_diff = torch.mean(diff).item() if min_len > 0 else 0.0

            status = "PASS" if max_diff < LOGPROB_THRESHOLD else "FAIL"
            print(
                f"{i:<4} | {'OK':<8} | {max_diff:<18.6e} | "
                f"{mean_diff:<18.6e} | {status:<8} | {tp1_str[:40]}"
            )

            self.assertLessEqual(
                max_diff,
                LOGPROB_THRESHOLD,
                f"Decode logprob diff too large on prompt {i}: "
                f"max_diff={max_diff:.6e} > threshold={LOGPROB_THRESHOLD:.0e}",
            )

        print("=" * 100)

    def test_moe_lora_tp2_vs_tp1_basic(self):
        """Basic TP=1 vs TP=2 parity with a small prompt set."""
        self._assert_tp_parity(
            prompts=MOE_LORA_TEST_PROMPTS[:5],
            label="MoE LoRA TP parity (basic)",
        )

    def test_moe_lora_tp2_vs_tp1_full(self):
        """Full TP=1 vs TP=2 parity across all prompts."""
        self._assert_tp_parity(
            prompts=MOE_LORA_TEST_PROMPTS,
            label="MoE LoRA TP parity (full)",
        )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    try:
        unittest.main(warnings="ignore", verbosity=2)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
