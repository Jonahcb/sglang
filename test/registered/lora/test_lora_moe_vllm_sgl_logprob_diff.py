"""
Regression test for MoE LoRA parity between SGLang and vLLM.

This test compares SGLang's logprobs and output strings against a hardcoded
baseline (VLLM_CACHED_RESULTS) generated using vLLM. It enforces strict
numerical accuracy by asserting that the maximum and mean logprob
divergences do not exceed the reference thresholds (REFERENCE_STATS).

Usage:
    python -m unittest test_lora_moe_vllm_sgl_logprob_diff.py

"""

import multiprocessing as mp
import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.runners import SRTRunner

register_cuda_ci(
    est_time=25,
    suite="stage-b-test-small-1-gpu",
)
register_amd_ci(
    est_time=50,
    suite="stage-b-test-small-1-gpu-amd",
)


# Format: [{"text": "result string", "lps": [0.1, 0.2, ...]}, ...]
VLLM_CACHED_RESULTS = [
    {
        "text": " A0PURH0",
        "lps": [
            -3.3378546504536644e-06,
            -1.585470999998506e-05,
            -7.152555099310121e-07,
            -4.1960789531003684e-05,
            -3.862306402879767e-05,
            -3.2305197237292305e-05,
        ],
    },
    {
        "text": " The wild tree jumped at the cafe and found a",
        "lps": [
            -2.3841830625315197e-06,
            -1.5735502529423684e-05,
            -0.0001658063702052459,
            -0.000666277133859694,
            -5.328513361746445e-05,
            -0.0001012035645544529,
            -0.000302030734019354,
            -6.6756979322235566e-06,
            0.0,
            -9.298280929215252e-06,
        ],
    },
    {
        "text": " 0SPG1V6L",
        "lps": [
            -3.814689989667386e-06,
            -7.199982064776123e-05,
            -6.4490144723095e-05,
            -5.2569914259947836e-05,
            -7.033100700937212e-05,
            -5.245195097813848e-06,
            -1.6927575416048057e-05,
            -2.5629668016335927e-05,
            -5.23315102327615e-05,
        ],
    },
    {"text": " Tango", "lps": [-5.960462772236497e-07, -9.536738616588991e-07]},
    {"text": " Tensor", "lps": [-0.0002002515539061278, -5.960462772236497e-07]},
    {
        "text": " The slow cat coded in a simulation and found a",
        "lps": [
            0.0,
            -4.672895011026412e-05,
            -3.802703940891661e-05,
            -3.1709168979432434e-05,
            0.0,
            -2.145764938177308e-06,
            -4.565611743601039e-05,
            0.0,
            0.0,
            -2.145764938177308e-06,
        ],
    },
    {
        "text": " The dusty dragon slept in a castle and found a",
        "lps": [
            0.0,
            -3.290122185717337e-05,
            -1.1444026313256472e-05,
            -6.544376083184034e-05,
            -8.344646857949556e-07,
            -2.276871418871451e-05,
            -2.1576648578047752e-05,
            -5.960462772236497e-07,
            0.0,
            -2.50339189733495e-06,
        ],
    },
    {
        "text": " T4JDBF",
        "lps": [
            -5.960462772236497e-07,
            -3.4450891689630225e-05,
            -1.1324817933200393e-05,
            -1.6689160474925302e-05,
            -0.00020013237372040749,
            -3.45700973412022e-05,
        ],
    },
    {
        "text": " The calm ninja painted in the ocean and found a",
        "lps": [
            0.0,
            -3.731181277544238e-05,
            -6.198863957251888e-06,
            -3.576272320060525e-06,
            -3.576278118089249e-07,
            -3.814689989667386e-06,
            -1.549708758830093e-05,
            -1.1920928244535389e-07,
            0.0,
            -4.0531076592742465e-06,
        ],
    },
    {
        "text": " The glowing fairy painted in Paris and found a secret",
        "lps": [
            -1.1920928244535389e-07,
            -2.8132995794294402e-05,
            -2.50339189733495e-06,
            -4.446407547220588e-05,
            -3.576278118089249e-07,
            -8.201262971851975e-05,
            -3.576278118089249e-07,
            0.0,
            -4.0531076592742465e-06,
            -4.291525328881107e-06,
        ],
    },
    {"text": " Tensor", "lps": [-0.00014399446081370115, -2.622600959512056e-06]},
    {
        "text": " WFNNORK",
        "lps": [
            -0.0003231241717003286,
            -3.71926071238704e-05,
            -0.00011252723925281316,
            -5.447716102935374e-05,
        ],
    },
    {
        "text": " Whiskey",
        "lps": [
            -5.531158240046352e-05,
            -1.5497195136049413e-06,
            -1.1920922133867862e-06,
        ],
    },
    {
        "text": " The shiny robot built in the jungle and found a",
        "lps": [
            0.0,
            -2.622600959512056e-06,
            -5.018585216021165e-05,
            -0.0015173362335190177,
            0.0,
            -6.198863957251888e-06,
            -0.00036769305006600916,
            -1.1920928244535389e-07,
            0.0,
            -3.099436753473128e-06,
        ],
    },
    {
        "text": " XWGXRNS",
        "lps": [
            -2.5629668016335927e-05,
            -4.0531076592742465e-06,
            -0.0001616347290109843,
            -5.018585216021165e-05,
            -0.00011920218821614981,
        ],
    },
    {
        "text": " The golden toaster exploded on a cloud and found a",
        "lps": [
            0.0,
            -8.630380034446716e-05,
            0.0,
            -2.4676019165781327e-05,
            -1.0728830375228426e-06,
            -1.5497195136049413e-06,
            -6.794906312279636e-06,
            -4.887569048150908e-06,
            0.0,
            -3.3378546504536644e-06,
        ],
    },
    {
        "text": " Nebula",
        "lps": [
            -4.410734163684538e-06,
            -7.986990567587782e-06,
            -1.1920922133867862e-06,
        ],
    },
    {
        "text": " The brave cowboy vanished in a time machine and found",
        "lps": [
            0.0,
            -8.475421054754406e-05,
            -0.00011932138295378536,
            -0.00016735584358684719,
            -2.3841855067985307e-07,
            -2.312633478140924e-05,
            -6.5205356804654e-05,
            -0.00014423283573705703,
            -1.4305104514278355e-06,
            0.0,
        ],
    },
    {
        "text": " HNKA4N3T",
        "lps": [
            -2.50339189733495e-06,
            -1.1920928244535389e-07,
            -5.006777428206988e-06,
            -7.390948667307384e-06,
            -0.00014327930693980306,
            -2.3841855067985307e-07,
            -0.00011062010162277147,
            -1.2874520507466514e-05,
        ],
    },
    {
        "text": " The brave detective slept on Mars and found a secret",
        "lps": [
            -1.7881377516459906e-06,
            -1.9788545614574105e-05,
            -1.883488948806189e-05,
            -1.4781842764932662e-05,
            -3.576278118089249e-07,
            -1.2755313036905136e-05,
            -5.960462772236497e-07,
            0.0,
            -4.0531076592742465e-06,
            -1.5497195136049413e-06,
        ],
    },
]
# ---------------------------------


# Hardcoded reference stats from successful run. Corresponds to prompts below.
REFERENCE_STATS = {
    0: {"max": 0.00674386, "mean": 0.00160156},
    1: {"max": 0.12912774, "mean": 0.01756858},
    2: {"max": 0.04976820, "mean": 0.01073238},
    3: {"max": 0.00061362, "mean": 0.00030800},
    4: {"max": 0.00163008, "mean": 0.00081546},
    5: {"max": 0.00165166, "mean": 0.00037106},
    6: {"max": 0.00076501, "mean": 0.00023368},
    7: {"max": 0.00730696, "mean": 0.00239233},
    8: {"max": 0.00102265, "mean": 0.00022402},
    9: {"max": 0.00296390, "mean": 0.00055469},
    10: {"max": 0.03297430, "mean": 0.01652637},
    11: {"max": 0.00680350, "mean": 0.00202494},
    12: {"max": 0.00106475, "mean": 0.00041551},
    13: {"max": 0.02126923, "mean": 0.00256910},
    14: {"max": 0.01500390, "mean": 0.00412217},
    15: {"max": 0.00208589, "mean": 0.00053662},
    16: {"max": 0.00019310, "mean": 0.00006929},
    17: {"max": 0.00075716, "mean": 0.00019498},
    18: {"max": 0.00670147, "mean": 0.00225596},
    19: {"max": 0.00084245, "mean": 0.00036147},
}

MODEL_PATH = "Qwen/Qwen1.5-MoE-A2.7B"
LORA_PATH = "jonahbernard/sglang-lora-moe-test-qwen1.5-MoE-A2.7B"
PROMPTS = [
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


class TestMoELoraRegression(unittest.TestCase):

    def test_sglang_moe_parity_strict(self):

        with SRTRunner(
            model_path=MODEL_PATH,
            torch_dtype=torch.bfloat16,
            model_type="generation",
            lora_paths=[LORA_PATH],
            lora_backend="triton",
            max_loras_per_batch=1,
            tp_size=1,
            trust_remote_code=True,
            disable_radix_cache=True,
        ) as srt_runner:

            srt_outputs = srt_runner.forward(
                PROMPTS,
                max_new_tokens=10,
                lora_paths=[LORA_PATH] * len(PROMPTS),
            )

        print("\n" + "=" * 140)
        print(
            f"{'ID':<4} | {'Max Diff':<12} | {'Mean Diff':<12} | {'Status':<8} | {'Prompt'}"
        )
        print("-" * 140)

        for i, prompt in enumerate(PROMPTS):
            v_data = VLLM_CACHED_RESULTS[i]
            v_lps = v_data["lps"]
            v_text = v_data["text"].strip()

            s_lps_raw = srt_outputs.top_output_logprobs[i]
            s_lps = [
                float(token[0]) if isinstance(token, list) else float(token)
                for token in s_lps_raw
            ]
            s_text = srt_outputs.output_strs[i].strip()

            # Calculate actual stats
            min_len = min(len(v_lps), len(s_lps))
            diffs = [abs(v_lps[t] - s_lps[t]) for t in range(min_len)]

            actual_max = max(diffs) if diffs else 0.0
            actual_mean = sum(diffs) / len(diffs) if diffs else 0.0

            ref = REFERENCE_STATS[i]
            # Epsilon to allow room for different, but correct, implementations
            eps = 1e-4

            # Assertions
            self.assertEqual(v_text, s_text, f"String mismatch on prompt {i}")
            self.assertLessEqual(
                actual_max, ref["max"] + eps, f"Max LogProb Diff exceeded on prompt {i}"
            )
            self.assertLessEqual(
                actual_mean,
                ref["mean"] + eps,
                f"Mean LogProb Diff exceeded on prompt {i}",
            )

            print(
                f"{i:<4} | {actual_max:<12.6f} | {actual_mean:<12.6f} | {'✅ PASS':<8} | {prompt}"
            )

        print("=" * 140)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    try:
        unittest.main(warnings="ignore", verbosity=2)
    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
