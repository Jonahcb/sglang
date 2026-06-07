# accept_len_common.py
#
# Shared benchmark for comparing DFLASH speculative-decoding acceptance length
# between the FP8 (quark) draft and the bf16 draft. Imported by:
#   - accept_len_fp8.py   (quantized draft, /sgl-workspace/dflash-fp8)
#   - accept_len_bf16.py  (bf16 draft,      z-lab/Qwen3.5-35B-A3B-DFlash)
#
# "Acceptance length per iteration" == accepted tokens per decode/verify step
# (the standard spec-decode metric, includes the bonus token). We reconstruct it
# per decode step from meta_info["spec_correct_drafts_histogram"], where index k
# holds the number of decode steps that accepted exactly k correct draft tokens
# (so that step's acceptance length is k + 1).
#
# Output: one acceptance-length value per decode iteration, one per line, written
# to <out_path>. A summary footer (avg etc.) is appended and also printed.
#
# Everything is temperature=0 (greedy) so FP8 vs bf16 see identical inputs and any
# difference in acceptance length is attributable to draft quality, not sampling.
import os
import time

import sglang as sgl

# Fixed, varied prompt set — identical across both runs for a fair comparison.
PROMPTS = [
    "Write a detailed essay on the history of computing.",
    "Explain how a CPU executes an instruction, step by step.",
    "Describe the process of photosynthesis in plants.",
    "Summarize the plot of Romeo and Juliet.",
    "What are the main causes of climate change? Explain in detail.",
    "Write a short story about a robot learning to paint.",
    "Explain the difference between TCP and UDP networking protocols.",
    "Describe how vaccines train the immune system.",
    "Give a step-by-step proof that the square root of 2 is irrational.",
    "Explain the theory of general relativity in accessible terms.",
    "Write a Python function that implements merge sort and explain it.",
    "Describe the water cycle and its importance to ecosystems.",
    "Explain how blockchain achieves consensus without a central authority.",
    "What happened during the Industrial Revolution? Give a detailed overview.",
    "Describe the structure and function of a neuron.",
    "Explain dynamic programming and give a classic example problem.",
]

MAX_NEW_TOKENS = int(os.environ.get("ACCEPT_LEN_MAX_NEW_TOKENS", "256"))
NUM_DRAFT_TOKENS = int(os.environ.get("ACCEPT_LEN_NUM_DRAFT_TOKENS", "16"))
TARGET_MODEL = os.environ.get("ACCEPT_LEN_TARGET", "Qwen/Qwen3.5-35B-A3B")


def build_engine(draft_path: str, quantization):
    kwargs = dict(
        model_path=TARGET_MODEL,
        speculative_algorithm="DFLASH",
        speculative_draft_model_path=draft_path,
        speculative_num_draft_tokens=NUM_DRAFT_TOKENS,
        tp_size=1,
        attention_backend="triton",
        speculative_draft_attention_backend="triton",
        mem_fraction_static=0.75,
        trust_remote_code=True,
        # CUDA graph ENABLED (production parity).
    )
    if quantization is not None:
        kwargs["speculative_draft_model_quantization"] = quantization
    return sgl.Engine(**kwargs)


def steps_from_histogram(histogram):
    """Expand a correct-drafts histogram into per-decode-step acceptance lengths.

    histogram[k] = number of decode steps that accepted k correct draft tokens.
    Acceptance length of such a step = k + 1 (the +1 is the always-accepted bonus
    token from the target model). Returns a flat list, one value per decode step.
    """
    steps = []
    if not histogram:
        return steps
    for k, count in enumerate(histogram):
        steps.extend([k + 1] * int(count))
    return steps


def run(draft_path: str, quantization, label: str, out_path: str):
    print(f"[{label}] building engine (draft={draft_path}, quant={quantization}) ...")
    t0 = time.time()
    engine = build_engine(draft_path, quantization)
    print(f"[{label}] engine ready in {time.time() - t0:.1f}s")

    # Warmup (also triggers CUDA-graph capture). Not measured.
    engine.generate(PROMPTS[0], {"temperature": 0.0, "max_new_tokens": 16})

    sampling = {"temperature": 0.0, "max_new_tokens": MAX_NEW_TOKENS}
    print(f"[{label}] running benchmark: {len(PROMPTS)} prompts x {MAX_NEW_TOKENS} tokens ...")
    t1 = time.time()
    outputs = engine.generate(PROMPTS, sampling)  # batched
    bench_s = time.time() - t1
    print(f"[{label}] generation done in {bench_s:.1f}s")

    # Collect per-decode-step acceptance lengths + per-request rows.
    all_steps = []
    per_request = []
    total_completion = 0
    total_verify = 0
    for i, out in enumerate(outputs):
        mi = out["meta_info"]
        verify_ct = mi.get("spec_verify_ct", 0) or 0
        accept_len = mi.get("spec_accept_length")
        hist = mi.get("spec_correct_drafts_histogram") or []
        completion = mi.get("completion_tokens", 0) or 0
        steps = steps_from_histogram(hist)
        all_steps.extend(steps)
        total_completion += completion
        total_verify += verify_ct
        per_request.append(
            {
                "idx": i,
                "verify_ct": verify_ct,
                "completion_tokens": completion,
                "accept_length": accept_len,
                "n_steps_from_hist": len(steps),
            }
        )

    # Averages.
    has_steps = len(all_steps) > 0
    step_avg = (sum(all_steps) / len(all_steps)) if has_steps else 0.0
    weighted_avg = (total_completion / total_verify) if total_verify else 0.0
    # Headline: prefer the per-iteration mean; fall back to token-weighted if the
    # histogram is unavailable (e.g. DFLASH build without the histogram patch).
    headline = step_avg if has_steps else weighted_avg
    req_avgs = [r["accept_length"] for r in per_request if r["accept_length"] is not None]
    req_mean = (sum(req_avgs) / len(req_avgs)) if req_avgs else 0.0

    # Write per-iteration acceptance lengths (one per line) + footer.
    with open(out_path, "w") as f:
        f.write(f"# DFLASH acceptance length per decode iteration -- {label}\n")
        f.write(f"# draft={draft_path} quant={quantization} target={TARGET_MODEL}\n")
        f.write(f"# num_draft_tokens={NUM_DRAFT_TOKENS} max_new_tokens={MAX_NEW_TOKENS}\n")
        f.write("# one value per line = accepted tokens that decode step (incl. bonus)\n")
        for v in all_steps:
            f.write(f"{v}\n")
        f.write("#\n")
        f.write("# ===== per-request summary =====\n")
        for r in per_request:
            al = r["accept_length"]
            al_s = f"{al:.4f}" if al is not None else "n/a"
            f.write(
                f"# req {r['idx']:2d}: verify_ct={r['verify_ct']:4d} "
                f"completion={r['completion_tokens']:4d} accept_len={al_s} "
                f"steps={r['n_steps_from_hist']}\n"
            )
        f.write("#\n")
        src = "decode iterations" if has_steps else "token-weighted (histogram empty)"
        f.write(f"# total decode iterations : {len(all_steps)}\n")
        f.write(f"# AVG acceptance length   : {headline:.4f}  (from {src})\n")
        f.write(f"# token-weighted accept   : {weighted_avg:.4f}  (sum completion / sum verify)\n")
        f.write(f"# per-request mean        : {req_mean:.4f}\n")
        f.write(f"# benchmark wall time     : {bench_s:.1f}s\n")

    src = "decode iterations" if has_steps else "token-weighted (histogram empty)"
    print("=" * 64)
    print(f"[{label}] RESULTS")
    print(f"  total decode iterations : {len(all_steps)}")
    print(f"  AVG acceptance length   : {headline:.4f}  (from {src})")
    print(f"  token-weighted accept   : {weighted_avg:.4f}  (sum completion / sum verify)")
    print(f"  per-request mean        : {req_mean:.4f}")
    print(f"  per-iteration data file : {out_path}")
    print("=" * 64)

    engine.shutdown()
    return headline
