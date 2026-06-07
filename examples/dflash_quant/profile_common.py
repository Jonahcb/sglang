# profile_common.py
#
# Shared torch-profiler harness for comparing the DFLASH draft model's GPU time
# in the FP8 (quark) path vs the bf16 path. Imported by:
#   - profile_fp8.py   (quantized draft, /sgl-workspace/dflash-fp8)
#   - profile_bf16.py  (bf16 draft,      z-lab/Qwen3.5-35B-A3B-DFlash)
#
# What it does, per the spec:
#   * profiles ONE prompt (single-stream, the cleanest signal for per-layer time)
#   * CUDA graph ENABLED (production parity) — graphs are captured during warmup,
#     OUTSIDE the profiling window, so the trace shows steady-state graph *replay*
#   * sets DEBUG_CLR_GRAPH_PACKET_CAPTURE=0 so the roctracer graph-replay bug does
#     NOT drop kernels (esp. the FP8 F8BS draft GEMMs) from the trace
#   * emits a PyTorch/Perfetto trace (.trace.json.gz) into a per-path TRACE_DIR
#   * then runs analyze_dflash_trace.py on that trace to report per-layer GEMM time
#
# IMPORTANT: the env var MUST be set before the ROCm/HIP runtime loads, i.e.
# before `import sglang` (which pulls in torch/hip). We set it here, at the top,
# BEFORE importing accept_len_common (which imports sglang).
import os

# --- must precede any torch/hip import ---------------------------------------
os.environ.setdefault("DEBUG_CLR_GRAPH_PACKET_CAPTURE", "0")

import glob  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

# Reuse the SAME engine config + prompt set as the acceptance-length benchmark so
# the profiled engine is identical to what we measure elsewhere.
from accept_len_common import PROMPTS, build_engine  # noqa: E402

# One prompt only (user asked for a single-prompt profile).
PROMPT = os.environ.get("PROFILE_PROMPT", PROMPTS[0])

WARMUP_TOKENS = int(os.environ.get("PROFILE_WARMUP_TOKENS", "16"))
PROFILE_TOKENS = int(os.environ.get("PROFILE_TOKENS", "128"))
# PROFILE_BATCH > 1 profiles a *batched* steady-state decode: submit that many
# prompts concurrently with fixed length + greedy so all requests decode in
# lockstep and the draft runs steady batch-N GEMM shapes for the whole window.
# This is the regime that exposes batch-shape kernel efficiency (fp8 vs bf16).
PROFILE_BATCH = int(os.environ.get("PROFILE_BATCH", "1"))

HERE = os.path.dirname(os.path.abspath(__file__))
# Analyzers may live colocated here or under /sgl-workspace.
ANALYZER_DIRS = [HERE, "/sgl-workspace"]


def _resolve_analyzer(name):
    if os.path.isabs(name):
        return name
    for d in ANALYZER_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"analyzer {name} not found in {ANALYZER_DIRS}")


def _newest_trace(trace_dir):
    cands = sorted(glob.glob(os.path.join(trace_dir, "*.trace.json.gz"))
                   + glob.glob(os.path.join(trace_dir, "*.json.gz")))
    if not cands:
        raise FileNotFoundError(f"no *.trace.json.gz under {trace_dir}")
    return cands[-1]


def run(draft_path, quantization, label, trace_dir, analyze_script, report_path):
    os.makedirs(trace_dir, exist_ok=True)
    print(f"[{label}] DEBUG_CLR_GRAPH_PACKET_CAPTURE="
          f"{os.environ.get('DEBUG_CLR_GRAPH_PACKET_CAPTURE')}  (0 => no kernels dropped)")
    print(f"[{label}] draft={draft_path}  quant={quantization}  trace_dir={trace_dir}")

    engine = build_engine(draft_path, quantization)

    # batch=1 -> single prompt (string); batch=N -> N prompts (list) submitted
    # together so they decode in lockstep at steady batch-N. Fixed length + greedy
    # keeps the batch full for the whole window.
    if PROFILE_BATCH > 1:
        gen_input = PROMPTS[:PROFILE_BATCH]
        print(f"[{label}] BATCHED profile: batch_size={len(gen_input)}")
    else:
        gen_input = PROMPT

    # Robust warmup, ALWAYS before start_profile and OUTSIDE the timed window.
    # It runs the IDENTICAL workload (same batch) for >= the profiled length, for
    # multiple passes, so that EVERY one-time cost happens here, not in the
    # measured window:
    #   * CUDA-graph capture for this batch size (trace shows steady-state replay)
    #   * hipBLASLt autotune -- the one-time torch._scaled_mm heuristic searches
    #     (~185ms each, one per distinct fp8 GEMM shape). A short 16-token warmup
    #     previously left these ~7 searches inside the timed window, inflating the
    #     fp8 batch numbers. Warming at >= PROFILE_TOKENS for >=2 passes absorbs
    #     them and reaches steady state.
    warmup_tokens = max(WARMUP_TOKENS, PROFILE_TOKENS)
    warmup_passes = max(1, int(os.environ.get("PROFILE_WARMUP_PASSES", "2")))
    bs = len(gen_input) if isinstance(gen_input, list) else 1
    print(f"[{label}] warmup: {warmup_passes} x {warmup_tokens} tokens @ batch={bs} "
          f"(captures CUDA graphs + hipBLASLt autotune; pre-profile)")
    for _ in range(warmup_passes):
        engine.generate(gen_input, {"temperature": 0.0, "max_new_tokens": warmup_tokens})

    engine.start_profile(
        output_dir=trace_dir,
        with_stack=False,    # drop the huge python_function event stream
        record_shapes=True,  # keep op input shapes for the per-shape breakdown
    )
    engine.generate(gen_input, {"temperature": 0.0, "max_new_tokens": PROFILE_TOKENS})
    engine.stop_profile()

    engine.shutdown()
    print(f"\n[{label}] === PROFILE DONE === trace under {trace_dir}")

    # Per-layer breakdown in the dflash_analysis.txt format: the rich analyzer
    # buckets GPU time to the dflash_draft annotation span (so the draft model is
    # cleanly isolated from the bf16 target, even in the bf16 run) and reports
    # per-forward kernel time by layer/op type. Output is written to report_path
    # (NEVER dflash_analysis.txt) and also echoed to stdout.
    analyze = _resolve_analyzer(analyze_script)
    trace_file = _newest_trace(trace_dir)
    print(f"\n[{label}] === {analyze_script} {trace_file} ===")
    print(f"[{label}] writing report -> {report_path}")
    proc = subprocess.run([sys.executable, analyze, trace_file],
                          capture_output=True, text=True)
    out = proc.stdout + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    with open(report_path, "w") as f:
        f.write(out)
    print(out)
    return report_path
