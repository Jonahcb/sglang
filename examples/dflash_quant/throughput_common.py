# throughput_common.py
#
# Raw throughput benchmark for DFLASH spec decoding, shared by:
#   - throughput_fp8.py   (quantized quark draft)
#   - throughput_bf16.py  (bf16 draft)
#
# Reuses the SAME engine config + prompts as accept_len_common (CUDA graph is
# ENABLED by default in that builder -- we do NOT pass disable_cuda_graph), so
# the throughput here is the real graph-replay steady state.
#
# Reports two complementary numbers (greedy / temperature=0 so fp8 vs bf16 are
# directly comparable):
#
#   1. Single-stream decode speed (tok/s per sequence): batch size 1, large
#      max_new_tokens so decode dominates prefill. This is where spec decoding
#      helps most -- it's the per-token latency view. Run several times, report
#      mean/median/best.
#
#   2. Batch output throughput (system tok/s): all prompts issued in one
#      generate() call; total generated tokens / wall time. This is the serving
#      throughput view.
#
# Timing uses time.perf_counter around generate(); engine build + warmup are
# excluded.
import os
import statistics
import time

from accept_len_common import PROMPTS, build_engine

# Single-stream phase.
SS_MAX_NEW_TOKENS = int(os.environ.get("TP_SS_MAX_NEW_TOKENS", "512"))
SS_REPEATS = int(os.environ.get("TP_SS_REPEATS", "3"))
SS_PROMPT = PROMPTS[0]

# Batch phase.
BATCH_MAX_NEW_TOKENS = int(os.environ.get("TP_BATCH_MAX_NEW_TOKENS", "256"))
BATCH_SIZE = int(os.environ.get("TP_BATCH_SIZE", str(len(PROMPTS))))


def _completion_tokens(out):
    return out["meta_info"].get("completion_tokens", 0) or 0


def run(draft_path: str, quantization, label: str, out_path: str):
    print(f"[{label}] building engine (draft={draft_path}, quant={quantization}) ...")
    t0 = time.perf_counter()
    engine = build_engine(draft_path, quantization)
    print(f"[{label}] engine ready in {time.perf_counter() - t0:.1f}s")

    # Robust warmup, EXCLUDED from timing. Must cover BOTH phases' shapes so every
    # one-time cost happens here, not inside a timed generate():
    #   * CUDA-graph capture for batch=1 AND batch=BATCH_SIZE
    #   * hipBLASLt autotune -- the one-time torch._scaled_mm heuristic searches
    #     (~185ms each per distinct fp8 GEMM shape). The previous warmup only ran
    #     single-stream/16 tokens, so the batch-shape autotune fired INSIDE the
    #     timed batch run and inflated the fp8 batch number. Warm each phase at its
    #     real length for >=2 passes to absorb them and reach steady state.
    batch_prompts = (PROMPTS * ((BATCH_SIZE // len(PROMPTS)) + 1))[:BATCH_SIZE]
    warmup_passes = max(1, int(os.environ.get("TP_WARMUP_PASSES", "2")))
    print(f"[{label}] warmup: {warmup_passes} passes "
          f"(single-stream {SS_MAX_NEW_TOKENS} tok + batch={len(batch_prompts)} "
          f"{BATCH_MAX_NEW_TOKENS} tok); captures graphs + hipBLASLt autotune")
    for _ in range(warmup_passes):
        engine.generate(SS_PROMPT, {"temperature": 0.0, "max_new_tokens": SS_MAX_NEW_TOKENS})
        engine.generate(batch_prompts, {"temperature": 0.0, "max_new_tokens": BATCH_MAX_NEW_TOKENS})

    # ---- Phase 1: single-stream decode speed (batch size 1) ----
    ss_records = []  # (tokens, seconds, tok/s)
    for r in range(SS_REPEATS):
        t = time.perf_counter()
        out = engine.generate(
            SS_PROMPT, {"temperature": 0.0, "max_new_tokens": SS_MAX_NEW_TOKENS}
        )
        dt = time.perf_counter() - t
        toks = _completion_tokens(out)
        tps = toks / dt if dt > 0 else 0.0
        ss_records.append((toks, dt, tps))
        print(f"[{label}] single-stream run {r+1}/{SS_REPEATS}: "
              f"{toks} tok in {dt:.3f}s -> {tps:.2f} tok/s")

    ss_tps = [x[2] for x in ss_records]
    ss_mean = statistics.mean(ss_tps) if ss_tps else 0.0
    ss_median = statistics.median(ss_tps) if ss_tps else 0.0
    ss_best = max(ss_tps) if ss_tps else 0.0

    # ---- Phase 2: batch output throughput (one generate call) ----
    prompts = batch_prompts  # same batch warmed above
    sampling = {"temperature": 0.0, "max_new_tokens": BATCH_MAX_NEW_TOKENS}
    t = time.perf_counter()
    outs = engine.generate(prompts, sampling)
    batch_dt = time.perf_counter() - t
    batch_tokens = sum(_completion_tokens(o) for o in outs)
    batch_tps = batch_tokens / batch_dt if batch_dt > 0 else 0.0
    req_per_s = len(prompts) / batch_dt if batch_dt > 0 else 0.0
    print(f"[{label}] batch: {batch_tokens} tok across {len(prompts)} reqs in "
          f"{batch_dt:.3f}s -> {batch_tps:.2f} tok/s, {req_per_s:.2f} req/s")

    # ---- Write results ----
    with open(out_path, "w") as f:
        f.write(f"# DFLASH raw throughput -- {label}\n")
        f.write(f"# draft={draft_path} quant={quantization}\n")
        f.write(f"# cuda_graph=ENABLED  temperature=0 (greedy)\n")
        f.write("#\n")
        f.write("# === single-stream decode speed (batch size 1) ===\n")
        f.write(f"# max_new_tokens={SS_MAX_NEW_TOKENS} repeats={SS_REPEATS}\n")
        for i, (toks, dt, tps) in enumerate(ss_records):
            f.write(f"run {i}: tokens={toks} seconds={dt:.4f} tok_s={tps:.4f}\n")
        f.write(f"# single-stream tok/s  mean={ss_mean:.4f} median={ss_median:.4f} best={ss_best:.4f}\n")
        f.write("#\n")
        f.write("# === batch output throughput (one generate call) ===\n")
        f.write(f"# batch_size={len(prompts)} max_new_tokens={BATCH_MAX_NEW_TOKENS}\n")
        f.write(f"batch_tokens={batch_tokens} seconds={batch_dt:.4f} "
                f"tok_s={batch_tps:.4f} req_s={req_per_s:.4f}\n")

    print("=" * 64)
    print(f"[{label}] THROUGHPUT RESULTS")
    print(f"  single-stream decode : mean {ss_mean:.2f} | median {ss_median:.2f} | best {ss_best:.2f} tok/s")
    print(f"  batch output         : {batch_tps:.2f} tok/s ({req_per_s:.2f} req/s, bs={len(prompts)})")
    print(f"  results file         : {out_path}")
    print("=" * 64)

    engine.shutdown()
    return {"ss_median": ss_median, "ss_best": ss_best, "batch_tps": batch_tps}
