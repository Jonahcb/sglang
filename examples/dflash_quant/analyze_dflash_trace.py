"""Bucket DFLASH draft-model GPU kernel time by dflash_draft / dflash_verify spans.

Works on both fp8 and bf16 traces: the GEMM census splits kernels by precision
(fp8 / bf16 / fp16 / ...) detected from the kernel name, so the same analyzer
reports either run.

Usage:
    python analyze_dflash_trace.py <trace.json.gz | trace_dir>

A directory argument resolves to the newest *.trace.json.gz inside it.
"""

import gzip
import json
import re
import sys
from bisect import bisect_right
from collections import Counter, defaultdict


PHASE_NAMES = ("dflash_draft",)
ITER_SPAN_NAMES = ("dflash_draft", "dflash_verify")


# Map kernel-name patterns -> layer / op-type bucket.
# Order matters: first match wins, so put the most specific patterns first.
LAYER_PATTERNS = [
    ("Attention",
        [r"_fwd_kernel", r"flash_attn", r"flash_fwd", r"mha_fwd", r"attention_fwd",
         r"paged_attn", r"page_attention", r"decode_attention"]),
    ("Linear / GEMM",
        [r"^Cijk_", r"rocblas", r"hipblas",
         r"\bckProfiler\b", r"ck_gemm", r"hgemm", r"\bgemm\b"]),
    ("RMSNorm / LayerNorm",
        [r"rmsnorm", r"layer_norm", r"layernorm", r"add_rmsnorm"]),
    ("Rotary embedding",
        [r"rotary_embedding", r"rotary_emb"]),
    ("SiLU / GLU activation",
        [r"act_and_mul", r"silu", r"swiglu", r"gelu"]),
    ("Embedding lookup",
        [r"embedding_kernel", r"index_select", r"embedding_dense"]),
    ("KV cache write",
        [r"store_kvcache", r"store_cache", r"reshape_and_cache", r"write_kv"]),
    ("Sampling / argmax / topk",
        [r"argmax", r"top_?k", r"sampling", r"ArgMaxOps", r"reduce_kernel.*ArgMax"]),
    ("All-reduce / collective",
        [r"all_reduce", r"allreduce", r"nccl", r"rccl", r"all_gather"]),
    ("Reduction (sum / max / mean)",
        [r"reduce_kernel", r"reduceOp", r"SumOps", r"MeanOps", r"MaxOps"]),
    # True DMA memcpy / memset only. The driver-side names are flat strings like
    # "Memcpy DtoD (Device -> Device)" with no template noise.
    ("DMA memcpy / memset",
        [r"^Memcpy ", r"^Memset ", r"hipMemcpy", r"hipMemset"]),
    # Elementwise compute kernels -- includes copy/cast (`direct_copy_kernel`),
    # fills (`FillFunctor`), arithmetic (`CUDAFunctor`), and casts. These are GPU
    # compute, NOT DMA, even though some implement Tensor.copy_() / .to(dtype).
    ("Elementwise (copy / cast / add / mul / fill / etc.)",
        [r"elementwise_kernel", r"CUDAFunctor", r"BinaryFunctor",
         r"direct_copy_kernel", r"copy_kernel", r"fill_kernel", r"FillFunctor",
         r"multi_tensor_apply_kernel"]),
    ("Index / scatter / gather",
        [r"index_kernel", r"scatter", r"gather", r"index_elementwise"]),
    ("Arange / range",
        [r"arange", r"range_kernel"]),
    ("Cumulative (cumsum / cumprod)",
        [r"cumsum", r"cumprod", r"scan_kernel"]),
    ("Quantize / dequantize",
        [r"quant", r"dequant", r"\bfp8\b", r"\bint8\b"]),
    ("Mamba / SSM / chunk scan",
        [r"mamba", r"ssm_", r"ChunkGatedDelta", r"chunk_scan"]),
]

LAYER_REGEX = [(label, re.compile("|".join(pats), re.IGNORECASE))
               for label, pats in LAYER_PATTERNS]


def classify_kernel(name: str) -> str:
    for label, rx in LAYER_REGEX:
        if rx.search(name):
            return label
    return "Other / unclassified"




def load_events(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        data = json.load(f)
    return data["traceEvents"]


def _is_gemm(name):
    return ("Cijk_" in name) or ("ck::" in name and "gemm" in name.lower())


def _gemm_precision(name):
    # fp8 markers first (Tensile F8BS, or any fp8 dtype token in the name).
    if re.search(r"_F8BS_|_F8_|fp8|float8|e4m3|e5m2|BFloat8|Float8", name, re.IGNORECASE):
        return "fp8"
    if "_BBS_" in name:
        return "bf16"
    if "_HHS_" in name or "_HAS_" in name:
        return "fp16"
    if "ck::" in name:
        return "moe(target)"
    return "other"


def _gemm_shape(name):
    m = re.search(r"_MT(\d+x\d+x\d+)_", name)
    return m.group(1) if m else "?"


def gemm_census(events, label=""):
    """Phase-independent audit that accounts for EVERY GEMM kernel in the trace.

    The per-phase tables elsewhere attribute kernels by *CPU launch ts*, which
    silently DROPS CUDA-graph-replayed kernels (their launch ts falls outside the
    record_function span). That makes an eager run (all GEMMs visible) look
    incomparable to a graphed run (most GEMMs dropped).

    This census instead buckets every GEMM by its own *GPU ts* into the innermost
    GPU-side annotation span (the way Perfetto shows them), and prints
    total == sum(by precision) == sum(by span incl. UNATTRIBUTED), so no GEMM can
    be silently lost. Use it to compare fp8 vs bf16 GEMM execution across runs.
    """
    spans = [
        (e["ts"], e["ts"] + e.get("dur", 0.0), e["name"])
        for e in events
        if e.get("ph") == "X" and e.get("cat") == "gpu_user_annotation"
    ]
    spans.sort()
    starts = [s for s, _, _ in spans]

    def which_span(ts):
        """Innermost (shortest) GPU annotation span containing ts, else UNATTRIBUTED."""
        best_name, best_len = "UNATTRIBUTED", None
        i = bisect_right(starts, ts) - 1
        while i >= 0:
            s, e, name = spans[i]
            if s <= ts <= e:
                length = e - s
                if best_len is None or length < best_len:
                    best_len, best_name = length, name
            i -= 1
        return best_name

    gemms = [
        e
        for e in events
        if e.get("ph") == "X" and e.get("cat") == "kernel" and _is_gemm(e["name"])
    ]
    total_time = sum(e.get("dur", 0.0) for e in gemms)

    by_prec, by_prec_t = Counter(), defaultdict(float)
    by_span, by_span_prec = Counter(), defaultdict(Counter)
    by_shape, by_shape_prec = Counter(), defaultdict(Counter)
    for e in gemms:
        n = e["name"]
        p = _gemm_precision(n)
        d = e.get("dur", 0.0)
        sp = which_span(e["ts"])
        sh = _gemm_shape(n)
        by_prec[p] += 1
        by_prec_t[p] += d
        by_span[sp] += 1
        by_span_prec[sp][p] += 1
        by_shape[sh] += 1
        by_shape_prec[sh][p] += 1

    print("=" * 80)
    print(f"GEMM CENSUS  {label}".rstrip())
    print("=" * 80)
    print(
        f"TOTAL GEMM kernels in trace: {len(gemms)}   (GPU time {total_time/1000:.3f} ms)"
    )
    print("  (GEMM = Tensile 'Cijk_*' or 'ck::*gemm*'; every one is bucketed below)")
    print()
    print("-- by precision (this is the robust fp8-vs-bf16 answer; phase-independent) --")
    for p, c in by_prec.most_common():
        print(f"   {p:14s} count={c:7d}  time={by_prec_t[p]/1000:10.3f} ms")
    print()
    print("-- by GPU-side annotation span (innermost; UNATTRIBUTED = inside no span) --")
    for sp, c in by_span.most_common():
        prec_str = "  ".join(f"{k}={v}" for k, v in by_span_prec[sp].most_common())
        print(f"   count={c:7d}  [{sp}]   {prec_str}")
    print()
    print("-- top GEMM shapes (MT n x m x k) by count, precision split --")
    for sh, c in by_shape.most_common(12):
        prec_str = "  ".join(f"{k}={v}" for k, v in by_shape_prec[sh].most_common())
        print(f"   MT{sh:14s} count={c:7d}   {prec_str}")
    print()
    n_prec, n_span = sum(by_prec.values()), sum(by_span.values())
    ok = (len(gemms) == n_prec == n_span)
    print(
        f"[completeness] total={len(gemms)}  sum(by precision)={n_prec}  "
        f"sum(by span incl UNATTRIBUTED)={n_span}  -> "
        f"{'ALL GEMMs accounted for' if ok else 'MISMATCH!'}"
    )
    print()


def build_launch_ts_maps(events):
    """Build two maps used to resolve a GPU kernel's CPU launch ts:

      ext_ts[External id] -> ts of the launching cpu_op
      corr_ts[correlation] -> ts of the matching cuda_runtime call

    Eager-launched kernels carry External id pointing at the leaf aten op.
    CUDA-graph-replayed kernels carry External id = None, but share a
    `correlation` id with the single hipGraphLaunch runtime call -- which
    is on CPU and falls inside the record_function span. We try ext_id
    first, then fall back to correlation.
    """
    ext_ts = {}
    corr_ts = {}
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat")
        if cat not in ("cpu_op", "cuda_runtime", "user_annotation"):
            continue
        args = e.get("args") or {}
        ts = e.get("ts")
        if ts is None:
            continue
        ext = args.get("External id")
        if ext is not None:
            if ext not in ext_ts or cat == "cpu_op":
                ext_ts[ext] = ts
        corr = args.get("correlation")
        if corr is not None and cat == "cuda_runtime":
            corr_ts[corr] = ts
    return ext_ts, corr_ts


def resolve_launch_ts(event, ext_ts, corr_ts):
    args = event.get("args") or {}
    ext = args.get("External id")
    if ext is not None:
        t = ext_ts.get(ext)
        if t is not None:
            return t
    corr = args.get("correlation")
    if corr is not None:
        t = corr_ts.get(corr)
        if t is not None:
            return t
    return event.get("ts")


def build_phase_index(events):
    """Return list of (start, end, phase_name) sorted by start, and starts list."""
    phases = []
    for e in events:
        if (
            e.get("ph") == "X"
            and e.get("cat") == "user_annotation"
            and e.get("name") in PHASE_NAMES
        ):
            ts = e["ts"]
            phases.append((ts, ts + e.get("dur", 0.0), e["name"]))
    phases.sort()
    starts = [p[0] for p in phases]
    return phases, starts


def phase_for(ts, phases, starts):
    """Return phase covering ts (start <= ts <= end), or None.

    Used with CPU-side launch timestamps (resolved via External id), so
    the strict containment is correct -- a kernel is attributed to the
    span its launching cpu_op was inside."""
    i = bisect_right(starts, ts) - 1
    if i < 0:
        return None
    s, e, name = phases[i]
    if ts <= e:
        return name
    return None


def main(path):
    events = load_events(path)
    gemm_census(events, label=path)
    phases, starts = build_phase_index(events)
    ext_ts, corr_ts = build_launch_ts_maps(events)
    n_draft = sum(1 for _, _, n in phases if n == "dflash_draft")
    # One draft forward pass per spec-decode iteration. There are two
    # dflash_draft spans per iteration (prep+forward and the trailing KV write),
    # so forward passes = n_draft / 2.
    n_forward = n_draft // 2

    # GPU execution time bucketed by phase.
    # Bucket by event ts falling inside a phase span. Phases are coarse enough
    # that drift between launch (CPU) and execution (GPU) is negligible.
    gpu_by_phase = defaultdict(lambda: defaultdict(float))
    gpu_count_by_phase = defaultdict(lambda: defaultdict(int))
    gpu_total = defaultdict(float)

    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat")
        if cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        dur = e.get("dur", 0.0)
        if dur is None:
            continue
        attr_ts = resolve_launch_ts(e, ext_ts, corr_ts)
        if attr_ts is None:
            continue
        ph = phase_for(attr_ts, phases, starts)
        if ph is None:
            continue
        gpu_by_phase[ph][e["name"]] += dur
        gpu_count_by_phase[ph][e["name"]] += 1
        gpu_total[ph] += dur

    def fmt_us(v):
        return f"{v/1000:>12.3f} ms"

    print(f"draft model forward passes: {n_forward}")
    print()

    print("=== GPU execution time (on-device compute + memcpy + memset) ===")
    for ph in PHASE_NAMES:
        n = n_forward
        per_iter = gpu_total[ph] / n if n else 0.0
        print(f"  {ph:<14} total={fmt_us(gpu_total[ph])}  per-forward={fmt_us(per_iter)}")
    print()

    for ph in PHASE_NAMES:
        n_phase = n_forward
        print(f"=== All GPU kernels (execution time) in {ph} ===")
        print(f"    count = total launches across {n_phase} draft forward passes")
        print(f"    per-forward = launches per single draft forward pass")
        items = sorted(gpu_by_phase[ph].items(), key=lambda kv: -kv[1])
        for name, t in items:
            pct = 100 * t / gpu_total[ph] if gpu_total[ph] else 0
            c = gpu_count_by_phase[ph][name]
            per_iter = c / n_phase if n_phase else 0
            print(f"  {fmt_us(t)}  {pct:5.1f}%  count={c:6d}  per-forward={per_iter:6.2f}  {name}")
        print()

    # Per-layer / op-type breakdown: group kernels by what kind of layer they
    # implement. Classification is based on kernel-name pattern matching --
    # see LAYER_PATTERNS at the top of the file to add or refine buckets.
    for ph in PHASE_NAMES:
        n_phase = n_forward
        layer_total = defaultdict(float)
        layer_count = defaultdict(int)
        layer_kernels = defaultdict(list)   # layer -> list of (name, time, count)
        for name, t in gpu_by_phase[ph].items():
            c = gpu_count_by_phase[ph][name]
            layer = classify_kernel(name)
            layer_total[layer] += t
            layer_count[layer] += c
            layer_kernels[layer].append((name, t, c))

        print(f"=== GPU execution time by layer / op-type in {ph} ===")
        total = gpu_total[ph]
        for layer, t in sorted(layer_total.items(), key=lambda kv: -kv[1]):
            pct = 100 * t / total if total else 0
            c = layer_count[layer]
            per_iter = c / n_phase if n_phase else 0
            print(f"  {fmt_us(t)}  {pct:5.1f}%  count={c:6d}  per-forward={per_iter:6.2f}  {layer}")
        print()

        for layer, t in sorted(layer_total.items(), key=lambda kv: -kv[1]):
            pct = 100 * t / total if total else 0
            c = layer_count[layer]
            per_iter = c / n_phase if n_phase else 0
            print(f"  -- {layer}  ({fmt_us(t)}, {pct:.1f}%, count={c}, per-forward={per_iter:.2f})")
            for name, kt, kc in sorted(layer_kernels[layer], key=lambda kv: -kv[1]):
                kpct = 100 * kt / t if t else 0
                kper_iter = kc / n_phase if n_phase else 0
                print(f"       {fmt_us(kt)}  {kpct:5.1f}%  count={kc:6d}  per-forward={kper_iter:6.2f}  {name}")
            print()
        print()


def print_per_iteration_breakdown(events):
    """Per-iteration kernel launch sequence.

    An iteration = one (dflash_draft, dflash_verify, dflash_draft) triple in
    timestamp order. For each iteration we print the chronological launch
    sequence of GPU kernels (consecutive duplicates collapsed as `name xN`).
    Use this to reverse-derive accepted tokens per iteration.
    """
    # Collect all relevant spans, sorted by ts.
    spans = []
    for e in events:
        if (e.get("ph") == "X"
                and e.get("cat") == "user_annotation"
                and e.get("name") in ITER_SPAN_NAMES):
            ts = e["ts"]
            spans.append((ts, ts + e.get("dur", 0.0), e["name"]))
    spans.sort()

    # Group into iterations. Expected per-iter pattern (in ts order):
    #   dflash_draft (prep + draft forward)
    #   dflash_verify
    #   dflash_draft (trailing KV write)
    iterations = []
    i = 0
    while i < len(spans):
        # Take the next draft, then any spans until (and including) the next
        # draft that follows a verify. Be tolerant of irregularities.
        cur = []
        saw_verify = False
        while i < len(spans):
            cur.append(spans[i])
            name = spans[i][2]
            i += 1
            if name == "dflash_verify":
                saw_verify = True
                continue
            if saw_verify and name == "dflash_draft":
                break
        iterations.append(cur)

    # Collect GPU kernel events. Attribute each to its launching cpu_op's
    # timestamp via External id (so spans bound by record_function on the
    # CPU correctly own all the kernels they launched, even ones whose GPU
    # execution finished after the span closed).
    ext_ts, corr_ts = build_launch_ts_maps(events)
    kernels = []
    for e in events:
        if e.get("ph") != "X":
            continue
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        attr_ts = resolve_launch_ts(e, ext_ts, corr_ts)
        if attr_ts is None:
            continue
        kernels.append((attr_ts, e["name"]))
    kernels.sort()
    k_starts = [k[0] for k in kernels]

    print("=" * 80)
    print("PER-ITERATION KERNEL LAUNCH SEQUENCE")
    print("=" * 80)
    print(f"Total iterations detected: {len(iterations)}")
    print("Layout per iteration:")
    print("  [draft span]  prep + draft model forward")
    print("  [verify span] target model verify")
    print("  [draft span]  trailing append-target-hidden-to-draft-kv")
    print("Consecutive identical kernel launches collapsed as `name xN`.")
    print()

    for idx, spans_in_iter in enumerate(iterations):
        iter_start = spans_in_iter[0][0]
        iter_end = max(s[1] for s in spans_in_iter)
        print("-" * 80)
        print(f"ITERATION {idx}  (ts {iter_start:.0f} -> {iter_end:.0f} us, "
              f"{(iter_end - iter_start)/1000:.3f} ms wall)")
        # Strict containment by CPU launch ts (via External id): a kernel
        # belongs to span [s_start, s_end] iff its launching cpu_op ts is in.
        for s_start, s_end, s_name in spans_in_iter:
            lo = bisect_right(k_starts, s_start - 1)
            hi = bisect_right(k_starts, s_end)
            seq = [kernels[j][1] for j in range(lo, hi)]

            # Per-span totals.
            counts = defaultdict(int)
            for n in seq:
                counts[n] += 1

            # Collapse consecutive duplicates.
            collapsed = []
            for n in seq:
                if collapsed and collapsed[-1][0] == n:
                    collapsed[-1] = (n, collapsed[-1][1] + 1)
                else:
                    collapsed.append((n, 1))

            print(f"  [{s_name}]  span={(s_end - s_start)/1000:.3f} ms  "
                  f"total_launches={len(seq)}  unique={len(counts)}")
            print(f"    -- counts (sorted by launches desc) --")
            for n, c in sorted(counts.items(), key=lambda kv: -kv[1]):
                print(f"       {c:5d}  {n}")
            print(f"    -- launch order with layer-boundary walls (consecutive dups collapsed) --")
            # Per-layer kernel order in DFlashDecoderLayer (Qwen3 style):
            #   input_layernorm (RMSNorm)        <-- layer entry
            #   qkv_proj (GEMM)
            #   q_norm, k_norm (RMSNorm x2, per-head)
            #   rotary_embedding                 <-- unique 1/layer marker
            #   store_kvcache                    <-- 1/layer marker (still inside attn)
            #   _fwd_kernel (Attention)
            #   o_proj (GEMM)                    <-- end of attn block
            #   post_attention_layernorm (RMSNorm)
            #   gate_up_proj (GEMM)
            #   act_and_mul (SwiGLU)             <-- unique 1/layer marker, mid-MLP
            #   down_proj (GEMM)                 <-- end of layer
            #
            # State machine: count rotary_embedding to derive attn_layer_idx;
            # transition attn->MLP on the first GEMM AFTER _fwd_kernel; new layer
            # starts on the next rotary_embedding.
            n_rotary = sum(1 for n in seq if "rotary_embedding" in n)
            layer_idx = -1
            in_attn = False
            saw_attn_kernel = False  # have we seen _fwd_kernel in current layer's attn?
            phase = "pre-decoder (eager prep)"
            print(f"    ===== {phase} =====")
            last_class = None
            for n, c in collapsed:
                klass = classify_kernel(n)
                is_rotary = "rotary_embedding" in n
                is_attn_fwd = klass == "Attention"
                is_gemm = klass == "Linear / GEMM"

                # Boundary: rotary -> start of new layer's attention block.
                if is_rotary:
                    layer_idx += 1
                    in_attn = True
                    saw_attn_kernel = False
                    phase = f"Layer {layer_idx}: attention"
                    print(f"    ===== {phase}  (input_layernorm + qkv_proj + q/k_norm already above) =====")
                    last_class = None
                # Boundary: first GEMM after _fwd_kernel = o_proj end-of-attn,
                # immediately followed by post_attn_layernorm + MLP.
                elif in_attn and saw_attn_kernel and is_gemm:
                    in_attn = False
                    phase = f"Layer {layer_idx}: MLP  (o_proj boundary; this GEMM = o_proj)"
                    print(f"    ===== {phase} =====")
                    last_class = None

                if klass != last_class:
                    print(f"      -- {klass} --")
                    last_class = klass
                if c == 1:
                    print(f"       {n}")
                else:
                    print(f"       {n}  x{c}")

                if is_attn_fwd:
                    saw_attn_kernel = True

            if layer_idx >= 0:
                print(f"    ===== Post-decoder  (final norm + lm_head + sampling + eager append-KV) =====")
            print(f"    [layer-count check: rotary_embedding launches = {n_rotary} (expect 8 for full draft forward)]")
            print()
    print()


def _resolve_trace(p):
    """Accept either a trace file or a directory; if a dir, pick newest *.trace.json.gz."""
    import glob
    import os

    if os.path.isdir(p):
        cands = sorted(
            glob.glob(os.path.join(p, "*.trace.json.gz")),
            key=os.path.getmtime,
        )
        if not cands:
            raise FileNotFoundError(f"No *.trace.json.gz found under {p}")
        return cands[-1]
    return p


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <trace.json.gz | trace_dir>")
    path = _resolve_trace(sys.argv[1])
    print(f"# analyzing trace: {path}\n")
    main(path)
    print_per_iteration_breakdown(load_events(path))
