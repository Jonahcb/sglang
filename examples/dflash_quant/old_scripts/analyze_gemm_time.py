# analyze_gemm_time.py
#
# Quantifies GPU time spent in linear-layer GEMM kernels from a PyTorch/Perfetto
# trace. Splits FP8 (F8BS, the quantized draft linears) from bf16 (BBS), reports
# total GEMM GPU time, per-distinct-kernel breakdown, and fp8 draft linear time
# per draft forward pass (32 fp8 GEMMs/pass).
#
#   python analyze_gemm_time.py [trace_dir_or_file]
#
# NOTE on validity: per-kernel GPU durations are accurate even under
# DEBUG_CLR_GRAPH_PACKET_CAPTURE=0 (the device execution is unchanged; only
# launch overhead/throughput are distorted). So GEMM GPU-time sums are valid.
import collections
import glob
import gzip
import json
import re
import sys

DEFAULT = "/sgl-workspace/dflash-fp8-trace-fixed"
GEMMS_PER_DRAFT_PASS = 32  # 4 fp8 linears/layer x 8 draft layers


def load(path):
    if path.endswith(".gz"):
        f = path
    else:
        cands = sorted(glob.glob(f"{path}/*.trace.json.gz") + glob.glob(f"{path}/*.json.gz"))
        if not cands:
            sys.exit(f"no trace .json.gz found under {path}")
        f = cands[-1]
    print(f"trace file: {f}\n")
    return json.load(gzip.open(f))["traceEvents"]


def precision(name):
    if "F8BS" in name:
        return "fp8"
    if "Cijk" in name or "_BBS_" in name:
        return "bf16"
    return "other"


# Short label for a Tensile GEMM: keep the macro-tile, drop the noise.
def short(name):
    m = re.search(r"(F8BS|BBS)", name)
    tag = m.group(1) if m else "?"
    mt = re.search(r"_MT(\d+x\d+x\d+)", name)
    return f"{tag}_MT{mt.group(1)}" if mt else (tag + "_" + name[:30])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    ev = load(path)
    ker = [e for e in ev if e.get("cat") == "kernel" and "dur" in e]

    total_gpu = sum(e["dur"] for e in ker)  # microseconds
    gemm = [e for e in ker if "Cijk" in e.get("name", "")]
    by_prec = collections.Counter()
    dur_prec = collections.Counter()
    for e in gemm:
        p = precision(e["name"])
        by_prec[p] += 1
        dur_prec[p] += e["dur"]

    print(f"total GPU kernel time : {total_gpu/1000:.2f} ms across {len(ker)} kernels")
    print(f"all GEMM time         : {sum(dur_prec.values())/1000:.2f} ms "
          f"({100*sum(dur_prec.values())/total_gpu:.1f}% of GPU) across {len(gemm)} GEMMs")
    for p in ("fp8", "bf16", "other"):
        if by_prec[p]:
            print(f"  {p:5s}: {by_prec[p]:6d} GEMMs   {dur_prec[p]/1000:8.2f} ms   "
                  f"mean {dur_prec[p]/by_prec[p]:7.2f} us")

    f8 = [e for e in gemm if precision(e["name"]) == "fp8"]
    if f8:
        passes = len(f8) / GEMMS_PER_DRAFT_PASS
        print(f"\n=== fp8 DRAFT linear layers ===")
        print(f"fp8 GEMM dispatches   : {len(f8)}  -> {passes:.1f} draft forward passes")
        print(f"fp8 draft linear GPU time per forward pass: "
              f"{sum(e['dur'] for e in f8)/passes:.2f} us  "
              f"(= 32 x {sum(e['dur'] for e in f8)/len(f8):.2f} us mean)")

    # per-distinct-GEMM breakdown (shape via macro-tile)
    agg = collections.defaultdict(lambda: [0, 0.0])  # label -> [count, dur_us]
    for e in gemm:
        a = agg[short(e["name"])]
        a[0] += 1
        a[1] += e["dur"]
    print(f"\nper-GEMM-shape (label = precision + macro-tile):")
    print(f"  {'label':24s} {'count':>7s} {'total_ms':>10s} {'mean_us':>9s}")
    for label, (c, d) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"  {label:24s} {c:7d} {d/1000:10.2f} {d/c:9.2f}")


if __name__ == "__main__":
    main()
