#!/usr/bin/env python3
# count_fp8_draft_kernels.py  <trace.json[.gz]>
#
# Manual verification that the FP8 *draft* actually runs.
#
# The draft is the ONLY fp8 module in the run (target/verify is bf16), so every
# `Cijk_*_F8BS_*` GEMM kernel and every Triton `_fwd_kernel` draft-attention
# dispatch must come from a draft forward. Count them per trace:
#
#   eager (no graph):  F8BS ~= (#draft forwards) * 32   (4 fp8 linears * 8 layers)
#   graph, healthy:    similar count (replays fire the captured kernels)
#   graph, EMPTY draft: near-zero (only the one-time capture shows up)
#
# Run it on both traces and compare:
#   python count_fp8_draft_kernels.py /sgl-workspace/dflash-fp8-trace-eager/*.json.gz
#   python count_fp8_draft_kernels.py /sgl-workspace/dflash-fp8-trace/*.json.gz
import gzip
import json
import sys
from collections import Counter


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def is_gpu_kernel(e):
    return e.get("cat") in ("kernel", "gpu_user_annotation") or (
        e.get("ph") == "X" and "DurationNs" in e.get("args", {}) and e.get("cat") == "kernel"
    )


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    data = load(path)
    events = data["traceEvents"] if isinstance(data, dict) else data

    f8bs = 0          # fp8 GEMM kernels -> draft-only
    bf16_gemm = 0     # Cijk *BBS* -> bf16 GEMM (target + draft-bf16 fc, etc.)
    fwd_kernel = 0    # triton attention forward kernels
    total_kernels = 0
    fp8_quant = 0     # scaled_fp8_quant / data_to_scale -> draft-only fp8 activation quant

    for e in events:
        if e.get("cat") != "kernel":
            continue
        name = e.get("name", "")
        total_kernels += 1
        if "F8BS" in name:
            f8bs += 1
        elif "Cijk_" in name and "BBS" in name:
            bf16_gemm += 1
        if "_fwd_kernel" in name:
            fwd_kernel += 1
        if "scale" in name.lower() and ("fp8" in name.lower() or "data_to_scale" in name.lower()):
            fp8_quant += 1

    print(f"trace: {path}")
    print(f"  total GPU kernels        : {total_kernels}")
    print(f"  F8BS  (fp8 GEMM, DRAFT)  : {f8bs}")
    print(f"  BBS   (bf16 GEMM)        : {bf16_gemm}")
    print(f"  fp8 activation-quant     : {fp8_quant}")
    print(f"  _fwd_kernel (attention)  : {fwd_kernel}")
    print()
    print(f"  >>> F8BS is the draft-only fp8 signal. Near-zero here means the")
    print(f"      draft GEMMs did NOT execute (empty draft graph).")


if __name__ == "__main__":
    main()
