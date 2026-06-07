# verify_trace_completeness.py
#
# Counts GPU kernel slices in a PyTorch/Perfetto trace, with emphasis on the
# FP8 (F8BS) draft GEMMs that the roctracer graph-replay bug drops. Use it to
# confirm DEBUG_CLR_GRAPH_PACKET_CAPTURE=0 restored complete kernel coverage.
#
#   python verify_trace_completeness.py [trace_dir_or_file]
#
# Reference (buggy roctracer batch path): 168 F8BS slices (~5 iterations worth).
# Expected (packet-capture off): ~32 * (number of draft forward passes).
import collections
import glob
import gzip
import json
import sys

DEFAULT = "/sgl-workspace/dflash-fp8-trace-fixed"


def load(path):
    if path.endswith(".gz"):
        f = path
    else:
        cands = sorted(glob.glob(f"{path}/*.trace.json.gz") + glob.glob(f"{path}/*.json.gz"))
        if not cands:
            sys.exit(f"no trace .json.gz found under {path}")
        f = cands[-1]
    print(f"trace file: {f}")
    return json.load(gzip.open(f))["traceEvents"]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    ev = load(path)
    ker = [e for e in ev if e.get("cat") == "kernel"]
    f8 = [e for e in ker if "F8BS" in e.get("name", "")]
    gemm = [e for e in ker if "Cijk" in e.get("name", "")]
    print(f"total GPU kernel slices : {len(ker)}")
    print(f"all Tensile GEMM slices  : {len(gemm)}")
    print(f"FP8 (F8BS) GEMM slices   : {len(f8)}")
    if f8:
        print(f"  -> ~{len(f8)/32:.1f} draft forward passes' worth (32 fp8 GEMMs/pass)")
    # distinct kernel names present (sanity)
    names = collections.Counter(e.get("name", "") for e in ker)
    print(f"distinct kernel names    : {len(names)}")


if __name__ == "__main__":
    main()
