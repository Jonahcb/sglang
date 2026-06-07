# summarize_graph_dots.py
#
# Lists just the kernels baked into each captured CUDA graph, labelled by the
# SGLang graph key (parsed from the .dot filename: <worker>_graph_<key>.dot).
#
# Input: the directory of .dot files dumped by cuda_graph_runner._dump_graph_dot
# (written when DFLASH_GRAPHDOT_DIR is set). Each KERNEL node in a .dot carries
# the kernel's mangled symbol; we extract a clean namespace::function name and
# count occurrences per graph.
#
#   python examples/dflash_quant/summarize_graph_dots.py /sgl-workspace/graph_dots
#   python examples/dflash_quant/summarize_graph_dots.py /sgl-workspace/graph_dots draft
#
# 2nd arg: optional filename substring filter (e.g. "draft" or "draft_graph_1").
import os
import re
import sys
from collections import Counter

# Kernel symbol sits on the line after "KERNEL":
#   | {ID | 24458 | <mangled><<<grid,block,...>>>}
# In DOT the <<< >>> are escaped as \<\<\< \>\>\>. Capture the symbol up to it.
KSYM = re.compile(r"\{ID \| \d+ \| (.+?)\\<\\<\\<")

# Precision markers (same spirit as analyze_dflash_trace.py), matched against
# the raw mangled symbol where the type tags live.
FP8 = re.compile(r"_F8|fp8|float8|e4m3|e5m2|DB8_|Float8|fp8_quant|to_scale|quant")
BF16 = re.compile(r"DF16b|BFloat16|bfloat16|_BBS_")
FP16 = re.compile(r"_HHS_|_HAS_|__half|Half|fp16")
MOE_CK = re.compile(r"\bck::|ck_tile|CK_")


def parse_itanium(sym):
    """Extract a readable namespace::function from an Itanium-mangled symbol.

    Handles the common shapes in these graphs (_ZN<ns><fn>I...E... nested names,
    and _Z<len><fn> single names). Non-mangled symbols (triton/ck/extern-C) are
    returned as-is. Best-effort: falls back to the raw symbol on any surprise.
    """
    if not sym.startswith("_Z"):
        return sym
    i = 2
    nested = False
    if i < len(sym) and sym[i] == "N":
        nested = True
        i += 1
    parts = []
    while i < len(sym) and sym[i].isdigit():
        j = i
        while j < len(sym) and sym[j].isdigit():
            j += 1
        n = int(sym[i:j])
        name = sym[j : j + n]
        if len(name) != n:
            break
        parts.append(name)
        i = j + n
        if not nested:  # single-name form: one identifier only
            break
    return "::".join(parts) if parts else sym


def shorten(name):
    """Compress the very long Tensile GEMM symbols to precision + macro-tile.

    e.g. Cijk_Alik_Bljk_F8BS_..._MT16x16x1024_...  ->  rocBLAS_GEMM[F8BS MT16x16x1024]
    Other names are returned unchanged.
    """
    if name.startswith("Cijk_"):
        tm = re.search(r"Bljk_([A-Z0-9]+?)_", name)
        mt = re.search(r"_(MT\d+x\d+x\d+)", name)
        prec = tm.group(1) if tm else "?"
        tile = mt.group(1) if mt else "?"
        return f"rocBLAS_GEMM[{prec} {tile}]"
    return name


def classify(sym):
    if MOE_CK.search(sym):
        return "moe_ck"
    if FP8.search(sym):
        return "fp8"
    if BF16.search(sym):
        return "bf16"
    if FP16.search(sym):
        return "fp16"
    return "other"


def key_from_filename(fn):
    m = re.match(r"(.+)_graph_(.+)\.dot$", fn)
    if m:
        return m.group(1), m.group(2)  # worker, key
    return "?", fn


def summarize_file(path):
    text = open(path, "r", errors="replace").read()
    syms = KSYM.findall(text)
    names = [shorten(parse_itanium(s)) for s in syms]
    precs = [classify(s) for s in syms]  # classify on raw mangled (has type tags)
    by_name = Counter(names)
    name_prec = {}
    for nm, pc in zip(names, precs):
        name_prec.setdefault(nm, pc)
    return len(syms), by_name, name_prec


def main():
    if len(sys.argv) < 2:
        print("usage: summarize_graph_dots.py <dot_dir> [filename_substring]")
        sys.exit(1)
    d = sys.argv[1]
    filt = sys.argv[2] if len(sys.argv) > 2 else ""
    files = sorted(
        (f for f in os.listdir(d) if f.endswith(".dot") and filt in f),
        key=lambda f: (key_from_filename(f)[0], _intish(key_from_filename(f)[1])),
    )
    if not files:
        print(f"no .dot files in {d}" + (f" matching '{filt}'" if filt else ""))
        return

    for f in files:
        worker, key = key_from_filename(f)
        total, by_name, name_prec = summarize_file(os.path.join(d, f))
        prec_tot = Counter()
        for nm, c in by_name.items():
            prec_tot[name_prec[nm]] += c
        print(f"\n=== {worker} graph  key={key}  ({f}) ===")
        print(
            f"  total kernels: {total}   unique: {len(by_name)}   "
            + " ".join(f"{k}={v}" for k, v in prec_tot.most_common())
        )
        for nm, c in by_name.most_common():
            print(f"    [{name_prec[nm]:6}] x{c:<4} {nm}")


def _intish(s):
    return (0, int(s)) if s.isdigit() else (1, s)


if __name__ == "__main__":
    main()
