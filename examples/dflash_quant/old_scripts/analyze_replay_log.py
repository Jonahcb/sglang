# analyze_replay_log.py
#
# Reads the per-replay JSONL written by dflash_fp8_replaytrace.py (env var
# DFLASH_REPLAY_LOG) and answers the perfetto puzzle directly:
#   - lists every CUDA graph replay (hipGraphLaunch) in order, split into draft
#     vs target, with the measured GPU time (gpu_ms) and the kernel count baked
#     into that graph.
#   - flags any replay whose gpu_ms is ~0 (i.e. a launch that did NO real work)
#     vs ones that did -> shows whether early iterations truly skip computation
#     or whether the kernels ran every time (perfetto attribution artifact).
#
#   python examples/dflash_quant/analyze_replay_log.py /sgl-workspace/replay_log.jsonl
import json
import sys

ZERO_MS = 0.02  # below this we treat a launch as "no real GPU work"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/replay_log.jsonl"
    recs = [json.loads(l) for l in open(path) if l.strip()]
    if not recs:
        print(f"no records in {path}")
        return

    draft = [r for r in recs if r.get("is_draft_worker")]
    target = [r for r in recs if not r.get("is_draft_worker")]

    def show(label, rows):
        if not rows:
            print(f"\n### {label}: none")
            return
        print(f"\n### {label}: {len(rows)} launches")
        kc = rows[0].get("kernels_in_graph")
        print(f"  kernels_in_graph (this key): {kc}")
        print(f"  {'idx':>4} {'key':>4} {'gpu_ms':>9} {'kernels':>7}  forward_mode")
        zero = 0
        gpu_vals = []
        for i, r in enumerate(rows):
            ms = r.get("gpu_ms")
            gpu_vals.append(ms if ms is not None else 0.0)
            flag = ""
            if ms is None:
                flag = " (no timing)"
            elif ms < ZERO_MS:
                flag = " <-- ~0, no work"
                zero += 1
            print(
                f"  {i:>4} {str(r.get('graph_key')):>4} "
                f"{(ms if ms is not None else float('nan')):>9.4f} "
                f"{str(r.get('kernels_in_graph')):>7}  {r.get('forward_mode')}{flag}"
            )
        nz = [v for v in gpu_vals if v >= ZERO_MS]
        print(
            f"  summary: {zero}/{len(rows)} launches did ~0 work; "
            f"nonzero gpu_ms min/mean/max = "
            f"{(min(nz) if nz else 0):.4f}/"
            f"{(sum(nz)/len(nz) if nz else 0):.4f}/"
            f"{(max(nz) if nz else 0):.4f}"
        )

    print(f"total replays: {len(recs)}  (draft={len(draft)}, target={len(target)})")
    show("DRAFT graph launches", draft)
    show("TARGET graph launches", target)

    print(
        "\nInterpretation: if DRAFT gpu_ms is similar & nonzero across ALL launches,"
        "\nthe fp8 kernels ran every iteration and perfetto just mis-attributed them."
        "\nIf early DRAFT launches are ~0 and later ones nonzero, the skip is real."
    )


if __name__ == "__main__":
    main()
