#!/usr/bin/env python3
# diff_draft_dumps.py  <graph.jsonl>  <eager.jsonl>
#
# Compares the per-iteration draft-model output captured by _dump_draft_output
# in graph vs eager mode.
#
# What to look for:
#  * ITER 0: inputs are identical in both runs (same prompt, temp=0), so the
#    draft hidden fingerprint and tokens MUST match if the graph replays the
#    draft correctly. A mismatch at iter 0 = the graph draft is NOT computing.
#  * WITHIN the graph run: if the draft graph is empty, the fingerprint is
#    constant (or zero) across iterations -> "stale buffer" proof on its own.
import json
import sys


def load(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def fp(rec):
    return (rec["hidden_mean"], rec["hidden_std"], rec["hidden_sum"])


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    graph = load(sys.argv[1])
    eager = load(sys.argv[2])
    print(f"graph iters: {len(graph)}   eager iters: {len(eager)}\n")

    # 1) within-graph staleness check
    g_fps = {fp(r) for r in graph}
    print(f"[graph] distinct hidden fingerprints across {len(graph)} iters: {len(g_fps)}")
    if len(g_fps) <= 1:
        print("        --> CONSTANT/stale draft output: the draft graph is not "
              "recomputing per iteration (empty/stale graph).")
    e_fps = {fp(r) for r in eager}
    print(f"[eager] distinct hidden fingerprints across {len(eager)} iters: {len(e_fps)}\n")

    # 2) iter-0 apples-to-apples (guaranteed same input)
    if graph and eager:
        g0, e0 = graph[0], eager[0]
        print("ITER 0 (identical inputs in both runs):")
        print(f"  graph hidden(mean,std,sum) = {fp(g0)}")
        print(f"  eager hidden(mean,std,sum) = {fp(e0)}")
        same_h = fp(g0) == fp(e0)
        same_t = g0["draft_tokens"] == e0["draft_tokens"]
        print(f"  hidden match: {same_h}    tokens match: {same_t}")
        if not (same_h and same_t):
            print("  --> Draft output DIFFERS at iter 0 with identical inputs: "
                  "the graph-mode draft did not produce the eager result.")
        print(f"  graph tokens: {g0['draft_tokens']}")
        print(f"  eager tokens: {e0['draft_tokens']}")


if __name__ == "__main__":
    main()
