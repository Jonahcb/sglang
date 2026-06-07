# dump_graphkeys_fp8.py
#
# Profiler-FREE companion to profile_dflash_engine_fp8.py.
#
# Same FP8 (Quark) DFlash engine + CUDA graphs as the profiling script, but it
# does NOT call start_profile()/stop_profile(). The torch profiler's trace
# export is the step that hangs for minutes on a CUDA-graph + spec-decode run;
# the graph-key dump does not need it.
#
# The graph-key dump is driven entirely by the DFLASH_GRAPHKEY_FILE env var,
# which cuda_graph_runner._dump_graph_key() reads on every graph replay. Run:
#
#   DFLASH_GRAPHKEY_FILE=/sgl-workspace/graph_keys.jsonl \
#       python examples/dflash_quant/dump_graphkeys_fp8.py
#
# Each generate() step appends one JSON line per CUDA graph replay
# (is_draft_worker=true -> draft graph, false -> target verify graph).
import os

import sglang as sgl

DRAFT_PATH = os.environ.get("DFLASH_FP8_DRAFT", "/sgl-workspace/dflash-fp8")
MAX_NEW_TOKENS = int(os.environ.get("DFLASH_MAX_NEW_TOKENS", "128"))


def main():
    if not os.environ.get("DFLASH_GRAPHKEY_FILE", "").strip():
        print(
            "WARNING: DFLASH_GRAPHKEY_FILE is not set; no graph-key dump will be "
            "written. Set it to a path before running."
        )

    engine = sgl.Engine(
        model_path="Qwen/Qwen3.5-35B-A3B",
        speculative_algorithm="DFLASH",
        speculative_draft_model_path=DRAFT_PATH,
        speculative_draft_model_quantization="quark",  # FP8 quark draft
        speculative_num_draft_tokens=16,
        tp_size=1,
        attention_backend="triton",
        speculative_draft_attention_backend="triton",
        mem_fraction_static=0.75,
        trust_remote_code=True,
        # CUDA graph is ENABLED by default — do NOT set disable_cuda_graph.
    )

    prompt = "Write a detailed essay on the history of computing."

    # Warmup: capture CUDA graphs first. Replays from this generate() are also
    # dumped, but capturing=true rows (one-time capture) are easy to filter out.
    engine.generate(prompt, {"temperature": 0.0, "max_new_tokens": 16})

    # Main run: this is the steady-state we care about; every replay is dumped.
    engine.generate(prompt, {"temperature": 0.0, "max_new_tokens": MAX_NEW_TOKENS})

    engine.shutdown()
    dump = os.environ.get("DFLASH_GRAPHKEY_FILE", "").strip()
    print(f"\n=== GRAPHKEY DUMP DONE === {dump or '(DFLASH_GRAPHKEY_FILE unset)'}")


if __name__ == "__main__":
    main()
