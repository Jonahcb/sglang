# dump_graphdots_fp8.py
#
# Profiler-FREE runner that produces the CUDA graph DOT dumps (and the graph-key
# dump), WITHOUT the torch profiler. The torch profiler's trace export is what
# makes profile_dflash_engine_fp8_graphkeys.py slow; neither the DOT dump nor
# the graph-key dump needs it.
#
# Both dumps are driven by env vars read once at import time inside
# cuda_graph_runner, so they are set here BEFORE `import sglang`:
#   DFLASH_GRAPHDOT_DIR   one .dot per captured graph (every kernel/memcpy node)
#   DFLASH_GRAPHKEY_FILE  one JSON line per graph replay (which graph was launched)
#
# The DOT files are written at CAPTURE time (engine init), so the short
# generate() below exists only to trigger graph capture + a few replays.
#
#   python examples/dflash_quant/dump_graphdots_fp8.py
#
# Then summarize:
#   python examples/dflash_quant/old_scripts/summarize_graph_dots.py /sgl-workspace/graph_dots draft
import os

# --- set the dump paths BEFORE importing sglang --------------------------------
GRAPHDOT_DIR = os.environ.get("DFLASH_GRAPHDOT_DIR", "/sgl-workspace/graph_dots")
os.environ["DFLASH_GRAPHDOT_DIR"] = GRAPHDOT_DIR

GRAPHKEY_FILE = os.environ.get("DFLASH_GRAPHKEY_FILE", "/sgl-workspace/graph_keys.jsonl")
os.environ["DFLASH_GRAPHKEY_FILE"] = GRAPHKEY_FILE

import sglang as sgl  # noqa: E402  (must follow the env var assignments above)

DRAFT_PATH = os.environ.get("DFLASH_FP8_DRAFT", "/sgl-workspace/dflash-fp8")
MAX_NEW_TOKENS = int(os.environ.get("DFLASH_MAX_NEW_TOKENS", "16"))


def main():
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

    # Graphs are captured during engine init; this short generate() just exercises
    # capture + a handful of replays so the dumps are populated.
    engine.generate(prompt, {"temperature": 0.0, "max_new_tokens": MAX_NEW_TOKENS})

    engine.shutdown()
    print(f"\n=== GRAPH DOT DUMP === {GRAPHDOT_DIR}")
    print(f"=== GRAPHKEY DUMP  === {GRAPHKEY_FILE}")
    print(
        "Summarize with: "
        f"python examples/dflash_quant/old_scripts/summarize_graph_dots.py {GRAPHDOT_DIR} draft"
    )


if __name__ == "__main__":
    main()
