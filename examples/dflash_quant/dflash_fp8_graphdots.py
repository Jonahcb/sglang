# dflash_fp8_graphdots.py
#
# Pair script #2 of 2 (sibling: dflash_fp8_perfetto.py).
# Both scripts run the IDENTICAL FP8 (Quark) DFlash engine + prompt with CUDA
# graph ENABLED. The only difference:
#   - dflash_fp8_perfetto.py captures the PyTorch (perfetto) profiler trace.
#   - this one captures NOTHING with the profiler, and instead dumps the draft
#     CUDA-graph DOTs + the graph keys (no slow profiler trace export).
#
# The DOT/key dumps are driven by env vars read ONCE at import time inside
# cuda_graph_runner, so they are set here BEFORE `import sglang`:
#   DFLASH_GRAPHDOT_DIR   one .dot per captured graph (named <worker>_graph_<key>.dot)
#   DFLASH_GRAPHKEY_FILE  one JSON line per graph replay (which graph_key launched)
#
#   python examples/dflash_quant/dflash_fp8_graphdots.py
# Then:
#   python examples/dflash_quant/old_scripts/summarize_graph_dots.py /sgl-workspace/graph_dots draft
import os

# --- set the dump paths BEFORE importing sglang --------------------------------
GRAPHDOT_DIR = os.environ.get("DFLASH_GRAPHDOT_DIR", "/sgl-workspace/graph_dots")
os.environ["DFLASH_GRAPHDOT_DIR"] = GRAPHDOT_DIR
GRAPHKEY_FILE = os.environ.get("DFLASH_GRAPHKEY_FILE", "/sgl-workspace/graph_keys.jsonl")
os.environ["DFLASH_GRAPHKEY_FILE"] = GRAPHKEY_FILE

import sglang as sgl  # noqa: E402  (must follow the env var assignments above)

DRAFT_PATH = os.environ.get("DFLASH_FP8_DRAFT", "/sgl-workspace/dflash-fp8")


def build_engine():
    return sgl.Engine(
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


PROMPT = "Write a detailed essay on the history of computing."


def main():
    engine = build_engine()

    # DOTs are written at CUDA graph capture (engine init). This warmup + main
    # generate mirror the perfetto script so the run is identical; here they also
    # populate the graph-key dump (one line per replay).
    engine.generate(PROMPT, {"temperature": 0.0, "max_new_tokens": 16})
    engine.generate(PROMPT, {"temperature": 0.0, "max_new_tokens": 128})

    engine.shutdown()
    print(f"\n=== GRAPH DOT DUMP === {GRAPHDOT_DIR}")
    print(f"=== GRAPHKEY DUMP  === {GRAPHKEY_FILE}")
    print(
        "Summarize: "
        f"python examples/dflash_quant/old_scripts/summarize_graph_dots.py {GRAPHDOT_DIR} draft"
    )


if __name__ == "__main__":
    main()
