# dflash_fp8_perfetto.py
#
# Pair script #1 of 2 (sibling: dflash_fp8_graphdots.py).
# Both scripts run the IDENTICAL FP8 (Quark) DFlash engine + prompt with CUDA
# graph ENABLED. The only difference:
#   - this one captures the PyTorch (perfetto) profiler trace.
#   - dflash_fp8_graphdots.py captures NOTHING with the profiler, and instead
#     dumps the draft CUDA-graph DOTs + the graph keys.
#
#   python examples/dflash_quant/dflash_fp8_perfetto.py
import os

import sglang as sgl

TRACE_DIR = os.environ.get("DFLASH_FP8_TRACE_DIR", "/sgl-workspace/dflash-fp8-trace")
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
    os.makedirs(TRACE_DIR, exist_ok=True)
    engine = build_engine()

    # Warmup: capture CUDA graphs OUTSIDE the profiling window, so the trace
    # shows graph *replay* (steady state), not one-time graph capture.
    engine.generate(PROMPT, {"temperature": 0.0, "max_new_tokens": 16})

    engine.start_profile(
        output_dir=TRACE_DIR,
        with_stack=False,     # kills the 115M python_function events (34 GB -> hundreds of MB)
        record_shapes=True,   # keep op input shapes; set False to shrink further
        # num_steps=40,       # optional: auto-stop after N forward passes (bounds trace size)
    )
    engine.generate(PROMPT, {"temperature": 0.0, "max_new_tokens": 128})
    engine.stop_profile()

    engine.shutdown()
    print(f"\n=== PERFETTO TRACE DONE === trace written under {TRACE_DIR}")


if __name__ == "__main__":
    main()
