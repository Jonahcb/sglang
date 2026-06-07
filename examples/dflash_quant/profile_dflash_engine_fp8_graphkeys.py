# profile_dflash_engine_fp8_graphkeys.py
#
# Combined runner: produces the torch profiler perfetto trace (same as
# profile_dflash_engine_fp8.py) AND dumps the CUDA graph key used on every
# graph replay to a JSONL file.
#
# The graph-key dump is driven by the DFLASH_GRAPHKEY_FILE env var, which
# cuda_graph_runner._dump_graph_key() reads ONCE at import time. So we set it
# here BEFORE importing sglang; otherwise the runner module would capture an
# empty value and never dump.
#
# Run:
#   python examples/dflash_quant/profile_dflash_engine_fp8_graphkeys.py
#
# Override paths/limits via env:
#   DFLASH_FP8_TRACE_DIR   trace output dir   (default /sgl-workspace/dflash-fp8-trace)
#   DFLASH_GRAPHKEY_FILE   graph-key jsonl    (default /sgl-workspace/graph_keys.jsonl)
#   DFLASH_FP8_DRAFT       fp8 draft path     (default /sgl-workspace/dflash-fp8)
#   DFLASH_MAX_NEW_TOKENS  profiled-gen len   (default 128)
#   DFLASH_PROFILE_STEPS   num_steps cap      (default 40; "" disables the cap)
import os

# --- set the dump paths BEFORE importing sglang --------------------------------
# These env vars are read once at import time inside cuda_graph_runner, so they
# must be set before `import sglang`.
GRAPHKEY_FILE = os.environ.get(
    "DFLASH_GRAPHKEY_FILE", "/sgl-workspace/graph_keys.jsonl"
)
os.environ["DFLASH_GRAPHKEY_FILE"] = GRAPHKEY_FILE

# One Graphviz .dot per captured graph (every kernel/memcpy node + edges).
# Written at capture time (engine init). Set to "" to skip the DOT dump.
GRAPHDOT_DIR = os.environ.get("DFLASH_GRAPHDOT_DIR", "/sgl-workspace/graph_dots")
os.environ["DFLASH_GRAPHDOT_DIR"] = GRAPHDOT_DIR

import sglang as sgl  # noqa: E402  (must follow the env var assignments above)

TRACE_DIR = os.environ.get("DFLASH_FP8_TRACE_DIR", "/sgl-workspace/dflash-fp8-trace")
DRAFT_PATH = os.environ.get("DFLASH_FP8_DRAFT", "/sgl-workspace/dflash-fp8")
MAX_NEW_TOKENS = int(os.environ.get("DFLASH_MAX_NEW_TOKENS", "128"))
# Bounding num_steps caps the trace size so stop_profile()'s export doesn't churn
# for minutes on a CUDA-graph + spec-decode run. Set DFLASH_PROFILE_STEPS="" to
# profile the whole generate() (matches the original script, may export slowly).
_steps_env = os.environ.get("DFLASH_PROFILE_STEPS", "40").strip()
PROFILE_STEPS = int(_steps_env) if _steps_env else None


def main():
    os.makedirs(TRACE_DIR, exist_ok=True)

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

    # Warmup: capture CUDA graphs OUTSIDE the profiling window, so the trace
    # shows graph *replay* (steady state), not one-time graph capture. (These
    # warmup replays are still dumped to the graph-key file.)
    engine.generate(prompt, {"temperature": 0.0, "max_new_tokens": 16})

    profile_kwargs = dict(
        output_dir=TRACE_DIR,
        with_stack=False,    # kills the 115M python_function events (34 GB -> hundreds of MB)
        record_shapes=True,  # keep op input shapes; set False to shrink further
    )
    if PROFILE_STEPS is not None:
        profile_kwargs["num_steps"] = PROFILE_STEPS  # auto-stop -> bounded trace

    engine.start_profile(**profile_kwargs)
    engine.generate(prompt, {"temperature": 0.0, "max_new_tokens": MAX_NEW_TOKENS})
    engine.stop_profile()

    engine.shutdown()
    print(f"\n=== PROFILE DONE === trace under {TRACE_DIR}")
    print(f"=== GRAPHKEY DUMP === {GRAPHKEY_FILE}")
    print(f"=== GRAPH DOT DUMP === {GRAPHDOT_DIR or '(disabled)'}")


if __name__ == "__main__":
    main()
