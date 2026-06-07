# dflash_fp8_replaytrace.py
#
# Diagnostic runner (no torch profiler) that answers, per CUDA graph replay:
#   - WHEN did it happen (monotonic seq + wall_t) -> iteration timing
#   - HOW MANY kernels are baked into that graph (kernels_in_graph)
#   - DID the GPU actually do work for THIS launch (gpu_ms, measured with CUDA
#     events + a sync -> ground truth, immune to perfetto attribution artifacts)
#   - draft vs target (is_draft_worker), graph_key, forward mode
#
# This exists to settle the perfetto puzzle where the draft graph's kernels only
# appear in a few iterations: gpu_ms here is the real per-launch GPU time, so if
# every draft replay shows a similar nonzero gpu_ms, the kernels DID run every
# iteration and the trace was just mis-attributing them.
#
# Same engine + prompt + CUDA graph as dflash_fp8_perfetto.py / _graphdots.py.
# The env vars are read once at import inside cuda_graph_runner, so set first:
#   DFLASH_REPLAY_LOG    per-replay JSONL  (the timing log)
#   DFLASH_GRAPHDOT_DIR  graph DOTs        (also supplies kernels_in_graph counts)
#
#   python examples/dflash_quant/dflash_fp8_replaytrace.py
#   python examples/dflash_quant/old_scripts/analyze_replay_log.py /sgl-workspace/replay_log.jsonl
import os

REPLAY_LOG = os.environ.get("DFLASH_REPLAY_LOG", "/sgl-workspace/replay_log.jsonl")
os.environ["DFLASH_REPLAY_LOG"] = REPLAY_LOG
GRAPHDOT_DIR = os.environ.get("DFLASH_GRAPHDOT_DIR", "/sgl-workspace/graph_dots")
os.environ["DFLASH_GRAPHDOT_DIR"] = GRAPHDOT_DIR

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
    engine.generate(PROMPT, {"temperature": 0.0, "max_new_tokens": 16})
    engine.generate(PROMPT, {"temperature": 0.0, "max_new_tokens": 128})
    engine.shutdown()
    print(f"\n=== REPLAY LOG === {REPLAY_LOG}")
    print(f"=== GRAPH DOTS === {GRAPHDOT_DIR}")
    print(
        "Analyze: "
        f"python examples/dflash_quant/old_scripts/analyze_replay_log.py {REPLAY_LOG}"
    )


if __name__ == "__main__":
    main()
