# profile_bf16.py
#
# Torch-profile the bf16 DFLASH draft path over a single prompt, CUDA graph
# enabled, no dropped kernels (DEBUG_CLR_GRAPH_PACKET_CAPTURE=0), then break
# down per-layer GEMM time. The bf16 baseline to compare against profile_fp8.py.
# See profile_common.py for details.
#
#   python profile_bf16.py
import os

from profile_common import run

DRAFT = os.environ.get("PROFILE_BF16_DRAFT", "z-lab/Qwen3.5-35B-A3B-DFlash")
TRACE_DIR = os.environ.get("PROFILE_BF16_TRACE_DIR", "/sgl-workspace/dflash-bf16-trace-prof")
REPORT = os.environ.get("PROFILE_BF16_REPORT", "/sgl-workspace/dflash_analysis_bf16.txt")

if __name__ == "__main__":
    run(
        draft_path=DRAFT,
        quantization=None,
        label="BF16",
        trace_dir=TRACE_DIR,
        # Single merged analyzer (handles both fp8 and bf16 traces).
        analyze_script="analyze_dflash_trace.py",
        report_path=REPORT,
    )
