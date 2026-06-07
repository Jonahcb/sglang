# profile_fp8.py
#
# Torch-profile the FP8 (quark) DFLASH draft path over a single prompt, CUDA
# graph enabled, no dropped kernels (DEBUG_CLR_GRAPH_PACKET_CAPTURE=0), then
# break down per-layer GEMM time. See profile_common.py for details.
#
#   python profile_fp8.py
import os

from profile_common import run

DRAFT = os.environ.get("PROFILE_FP8_DRAFT", "/sgl-workspace/dflash-fp8")
TRACE_DIR = os.environ.get("PROFILE_FP8_TRACE_DIR", "/sgl-workspace/dflash-fp8-trace-prof")
REPORT = os.environ.get("PROFILE_FP8_REPORT", "/sgl-workspace/dflash_analysis_fp8.txt")

if __name__ == "__main__":
    run(
        draft_path=DRAFT,
        quantization="quark",
        label="FP8",
        trace_dir=TRACE_DIR,
        analyze_script="analyze_dflash_trace.py",
        report_path=REPORT,
    )
