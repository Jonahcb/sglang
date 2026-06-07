# throughput_fp8.py
#
# Raw throughput for the QUANTIZED (FP8 / quark) DFLASH draft, CUDA graph enabled.
#   python throughput_fp8.py
import os

from throughput_common import run

DRAFT_PATH = os.environ.get("ACCEPT_LEN_FP8_DRAFT", "/sgl-workspace/dflash-fp8")
OUT_PATH = os.environ.get("TP_FP8_OUT", "/sgl-workspace/throughput_fp8.txt")

if __name__ == "__main__":
    run(draft_path=DRAFT_PATH, quantization="quark", label="FP8", out_path=OUT_PATH)
