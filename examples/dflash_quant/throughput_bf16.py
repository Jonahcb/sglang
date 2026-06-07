# throughput_bf16.py
#
# Raw throughput for the NON-QUANTIZED (bf16) DFLASH draft, CUDA graph enabled.
#   python throughput_bf16.py
import os

from throughput_common import run

DRAFT_PATH = os.environ.get("ACCEPT_LEN_BF16_DRAFT", "z-lab/Qwen3.5-35B-A3B-DFlash")
OUT_PATH = os.environ.get("TP_BF16_OUT", "/sgl-workspace/throughput_bf16.txt")

if __name__ == "__main__":
    run(draft_path=DRAFT_PATH, quantization=None, label="BF16", out_path=OUT_PATH)
