from sglang.srt.models.gr00t import GR00T_N1_5
from sglang.srt.multimodal.processors.eagle2_5_vl import Eagle2_5_VLProcessor


class Gr00tProcessor(Eagle2_5_VLProcessor):
    models = [GR00T_N1_5]
