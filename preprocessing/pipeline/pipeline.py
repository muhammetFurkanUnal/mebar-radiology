from enum import Enum
from config import (
    NoAlignmentConfig,
    AlignmentConfig
    )

class Pipeline:
    def __init__(self, config, modality):
        self.pipeline = []

        # if CURRENT CONFIGURATION
        # no_alignment.json:
        if isinstance(config, NoAlignmentConfig):
            from .steps import (
                Resample,
                CropROI
            )
            resample = Resample(
                target_spacing=config.resample.ADC_spacing_mm if modality == "ADC" else config.resample.T2_spacing_mm
            )
            self.pipeline.append(resample.step)
            crop_roi = CropROI(
                volume_mm=config.crop.volume_mm
            )
            self.pipeline.append(crop_roi.step)

        # if CURRENT CONFIGURATION
        # alignment.json:
        elif isinstance(config, AlignmentConfig):
            pass

    def run_pipeline(self, mask, image):
        for step in self.pipeline:
            mask, image = step(mask, image)
        return mask, image