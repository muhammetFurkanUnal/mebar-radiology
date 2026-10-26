from config import (
    NoAlignmentConfig,
    AlignmentConfig
    )

from .steps import (
    Identity,
    ZScoreNorm,
    IntensityThreshold,
    Resample,
    BiasFieldCorrection,
    CropROI,
    Pad,
    Write
)


class Pipeline:
    def __init__(self, config, modality, records=None):
        self.records = records if records is not None else []
        self.pipeline = []

        if isinstance(config, NoAlignmentConfig):
            if modality == "ADC":
                self.pipeline = self._build_ADC(config)
            elif modality == "T2":
                self.pipeline = self._build_T2(config)
            else:
                raise ValueError(f"unknown modality: {modality}")
        elif isinstance(config, AlignmentConfig):
            pass

    def _build_ADC(self, config):
        steps = [
            Identity().step
            # Resample(target_spacing=config.resample.ADC_spacing_mm).step,
            # CropROI(volume_mm=config.crop.volume_mm).step,
            # Pad(target_size=config.pad.ADC_tensor_size).step,
            # Checkpoint(self.records, "after_pad").step,
        ]
        # ADC-specific steps go here (ADC_zero_negatives, ADC_scale_imbalance_fix)
        return steps

    def _build_T2(self, config):
        steps = [
            Resample(target_spacing=config.resample.T2_spacing_mm).step,
            BiasFieldCorrection(config.bias_field_correction.shrink_factor).step,
            IntensityThreshold(
                lower=config.T2_intensity_threshold.lower,
                upper=config.T2_intensity_threshold.upper,
                ).step,
            ZScoreNorm().step,
            CropROI(volume_mm=config.crop.volume_mm).step,
            Pad(target_size=config.pad.T2_tensor_size).step,
            Write().step
        ]
        return steps

    def run_pipeline(self, ctx):
        for step in self.pipeline:
            ctx = step(ctx)
        return ctx
