"""Preprocessing configurations.

Each configuration lives in its own folder under `configurations/`:

    configurations/
    ├── alignment/config.json
    └── no_alignment/
        ├── config.json
        ├── images/                  (processed outputs)
        └── pipeline_checkpoints.csv (run statistics)

Each folder's config.json has its own pydantic model below.
Load one by its folder name:

    from preprocessing.config import load_config
    cfg = load_config("no_alignment")
"""

import json
from pathlib import Path

from pydantic import BaseModel

CONFIGS_DIR = Path(__file__).with_name("configurations")


# --- Sections shared by the configs -----------------------------------------


class Apply(BaseModel):
    apply: bool


class Crop(BaseModel):
    apply: bool
    volume_mm: tuple[float, float, float]
    

class Interval(BaseModel):
    apply: bool
    lower: float
    upper: float


class Intensity(BaseModel):
    interval: Interval
    zscore: Apply

class BiasFieldCorrection(BaseModel):
    apply: bool
    shrink_factor: int


# --- alignment.json ------------------------------------------


class AlignmentResample(BaseModel):
    apply: bool
    spacing_mm: tuple[float, float, float]


class AlignmentConfig(BaseModel):
    excluded_sample_ids: list[str]
    resample: AlignmentResample
    crop: Crop
    bias_field_correction: Apply
    intensity: Intensity


# --- no_alignment.json ------------------------------------------------------

class NoAlignmentPad(BaseModel):
    apply: bool
    ADC_tensor_size: list[int]
    T2_tensor_size: list[int]

class NoAlignmentResample(BaseModel):
    apply: bool
    ADC_spacing_mm: tuple[float, float, float]
    T2_spacing_mm: tuple[float, float, float]


class ADCScaleImbalanceFix(BaseModel):
    apply: bool
    threshold: float
    crop_out_of_bounds: bool
    global_zscore: bool


class NoAlignmentConfig(BaseModel):
    excluded_sample_ids: list[str]
    manifest_path: str
    resample: NoAlignmentResample
    crop: Crop
    pad:NoAlignmentPad
    bias_field_correction: BiasFieldCorrection
    T2_intensity_threshold: Interval
    ADC_intensity_threshold: Interval
    ADC_scale_imbalance_fix: ADCScaleImbalanceFix
    output_folder_path: str


# --- Loader -----------------------------------------------------------------

# Config name (JSON file name without .json) -> its pydantic model.
CONFIGS = {
    "alignment": AlignmentConfig,
    "no_alignment": NoAlignmentConfig,
}


def load_config(name: str) -> AlignmentConfig | NoAlignmentConfig:
    """Read `configurations/<name>/config.json` and return it as its pydantic model."""
    with open(CONFIGS_DIR / name / "config.json", encoding="utf-8") as f:
        data = json.load(f)
    return CONFIGS[name](**data)
