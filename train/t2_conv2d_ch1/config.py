import json
from pathlib import Path

from pydantic import BaseModel

CONFIG_PATH = Path(__file__).with_name("config.json")
TEST_CONFIG_PATH = Path(__file__).with_name("test_config.json")


class OptimizerConfig(BaseModel):
    name: str
    lr: float
    weight_decay: float


class XGBoostConfig(BaseModel):
    csv_path: str = ""
    n_estimators: int = 1000
    learning_rate: float = 0.05
    max_depth: int = 4
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    eval_metric: str = "logloss"
    early_stopping_rounds: int = 50
    random_state: int = 42


class TransformConfig(BaseModel):
    rnd_h_flip_p: float
    rnd_rot_deg: float
    rnd_aff_p: float
    rnd_down_scale: float
    rnd_up_scale: float
    translate_down: float
    translate_up: float
    intensity_scale_p: float
    intensity_scale_factor: float
    shift_intensity_p: float
    shift_intensity_offsets: float
    adjust_contrast_p: float
    adjust_contrast_gamma: tuple[float, float]
    gaussian_noise_p: float
    gaussian_noise_mean: float
    gaussian_noise_std: float
    gaussian_smooth_p: float
    gaussian_smooth_sigma_x: tuple[float, float]
    gaussian_smooth_sigma_y: tuple[float, float]
    bias_field_p: float
    bias_field_coeff_range: tuple[float, float]
    bias_field_degree: int


class Config(BaseModel):
    name: str
    batch_size: int
    seed: int
    manifest_path:str
    optimizer: OptimizerConfig
    transform: TransformConfig | None = None
    epochs: int
    early_stop: int
    ckp_frq: int
    ckp_best_check_frq: int
    xgboost: XGBoostConfig | None = None


class TestConfig(BaseModel):
    name: str
    test_csv_filename: str = "default"
    model_pt_filename: str = "best.pt"


def load_config(path: Path | str = CONFIG_PATH) -> Config:
    # Reads the json configuration file and returns it as a validated config model.
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Config(**data)


def load_test_config(path: Path | str = TEST_CONFIG_PATH) -> TestConfig:
    # Reads the test json configuration file and returns it as a validated test config model.
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return TestConfig(**data)


config = load_config()
test_config = load_test_config()

