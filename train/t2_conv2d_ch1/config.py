import json
from pathlib import Path

from pydantic import BaseModel

CONFIG_PATH = Path(__file__).with_name("config.json")


class OptimizerConfig(BaseModel):
    name: str
    lr: float
    weight_decay: float


class TransformConfig(BaseModel):
    pass

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


def load_config(path: Path | str = CONFIG_PATH) -> Config:
    # Reads the json configuration file and returns it as a validated config model.
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Config(**data)


config = load_config()

