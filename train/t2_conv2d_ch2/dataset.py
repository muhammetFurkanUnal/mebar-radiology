from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Sample(NamedTuple):
    sample_id: str
    image: torch.Tensor
    mask: torch.Tensor
    label: bool


class T2Conv2DDataset(Dataset):
    # Doctor-approved: intermediate is treated as nonresponder.
    response_labels = {"responder": True, "nonresponder": False, "intermediate": False}

    def __init__(self, manifest_path, transform=None, manifest=None):
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        self.transform = transform

        if manifest is None:
            manifest = pd.read_csv(self.manifest_path)
        else:
            manifest = manifest.copy()

        if "response_group" not in manifest.columns:
            raise ValueError("manifest is missing response_group; rebuild the 2D dataset")
        if manifest["response_group"].isna().any():
            raise ValueError("manifest has rows without response_group")

        manifest["sample_id"] = manifest["sample_id"].astype(str)
        manifest["label"] = manifest["response_group"].map(self.response_labels)

        unknown = manifest.loc[manifest["label"].isna(), "response_group"].unique()
        if len(unknown) > 0:
            raise ValueError(f"unknown response_group values: {sorted(unknown)}")

        manifest["label"] = manifest["label"].astype(bool)
        self.manifest = manifest.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx) -> Sample:
        row = self.manifest.iloc[idx]
        image = self._read_slice(self.root / row["image_path"])
        mask = self._read_slice(self.root / row["mask_path"])

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return Sample(sample_id=row["sample_id"], image=image, mask=mask, label=bool(row["label"]))

    def _read_slice(self, path: Path) -> torch.Tensor:
        # Reads one 2D slice and returns a single-channel float tensor.
        array = np.load(path)
        return torch.from_numpy(array).float().unsqueeze(0)


def build_dataset(
    manifest_path,
    test_split_ratio: float,
    train_transform=None,
    test_transform=None,
    seed: int = 42,
):
    """Builds train and test datasets, keeping each patient on only one side."""
    if not 0.0 < test_split_ratio < 1.0:
        raise ValueError("test_split_ratio must be between 0 and 1")

    full = T2Conv2DDataset(manifest_path)
    manifest = full.manifest

    patients = manifest.drop_duplicates("sample_id")[["sample_id", "response_group"]]
    rng = np.random.default_rng(seed)

    test_ids = []
    for _, rows in patients.groupby("response_group"):
        ids = np.sort(rows["sample_id"].to_numpy())
        rng.shuffle(ids)
        count = int(round(len(ids) * test_split_ratio))
        count = min(count, len(ids) - 1)
        test_ids.extend(ids[:count])

    if len(test_ids) == 0:
        raise ValueError("test split is empty; increase test_split_ratio")

    is_test = manifest["sample_id"].isin(set(test_ids))

    train_dataset = T2Conv2DDataset(
        manifest_path,
        transform=train_transform,
        manifest=manifest.loc[~is_test],
    )
    test_dataset = T2Conv2DDataset(
        manifest_path,
        transform=test_transform,
        manifest=manifest.loc[is_test],
    )
    return train_dataset, test_dataset