from pathlib import Path
import sys
import time
import pandas as pd
import torch
import torchvision
from dataset import T2Conv2DDataset
from model import T2Conv2DModel_Ch1
from torch.utils.data import DataLoader
from config import config, test_config
from monai.data.utils import worker_init_fn

PARENT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parents[0]
sys.path.append(str(PARENT))
from lib import *

def build_model():
    return T2Conv2DModel_Ch1()

def build_loader(config):
    if config.transform:
        from transform import test_transform
    else:
        test_transform = None

    dataset = T2Conv2DDataset(
        manifest_path=config.manifest_path,
        transform=test_transform,
    )
    batch_size = config.batch_size
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    return loader

def test_pytorch():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)

    seed = config.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = build_model()
    model = model.to(device)

    testloader = build_loader(config)

    prep_test_folder(SCRIPT_DIR, test_config)

    ckptManager = CheckpointManager(SCRIPT_DIR, test_config.name, model)
    loaded_epoch = ckptManager.load(which=test_config.model_pt_filename, model_only=True)
    print(f"Loaded checkpoint '{test_config.model_pt_filename}' from epoch: {loaded_epoch}")

    model.eval()

    records = []
    with torch.no_grad():
        for batch in testloader:
            images = batch.image.to(device)
            labels = batch.label.to(device)
            sample_ids = batch.sample_id
            slice_indices = batch.slice_idx

            logits = model(images)
            scores = logits.cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            slice_idx_list = (
                slice_indices.cpu().numpy()
                if isinstance(slice_indices, torch.Tensor)
                else list(slice_indices)
            )

            for s_id, sl_idx, score, label, pred in zip(sample_ids, slice_idx_list, scores, labels_np, preds):
                records.append({
                    "sample_id": s_id,
                    "slice_idx": int(sl_idx),
                    "score_0": float(score[0]), # nonresponder + intermediate
                    "score_1": float(score[1]), # responder
                    "label": int(label),
                    "pred": int(pred),
                })

    df = pd.DataFrame(records)
    filename = test_config.test_csv_filename.strip() if test_config.test_csv_filename else ""
    if not filename or filename.lower() == "default":
        filename = "test_results.csv"
    elif not filename.endswith(".csv"):
        filename += ".csv"

    out_path = SCRIPT_DIR / "runs" / test_config.name / "inference" / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Test results ({len(df)} slices) saved to: {out_path}")


def test_xgboost():
    pass


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test model with PyTorch or XGBoost.")
    parser.add_argument(
        "framework",
        choices=["pytorch", "xgboost"],
        help="Testing framework: 'pytorch' or 'xgboost'",
    )
    args = parser.parse_args()

    if args.framework == "pytorch":
        test_pytorch()
    elif args.framework == "xgboost":
        test_xgboost()


if __name__ == "__main__":
    main()


