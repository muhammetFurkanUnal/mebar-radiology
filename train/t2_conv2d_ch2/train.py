from datetime import datetime
from pathlib import Path
import sys
import time
import torch
import torchvision
from dataset import build_dataset
from model import T2Conv2DModel_Ch2
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from config import config
from monai.data.utils import worker_init_fn

PARENT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parents[0]
sys.path.append(str(PARENT))
from lib import *

writer = SummaryWriter(SCRIPT_DIR / "runs" / config.name)

def build_model():
    return T2Conv2DModel_Ch2()

def build_optimizer(model, config):
    opt_cfg = config.optimizer
    if opt_cfg.name == "adamw":
        from torch.optim import AdamW
        return AdamW(
            model.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay
        )

def build_loaders(config):
    if config.transform:
        from transform import train_transform, test_transform
    else:
        train_transform = None
        test_transform = None

    train_set, test_set = build_dataset(
        manifest_path=config.manifest_path,
        test_split_ratio=0.2,
        train_transform=train_transform,
        test_transform=test_transform
    )

    batch_size = config.batch_size
    trainloader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    testloader = DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    return trainloader, testloader


def train_pytorch():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)

    seed = config.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = build_model()
    model = model.to(device)

    optimizer = build_optimizer(model, config)
    loss_fn = CrossEntropyLoss()

    trainloader, testloader = build_loaders(config)

    prep_train_run_folder(SCRIPT_DIR, config)
    ckptManager = CheckpointManager(SCRIPT_DIR, config.name, model, optimizer)

    best_acc = 0
    best_ep = -1
    for epoch in range(config.epochs):

        # training
        train_loss = 0.
        num_batches = 0
        total_correct = 0
        total_data = 0
        model.train(True)
        for batch in trainloader:
            inputs, labels = batch.image_w_mask, batch.label
            inputs = inputs.to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            pred = model(inputs)
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            total_correct += (pred.argmax(1) == labels).sum().item()
            total_data += labels.size(0) # number of samples in a batch
        
        train_loss = train_loss/num_batches
        train_acc = total_correct / total_data
        print(f"Epochs [{epoch}/{config.epochs}]")
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Acc.", train_acc, epoch)

        # validation
        test_loss = 0.
        num_batches = 0
        total_correct = 0
        total_data = 0
        model.eval()
        with torch.no_grad():
            for batch in testloader:
                inputs, labels = batch.image_w_mask, batch.label
                inputs = inputs.to(device)
                labels = labels.long().to(device)

                pred = model(inputs)
                loss = loss_fn(pred, labels)

                test_loss += loss.item()
                num_batches += 1
                total_correct += (pred.argmax(1) == labels).sum().item()
                total_data += labels.size(0) # number of samples in a batch

        test_loss = test_loss / num_batches
        test_acc = total_correct / total_data
        writer.add_scalar("Test/Loss", test_loss, epoch)
        writer.add_scalar("Test/Acc.", test_acc, epoch)

        writer.flush()

        if (epoch+1) % config.ckp_frq == 0:
            ckptManager.save(epoch)

        if (epoch+1) % config.ckp_best_check_frq == 0 and test_acc > best_acc:
            best_acc = test_acc
            ckptManager.save(epoch, best=True)
            best_ep = epoch

    print("best epoch: ", best_ep)


def prepare_xgb_data(config, test_split_ratio: float = 0.2):
    import pandas as pd

    train_set, test_set = build_dataset(
        manifest_path=config.manifest_path,
        test_split_ratio=test_split_ratio,
        seed=config.seed,
    )

    train_ids = set(train_set.manifest["sample_id"].unique())
    test_ids = set(test_set.manifest["sample_id"].unique())

    xgb_df = pd.read_csv(config.xgboost.csv_path)
    xgb_df["sample_id"] = xgb_df["sample_id"].astype(str)

    missing_train = train_ids - set(xgb_df["sample_id"])
    missing_test = test_ids - set(xgb_df["sample_id"])
    if missing_train or missing_test:
        raise ValueError(
            f"Sample ID mismatch between PyTorch manifest and XGBoost CSV. "
            f"Missing in XGBoost CSV -> Train: {len(missing_train)}, Test: {len(missing_test)}"
        )

    train_df = xgb_df[xgb_df["sample_id"].isin(train_ids)].reset_index(drop=True)
    test_df = xgb_df[xgb_df["sample_id"].isin(test_ids)].reset_index(drop=True)

    X_train = train_df.drop(columns=["sample_id", "label"])
    y_train = train_df["label"]

    X_test = test_df.drop(columns=["sample_id", "label"])
    y_test = test_df["label"]

    return X_train, X_test, y_train, y_test


def train_xgboost():
    import xgboost as xgb
    import optuna
    import numpy as np
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.dummy import DummyClassifier

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cfg_xgb = config.xgboost
    seed = cfg_xgb.random_state
    X_train, X_test, y_train, y_test = prepare_xgb_data(config, 0.2)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(skf.split(X_train, y_train))  # her denemede aynı fold'lar

    fixed_params = {
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": seed,
    }

    def objective(trial):
        params = {
            **fixed_params,
            "n_estimators": 2000,
            "early_stopping_rounds": 50,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        }

        losses, best_iters = [], []
        for i, (tr_idx, val_idx) in enumerate(folds):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            clf = xgb.XGBClassifier(**params)
            clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            proba = clf.predict_proba(X_val)[:, 1]
            losses.append(log_loss(y_val, proba))
            best_iters.append(clf.best_iteration + 1)

            trial.report(np.mean(losses), i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("n_estimators", int(np.mean(best_iters)))
        trial.set_user_attr("cv_std", float(np.std(losses)))
        return float(np.mean(losses))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=100)

    best = study.best_trial
    print("CONFIGURATION:", config.name)
    print(f"Best CV log loss: {best.value:.4f} ± {best.user_attrs['cv_std']:.4f}")
    print("Best params:", best.params)
    print("n_estimators:", best.user_attrs["n_estimators"])

    best_model = xgb.XGBClassifier(
        **fixed_params, **best.params,
        n_estimators=best.user_attrs["n_estimators"],
    )
    best_model.fit(X_train, y_train)

    # Referans: hep çoğunluk sınıfını tahmin eden model
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    print("Baseline test accuracy:", accuracy_score(y_test, dummy.predict(X_test)))

    test_proba = best_model.predict_proba(X_test)[:, 1]
    print("Train accuracy:", accuracy_score(y_train, best_model.predict(X_train)))
    print("Test accuracy: ", accuracy_score(y_test, (test_proba >= 0.5).astype(int)))
    print("Test log loss: ", log_loss(y_test, test_proba))
    print("Test ROC-AUC:  ", roc_auc_score(y_test, test_proba))

    return best_model, study


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train model with PyTorch or XGBoost.")
    parser.add_argument(
        "framework",
        choices=["pytorch", "xgboost"],
        help="Training framework: 'pytorch' or 'xgboost'",
    )
    args = parser.parse_args()

    if args.framework == "pytorch":
        train_pytorch()
    elif args.framework == "xgboost":
        train_xgboost()


if __name__ == "__main__":
    main()


