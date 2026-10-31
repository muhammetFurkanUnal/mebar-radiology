from datetime import datetime
from pathlib import Path
from IPython.display import display
import socket
import subprocess
import sys
import time
import torch
import torchvision
from dataset import build_dataset
from model import T2Conv2DModel_Ch1
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from config import config

PARENT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parents[0]
sys.path.append(str(PARENT))
from lib import *

writer = SummaryWriter(SCRIPT_DIR / "runs" / config.name)

def build_model():
    return T2Conv2DModel_Ch1()

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
    train_set, test_set = build_dataset(
        manifest_path=config.manifest_path,
        test_split_ratio=0.2,
    )

    batch_size = config.batch_size
    trainloader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    testloader = DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )

    return trainloader, testloader


if __name__ == "__main__":
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

    prep_run_folder(SCRIPT_DIR, config)
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
            inputs, labels = batch.image, batch.label
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
                inputs, labels = batch.image, batch.label
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

