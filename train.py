import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch
from torch import nn, optim

from data.cifar10 import build_dataloaders
from engine.checkpoint import load_checkpoint, save_checkpoint
from engine.trainer import evaluate, train_one_epoch
from models.factory import build_model
from utils.config import load_config
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 classifier")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def write_csv_header(csv_path: Path):
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "lr",
        ])


def append_csv_row(csv_path: Path, row: list):
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def build_optimizer(model, cfg: dict):
    opt_cfg = cfg["optimizer"]
    name = opt_cfg["name"].lower()

    if name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=opt_cfg["lr"],
            momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=opt_cfg.get("weight_decay", 0.0),
        )

    if name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg.get("weight_decay", 0.0),
        )

    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, cfg: dict):
    sch_cfg = cfg["scheduler"]
    name = sch_cfg["name"].lower()

    if name == "none":
        return None

    if name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sch_cfg.get("step_size", 10),
            gamma=sch_cfg.get("gamma", 0.1),
        )

    if name == "cosine":
        t_max = sch_cfg.get("t_max") or cfg["train"]["epochs"]
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    raise ValueError(f"Unsupported scheduler: {name}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_run_name = cfg["run_name"] or cfg["model"]["name"]
    if args.resume:
        resume_checkpoint_path = Path(args.resume)
        run_dir = resume_checkpoint_path.parent
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{base_run_name}_{timestamp}"
        run_dir = Path(cfg["train"]["output_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.csv"
    best_checkpoint_path = run_dir / "best.pt"
    last_checkpoint_path = run_dir / "last.pt"

    if not args.resume:
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        write_csv_header(metrics_path)
    elif not metrics_path.exists():
        write_csv_header(metrics_path)

    train_loader, val_loader, class_names = build_dataloaders(
        data_dir=cfg["data"]["data_dir"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        data_aug=cfg["data"]["data_aug"],
        label_noise_rate=cfg["data"].get("label_noise_rate", 0.0),
        label_noise_seed=cfg["data"].get("label_noise_seed", cfg["seed"]),
    )

    model = build_model(cfg["model"]["name"], num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    start_epoch = 1
    best_val_acc = 0.0
    epochs = cfg["train"]["epochs"]

    if args.resume:
        checkpoint = load_checkpoint(str(resume_checkpoint_path), device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if not checkpoint.get("scheduler_stepped_after_save", False):
                scheduler.step()
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        start_epoch = checkpoint["epoch"] + 1

    print(f"Device: {device}")
    print(f"Run directory: {run_dir}")
    if args.resume:
        print(f"Resumed from: {resume_checkpoint_path}")
        print(f"Continuing from epoch {start_epoch} to {epochs}")
    print(f"Classes: {class_names}")
    print(f"Model: {cfg['model']['name']}")
    print(f"Optimizer: {cfg['optimizer']['name']}")
    print(f"Scheduler: {cfg['scheduler']['name']}")

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        val_metrics = evaluate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        append_csv_row(metrics_path, [
            epoch,
            f"{train_metrics['loss']:.6f}",
            f"{train_metrics['acc']:.6f}",
            f"{val_metrics['loss']:.6f}",
            f"{val_metrics['acc']:.6f}",
            f"{current_lr:.8f}",
        ])

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            is_best = True
        else:
            is_best = False

        if scheduler is not None:
            scheduler.step()

        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scheduler_stepped_after_save": True,
            "best_val_acc": best_val_acc,
            "config": cfg,
            "class_names": class_names,
        }

        save_checkpoint(checkpoint_state, str(last_checkpoint_path))
        if is_best:
            save_checkpoint(checkpoint_state, str(best_checkpoint_path))

        print(
            f"Epoch [{epoch:03d}/{epochs:03d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"lr={current_lr:.6f} "
            f"best_val_acc={best_val_acc:.4f}"
        )

    print("Training finished.")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Metrics CSV: {metrics_path}")


if __name__ == "__main__":
    main()
