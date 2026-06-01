import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch import optim

from data.cifar10 import build_dataloaders
from engine.checkpoint import save_checkpoint
from engine.trainer import evaluate, train_one_epoch
from models.simple_cnn import SimpleCNN
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train SimpleCNN baseline on CIFAR-10")

    parser.add_argument("--data-dir", type=str, default="./datasets")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-aug", action="store_true")
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

def main():
    args=parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"simple_cnn_baseline_{timestamp}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    args_path = run_dir / "args.json"
    metrics_path = run_dir / "metrics.csv"
    best_checkpoint_path = run_dir / "best.pt"
    last_checkpoint_path = run_dir / "last.pt"

    with args_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    write_csv_header(metrics_path)


    train_loader, val_loader, class_names = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_aug=args.data_aug,
    )

    model = SimpleCNN(num_classes=len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    best_val_acc = 0.0

    print(f"Device: {device}")
    print(f"Run directory: {run_dir}")
    print(f"Classes: {class_names}")
    for epoch in range(1, args.epochs + 1):
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

        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "args": vars(args),
            "class_names": class_names,
        }

        save_checkpoint(checkpoint_state, str(last_checkpoint_path))

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            checkpoint_state["best_val_acc"] = best_val_acc
            save_checkpoint(checkpoint_state, str(best_checkpoint_path))

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"best_val_acc={best_val_acc:.4f}"
        )

    print("Training finished.")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Metrics CSV: {metrics_path}")


if __name__ == "__main__":
    main()