import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.cifar10 import build_dataloaders
from engine.checkpoint import load_checkpoint
from models.factory import build_model
from utils.config import DEFAULT_CONFIG, deep_update


def parse_args():
    parser = argparse.ArgumentParser(description="Generate confusion matrix for CIFAR-10 checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()

def config_from_checkpoint(checkpoint: dict) -> dict:
    if "config" in checkpoint:
        return checkpoint["config"]

    old_args = checkpoint.get("args", {})
    legacy_config = {
        "run_name": old_args.get("run_name"),
        "seed": old_args.get("seed", DEFAULT_CONFIG["seed"]),
        "data": {
            "data_dir": old_args.get("data_dir", DEFAULT_CONFIG["data"]["data_dir"]),
            "batch_size": old_args.get("batch_size", DEFAULT_CONFIG["data"]["batch_size"]),
            "num_workers": old_args.get("num_workers", DEFAULT_CONFIG["data"]["num_workers"]),
            "data_aug": old_args.get("data_aug", False),
        },
        "model": {
            "name": "simple_cnn",
        },
        "train": {
            "epochs": old_args.get("epochs", DEFAULT_CONFIG["train"]["epochs"]),
            "output_dir": old_args.get("output_dir", DEFAULT_CONFIG["train"]["output_dir"]),
        },
        "optimizer": {
            "name": "sgd",
            "lr": old_args.get("lr", DEFAULT_CONFIG["optimizer"]["lr"]),
            "momentum": old_args.get("momentum", DEFAULT_CONFIG["optimizer"]["momentum"]),
            "weight_decay": old_args.get(
                "weight_decay",
                DEFAULT_CONFIG["optimizer"]["weight_decay"],
            ),
        },
        "scheduler": {
            "name": "none",
        },
    }
    return deep_update(DEFAULT_CONFIG, legacy_config)

def collect_predictions(model,dataloader,device):
    model.eval()

    all_targets=[]
    all_preds=[]

    with torch.no_gard():
        for images,label in dataloader:
            images=images.to(device)
            labels=labels.to(device)

            logits=model(images)
            preds=torch.argmax(logits,dim=1)

            all_targets.append(labels.cpu())
            all_preds.append(preds.cpu())
        targets=torch.cat(all_targets).numpy()
        preds=torch.cat(all_preds).numpy()

        return targets,preds
def build_confusion_matrix(targets,preds,num_classes:int):
    martrix=np.zeros((num_classes,num_classes),dtype=np.int64)

    for true_label,pred_label in zip(targets,preds):
        martrix[true_label,pred_label]+=1
    return martrix

def compute_per_class_accuracy(matrix, class_names):
    rows = []

    for idx, class_name in enumerate(class_names):
        total = matrix[idx].sum()
        correct = matrix[idx, idx]

        if total == 0:
            acc = 0.0
        else:
            acc = correct / total

        rows.append({
            "class": class_name,
            "correct": int(correct),
            "total": int(total),
            "accuracy": acc,
        })

    return rows

def save_per_class_accuracy(rows, output_path: Path):
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class", "correct", "total", "accuracy"],
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({
                "class": row["class"],
                "correct": row["correct"],
                "total": row["total"],
                "accuracy": f"{row['accuracy']:.6f}",
            })   

def plot_confusion_matrix(matrix, class_names, output_path: Path):
    plt.figure(figsize=(9, 8))

    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    threshold = matrix.max() / 2

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if value > threshold else "black"
            plt.text(
                j,
                i,
                str(value),
                horizontalalignment="center",
                verticalalignment="center",
                color=color,
                fontsize=8,
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_confusion_matrix(matrix, class_names, output_path: Path):
    plt.figure(figsize=(9, 8))

    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    threshold = matrix.max() / 2

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if value > threshold else "black"
            plt.text(
                j,
                i,
                str(value),
                horizontalalignment="center",
                verticalalignment="center",
                color=color,
                fontsize=8,
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(str(checkpoint_path), device)
    cfg = config_from_checkpoint(checkpoint)

    data_dir = args.data_dir or cfg["data"]["data_dir"]
    batch_size = args.batch_size or cfg["data"]["batch_size"]
    num_workers = args.num_workers or cfg["data"]["num_workers"]

    if args.output_dir is None:
        output_dir = checkpoint_path.parent
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    _, val_loader, class_names = build_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        data_aug=False,
    )

    model = build_model(cfg["model"]["name"], num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    targets, preds = collect_predictions(
        model=model,
        dataloader=val_loader,
        device=device,
    )

    matrix = build_confusion_matrix(
        targets=targets,
        preds=preds,
        num_classes=len(class_names),
    )

    matrix_path = output_dir / "confusion_matrix.png"
    per_class_path = output_dir / "per_class_accuracy.csv"

    plot_confusion_matrix(
        matrix=matrix,
        class_names=class_names,
        output_path=matrix_path,
    )

    per_class_rows = compute_per_class_accuracy(
        matrix=matrix,
        class_names=class_names,
    )
    save_per_class_accuracy(per_class_rows, per_class_path)

    overall_acc = np.trace(matrix) / matrix.sum()

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model: {cfg['model']['name']}")
    print(f"Overall accuracy: {overall_acc:.4f}")
    print(f"Saved confusion matrix to: {matrix_path}")
    print(f"Saved per-class accuracy to: {per_class_path}")

    print("\nPer-class accuracy:")
    for row in per_class_rows:
        print(
            f"{row['class']:>10s}: "
            f"{row['accuracy']:.4f} "
            f"({row['correct']}/{row['total']})"
        )


if __name__ == "__main__":
    main()