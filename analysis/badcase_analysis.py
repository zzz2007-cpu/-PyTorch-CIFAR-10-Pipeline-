import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from data.cifar10 import CIFAR10_MEAN, CIFAR10_STD, build_dataloaders
from engine.checkpoint import load_checkpoint
from models.factory import build_model
from utils.config import DEFAULT_CONFIG, deep_update

def parse_args():
    parser = argparse.ArgumentParser(description="Export bad cases for CIFAR-10 checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-per-class", type=int, default=20)
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

def denormalize(images: torch.Tensor):
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    images = images.cpu() * std + mean
    return torch.clamp(images, 0.0, 1.0)
def safe_name(name: str):
    return name.replace(" ", "_").replace("/", "_")

def export_badcases(
    model,
    dataloader,
    class_names,
    device,
    output_dir: Path,
    max_per_class: int,
):
    model.eval()

    badcase_dir = output_dir / "badcases"
    badcase_dir.mkdir(parents=True, exist_ok=True)

    csv_path = badcase_dir / "badcases.csv"
    saved_per_class = {idx: 0 for idx in range(len(class_names))}
    rows = []
    global_index = 0

    with torch.no_grad():
        for images,labels in dataloader:
            images=images.to(device)
            labels=labels.to(device)

            logits=model(images)
            probs=F.softmax(logits,dim=1)
            confidences,preds=torch.max(probs,dim=1)

            wrong_mask=preds!=labels

            if wrong_mask.sum().item()==0:
                continue
            raw_images=denormalize(images)

            for i in range(images.size(0)):
                if not wrong_mask[i]:
                    continue
            true_idx = labels[i].item()
            pred_idx = preds[i].item()

            if saved_per_class[true_idx] >= max_per_class:
                continue
            confidence = confidences[i].item()
            true_name = safe_name(class_names[true_idx])
            pred_name = safe_name(class_names[pred_idx])
            saved_per_class[true_idx] += 1
            global_index += 1
            filename = (
                f"{true_name}_pred_{pred_name}_"
                f"conf_{confidence:.2f}_{global_index:04d}.png"
            )
            image_path = badcase_dir / filename
            save_image(raw_images[i], image_path)
            rows.append({
                "image": filename,
                "true_label": class_names[true_idx],
                "pred_label": class_names[pred_idx],
                "confidence": confidence,
            })

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "true_label", "pred_label", "confidence"],
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({
                "image": row["image"],
                "true_label": row["true_label"],
                "pred_label": row["pred_label"],
                "confidence": f"{row['confidence']:.6f}",
            })

    return badcase_dir, csv_path, rows

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

    badcase_dir, csv_path, rows = export_badcases(
        model=model,
        dataloader=val_loader,
        class_names=class_names,
        device=device,
        output_dir=output_dir,
        max_per_class=args.max_per_class,
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model: {cfg['model']['name']}")
    print(f"Saved badcase images to: {badcase_dir}")
    print(f"Saved badcase table to: {csv_path}")
    print(f"Total exported badcases: {len(rows)}")

    print("\nTop high-confidence mistakes:")
    rows = sorted(rows, key=lambda x: x["confidence"], reverse=True)
    for row in rows[:10]:
        print(
            f"{row['image']} | "
            f"true={row['true_label']} "
            f"pred={row['pred_label']} "
            f"conf={row['confidence']:.4f}"
        )


if __name__ == "__main__":
    main()