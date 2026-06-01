import argparse

import torch
from torch import nn

from data.cifar10 import build_dataloaders
from engine.checkpoint import load_checkpoint
from engine.trainer import evaluate
from models.factory import build_model
from utils.config import DEFAULT_CONFIG, deep_update


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CIFAR-10 checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
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


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = load_checkpoint(args.checkpoint, device)
    cfg = config_from_checkpoint(checkpoint)

    data_dir = args.data_dir or cfg["data"]["data_dir"]
    batch_size = args.batch_size or cfg["data"]["batch_size"]
    num_workers = args.num_workers or cfg["data"]["num_workers"]

    _, val_loader, class_names = build_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        data_aug=False,
    )

    model = build_model(cfg["model"]["name"], num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()

    metrics = evaluate(
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {cfg['model']['name']}")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Best val acc in training: {checkpoint['best_val_acc']:.4f}")
    print(f"Eval loss: {metrics['loss']:.4f}")
    print(f"Eval acc: {metrics['acc']:.4f}")


if __name__ == "__main__":
    main()
