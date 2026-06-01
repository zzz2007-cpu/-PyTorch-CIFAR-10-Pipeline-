import argparse

import torch
from torch import nn

from data.cifar10 import build_dataloaders
from engine.checkpoint import load_checkpoint
from engine.trainer import evaluate
from models.simple_cnn import SimpleCNN


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SimpleCNN checkpoint on CIFAR-10")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="./datasets")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, class_names = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_aug=False,
    )

    model = SimpleCNN(num_classes=len(class_names))
    model = model.to(device)

    checkpoint = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()

    metrics = evaluate(
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Best val acc in training: {checkpoint['best_val_acc']:.4f}")
    print(f"Eval loss: {metrics['loss']:.4f}")
    print(f"Eval acc: {metrics['acc']:.4f}")


if __name__ == "__main__":
    main()
