import torch
from tqdm import tqdm

from utils.metrics import AverageMeter, accuracy


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch: int):
    model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    progress_bar = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        batch_acc = accuracy(logits, labels)

        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(batch_acc, batch_size)

        progress_bar.set_postfix(
            {
                "loss": f"{loss_meter.avg:.4f}",
                "acc": f"{acc_meter.avg:.4f}",
            }
        )

    return {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg,
    }


def evaluate(model, val_loader, criterion, device):
    model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    progress_bar = tqdm(val_loader, desc="Evaluate", leave=False)

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = images.size(0)
            batch_acc = accuracy(logits, labels)

            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(batch_acc, batch_size)

            progress_bar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "acc": f"{acc_meter.avg:.4f}",
            })

    return {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg,
    }
