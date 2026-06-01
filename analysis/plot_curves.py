import argparse
import csv
from pathlib import  Path

import matplotlib.pyplot as plt

def read_metrics(metrics_path:str):
    metrics_path=Path(metrics_path)

    epochs=[]
    train_loss=[]
    train_acc=[]
    val_loss=[]
    val_acc=[]

    with metrics_path.open("r",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        for row in reader:

            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_acc.append(float(row["train_acc"]))
            val_loss.append(float(row["val_loss"]))
            val_acc.append(float(row["val_acc"]))
    return epochs,train_loss,train_acc,val_loss,val_acc


def plot_loss(epochs, train_loss, val_loss, output_path: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", label="Train Loss")
    plt.plot(epochs, val_loss, marker="o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_accuracy(epochs,train_acc,val_acc,output_path:Path):
    plt.figure(figsize=(8,5))
    plt.plot(epochs,train_acc,marker='o',label="train Acc")
    plt.plot(epochs,val_acc,marker='o',label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path,dpi=200)
    plt.close()


def parse_args():
    parser=argparse.ArgumentParser(description="Plot loss and accuracy curves from metrics.csv")
    parser.add_argument("--metrics",type=str,required=True)
    parser.add_argument("--output_dir",type=str,default=None)

    return parser.parse_args()


def main():
    args=parse_args()

    metrics_path=Path(args.metrics)

    if args.output_dir is None:
        output_dir=metrics_path.parent
    else:
        output_dir=Path(args.output_dir)

    output_dir.mkdir(parents=True,exist_ok=True)

    epochs,train_loss,train_acc,val_loss,val_acc=read_metrics(metrics_path)

    loss_path=output_dir/"loss_curve.png"
    acc_path=output_dir/"accuracy_curve.png"
    plot_loss(epochs,train_loss,val_loss,loss_path)
    plot_accuracy(epochs,train_acc,val_acc,acc_path)
    print(f"Saved loss curve to{loss_path}")
    print(f"Saved accuracy curve to: {acc_path}")

if __name__=="__main__":
    main()


