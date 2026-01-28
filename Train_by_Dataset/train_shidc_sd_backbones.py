import os
import sys
import time
import random
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, precision_recall_fscore_support


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class EarlyStopping:
    """Stop if val_loss doesn't improve by min_delta for 'patience' epochs."""
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.counter = 0
            return

        if val_loss > (self.best_loss - self.min_delta):
            self.counter += 1
            print(f"EarlyStopping: No mejora ({self.counter}/{self.patience}) | "
                  f"Best: {self.best_loss:.4f} Curr: {val_loss:.4f}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def infer_class_indices(class_to_idx: dict):
    """
    Try to find indices for bg/neg/pos by folder names.
    Returns: idx_bg, idx_neg, idx_pos, class_names_ordered(list by idx)
    """
    idx_to_name = {v: k for k, v in class_to_idx.items()}
    class_names_ordered = [idx_to_name[i] for i in range(len(idx_to_name))]

    def find_idx(keys):
        for i, name in idx_to_name.items():
            n = name.lower()
            if any(k in n for k in keys):
                return i
        return None

    idx_bg  = find_idx(["bg", "back", "background", "fondo"])
    idx_pos = find_idx(["pos", "positive", "ki67_pos", "ki67+"])
    idx_neg = find_idx(["neg", "negative", "ki67_neg", "ki67-"])
    return idx_bg, idx_neg, idx_pos, class_names_ordered

def compute_class_weights(targets, num_classes, device):
    """
    weight[c] = total / (num_classes * count[c])  (stable inverse-frequency)
    """
    counts = Counter(targets)
    freq = np.array([counts.get(i, 0) for i in range(num_classes)], dtype=np.float64)
    if np.any(freq == 0):
        freq[freq == 0] = 1.0
    weights = freq.sum() / (num_classes * freq)
    return torch.tensor(weights, dtype=torch.float32, device=device)

def save_checkpoint(path: Path, payload: dict):
    torch.save(payload, str(path))


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0), #No Hue para no afectar la tincion
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf

def get_data_loaders(dataset_root: Path, batch_size: int, num_workers: int):
    train_tf, val_tf = build_transforms()

    train_dir = dataset_root / "Train"
    val_dir = dataset_root / "Validation"
    if not train_dir.exists():
        raise FileNotFoundError(f"No se encuentra {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"No se encuentra {val_dir}")

    train_ds = datasets.ImageFolder(root=train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(root=val_dir, transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_ds, val_ds, train_loader, val_loader

def build_model(arch: str, num_classes: int, device):
    arch = arch.lower().strip()

    if arch == "convnext_tiny":
        model = models.convnext_tiny(weights="DEFAULT")
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    elif arch == "resnext50":
        model = models.resnext50_32x4d(weights="IMAGENET1K_V1")
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif arch == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights="DEFAULT")
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Arquitectura no soportada: {arch}. Usa convnext_tiny | resnext50 | efficientnet_v2_s")

    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    device, amp_device, use_cuda):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="   Training", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=amp_device, enabled=use_cuda):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)     
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, num_classes,
             idx_neg, idx_pos, device, amp_device, use_cuda):

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(loader, desc="   Validating", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=amp_device, enabled=use_cuda):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    val_loss = running_loss / total
    val_acc = 100.0 * correct / total

    val_f1_macro = f1_score(all_labels, all_preds, average="macro")

    # F1 macro solo POS/NEG (para Ki-67 index)
    if idx_neg is not None and idx_pos is not None:
        val_f1_posneg = f1_score(all_labels, all_preds, labels=[idx_neg, idx_pos], average="macro")
    else:
        val_f1_posneg = float("nan")

    prec, rec, f1, sup = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(num_classes)), zero_division=0
    )

    per_class = {}
    for i in range(num_classes):
        per_class[f"val_prec_c{i}"] = float(prec[i])
        per_class[f"val_rec_c{i}"]  = float(rec[i])
        per_class[f"val_f1_c{i}"]   = float(f1[i])
        per_class[f"val_sup_c{i}"]  = int(sup[i])

    return val_loss, val_acc, val_f1_macro, val_f1_posneg, per_class



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="/media/HDD2/PatchBCData")
    parser.add_argument("--out_root", type=str, default="/media/HDD2/BCData_Training_Compare")
    parser.add_argument("--arch", type=str, required=True, choices=["convnext_tiny", "resnext50", "efficientnet_v2_s"])

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--wd", type=float, default=1e-4)

    parser.add_argument("--save_every", type=int, default=5)

    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min_delta", type=float, default=0.001)

    parser.add_argument("--sched_patience", type=int, default=3)
    parser.add_argument("--sched_factor", type=float, default=0.5)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()
    amp_device = "cuda" if use_cuda else "cpu"

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)
    run_dir = out_root / f"{args.arch}_lr{args.lr}_wd{args.wd}_bs{args.batch}"
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEntrenamiento BCData")
    print(f"Arch: {args.arch}")
    print(f"Device: {device}")
    print(f"Batch: {args.batch} | Epochs: {args.epochs}")
    print(f"LR: {args.lr} | WD: {args.wd}")
    print(f"Output: {run_dir}")

    train_ds, val_ds, train_loader, val_loader = get_data_loaders(dataset_root, args.batch, args.num_workers)
    num_classes = len(train_ds.classes)

    print(f"   class_to_idx: {train_ds.class_to_idx}")
    idx_bg, idx_neg, idx_pos, class_names = infer_class_indices(train_ds.class_to_idx)
    print(f"   idx->name: {list(enumerate(class_names))}")
    print(f"   idx_neg={idx_neg}, idx_pos={idx_pos}, idx_bg={idx_bg}")

    class_weights = compute_class_weights(train_ds.targets, num_classes, device)
    print(f"   class_weights(idx order): {class_weights.detach().cpu().numpy()}")

    model = build_model(args.arch, num_classes, device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=args.sched_patience, factor=args.sched_factor
    )

    scaler = torch.amp.GradScaler(device=amp_device, enabled=use_cuda)
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    history = []
    start_time = time.time()

    best_val_loss = float("inf")
    best_f1_posneg = -1.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        try:
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler,
                device, amp_device, use_cuda
            )
            val_loss, val_acc, val_f1_macro, val_f1_posneg, per_class = validate(
                model, val_loader, criterion, num_classes,
                idx_neg, idx_pos, device, amp_device, use_cuda
            )


            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            print(f"   Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | LR: {current_lr:.2e}")
            print(f"   Val   Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | "
                  f"F1(Macro): {val_f1_macro:.4f} | F1(Pos/Neg): {val_f1_posneg:.4f}")

            row = {
                "arch": args.arch,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1_macro": val_f1_macro,
                "val_f1_posneg": val_f1_posneg,
                "lr": current_lr,
            }
            row.update(per_class)
            history.append(row)

            pd.DataFrame(history).to_csv(run_dir / "training_log.csv", index=False)

            if epoch % args.save_every == 0:
                save_checkpoint(ckpt_dir / f"epoch_{epoch}.pth", {
                    "epoch": epoch,
                    "arch": args.arch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_f1_posneg": best_f1_posneg,
                    "early_best_loss": early_stopping.best_loss,
                    "early_counter": early_stopping.counter,
                    "class_to_idx": train_ds.class_to_idx
                })

            # best by val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(run_dir / "best_loss.pth", {
                    "epoch": epoch,
                    "arch": args.arch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_f1_posneg": best_f1_posneg,
                    "class_to_idx": train_ds.class_to_idx
                })
                print(f"  ** Best(Loss): {best_val_loss:.4f} (epoch {epoch})")

            # best by F1 pos/neg
            if not np.isnan(val_f1_posneg) and val_f1_posneg > best_f1_posneg:
                best_f1_posneg = val_f1_posneg
                save_checkpoint(run_dir / "best_f1posneg.pth", {
                    "epoch": epoch,
                    "arch": args.arch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_f1_posneg": best_f1_posneg,
                    "class_to_idx": train_ds.class_to_idx
                })
                print(f"   ** Best(F1 Pos/Neg): {best_f1_posneg:.4f} (epoch {epoch})")

            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f"\nEARLY STOPPING ACTIVADO en época {epoch}.")
                break

        except torch.cuda.OutOfMemoryError:
            print("OOM: Baja --batch (p.ej. 16) o cierra otros procesos GPU.")
            break

    total_min = (time.time() - start_time) / 60.0
    print(f"\nTerminado en {total_min:.2f} min")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"Best F1 pos/neg: {best_f1_posneg:.4f}")
    print(f"Logs: {run_dir / 'training_log.csv'}")


if __name__ == "__main__":
    main()
