import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import ImageFile
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from resnet_cbam import resnet34_cbam, resnet50_cbam

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


ImageFile.LOAD_TRUNCATED_IMAGES = True
BASE_DIR = Path(__file__).resolve().parent


def resolve_path(path: Optional[Union[str, os.PathLike]]) -> Optional[Path]:
    if not path:
        return None
    value = Path(path)
    return value if value.is_absolute() else BASE_DIR / value


class PhoneClassifier:
    """Train a ResNet/CBAM phone classifier from an ImageFolder dataset."""

    def __init__(
        self,
        dataset_dir,
        model_path,
        batch_size=16,
        epochs=10,
        step_size=5,
        gamma=0.1,
        load_method="init_and_load_model",
        save=None,
        lr=0.001,
        weight_decay=1e-4,
        patience=5,
        num_workers=8,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset_dir = resolve_path(dataset_dir)
        self.model_path = resolve_path(model_path)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.step_size = int(float(step_size))
        self.gamma = float(gamma)
        self.load_method = load_method
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.num_workers = int(num_workers)
        self.output_dir = resolve_path(save) or self.dataset_dir
        self.checkpoint_dir = self.output_dir / "checkpoint"

        self._validate_paths()
        self.train_transform, self.test_transform = self._build_transforms()
        self.train_path = self.dataset_dir / "train"
        self.test_path = self.dataset_dir / "val"

        self.train_dataset = datasets.ImageFolder(str(self.train_path), self.train_transform)
        self.test_dataset = datasets.ImageFolder(str(self.test_path), self.test_transform)
        self.class_names = self.train_dataset.classes
        self.n_class = len(self.class_names)
        self.idx_to_labels = {y: x for x, y in self.train_dataset.class_to_idx.items()}

        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.output_dir / "idx_to_labels.npy", self.idx_to_labels)
        np.save(self.output_dir / "labels_to_idx.npy", self.train_dataset.class_to_idx)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

        self.model, self.optimizer = self._load_model()
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=1e-6,
        )
        self.best_test_accuracy = 0.0
        self.epoch = 0
        self.batch_idx = 0
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")

    def _validate_paths(self):
        if not self.dataset_dir or not self.dataset_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在: {self.dataset_dir}")
        if not (self.dataset_dir / "train").exists():
            raise FileNotFoundError(f"缺少训练目录: {self.dataset_dir / 'train'}")
        if not (self.dataset_dir / "val").exists():
            raise FileNotFoundError(f"缺少验证目录: {self.dataset_dir / 'val'}")
        if not self.model_path or not self.model_path.exists():
            raise FileNotFoundError(f"预训练模型不存在: {self.model_path}")

    @staticmethod
    def _build_transforms():
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return train_transform, test_transform

    def _has_cbam(self):
        return self.load_method in {"init_and_load_model", "download_and_load_model"}

    def _create_model(self):
        model_name = self.model_path.name.lower()
        use_cbam = self._has_cbam()

        if "resnet34" in model_name:
            return resnet34_cbam(weights=None) if use_cbam else models.resnet34(weights=None)
        if "resnet50" in model_name:
            return resnet50_cbam(weights=None) if use_cbam else models.resnet50(weights=None)
        raise ValueError(f"无法根据文件名识别模型结构: {self.model_path.name}")

    def _load_model(self) -> Tuple[nn.Module, optim.Optimizer]:
        model = self._create_model()
        raw = torch.load(self.model_path, map_location="cpu")

        # Extract state_dict from full model or dict format
        if hasattr(raw, "state_dict"):
            state_dict = raw.state_dict()
        elif isinstance(raw, dict):
            state_dict = raw
        else:
            raise ValueError(f"不支持的模型文件格式: {self.model_path}")

        # Handle key prefix mismatch: torchvision state_dict keys lack "base."
        # prefix, but CBAM-wrapped model expects them.
        model_keys = list(model.state_dict().keys())
        if (
            self._has_cbam()
            and model_keys
            and model_keys[0].startswith("base.")
            and not any(k.startswith("base.") for k in state_dict)
        ):
            state_dict = {"base." + k: v for k, v in state_dict.items()}

        model_dict = model.state_dict()
        compatible_state = {
            key: value
            for key, value in state_dict.items()
            if key in model_dict and model_dict[key].shape == value.shape
        }
        skipped = len(state_dict) - len(compatible_state)
        if skipped:
            print(f"跳过 {skipped} 个不兼容的权重层（通常是分类头尺寸不匹配）")

        model_dict.update(compatible_state)
        model.load_state_dict(model_dict, strict=False)

        # Replace classifier head — CBAM model uses base.fc, plain ResNet uses fc
        if self._has_cbam():
            in_features = model.base.fc.in_features
            model.base.fc = nn.Linear(in_features, self.n_class)
        else:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, self.n_class)

        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return model, optimizer

    @staticmethod
    def _metrics(labels, preds, prefix: str) -> Dict[str, float]:
        return {
            f"{prefix}_accuracy": accuracy_score(labels, preds),
            f"{prefix}_precision": precision_score(labels, preds, average="macro", zero_division=0),
            f"{prefix}_recall": recall_score(labels, preds, average="macro", zero_division=0),
            f"{prefix}_f1": f1_score(labels, preds, average="macro", zero_division=0),
        }

    def train_one_batch(self, images, labels):
        self.model.train()
        images = images.to(self.device)
        labels = labels.to(self.device)

        with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        log_train = {
            "epoch": self.epoch,
            "batch": self.batch_idx,
            "train_loss": float(loss.detach().cpu().item()),
        }
        log_train.update(self._metrics(labels_np, preds, "train"))
        return log_train

    def evaluate_testset(self):
        self.model.eval()
        loss_list = []
        labels_list = []
        preds_list = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)

                loss_list.append(float(loss.detach().cpu().item()))
                labels_list.extend(labels.detach().cpu().numpy())
                preds_list.extend(preds.detach().cpu().numpy())

        log_test = {
            "epoch": self.epoch,
            "test_loss": float(np.mean(loss_list)) if loss_list else 0.0,
        }
        log_test.update(self._metrics(labels_list, preds_list, "test"))
        return log_test

    def _save_best_checkpoint(self, accuracy: float) -> Path:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        suffix = "-cbam.pth" if self._has_cbam() else ".pth"
        checkpoint_path = self.checkpoint_dir / f"best-{accuracy:.3f}{suffix}"
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def train(self, callback=None):
        emit = self._normalize_callback(callback)
        emit(f"设备: {self.device}\n")
        emit(f"训练集: {self.train_path}\n")
        emit(f"验证集: {self.test_path}\n")
        emit(f"类别数: {self.n_class}\n")

        if _WANDB_AVAILABLE:
            wandb.init(
                project="phone",
                name=f"{self.model_path.stem}-{time.strftime('%m%d-%H%M')}",
                config={
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                    "patience": self.patience,
                    "load_method": self.load_method,
                    "n_class": self.n_class,
                    "device": str(self.device),
                    "model": self.model_path.name,
                },
                dir=str(self.output_dir),
                reinit=True,
            )
            emit("wandb 已启用\n")

        train_logs = []
        test_logs = []
        epochs_no_improve = 0

        for self.epoch in range(1, self.epochs + 1):
            emit(f"\nEpoch {self.epoch}/{self.epochs}\n")
            self.model.train()

            for images, labels in tqdm(self.train_loader, desc=f"Epoch {self.epoch}", leave=False):
                self.batch_idx += 1
                log_train = self.train_one_batch(images, labels)
                train_logs.append(log_train)

            self.lr_scheduler.step()
            log_test = self.evaluate_testset()
            test_logs.append(log_test)

            # Log epoch metrics to wandb
            if _WANDB_AVAILABLE and wandb.run:
                epoch_train = pd.DataFrame(train_logs)
                epoch_train = epoch_train[epoch_train["epoch"] == self.epoch]
                wandb.log(
                    {
                        "epoch": self.epoch,
                        "train/loss": float(epoch_train["train_loss"].mean()),
                        "train/accuracy": float(epoch_train["train_accuracy"].mean()),
                        "train/f1": float(epoch_train["train_f1"].mean()),
                        "test/loss": log_test["test_loss"],
                        "test/accuracy": log_test["test_accuracy"],
                        "test/precision": log_test["test_precision"],
                        "test/recall": log_test["test_recall"],
                        "test/f1": log_test["test_f1"],
                        "lr": self.lr_scheduler.get_last_lr()[0],
                    },
                    step=self.epoch,
                )

            emit(
                "验证: loss={test_loss:.4f}, acc={test_accuracy:.4f}, "
                "precision={test_precision:.4f}, recall={test_recall:.4f}, f1={test_f1:.4f}\n".format(**log_test)
            )

            if log_test["test_accuracy"] > self.best_test_accuracy:
                self.best_test_accuracy = log_test["test_accuracy"]
                checkpoint_path = self._save_best_checkpoint(self.best_test_accuracy)
                epochs_no_improve = 0
                emit(f"保存最佳模型: {checkpoint_path}\n")
            else:
                epochs_no_improve += 1
                emit(f"验证准确率未提升 ({epochs_no_improve}/{self.patience})\n")

            if epochs_no_improve >= self.patience:
                emit(f"早停: {self.patience} 轮未提升，停止训练\n")
                break

        df_train_log = pd.DataFrame(train_logs)
        df_test_log = pd.DataFrame(test_logs)
        train_log_path = self.output_dir / "train_log.csv"
        test_log_path = self.output_dir / "val_log.csv"
        df_train_log.to_csv(train_log_path, index=False, encoding="utf-8-sig")
        df_test_log.to_csv(test_log_path, index=False, encoding="utf-8-sig")
        emit(f"训练日志: {train_log_path}\n")
        emit(f"验证日志: {test_log_path}\n")

        curves_path = self._plot_curves(train_logs, test_logs)
        if curves_path:
            emit(f"训练曲线: {curves_path}\n")

        if _WANDB_AVAILABLE and wandb.run:
            if curves_path:
                wandb.log({"curves": wandb.Image(str(curves_path))})
            wandb.log({"best/test_accuracy": self.best_test_accuracy})
            wandb.finish()

        emit(f"最佳验证准确率: {self.best_test_accuracy:.4f}\n")
        return {
            "best_test_accuracy": self.best_test_accuracy,
            "train_log_path": str(train_log_path),
            "test_log_path": str(test_log_path),
        }

    def _plot_curves(self, train_logs, test_logs):
        """Plot 2x2 evaluation curves (loss, accuracy, precision/recall, F1)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm
        except ImportError:
            return None

        font_path = BASE_DIR / "SimHei.ttf"
        if font_path.exists():
            try:
                fm.fontManager.addfont(str(font_path))
                plt.rcParams["font.family"] = fm.FontProperties(fname=str(font_path)).get_name()
            except Exception:
                pass
        plt.rcParams["axes.unicode_minus"] = False

        df_train = pd.DataFrame(train_logs)
        df_test = pd.DataFrame(test_logs)
        train_epoch = df_train.groupby("epoch", sort=True).agg(train_loss=("train_loss", "mean")).reset_index()

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        fig.suptitle("Training Evaluation Curves", fontsize=14, fontweight="bold", y=0.98)

        # 1. Loss
        ax = axes[0, 0]
        ax.plot(train_epoch["epoch"], train_epoch["train_loss"], "b-", label="Train Loss", linewidth=1.5)
        ax.plot(df_test["epoch"], df_test["test_loss"], "r-", label="Test Loss", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 2. Accuracy
        ax = axes[0, 1]
        ax.plot(df_test["epoch"], df_test["test_accuracy"], "g-", linewidth=2)
        best_epoch = df_test.loc[df_test["test_accuracy"].idxmax(), "epoch"]
        best_acc = df_test["test_accuracy"].max()
        ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.5)
        ax.annotate(f"best {best_acc:.3f}", xy=(best_epoch, best_acc),
                    xytext=(8, 8), textcoords="offset points", fontsize=9, color="gray")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Test Accuracy")
        ax.grid(True, alpha=0.3)

        # 3. Precision & Recall
        ax = axes[1, 0]
        ax.plot(df_test["epoch"], df_test["test_precision"], "b-", label="Precision", linewidth=1.5)
        ax.plot(df_test["epoch"], df_test["test_recall"], color="orange", label="Recall", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("Precision & Recall")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 4. F1 Score
        ax = axes[1, 1]
        ax.plot(df_test["epoch"], df_test["test_f1"], "m-", linewidth=2)
        best_f1 = df_test["test_f1"].max()
        ax.axhline(best_f1, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1")
        ax.set_title(f"Test F1 (best {best_f1:.3f})")
        ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = self.output_dir / "training_curves.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    @staticmethod
    def _normalize_callback(callback):
        if callback is None:
            return print
        if hasattr(callback, "write"):
            return callback.write
        return callback


if __name__ == "__main__":
    with open(BASE_DIR / "parameters.json", "r", encoding="utf-8") as f:
        parameters = json.load(f)

    classifier = PhoneClassifier(
        dataset_dir=parameters["dataset_dir"],
        model_path=parameters["model_path"],
        batch_size=parameters.get("batch_size", 16),
        epochs=parameters.get("epochs", 10),
        step_size=parameters.get("step_size", 5),
        gamma=parameters.get("gamma", 0.1),
        load_method=parameters.get("load_method", "init_and_load_model"),
        save=parameters.get("save") or parameters["dataset_dir"],
        lr=parameters.get("lr", 0.001),
        weight_decay=parameters.get("weight_decay", 1e-4),
        patience=parameters.get("patience", 5),
        num_workers=parameters.get("num_workers", 0),
    )
    classifier.train()
