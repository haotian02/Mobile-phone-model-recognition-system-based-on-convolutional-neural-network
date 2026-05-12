import io
import json
import os
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
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset_dir = resolve_path(dataset_dir)
        self.model_path = resolve_path(model_path)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.step_size = int(float(step_size))
        self.gamma = float(gamma)
        self.load_method = load_method
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
            num_workers=0,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        self.model, self.optimizer = self._load_model()
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.step_size,
            gamma=self.gamma,
        )
        self.best_test_accuracy = 0.0
        self.epoch = 0
        self.batch_idx = 0

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

    def _create_model(self):
        model_name = self.model_path.name.lower()
        use_cbam = self.load_method in {"init_and_load_model", "download_and_load_model"}

        if "resnet34" in model_name:
            return resnet34_cbam(pretrained=False) if use_cbam else models.resnet34(weights=None)
        if "resnet50" in model_name:
            return resnet50_cbam(pretrained=False) if use_cbam else models.resnet50(weights=None)
        raise ValueError(f"无法根据文件名识别模型结构: {self.model_path.name}")

    def _load_model(self) -> Tuple[nn.Module, optim.Optimizer]:
        model = self._create_model()
        checkpoint = torch.load(self.model_path, map_location="cpu")
        state_dict = checkpoint.state_dict() if hasattr(checkpoint, "state_dict") else checkpoint
        if not isinstance(state_dict, dict):
            raise ValueError(f"模型文件格式不支持: {self.model_path}")

        model_dict = model.state_dict()
        compatible_state = {
            key: value
            for key, value in state_dict.items()
            if key in model_dict and model_dict[key].shape == value.shape
        }
        model_dict.update(compatible_state)
        model.load_state_dict(model_dict, strict=False)
        model.fc = nn.Linear(model.fc.in_features, self.n_class)
        optimizer = optim.Adam(model.parameters())
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

        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
        checkpoint_path = self.checkpoint_dir / f"best-{accuracy:.3f}.pth"
        torch.save(self.model, checkpoint_path)
        return checkpoint_path

    def train(self, callback=None):
        emit = self._normalize_callback(callback)
        emit(f"设备: {self.device}\n")
        emit(f"训练集: {self.train_path}\n")
        emit(f"验证集: {self.test_path}\n")
        emit(f"类别数: {self.n_class}\n")

        df_train_log = pd.DataFrame()
        df_test_log = pd.DataFrame()

        for self.epoch in range(1, self.epochs + 1):
            emit(f"\nEpoch {self.epoch}/{self.epochs}\n")
            self.model.train()

            for images, labels in tqdm(self.train_loader, desc=f"Epoch {self.epoch}", leave=False):
                self.batch_idx += 1
                log_train = self.train_one_batch(images, labels)
                df_train_log = pd.concat([df_train_log, pd.DataFrame([log_train])], ignore_index=True)

            self.lr_scheduler.step()
            log_test = self.evaluate_testset()
            df_test_log = pd.concat([df_test_log, pd.DataFrame([log_test])], ignore_index=True)
            emit(
                "验证: loss={test_loss:.4f}, acc={test_accuracy:.4f}, "
                "precision={test_precision:.4f}, recall={test_recall:.4f}, f1={test_f1:.4f}\n".format(**log_test)
            )

            if log_test["test_accuracy"] > self.best_test_accuracy:
                self.best_test_accuracy = log_test["test_accuracy"]
                checkpoint_path = self._save_best_checkpoint(self.best_test_accuracy)
                emit(f"保存最佳模型: {checkpoint_path}\n")

        train_log_path = self.output_dir / "train_log.csv"
        test_log_path = self.output_dir / "val_log.csv"
        df_train_log.to_csv(train_log_path, index=False, encoding="utf-8-sig")
        df_test_log.to_csv(test_log_path, index=False, encoding="utf-8-sig")
        emit(f"训练日志: {train_log_path}\n")
        emit(f"验证日志: {test_log_path}\n")
        emit(f"最佳验证准确率: {self.best_test_accuracy:.4f}\n")
        return {
            "best_test_accuracy": self.best_test_accuracy,
            "train_log_path": str(train_log_path),
            "test_log_path": str(test_log_path),
        }

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
    )
    classifier.train()
