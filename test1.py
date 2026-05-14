import io
import os
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from resnet_cbam import resnet34_cbam, resnet50_cbam


BASE_DIR = Path(__file__).resolve().parent


def resolve_path(path: Optional[Union[str, os.PathLike]]) -> Optional[Path]:
    if not path:
        return None
    value = Path(path)
    return value if value.is_absolute() else BASE_DIR / value


def _build_model(model_path: Path, num_classes: int):
    """Build model architecture matching the checkpoint file."""
    name = model_path.name.lower()
    if "resnet34" in name:
        model = resnet34_cbam(pretrained=False, num_classes=num_classes)
    else:
        model = resnet50_cbam(pretrained=False, num_classes=num_classes)
    return model


class VideoProcessor:
    """Phone model inference helper for image files and webcam frames."""

    def __init__(
        self,
        model1_path,
        idx_to_labels_path,
        image_path=None,
        font_path="./SimHei.ttf",
        top_k=5,
    ):
        self.image_path = resolve_path(image_path)
        self.model_path = resolve_path(model1_path)
        self.labels_path = resolve_path(idx_to_labels_path)
        self.font_path = resolve_path(font_path)
        self.top_k = int(top_k)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if not self.model_path or not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        if not self.labels_path or not self.labels_path.exists():
            raise FileNotFoundError(f"标签文件不存在: {self.labels_path}")

        self.font = self._load_font(self.font_path)
        self.idx_to_labels = np.load(self.labels_path, allow_pickle=True).item()

        self.model = self._load_model()
        self.model = self.model.eval().to(self.device)

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.stop = False

        # EMA FPS smoothing
        self._fps_alpha = 0.1
        self._smooth_fps = 0.0

    def _load_font(self, font_path: Optional[Path]):
        try:
            if font_path and font_path.exists():
                return ImageFont.truetype(str(font_path), 32)
        except OSError:
            pass
        return ImageFont.load_default()

    def _load_model(self):
        """Load state_dict with fallback for legacy full-model pickle."""
        raw = torch.load(self.model_path, map_location=self.device)

        # State dict format — build model explicitly
        if isinstance(raw, dict):
            n_class = len(self.idx_to_labels)
            model = _build_model(self.model_path, n_class)

            model_dict = model.state_dict()
            compatible = {
                k: v
                for k, v in raw.items()
                if k in model_dict and model_dict[k].shape == v.shape
            }
            model.load_state_dict(compatible, strict=False)
            return model

        # Full model pickle (legacy format) — register CBAM classes for torch.load
        import torchvision.models.resnet as tv_resnet

        import resnet_cbam

        for name in ("ChannelAttention", "SpatialAttention", "CBAM", "ResNetWithCBAM"):
            if hasattr(resnet_cbam, name):
                setattr(tv_resnet, name, getattr(resnet_cbam, name))
        return raw

    def _predict_pil(self, image: Image.Image) -> List[Tuple[str, float]]:
        image = image.convert("RGB")
        input_img = self.test_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_logits = self.model(input_img)
            pred_softmax = F.softmax(pred_logits, dim=1)
            k = min(self.top_k, pred_softmax.shape[1])
            confs, pred_ids = torch.topk(pred_softmax, k)

        pred_ids = pred_ids.cpu().numpy().squeeze().tolist()
        confs = confs.cpu().numpy().squeeze().tolist()
        if not isinstance(pred_ids, list):
            pred_ids = [pred_ids]
            confs = [confs]

        results = []
        for pred_id, confidence in zip(pred_ids, confs):
            label = self.idx_to_labels.get(int(pred_id), str(pred_id))
            results.append((label, float(confidence)))
        return results

    def _update_fps(self, elapsed):
        fps = 1.0 / elapsed
        if self._smooth_fps == 0.0:
            self._smooth_fps = fps
        else:
            self._smooth_fps = (
                self._fps_alpha * fps + (1 - self._fps_alpha) * self._smooth_fps
            )
        return self._smooth_fps

    def process_frame(self, frame):
        start_time = time.time()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        predictions = self._predict_pil(img_pil)

        # Dynamic text positioning based on frame size
        h, w = frame.shape[:2]
        text_x = int(w * 0.03)
        text_y = int(h * 0.12)
        line_height = int(h * 0.055)

        draw = ImageDraw.Draw(img_pil)
        for index, (label, confidence) in enumerate(predictions):
            text = f"{label:<15} {confidence * 100:>.2f}%"
            draw.text((text_x, text_y + line_height * index), text, font=self.font, fill=(255, 0, 0))

        elapsed = max(time.time() - start_time, 1e-6)
        smoothed = self._update_fps(elapsed)
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        cv2.putText(
            frame,
            f"FPS {int(smoothed)}",
            (int(w * 0.03), int(h * 0.08)),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            4,
            cv2.LINE_AA,
        )
        return frame

    def _init_camera(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头: {camera_index}")
        return cap

    def run(self, camera_index=0):
        cap = self._init_camera(camera_index)
        try:
            while cap.isOpened() and not self.stop:
                success, frame = cap.read()
                if not success:
                    break
                cv2.imshow("phone", self.process_frame(frame))
                if cv2.waitKey(1) in [ord("q"), 27]:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def capture_and_predict(self):
        cap = self._init_camera(0)
        try:
            success, frame = cap.read()
            if not success:
                raise RuntimeError("无法读取摄像头画面")
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return self._format_results(self._predict_pil(image))
        finally:
            cap.release()

    def predict_image(self, image_path=None):
        target = resolve_path(image_path) or self.image_path
        if not target or not target.exists():
            raise FileNotFoundError(f"图片文件不存在: {target}")
        image = Image.open(target)
        return self._format_results(self._predict_pil(image))

    @staticmethod
    def _format_results(results: Sequence[Tuple[str, float]]) -> List[str]:
        return [f"{label}: {confidence * 100:.2f}%" for label, confidence in results]

    def run_detection(self, mode, image_path=None, output=None):
        output = output or io.StringIO()
        if mode == "realtime":
            self.run()
        elif mode == "image":
            for line in self.predict_image(image_path):
                print(line, file=output)
        else:
            print(f"未知检测模式: {mode}", file=output)
        return output


if __name__ == "__main__":
    import json

    with open(BASE_DIR / "parameters.json", "r", encoding="utf-8") as f:
        parameters = json.load(f)

    processor = VideoProcessor(
        model1_path=parameters["model1_path"],
        idx_to_labels_path=parameters["idx_to_labels_path"],
        image_path=parameters.get("image_path"),
    )
    buffer = io.StringIO()
    processor.run_detection("image", parameters.get("image_path"), buffer)
    print(buffer.getvalue())
