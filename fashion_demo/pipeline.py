"""Inference and visualization shared by the command line and Streamlit UI."""
from __future__ import annotations

import io
import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
import torch
import torch.nn.functional as nnf
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

ROOT = Path(__file__).resolve().parents[1]
FASHION_CATEGORIES = {"person", "backpack", "umbrella", "handbag", "tie", "suitcase"}
CHINESE_NAMES = {"person": "人物", "backpack": "背包", "umbrella": "雨伞", "handbag": "手提包", "tie": "领带", "suitcase": "行李箱"}
COLORS = {"person": "#009E73", "backpack": "#E69F00", "umbrella": "#56B4E9", "handbag": "#D55E00", "tie": "#CC79A7", "suitcase": "#0072B2"}


def read_image(source: str | Path | BinaryIO) -> Image.Image:
    """Honor EXIF orientation and retain the complete image (no center crop)."""
    try:
        with Image.open(source) as image:
            if image.width * image.height > 24_000_000:
                raise ValueError("图片超过2400万像素，请先缩小后再试。")
            result = ImageOps.exif_transpose(image).convert("RGB")
            result.load()
            return result
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("无法读取图片，请使用完整的JPG、PNG或WebP文件。") from exc


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def draw_boxes(image: Image.Image, detections: list[dict]) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font_size = max(14, round(min(image.size) / 38))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default(size=font_size)
    line_width = max(2, round(min(image.size) / 230))
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        color = COLORS[detection["category"]]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        label = f'{detection["category"]} {detection["score"]:.2f}'
        bounds = draw.textbbox((0, 0), label, font=font)
        text_width, text_height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        left = max(0, min(x1, canvas.width - text_width - 8))
        top = max(0, y1 - text_height - 8)
        draw.rectangle((left, top, left + text_width + 8, top + text_height + 8), fill=color)
        draw.text((left + 4, top + 4 - bounds[1]), label, fill="white", font=font)
    return canvas


def overlay_mask(image: Image.Image, mask: np.ndarray, alpha: float) -> Image.Image:
    if mask.shape != (image.height, image.width):
        raise ValueError("分割掩码与原图尺寸不一致。")
    array = np.asarray(image).copy()
    array[mask] = np.round(array[mask] * (1 - alpha) + np.array([40, 130, 230]) * alpha).astype(np.uint8)
    return Image.fromarray(array)


@dataclass
class Result:
    original: Image.Image
    detection: Image.Image
    segmentation: Image.Image
    combined: Image.Image
    mask: Image.Image
    metadata: dict

    def archive(self) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in ("original", "detection", "segmentation", "combined", "mask"):
                archive.writestr(f"{name}.png", png_bytes(getattr(self, name)))
            archive.writestr("metrics.json", json.dumps(self.metadata, ensure_ascii=False, indent=2))
        return buffer.getvalue()

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("original", "detection", "segmentation", "combined", "mask"):
            getattr(self, name).save(directory / f"{name}.png")
        (directory / "metrics.json").write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")


class FashionPipeline:
    def __init__(self, device: str = "auto"):
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device必须是auto、cpu或cuda。")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("当前PyTorch环境没有可用的CUDA，请选择CPU。")
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
        torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
        torch.hub.set_dir(str(ROOT / ".cache" / "torch" / "hub"))
        self.detection_weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        self.segmentation_weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        self.detector = fasterrcnn_resnet50_fpn(weights=self.detection_weights).to(self.device).eval()
        self.segmenter = deeplabv3_resnet50(weights=self.segmentation_weights).to(self.device).eval()
        self.categories = self.detection_weights.meta["categories"]
        self.person_index = self.segmentation_weights.meta["categories"].index("person")

    @torch.inference_mode()
    def run(self, image: Image.Image, threshold: float = 0.5, alpha: float = 0.45) -> Result:
        if not 0 <= threshold <= 1 or not 0 <= alpha <= 1:
            raise ValueError("置信度和透明度必须在0到1之间。")
        image = image.convert("RGB")
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        det_tensor = self.detection_weights.transforms()(image).to(self.device)
        predictions = self.detector([det_tensor])[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        detection_end = time.perf_counter()
        seg_tensor = self.segmentation_weights.transforms()(image).unsqueeze(0).to(self.device)
        logits = self.segmenter(seg_tensor)["out"]
        logits = nnf.interpolate(logits, size=(image.height, image.width), mode="bilinear", align_corners=False)
        mask = (logits.argmax(1)[0] == self.person_index).cpu().numpy()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        segmentation_end = time.perf_counter()
        detections = []
        for box, label, score in zip(predictions["boxes"].cpu().tolist(), predictions["labels"].cpu().tolist(), predictions["scores"].cpu().tolist()):
            category = self.categories[label]
            if score >= threshold and category in FASHION_CATEGORIES:
                x1, y1, x2, y2 = box
                clipped = [max(0, min(image.width - 1, x1)), max(0, min(image.height - 1, y1)), max(0, min(image.width - 1, x2)), max(0, min(image.height - 1, y2))]
                detections.append({"category": category, "category_zh": CHINESE_NAMES[category], "score": round(score, 6), "box": [round(v, 2) for v in clipped]})
        segmented = overlay_mask(image, mask, alpha)
        metadata = {
            "width": image.width, "height": image.height, "device": str(self.device),
            "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
            "detection_weights": self.detection_weights.name, "segmentation_weights": self.segmentation_weights.name,
            "threshold": threshold, "alpha": alpha, "detections": detections,
            "person_area_ratio": round(float(mask.mean()), 6),
            "detection_seconds": round(detection_end - start, 4),
            "segmentation_seconds": round(segmentation_end - detection_end, 4),
            "inference_seconds": round(segmentation_end - start, 4),
            "timing_note": "单次测量，包含模型预处理，不含模型加载、可视化与文件保存。非严格性能基准。",
            "limitation": "分割掩码表示人物区域，包含衣物、皮肤和头发，不区分上衣、裤子等服装类别。",
        }
        return Result(image.copy(), draw_boxes(image, detections), segmented, draw_boxes(segmented, detections), Image.fromarray(mask.astype(np.uint8) * 255), metadata)
