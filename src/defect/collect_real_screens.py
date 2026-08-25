from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phone.phone_crop_pipeline import crop_phone
from screen.perspective_normalize import extract_detection, orientation_and_transform

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_INPUT = PROJECT_ROOT / "datasets" / "real_world"
DEFAULT_OUTPUT = PROJECT_ROOT / "processed" / "defect_real" / "images"
CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def list_images(directory: Path) -> list[Path]:
    return sorted([path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS], key=lambda path: path.name.lower()) if directory.is_dir() else []


def load_weights() -> tuple[Path, Path]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
    phone = resolve_path(config.get("phone", "models/phone/phone_yolo11n_baseline_best.pt"))
    screen = resolve_path(config.get("screen", "models/screen/screen_yolo11n_baseline_best.pt"))
    return phone, screen


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect perspective-normalized real-world screen images.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    phone_weights, screen_weights = load_weights()
    for path in (phone_weights, screen_weights):
        if not path.exists():
            raise FileNotFoundError(f"Weights not found: {path}")

    from ultralytics import YOLO

    phone_model = YOLO(str(phone_weights))
    screen_model = YOLO(str(screen_weights))
    inputs = list_images(input_dir)
    results: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="real_screen_collect_") as temporary:
        for image_path in inputs:
            result: dict[str, Any] = {"input": str(image_path), "status": "SCREEN_NOT_FOUND", "output": None}
            image = cv2.imread(str(image_path))
            if image is None:
                result["reason"] = "unreadable image"
                print(f"{image_path.name}: SCREEN_NOT_FOUND (unreadable image)")
                results.append(result)
                continue
            crop_path = Path(temporary) / f"{image_path.stem}.jpg"
            crop_report = crop_phone(image_path, phone_weights, crop_path, 0.05, 0.25, 640, "cpu")
            result["phone_status"] = crop_report.get("status")
            result["phone_confidence"] = crop_report.get("confidence")
            if crop_report.get("status") != "PHONE_DETECTED":
                result["reason"] = "phone not detected with confidence >= 0.25"
                print(f"{image_path.name}: {result['status']} ({result['reason']})")
                results.append(result)
                continue
            crop = cv2.imread(str(crop_path))
            prediction = screen_model.predict(source=crop, imgsz=640, conf=0.001, device="cpu", verbose=False)[0]
            corners, confidence, screen_status = extract_detection(prediction, 0.5)
            result["screen_confidence"] = confidence
            result["screen_status"] = screen_status
            if corners is None or screen_status != "SCREEN_NORMALIZED":
                result["status"] = screen_status
                result["reason"] = "screen polygon missing, degenerate, or below confidence threshold"
                print(f"{image_path.name}: {result['status']} ({result['reason']})")
                results.append(result)
                continue
            orientation, matrix, _inverse, _ordered = orientation_and_transform(corners)
            normalized = cv2.warpPerspective(crop, matrix, (640, 360))
            output_path = output_dir / f"{image_path.stem}.jpg"
            cv2.imwrite(str(output_path), normalized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            result.update({"status": "SCREEN_NORMALIZED", "orientation": orientation, "output": str(output_path)})
            print(f"{image_path.name}: SCREEN_NORMALIZED")
            results.append(result)
    report_path = output_dir.parent / "collect_report.json"
    counts = {status: sum(item["status"] == status for item in results) for status in sorted({item["status"] for item in results})}
    report = {"input_dir": str(input_dir), "output_dir": str(output_dir), "counts": counts, "results": results}
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Final counts: {counts}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
