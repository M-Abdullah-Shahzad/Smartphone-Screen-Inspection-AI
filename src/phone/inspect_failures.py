from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "processed" / "phone_detection"
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "phone" / "phone_yolo11n_cpu_baseline" / "weights" / "best.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "phone" / "phone_yolo11n_cpu_baseline" / "failure_analysis"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_ground_truth(label_path: Path) -> list[tuple[float, float, float, float]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5 or fields[0] != "0":
            continue
        values = tuple(float(value) for value in fields[1:])
        center_x, center_y, width, height = values
        boxes.append((center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2))
    return boxes


def iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect phone detector false positives, false negatives, and low-confidence detections.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--low-confidence", type=float, default=0.50)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    args = parser.parse_args()

    if not args.weights.is_absolute():
        args.weights = PROJECT_ROOT / args.weights
    if not args.dataset.is_absolute():
        args.dataset = PROJECT_ROOT / args.dataset
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    image_dir = args.dataset / "images" / args.split
    label_dir = args.dataset / "labels" / args.split
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Missing split directories under {args.dataset}: {args.split}")

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    args.output.mkdir(parents=True, exist_ok=True)
    findings: dict[str, list[dict[str, Any]]] = {"false_positives": [], "false_negatives": [], "low_confidence": []}
    images = sorted((path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS), key=lambda path: path.name.lower())
    for image_path in images:
        ground_truth = read_ground_truth(label_dir / f"{image_path.stem}.txt")
        result = model.predict(source=str(image_path), imgsz=args.imgsz, conf=0.001, device="cpu", verbose=False)[0]
        height, width = result.orig_shape
        predictions = []
        for box, confidence in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
            predicted = (box[0] / width, box[1] / height, box[2] / width, box[3] / height)
            predictions.append({"box": predicted, "confidence": float(confidence)})
        matched = any(iou(prediction["box"], truth) >= args.iou_threshold for prediction in predictions for truth in ground_truth)
        if not ground_truth and predictions:
            findings["false_positives"].append({"image": image_path.name, "predictions": predictions})
        elif ground_truth and not matched:
            findings["false_negatives"].append({"image": image_path.name, "ground_truth_count": len(ground_truth), "predictions": predictions})
        low_predictions = [prediction for prediction in predictions if prediction["confidence"] < args.low_confidence]
        if low_predictions:
            findings["low_confidence"].append({"image": image_path.name, "predictions": low_predictions})
        if not (not ground_truth and predictions) and not (ground_truth and not matched) and not low_predictions:
            continue
        plotted = result.plot()
        Image.fromarray(plotted[:, :, ::-1]).save(args.output / image_path.name)

    report = {
        "weights": str(args.weights),
        "split": args.split,
        "low_confidence_threshold": args.low_confidence,
        "iou_threshold": args.iou_threshold,
        "counts": {key: len(value) for key, value in findings.items()},
        "findings": findings,
        "output_directory": str(args.output),
    }
    (args.output / "failure_analysis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
