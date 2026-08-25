from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "phone" / "phone_yolo11n_cpu_baseline_best.pt"


def crop_phone(image_path: Path, weights_path: Path, output_path: Path, padding: float, min_confidence: float, size: int, device: str) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    result = model.predict(source=str(image_path), imgsz=size, device=device, conf=0.001, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        status = "PHONE_NOT_FOUND"
        report = {"status": status, "input": str(image_path), "output": None}
        return report

    confidences = result.boxes.conf.cpu().tolist()
    best_index = max(range(len(confidences)), key=confidences.__getitem__)
    confidence = float(confidences[best_index])
    if confidence < min_confidence:
        return {"status": "PHONE_TOO_FAR", "confidence": confidence, "input": str(image_path), "output": None}

    with Image.open(image_path).convert("RGB") as image:
        width, height = image.size
        x1, y1, x2, y2 = result.boxes.xyxy[best_index].cpu().tolist()
        box_width = x2 - x1
        box_height = y2 - y1
        x1 = max(0, int(x1 - box_width * padding))
        y1 = max(0, int(y1 - box_height * padding))
        x2 = min(width, int(x2 + box_width * padding))
        y2 = min(height, int(y2 + box_height * padding))
        if x2 <= x1 or y2 <= y1:
            return {"status": "PHONE_NOT_FOUND", "confidence": confidence, "input": str(image_path), "output": None}
        crop = image.crop((x1, y1, x2, y2))
        fitted = ImageOps.contain(crop, (size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), "black")
        canvas.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)

    return {
        "status": "PHONE_DETECTED",
        "confidence": confidence,
        "input": str(image_path),
        "output": str(output_path),
        "box_with_padding": [x1, y1, x2, y2],
        "resize": [size, size],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and crop a phone from one input image.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--padding", type=float, default=0.05)
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    for path_name, path in (("input", args.input), ("weights", args.weights)):
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path_name == "input":
            args.input = path
        else:
            args.weights = path
    if not args.input.exists():
        raise FileNotFoundError(f"Input image not found: {args.input}")
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    report = crop_phone(args.input, args.weights, args.output, args.padding, args.min_confidence, args.size, args.device)
    report_path = args.report or args.output.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
