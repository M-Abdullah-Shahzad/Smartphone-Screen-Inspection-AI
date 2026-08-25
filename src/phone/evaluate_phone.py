from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "configs" / "phone.yaml"
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "phone" / "phone_yolo11n_cpu_baseline" / "weights" / "best.pt"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "phone_detection_metrics.json"


def metric(metrics: Any, name: str) -> float | None:
    value = getattr(getattr(metrics, "box", None), name, None)
    return None if value is None else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained phone detector on a held-out split.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs" / "phone")
    parser.add_argument("--name", default="phone_detection_evaluation")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    if not args.weights.is_absolute():
        args.weights = PROJECT_ROOT / args.weights
    if not args.data.is_absolute():
        args.data = PROJECT_ROOT / args.data
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset config not found: {args.data}")

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        project=str(args.project),
        name=args.name,
        exist_ok=False,
        plots=True,
        verbose=True,
    )
    report = {
        "weights": str(args.weights),
        "data": str(args.data),
        "split": args.split,
        "imgsz": args.imgsz,
        "precision": metric(metrics, "mp"),
        "recall": metric(metrics, "mr"),
        "mAP50": metric(metrics, "map50"),
        "mAP50-95": metric(metrics, "map"),
        "speed": getattr(metrics, "speed", {}) or {},
        "evaluation_directory": str(getattr(metrics, "save_dir", "")),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
