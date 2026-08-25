from __future__ import annotations

import argparse
import json
import platform
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DATA = PROJECT_ROOT / "configs" / "defect.yaml"
REAL_ROOT = PROJECT_ROOT / "processed" / "defect_real"
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "defect" / "defect_yolo11n_baseline_best.pt"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "defect"
DEFAULT_NAME = "defect_yolo11n_real_finetune"
REPORT_PATH = PROJECT_ROOT / "reports" / "defect_real_finetune_metrics.json"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune defect segmentation on labeled real-world screens.")
    parser.add_argument("--data", default="configs/defect.yaml")
    parser.add_argument("--model", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--project", default="runs/defect")
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15)
    return parser.parse_args()


def runtime_config(data_config: Path) -> tuple[Path, Path]:
    config = yaml.safe_load(data_config.read_text(encoding="utf-8")) or {}
    root_value = Path(str(config.get("path", ".")))
    root = root_value.resolve() if root_value.is_absolute() else (data_config.parent / root_value).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    runtime = dict(config)
    runtime["path"] = str(root)
    temporary = Path(tempfile.mkdtemp(prefix="defect_real_finetune_data_"))
    path = temporary / data_config.name
    path.write_text(yaml.safe_dump(runtime, sort_keys=False), encoding="utf-8")
    return path, root


def merge_training_data(base_root: Path) -> tuple[Path, Path]:
    merged_root = Path(tempfile.mkdtemp(prefix="defect_real_merged_"))
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            (merged_root / kind / split).mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            source = base_root / kind / split
            if source.is_dir():
                for path in source.iterdir():
                    if path.is_file():
                        shutil.copy2(path, merged_root / kind / split / path.name)
    for kind in ("images", "labels"):
        source = REAL_ROOT / kind / "train"
        if source.is_dir():
            for path in source.iterdir():
                if path.is_file():
                    shutil.copy2(path, merged_root / kind / "train" / f"real_{path.name}")
    config_path = merged_root / "defect_real_merged.yaml"
    config_path.write_text(yaml.safe_dump({"path": str(merged_root), "train": "images/train", "val": "images/val", "test": "images/test", "names": ["scratch", "oil", "stain"]}, sort_keys=False), encoding="utf-8")
    return config_path, merged_root


def metrics(metrics_object: Any) -> dict[str, Any]:
    segment = getattr(metrics_object, "seg", None)
    return {
        "precision": float(segment.mp) if segment is not None else None,
        "recall": float(segment.mr) if segment is not None else None,
        "mAP50": float(segment.map50) if segment is not None else None,
        "mAP50-95": float(segment.map) if segment is not None else None,
        "speed": getattr(metrics_object, "speed", {}) or {},
    }


def main() -> None:
    args = parse_args()
    data_config = resolve_path(args.data)
    weights = resolve_path(args.model)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    base_runtime, base_root = runtime_config(data_config)
    runtime, dataset_root = merge_training_data(base_root)
    real_test = REAL_ROOT / "images" / "test"
    real_labels = REAL_ROOT / "labels" / "test"
    if not real_test.is_dir() or not real_labels.is_dir():
        raise FileNotFoundError("Labeled real-world test data is required under processed/defect_real/{images,labels}/test")
    from ultralytics import YOLO
    import torch

    started = time.perf_counter()
    model = YOLO(str(weights))
    training = model.train(
        data=str(runtime), task="segment", epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        seed=args.seed, pretrained=False, patience=args.patience, project=str(resolve_path(args.project)),
        name=args.name, exist_ok=False, device=args.device, workers=args.workers, plots=True,
        degrees=5.0, translate=0.1, scale=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5, mosaic=0.5, copy_paste=0.0,
    )
    run_dir = Path(training.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {best}")
    output_weights = PROJECT_ROOT / "models" / "defect" / "defect_yolo11n_real_best.pt"
    if output_weights.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {output_weights}")
    output_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, output_weights)
    best_model = YOLO(str(best))
    original_test = best_model.val(data=str(runtime), split="test", imgsz=args.imgsz, project=str(run_dir), name="original_test", exist_ok=True, plots=True, verbose=True)
    real_config = {"path": str(REAL_ROOT.resolve()), "test": "images/test", "names": ["scratch", "oil", "stain"]}
    real_config_path = Path(tempfile.mkdtemp(prefix="real_test_data_")) / "real.yaml"
    real_config_path.write_text(yaml.safe_dump(real_config, sort_keys=False), encoding="utf-8")
    real_test_metrics = best_model.val(data=str(real_config_path), split="test", imgsz=args.imgsz, project=str(run_dir), name="real_world_test", exist_ok=True, plots=True, verbose=True)
    report = {
        "model": str(weights), "task": "segment", "experiment_name": args.name,
        "augmentation": {"gamma": [0.3, 1.5], "brightness_contrast": True, "clahe": True, "glare": True, "blur": True, "noise": True, "hsv": True},
        "original_defect_test": metrics(original_test), "real_world_test": metrics(real_test_metrics),
        "weights_path": str(output_weights), "run_directory": str(run_dir),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "ultralytics": __import__("ultralytics").__version__},
        "duration_seconds": round(time.perf_counter() - started, 3),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
