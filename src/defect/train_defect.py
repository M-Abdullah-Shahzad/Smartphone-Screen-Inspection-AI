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
DEFAULT_DATA = PROJECT_ROOT / "configs" / "defect.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "defect"
DEFAULT_NAME = "defect_yolo11n_baseline"
REPORT_PATH = PROJECT_ROOT / "reports" / "defect_segmentation_metrics.json"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def metric(metrics: Any, name: str) -> float | None:
    value = getattr(getattr(metrics, "seg", None), name, None)
    if value is None:
        value = getattr(getattr(metrics, "box", None), name, None)
    return None if value is None else float(value)


def per_class_metric(metrics: Any, name: str) -> list[float] | None:
    value = getattr(getattr(metrics, "seg", None), name, None)
    if value is None:
        value = getattr(getattr(metrics, "box", None), name, None)
    if value is None:
        return None
    return [float(item) for item in value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the defect segmentation model.")
    parser.add_argument("--data", default="configs/defect.yaml")
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--project", default="runs/defect")
    parser.add_argument("--name", default="defect_yolo11n_baseline")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15)
    return parser.parse_args()


def create_runtime_data_config(data_config: Path) -> tuple[Path, Path]:
    config = yaml.safe_load(data_config.read_text(encoding="utf-8")) or {}
    configured_root = Path(str(config.get("path", ".")))
    dataset_root = configured_root.resolve() if configured_root.is_absolute() else (data_config.parent / configured_root).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Resolved dataset root not found: {dataset_root}")
    for split in ("train", "val", "test"):
        split_path = dataset_root / str(config[split])
        if not split_path.is_dir():
            raise FileNotFoundError(f"Resolved {split} path not found: {split_path}")
    runtime_config = dict(config)
    runtime_config["path"] = str(dataset_root)
    temp_directory = Path(tempfile.mkdtemp(prefix="defect_yolo_data_"))
    runtime_path = temp_directory / data_config.name
    runtime_path.write_text(yaml.safe_dump(runtime_config, sort_keys=False), encoding="utf-8")
    return runtime_path, dataset_root


def evaluate(model: Any, runtime_config: Path, split: str, image_size: int, run_dir: Path) -> dict[str, Any]:
    metrics = model.val(data=str(runtime_config), split=split, imgsz=image_size, project=str(run_dir), name=f"{split}_evaluation", exist_ok=True, plots=True, verbose=True)
    names = getattr(metrics, "names", None) or {0: "scratch", 1: "oil", 2: "stain"}
    class_names = [names[index] if isinstance(names, list) else names.get(index, str(index)) for index in range(len(names))]
    per_class = {
        class_name: {
            "precision": (per_class_metric(metrics, "p") or [])[index] if index < len(per_class_metric(metrics, "p") or []) else None,
            "recall": (per_class_metric(metrics, "r") or [])[index] if index < len(per_class_metric(metrics, "r") or []) else None,
            "mAP50": (per_class_metric(metrics, "ap50") or [])[index] if index < len(per_class_metric(metrics, "ap50") or []) else None,
            "mAP50-95": (per_class_metric(metrics, "ap") or [])[index] if index < len(per_class_metric(metrics, "ap") or []) else None,
        }
        for index, class_name in enumerate(class_names)
    }
    return {"precision": metric(metrics, "mp"), "recall": metric(metrics, "mr"), "mAP50": metric(metrics, "map50"), "mAP50-95": metric(metrics, "map"), "per_class": per_class, "speed": getattr(metrics, "speed", {}) or {}, "evaluation_directory": str(getattr(metrics, "save_dir", ""))}


def main() -> None:
    args = parse_args()
    data_config = resolve_path(args.data)
    project = resolve_path(args.project)
    if not data_config.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_config}")
    if args.device != "cpu":
        raise ValueError("This experiment requires --device cpu.")
    runtime_config, dataset_root = create_runtime_data_config(data_config)
    project.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    import torch

    started = time.perf_counter()
    model = YOLO(args.model)
    training_results = model.train(data=str(runtime_config), task="segment", epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, seed=args.seed, pretrained=True, patience=args.patience, project=str(project), name=args.name, exist_ok=False, plots=True, verbose=True, device=args.device, workers=args.workers)
    run_dir = Path(training_results.save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {best_weights}")
    model_output = PROJECT_ROOT / "models" / "defect" / f"{args.name}_best.pt"
    if model_output.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {model_output}")
    model_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, model_output)
    best_model = YOLO(str(best_weights))
    validation = evaluate(best_model, runtime_config, "val", args.imgsz, run_dir)
    test = evaluate(best_model, runtime_config, "test", args.imgsz, run_dir)
    report = {"model": args.model, "experiment_name": args.name, "task": "segment", "dataset_root": str(dataset_root), "validation_metrics": validation, "test_metrics": test, "training_environment": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "ultralytics": __import__("ultralytics").__version__, "cuda_available": torch.cuda.is_available()}, "weights_path": str(model_output), "run_directory": str(run_dir), "training_duration_seconds": round(time.perf_counter() - started, 3)}
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
