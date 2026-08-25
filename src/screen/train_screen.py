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
DEFAULT_DATA = PROJECT_ROOT / "configs" / "screen.yaml"
DEFAULT_MODEL = "yolo11n-seg.pt"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "screen"
DEFAULT_NAME = "screen_yolo11n_baseline"
REPORT_PATH = PROJECT_ROOT / "reports" / "screen_detection_metrics.json"
MODEL_OUTPUT = PROJECT_ROOT / "models" / "screen" / "screen_yolo11n_baseline_best.pt"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def metric(metrics: Any, name: str) -> float | None:
    value = getattr(getattr(metrics, "seg", None), name, None)
    if value is None:
        value = getattr(getattr(metrics, "box", None), name, None)
    return None if value is None else float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the screen segmentation model.")
    parser.add_argument("--data", default="configs/screen.yaml")
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--project", default="runs/screen")
    parser.add_argument("--name", default="screen_yolo11n_baseline")
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
    temp_directory = Path(tempfile.mkdtemp(prefix="screen_yolo_data_"))
    runtime_path = temp_directory / data_config.name
    runtime_path.write_text(yaml.safe_dump(runtime_config, sort_keys=False), encoding="utf-8")
    return runtime_path, dataset_root


def evaluate(model: Any, runtime_config: Path, split: str, image_size: int, run_dir: Path) -> dict[str, Any]:
    metrics = model.val(
        data=str(runtime_config),
        split=split,
        imgsz=image_size,
        project=str(run_dir),
        name=f"{split}_evaluation",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    return {
        "precision": metric(metrics, "mp"),
        "recall": metric(metrics, "mr"),
        "mAP50": metric(metrics, "map50"),
        "mAP50-95": metric(metrics, "map"),
        "speed": getattr(metrics, "speed", {}) or {},
        "evaluation_directory": str(getattr(metrics, "save_dir", "")),
    }


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
    training_results = model.train(
        data=str(runtime_config),
        task="segment",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        pretrained=True,
        patience=args.patience,
        project=str(project),
        name=args.name,
        exist_ok=False,
        plots=True,
        verbose=True,
        device=args.device,
        workers=args.workers,
    )
    run_dir = Path(training_results.save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {best_weights}")
    if MODEL_OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {MODEL_OUTPUT}")
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, MODEL_OUTPUT)

    best_model = YOLO(str(best_weights))
    validation = evaluate(best_model, runtime_config, "val", args.imgsz, run_dir)
    test = evaluate(best_model, runtime_config, "test", args.imgsz, run_dir)
    report = {
        "model": args.model,
        "experiment_name": args.name,
        "task": "segment",
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "patience": args.patience,
        "device": args.device,
        "dataset_root": str(dataset_root),
        "validation_metrics": validation,
        "test_metrics": test,
        "precision": validation["precision"],
        "recall": validation["recall"],
        "mAP50": validation["mAP50"],
        "mAP50-95": validation["mAP50-95"],
        "training_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": __import__("ultralytics").__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "weights_path": str(MODEL_OUTPUT),
        "run_directory": str(run_dir),
        "training_duration_seconds": round(time.perf_counter() - started, 3),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
