from __future__ import annotations

import json
import argparse
import csv
import platform
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CONFIG = PROJECT_ROOT / "configs" / "phone.yaml"
DATASET_ROOT = PROJECT_ROOT / "processed" / "phone_detection"
RUNS_ROOT = PROJECT_ROOT / "runs" / "phone"
REPORT_PATH = PROJECT_ROOT / "reports" / "phone_baseline_report.json"
MODEL_OUTPUT_ROOT = PROJECT_ROOT / "models" / "phone"
EXPERIMENT_NAME = "phone_yolo11n_cpu_baseline"
MODEL_NAME = "yolo11n.pt"
EPOCHS = 45
IMAGE_SIZE = 640
SEED = 42
PATIENCE = 15


def count_images(split: str) -> int:
    image_dir = DATASET_ROOT / "images" / split
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sum(1 for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions)


def metric(metrics: Any, name: str) -> float | None:
    value = getattr(getattr(metrics, "box", None), name, None)
    if value is None:
        return None
    return float(value)


def best_epoch(results_csv: Path) -> int | None:
    if not results_csv.exists():
        return None
    with results_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return None
    map_key = next((key for key in rows[0] if "metrics/mAP50-95" in key), None)
    if map_key is None:
        return None
    return max(rows, key=lambda row: float(row[map_key]))["epoch"]


def training_observation(run_dir: Path, validation_metrics: Any) -> str:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return "Training results CSV unavailable; no train-vs-validation classification was made."
    with results_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return "Training results CSV empty; no train-vs-validation classification was made."
    train_map_key = next((key for key in rows[0] if "metrics/mAP50-95" in key), None)
    val_map_key = next((key for key in rows[0] if "metrics/mAP50(B)" in key), None)
    if train_map_key is None or val_map_key is None:
        return "Train/validation metric columns unavailable; classification requires manual review."
    train_map = float(rows[-1][train_map_key])
    val_map = float(rows[-1][val_map_key])
    gap = train_map - val_map
    if val_map < 0.25:
        return f"Validation mAP50-95 is {val_map:.4f} while train mAP50-95 is {train_map:.4f}; this indicates underfitting or weak learning."
    if gap > 0.20:
        return f"Train/validation mAP50-95 gap is {gap:.4f} ({train_map:.4f} vs {val_map:.4f}); possible overfitting."
    return f"Train/validation mAP50-95 gap is {gap:.4f} ({train_map:.4f} vs {val_map:.4f}); no large overfitting gap observed."


def save_predictions(model: Any, split: str, run_dir: Path, dataset_root: Path, image_size: int) -> Path:
    prediction_dir = run_dir / f"predictions_{split}"
    model.predict(
        source=str(dataset_root / "images" / split),
        imgsz=image_size,
        project=str(run_dir),
        name=f"predictions_{split}",
        exist_ok=True,
        save=True,
        save_txt=True,
        save_conf=True,
        verbose=False,
    )
    return prediction_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the phone detector.")
    parser.add_argument("--data", default="configs/phone.yaml", help="Ultralytics dataset YAML path.")
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained YOLO model or checkpoint path.")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", default="auto", help="CPU batch size: auto selects 4 or 2 based on available RAM.")
    parser.add_argument("--project", default="runs/phone")
    parser.add_argument("--name", default="phone_yolo11n_baseline")
    parser.add_argument("--device", default="cpu", help="Training device; this experiment requires CPU.")
    parser.add_argument("--workers", type=int, default=0, help="Data-loader worker processes.")
    parser.add_argument("--startup-check", action="store_true", help="Validate paths and dataset counts without training.")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_batch(value: str) -> int | float:
    if value == "auto":
        return -1
    try:
        return int(value)
    except ValueError:
        return float(value)


def choose_cpu_batch(requested: str) -> int:
    if requested != "auto":
        return int(requested)
    try:
        import psutil

        return 4 if psutil.virtual_memory().available >= 4 * 1024**3 else 2
    except ImportError:
        return 4


def startup_check(data_config: Path, dataset_root: Path) -> dict[str, Any]:
    errors: list[str] = []
    if not data_config.exists():
        errors.append(f"missing dataset config: {data_config}")
    expected_counts = {"train": 70, "val": 15, "test": 15}
    counts: dict[str, int] = {}
    for split, expected in expected_counts.items():
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        image_count = count_images_from_root(dataset_root, split)
        label_count = len(list(label_dir.glob("*.txt"))) if label_dir.exists() else 0
        counts[split] = image_count
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f"missing split directories: {split}")
        if image_count != expected or label_count != expected:
            errors.append(f"unexpected {split} counts: images={image_count}, labels={label_count}")
        if image_count != label_count:
            errors.append(f"image/label mismatch: {split}")
    return {"status": "PASS" if not errors else "FAIL", "counts": counts, "errors": errors}


def count_images_from_root(dataset_root: Path, split: str) -> int:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_dir = dataset_root / "images" / split
    return sum(1 for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions) if image_dir.exists() else 0


def create_runtime_data_config(data_config: Path) -> tuple[Path, Path]:
    config = yaml.safe_load(data_config.read_text(encoding="utf-8")) or {}
    configured_root = Path(str(config.get("path", ".")))
    dataset_root = (data_config.parent / configured_root).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Resolved dataset root not found: {dataset_root}")
    for split in ("train", "val", "test"):
        split_path = (dataset_root / str(config[split])).resolve()
        if not split_path.is_dir():
            raise FileNotFoundError(f"Resolved {split} path not found: {split_path}")

    runtime_config = dict(config)
    runtime_config["path"] = str(dataset_root)
    temp_directory = Path(tempfile.mkdtemp(prefix="phone_yolo_data_"))
    runtime_path = temp_directory / data_config.name
    runtime_path.write_text(yaml.safe_dump(runtime_config, sort_keys=False), encoding="utf-8")
    return runtime_path, dataset_root


def verify_ultralytics_resolution(runtime_config: Path, expected_root: Path) -> dict[str, Any]:
    from ultralytics.data.utils import check_det_dataset

    resolved = check_det_dataset(str(runtime_config))
    paths = {split: str(Path(resolved[split]).resolve()) for split in ("train", "val", "test")}
    expected = {split: str((expected_root / split).resolve()) for split in ("train", "val", "test")}
    return {
        "resolved_paths": paths,
        "expected_paths": expected,
        "valid": paths == expected,
    }


def main() -> None:
    args = parse_args()
    data_config = resolve_path(args.data)
    project_root = resolve_path(args.project)
    if not data_config.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_config}")
    if args.device != "cpu":
        raise ValueError("This CPU-only experiment requires --device cpu.")
    runtime_config, dataset_root = create_runtime_data_config(data_config)

    check = startup_check(data_config, dataset_root)
    resolution = verify_ultralytics_resolution(runtime_config, dataset_root / "images")
    check["ultralytics_resolution"] = resolution
    if not resolution["valid"]:
        check["status"] = "FAIL"
        check["errors"].append("Ultralytics resolved one or more dataset paths incorrectly")
    print(json.dumps({"startup_check": check}, indent=2))
    if args.startup_check:
        if check["status"] != "PASS":
            raise RuntimeError(json.dumps(check, indent=2))
        return
    if check["status"] != "PASS":
        raise RuntimeError(json.dumps(check, indent=2))

    from ultralytics import YOLO
    import torch

    started = time.perf_counter()
    model = YOLO(args.model)
    training_results = model.train(
        data=str(runtime_config),
        epochs=args.epochs,
        imgsz=args.imgsz,
        seed=SEED,
        pretrained=True,
        batch=choose_cpu_batch(args.batch),
        patience=PATIENCE,
        project=str(project_root),
        name=args.name,
        exist_ok=False,
        plots=True,
        verbose=True,
        device="cpu",
        workers=args.workers,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.0,
    )
    run_dir = Path(training_results.save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {best_weights}")
    if not last_weights.exists():
        raise FileNotFoundError(f"Last checkpoint not found: {last_weights}")
    MODEL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    saved_best = MODEL_OUTPUT_ROOT / f"{args.name}_best.pt"
    saved_last = MODEL_OUTPUT_ROOT / f"{args.name}_last.pt"
    if saved_best.exists() or saved_last.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint(s): {saved_best}, {saved_last}")
    shutil.copy2(best_weights, saved_best)
    shutil.copy2(last_weights, saved_last)

    best_model = YOLO(str(best_weights))
    validation_metrics = best_model.val(
        data=str(runtime_config),
        split="val",
        imgsz=args.imgsz,
        project=str(run_dir),
        name="validation",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    test_metrics = best_model.val(
        data=str(runtime_config),
        split="test",
        imgsz=args.imgsz,
        project=str(run_dir),
        name="test",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    validation_predictions = save_predictions(best_model, "val", run_dir, dataset_root, args.imgsz)
    test_predictions = save_predictions(best_model, "test", run_dir, dataset_root, args.imgsz)
    validation_speed = getattr(validation_metrics, "speed", {}) or {}
    test_speed = getattr(test_metrics, "speed", {}) or {}

    report = {
        "model": args.model,
        "experiment_name": args.name,
        "epochs": args.epochs,
        "image_size": args.imgsz,
        "imgsz": args.imgsz,
        "seed": SEED,
        "pretrained": True,
        "batch": choose_cpu_batch(args.batch),
        "patience": PATIENCE,
        "device": "cpu",
        "train_count": count_images("train"),
        "val_count": count_images("val"),
        "test_count": count_images("test"),
        "precision": metric(validation_metrics, "mp"),
        "recall": metric(validation_metrics, "mr"),
        "mAP50": metric(validation_metrics, "map50"),
        "mAP50-95": metric(validation_metrics, "map"),
        "test_precision": metric(test_metrics, "mp"),
        "test_recall": metric(test_metrics, "mr"),
        "test_mAP50": metric(test_metrics, "map50"),
        "test_mAP50-95": metric(test_metrics, "map"),
        "validation_metrics": {
            "precision": metric(validation_metrics, "mp"),
            "recall": metric(validation_metrics, "mr"),
            "mAP50": metric(validation_metrics, "map50"),
            "mAP50-95": metric(validation_metrics, "map"),
            "speed": validation_speed,
        },
        "test_metrics": {
            "precision": metric(test_metrics, "mp"),
            "recall": metric(test_metrics, "mr"),
            "mAP50": metric(test_metrics, "map50"),
            "mAP50-95": metric(test_metrics, "map"),
            "speed": test_speed,
        },
        "training_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": __import__("ultralytics").__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 3) if torch.cuda.is_available() else None,
        },
        "best_epoch": best_epoch(run_dir / "results.csv"),
        "best_weights_path": str(saved_best),
        "last_weights_path": str(saved_last),
        "training_duration_seconds": round(time.perf_counter() - started, 3),
        "inference_speed": {"validation": validation_speed, "test": test_speed},
        "baseline_classification": "REVIEW_REQUIRED",
        "important_observations": [training_observation(run_dir, validation_metrics)],
        "warnings_errors": [],
        "run_directory": str(run_dir),
        "artifacts": {
            "training_curves": str(run_dir / "results.png"),
            "validation_confusion_matrix": str(run_dir / "validation" / "confusion_matrix.png"),
            "test_confusion_matrix": str(run_dir / "test" / "confusion_matrix.png"),
            "validation_predictions": str(validation_predictions),
            "test_predictions": str(test_predictions),
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
