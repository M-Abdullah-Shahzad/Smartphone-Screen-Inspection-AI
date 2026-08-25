from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHONE_ROOT = PROJECT_ROOT / "processed" / "phone_detection"
CONFIG_PATH = PROJECT_ROOT / "configs" / "phone.yaml"
REPORT_PATH = PROJECT_ROOT / "reports" / "phone_training_readiness.json"
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def validate_split(split: str) -> dict[str, Any]:
    image_dir = PHONE_ROOT / "images" / split
    label_dir = PHONE_ROOT / "labels" / split
    images = sorted(
        [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.name.lower(),
    ) if image_dir.exists() else []
    labels = sorted(label_dir.glob("*.txt"), key=lambda path: path.name.lower()) if label_dir.exists() else []
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    missing_labels = sorted(image_stems - label_stems)
    orphan_labels = sorted(label_stems - image_stems)
    invalid_labels: list[dict[str, Any]] = []
    empty_labels: list[str] = []
    class_ids: Counter[str] = Counter()
    invalid_class_ids: list[dict[str, Any]] = []
    zero_area_boxes: list[dict[str, Any]] = []
    out_of_range_boxes: list[dict[str, Any]] = []
    invalid_coordinate_rows: list[dict[str, Any]] = []

    for label_path in labels:
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            empty_labels.append(label_path.name)
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            fields = line.split()
            if len(fields) != 5:
                invalid_labels.append({"file": label_path.name, "line": line_number, "reason": "expected_5_fields", "value": line})
                continue
            class_id = fields[0]
            class_ids[class_id] += 1
            try:
                values = [float(value) for value in fields[1:]]
            except ValueError:
                invalid_coordinate_rows.append({"file": label_path.name, "line": line_number, "reason": "non_numeric_coordinate", "value": line})
                continue
            center_x, center_y, box_width, box_height = values
            if class_id != "0":
                invalid_class_ids.append({"file": label_path.name, "line": line_number, "class_id": class_id})
            if box_width <= 0 or box_height <= 0:
                zero_area_boxes.append({"file": label_path.name, "line": line_number, "values": values})
            if any(value < 0 or value > 1 for value in values):
                out_of_range_boxes.append({"file": label_path.name, "line": line_number, "values": values})

    unreadable_images: list[str] = []
    for image_path in images:
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as error:
            unreadable_images.append(f"{image_path.name}: {error}")

    split_valid = (
        image_dir.exists()
        and label_dir.exists()
        and not missing_labels
        and not orphan_labels
        and not invalid_labels
        and not empty_labels
        and not invalid_class_ids
        and not zero_area_boxes
        and not out_of_range_boxes
        and not invalid_coordinate_rows
        and not unreadable_images
    )
    return {
        "image_count": len(images),
        "label_count": len(labels),
        "missing_labels": missing_labels,
        "orphan_labels": orphan_labels,
        "invalid_labels": invalid_labels,
        "empty_labels": empty_labels,
        "class_ids": dict(sorted(class_ids.items())),
        "invalid_class_ids": invalid_class_ids,
        "zero_area_boxes": zero_area_boxes,
        "boxes_outside_normalized_range": out_of_range_boxes,
        "invalid_coordinate_rows": invalid_coordinate_rows,
        "unreadable_images": unreadable_images,
        "readiness": "PASS" if split_valid else "FAIL",
    }


def validate_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {"exists": False, "valid": False, "errors": ["config_missing"]}
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    errors: list[str] = []
    dataset_root = (CONFIG_PATH.parent / str(config.get("path", "."))).resolve()
    resolved_paths: dict[str, str] = {}
    for split in SPLITS:
        configured_path = config.get(split)
        if configured_path is None:
            errors.append(f"missing_{split}_path")
            continue
        resolved = (dataset_root / str(configured_path)).resolve()
        resolved_paths[split] = str(resolved)
        if not resolved.is_dir():
            errors.append(f"invalid_{split}_path")
    if config.get("nc") != 1:
        errors.append("invalid_class_count")
    if config.get("names") != ["phone"]:
        errors.append("invalid_class_names")
    return {
        "exists": True,
        "valid": not errors,
        "config": config,
        "resolved_paths": resolved_paths,
        "errors": errors,
    }


def main() -> None:
    split_reports = {split: validate_split(split) for split in SPLITS}
    config_report = validate_config()
    required_paths = {
        "images_train": (PHONE_ROOT / "images" / "train").exists(),
        "images_val": (PHONE_ROOT / "images" / "val").exists(),
        "images_test": (PHONE_ROOT / "images" / "test").exists(),
        "labels_train": (PHONE_ROOT / "labels" / "train").exists(),
        "labels_val": (PHONE_ROOT / "labels" / "val").exists(),
        "labels_test": (PHONE_ROOT / "labels" / "test").exists(),
        "phone_yaml": CONFIG_PATH.exists(),
    }
    all_pass = all(required_paths.values()) and config_report["valid"] and all(item["readiness"] == "PASS" for item in split_reports.values())
    report = {
        "dataset": "phone_detection",
        "processed_root": str(PHONE_ROOT),
        "config_path": str(CONFIG_PATH),
        "required_paths": required_paths,
        "config_validation": config_report,
        "splits": split_reports,
        "all_class_ids": sorted({class_id for item in split_reports.values() for class_id in item["class_ids"]}),
        "all_class_ids_are_zero": all(
            all(class_id == "0" for class_id in item["class_ids"])
            and not item["invalid_class_ids"]
            for item in split_reports.values()
        ),
        "training_readiness": "PASS" if all_pass else "FAIL",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "image_count_per_split": {split: item["image_count"] for split, item in split_reports.items()},
        "label_count_per_split": {split: item["label_count"] for split, item in split_reports.items()},
        "invalid_labels": sum(len(item["invalid_labels"]) + len(item["invalid_coordinate_rows"]) for item in split_reports.values()),
        "empty_labels": sum(len(item["empty_labels"]) for item in split_reports.values()),
        "class_ids": report["all_class_ids"],
        "training_readiness": report["training_readiness"],
    }, indent=2))


if __name__ == "__main__":
    main()
