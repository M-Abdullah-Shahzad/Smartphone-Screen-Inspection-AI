from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

RAW_DEFECT_DIRS = ["good", "oil", "scratch", "stain"]
MASK_DEFECT_DIRS = ["ground_truth_1", "ground_truth_2"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name.lower(),
    )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def phash_digest(path: Path) -> str:
    with Image.open(path).convert("L") as img:
        img = img.resize((8, 8), Image.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.float32)
        mean = float(arr.mean())
        bits = "".join("1" if px >= mean else "0" for px in arr.flatten())
        return hashlib.md5(bits.encode("utf-8")).hexdigest()


def hamming_distance(a: str, b: str) -> int:
    if len(a) != len(b):
        return 1000
    return sum(1 for x, y in zip(a, b) if x != y)


def build_duplicate_report(paths: list[Path], threshold: int = 4) -> dict[str, Any]:
    sha_map: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        sha_map[file_sha256(path)].append(path.name)

    exact_duplicate_groups = [
        {"hash": digest, "files": sorted(names)}
        for digest, names in sorted(sha_map.items())
        if len(names) > 1
    ]

    phash_items = []
    for path in paths:
        if is_valid_image(path):
            phash_items.append((path.name, phash_digest(path)))

    near_pairs: list[dict[str, Any]] = []
    for i in range(len(phash_items)):
        for j in range(i + 1, len(phash_items)):
            a_name, a_hash = phash_items[i]
            b_name, b_hash = phash_items[j]
            dist = hamming_distance(a_hash, b_hash)
            if dist <= threshold:
                near_pairs.append({"image_a": a_name, "image_b": b_name, "hamming_distance": dist})

    return {
        "exact_duplicate_groups": exact_duplicate_groups,
        "exact_duplicate_count": sum(len(group["files"]) - 1 for group in exact_duplicate_groups),
        "near_duplicate_pairs": near_pairs,
        "near_duplicate_count": len(near_pairs),
    }


def validate_phone_dataset(phone_root: Path) -> dict[str, Any]:
    image_dir = phone_root / "Mobile_image" / "Mobile_image"
    annotation_dir = phone_root / "Annotations" / "Annotations"
    image_paths = list_image_files(image_dir)
    xml_paths = sorted(annotation_dir.glob("*.xml"), key=lambda p: p.name.lower())

    image_stems = {p.stem for p in image_paths}
    xml_stems = {p.stem for p in xml_paths}

    missing_annotations = sorted(image_stems - xml_stems)
    missing_images = sorted(xml_stems - image_stems)

    invalid_boxes: list[dict[str, Any]] = []
    boxes_outside_image: list[dict[str, Any]] = []
    obj_counts: Counter[int] = Counter()
    class_counts: Counter[str] = Counter()

    for xml_path in xml_paths:
        try:
            tree = ET.parse(xml_path)
            xml_root = tree.getroot()
        except Exception:
            invalid_boxes.append({"annotation": xml_path.name, "reason": "xml_parse_error"})
            continue

        filename = xml_root.findtext("filename")
        image_path = image_dir / filename if filename else None
        if filename is None or not image_path or not image_path.exists():
            missing_images.append(xml_path.stem)
            continue

        try:
            with Image.open(image_path) as img:
                width, height = img.width, img.height
        except Exception:
            invalid_boxes.append({"annotation": xml_path.name, "reason": "image_not_readable", "image": filename})
            continue

        objects = xml_root.findall("object")
        obj_counts[len(objects)] += 1
        for obj in objects:
            name = (obj.findtext("name") or "unknown").strip()
            class_counts[name] += 1
            bndbox = obj.find("bndbox")
            if bndbox is None:
                invalid_boxes.append({"annotation": xml_path.name, "reason": "missing_bndbox", "object_name": name})
                continue
            values: dict[str, float | None] = {}
            for key in ("xmin", "ymin", "xmax", "ymax"):
                raw = bndbox.findtext(key)
                if raw is None:
                    values[key] = None
                    invalid_boxes.append({"annotation": xml_path.name, "reason": f"missing_{key}", "object_name": name})
                    continue
                try:
                    values[key] = float(raw)
                except ValueError:
                    values[key] = None
                    invalid_boxes.append({"annotation": xml_path.name, "reason": f"invalid_{key}", "object_name": name, "value": raw})
            if any(v is None for v in values.values()):
                continue
            xmin = float(values["xmin"]) ; ymin = float(values["ymin"])
            xmax = float(values["xmax"]) ; ymax = float(values["ymax"])
            if xmin < 0 or ymin < 0 or xmax > width or ymax > height or xmax <= xmin or ymax <= ymin:
                boxes_outside_image.append({
                    "annotation": xml_path.name,
                    "image": filename,
                    "object_name": name,
                    "box": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
                    "image_size": {"width": width, "height": height},
                })

    phone_duplicates = build_duplicate_report(image_paths)

    return {
        "dataset_root": str(phone_root),
        "image_count": len(image_paths),
        "annotation_count": len(xml_paths),
        "image_extensions": dict(sorted(Counter(p.suffix.lower() for p in image_paths).items())),
        "class_distribution": dict(sorted(class_counts.items())),
        "object_count_distribution": dict(sorted((int(k), v) for k, v in obj_counts.items())),
        "annotation_mismatches": {
            "missing_annotations": missing_annotations,
            "missing_images": missing_images,
            "invalid_boxes": invalid_boxes,
            "boxes_outside_image": boxes_outside_image,
        },
        "image_dimensions": {
            "unique_dimensions": {
                f"{width}x{height}": count
                for (width, height), count in sorted(
                    Counter((Image.open(p).width, Image.open(p).height) for p in image_paths).items(),
                    key=lambda kv: (-kv[1], kv[0]),
                )[:10]
            },
            "min_width": min(Image.open(p).width for p in image_paths),
            "max_width": max(Image.open(p).width for p in image_paths),
            "min_height": min(Image.open(p).height for p in image_paths),
            "max_height": max(Image.open(p).height for p in image_paths),
        },
        "duplicate_summary": {
            "exact_duplicate_count": phone_duplicates["exact_duplicate_count"],
            "near_duplicate_count": phone_duplicates["near_duplicate_count"],
            "exact_duplicate_groups": phone_duplicates["exact_duplicate_groups"],
            "near_duplicate_pairs": phone_duplicates["near_duplicate_pairs"],
        },
    }


def validate_defect_dataset(defect_root: Path) -> dict[str, Any]:
    raw_dirs = [defect_root / folder for folder in RAW_DEFECT_DIRS]
    mask_dirs = [defect_root / folder for folder in MASK_DEFECT_DIRS]
    raw_images = []
    for folder in raw_dirs:
        raw_images.extend(list_image_files(folder))
    mask_images = []
    for folder in mask_dirs:
        mask_images.extend(list_image_files(folder))

    raw_by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in raw_images:
        raw_by_stem[path.stem].append(path)
    mask_by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in mask_images:
        mask_by_stem[path.stem].append(path)

    class_reports: dict[str, Any] = {}
    for class_name in RAW_DEFECT_DIRS:
        class_files = list_image_files(defect_root / class_name)
        per_class_mask_values: Counter[str] = Counter()
        missing_mask_files: list[str] = []
        mismatched_dimensions: list[dict[str, Any]] = []
        valid_pair_count = 0
        invalid_pair_count = 0
        paired_mask_count = 0
        empty_mask_count = 0

        for image_path in class_files:
            candidates = mask_by_stem.get(image_path.stem, [])
            if not candidates:
                missing_mask_files.append(image_path.name)
                invalid_pair_count += 1
                continue
            paired_mask_count += 1
            mask_path = candidates[0]
            try:
                with Image.open(image_path) as img_i, Image.open(mask_path) as img_m:
                    if img_i.size != img_m.size:
                        mismatched_dimensions.append({
                            "image": image_path.name,
                            "mask": mask_path.name,
                            "image_size": img_i.size,
                            "mask_size": img_m.size,
                        })
                        invalid_pair_count += 1
                        continue
                    mask_array = np.asarray(img_m)
                    unique_vals = np.unique(mask_array)
                    for value in unique_vals.tolist():
                        per_class_mask_values[str(value)] += int(np.sum(mask_array == value))
                    if mask_array.size == 0 or np.all(mask_array == 0):
                        empty_mask_count += 1
                    valid_pair_count += 1
            except Exception:
                invalid_pair_count += 1
                mismatched_dimensions.append({
                    "image": image_path.name,
                    "mask": mask_path.name,
                    "reason": "unreadable_image_or_mask",
                })

        class_reports[class_name] = {
            "image_count": len(class_files),
            "paired_mask_count": paired_mask_count,
            "valid_pair_count": valid_pair_count,
            "invalid_pair_count": invalid_pair_count,
            "mask_pixel_value_distribution": dict(sorted(per_class_mask_values.items(), key=lambda kv: kv[0])),
            "empty_mask_count": empty_mask_count,
            "missing_mask_files": missing_mask_files,
            "mismatched_dimensions": mismatched_dimensions,
        }

    defect_duplicates = build_duplicate_report(raw_images)
    mask_value_summary: dict[str, dict[str, int]] = {}
    for folder_name in MASK_DEFECT_DIRS:
        values: Counter[str] = Counter()
        for mask_path in list_image_files(defect_root / folder_name):
            try:
                arr = np.asarray(Image.open(mask_path))
            except Exception:
                continue
            for value in np.unique(arr).tolist():
                values[str(value)] += int(np.sum(arr == value))
        mask_value_summary[folder_name] = dict(sorted(values.items(), key=lambda kv: kv[0]))

    # infer direct stem mapping from filenames
    stem_relationship = {
        "good": {"pattern": "0001.png ... 0020.png", "mask_relation": "no mask expected"},
        "oil": {"pattern": "Oil_0001.jpg -> Oil_0001.png", "mask_relation": "ground_truth_2"},
        "scratch": {"pattern": "Scr_0001.jpg -> Scr_0001.png", "mask_relation": "ground_truth_1"},
        "stain": {"pattern": "Sta_0001.jpg -> Sta_0001.png", "mask_relation": "ground_truth_1"},
        "ground_truth_1": {"pattern": "Scr_* and Sta_* only", "mask_relation": "matches scratch and stain docs"},
        "ground_truth_2": {"pattern": "Oil_* only", "mask_relation": "matches oil docs"},
    }

    return {
        "dataset_root": str(defect_root),
        "raw_image_count": len(raw_images),
        "mask_image_count": len(mask_images),
        "folder_counts": {
            folder.name: len(list_image_files(folder))
            for folder in sorted(defect_root.iterdir(), key=lambda p: p.name.lower())
            if folder.is_dir()
        },
        "image_extensions": dict(sorted(Counter(p.suffix.lower() for p in raw_images).items())),
        "mask_extensions": dict(sorted(Counter(p.suffix.lower() for p in mask_images).items())),
        "raw_image_dimensions": {
            "unique_dimensions": {
                f"{width}x{height}": count
                for (width, height), count in sorted(
                    Counter((Image.open(p).width, Image.open(p).height) for p in raw_images).items(),
                    key=lambda kv: (-kv[1], kv[0]),
                )[:10]
            },
            "min_width": min(Image.open(p).width for p in raw_images),
            "max_width": max(Image.open(p).width for p in raw_images),
            "min_height": min(Image.open(p).height for p in raw_images),
            "max_height": max(Image.open(p).height for p in raw_images),
        },
        "mask_dimensions": {
            "unique_dimensions": {
                f"{width}x{height}": count
                for (width, height), count in sorted(
                    Counter((Image.open(p).width, Image.open(p).height) for p in mask_images).items(),
                    key=lambda kv: (-kv[1], kv[0]),
                )[:10]
            },
            "min_width": min(Image.open(p).width for p in mask_images),
            "max_width": max(Image.open(p).width for p in mask_images),
            "min_height": min(Image.open(p).height for p in mask_images),
            "max_height": max(Image.open(p).height for p in mask_images),
        },
        "relationship_by_folder": stem_relationship,
        "class_reports": class_reports,
        "mask_value_summary": mask_value_summary,
        "duplicate_summary": {
            "exact_duplicate_count": defect_duplicates["exact_duplicate_count"],
            "near_duplicate_count": defect_duplicates["near_duplicate_count"],
            "exact_duplicate_groups": defect_duplicates["exact_duplicate_groups"],
            "near_duplicate_pairs": defect_duplicates["near_duplicate_pairs"],
        },
    }


def build_report() -> dict[str, Any]:
    phone_report = validate_phone_dataset(PROJECT_ROOT / "datasets" / "phone_dataset")
    defect_report = validate_defect_dataset(PROJECT_ROOT / "datasets" / "defect_dataset")

    return {
        "PHONE DATASET": {
            "dataset_root": str(PROJECT_ROOT / "datasets" / "phone_dataset"),
            "image_count": phone_report["image_count"],
            "annotation_count": phone_report["annotation_count"],
            "image_extensions": phone_report["image_extensions"],
            "image_dimensions": phone_report["image_dimensions"],
            "class_distribution": phone_report["class_distribution"],
            "object_count_distribution": phone_report["object_count_distribution"],
            "annotation_mismatches": phone_report["annotation_mismatches"],
        },
        "DEFECT DATASET": {
            "dataset_root": str(PROJECT_ROOT / "datasets" / "defect_dataset"),
            "raw_image_count": defect_report["raw_image_count"],
            "mask_image_count": defect_report["mask_image_count"],
            "folder_counts": defect_report["folder_counts"],
            "image_extensions": defect_report["image_extensions"],
            "mask_extensions": defect_report["mask_extensions"],
            "raw_image_dimensions": defect_report["raw_image_dimensions"],
            "mask_dimensions": defect_report["mask_dimensions"],
            "relationship_by_folder": defect_report["relationship_by_folder"],
            "class_reports": defect_report["class_reports"],
        },
        "DUPLICATES": {
            "PHONE_DATASET": phone_report["duplicate_summary"],
            "DEFECT_DATASET": defect_report["duplicate_summary"],
        },
        "IMAGE-MASK PAIRING": {
            "raw_vs_mask_by_stem": {
                class_name: {
                    "raw_count": len(list_image_files(PROJECT_ROOT / "datasets" / "defect_dataset" / class_name)),
                    "mask_count": len(list_image_files(PROJECT_ROOT / "datasets" / "defect_dataset" / ("ground_truth_2" if class_name == "oil" else "ground_truth_1"))),
                    "match_count": sum(1 for p in list_image_files(PROJECT_ROOT / "datasets" / "defect_dataset" / class_name) if any(m.stem == p.stem for m in list_image_files(PROJECT_ROOT / "datasets" / "defect_dataset" / ("ground_truth_2" if class_name == "oil" else "ground_truth_1"))))
                }
                for class_name in ["good", "oil", "scratch", "stain"]
            },
            "defect_class_reports": defect_report["class_reports"],
        },
        "ANNOTATION VALIDATION": {
            "missing_annotations": phone_report["annotation_mismatches"]["missing_annotations"],
            "missing_images": phone_report["annotation_mismatches"]["missing_images"],
            "invalid_boxes": phone_report["annotation_mismatches"]["invalid_boxes"],
            "boxes_outside_image": phone_report["annotation_mismatches"]["boxes_outside_image"],
            "object_count_distribution": phone_report["object_count_distribution"],
        },
        "MASK VALIDATION": defect_report["mask_value_summary"],
    }


def main() -> None:
    payload = build_report()
    report_path = REPORTS_DIR / "phase2_validation_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(json.dumps({
        "phone_images": payload["PHONE DATASET"]["image_count"],
        "phone_annotations": payload["PHONE DATASET"]["annotation_count"],
        "defect_raw_images": payload["DEFECT DATASET"]["raw_image_count"],
        "defect_mask_images": payload["DEFECT DATASET"]["mask_image_count"],
        "phone_duplicate_count": payload["DUPLICATES"]["PHONE_DATASET"]["near_duplicate_count"],
        "defect_duplicate_count": payload["DUPLICATES"]["DEFECT_DATASET"]["near_duplicate_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
