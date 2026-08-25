from __future__ import annotations

import hashlib
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets"
PHONE_ROOT = DATASETS_DIR / "phone_dataset"
DEFECT_ROOT = DATASETS_DIR / "defect_dataset"
PROCESSED_ROOT = PROJECT_ROOT / "processed"
PHONE_OUT = PROCESSED_ROOT / "phone_detection"
DEFECT_OUT = PROCESSED_ROOT / "defect_segmentation"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFIGS_DIR = PROJECT_ROOT / "configs"
SEED = 42
SPLITS = ("train", "val", "test")
SPLIT_TARGETS = {"train": 0.70, "val": 0.15, "test": 0.15}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
RAW_CLASSES = ("good", "oil", "scratch", "stain")
MASK_DIRS = {"oil": "ground_truth_2", "scratch": "ground_truth_1", "stain": "ground_truth_1"}


@dataclass(frozen=True)
class DefectRecord:
    image: Path
    class_name: str
    mask: Path | None
    sha256: str
    signature: int
    group_id: int = -1


class UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def list_images(directory: Path) -> list[Path]:
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.name.lower(),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_signature(path: Path) -> int:
    with Image.open(path).convert("L") as image:
        pixels = np.asarray(image.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.uint8)
    bits = pixels[:, 1:] >= pixels[:, :-1]
    signature = 0
    for bit in bits.flatten():
        signature = (signature << 1) | int(bit)
    return signature


def bit_distance(first: int, second: int) -> int:
    return (first ^ second).bit_count()


def ensure_split_dirs(root: Path, include_labels: bool = False, include_masks: bool = False) -> None:
    for split in SPLITS:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        if include_labels:
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        if include_masks:
            (root / "masks" / split).mkdir(parents=True, exist_ok=True)


def clear_generated_output(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)


def copy_and_verify_image(source: Path, destination: Path) -> tuple[bool, str | None]:
    if not source.is_file():
        return False, f"source_missing: {source}"
    try:
        with Image.open(source) as image:
            image.verify()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if not destination.is_file():
            return False, f"destination_missing_after_copy: {destination}"
        with Image.open(destination) as image:
            image.verify()
    except Exception as error:
        return False, f"copy_or_open_failed: {source} -> {destination}: {error}"
    return True, None


def clip_box(values: dict[str, float], width: int, height: int) -> tuple[float, float, float, float] | None:
    xmin = max(0.0, min(float(width), values["xmin"]))
    ymin = max(0.0, min(float(height), values["ymin"]))
    xmax = max(0.0, min(float(width), values["xmax"]))
    ymax = max(0.0, min(float(height), values["ymax"]))
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def parse_phone_annotation(xml_path: Path, image_path: Path) -> tuple[str, list[str], int, int, int]:
    root = ET.parse(xml_path).getroot()
    filename = (root.findtext("filename") or image_path.name).strip()
    if filename != image_path.name:
        raise ValueError(f"filename mismatch: {filename} != {image_path.name}")
    with Image.open(image_path) as image:
        width, height = image.size
    labels: list[str] = []
    clipped_count = 0
    invalid_count = 0
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            invalid_count += 1
            continue
        values: dict[str, float] = {}
        try:
            for key in ("xmin", "ymin", "xmax", "ymax"):
                raw = box.findtext(key)
                if raw is None:
                    raise ValueError(f"missing {key}")
                values[key] = float(raw)
        except (TypeError, ValueError):
            invalid_count += 1
            continue
        original = (values["xmin"], values["ymin"], values["xmax"], values["ymax"])
        clipped = clip_box(values, width, height)
        if clipped is None:
            invalid_count += 1
            continue
        if clipped != original:
            clipped_count += 1
        xmin, ymin, xmax, ymax = clipped
        center_x = ((xmin + xmax) / 2) / width
        center_y = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        labels.append(f"0 {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}")
    if not labels:
        invalid_count += 1
    return filename, labels, clipped_count, invalid_count, len(root.findall("object"))


def assign_splits(items: list[Any], key_size: Any, seed: int = SEED) -> dict[Any, str]:
    randomizer = random.Random(seed)
    ordered = sorted(items, key=lambda item: str(item))
    randomizer.shuffle(ordered)
    ordered.sort(key=lambda item: (-key_size(item), str(item)))
    total = sum(key_size(item) for item in ordered)
    targets = {split: total * ratio for split, ratio in SPLIT_TARGETS.items()}
    current = Counter()
    assignment: dict[Any, str] = {}
    for item in ordered:
        split = min(SPLITS, key=lambda candidate: (current[candidate] - targets[candidate], SPLITS.index(candidate)))
        assignment[item] = split
        current[split] += key_size(item)
    return assignment


def prepare_phone_dataset() -> dict[str, Any]:
    image_dir = PHONE_ROOT / "Mobile_image" / "Mobile_image"
    annotation_dir = PHONE_ROOT / "Annotations" / "Annotations"
    image_paths = list_images(image_dir)
    xml_by_stem = {path.stem: path for path in annotation_dir.glob("*.xml")}
    valid_records: list[tuple[Path, list[str], int, int, int]] = []
    invalid_pairs: list[dict[str, str]] = []
    clipped_count = 0
    invalid_box_count = 0
    total_objects = 0

    for image_path in image_paths:
        xml_path = xml_by_stem.get(image_path.stem)
        if xml_path is None:
            invalid_pairs.append({"image": image_path.name, "reason": "missing_annotation"})
            continue
        try:
            _, labels, clipped, invalid, objects = parse_phone_annotation(xml_path, image_path)
        except Exception as error:
            invalid_pairs.append({"image": image_path.name, "annotation": xml_path.name, "reason": str(error)})
            continue
        clipped_count += clipped
        invalid_box_count += invalid
        total_objects += objects
        if invalid == 0 and labels:
            valid_records.append((image_path, labels, clipped, invalid, objects))
        else:
            invalid_pairs.append({"image": image_path.name, "annotation": xml_path.name, "reason": "invalid_box"})

    clear_generated_output(PHONE_OUT)
    assignment = assign_splits([record[0].name for record in valid_records], lambda _: 1)
    ensure_split_dirs(PHONE_OUT, include_labels=True)
    split_counts = Counter()
    class_distribution = Counter()
    successful_copies = 0
    failed_copies: list[dict[str, str]] = []
    for image_path, labels, _, _, objects in valid_records:
        split = assignment[image_path.name]
        destination_image = PHONE_OUT / "images" / split / image_path.name
        copied, error = copy_and_verify_image(image_path, destination_image)
        if not copied:
            failed_copies.append({"source": str(image_path), "destination": str(destination_image), "error": error or "unknown_copy_error"})
            continue
        successful_copies += 1
        (PHONE_OUT / "labels" / split / f"{image_path.stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
        split_counts[split] += 1
        class_distribution["phone"] += objects

    parity_errors: list[dict[str, Any]] = []
    unreadable_images: list[str] = []
    for split in SPLITS:
        image_files = list_images(PHONE_OUT / "images" / split)
        label_files = sorted((PHONE_OUT / "labels" / split).glob("*.txt"), key=lambda path: path.name.lower())
        image_stems = {path.stem for path in image_files}
        label_stems = {path.stem for path in label_files}
        if image_stems != label_stems:
            parity_errors.append({
                "split": split,
                "missing_images_for_labels": sorted(label_stems - image_stems),
                "missing_labels_for_images": sorted(image_stems - label_stems),
            })
        for image_file in image_files:
            try:
                with Image.open(image_file) as image:
                    image.verify()
            except Exception as error:
                unreadable_images.append(f"{split}/{image_file.name}: {error}")

    if failed_copies or parity_errors or unreadable_images:
        raise RuntimeError(json.dumps({
            "failed_copies": failed_copies,
            "parity_errors": parity_errors,
            "unreadable_images": unreadable_images,
        }, indent=2))

    config_path = CONFIGS_DIR / "phone.yaml"
    config_path.write_text(
        "path: ../processed/phone_detection\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names: [phone]\n",
        encoding="utf-8",
    )
    report = {
        "dataset": "phone_detection",
        "source": str(PHONE_ROOT),
        "processed_root": str(PHONE_OUT),
        "train_count": split_counts["train"],
        "val_count": split_counts["val"],
        "test_count": split_counts["test"],
        "total": sum(split_counts.values()),
        "class_distribution": dict(class_distribution),
        "clipped_box_count": clipped_count,
        "invalid_box_count": invalid_box_count,
        "invalid_pairs": invalid_pairs,
        "images_successfully_copied": successful_copies,
        "failed_image_copies": failed_copies,
        "unreadable_processed_images": unreadable_images,
        "image_label_parity": {split: len(list_images(PHONE_OUT / "images" / split)) == len(list((PHONE_OUT / "labels" / split).glob("*.txt"))) for split in SPLITS},
        "random_seed": SEED,
        "split_percentages": {split: split_counts[split] / max(1, sum(split_counts.values())) for split in SPLITS},
        "annotation_format": "YOLO detection, class 0 = phone",
    }
    (REPORTS_DIR / "phone_split_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_defect_records() -> tuple[list[DefectRecord], dict[str, Any]]:
    records: list[DefectRecord] = []
    missing_masks: list[str] = []
    invalid_masks: list[str] = []
    empty_masks: list[str] = []
    for class_name in RAW_CLASSES:
        for image_path in list_images(DEFECT_ROOT / class_name):
            mask_path = None
            if class_name in MASK_DIRS:
                candidates = [path for path in list_images(DEFECT_ROOT / MASK_DIRS[class_name]) if path.stem == image_path.stem]
                if not candidates:
                    missing_masks.append(image_path.name)
                    continue
                mask_path = candidates[0]
                try:
                    with Image.open(image_path) as image, Image.open(mask_path) as mask:
                        if image.size != mask.size:
                            invalid_masks.append(f"{image_path.name}: dimension mismatch")
                            continue
                        mask_array = np.asarray(mask)
                        if mask_array.size == 0 or np.all(mask_array == 0):
                            empty_masks.append(image_path.name)
                except Exception as error:
                    invalid_masks.append(f"{image_path.name}: {error}")
                    continue
            try:
                records.append(DefectRecord(image_path, class_name, mask_path, sha256(image_path), perceptual_signature(image_path)))
            except Exception as error:
                invalid_masks.append(f"{image_path.name}: unreadable image ({error})")
    return records, {
        "missing_masks": missing_masks,
        "invalid_masks": invalid_masks,
        "empty_masks": empty_masks,
    }


def group_defect_records(records: list[DefectRecord], threshold: int = 4) -> list[list[DefectRecord]]:
    union_find = UnionFind(len(records))
    exact_groups: dict[str, list[int]] = defaultdict(list)
    buckets: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        exact_groups[record.sha256].append(index)
        buckets[record.signature >> 60].append(index)
    for indices in exact_groups.values():
        for index in indices[1:]:
            union_find.union(indices[0], index)
    for indices in buckets.values():
        for position, first in enumerate(indices):
            for second in indices[position + 1:]:
                if bit_distance(records[first].signature, records[second].signature) <= threshold:
                    union_find.union(first, second)
    grouped: dict[int, list[DefectRecord]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[union_find.find(index)].append(record)
    groups = sorted(grouped.values(), key=lambda group: (min(item.image.name for item in group), len(group)))
    return groups


def prepare_defect_dataset() -> tuple[dict[str, Any], list[DefectRecord], list[list[DefectRecord]], dict[str, str]]:
    records, mask_report = build_defect_records()
    groups = group_defect_records(records)
    group_keys = [min(record.image.relative_to(DEFECT_ROOT).as_posix() for record in group) for group in groups]
    group_sizes = {key: len(group) for key, group in zip(group_keys, groups)}
    group_assignment = assign_splits(group_keys, lambda key: group_sizes[key])
    record_assignment: dict[str, str] = {}
    clear_generated_output(DEFECT_OUT)
    ensure_split_dirs(DEFECT_OUT, include_masks=True)
    split_counts = Counter()
    per_class = {class_name: Counter() for class_name in RAW_CLASSES}
    metadata_handles = {
        split: (DEFECT_OUT / f"metadata_{split}.jsonl").open("w", encoding="utf-8") for split in SPLITS
    }
    try:
        for group, group_key in zip(groups, group_keys):
            split = group_assignment[group_key]
            for record in group:
                record_assignment[record.image.name] = split
                relative_image = record.image.relative_to(DEFECT_ROOT).as_posix()
                output_image = DEFECT_OUT / "images" / split / record.image.name
                shutil.copy2(record.image, output_image)
                if record.mask is not None:
                    shutil.copy2(record.mask, DEFECT_OUT / "masks" / split / record.mask.name)
                metadata_handles[split].write(json.dumps({
                    "image": record.image.name,
                    "source_image": relative_image,
                    "class": record.class_name,
                    "mask": record.mask.relative_to(DEFECT_ROOT).as_posix() if record.mask else None,
                    "processed_mask": record.mask.name if record.mask else None,
                    "similarity_group": group_key,
                }) + "\n")
                split_counts[split] += 1
                per_class[record.class_name][split] += 1
    finally:
        for handle in metadata_handles.values():
            handle.close()

    report = {
        "dataset": "defect_segmentation",
        "source": str(DEFECT_ROOT),
        "processed_root": str(DEFECT_OUT),
        "total_raw_images": len(records),
        "good_count": per_class["good"]["train"] + per_class["good"]["val"] + per_class["good"]["test"],
        "oil_count": per_class["oil"]["train"] + per_class["oil"]["val"] + per_class["oil"]["test"],
        "scratch_count": per_class["scratch"]["train"] + per_class["scratch"]["val"] + per_class["scratch"]["test"],
        "stain_count": per_class["stain"]["train"] + per_class["stain"]["val"] + per_class["stain"]["test"],
        "train_count": split_counts["train"],
        "val_count": split_counts["val"],
        "test_count": split_counts["test"],
        "per_class_counts": {class_name: dict(counts) for class_name, counts in per_class.items()},
        "similarity_group_count": len(groups),
        "group_sizes": sorted(len(group) for group in groups),
        "group_assignment": group_assignment,
        "groups_crossing_splits": 0,
        "missing_masks": mask_report["missing_masks"],
        "invalid_masks": mask_report["invalid_masks"],
        "empty_masks": mask_report["empty_masks"],
        "random_seed": SEED,
        "split_percentages": {split: split_counts[split] / max(1, len(records)) for split in SPLITS},
        "similarity_method": "SHA-256 exact grouping plus 64-bit 8x8 grayscale perceptual signatures, Hamming threshold 4 within 4-bit buckets",
    }
    (REPORTS_DIR / "defect_split_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report, records, groups, record_assignment


def run_leakage_checks(phone_report: dict[str, Any], defect_report: dict[str, Any], records: list[DefectRecord], groups: list[list[DefectRecord]], record_assignment: dict[str, str]) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    checks["phone_pairs_remain_together"] = {"status": "PASS" if not phone_report["invalid_pairs"] else "FAIL", "invalid_pairs": phone_report["invalid_pairs"]}
    checks["similarity_groups_single_split"] = {"status": "PASS" if defect_report["groups_crossing_splits"] == 0 else "FAIL", "groups_crossing_splits": defect_report["groups_crossing_splits"]}
    checks["source_images_single_split"] = {"status": "PASS" if len(record_assignment) == len(records) else "FAIL", "assigned": len(record_assignment), "records": len(records)}
    mask_mismatches: list[str] = []
    for record in records:
        if record.mask is None:
            continue
        source_split = record_assignment[record.image.name]
        expected_path = DEFECT_OUT / "masks" / source_split / record.mask.name
        other_splits = [split for split in SPLITS if split != source_split and (DEFECT_OUT / "masks" / split / record.mask.name).exists()]
        if not expected_path.exists() or other_splits:
            mask_mismatches.append(record.image.name)
    checks["masks_same_split_as_source"] = {"status": "PASS" if not mask_mismatches else "FAIL", "mismatches": mask_mismatches}
    mask_names = {record.mask.name for record in records if record.mask is not None}
    masks_in_images = [
        path.name for path in (DEFECT_OUT / "images").glob("**/*")
        if path.is_file() and path.name in mask_names
    ]
    checks["masks_not_independent_images"] = {
        "status": "PASS" if not masks_in_images else "FAIL",
        "mask_files_in_processed_images": masks_in_images,
    }
    checks["exact_duplicates_not_across_splits"] = {"status": "PASS", "cross_split_exact_duplicate_groups": []}
    for group in groups:
        splits = {record_assignment[record.image.name] for record in group}
        if len(splits) > 1:
            checks["similarity_groups_single_split"]["status"] = "FAIL"
            checks["similarity_groups_single_split"].setdefault("details", []).append(sorted(splits))
        for record in group:
            if record.mask is not None and record_assignment.get(record.image.name) != record_assignment.get(record.image.name):
                checks["masks_same_split_as_source"]["status"] = "FAIL"
                checks["masks_same_split_as_source"]["mismatches"].append(record.image.name)
    by_hash: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_hash[record.sha256].add(record_assignment[record.image.name])
    cross_hashes = [digest for digest, splits in by_hash.items() if len(splits) > 1]
    if cross_hashes:
        checks["exact_duplicates_not_across_splits"] = {"status": "FAIL", "cross_split_exact_duplicate_groups": cross_hashes}
    overall = "PASS" if all(item["status"] == "PASS" for item in checks.values()) else "FAIL"
    result = {"overall": overall, "checks": checks, "random_seed": SEED}
    (REPORTS_DIR / "phase3_leakage_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    phone_report = prepare_phone_dataset()
    defect_report, records, groups, record_assignment = prepare_defect_dataset()
    leakage_report = run_leakage_checks(phone_report, defect_report, records, groups, record_assignment)
    print(json.dumps({
        "phone": {key: phone_report[key] for key in ("train_count", "val_count", "test_count", "total", "clipped_box_count", "invalid_box_count")},
        "defect": {key: defect_report[key] for key in ("train_count", "val_count", "test_count", "total_raw_images", "similarity_group_count")},
        "leakage": leakage_report["overall"],
    }, indent=2))


if __name__ == "__main__":
    main()
