from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "datasets" / "defect_dataset"
OUTPUT_ROOT = PROJECT_ROOT / "processed" / "defect_segmentation"
CONFIG_PATH = PROJECT_ROOT / "configs" / "defect.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_BY_FOLDER = {"scratch": 0, "oil": 1, "stain": 2}
IMAGE_FOLDERS = ("good", "scratch", "oil", "stain")
MASK_FOLDERS = ("ground_truth_1", "ground_truth_2")
SEED = 42


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    source_class: str
    stem: str
    file_hash: str
    perceptual_hash: str


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

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


def image_paths(directory: Path) -> list[Path]:
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.name.lower(),
    ) if directory.is_dir() else []


def perceptual_hash(path: Path) -> str:
    with Image.open(path).convert("L") as image:
        pixels = np.asarray(image.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    low_frequency = cv2.resize(pixels, (8, 8), interpolation=cv2.INTER_AREA)
    mean = float(low_frequency.mean())
    return "".join("1" if value >= mean else "0" for value in low_frequency.flat)


def build_records() -> list[ImageRecord]:
    records = []
    for source_class in IMAGE_FOLDERS:
        for path in image_paths(RAW_ROOT / source_class):
            records.append(
                ImageRecord(
                    path=path,
                    source_class=source_class,
                    stem=path.stem,
                    file_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
                    perceptual_hash=perceptual_hash(path),
                )
            )
    return records


def duplicate_groups(records: list[ImageRecord]) -> list[list[ImageRecord]]:
    union_find = UnionFind(len(records))
    first_by_key: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        for key in (("stem", record.stem), ("file", record.file_hash), ("perceptual", record.perceptual_hash)):
            if key in first_by_key:
                union_find.union(index, first_by_key[key])
            else:
                first_by_key[key] = index
    grouped: dict[int, list[ImageRecord]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[union_find.find(index)].append(record)
    return [sorted(group, key=lambda item: str(item.path).lower()) for group in grouped.values()]


def mask_lookup() -> dict[str, list[Path]]:
    lookup: dict[str, list[Path]] = defaultdict(list)
    for folder in MASK_FOLDERS:
        for path in image_paths(RAW_ROOT / folder):
            lookup[path.stem].append(path)
    return lookup


def mask_array(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"unable to read mask: {mask_path}")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def mask_class(mask_path: Path, source_class: str) -> int:
    if source_class not in CLASS_BY_FOLDER:
        raise ValueError(f"Masks are not expected for source class {source_class}: {mask_path}")
    return CLASS_BY_FOLDER[source_class]


def mask_polygons(mask_path: Path, source_class: str) -> list[str]:
    mask = mask_array(mask_path)
    height, width = mask.shape[:2]
    foreground = np.where(mask != 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    class_id = mask_class(mask_path, source_class)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 0:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, max(0.002 * perimeter, 0.5), True).reshape(-1, 2)
        if len(polygon) < 3:
            continue
        coordinates = []
        for x, y in polygon:
            coordinates.extend((float(np.clip(x / width, 0.0, 1.0)), float(np.clip(y / height, 0.0, 1.0))))
        lines.append(str(class_id) + " " + " ".join(f"{value:.6f}" for value in coordinates))
    return lines


def choose_splits(groups: list[list[ImageRecord]], seed: int) -> dict[str, list[ImageRecord]]:
    by_combination: dict[tuple[str, ...], list[list[ImageRecord]]] = defaultdict(list)
    for group in groups:
        combination = tuple(sorted({record.source_class for record in group}))
        by_combination[combination].append(group)
    rng = random.Random(seed)
    selected: dict[str, list[ImageRecord]] = {"train": [], "val": [], "test": []}
    for combination, combination_groups in sorted(by_combination.items()):
        rng.shuffle(combination_groups)
        total = len(combination_groups)
        train_end = round(total * 0.70)
        val_end = train_end + round(total * 0.15)
        for split, split_groups in (("train", combination_groups[:train_end]), ("val", combination_groups[train_end:val_end]), ("test", combination_groups[val_end:])):
            for group in split_groups:
                selected[split].append(group[0])
    return selected


def make_label(record: ImageRecord, masks: dict[str, list[Path]]) -> str:
    if record.source_class == "good":
        return ""
    lines = []
    for mask_path in masks.get(record.stem, []):
        lines.extend(mask_polygons(mask_path, record.source_class))
    return "\n".join(lines) + ("\n" if lines else "")


def write_dataset(splits: dict[str, list[ImageRecord]], masks: dict[str, list[Path]]) -> dict[str, dict[str, int]]:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    summary: dict[str, dict[str, int]] = {}
    for split, records in splits.items():
        image_dir = OUTPUT_ROOT / "images" / split
        label_dir = OUTPUT_ROOT / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        class_counts: Counter[str] = Counter()
        empty_labels = 0
        for record in records:
            shutil.copy2(record.path, image_dir / record.path.name)
            label_text = make_label(record, masks)
            (label_dir / f"{record.stem}.txt").write_text(label_text, encoding="utf-8")
            class_counts[record.source_class] += 1
            empty_labels += label_text == ""
        summary[split] = {"images": len(records), "empty_labels": empty_labels, **{f"{name}_images": class_counts[name] for name in IMAGE_FOLDERS}}
    return summary


def write_config() -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        f"path: {OUTPUT_ROOT.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames: [scratch, oil, stain]\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a deduplicated YOLO segmentation dataset for defects.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    records = build_records()
    groups = duplicate_groups(records)
    retained = [group[0] for group in groups]
    split_records = choose_splits(groups, args.seed)
    masks = mask_lookup()
    summary = write_dataset(split_records, masks)
    write_config()
    report = {
        "source_images": len(records),
        "deduplicated_images": len(retained),
        "duplicate_groups_removed": len(records) - len(retained),
        "duplicate_groups": len(groups),
        "split_summary": summary,
        "config": str(CONFIG_PATH),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
