from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import hashlib
import xml.etree.ElementTree as ET

from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def image_metadata(path: Path) -> dict[str, Any] | None:
    try:
        with Image.open(path) as img:
            return {
                "filename": path.name,
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "size_bytes": path.stat().st_size,
            }
    except Exception:
        return None


def phash(path: Path) -> str:
    with Image.open(path).convert("L") as img:
        img = img.resize((8, 8), Image.Resampling.LANCZOS)
        pixels = np.asarray(img, dtype=np.float32)
        mean = float(pixels.mean())
        bits = "".join("1" if pixel >= mean else "0" for pixel in pixels.flatten())
        h = int(bits, 2)
        return hashlib.md5(str(h).encode("utf-8")).hexdigest()


def hamming_distance(a: str, b: str) -> int:
    if len(a) != len(b):
        return 1000
    return sum(1 for x, y in zip(a, b) if x != y)


def collect_phone_dataset(root: Path) -> dict[str, Any]:
    image_dir = root / "Mobile_image" / "Mobile_image"
    annotation_dir = root / "Annotations" / "Annotations"

    image_files_all = list_image_files(image_dir)
    xml_files = sorted(annotation_dir.glob("*.xml"), key=lambda p: p.name.lower())

    image_meta = []
    corrupt_images = []
    for image_path in image_files_all:
        meta = image_metadata(image_path)
        if meta is None:
            corrupt_images.append({"filename": image_path.name, "reason": "unable_to_read"})
        else:
            image_meta.append(meta)

    image_stems = {path.stem for path in image_files_all}
    xml_stems = {path.stem for path in xml_files}
    missing_annotations = sorted(image_stems - xml_stems)
    missing_images = sorted(xml_stems - image_stems)

    class_counts: Counter[str] = Counter()
    annotation_object_count = 0
    annotation_classes = []
    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            xml_root = tree.getroot()
        except Exception:
            continue
        for obj in xml_root.findall("object"):
            name = obj.findtext("name")
            if name:
                class_counts[name] += 1
                annotation_classes.append(name)
                annotation_object_count += 1

    width_values = [item["width"] for item in image_meta]
    height_values = [item["height"] for item in image_meta]
    dim_counter = Counter((item["width"], item["height"]) for item in image_meta)
    dim_counter_serializable = {
        f"{width}x{height}": count
        for (width, height), count in sorted(dim_counter.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    }

    phashes = []
    for img_path in image_files_all:
        if is_valid_image(img_path):
            phashes.append((img_path.name, phash(img_path)))

    duplicate_candidates: list[dict[str, Any]] = []
    for i in range(len(phashes)):
        for j in range(i + 1, len(phashes)):
            name_a, hash_a = phashes[i]
            name_b, hash_b = phashes[j]
            dist = hamming_distance(hash_a, hash_b)
            if dist <= 4:
                duplicate_candidates.append({"image_a": name_a, "image_b": name_b, "hamming_distance": dist})

    report = {
        "dataset_name": "phone_dataset",
        "root": str(root),
        "structure": {
            "image_directory": str(image_dir),
            "annotation_directory": str(annotation_dir),
            "top_level_contents": [p.name for p in root.iterdir()],
        },
        "image_count": len(image_files_all),
        "annotation_count": len(xml_files),
        "image_extensions": dict(sorted(Counter(path.suffix.lower() for path in image_files_all).items())),
        "annotation_format": "Pascal VOC XML",
        "classes": dict(sorted(class_counts.items())),
        "class_count": len(class_counts),
        "annotation_objects": annotation_object_count,
        "image_dimensions": {
            "width_values": sorted(width_values),
            "height_values": sorted(height_values),
            "unique_dimensions": dim_counter_serializable,
            "min_width": min(width_values) if width_values else None,
            "max_width": max(width_values) if width_values else None,
            "min_height": min(height_values) if height_values else None,
            "max_height": max(height_values) if height_values else None,
        },
        "missing_annotations": missing_annotations,
        "missing_images": missing_images,
        "corrupt_files": corrupt_images,
        "duplicate_candidates": duplicate_candidates[:50],
        "duplicate_candidate_count": len(duplicate_candidates),
        "notes": "Dataset contains nested Mobile_image/Mobile_image and Annotations/Annotations folders; object class names are read directly from Pascal VOC XML labels.",
    }
    return report


def collect_defect_dataset(root: Path) -> dict[str, Any]:
    all_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    folder_counts = {}
    image_extensions = Counter()
    all_image_meta = []
    corrupt_files = []
    all_image_paths = []

    for folder in all_dirs:
        files = list_image_files(folder)
        folder_counts[folder.name] = len(files)
        all_image_paths.extend(files)
        for image in files:
            image_extensions[image.suffix.lower()] += 1
            meta = image_metadata(image)
            if meta is None:
                corrupt_files.append({"filename": image.name, "directory": folder.name, "reason": "unable_to_read"})
            else:
                all_image_meta.append({"directory": folder.name, **meta})

    ground_truth_dirs = [p for p in all_dirs if p.name.startswith("ground_truth")]
    mask_summary: dict[str, Any] = {}
    mask_values: dict[str, list[int]] = {}
    mask_files = []
    for gt_dir in ground_truth_dirs:
        files = list_image_files(gt_dir)
        mask_files.extend(files)
        mask_summary[gt_dir.name] = {"file_count": len(files), "sample_files": [p.name for p in files[:5]]}
        for mask_path in files[:5]:
            try:
                with Image.open(mask_path) as img:
                    arr = np.asarray(img)
                    vals = np.unique(arr).tolist()
                    if isinstance(vals[0], np.integer):
                        mask_values[mask_path.name] = [int(v) for v in vals[:10]]
                    else:
                        mask_values[mask_path.name] = [v for v in vals[:10]]
            except Exception:
                mask_values[mask_path.name] = []

    by_prefix = defaultdict(list)
    for image_path in all_image_paths:
        prefix = image_path.stem.split("_")[0].lower()
        by_prefix[prefix].append(image_path.name)

    phashes = []
    for path in all_image_paths:
        if is_valid_image(path):
            phashes.append((path.name, phash(path)))

    duplicate_candidates = []
    for i in range(len(phashes)):
        for j in range(i + 1, len(phashes)):
            name_a, hash_a = phashes[i]
            name_b, hash_b = phashes[j]
            dist = hamming_distance(hash_a, hash_b)
            if dist <= 4:
                duplicate_candidates.append({"image_a": name_a, "image_b": name_b, "hamming_distance": dist})

    width_values = [item["width"] for item in all_image_meta]
    height_values = [item["height"] for item in all_image_meta]
    dim_counter = Counter((item["width"], item["height"]) for item in all_image_meta)
    dim_counter_serializable = {
        f"{width}x{height}": count
        for (width, height), count in sorted(dim_counter.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    }

    report = {
        "dataset_name": "defect_dataset",
        "root": str(root),
        "structure": {
            "top_level_directories": [p.name for p in all_dirs],
            "folder_counts": folder_counts,
        },
        "image_count": len(all_image_paths),
        "image_extensions": dict(sorted(image_extensions.items())),
        "image_dimensions": {
            "width_values": sorted(width_values),
            "height_values": sorted(height_values),
            "unique_dimensions": dim_counter_serializable,
            "min_width": min(width_values) if width_values else None,
            "max_width": max(width_values) if width_values else None,
            "min_height": min(height_values) if height_values else None,
            "max_height": max(height_values) if height_values else None,
        },
        "mask_directories": {name: summary for name, summary in mask_summary.items()},
        "mask_format": {
            "type": "PNG segmentation masks",
            "sample_unique_values": mask_values,
        },
        "classes": {
            "by_folder_name": folder_counts,
            "by_name_prefix": {key: len(value) for key, value in sorted(by_prefix.items())},
        },
        "corrupt_files": corrupt_files,
        "duplicate_candidates": duplicate_candidates[:50],
        "duplicate_candidate_count": len(duplicate_candidates),
        "notes": "Defect images are organized by class folder and paired with ground-truth PNG masks. Sample mask pixel values reveal the raw segmentation mask format used in the dataset.",
    }
    return report


def write_json_report(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    phone_root = PROJECT_ROOT / "datasets" / "phone_dataset"
    defect_root = PROJECT_ROOT / "datasets" / "defect_dataset"

    phone_report = collect_phone_dataset(phone_root)
    defect_report = collect_defect_dataset(defect_root)

    write_json_report(REPORTS_DIR / "phone_dataset_report.json", phone_report)
    write_json_report(REPORTS_DIR / "defect_dataset_report.json", defect_report)

    print(json.dumps({
        "phone_dataset": {
            "image_count": phone_report["image_count"],
            "annotation_count": phone_report["annotation_count"],
            "classes": phone_report["classes"],
            "missing_annotations": len(phone_report["missing_annotations"]),
            "duplicate_candidates": phone_report["duplicate_candidate_count"],
        },
        "defect_dataset": {
            "image_count": defect_report["image_count"],
            "classes": defect_report["classes"]["by_folder_name"],
            "mask_directories": defect_report["mask_directories"],
            "duplicate_candidates": defect_report["duplicate_candidate_count"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
