from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "datasets" / "defect_dataset"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "defect_dataset_structure.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXPECTED_FOLDERS = ("good", "ground_truth_1", "ground_truth_2", "oil", "scratch", "stain")
MASK_FOLDERS = {"ground_truth_1", "ground_truth_2"}
IMAGE_FOLDERS = {"good", "oil", "scratch", "stain"}


def files_in(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.name.lower(),
    )


def inspect_image(path: Path) -> dict[str, Any] | None:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return {"width": image.width, "height": image.height, "mode": image.mode, "format": image.format}
    except Exception:
        return None


def image_hash(path: Path) -> str:
    with Image.open(path).convert("L") as image:
        resized = image.resize((8, 8), Image.Resampling.LANCZOS)
        values = np.asarray(resized, dtype=np.float32)
    bits = "".join("1" if value >= float(values.mean()) else "0" for value in values.flat)
    return bits


def hamming_distance(first: str, second: str) -> int:
    return sum(left != right for left, right in zip(first, second))


def mask_encoding(paths: list[Path]) -> dict[str, Any]:
    value_counts: Counter[str] = Counter()
    rgb_maps: Counter[str] = Counter()
    sample_details: list[dict[str, Any]] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                array = np.asarray(image)
                unique = np.unique(array)
                value_counts.update(str(int(value)) for value in unique if np.issubdtype(unique.dtype, np.integer))
                if array.ndim == 3:
                    colors = np.unique(array.reshape(-1, array.shape[-1]), axis=0)
                    rgb_maps.update("[" + ", ".join(str(int(value)) for value in color) + "]" for color in colors)
                if len(sample_details) < 5:
                    sample_details.append({"filename": path.name, "mode": image.mode, "unique_values": [int(value) for value in unique[:32]]})
        except Exception:
            continue
    return {
        "type": "indexed/grayscale mask" if not rgb_maps else "RGB mask",
        "unique_pixel_values": dict(sorted(value_counts.items(), key=lambda item: int(item[0]))),
        "rgb_value_map": dict(rgb_maps),
        "sample_details": sample_details,
    }


def mapping_for_folder(folder_name: str, root: Path, mask_paths: list[Path], image_paths_by_stem: dict[str, list[Path]]) -> dict[str, Any]:
    mappings = []
    missing_images = []
    for mask_path in mask_paths:
        matches = image_paths_by_stem.get(mask_path.stem, [])
        if matches:
            mappings.append({"mask": mask_path.name, "images": [str(path.relative_to(root)) for path in matches]})
        else:
            missing_images.append(mask_path.name)
    return {"mask_count": len(mask_paths), "matched_mask_count": len(mappings), "missing_image_for_mask": missing_images, "samples": mappings[:10]}


def collect_report(root: Path) -> dict[str, Any]:
    folders = {name: root / name for name in EXPECTED_FOLDERS}
    all_paths = [path for directory in folders.values() for path in files_in(directory)]
    image_paths = [path for name, directory in folders.items() if name in IMAGE_FOLDERS for path in files_in(directory)]
    image_paths_by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in image_paths:
        image_paths_by_stem[path.stem].append(path)

    folder_summary: dict[str, Any] = {}
    corrupt_files: list[dict[str, str]] = []
    missing_pairs: list[dict[str, str]] = []
    for name, directory in folders.items():
        paths = files_in(directory)
        roles = "masks" if name in MASK_FOLDERS else "images"
        metadata = Counter()
        for path in paths:
            details = inspect_image(path)
            if details is None:
                corrupt_files.append({"file": str(path.relative_to(root)), "reason": "unable_to_read_as_image"})
            else:
                metadata[f"{details['width']}x{details['height']}/{details['mode']}"] += 1
        folder_summary[name] = {
            "role": roles,
            "file_count": len(paths),
            "extensions": dict(Counter(path.suffix.lower() for path in paths)),
            "dimensions_and_modes": dict(metadata),
        }

    mask_mapping: dict[str, Any] = {}
    for name in MASK_FOLDERS:
        mask_mapping[name] = mapping_for_folder(name, root, files_in(folders[name]), image_paths_by_stem)
    for name in IMAGE_FOLDERS:
        paths = files_in(folders[name])
        expected_mask_folder = "ground_truth_1" if name == "scratch" else "ground_truth_2" if name == "oil" else None
        if expected_mask_folder:
            mask_stems = {path.stem for path in files_in(folders[expected_mask_folder])}
            missing_pairs.extend(
                {"image": str(path.relative_to(root)), "expected_mask_folder": expected_mask_folder, "reason": "missing_mask"}
                for path in paths
                if path.stem not in mask_stems
            )

    multiple_mask_images = [
        {"image_stem": stem, "mask_count": len([path for folder in MASK_FOLDERS for path in files_in(folders[folder]) if path.stem == stem])}
        for stem in sorted(image_paths_by_stem)
        if len([path for folder in MASK_FOLDERS for path in files_in(folders[folder]) if path.stem == stem]) > 1
    ]
    valid_images = [path for path in image_paths if inspect_image(path) is not None]
    hashes = [(path, image_hash(path)) for path in valid_images]
    exact_groups: dict[str, list[str]] = defaultdict(list)
    for path, value in hashes:
        exact_groups[value].append(str(path.relative_to(root)))
    duplicate_groups = [paths for paths in exact_groups.values() if len(paths) > 1]
    near_duplicates = []
    for index, (first_path, first_hash) in enumerate(hashes):
        for second_path, second_hash in hashes[index + 1 :]:
            distance = hamming_distance(first_hash, second_hash)
            if 0 < distance <= 4:
                near_duplicates.append({"image_a": str(first_path.relative_to(root)), "image_b": str(second_path.relative_to(root)), "hamming_distance": distance})

    return {
        "dataset_name": "defect_dataset",
        "root": str(root.resolve()),
        "folder_roles": folder_summary,
        "mask_to_image_mapping": mask_mapping,
        "inferred_mask_class_mapping": {
            "ground_truth_1": {"Scr": "scratch", "Sta": "stain"},
            "ground_truth_2": {"Oil": "oil"},
        },
        "mask_encoding": {name: mask_encoding(files_in(folders[name])) for name in MASK_FOLDERS},
        "multiple_defect_masks_per_image": {"supported_by_mapping": True, "observed_images_with_multiple_masks": multiple_mask_images},
        "counts": {"all_supported_files": len(all_paths), "image_files": len(image_paths), "mask_files": sum(len(files_in(folders[name])) for name in MASK_FOLDERS)},
        "missing_pairs": missing_pairs,
        "corrupt_files": corrupt_files,
        "duplicate_images": {"exact_groups": duplicate_groups, "near_duplicate_count": len(near_duplicates), "near_duplicate_pairs": near_duplicates[:200]},
        "conclusion": "Folder roles and mask mappings are explicit; conversion should preserve multiple mask lines per image and create empty labels for good images." if not corrupt_files else "Corrupt files require review before conversion.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect raw defect images and segmentation masks.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = collect_report(args.root.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"root": report["root"], "folder_roles": report["folder_roles"], "counts": report["counts"], "missing_pairs": len(report["missing_pairs"]), "corrupt_files": len(report["corrupt_files"]), "exact_duplicate_groups": len(report["duplicate_images"]["exact_groups"]), "near_duplicate_count": report["duplicate_images"]["near_duplicate_count"], "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
