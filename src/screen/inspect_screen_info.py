from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "screen_annotation_report.json"
SCREEN_TERMS = ("screen", "display", "monitor", "glass", "lcd", "oled", "screen_region")
ANNOTATION_SUFFIXES = {".xml", ".json", ".txt", ".csv"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_DIR_NAMES = {"ground_truth_1", "ground_truth_2", "masks", "mask"}


def iter_existing_roots(project_root: Path) -> list[Path]:
    roots = [
        project_root / "datasets" / "phone_dataset",
        project_root / "datasets" / "defect_dataset",
        project_root / "processed" / "phone_detection",
        project_root / "processed" / "defect_segmentation",
    ]
    return [root for root in roots if root.is_dir()]


def contains_screen_term(value: str) -> bool:
    normalized = value.lower()
    return any(re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", normalized) for term in SCREEN_TERMS)


def annotation_files(roots: Iterable[Path]) -> list[Path]:
    return sorted(
        [path for root in roots for path in root.rglob("*") if path.is_file() and path.suffix.lower() in ANNOTATION_SUFFIXES],
        key=lambda path: str(path).lower(),
    )


def inspect_xml(path: Path) -> list[str]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return []
    values = []
    for element in root.iter():
        if element.tag.lower() in {"name", "label", "class", "category", "filename"} and element.text:
            values.append(element.text.strip())
    return [value for value in values if contains_screen_term(value)]


def inspect_structured_text(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
            text = json.dumps(payload)
        except json.JSONDecodeError:
            pass
    elif path.suffix.lower() == ".csv":
        try:
            rows = csv.reader(text.splitlines())
            text = "\n".join(",".join(row) for row in rows)
        except csv.Error:
            pass
    return sorted(set(re.findall(r"[^,;:\n\[\]{}\"']*?(?:screen|display|monitor|glass|lcd|oled)[^,;:\n\[\]{}\"']*", text, flags=re.IGNORECASE)))


def inspect_annotations(roots: Iterable[Path]) -> dict[str, Any]:
    files = annotation_files(roots)
    matches: list[dict[str, Any]] = []
    format_counts: Counter[str] = Counter()
    images_with_annotations: set[str] = set()
    for path in files:
        suffix = path.suffix.lower()
        values = inspect_xml(path) if suffix == ".xml" else inspect_structured_text(path)
        if not values:
            continue
        format_name = suffix[1:].upper()
        format_counts[format_name] += 1
        matches.append({
            "file": path.relative_to(PROJECT_ROOT).as_posix(),
            "format": format_name,
            "screen_like_values": values[:50],
        })
        images_with_annotations.add(path.stem)
    return {
        "annotation_files_scanned": len(files),
        "screen_annotation_file_count": len(matches),
        "screen_annotations_exist": bool(matches),
        "formats": dict(sorted(format_counts.items())),
        "images_with_screen_annotations": len(images_with_annotations),
        "matches": matches,
    }


def mask_files(roots: Iterable[Path]) -> list[Path]:
    paths = []
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if path.parent.name.lower() in MASK_DIR_NAMES or "ground_truth" in path.parent.name.lower():
                paths.append(path)
    return sorted(paths, key=lambda path: str(path).lower())


def summarize_masks(paths: list[Path]) -> dict[str, Any]:
    value_counts: Counter[str] = Counter()
    foreground_ratios: list[float] = []
    dimensions: Counter[str] = Counter()
    invalid_files: list[dict[str, str]] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                array = np.asarray(image)
                dimensions[f"{image.width}x{image.height}"] += 1
            if array.size == 0:
                invalid_files.append({"file": path.relative_to(PROJECT_ROOT).as_posix(), "reason": "empty_array"})
                continue
            values, counts = np.unique(array, return_counts=True)
            for value, count in zip(values.tolist(), counts.tolist()):
                value_counts[str(value)] += int(count)
            foreground_ratios.append(float(np.count_nonzero(array) / array.size))
        except Exception as error:
            invalid_files.append({"file": path.relative_to(PROJECT_ROOT).as_posix(), "reason": str(error)})
    return {
        "mask_file_count": len(paths),
        "mask_formats": dict(sorted(Counter(path.suffix.lower() for path in paths).items())),
        "unique_dimensions": dict(dimensions),
        "unique_pixel_values": dict(sorted(value_counts.items(), key=lambda item: item[0])),
        "foreground_ratio": {
            "min": min(foreground_ratios) if foreground_ratios else None,
            "max": max(foreground_ratios) if foreground_ratios else None,
            "mean": float(np.mean(foreground_ratios)) if foreground_ratios else None,
            "large_mask_count_ratio_at_least_50_percent": sum(ratio >= 0.5 for ratio in foreground_ratios),
        },
        "invalid_mask_files": invalid_files,
        "screen_like_mask_evidence": bool(foreground_ratios and sum(ratio >= 0.5 for ratio in foreground_ratios) / len(foreground_ratios) >= 0.8),
        "interpretation": "These files are treated as defect masks based on their source folders; mask statistics alone do not establish screen-region annotations.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect datasets for existing screen-region annotations or masks.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    roots = iter_existing_roots(project_root)
    annotation_report = inspect_annotations(roots)
    mask_paths = mask_files(roots)
    mask_report = summarize_masks(mask_paths)
    report = {
        "project_root": str(project_root),
        "inspected_roots": [path.relative_to(project_root).as_posix() for path in roots],
        "screen_annotations": annotation_report,
        "segmentation_mask_analysis": mask_report,
        "screen_information_available": annotation_report["screen_annotations_exist"] or mask_report["screen_like_mask_evidence"],
        "conclusion": "Existing screen annotations were found." if annotation_report["screen_annotations_exist"] else "No explicit screen annotations were found; existing masks are defect-mask artifacts and do not provide reliable screen-region labels.",
    }
    report_path = args.report if args.report.is_absolute() else project_root / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Screen annotations exist: {report['screen_annotations']['screen_annotations_exist']}")
    print(f"Annotation files containing screen-like labels: {report['screen_annotations']['screen_annotation_file_count']}")
    print(f"Images with screen annotations: {report['screen_annotations']['images_with_screen_annotations']}")
    print(f"Segmentation masks analyzed: {report['segmentation_mask_analysis']['mask_file_count']}")
    print(f"Screen-like mask evidence: {report['segmentation_mask_analysis']['screen_like_mask_evidence']}")
    print(report["conclusion"])


if __name__ == "__main__":
    main()
