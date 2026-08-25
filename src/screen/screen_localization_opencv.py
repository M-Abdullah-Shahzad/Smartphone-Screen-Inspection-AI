from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFECT_ROOT = PROJECT_ROOT / "datasets" / "defect_dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MIN_AREA_RATIO = 0.20
MAX_AREA_RATIO = 0.95
MIN_ASPECT = 1.20
MAX_ASPECT = 2.60


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and not path.stem.endswith("_screen")
        ],
        key=lambda path: path.name.lower(),
    )


def contour_touches_all_borders(contour: np.ndarray, width: int, height: int) -> bool:
    points = contour.reshape(-1, 2)
    tolerance = max(1, round(min(width, height) * 0.01))
    return (
        bool(np.any(points[:, 0] <= tolerance))
        and bool(np.any(points[:, 1] <= tolerance))
        and bool(np.any(points[:, 0] >= width - 1 - tolerance))
        and bool(np.any(points[:, 1] >= height - 1 - tolerance))
    )


def order_corners(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def candidate_metrics(
    contour: np.ndarray,
    width: int,
    height: int,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if diagnostics is not None:
        diagnostics["contour_count"] += 1
        diagnostics["contours"].append(contour)
    crop_area = float(width * height)
    area_ratio = float(cv2.contourArea(contour) / crop_area)
    if not MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO:
        if diagnostics is not None:
            diagnostics["reasons"]["area out of range"] += 1
        return None
    if not cv2.isContourConvex(contour):
        if diagnostics is not None:
            diagnostics["reasons"]["no quad found"] += 1
        return None
    if contour_touches_all_borders(contour, width, height):
        if diagnostics is not None:
            diagnostics["reasons"]["border touching"] += 1
        return None

    x, y, box_width, box_height = cv2.boundingRect(contour)
    if box_width <= 0 or box_height <= 0:
        if diagnostics is not None:
            diagnostics["reasons"]["no quad found"] += 1
        return None
    portrait_ratio = box_height / box_width
    landscape_ratio = box_width / box_height
    if not (MIN_ASPECT <= portrait_ratio <= MAX_ASPECT or MIN_ASPECT <= landscape_ratio <= MAX_ASPECT):
        if diagnostics is not None:
            diagnostics["reasons"]["aspect rejected"] += 1
        return None
    aspect = max(portrait_ratio, landscape_ratio)
    return {
        "area_ratio": area_ratio,
        "aspect_ratio_long_over_short": float(aspect),
        "bbox": {"x": int(x), "y": int(y), "width": int(box_width), "height": int(box_height)},
        "area": float(cv2.contourArea(contour)),
        "perimeter": float(cv2.arcLength(contour, True)),
    }


def contour_candidate(
    contour: np.ndarray,
    width: int,
    height: int,
    strategy: str,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    metrics = candidate_metrics(contour, width, height, diagnostics)
    if metrics is None:
        return None
    perimeter = metrics["perimeter"]
    polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    corners = None
    if len(polygon) == 4 and cv2.isContourConvex(polygon):
        corners = order_corners(polygon.reshape(4, 2)).tolist()
    return {**metrics, "contour": contour, "corners": corners, "strategy": strategy}


def find_bright_region(gray: np.ndarray, diagnostics: dict[str, Any]) -> dict[str, Any] | None:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [candidate for contour in contours if (candidate := contour_candidate(contour, gray.shape[1], gray.shape[0], "a", diagnostics))]
    return max(candidates, key=lambda candidate: candidate["area"]) if candidates else None


def find_adaptive_region(gray: np.ndarray, diagnostics: dict[str, Any]) -> dict[str, Any] | None:
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [candidate for contour in contours if (candidate := contour_candidate(contour, gray.shape[1], gray.shape[0], "b", diagnostics))]
    return max(candidates, key=lambda candidate: candidate["area"]) if candidates else None


def find_edge_quadrilateral(gray: np.ndarray, diagnostics: dict[str, Any]) -> dict[str, Any] | None:
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        candidate = contour_candidate(contour, gray.shape[1], gray.shape[0], "c", diagnostics)
        if candidate is not None and candidate["corners"] is not None:
            candidates.append(candidate)
        elif candidate is not None:
            diagnostics["reasons"]["no quad found"] += 1
    return max(candidates, key=lambda candidate: candidate["area"]) if candidates else None


def find_min_area_fallback(gray: np.ndarray, diagnostics: dict[str, Any]) -> dict[str, Any] | None:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        metrics = candidate_metrics(contour, gray.shape[1], gray.shape[0], diagnostics)
        if metrics is None:
            continue
        rectangle = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
        candidate = {**metrics, "contour": contour, "corners": order_corners(rectangle).tolist(), "strategy": "fallback"}
        return candidate
    return None


def localize_screen(image: np.ndarray) -> dict[str, Any]:
    if image is None or image.size == 0:
        return {"status": "SCREEN_NOT_FOUND", "bbox": None, "corners": None, "area_ratio": None, "strategy": None}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = gray.shape[:2]
    diagnostics = {
        "contour_count": 0,
        "contours": [],
        "reasons": Counter(),
    }
    strategies = (
        find_bright_region,
        find_adaptive_region,
        find_edge_quadrilateral,
        find_min_area_fallback,
    )
    candidate = None
    for strategy in strategies:
        candidate = strategy(gray, diagnostics)
        if candidate is not None:
            break
    if candidate is None:
        reasons = dict(diagnostics["reasons"])
        if not reasons:
            reasons["no quad found"] = 1
        return {
            "status": "SCREEN_NOT_FOUND",
            "bbox": None,
            "corners": None,
            "area_ratio": None,
            "strategy": None,
            "failure_reasons": reasons,
            "contour_count": diagnostics["contour_count"],
            "_contours": diagnostics["contours"],
        }
    status = "SCREEN_FOUND" if candidate["corners"] is not None else "UNCERTAIN"
    return {
        "status": status,
        "bbox": candidate["bbox"],
        "corners": candidate["corners"],
        "area_ratio": candidate["area_ratio"],
        "aspect_ratio_long_over_short": candidate["aspect_ratio_long_over_short"],
        "strategy": candidate["strategy"],
        "failure_reasons": dict(diagnostics["reasons"]),
        "contour_count": diagnostics["contour_count"],
        "image_size": {"width": width, "height": height},
        "acceptance_rules": {
            "area_ratio": [MIN_AREA_RATIO, MAX_AREA_RATIO],
            "aspect_ratio_long_over_short": [MIN_ASPECT, MAX_ASPECT],
            "contour_must_not_touch_all_borders": True,
        },
        "_contour": candidate["contour"],
        "_contours": diagnostics["contours"],
    }


def save_visualization(image: np.ndarray, result: dict[str, Any], output_path: Path) -> None:
    overlay = image.copy()
    for contour in result.get("_contours", []):
        cv2.drawContours(overlay, [contour], -1, (128, 128, 128), 1)
    contour = result.get("_contour")
    if contour is not None:
        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 3)
    bbox = result.get("bbox")
    if bbox is not None:
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 3)
    corners = result.get("corners")
    if corners is not None:
        points = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [points], True, (255, 0, 0), 3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)


def clean_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if not key.startswith("_")}


def aspect_ratio_distribution() -> dict[str, Any]:
    ratios: dict[str, list[float]] = {}
    for class_name in ("good", "oil", "scratch", "stain"):
        values = []
        for path in list_images(DEFECT_ROOT / class_name):
            image = cv2.imread(str(path))
            if image is not None and image.shape[0] > 0:
                values.append(float(image.shape[1] / image.shape[0]))
        ratios[class_name] = values
    all_values = [value for values in ratios.values() for value in values]
    summary = {
        class_name: {
            "count": len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "unique_values": dict(Counter(round(value, 6) for value in values)),
        }
        for class_name, values in ratios.items()
    }
    summary["all_raw_defect_images"] = {
        "count": len(all_values),
        "min": min(all_values) if all_values else None,
        "max": max(all_values) if all_values else None,
        "mean": float(np.mean(all_values)) if all_values else None,
        "median": float(np.median(all_values)) if all_values else None,
    }
    print("Raw defect-image aspect-ratio distribution (width / height):")
    for class_name, values in summary.items():
        if isinstance(values, dict) and "count" in values:
            print(f"  {class_name}: count={values['count']}, mean={values['mean']}, median={values['median']}, min={values['min']}, max={values['max']}")
    return summary


def process_one(input_path: Path, output_path: Path) -> dict[str, Any]:
    image = cv2.imread(str(input_path))
    if image is None:
        result = {"status": "SCREEN_NOT_FOUND", "bbox": None, "corners": None, "area_ratio": None, "strategy": None, "error": "image_unreadable"}
    else:
        result = localize_screen(image)
        save_visualization(image, result, output_path.with_suffix(".png"))
    result = {"input": str(input_path), **clean_result(result), "visualization": str(output_path.with_suffix(".png"))}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Localize a phone screen using OpenCV contours.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path)
    group.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path, help="Output JSON path for single-image mode.")
    parser.add_argument("--output-dir", type=Path, help="Output directory for batch JSON and visualization files.")
    args = parser.parse_args()

    aspect_ratio_distribution()
    if args.input is not None:
        output = args.output or args.input.with_name(f"{args.input.stem}_screen.json")
        result = process_one(args.input, output)
        print(json.dumps(clean_result(result), indent=2))
        return

    if args.output_dir is None:
        parser.error("--output-dir is required with --input-dir")
    inputs = list_images(args.input_dir)
    results = [process_one(path, args.output_dir / f"{path.stem}_screen.json") for path in inputs]
    counts = Counter(result["status"] for result in results)
    strategy_counts = Counter({strategy: 0 for strategy in ("a", "b", "c")})
    strategy_counts.update(result["strategy"] for result in results if result["strategy"] is not None)
    success_rate = sum(counts.get(status, 0) for status in ("SCREEN_FOUND",)) / len(results) if results else 0.0
    print(f"Batch images: {len(results)}")
    print(f"Status counts: {dict(counts)}")
    print(f"Strategy usage counts: {dict(strategy_counts)}")
    succeeded = [Path(result["input"]).name for result in results if result["status"] == "SCREEN_FOUND"]
    print(f"Succeeded filenames: {succeeded}")
    for result in results:
        if result["status"] != "SCREEN_FOUND":
            print(
                f"Failed {Path(result['input']).name}: "
                f"{result.get('failure_reasons', {})}; "
                f"contour count={result.get('contour_count', 0)}"
            )
    print(f"SCREEN_FOUND success rate: {success_rate:.3f} ({success_rate * 100:.1f}%)")


if __name__ == "__main__":
    main()
