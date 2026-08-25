from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "screen" / "screen_yolo11n_baseline_best.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 360


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.name.lower(),
    )


def order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = points.sum(axis=1)
    differences = points[:, 0] - points[:, 1]
    return np.array(
        [
            points[np.argmin(sums)],
            points[np.argmax(differences)],
            points[np.argmax(sums)],
            points[np.argmin(differences)],
        ],
        dtype=np.float32,
    )


def cross(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> float:
    first_vector = second - first
    second_vector = third - second
    return float(first_vector[0] * second_vector[1] - first_vector[1] * second_vector[0])


def segments_intersect(first: np.ndarray, second: np.ndarray, third: np.ndarray, fourth: np.ndarray) -> bool:
    def orientation(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> float:
        first_vector = end - start
        second_vector = point - start
        return float(first_vector[0] * second_vector[1] - first_vector[1] * second_vector[0])

    def on_segment(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> bool:
        return bool(
            min(start[0], end[0]) <= point[0] <= max(start[0], end[0])
            and min(start[1], end[1]) <= point[1] <= max(start[1], end[1])
        )

    first_orientation = orientation(first, second, third)
    second_orientation = orientation(first, second, fourth)
    third_orientation = orientation(third, fourth, first)
    fourth_orientation = orientation(third, fourth, second)
    if first_orientation == 0 and on_segment(first, second, third):
        return True
    if second_orientation == 0 and on_segment(first, second, fourth):
        return True
    if third_orientation == 0 and on_segment(third, fourth, first):
        return True
    if fourth_orientation == 0 and on_segment(third, fourth, second):
        return True
    return (first_orientation > 0) != (second_orientation > 0) and (third_orientation > 0) != (fourth_orientation > 0)


def polygon_is_degenerate(points: np.ndarray) -> bool:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(points).all() or len(np.unique(points, axis=0)) != 4:
        return True
    if abs(float(cv2.contourArea(points.reshape(-1, 1, 2)))) < 1.0:
        return True
    if segments_intersect(points[0], points[1], points[2], points[3]):
        return True
    if segments_intersect(points[1], points[2], points[3], points[0]):
        return True
    edge_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    return bool(np.min(edge_lengths) < 1.0)


def four_corners(polygon: np.ndarray) -> np.ndarray | None:
    polygon = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(polygon) < 4:
        return None
    perimeter = cv2.arcLength(polygon.reshape(-1, 1, 2), True)
    approximation = cv2.approxPolyDP(polygon.reshape(-1, 1, 2), 0.02 * perimeter, True).reshape(-1, 2)
    if len(approximation) == 4:
        return order_points(approximation)
    rectangle = cv2.boxPoints(cv2.minAreaRect(polygon)).astype(np.float32)
    return order_points(rectangle)


def orientation_and_transform(corners: np.ndarray) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    ordered = order_points(corners)
    width = max(np.linalg.norm(ordered[1] - ordered[0]), np.linalg.norm(ordered[2] - ordered[3]))
    height = max(np.linalg.norm(ordered[3] - ordered[0]), np.linalg.norm(ordered[2] - ordered[1]))
    if width > height:
        orientation = "landscape"
        destination = np.array([[0, 0], [OUTPUT_WIDTH - 1, 0], [OUTPUT_WIDTH - 1, OUTPUT_HEIGHT - 1], [0, OUTPUT_HEIGHT - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(ordered, destination)
    else:
        orientation = "portrait"
        portrait_destination = np.array([[0, 0], [359, 0], [359, 639], [0, 639]], dtype=np.float32)
        portrait_matrix = cv2.getPerspectiveTransform(ordered, portrait_destination)
        rotate_clockwise = np.array([[0, -1, 639], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        matrix = rotate_clockwise @ portrait_matrix
    return orientation, matrix, np.linalg.inv(matrix), ordered


def round_trip_self_test() -> bool:
    source = np.array([[20, 30], [500, 40], [480, 700], [30, 680]], dtype=np.float32)
    _, matrix, inverse, _ = orientation_and_transform(source)
    points = np.array([[20, 30], [250, 35], [500, 40], [480, 700], [30, 680]], dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points, matrix)
    recovered = cv2.perspectiveTransform(transformed, inverse)
    error = float(np.max(np.linalg.norm(recovered.reshape(-1, 2) - points.reshape(-1, 2), axis=1)))
    passed = error < 1.0
    print(f"Round-trip self-test: {'PASS' if passed else 'FAIL'} (max error {error:.6f} px)")
    return passed


def extract_detection(result: Any, confidence_threshold: float) -> tuple[np.ndarray | None, float | None, str]:
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return None, None, "SCREEN_NOT_FOUND"
    confidences = result.boxes.conf.detach().cpu().numpy().reshape(-1)
    best_index = int(np.argmax(confidences))
    confidence = float(confidences[best_index])
    polygons = result.masks.xy
    if best_index >= len(polygons):
        return None, confidence, "SCREEN_NOT_FOUND"
    corners = four_corners(polygons[best_index])
    if corners is None or polygon_is_degenerate(corners):
        return corners, confidence, "UNCERTAIN"
    if confidence < confidence_threshold:
        return corners, confidence, "UNCERTAIN"
    return corners, confidence, "SCREEN_NORMALIZED"


def process_one(image_path: Path, output_dir: Path, model: Any, confidence_threshold: float, image_size: int) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    output_image = output_dir / f"{image_path.stem}_normalized.jpg"
    output_json = output_dir / f"{image_path.stem}_normalized.json"
    result: dict[str, Any] = {"input": str(image_path), "status": "SCREEN_NOT_FOUND", "output": None, "source_corners": None, "confidence": None}
    if image is not None:
        prediction = model.predict(source=image, imgsz=image_size, conf=0.001, device="cpu", verbose=False)[0]
        corners, confidence, status = extract_detection(prediction, confidence_threshold)
        result["confidence"] = confidence
        result["status"] = status
        if corners is not None:
            orientation, matrix, inverse, ordered = orientation_and_transform(corners)
            result["source_corners"] = ordered.tolist()
            result["orientation"] = orientation
            result["M"] = matrix.tolist()
            result["M_inv"] = inverse.tolist()
            if status == "SCREEN_NORMALIZED":
                if orientation == "landscape":
                    normalized = cv2.warpPerspective(image, matrix, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
                else:
                    portrait = cv2.warpPerspective(image, cv2.getPerspectiveTransform(ordered, np.array([[0, 0], [359, 0], [359, 639], [0, 639]], dtype=np.float32)), (360, 640))
                    normalized = cv2.rotate(portrait, cv2.ROTATE_90_CLOCKWISE)
                output_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_image), normalized, [cv2.IMWRITE_JPEG_QUALITY, 95])
                result["output"] = str(output_image)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize phone screen perspective using a trained YOLO segmentation model.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path)
    group.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    weights = args.weights if args.weights.is_absolute() else PROJECT_ROOT / args.weights
    if not weights.exists():
        raise FileNotFoundError(f"Screen model weights not found: {weights}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1")
    if not round_trip_self_test():
        raise RuntimeError("Perspective matrix round-trip self-test failed")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    inputs = [args.input] if args.input is not None else list_images(args.input_dir)
    results = [process_one(path, args.output_dir, model, args.conf, args.imgsz) for path in inputs]
    if args.input_dir is not None:
        success = sum(result["status"] == "SCREEN_NORMALIZED" for result in results)
        rate = success / len(results) if results else 0.0
        print(f"Batch images: {len(results)}")
        print(f"SCREEN_NORMALIZED success rate: {rate:.3f} ({rate * 100:.1f}%)")
    else:
        print(json.dumps(results[0], indent=2) if results else "No input image")


if __name__ == "__main__":
    main()
