from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pipeline.decision_engine import decide, load_thresholds
from pipeline.mask_analysis import analyze_masks
from screen.perspective_normalize import orientation_and_transform, polygon_is_degenerate, four_corners
PHONE_WEIGHTS = PROJECT_ROOT / "models" / "phone" / "phone_yolo11n_baseline_best.pt"
SCREEN_WEIGHTS = PROJECT_ROOT / "models" / "screen" / "screen_yolo11n_baseline_best.pt"
DEFECT_WEIGHTS = PROJECT_ROOT / "models" / "defect" / "defect_yolo11n_baseline_best.pt"
PIPELINE_CONFIG = PROJECT_ROOT / "configs" / "pipeline.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(directory: Path) -> list[Path]:
    return sorted([path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS], key=lambda path: path.name.lower()) if directory.is_dir() else []


def model_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_pipeline_weights(config_path: Path = PIPELINE_CONFIG) -> dict[str, Path]:
    defaults = {"phone": PHONE_WEIGHTS, "screen": SCREEN_WEIGHTS, "defect": DEFECT_WEIGHTS}
    if not config_path.exists():
        return defaults
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return {name: model_path(config.get(name, default)) for name, default in defaults.items()}


def best_box(result: Any) -> tuple[float, float, float, float, float] | None:
    if result.boxes is None or len(result.boxes) == 0:
        return None
    confidence = result.boxes.conf.detach().cpu().numpy().reshape(-1)
    index = int(np.argmax(confidence))
    box = result.boxes.xyxy[index].detach().cpu().numpy().tolist()
    return (*[float(value) for value in box], float(confidence[index]))


def crop_phone(image: np.ndarray, result: Any) -> tuple[np.ndarray | None, float | None]:
    detection = best_box(result)
    if detection is None:
        return None, None
    x1, y1, x2, y2, confidence = detection
    height, width = image.shape[:2]
    box_width, box_height = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - box_width * 0.05))
    y1 = max(0, int(y1 - box_height * 0.05))
    x2 = min(width, int(x2 + box_width * 0.05))
    y2 = min(height, int(y2 + box_height * 0.05))
    if x2 <= x1 or y2 <= y1:
        return None, confidence
    crop = image[y1:y2, x1:x2]
    scale = min(640 / crop.shape[1], 640 / crop.shape[0])
    resized = cv2.resize(crop, (round(crop.shape[1] * scale), round(crop.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((640, 640, 3), dtype=np.uint8)
    top = (640 - resized.shape[0]) // 2
    left = (640 - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas, confidence


def screen_transform(crop: np.ndarray, result: Any, threshold: float) -> tuple[np.ndarray | None, float | None, np.ndarray | None, str]:
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return None, None, None, "SCREEN_NOT_FOUND"
    confidences = result.boxes.conf.detach().cpu().numpy().reshape(-1)
    index = int(np.argmax(confidences))
    confidence = float(confidences[index])
    corners = four_corners(result.masks.xy[index]) if index < len(result.masks.xy) else None
    if corners is None or polygon_is_degenerate(corners):
        return None, confidence, None, "UNCERTAIN"
    orientation, matrix, _inverse, ordered = orientation_and_transform(corners)
    if confidence < threshold:
        return None, confidence, matrix, "UNCERTAIN"
    if orientation == "landscape":
        normalized = cv2.warpPerspective(crop, matrix, (640, 360))
    else:
        portrait_destination = np.array([[0, 0], [359, 0], [359, 639], [0, 639]], dtype=np.float32)
        portrait_matrix = cv2.getPerspectiveTransform(ordered, portrait_destination)
        portrait = cv2.warpPerspective(crop, portrait_matrix, (360, 640))
        normalized = cv2.rotate(portrait, cv2.ROTATE_90_CLOCKWISE)
    return normalized, confidence, matrix, "SCREEN_NORMALIZED"


def annotate(image: np.ndarray, result: Any, output_path: Path) -> None:
    canvas = image.copy()
    if result is not None and result.masks is not None and result.boxes is not None:
        masks = result.masks.data.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        colors = ((0, 0, 255), (0, 165, 255), (255, 0, 0))
        for mask, class_id in zip(masks, classes):
            mask = cv2.resize(mask.astype(np.float32), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0.5
            color = np.asarray(colors[class_id % len(colors)], dtype=np.uint8)
            canvas[mask] = (canvas[mask] * 0.45 + color * 0.55).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def enhancement_views(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    gamma_table = np.array([min(255, int(((value / 255.0) ** 0.5) * 255.0)) for value in range(256)], dtype=np.uint8)
    gamma = cv2.LUT(image, gamma_table)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, green_red, blue_yellow = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    clahe = cv2.cvtColor(cv2.merge((lightness, green_red, blue_yellow)), cv2.COLOR_LAB2BGR)
    return [("original", image), ("gamma_0.5", gamma), ("clahe", clahe)]


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / union) if union else 0.0


def segment_enhanced_views(model: Any, image: np.ndarray) -> tuple[dict[str, Any], list[Any]]:
    detections: list[dict[str, Any]] = []
    results: list[Any] = []
    for view_name, view in enhancement_views(image):
        prediction = model.predict(source=view, imgsz=640, conf=0.001, device="cpu", verbose=False)[0]
        results.append(prediction)
        if prediction.masks is None or prediction.boxes is None:
            continue
        masks = prediction.masks.data.detach().cpu().numpy()
        classes = prediction.boxes.cls.detach().cpu().numpy().astype(int)
        confidences = prediction.boxes.conf.detach().cpu().numpy()
        for mask, class_id, confidence in zip(masks, classes, confidences):
            resized = cv2.resize(mask.astype(np.float32), (640, 360), interpolation=cv2.INTER_NEAREST) > 0.5
            duplicate_index = next((item_index for item_index, item in enumerate(detections) if item["class_id"] == int(class_id) and mask_iou(item["mask"], resized) > 0.5), None)
            if duplicate_index is None or float(confidence) > detections[duplicate_index]["confidence"]:
                item = {"mask": resized, "class_id": int(class_id), "confidence": float(confidence), "view": view_name}
                if duplicate_index is not None:
                    detections[duplicate_index] = item
                else:
                    detections.append(item)
    analysis = analyze_masks([item["mask"] for item in detections], [item["class_id"] for item in detections], [item["confidence"] for item in detections], 640, 360, confidence_threshold=0.25)
    analysis["inference_views"] = [name for name, _ in enhancement_views(image)]
    return analysis, results


def _result_with_decision(result: dict[str, Any], decision: dict[str, Any], details: dict[str, Any]) -> dict[str, Any]:
    result["status"] = decision["decision"]
    result["simple"] = decision["simple"]
    result["technical"] = {**details, "decision_detail": decision}
    return result


def process_one(path: Path, output_dir: Path, models: dict[str, Any], screen_confidence: float, save_annotated: bool) -> dict[str, Any]:
    image = cv2.imread(str(path))
    output_json = output_dir / f"{path.stem}_pipeline.json"
    result: dict[str, Any] = {"input": str(path), "status": "UNCERTAIN", "simple": {}, "technical": {}}
    if image is None:
        result["statuses"] = {"image_quality": "POOR_IMAGE_QUALITY", "phone_status": "PHONE_NOT_FOUND"}
        decision = decide({}, result["statuses"], None, load_thresholds())
        _result_with_decision(result, decision, {"blur_score": None, "statuses": result["statuses"]})
        output_dir.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    blur_score = float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    phone_result = models["phone"].predict(source=image, imgsz=640, conf=0.001, device="cpu", verbose=False)[0]
    crop, phone_conf = crop_phone(image, phone_result)
    statuses: dict[str, Any] = {"image_quality": "OK", "phone_status": "PHONE_DETECTED" if crop is not None else "PHONE_NOT_FOUND", "phone_confidence": phone_conf}
    if crop is None:
        decision = decide({}, statuses, blur_score)
        _result_with_decision(result, decision, {"blur_score": blur_score, "statuses": statuses})
    else:
        screen_result = models["screen"].predict(source=crop, imgsz=640, conf=0.001, device="cpu", verbose=False)[0]
        normalized, confidence, _matrix, screen_status = screen_transform(crop, screen_result, screen_confidence)
        statuses.update({"screen_status": screen_status, "screen_confidence": confidence})
        if normalized is None:
            decision = decide({}, statuses, blur_score)
            _result_with_decision(result, decision, {"blur_score": blur_score, "statuses": statuses})
        else:
            analysis, defect_results = segment_enhanced_views(models["defect"], normalized)
            statuses["defect_status"] = "DEFECT_SEGMENTED"
            decision = decide(analysis, statuses, blur_score)
            _result_with_decision(result, decision, {"blur_score": blur_score, "statuses": statuses, "defects": analysis["defects"], "defect_types": analysis["defect_types"], "total_affected_area": analysis["total_affected_area"], "real_total_affected_area": analysis["real_total_affected_area"], "screen_dimensions": analysis["screen_dimensions"], "inference_views": analysis["inference_views"]})
            if save_annotated:
                annotate(normalized, defect_results[0] if defect_results else None, output_dir / f"{path.stem}_annotated.jpg")
                result["annotated_image"] = str(output_dir / f"{path.stem}_annotated.jpg")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run phone, screen, and defect models as one inspection pipeline.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path)
    group.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phone-weights", type=Path, default=PHONE_WEIGHTS)
    parser.add_argument("--screen-weights", type=Path, default=SCREEN_WEIGHTS)
    parser.add_argument("--defect-weights", type=Path, default=DEFECT_WEIGHTS)
    parser.add_argument("--screen-conf", type=float, default=0.5)
    parser.add_argument("--save-annotated", action="store_true")
    args = parser.parse_args()
    configured = load_pipeline_weights()
    weights = {name: model_path(value) for name, value in (("phone", args.phone_weights), ("screen", args.screen_weights), ("defect", args.defect_weights))}
    defaults = {"phone": PHONE_WEIGHTS, "screen": SCREEN_WEIGHTS, "defect": DEFECT_WEIGHTS}
    weights = {name: configured[name] if value == defaults[name] else value for name, value in weights.items()}
    missing = [str(path) for path in weights.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing model weights: " + ", ".join(missing))
    from ultralytics import YOLO
    models = {name: YOLO(str(path)) for name, path in weights.items()}
    inputs = [args.input] if args.input is not None else list_images(args.input_dir)
    results = [process_one(path, args.output_dir, models, args.screen_conf, args.save_annotated) for path in inputs]
    if args.input_dir is not None:
        normalized = sum(item.get("statuses", {}).get("screen_status") == "SCREEN_NORMALIZED" for item in results)
        rate = normalized / len(results) if results else 0.0
        print(f"Batch images: {len(results)}")
        print(f"SCREEN_NORMALIZED success rate: {rate:.3f} ({rate * 100:.1f}%)")
    else:
        print(json.dumps(results[0], indent=2) if results else "No input image")


if __name__ == "__main__":
    main()
