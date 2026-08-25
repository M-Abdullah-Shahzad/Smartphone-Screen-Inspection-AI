from __future__ import annotations

from collections import deque
from typing import Any, Iterable

import cv2
import numpy as np

CLASS_NAMES = {0: "scratch", 1: "oil", 2: "stain"}
ZONE_NAMES = ("top-left", "top-center", "top-right", "middle-left", "center", "middle-right", "bottom-left", "bottom-center", "bottom-right")


def _as_mask_array(mask: Any, height: int, width: int) -> np.ndarray:
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    array = np.asarray(mask)
    if array.ndim > 2:
        array = np.squeeze(array)
    if array.shape != (height, width):
        array = cv2.resize(array.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    return array > 0.5


def _zone(x: float, y: float, width: int, height: int) -> str:
    column = min(2, int(x / max(width, 1) * 3))
    row = min(2, int(y / max(height, 1) * 3))
    return ZONE_NAMES[row * 3 + column]


def _components(mask: np.ndarray) -> list[dict[str, Any]]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [
        {"area_pixels": int(stats[index, cv2.CC_STAT_AREA]), "centroid": [float(centroids[index, 0]), float(centroids[index, 1])]}
        for index in range(1, count)
    ]


def analyze_masks(masks: Iterable[Any], classes: Iterable[Any], confidences: Iterable[Any], screen_width: int, screen_height: int, min_area_ratio: float = 0.001, confidence_threshold: float = 0.25) -> dict[str, Any]:
    mask_list = list(masks)
    class_list = list(classes)
    confidence_list = list(confidences)
    cleaned_union = np.zeros((screen_height, screen_width), dtype=bool)
    defects: list[dict[str, Any]] = []
    real_union = np.zeros((screen_height, screen_width), dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    minimum_pixels = screen_width * screen_height * min_area_ratio
    for index, raw_mask in enumerate(mask_list):
        mask = _as_mask_array(raw_mask, screen_height, screen_width)
        cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel).astype(bool)
        component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(cleaned.astype(np.uint8), 8)
        retained = np.zeros_like(cleaned)
        for component_index in range(1, component_count):
            if component_stats[component_index, cv2.CC_STAT_AREA] >= minimum_pixels:
                retained[component_labels == component_index] = True
        cleaned = retained
        if not cleaned.any():
            continue
        class_id = int(np.asarray(class_list[index]).item()) if index < len(class_list) else -1
        confidence = float(np.asarray(confidence_list[index]).item()) if index < len(confidence_list) else None
        components = _components(cleaned)
        for component in components:
            x, y = component["centroid"]
            component["location"] = _zone(x, y, screen_width, screen_height)
        defects.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
            "confidence": confidence,
            "area_pixels": int(cleaned.sum()),
            "area_percent": float(cleaned.sum() / (screen_width * screen_height) * 100.0),
            "components": components,
        })
        cleaned_union |= cleaned
        if confidence is not None and confidence >= confidence_threshold:
            real_union |= cleaned
    affected_pixels = int(cleaned_union.sum())
    real_affected_pixels = int(real_union.sum())
    real_defects = [defect for defect in defects if (defect["confidence"] or 0.0) >= confidence_threshold]
    return {
        "defects": defects,
        "defect_types": sorted({defect["class_name"] for defect in defects}),
        "total_affected_area": {"pixels": affected_pixels, "percent": float(affected_pixels / (screen_width * screen_height) * 100.0)},
        "real_defects": real_defects,
        "real_defect_types": sorted({defect["class_name"] for defect in real_defects}),
        "real_total_affected_area": {"pixels": real_affected_pixels, "percent": float(real_affected_pixels / (screen_width * screen_height) * 100.0)},
        "screen_dimensions": {"width": screen_width, "height": screen_height},
        "minimum_area_percent": min_area_ratio * 100.0,
    }


def analyze_ultralytics_result(result: Any, screen_width: int, screen_height: int, min_area_ratio: float = 0.001, confidence_threshold: float = 0.25) -> dict[str, Any]:
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return analyze_masks([], [], [], screen_width, screen_height, min_area_ratio, confidence_threshold)
    masks = result.masks.data
    classes = result.boxes.cls
    confidences = result.boxes.conf
    return analyze_masks(masks, classes, confidences, screen_width, screen_height, min_area_ratio, confidence_threshold)
