from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLDS = PROJECT_ROOT / "configs" / "decision_thresholds.yaml"
DISPLAY_NAMES = {"oil": "oil (grease/fingerprints)", "scratch": "scratch", "stain": "stain"}
MESSAGES = {
    "OK": "No defects found. The screen looks clean.",
    "OIL": "Grease or fingerprints found. Wipe the screen.",
    "SCRATCH": "Scratches found on the screen.",
    "STAIN": "Stains found on the screen.",
    "UNCERTAIN": "Something may be on the screen. Wipe it and retake the photo in good light.",
    "PHONE_TOO_FAR": "Phone is too small in the photo. Move closer.",
    "POOR_IMAGE_QUALITY": "Photo is blurry. Please retake it.",
    "NO_PHONE": "No phone found in the photo. Center the phone and try again.",
    "MULTIPLE": "Multiple possible defects found on the screen.",
}
ADVICE = {
    "OK": "No action needed.",
    "OIL": "Wipe the screen and retake the photo in good light.",
    "SCRATCH": "Check the screen closely and retake the photo in good light.",
    "STAIN": "Clean the screen and retake the photo in good light.",
    "UNCERTAIN": "Retake the photo in better light with the phone centered.",
    "PHONE_TOO_FAR": "Move closer and keep the whole phone in the photo.",
    "POOR_IMAGE_QUALITY": "Hold the camera steady and retake the photo.",
    "NO_PHONE": "Center the phone and retake the photo.",
    "MULTIPLE": "Wipe the screen and retake the photo in good light.",
}


def load_thresholds(path: Path = DEFAULT_THRESHOLDS) -> dict[str, float]:
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(key): float(value) for key, value in values.items()}


def _filtered_defects(defects: list[dict[str, Any]], thresholds: dict[str, float]) -> tuple[list[dict[str, Any]], bool]:
    real = [defect for defect in defects if float(defect.get("confidence") or 0.0) >= thresholds["defect_confidence_real"]]
    best = max((float(defect.get("confidence") or 0.0) for defect in defects), default=0.0)
    return real, thresholds["defect_confidence_uncertain"] <= best < thresholds["defect_confidence_real"]


def _simple(decision: str, defects: list[dict[str, Any]], post_processed: dict[str, Any], reason: str) -> dict[str, Any]:
    types = sorted({str(defect.get("class_name", "unknown")) for defect in defects})
    locations = sorted({component["location"] for defect in defects for component in defect.get("components", []) if component.get("location")})
    area = post_processed.get("real_total_affected_area", post_processed.get("total_affected_area", {})).get("percent")
    affected_area = f"about {area:.0f}% of the screen" if area is not None and area >= 1 else f"about {area:.1f}% of the screen" if area is not None else "not available"
    result_name = {"MULTIPLE": "MULTIPLE DEFECTS", "UNCERTAIN": "NOT SURE", "PHONE_TOO_FAR": "PHONE TOO FAR", "POOR_IMAGE_QUALITY": "BLURRY PHOTO", "NO_PHONE": "NO PHONE"}.get(decision, "OK" if decision == "OK" else "DEFECT FOUND")
    return {"result": result_name, "message": MESSAGES[decision], "defects_simple": [DISPLAY_NAMES.get(name, name) for name in types], "affected_area": affected_area, "locations": locations, "advice": ADVICE[decision], "reason": reason}


def decide(post_processed: dict[str, Any], statuses: dict[str, Any], blur_score: float | None = None, thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    thresholds = thresholds or load_thresholds()
    phone_conf = statuses.get("phone_confidence")
    screen_conf = statuses.get("screen_confidence")
    if statuses.get("phone_status") == "PHONE_NOT_FOUND":
        decision, reason = "NO_PHONE", "phone detection failed"
    elif phone_conf is not None and phone_conf < thresholds["phone_confidence_too_far"]:
        decision, reason = "PHONE_TOO_FAR", "phone confidence below configured threshold"
    elif blur_score is not None and blur_score < thresholds["blur_score_poor"]:
        decision, reason = "POOR_IMAGE_QUALITY", "Laplacian variance below configured threshold"
    elif statuses.get("screen_status") not in (None, "SCREEN_NORMALIZED") or (screen_conf is not None and screen_conf < thresholds["screen_confidence_uncertain"]):
        decision, reason = "UNCERTAIN", "screen localization confidence is too low"
    elif statuses.get("defect_status") not in (None, "DEFECT_SEGMENTED"):
        decision, reason = "UNCERTAIN", "defect segmentation failed"
    else:
        real, possible = _filtered_defects(post_processed.get("defects", []), thresholds)
        post_processed = {**post_processed, "defects": real, "defect_types": sorted({str(defect.get("class_name", "unknown")) for defect in real}), "total_affected_area": post_processed.get("real_total_affected_area", {"pixels": 0, "percent": 0.0}), "real_total_affected_area": post_processed.get("real_total_affected_area", {"pixels": 0, "percent": 0.0})}
        if possible and not real:
            decision, reason = "UNCERTAIN", "possible defect below the real-defect confidence threshold"
        else:
            types = post_processed["defect_types"]
            decision = "OK" if not types else types[0].upper() if len(types) == 1 else "MULTIPLE"
            reason = "confidence-filtered defect types classified"
    filtered = post_processed.get("defects", [])
    return {"decision": decision, "reason": reason, "defect_types": post_processed.get("defect_types", []), "filtered_defects": filtered, "blur_score": blur_score, "simple": _simple(decision, filtered, post_processed, reason), "thresholds": thresholds}
