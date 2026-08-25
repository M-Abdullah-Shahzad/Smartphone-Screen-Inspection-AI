from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_report(name: str) -> dict[str, Any]:
    return json.loads((REPORTS_DIR / name).read_text(encoding="utf-8"))


def main() -> None:
    phase2 = load_report("phase2_validation_report.json")
    phone_review = load_report("phone_box_review.json")
    final_review = load_report("phase2_5_quality_review.json")
    duplicate_report = load_report("duplicate_category_report.json")

    phase2_records = phase2["ANNOTATION VALIDATION"]["boxes_outside_image"]
    unique_annotations = sorted({record["annotation"] for record in phase2_records})
    phone_visualization_files = sorted(
        (REPORTS_DIR / "visualizations" / "phone_box_review").glob("*.png")
    )

    cross_class_categories = [
        "good__oil",
        "good__scratch",
        "good__stain",
        "oil__scratch",
        "oil__stain",
        "scratch__stain",
    ]
    within_class_categories = [
        "good__good",
        "oil__oil",
        "scratch__scratch",
        "stain__stain",
    ]

    report = {
        "SOURCE_COUNTS": {
            "historical_phase2_claimed_out_of_bounds_boxes": 13,
            "phase2_report_out_of_bounds_records": len(phase2_records),
            "phase2_unique_affected_annotations": len(unique_annotations),
            "phone_box_review_details": len(phone_review["PHONE BOX REVIEW"]["details"]),
            "phase2_5_report_total_affected": final_review["PHONE_BOX_REVIEW"]["total_affected"],
            "phone_box_visualization_files": len(phone_visualization_files),
            "phone_box_visualization_pairs": len(phone_visualization_files),
            "unique_affected_annotation_names": unique_annotations,
        },
        "INTERPRETATION": {
            "what_28_represents": "28 unique affected phone annotations, 28 phone-box review records, and 28 phone-box visualization pairs/files.",
            "historical_13_status": "Not reproduced by the current Phase 2 report; the current report contains 28 out-of-bounds records and 28 unique annotation names.",
            "phase2_5_ready_for_phase3": False,
            "readiness_reason": "The count discrepancy with the historical Phase 2 value of 13 must be resolved before Phase 3.",
        },
        "DUPLICATE_COUNTS_BY_CATEGORY": {
            category: {
                "exact_duplicates": duplicate_report[category]["exact_duplicate_count"],
                "near_duplicates": duplicate_report[category]["near_duplicate_count"],
            }
            for category in duplicate_report
        },
        "CROSS_CLASS_EXACT_DUPLICATES": {
            category: duplicate_report[category]["exact_duplicate_count"]
            for category in cross_class_categories
        },
        "CROSS_CLASS_NEAR_DUPLICATES": {
            category: duplicate_report[category]["near_duplicate_count"]
            for category in cross_class_categories
        },
        "WITHIN_CLASS_DUPLICATES": {
            category: {
                "exact_duplicates": duplicate_report[category]["exact_duplicate_count"],
                "near_duplicates": duplicate_report[category]["near_duplicate_count"],
            }
            for category in within_class_categories
        },
    }

    output_path = REPORTS_DIR / "phase2_5_consistency_check.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
