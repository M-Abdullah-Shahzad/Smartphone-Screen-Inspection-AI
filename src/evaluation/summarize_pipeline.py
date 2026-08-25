from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "pipeline_summary.json"


def number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def summarize(input_dir: Path) -> dict[str, Any]:
    files = sorted(input_dir.glob("*_pipeline.json"), key=lambda path: path.name.lower()) if input_dir.is_dir() else []
    records: list[dict[str, Any]] = []
    unreadable: list[dict[str, str]] = []
    for path in files:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(record, dict):
                records.append(record)
            else:
                unreadable.append({"file": str(path), "reason": "JSON root is not an object"})
        except (OSError, json.JSONDecodeError) as error:
            unreadable.append({"file": str(path), "reason": str(error)})

    decisions: defaultdict[str, list[str]] = defaultdict(list)
    blur_scores: list[float] = []
    screen_confidences: list[float] = []
    defect_areas: list[float] = []
    for record in records:
        filename = Path(str(record.get("input", ""))).name or "<unknown>"
        decision = str(record.get("decision", record.get("status", "UNKNOWN")))
        decisions[decision].append(filename)
        blur = number(record.get("blur_score"))
        screen_confidence = number(record.get("statuses", {}).get("screen_confidence"))
        area = number(record.get("total_affected_area", {}).get("percent"))
        if blur is not None:
            blur_scores.append(blur)
        if screen_confidence is not None:
            screen_confidences.append(screen_confidence)
        if area is not None:
            defect_areas.append(area)

    summary = {
        "input_dir": str(input_dir.resolve()),
        "total_images_processed": len(records),
        "files_found": len(files),
        "unreadable_files": unreadable,
        "decision_counts": {decision: len(filenames) for decision, filenames in sorted(decisions.items())},
        "images_by_decision": {decision: filenames for decision, filenames in sorted(decisions.items())},
        "averages": {
            "blur_score": mean(blur_scores) if blur_scores else None,
            "screen_confidence": mean(screen_confidences) if screen_confidences else None,
            "defect_area_percent": mean(defect_areas) if defect_areas else None,
        },
        "observations": {
            "blur_scores_available": len(blur_scores),
            "screen_confidences_available": len(screen_confidences),
            "defect_areas_available": len(defect_areas),
        },
    }
    return summary


def print_table(summary: dict[str, Any]) -> None:
    print("Pipeline Evaluation Summary")
    print("=" * 28)
    print(f"Total images processed: {summary['total_images_processed']}")
    print("\nFinal decisions")
    print("-" * 30)
    print(f"{'Decision':<24} {'Count':>5}")
    for decision, count in summary["decision_counts"].items():
        print(f"{decision:<24} {count:>5}")
    print("\nAverages")
    print("-" * 30)
    averages = summary["averages"]
    for name, value in averages.items():
        label = name.replace("_", " ").title()
        print(f"{label:<24} {value:.4f}" if value is not None else f"{label:<24} n/a")
    print("\nImages by decision")
    print("-" * 30)
    for decision, filenames in summary["images_by_decision"].items():
        print(f"{decision}: {', '.join(filenames)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize pipeline JSON results.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    summary = summarize(args.input_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print_table(summary)
    print(f"\nSaved: {args.report}")


if __name__ == "__main__":
    main()
