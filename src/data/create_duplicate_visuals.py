from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFECT_ROOT = PROJECT_ROOT / "datasets" / "defect_dataset"
REPORT_PATH = PROJECT_ROOT / "reports" / "duplicate_category_report.json"
OUTPUT_ROOT = PROJECT_ROOT / "reports" / "visualizations" / "duplicate_review"


def find_image(filename: str) -> Path | None:
    for path in DEFECT_ROOT.glob(f"*/{filename}"):
        if path.is_file():
            return path
    return None


def main() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    created = 0
    for category, details in report.items():
        examples = details.get("near_duplicate_examples") or details.get("exact_duplicate_examples")
        for index, example in enumerate(examples[:5], start=1):
            left_path = find_image(example["image_a"])
            right_path = find_image(example["image_b"])
            if left_path is None or right_path is None:
                continue
            with Image.open(left_path).convert("RGB") as left, Image.open(right_path).convert("RGB") as right:
                left.thumbnail((640, 360), Image.Resampling.LANCZOS)
                right.thumbnail((640, 360), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (left.width + right.width + 40, max(left.height, right.height) + 60), "white")
                canvas.paste(left, (10, 30))
                canvas.paste(right, (left.width + 30, 30))
                draw = ImageDraw.Draw(canvas)
                draw.text((10, 8), example["image_a"], fill="black")
                draw.text((left.width + 30, 8), example["image_b"], fill="black")
                output_dir = OUTPUT_ROOT / category
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"pair_{index:02d}_{Path(example['image_a']).stem}_{Path(example['image_b']).stem}.png"
                canvas.save(output_path)
                created += 1
    print(json.dumps({"visualizations_created": created}, indent=2))


if __name__ == "__main__":
    main()
