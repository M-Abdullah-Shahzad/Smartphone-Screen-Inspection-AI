from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHONE_ROOT = PROJECT_ROOT / "datasets" / "phone_dataset"
IMAGE_DIR = PHONE_ROOT / "Mobile_image" / "Mobile_image"
ANNOTATION_DIR = PHONE_ROOT / "Annotations" / "Annotations"
OUT_DIR = PROJECT_ROOT / "reports" / "visualizations" / "phone_box_review"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def clamp_box(xmin: float, ymin: float, xmax: float, ymax: float, width: int, height: int) -> tuple[float, float, float, float]:
    left = max(0.0, xmin)
    top = max(0.0, ymin)
    right = min(float(width), xmax)
    bottom = min(float(height), ymax)
    return left, top, right, bottom


def compute_outside_pixels(xmin: float, ymin: float, xmax: float, ymax: float, width: int, height: int) -> dict[str, float]:
    return {
        "left": max(0.0, 0.0 - xmin),
        "top": max(0.0, 0.0 - ymin),
        "right": max(0.0, xmax - float(width)),
        "bottom": max(0.0, ymax - float(height)),
    }


def find_phone_box_outliers() -> list[dict[str, Any]]:
    outliers: list[dict[str, Any]] = []
    for xml_path in sorted(ANNOTATION_DIR.glob("*.xml"), key=lambda p: p.name.lower()):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception:
            continue

        image_name = (root.findtext("filename") or "").strip()
        image_path = IMAGE_DIR / image_name
        if not image_path.exists():
            continue

        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception:
            continue

        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            values = {}
            for key in ("xmin", "ymin", "xmax", "ymax"):
                raw = bndbox.findtext(key)
                if raw is None:
                    values[key] = None
                    continue
                try:
                    values[key] = float(raw)
                except ValueError:
                    values[key] = None
            if any(v is None for v in values.values()):
                continue

            xmin, ymin, xmax, ymax = values["xmin"], values["ymin"], values["xmax"], values["ymax"]
            if xmin < 0 or ymin < 0 or xmax > width or ymax > height or xmax <= xmin or ymax <= ymin:
                left, top, right, bottom = clamp_box(xmin, ymin, xmax, ymax, width, height)
                valid = left < right and top < bottom
                outliers.append({
                    "annotation": xml_path.name,
                    "image": image_name,
                    "image_size": {"width": width, "height": height},
                    "original": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
                    "clipped": {"xmin": left, "ymin": top, "xmax": right, "ymax": bottom},
                    "outside_pixels": compute_outside_pixels(xmin, ymin, xmax, ymax, width, height),
                    "clipped_box_width": max(0.0, right - left),
                    "clipped_box_height": max(0.0, bottom - top),
                    "valid_after_clipping": valid,
                })
    return outliers


def draw_box(image: Image.Image, box: tuple[float, float, float, float], color: tuple[int, int, int], width: int = 4) -> None:
    draw = ImageDraw.Draw(image)
    draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color, width=width)


def save_box_review(outlier: dict[str, Any]) -> str:
    image_path = IMAGE_DIR / outlier["image"]
    source = Image.open(image_path).convert("RGB")
    original_image = source.copy()
    clipped_image = source.copy()

    original_box = (
        outlier["original"]["xmin"],
        outlier["original"]["ymin"],
        outlier["original"]["xmax"],
        outlier["original"]["ymax"],
    )
    clipped_box = (
        outlier["clipped"]["xmin"],
        outlier["clipped"]["ymin"],
        outlier["clipped"]["xmax"],
        outlier["clipped"]["ymax"],
    )

    draw_box(original_image, original_box, (255, 0, 0), width=4)
    draw_box(clipped_image, clipped_box, (0, 255, 0), width=4)

    canvas_width = original_image.width + clipped_image.width + 30
    canvas_height = max(original_image.height, clipped_image.height) + 60
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    canvas.paste(original_image, (10, 30))
    canvas.paste(clipped_image, (original_image.width + 20, 30))

    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), f"Original: {outlier['image']}", fill=(0, 0, 0))
    draw.text((original_image.width + 20, 10), f"Clipped: {outlier['image']}", fill=(0, 0, 0))

    save_name = Path(outlier["image"]).stem + "_box_review.png"
    save_path = OUT_DIR / save_name
    canvas.save(save_path)
    return str(save_path)


def main() -> None:
    outliers = find_phone_box_outliers()
    report = {
        "PHONE BOX REVIEW": {
            "total_affected": len(outliers),
            "valid_after_clipping": sum(1 for item in outliers if item["valid_after_clipping"]),
            "invalid_after_clipping": sum(1 for item in outliers if not item["valid_after_clipping"]),
            "likely_safe_to_clip": sum(1 for item in outliers if item["valid_after_clipping"]),
            "requires_manual_review": sum(1 for item in outliers if not item["valid_after_clipping"]),
            "details": [
                {
                    "filename": item["image"],
                    "image_width": item["image_size"]["width"],
                    "image_height": item["image_size"]["height"],
                    "original_xmin": item["original"]["xmin"],
                    "original_ymin": item["original"]["ymin"],
                    "original_xmax": item["original"]["xmax"],
                    "original_ymax": item["original"]["ymax"],
                    "clipped_xmin": item["clipped"]["xmin"],
                    "clipped_ymin": item["clipped"]["ymin"],
                    "clipped_xmax": item["clipped"]["xmax"],
                    "clipped_ymax": item["clipped"]["ymax"],
                    "pixels_outside_each_boundary": item["outside_pixels"],
                    "clipped_box_width": item["clipped_box_width"],
                    "clipped_box_height": item["clipped_box_height"],
                    "valid_after_clipping": item["valid_after_clipping"],
                    "visualization": save_box_review(item),
                }
                for item in outliers
            ],
        }
    }

    phone_report_path = PROJECT_ROOT / "reports" / "phone_box_review.json"
    with phone_report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(json.dumps({
        "total_affected": len(outliers),
        "valid_after_clipping": sum(1 for item in outliers if item["valid_after_clipping"]),
        "invalid_after_clipping": sum(1 for item in outliers if not item["valid_after_clipping"]),
    }, indent=2))


if __name__ == "__main__":
    main()
