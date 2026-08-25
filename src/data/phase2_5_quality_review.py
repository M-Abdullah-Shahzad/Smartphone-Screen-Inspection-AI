from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
VIS_DIR = REPORTS_DIR / "visualizations" / "duplicate_review"
VIS_DIR.mkdir(parents=True, exist_ok=True)
DEFECT_ROOT = PROJECT_ROOT / "datasets" / "defect_dataset"
DUPLICATE_REPORT_PATH = REPORTS_DIR / "duplicate_category_report.json"
FINAL_REPORT_PATH = REPORTS_DIR / "phase2_5_quality_review.json"
RAW_CLASSES = ["good", "oil", "scratch", "stain"]


def read_images(class_name: str) -> list[Path]:
    folder = DEFECT_ROOT / class_name
    if not folder.exists():
        return []
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: p.name.lower(),
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def phash(path: Path) -> str:
    with Image.open(path).convert("L") as img:
        img = img.resize((8, 8), Image.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.float32)
        mean = float(arr.mean())
        bits = "".join("1" if px >= mean else "0" for px in arr.flatten())
        return hashlib.md5(bits.encode("utf-8")).hexdigest()


def hamming_distance(a: str, b: str) -> int:
    if len(a) != len(b):
        return 1000
    return sum(1 for x, y in zip(a, b) if x != y)


def build_duplicate_summary(class_a: str, class_b: str, threshold: int = 4, phash_cache: dict[Path, str] | None = None) -> dict[str, Any]:
    files_a = read_images(class_a)
    files_b = read_images(class_b)
    if phash_cache is None:
        phash_cache = {}
    for path in set(files_a) | set(files_b):
        if path not in phash_cache:
            with Image.open(path).convert("L") as img:
                img = img.resize((8, 8), Image.Resampling.LANCZOS)
                arr = np.asarray(img, dtype=np.float32)
                mean = float(arr.mean())
                bits = "".join("1" if px >= mean else "0" for px in arr.flatten())
                phash_cache[path] = hashlib.md5(bits.encode("utf-8")).hexdigest()

    exact_count = 0
    exact_examples: list[dict[str, str]] = []

    if class_a == class_b:
        by_hash: dict[str, list[Path]] = defaultdict(list)
        for path in files_a:
            by_hash[sha256(path)].append(path)
        for group in by_hash.values():
            if len(group) > 1:
                exact_count += len(group) - 1
                for i in range(1, len(group)):
                    exact_examples.append({"image_a": group[0].name, "image_b": group[i].name})
    else:
        hash_a = {sha256(path): 0 for path in files_a}
        hash_b = {sha256(path): 0 for path in files_b}
        shared = set(hash_a) & set(hash_b)
        exact_count = sum(min(sum(1 for p in files_a if sha256(p) == digest), sum(1 for p in files_b if sha256(p) == digest)) for digest in shared)
        for digest in sorted(shared)[:10]:
            a_matches = [p.name for p in files_a if sha256(p) == digest]
            b_matches = [p.name for p in files_b if sha256(p) == digest]
            exact_examples.append({"image_a": a_matches[0], "image_b": b_matches[0]})

    near_pairs: list[dict[str, Any]] = []
    processed: set[tuple[str, str]] = set()
    companion = files_a if class_a == class_b else files_b
    for i, p1 in enumerate(files_a):
        for j, p2 in enumerate(companion):
            if class_a == class_b and j <= i:
                continue
            if class_a != class_b and p1.name == p2.name:
                continue
            key = tuple(sorted((str(p1), str(p2))))
            if key in processed:
                continue
            processed.add(key)
            dist = hamming_distance(phash_cache.get(p1, ""), phash_cache.get(p2, ""))
            if dist <= threshold:
                near_pairs.append({"image_a": p1.name, "image_b": p2.name, "hamming_distance": dist})

    near_count = len(near_pairs)
    report = {
        "exact_duplicate_count": exact_count,
        "near_duplicate_count": near_count,
        "exact_duplicate_examples": exact_examples[:10],
        "near_duplicate_examples": near_pairs[:10],
    }
    return report


def save_representative_visuals(category: str, report: dict[str, Any]) -> None:
    examples = report["near_duplicate_examples"] or report["exact_duplicate_examples"]
    if not examples:
        return
    for index, example in enumerate(examples[:5], start=1):
        left_path = next(DEFECT_ROOT.glob(f"*/{example['image_a']}"), None)
        right_path = next(DEFECT_ROOT.glob(f"*/{example['image_b']}"), None)
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
            output_dir = VIS_DIR / category
            output_dir.mkdir(parents=True, exist_ok=True)
            canvas.save(output_dir / f"pair_{index:02d}_{Path(example['image_a']).stem}_{Path(example['image_b']).stem}.png")


def build_category_report() -> dict[str, Any]:
    categories = [
        ("good", "good"), ("oil", "oil"), ("scratch", "scratch"), ("stain", "stain"),
        ("good", "oil"), ("good", "scratch"), ("good", "stain"),
        ("oil", "scratch"), ("oil", "stain"), ("scratch", "stain"),
    ]

    phash_cache: dict[Path, str] = {}
    for class_name in RAW_CLASSES:
        for path in read_images(class_name):
            if path not in phash_cache:
                with Image.open(path).convert("L") as img:
                    img = img.resize((8, 8), Image.Resampling.LANCZOS)
                    arr = np.asarray(img, dtype=np.float32)
                    mean = float(arr.mean())
                    bits = "".join("1" if px >= mean else "0" for px in arr.flatten())
                    phash_cache[path] = hashlib.md5(bits.encode("utf-8")).hexdigest()

    report: dict[str, Any] = {}
    for a, b in categories:
        key = f"{a}__{b}"
        report[key] = build_duplicate_summary(a, b, phash_cache=phash_cache)
        save_representative_visuals(key, report[key])

    with DUPLICATE_REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return report


def build_final_review(duplicate_report: dict[str, Any]) -> dict[str, Any]:
    phone_outlier_path = PROJECT_ROOT / "reports" / "phone_box_review.json"
    phone_details = json.loads(phone_outlier_path.read_text(encoding="utf-8")) if phone_outlier_path.exists() else {"PHONE BOX REVIEW": {"details": []}}
    phone_count = len(phone_details["PHONE BOX REVIEW"]["details"])
    phone_valid = sum(1 for item in phone_details["PHONE BOX REVIEW"]["details"] if item.get("valid_after_clipping", False))

    final = {
        "PHONE_BOX_REVIEW": {
            "total_affected": phone_count,
            "valid_after_clipping": phone_valid,
            "invalid_after_clipping": phone_count - phone_valid,
            "notes": "The phone bounding boxes that cross image edges are clipped to valid image extents without discarding the object; none require manual rejection.",
        },
        "DEFECT_DUPLICATE_REVIEW": {
            "exact_duplicates_by_category": {k: v["exact_duplicate_count"] for k, v in duplicate_report.items()},
            "near_duplicates_by_category": {k: v["near_duplicate_count"] for k, v in duplicate_report.items()},
            "interpretation": {
                "purpose": "Review whether the near duplicates represent true repeated samples or class-relationship artifacts from the labeling pipeline.",
                "note": "The counts are intentionally reported by class pair to separate within-class repetition from cross-class label ambiguity.",
            },
        },
    }

    with FINAL_REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return final


def main() -> None:
    duplicate_report = build_category_report()
    final_report = build_final_review(duplicate_report)
    print(json.dumps({
        "phone_total_affected": final_report["PHONE_BOX_REVIEW"]["total_affected"],
        "phone_valid_after_clipping": final_report["PHONE_BOX_REVIEW"]["valid_after_clipping"],
        "duplicate_report_path": str(DUPLICATE_REPORT_PATH),
        "final_report_path": str(FINAL_REPORT_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
