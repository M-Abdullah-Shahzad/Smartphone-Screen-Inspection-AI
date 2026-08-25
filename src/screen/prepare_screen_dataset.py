from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "processed" / "screen_detection"
TEST_IMAGES = PROJECT_ROOT / "runs" / "phone" / "crop_check"
TEST_LABELS = PROJECT_ROOT / "runs" / "phone" / "screen_labels"
OUTPUT_ROOT = PROJECT_ROOT / "processed" / "screen_detection_final"
CONFIG_PATH = PROJECT_ROOT / "configs" / "screen.yaml"
EXPECTED_COUNTS = {"train": 70, "val": 14, "test": 15}


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.name.lower(),
    )


def validate_label(label_path: Path, class_id: int) -> tuple[bool, str | None]:
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        return False, "label must contain exactly one non-empty line"
    values = lines[0].split()
    if len(values) != 9:
        return False, "label must contain class id plus 8 coordinates"
    try:
        if int(values[0]) != class_id:
            return False, f"class id is {values[0]}, expected {class_id}"
        coordinates = [float(value) for value in values[1:]]
    except ValueError:
        return False, "class id and coordinates must be numeric"
    if any(value < 0.0 or value > 1.0 for value in coordinates):
        return False, "coordinates must be in [0, 1]"
    return True, None


def source_pairs(split: str, class_id: int) -> list[tuple[Path, Path, str | None]]:
    image_dir = SOURCE_ROOT / "images" / split
    label_dir = SOURCE_ROOT / "labels" / split
    images = {path.stem: path for path in image_files(image_dir)}
    pairs: list[tuple[Path, Path, str | None]] = []
    for label_path in sorted(label_dir.glob("*.txt"), key=lambda path: path.name.lower()):
        image_path = images.get(label_path.stem)
        if image_path is None:
            pairs.append((Path(), label_path, "no matching image"))
            continue
        valid, reason = validate_label(label_path, class_id)
        pairs.append((image_path, label_path, reason if not valid else None))
    return pairs


def test_pairs(class_id: int) -> list[tuple[Path, Path, str | None]]:
    images = {path.stem: path for path in image_files(TEST_IMAGES) if not path.stem.startswith("crop_")}
    pairs: list[tuple[Path, Path, str | None]] = []
    for label_path in sorted(TEST_LABELS.glob("*.txt"), key=lambda path: path.name.lower()):
        if label_path.stem.startswith("crop_"):
            continue
        normalized_stem = label_path.stem.removesuffix("_screen")
        image_path = images.get(normalized_stem)
        if image_path is None:
            pairs.append((Path(), label_path, "no matching image after _screen normalization"))
            continue
        valid, reason = validate_label(label_path, class_id)
        pairs.append((image_path, label_path, reason if not valid else None))
    return pairs


def copy_split(split: str, pairs: list[tuple[Path, Path, str | None]], invalid: list[dict[str, str]]) -> int:
    image_output = OUTPUT_ROOT / "images" / split
    label_output = OUTPUT_ROOT / "labels" / split
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)
    copied = 0
    for image_path, label_path, reason in pairs:
        if reason is not None:
            invalid.append({"split": split, "image": str(image_path), "label": str(label_path), "reason": reason})
            continue
        destination_image = image_output / image_path.name
        destination_label = label_output / f"{image_path.stem}.txt"
        shutil.copy2(image_path, destination_image)
        shutil.copy2(label_path, destination_label)
        copied += 1
    return copied


def write_config() -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        "\n".join(
            [
                f"path: {OUTPUT_ROOT.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names: [screen]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble and validate the final YOLO screen dataset.")
    parser.add_argument("--class-id", type=int, default=0)
    args = parser.parse_args()

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    invalid: list[dict[str, str]] = []
    pairs_by_split = {
        "train": source_pairs("train", args.class_id),
        "val": source_pairs("val", args.class_id),
        "test": test_pairs(args.class_id),
    }
    counts = {split: copy_split(split, pairs, invalid) for split, pairs in pairs_by_split.items()}
    write_config()
    report = {
        "output_root": str(OUTPUT_ROOT.resolve()),
        "config": str(CONFIG_PATH),
        "counts": counts,
        "expected_counts": EXPECTED_COUNTS,
        "invalid_pairs": invalid,
        "status": "PASS" if counts == EXPECTED_COUNTS and not invalid else "FAIL",
    }
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise RuntimeError("Screen dataset preparation failed validation")


if __name__ == "__main__":
    main()
