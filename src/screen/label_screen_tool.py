from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
WINDOW_NAME = "Screen labeling"
DISPLAY_SIZE = 640


def list_images(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        return []
    return sorted(
        [path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.name.lower(),
    )


def normalize_points(points: list[tuple[int, int]], width: int, height: int) -> list[float]:
    normalized: list[float] = []
    for x, y in points:
        normalized.extend((x / width, y / height))
    return normalized


def save_label(label_path: Path, class_id: int, points: list[tuple[int, int]], width: int, height: int) -> None:
    values = normalize_points(points, width, height)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(
        f"{class_id} " + " ".join(f"{value:.6f}" for value in values) + "\n",
        encoding="utf-8",
    )


def run_labeler(images_dir: Path, labels_dir: Path, class_id: int) -> None:
    images = list_images(images_dir)
    labeled = 0
    skipped = 0
    index = 0

    try:
        while 0 <= index < len(images):
            image_path = images[index]
            label_path = labels_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                skipped += 1
                index += 1
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                print(f"[{index + 1}/{len(images)}] unreadable, skipped: {image_path.name}")
                skipped += 1
                index += 1
                continue
            height, width = image.shape[:2]
            scale = min(DISPLAY_SIZE / width, DISPLAY_SIZE / height)
            display_width = max(1, round(width * scale))
            display_height = max(1, round(height * scale))
            resized = cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_AREA)
            points: list[tuple[int, int]] = []

            def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
                if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                    points.append((x, y))

            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, DISPLAY_SIZE, DISPLAY_SIZE)
            cv2.setMouseCallback(WINDOW_NAME, on_mouse)
            print(f"[{index + 1}/{len(images)}] {image_path.name}: click TL, TR, BR, BL")

            while True:
                canvas = resized.copy()
                for x, y in points:
                    cv2.circle(canvas, (x, y), 4, (0, 255, 0), -1)
                if len(points) == 4:
                    cv2.polylines(canvas, [np.array(points, np.int32)], True, (0, 255, 0), 2)
                cv2.imshow(WINDOW_NAME, canvas)
                key = cv2.waitKey(20) & 0xFF
                if key in (13, 32):
                    if len(points) != 4:
                        print("  Need exactly 4 corners before saving.")
                        continue
                    original_points = [(round(x / scale), round(y / scale)) for x, y in points]
                    save_label(label_path, class_id, original_points, width, height)
                    labeled += 1
                    print(f"  saved: {label_path}")
                    index += 1
                    break
                if key == ord("r"):
                    points.clear()
                elif key == ord("s"):
                    skipped += 1
                    index += 1
                    break
                elif key == ord("p"):
                    index = max(0, index - 1)
                    break
                elif key == ord("q"):
                    index = len(images)
                    break
    finally:
        cv2.destroyAllWindows()
    print(f"Final counts: labeled={labeled}, skipped={skipped}, total={len(images)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manually label phone screen corners in YOLO segmentation format.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--class-id", type=int, default=0)
    args = parser.parse_args()
    run_labeler(args.images_dir, args.labels_dir, args.class_id)


if __name__ == "__main__":
    main()
