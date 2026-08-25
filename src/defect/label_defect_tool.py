from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

WINDOW = "Defect labeling"
COLORS = {1: (0, 0, 255), 2: (0, 165, 255), 3: (255, 0, 0)}
NAMES = {1: "scratch", 2: "oil", 3: "stain"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def images_in(directory: Path) -> list[Path]:
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS], key=lambda p: p.name.lower()) if directory.is_dir() else []


def normalize(points: list[tuple[int, int]], width: int, height: int) -> str:
    values = [value for point in points for value in (point[0] / width, point[1] / height)]
    return " ".join(f"{value:.6f}" for value in values)


def save_shapes(label_path: Path, shapes: list[tuple[int, list[tuple[int, int]]]], width: int, height: int) -> None:
    lines = []
    for class_id, shape in shapes:
        if class_id == 1:
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.polylines(mask, [np.asarray(shape, np.int32)], False, 255, 5)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            shape = contours[0].reshape(-1, 2).tolist() if contours else shape
        lines.append(f"{class_id - 1} {normalize(shape, width, height)}")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def run(images_dir: Path, labels_dir: Path, split: str) -> None:
    images = images_in(images_dir)
    all_label_dirs = [labels_dir, labels_dir.parent / "train", labels_dir.parent / "test"]
    index = 0
    labeled = skipped = clean = 0
    try:
        while 0 <= index < len(images):
            image_path = images[index]
            label_path = labels_dir / f"{image_path.stem}.txt"
            if any((directory / label_path.name).exists() for directory in all_label_dirs):
                skipped += 1
                index += 1
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                skipped += 1
                index += 1
                continue
            height, width = image.shape[:2]
            points: list[tuple[int, int]] = []
            shapes: list[tuple[int, list[tuple[int, int]]]] = []
            active_class: int | None = None
            print(f"[{index + 1}/{len(images)}] {image_path.name}: 1=scratch 2=oil 3=stain c=clean")

            def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
                if event == cv2.EVENT_LBUTTONDOWN:
                    points.append((x, y))

            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, 640, 360)
            cv2.setMouseCallback(WINDOW, on_mouse)
            while True:
                canvas = image.copy()
                for class_id, shape in shapes:
                    cv2.polylines(canvas, [np.asarray(shape, np.int32)], True, COLORS[class_id], 2)
                if points:
                    cv2.polylines(canvas, [np.asarray(points, np.int32)], False, COLORS.get(active_class or 1, (255, 255, 255)), 2)
                    for point in points:
                        cv2.circle(canvas, point, 4, COLORS.get(active_class or 1, (255, 255, 255)), -1)
                class_text = f"class: {NAMES.get(active_class, 'none')}"
                cv2.setWindowTitle(WINDOW, f"{image_path.name} | {class_text} | points: {len(points)}")
                cv2.imshow(WINDOW, canvas)
                key = cv2.waitKey(20) & 0xFF
                if key in (ord("1"), ord("2"), ord("3")):
                    active_class = int(chr(key))
                elif key == 13:
                    if active_class is not None and len(points) >= 2:
                        shapes.append((active_class, points.copy()))
                        save_shapes(label_path, shapes, width, height)
                        points.clear()
                        active_class = None
                        print(f"  saved shape: {NAMES[shapes[-1][0]]}")
                elif key == ord("u"):
                    if points:
                        points.pop()
                elif key == ord("d"):
                    if shapes:
                        shapes.pop()
                        save_shapes(label_path, shapes, width, height)
                elif key == ord("n"):
                    if shapes:
                        save_shapes(label_path, shapes, width, height)
                        labeled += 1
                    else:
                        skipped += 1
                    index += 1
                    break
                elif key == ord("c"):
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    label_path.write_text("", encoding="utf-8")
                    clean += 1
                    index += 1
                    break
                elif key == ord("s"):
                    skipped += 1
                    index += 1
                    break
                elif key == ord("q"):
                    index = len(images)
                    break
    finally:
        cv2.destroyAllWindows()
    print(f"Final counts: labeled={labeled}, clean={clean}, skipped={skipped}, total={len(images)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Label defect shapes on normalized screen images.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    args = parser.parse_args()
    run(args.images_dir, args.labels_dir / args.split, args.split)


if __name__ == "__main__":
    main()
