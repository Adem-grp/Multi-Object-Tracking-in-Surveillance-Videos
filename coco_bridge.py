
# coco_bridge.py
"""
COCO-pretrained bridge -> YOLO dataset for GMOT taxonomy (train-first layout).

- Input: YOLO image folders under GMOT root:
    datasets/gmot_yolo/train/images
    datasets/gmot_yolo/val/images (optional)
- Output: bridge_yolo with the SAME train-first layout:
    datasets/bridge_yolo/train/images
    datasets/bridge_yolo/train/labels
    datasets/bridge_yolo/val/images   (if GMOT val exists)
    datasets/bridge_yolo/val/labels   (if GMOT val exists)
- Mapping: COCO classes -> GMOT IDs (cow/sheep -> stock=8)
- YAML: writes gmot_coco_bridge.yaml and includes only existing train/val paths
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, List

from PIL import Image
from ultralytics import YOLO


# ===================== GMOT taxonomy =====================
GMOT_NAMES = {
    0: "airplane",
    1: "fish",
    2: "ball",
    3: "bird",
    4: "boat",
    5: "balloon",
    6: "person",
    7: "insect",
    8: "stock",
    9: "car",
}

# ===================== COCO -> GMOT mapping =====================
COCO_TO_GMOT_ID: Dict[str, int] = {
    "airplane": 0,
    "sports ball": 2,
    "bird": 3,
    "boat": 4,
    "person": 6,
    "cow": 8,
    "sheep": 8,
    "car": 9,
}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def yolo_label_path(dst_labels_dir: Path, image_name: str) -> Path:
    """Make per-image label path: train-first layout keeps the same filename stem."""
    stem = Path(image_name).stem
    return dst_labels_dir / f"{stem}.txt"


def write_yolo_lines(lbl_path: Path, lines: List[str]):
    """Overwrite label file; remove stale file if no detections."""
    if not lines:
        if lbl_path.exists():
            lbl_path.unlink()
        return
    with lbl_path.open("w") as f:
        for line in lines:
            f.write(line + "\n")


def process_images(model: YOLO, src_dir: Path, dst_images_dir: Path, dst_labels_dir: Path, conf: float, iou: float):
    """Run COCO model on a YOLO images folder and write bridge labels/images (train-first layout)."""
    print(f"[INFO] Processing folder: {src_dir}")
    ensure_dir(dst_images_dir)
    ensure_dir(dst_labels_dir)

    results = model.predict(source=str(src_dir), conf=conf, iou=iou, stream=True)
    coco_names = model.names

    count_imgs = 0
    for r in results:
        img_path = Path(getattr(r, "path", ""))
        if not img_path.exists():
            continue

        count_imgs += 1
        # Copy image to bridge dataset (same filename)
        dst_img = dst_images_dir / img_path.name
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)

        # Image size
        with Image.open(dst_img) as im:
            W, H = im.size

        # Prepare label file
        dst_lbl = yolo_label_path(dst_labels_dir, img_path.name)

        boxes = r.boxes
        if boxes is None:
            write_yolo_lines(dst_lbl, [])
            continue

        import numpy as np
        xywh = boxes.xywh.cpu().numpy() if boxes.xywh is not None else np.empty((0, 4))
        cls = boxes.cls.cpu().numpy() if boxes.cls is not None else np.empty((0,))

        lines: List[str] = []
        for k in range(len(xywh)):
            name = coco_names.get(int(cls[k]), None)
            if name is None:
                continue
            g_id = COCO_TO_GMOT_ID.get(name, None)
            if g_id is None:
                continue

            x_c, y_c, w, h = xywh[k]  # center-format in pixels
            xc = float(x_c) / W
            yc = float(y_c) / H
            wn = float(w) / W
            hn = float(h) / H
            lines.append(f"{g_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        write_yolo_lines(dst_lbl, lines)

    print(f"[OK] Completed: {src_dir} | images processed: {count_imgs}")


def make_yaml(out_yaml: Path, gmot_root: Path, bridge_root: Path):
    """Write a combined YAML that points to train-first image folders and includes val only if present."""
    g_train = gmot_root / "train" / "images"
    b_train = bridge_root / "train" / "images"
    g_val = gmot_root / "val" / "images"
    b_val = bridge_root / "val" / "images"

    lines = [
        "# gmot_coco_bridge.yaml",
        "path: .",
        "train:",
        f"  - {g_train.as_posix()}",
        f"  - {b_train.as_posix()}",
        "val:",
    ]
    # Only include existing val paths
    if g_val.exists():
        lines.append(f"  - {g_val.as_posix()}")
    if b_val.exists():
        lines.append(f"  - {b_val.as_posix()}")

    lines += [
        "nc: 10",
        "names:",
        "  0: airplane",
        "  1: fish",
        "  2: ball",
        "  3: bird",
        "  4: boat",
        "  5: balloon",
        "  6: person",
        "  7: insect",
        "  8: stock",
        "  9: car",
    ]

    out_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Wrote YAML: {out_yaml}")


def main():
    ap = argparse.ArgumentParser(description="COCO bridge for YOLO-ready GMOT dataset (train-first layout)")
    ap.add_argument("--gmot-root", default="./datasets/gmot_yolo", help="Root of GMOT YOLO dataset (train-first)")
    ap.add_argument("--out",       default="./datasets/bridge_yolo", help="Output root for bridge dataset (train-first)")
    ap.add_argument("--splits",    nargs="+", default=["train", "val"], choices=["train", "val"],
                    help="Splits to process (default: both). Will skip missing ones gracefully.")
    ap.add_argument("--conf",      type=float, default=0.45, help="Confidence threshold for COCO model")
    ap.add_argument("--iou",       type=float, default=0.50, help="IoU threshold")
    ap.add_argument("--make-yaml", default="./datasets/gmot_coco_bridge.yaml", help="Path to write combined YAML")
    args = ap.parse_args()

    print("[INFO] Loading COCO-pretrained model: yolov8n.pt")
    model = YOLO("yolov8n.pt")

    gmot_root = Path(args.gmot_root)
    bridge_root = Path(args.out)

    # Process requested splits in train-first layout
    for split in args.splits:
        src_dir = gmot_root / split / "images"
        if not src_dir.exists():
            print(f"[WARN] GMOT {split} images not found: {src_dir}. Skipping {split}.")
            continue

        dst_images_dir = bridge_root / split / "images"
        dst_labels_dir = bridge_root / split / "labels"
        ensure_dir(dst_images_dir)
        ensure_dir(dst_labels_dir)

        process_images(model, src_dir, dst_images_dir, dst_labels_dir, args.conf, args.iou)

    # Write YAML including only existing paths
    make_yaml(Path(args.make_yaml), gmot_root, bridge_root)
    print(f"[DONE] Bridge dataset at: {bridge_root}")


if __name__ == "__main__":
    main()
