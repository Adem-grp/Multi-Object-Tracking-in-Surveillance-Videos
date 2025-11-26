
# gmot_bridge.py
"""
GMOT -> Bridge (COCO-pretrained) with GMOT taxonomy (train/val only).

- Input:
    datasets/gmot_yolo/<split>/images      # GMOT already arranged in YOLO format
- Output:
    datasets/bridge_yolo/gmot/<split>/images
    datasets/bridge_yolo/gmot/<split>/labels

- Mapping:
    COCO classes -> GMOT IDs (e.g., cow/sheep -> stock=8, person -> 6, car -> 9)

- IMPORTANT:
    * GMOT ground truths are NOT modified.
    * This script does NOT write a training YAML. Use the unified YAML you provided.
    * 'test' is skipped by design (bridge is only for train/val).
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, List

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
# Minimal mapping per your spec; you can extend this safely (examples commented).
COCO_TO_GMOT_ID: Dict[str, int] = {
    "airplane": 0,
    "sports ball": 2,
    "bird": 3,
    "boat": 4,
    "person": 6,
    "cow": 8,
    "sheep": 8,
    "car": 9,
    # Optional reasonable extensions (uncomment if desired):
    # "bus": 9,
    # "truck": 9,
    # "motorcycle": 9,
    # "bicycle": 9,
    # "frisbee": 2,
    # "skateboard": 2,
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def yolo_label_path(dst_labels_dir: Path, image_name: str) -> Path:
    return dst_labels_dir / f"{Path(image_name).stem}.txt"


def write_yolo_lines(lbl_path: Path, lines: List[str], write_empty: bool = True):
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    if not lines:
        if write_empty:
            lbl_path.write_text("", encoding="utf-8")
        else:
            if lbl_path.exists():
                lbl_path.unlink()
        return
    lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def has_images(dir_path: Path) -> bool:
    return dir_path.exists() and any(
        p.is_file() and p.suffix.lower() in IMG_EXTS for p in dir_path.iterdir()
    )


def process_images(
    model: YOLO,
    src_dir: Path,
    dst_images_dir: Path,
    dst_labels_dir: Path,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    batch: int,
    write_empty: bool,
):
    if not has_images(src_dir):
        print(f"[WARN] No images found in: {src_dir}. Skipping.")
        return

    print(f"[INFO] Processing: {src_dir}")
    ensure_dir(dst_images_dir)
    ensure_dir(dst_labels_dir)

    # Run inference using the COCO-pretrained weights
    results = model.predict(
        source=str(src_dir),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        stream=True,        # generator
        batch=batch,        # speed up
        verbose=False,
        save=False,
    )
    coco_names = model.names  # dict: idx -> name

    count_imgs = 0
    for r in results:
        # r.path is the original input image path
        img_path_str = getattr(r, "path", None)
        if not img_path_str:
            continue
        img_path = Path(img_path_str)
        if not img_path.exists():
            continue

        count_imgs += 1

        # Copy image to bridge destination
        out_img = dst_images_dir / img_path.name
        if not out_img.exists():
            try:
                shutil.copy2(img_path, out_img)
            except Exception as e:
                print(f"[WARN] Copy failed {img_path} -> {out_img}: {e}")
                continue

        # Prepare label path
        lbl_path = yolo_label_path(dst_labels_dir, img_path.name)

        boxes = getattr(r, "boxes", None)
        if boxes is None or boxes.cls is None:
            write_yolo_lines(lbl_path, [], write_empty)
            continue

        # Prefer normalized boxes directly (if available in your Ultralytics version)
        xywhn = getattr(boxes, "xywhn", None)
        xywh = getattr(boxes, "xywh", None)
        cls = boxes.cls.cpu().numpy()

        lines: List[str] = []
        if xywhn is not None:
            # Already normalized
            xywhn_np = xywhn.cpu().numpy()
            for k in range(len(xywhn_np)):
                name = coco_names.get(int(cls[k]), None)
                if name is None:
                    continue
                g_id = COCO_TO_GMOT_ID.get(name, None)
                if g_id is None:
                    continue
                x_c, y_c, w, h = xywhn_np[k]
                lines.append(f"{g_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        elif xywh is not None:
            # Normalize using original image size from r.orig_shape
            xywh_np = xywh.cpu().numpy()
            H, W = r.orig_shape[:2]
            if W <= 0 or H <= 0:
                write_yolo_lines(lbl_path, [], write_empty)
                continue
            for k in range(len(xywh_np)):
                name = coco_names.get(int(cls[k]), None)
                if name is None:
                    continue
                g_id = COCO_TO_GMOT_ID.get(name, None)
                if g_id is None:
                    continue
                x_c, y_c, w, h = xywh_np[k]
                lines.append(f"{g_id} {x_c / W:.6f} {y_c / H:.6f} {w / W:.6f} {h / H:.6f}")
        else:
            write_yolo_lines(lbl_path, [], write_empty)
            continue

        write_yolo_lines(lbl_path, lines, write_empty)

    print(f"[OK] Completed: {src_dir} | images processed: {count_imgs}")


def main():
    ap = argparse.ArgumentParser(description="GMOT -> Bridge using COCO-pretrained YOLOv8 mapped to GMOT taxonomy")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to process (test ignored)")
    ap.add_argument("--conf", type=float, default=0.45, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.50, help="IoU threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    ap.add_argument("--batch", type=int, default=16, help="Batch size for inference")
    ap.add_argument("--device", type=str, default="", help="Device: 'cpu', 'cuda', '0', '0,1', or '' for auto")
    ap.add_argument("--model", type=str, default="yolov8s.pt", help="COCO-pretrained YOLOv8 model weights")
    ap.add_argument("--src-root", type=str, default="datasets/gmot_yolo", help="Source GMOT YOLO root")
    ap.add_argument("--dst-root", type=str, default="datasets/bridge_yolo/gmot", help="Destination bridge root")
    ap.add_argument("--write-empty", action="store_true", help="Write empty .txt when no detections (recommended)")
    args = ap.parse_args()

    # Load COCO-pretrained weights (no COCO dataset download required)
    model = YOLO(args.model)

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    ensure_dir(dst_root)

    for split in args.splits:
        if split.lower() == "test":
            print("[INFO] Skipping test split for bridge.")
            continue
        src_dir = src_root / split / "images"
        dst_images = dst_root / split / "images"
        dst_labels = dst_root / split / "labels"
        process_images(
            model,
            src_dir,
            dst_images,
            dst_labels,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            batch=args.batch,
            write_empty=args.write_empty,
        )

    print("[DONE] GMOT Bridge written to:", dst_root)


if __name__ == "__main__":
    main()
