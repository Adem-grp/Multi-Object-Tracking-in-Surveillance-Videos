"""
This script generates YOLO-format pseudo-labels for anomaly datasets
that do not provide the required object-detection annotations.

A COCO-pretrained YOLO11m model is used to generate detections for the
training and validation splits.

Detected objects are assigned classes according to the GMOT taxonomy and
saved as YOLO-format label files alongside copies of the images.

COCO classes are remapped to the corresponding GMOT classes using a predefined
mapping.

The test split is excluded from pseudo-labelling to prevent pseudo-labels
from being introduced into the evaluation set.
"""

import gc
import shutil
from pathlib import Path
from typing import Dict, List, Iterable
from ultralytics import YOLO

# The dataset that will be bridged to YOLO format. This should match the folder name in the datasets directory.
SOURCE_DATASET = "shanghaitech"

# CONFIG
SRC_ROOT = Path(r"C:/Users/USER/PycharmProjects/Multi-Object-Tracking-in-Surveillance-Videos"
                rf"/datasets/{SOURCE_DATASET}_yolo") # source
DST_ROOT = Path(r"C:/Users/USER/PycharmProjects/Multi-Object-Tracking-in-Surveillance-Videos"
                rf"/datasets/bridge_yolo/{SOURCE_DATASET}") # destination

MODEL = "yolo11m.pt"  # COCO-pretrained weights
CONF = 0.5  # Detection confidence threshold
IOU = 0.50  # IoU threshold used during non-maximum suppression (NMS) to filter overlapping boxes
IMGSZ = 640
BATCH = 32
DEVICE = "cuda"
WRITE_EMPTY = True  # write empty label files for images with no detections
OVERWRITE = False  # preserve existing outputs unless explicitly enabled

SPLITS = ["train", "val"]  # test intentionally skipped
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# limit memory and file-handle usage by processing images in chunks
PENDING_CHUNK = 128

# Class taxonomy used by the project
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
# Map COCO class names detected by the YOLO model to GMOT class IDs
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


# HELPERS
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def dir_has_images(dir_path: Path) -> bool:
    return dir_path.exists() and any(p.is_file() and p.suffix.lower() in IMG_EXTS for p in dir_path.iterdir())


def list_images(dir_path: Path) -> List[Path]:
    return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def yolo_label_path(dst_labels_dir: Path, image_name: str) -> Path:
    return dst_labels_dir / f"{Path(image_name).stem}.txt"


# Writes YOLO label lines depending on the empty-label and overwrite settings
def write_yolo_lines(lbl_path: Path, lines: List[str]):
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    if lbl_path.exists() and not OVERWRITE:
        return
    if not lines:
        if WRITE_EMPTY:
            lbl_path.write_text("", encoding="utf-8")
        else:
            if lbl_path.exists():
                lbl_path.unlink()
        return
    lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def chunked(iterable: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


# CORE
def process_split(model: YOLO, split: str):
    src_dir = SRC_ROOT / split / "images"
    dst_images = DST_ROOT / split / "images"
    dst_labels = DST_ROOT / split / "labels"
    ensure_dir(dst_images)
    ensure_dir(dst_labels)

    if not dir_has_images(src_dir):
        print(f"[WARN] No images found in: {src_dir}. Skipping {split}.")
        return

    # Select images that will require processing based on the overwrite setting and existing outputs
    pending = []
    for img_path in list_images(src_dir):
        out_img = dst_images / img_path.name
        lbl_path = yolo_label_path(dst_labels, img_path.name)
        if OVERWRITE or (not out_img.exists() or not lbl_path.exists()):
            pending.append(str(img_path))
    if not pending:
        print(f"[INFO] {split}: nothing to do (overwrite={OVERWRITE}).")
        return

    print(f"[INFO] {split}: running inference on {len(pending)} images in chunks of {PENDING_CHUNK}...")

    coco_names = model.names
    count_imgs = 0
    for batch_paths in chunked(pending, PENDING_CHUNK):
        # Detects objects in the images using the YOLO model
        results = model.predict(
            source=batch_paths,
            conf=CONF,
            iou=IOU,
            imgsz=IMGSZ,
            device=DEVICE,
            stream=False,
            batch=BATCH,
            verbose=False,
            save=False,
            workers=0,
        )
        for r, original_path in zip(results, batch_paths):
            src_img = Path(original_path).resolve()
            if not src_img.exists():
                continue

            count_imgs += 1
            # Copy the source image to the destination images directory if it doesn't already exist or if overwrite is enabled
            out_img = dst_images / src_img.name
            if not out_img.exists() or OVERWRITE:
                try:
                    shutil.copy2(src_img, out_img)
                except Exception as e:
                    print(f"[WARN] Copy failed {src_img} -> {out_img}: {e}")
                    continue

            lbl_path = yolo_label_path(dst_labels, src_img.name)
            if lbl_path.exists() and not OVERWRITE:
                continue

            boxes = getattr(r, "boxes", None)
            if boxes is None or boxes.cls is None:
                write_yolo_lines(lbl_path, [])
                continue

            # Extract the class IDs and bounding box coordinates from the detection results
            cls = boxes.cls.cpu().numpy()
            xywhn = getattr(boxes, "xywhn", None)
            xywh = getattr(boxes, "xywh", None)

            lines: List[str] = []
            if xywhn is not None:
                xywhn_np = xywhn.cpu().numpy()
                for k in range(len(xywhn_np)):
                    name = coco_names.get(int(cls[k]), None) # ignores COCO detections that have no corresponding GMOT class
                    g_id = COCO_TO_GMOT_ID.get(name, None)
                    if g_id is None:
                        continue
                    x_c, y_c, w, h = xywhn_np[k]
                    lines.append(f"{g_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
            elif xywh is not None: # fallback to absolute coordinates if normalized coordinates are not available
                xywh_np = xywh.cpu().numpy()
                H, W = r.orig_shape[:2]
                if W <= 0 or H <= 0:
                    write_yolo_lines(lbl_path, [])
                    continue
                for k in range(len(xywh_np)):
                    name = coco_names.get(int(cls[k]), None)
                    g_id = COCO_TO_GMOT_ID.get(name, None)
                    if g_id is None:
                        continue
                    x_c, y_c, w, h = xywh_np[k]
                    lines.append(f"{g_id} {x_c / W:.6f} {y_c / H:.6f} {w / W:.6f} {h / H:.6f}")
            else:
                write_yolo_lines(lbl_path, [])
                continue

            write_yolo_lines(lbl_path, lines)

        # Release batch results before processing the next chunk
        del results
        gc.collect()

    print(f"[OK] {split}: processed images: {count_imgs}")


# RUN
def main():
    print(repr(SRC_ROOT))
    print(SRC_ROOT.exists())

    print(f"{SOURCE_DATASET} is being bridged")
    print(f"Source root: {SRC_ROOT}")
    print(f"Output root: {DST_ROOT}")
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    model = YOLO(MODEL)
    for split in SPLITS:
        process_split(model, split)
    print(f"[DONE] {SOURCE_DATASET} Bridge written to: {DST_ROOT}")


if __name__ == "__main__":
    main()
