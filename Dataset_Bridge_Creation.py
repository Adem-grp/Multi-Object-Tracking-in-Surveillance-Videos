"""
bridge.py
=========
Universal bridge script. Replaces both coco_bridge.py and avenue_bridge.py.

PURPOSE:
  You have datasets whose images have no bounding-box annotations (Avenue, etc.)
  or datasets where you want to supplement ground-truth with pseudo-labels
  (GMOT frames not covered by the track_label files).

  This script runs a COCO-pretrained YOLOv11 model on those images and converts
  the detections into YOLO-format label files using the GMOT class taxonomy.
  The result is placed in datasets/bridge_yolo/<dataset_name>/ so it can be
  listed alongside real labels in gmot.yaml.

HOW TO USE:
  Set SOURCE_DATASET to the name of your dataset (e.g., "avenue", "gmot").
  Set SRC_ROOT to the folder that contains train/images, val/images, etc.
  Run the script. Done.

WHY stream=False + chunking:
  stream=True returns a generator that keeps image file handles open until the
  generator is exhausted. On Windows this quickly hits the OS limit on open file
  descriptors when processing large directories. stream=False returns a complete
  list per batch, so Python can garbage-collect and close handles between chunks.
  PENDING_CHUNK controls how many images are passed to model.predict() at once.

Of course, the test split is intentionally excluded from SPLITS since we don't want to contaminate the test set with pseudo-labels. Only train and val are processed.
"""

import gc
import shutil
from pathlib import Path
from typing import Dict, List, Iterable
from ultralytics import YOLO

# Source dataset is used to specify the dataset being bridged of course name of the output folder needs to be used
SOURCE_DATASET = "avenue"

# ===================== CONFIG  =====================
SRC_ROOT = Path(r"C:/Users/USER/PycharmProjects/Multi-Object-Tracking-in-Surveillance-Videos"
                rf"/datasets/{SOURCE_DATASET}_yolo")
DST_ROOT = Path(r"C:/Users/USER/PycharmProjects/Multi-Object-Tracking-in-Surveillance-Videos"
                rF"/datasets/bridge_yolo/{SOURCE_DATASET}")

MODEL = "yolo11n.pt"  # COCO-pretrained weights
CONF = 0.5  # below this treshold is discarded for  higher precision, above for higher recall
IOU = 0.50  # NMS IoU threshold for suppressing overlapping boxes
IMGSZ = 640  # this is inference resolution matches YOLO training defaut
BATCH = 32  # first try 16 batch if computer can handle it move it to 32
DEVICE = "cuda"  # "", "cpu", "cuda", "0", "0,1"
WRITE_EMPTY = True  # write empty .txt when no detections # can check the failures and modify conf etc.
OVERWRITE = False  # skip existing outputs unless True

SPLITS = ["train", "val"]  # test intentionally skipped
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# controls how many image paths are passed to model.predict() at once smaller batches reduce memory
PENDING_CHUNK = 128

# ===================== GMOT taxonomy & mapping =====================
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

COCO_TO_GMOT_ID: Dict[str, int] = {
    "airplane": 0,
    "sports ball": 2,
    "bird": 3,
    "boat": 4,
    "person": 6,
    "cow": 8,
    "sheep": 8,
    "car": 9,
    # Optional extensions:
    # "bus": 9, "truck": 9, "motorcycle": 9, "bicycle": 9,
    # "frisbee": 2, "skateboard": 2,
}


# ===================== HELPERS =====================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def dir_has_images(dir_path: Path) -> bool:
    return dir_path.exists() and any(p.is_file() and p.suffix.lower() in IMG_EXTS for p in dir_path.iterdir())


def list_images(dir_path: Path) -> List[Path]:
    return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def yolo_label_path(dst_labels_dir: Path, image_name: str) -> Path:
    return dst_labels_dir / f"{Path(image_name).stem}.txt"


# Writes YOLO label lines to disk
# if lines are empty and Write empty is true writes an empty file
# if overwrite is false skips the file if exists
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


# ===================== CORE =====================
def process_split(model: YOLO, split: str):
    src_dir = SRC_ROOT / split / "images"
    dst_images = DST_ROOT / split / "images"
    dst_labels = DST_ROOT / split / "labels"
    ensure_dir(dst_images)
    ensure_dir(dst_labels)

    if not dir_has_images(src_dir):
        print(f"[WARN] No images found in: {src_dir}. Skipping {split}.")
        return

    # Collect pending images (skip ones already done unless OVERWRITE=True)
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
        # I need to use stream=False here to let Ultralytics close files per batch more promptly
        results = model.predict(
            source=batch_paths,
            conf=CONF,
            iou=IOU,
            imgsz=IMGSZ,
            device=DEVICE,
            stream=False,  # <-- batch returns a list; helps resource cleanup
            batch=BATCH,
            verbose=False,
            save=False,
            workers=0,  # fewer background workers -> fewer open handles on Windows
        )
        for r in results:
            img_path_str = getattr(r, "path", None)
            if not img_path_str:
                continue
            src_img = Path(img_path_str)
            if not src_img.exists():
                continue

            count_imgs += 1
            # for copying image to bridge destination
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

            # xywhn is basically bounding boxed that are normalised by image size
            cls = boxes.cls.cpu().numpy()
            xywhn = getattr(boxes, "xywhn", None)
            xywh = getattr(boxes, "xywh", None)

            lines: List[str] = []
            if xywhn is not None:
                xywhn_np = xywhn.cpu().numpy()
                for k in range(len(xywhn_np)):
                    name = coco_names.get(int(cls[k]), None)
                    g_id = COCO_TO_GMOT_ID.get(name, None)
                    if g_id is None:
                        continue
                    x_c, y_c, w, h = xywhn_np[k]
                    lines.append(f"{g_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
            elif xywh is not None:
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

        # Force cleanup between batches to release file handles
        del results
        gc.collect()

    print(f"[OK] {split}: processed images: {count_imgs}")


# ===================== RUN =====================
def main():
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
