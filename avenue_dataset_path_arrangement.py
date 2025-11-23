
import os
import glob
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
from scipy.io import loadmat

# =============== USER PATHS ===============
AVENUE_ROOT = r"C:/Users/USER/PycharmProjects/Multi-Object-Tracking-in-Surveillance-Videos/Avenue Dataset"
GT_ROOT     = r"C:/Users/USER/PycharmProjects/Multi-Object-Tracking-in-Surveillance-Videos/Avenue-Ground-Truth"
TEST_VIDEOS_DIR       = os.path.join(AVENUE_ROOT, "testing_videos")      # *.avi
TEST_LABEL_MASK_DIR   = os.path.join(GT_ROOT,    "testing_label_mask")   # X_label.mat (rects or masks)
TEST_VOL_DIR          = os.path.join(AVENUE_ROOT, "testing_vol")         # volXX.mat (mask volume), optional

YOLO_ROOT   = r"C:/Users/USER/PycharmProjects/Multi-Object-Tracking-in-Surveillance-Videos/avenue_yolo"
TRAIN_RATIO = 0.8  # split by video

# If you want to add normal training videos as background (empty labels), set True
ADD_BACKGROUND_TRAIN = True
TRAIN_VIDEOS_DIR     = os.path.join(AVENUE_ROOT, "training_videos")

# Single-class anomaly detection (you can expand later if you want multi-class)
CLASS_NAMES = ["anomaly"]

# =============== HELPERS ===============
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def yolo_from_xyxy(xmin, ymin, xmax, ymax, W, H):
    w = max(0.0, xmax - xmin)
    h = max(0.0, ymax - ymin)
    x_c = (xmin + w / 2.0) / W
    y_c = (ymin + h / 2.0) / H
    return x_c, y_c, w / W, h / H

def extract_frames(video_path: str, out_dir: str, prefix: str):
    """Extract frames and return (count, W, H)."""
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    count, W, H = 0, None, None
    while True:
        ok, frame = cap.read()
        if not ok: break
        if W is None:
            H, W = frame.shape[:2]
        fname = f"{prefix}_frame_{count+1:05d}.jpg"
        cv2.imwrite(os.path.join(out_dir, fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        count += 1
    cap.release()
    return count, W, H

def load_rects_from_mat(label_mat_path: str):
    """
    Return: (rects_per_frame, native_size)
    - rects_per_frame: dict 1-based frame -> list of (x,y,w,h) rectangles
    - native_size: (Hm, Wm) if rectangles/masks are stored in a low-res grid
                   None if already in frame coordinates
    Supports 'volLabel' (3D ndarray or object-array) and common MOT-like tabular keys.
    """
    rects = {}
    native_size = None

    if not os.path.exists(label_mat_path):
        print(f"[info] no label mat: {label_mat_path}")
        return rects, native_size

    data = loadmat(label_mat_path, squeeze_me=True, struct_as_record=False)
    keys = [k for k in data.keys() if not k.startswith('__')]

    # --- Case A: volLabel holds per-frame masks we can convert to rectangles ---
    if 'volLabel' in data:
        vl = data['volLabel']
        # Numeric 3D volume (H, W, N)
        if isinstance(vl, np.ndarray) and vl.dtype != np.object_ and vl.ndim == 3:
            Hm, Wm, N = vl.shape
            native_size = (Hm, Wm)
            for i in range(N):
                m = np.array(vl[..., i])
                if np.any(m):
                    ys, xs = np.where(m > 0)
                    x0, y0 = int(xs.min()), int(ys.min())
                    x1, y1 = int(xs.max()), int(ys.max())
                    rects.setdefault(i+1, []).append((x0, y0, x1 - x0 + 1, y1 - y0 + 1))
            return rects, native_size

        # Object-array of per-frame masks
        if isinstance(vl, np.ndarray) and vl.dtype == np.object_:
            rects = {}
            Hm = Wm = None
            flat = vl.ravel()
            for i, cell in enumerate(flat):
                arr = np.array(cell)
                if arr.ndim >= 2 and np.any(arr):
                    ys, xs = np.where(arr > 0)
                    x0, y0 = int(xs.min()), int(ys.min())
                    x1, y1 = int(xs.max()), int(ys.max())
                    rects.setdefault(i+1, []).append((x0, y0, x1 - x0 + 1, y1 - y0 + 1))
                    Hm, Wm = arr.shape[:2]
            if Hm and Wm:
                native_size = (Hm, Wm)
            return rects, native_size

        # Struct-like with .vol
        if hasattr(vl, 'vol'):
            arr = np.array(vl.vol)
            if arr.ndim == 3:
                Hm, Wm, N = arr.shape
                native_size = (Hm, Wm)
                for i in range(N):
                    m = arr[..., i]
                    if np.any(m):
                        ys, xs = np.where(m > 0)
                        x0, y0 = int(xs.min()), int(ys.min())
                        x1, y1 = int(xs.max()), int(ys.max())
                        rects.setdefault(i+1, []).append((x0, y0, x1 - x0 + 1, y1 - y0 + 1))
                return rects, native_size

    # --- Case B: tabular rectangles [frame, x, y, w, h, ...] ---
    for key in ["gt", "rect", "boxes", "labels"]:
        if key in data:
            arr = np.array(data[key]).squeeze()
            if arr.ndim == 2 and arr.shape[1] >= 5:
                for row in arr:
                    fid = int(row[0])
                    x, y, w, h = map(float, row[1:5])
                    rects.setdefault(fid, []).append((int(x), int(y), int(w), int(h)))
                native_size = None  # assume frame-space
                return rects, native_size

    print(f"[warn] No rectangles found in {label_mat_path}. Keys: {keys}")
    return rects, native_size

def write_yolo_label(lines, label_path):
    ensure_dir(Path(label_path).parent)
    with open(label_path, "w") as f:
        f.write("\n".join(lines))

# =============== MAIN ===============
def arrange_avenue_to_yolo():
    # Discover test videos
    test_videos = sorted(glob.glob(os.path.join(TEST_VIDEOS_DIR, "*.avi")))
    if not test_videos:
        raise SystemExit("No testing videos found.")

    # Split by videos
    split_idx = int(len(test_videos) * TRAIN_RATIO)
    train_vids = test_videos[:split_idx]
    val_vids   = test_videos[split_idx:]

    out_images_train = os.path.join(YOLO_ROOT, "images", "train")
    out_images_val   = os.path.join(YOLO_ROOT, "images", "val")
    out_labels_train = os.path.join(YOLO_ROOT, "labels", "train")
    out_labels_val   = os.path.join(YOLO_ROOT, "labels", "val")
    ensure_dir(out_images_train); ensure_dir(out_images_val)
    ensure_dir(out_labels_train); ensure_dir(out_labels_val)

    def process_split(split_name, vids):
        img_out = out_images_train if split_name == "train" else out_images_val
        lbl_out = out_labels_train if split_name == "train" else out_labels_val

        for vid in tqdm(vids, desc=f"Extract+Label ({split_name})"):
            vid_id_str = Path(vid).stem  # e.g., '01'
            # Mat paths
            label_mat_path = os.path.join(TEST_LABEL_MASK_DIR, f"{int(vid_id_str)}_label.mat")
            vol_path       = os.path.join(TEST_VOL_DIR, f"vol{int(vid_id_str):02d}.mat")

            # Extract frames
            prefix = f"test{int(vid_id_str):02d}"
            frame_count, W, H = extract_frames(vid, img_out, prefix)

            # Load rectangles if available
            rects_per_frame, native_size = load_rects_from_mat(label_mat_path)
            Wm = Hm = None
            if native_size is not None:
                Hm, Wm = native_size

            # Optional: load mask volume if no rectangles
            vol = None
            if (not rects_per_frame) and os.path.exists(vol_path):
                data = loadmat(vol_path)
                # find a 3D volume
                for k, v in data.items():
                    if k.startswith("__"): continue
                    vnp = np.array(v)
                    if vnp.ndim == 3 and vnp.dtype != np.object_:
                        vol = vnp
                        break

            # Per frame label writing
            for i in range(frame_count):
                fname = f"{prefix}_frame_{i+1:05d}"
                lbl_path = os.path.join(lbl_out, f"{fname}.txt")
                lines = []

                if (i+1) in rects_per_frame and len(rects_per_frame[i+1]) > 0:
                    # Rectangles: scale if needed (from native mask grid to frame size)
                    for (rx, ry, rw, rh) in rects_per_frame[i+1]:
                        if (Wm is not None) and (Hm is not None):
                            sx = W / float(Wm)
                            sy = H / float(Hm)
                            xmin = int(rx * sx)
                            ymin = int(ry * sy)
                            xmax = int((rx + rw) * sx)
                            ymax = int((ry + rh) * sy)
                        else:
                            xmin = int(rx); ymin = int(ry)
                            xmax = int(rx + rw); ymax = int(ry + rh)

                        # clamp
                        xmin = max(0, min(xmin, W-1)); ymin = max(0, min(ymin, H-1))
                        xmax = max(0, min(xmax, W-1)); ymax = max(0, min(ymax, H-1))
                        # skip degenerate
                        if xmax <= xmin or ymax <= ymin: continue

                        x_c, y_c, w_n, h_n = yolo_from_xyxy(xmin, ymin, xmax, ymax, W, H)
                        lines.append(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

                elif vol is not None and i < vol.shape[2]:
                    # Fallback: per-frame mask -> connected components -> boxes
                    mask = np.array(vol[..., i])
                    # robust binarization
                    m_u8 = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    _, m_bin = cv2.threshold(m_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    m_bin = cv2.resize(m_bin, (W, H), interpolation=cv2.INTER_NEAREST)
                    # opening + closing to reduce edge artifacts
                    kernel = np.ones((3,3), np.uint8)
                    m_bin = cv2.morphologyEx(m_bin, cv2.MORPH_OPEN, kernel)
                    m_bin = cv2.morphologyEx(m_bin, cv2.MORPH_CLOSE, kernel)
                    # contours
                    contours, _ = cv2.findContours((m_bin > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    min_area = int(0.001 * W * H)
                    for c in contours:
                        x, y, w, h = cv2.boundingRect(c)
                        if w*h >= min_area and (w*h) < int(0.95 * W * H):
                            x_c, y_c, w_n, h_n = yolo_from_xyxy(x, y, x+w, y+h, W, H)
                            lines.append(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

                # Always write file (empty -> no objects)
                write_yolo_label(lines, lbl_path)

    # Process splits
    process_split("train", train_vids)
    process_split("val",   val_vids)

    # Optional: add normal training videos as background in TRAIN only (empty labels)
    if ADD_BACKGROUND_TRAIN and os.path.isdir(TRAIN_VIDEOS_DIR):
        for vid in tqdm(sorted(glob.glob(os.path.join(TRAIN_VIDEOS_DIR, "*.avi"))),
                        desc="Extract normal training videos (background)"):
            prefix = f"trainbg{Path(vid).stem}"
            frame_count, Wbg, Hbg = extract_frames(vid, out_images_train, prefix)
            for i in range(frame_count):
                fname = f"{prefix}_frame_{i+1:05d}"
                lbl_path = os.path.join(out_labels_train, f"{fname}.txt")
                write_yolo_label([], lbl_path)  # empty: no objects

    # Write YAML
    yaml_path = os.path.join(YOLO_ROOT, "avenue.yaml")
    yaml_content = f"""# Avenue -> YOLO dataset config
path: {YOLO_ROOT}
train: {os.path.join(YOLO_ROOT, 'images', 'train')}
val:   {os.path.join(YOLO_ROOT, 'images', 'val')}
nc: {len(CLASS_NAMES)}
names:
  0: {CLASS_NAMES[0]}
"""
    ensure_dir(Path(yaml_path).parent)
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    # Summary
    print("\n========== SUMMARY ==========")
    print("Train images:", len(list(Path(os.path.join(YOLO_ROOT, 'images', 'train')).glob('*.jpg'))))
    print("Train labels:", len(list(Path(os.path.join(YOLO_ROOT, 'labels', 'train')).glob('*.txt'))))
    print("Val images:",   len(list(Path(os.path.join(YOLO_ROOT, 'images', 'val')).glob('*.jpg'))))
    print("Val labels:",   len(list(Path(os.path.join(YOLO_ROOT, 'labels', 'val')).glob('*.txt'))))
    print("YAML saved to:", yaml_path)
    print("==============================")
    print(f"You can now train YOLOv8 with:\n yolo detect train data={yaml_path} model=yolov8n.pt epochs=50 imgsz=640")

if __name__ == "__main__":
    arrange_avenue_to_yolo()
