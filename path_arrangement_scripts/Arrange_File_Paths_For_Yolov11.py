import configparser
import os
import glob
import shutil
import random

from sqlalchemy.dialects.oracle.dictionary import all_sequences
from tqdm import tqdm
import pandas as pd
import cv2

# turning GMOT to yolo format
GMOT_ROOT = r"C:/Users/USER/PycharmProjects/Multi-Object-Tracking-in-Surveillance-Videos/GenericMOT_JPEG_Sequence"
TRACK_LABEL_ROOT = r"C:/Users/USER/PycharmProjects/Multi-Object-Tracking-in-Surveillance-Videos/track_label"
YOLO_ROOT = r"C:/Users/USER/PycharmProjects/Multi-Object-Tracking-in-Surveillance-Videos/datasets/gmot_yolo"

# ================= Config =================
RANDOM_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
AUTO_FRAME_OFFSET = True  # try both 0 and 1, pick the higher hit count

# ================= GMOT taxonomy =================
CLASS_MAP = {
    "airplane": 0, "fish": 1, "ball": 2, "bird": 3,
    "boat": 4, "balloon": 5, "person": 6, "insect": 7,
    "stock": 8, "car": 9
}
CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}

# Optional alias normalization if some folder names vary
ALIASES = {
    "plane": "airplane", "aeroplane": "airplane",
    "human": "person", "people": "person",
    "auto": "car", "vehicle": "car",
    # add more if needed
}


# ================= Helpers =================
def normalize_class_name(name: str) -> str:
    name = name.strip().lower()
    return ALIASES.get(name, name)


def safe_int(s):
    try:
        return int(s)
    except Exception:
        return None


def yolo_bbox(x, y, w, h, img_w, img_h):
    """Convert pixel bbox (top-left x,y,w,h) to YOLO normalized (xc,yc,w,h)."""
    x_c = (x + w / 2.0) / max(img_w, 1)
    y_c = (y + h / 2.0) / max(img_h, 1)
    w_n = w / max(img_w, 1)  # normalization is done with max(img_w/h,1)
    h_n = h / max(img_h, 1)
    return x_c, y_c, w_n, h_n


def detect_first_index(imgs):
    """Infer the first numeric index from the first filename."""
    if not imgs:
        return 0
    first_basename = os.path.splitext(os.path.basename(imgs[0]))[0]
    stripped = first_basename.lstrip("0")
    return safe_int(stripped) if stripped != "" else 0


def read_seq_dims(folder: str):  # create seqinfo.ini and read the width and height from there
    seq_info_path = os.path.join(folder, "seqinfo.ini")
    # if the file doesn't exist or is malformed cv2.imread is performed tho
    if not os.path.isfile(seq_info_path):
        return None
    cfg = configparser.ConfigParser()
    cfg.read(seq_info_path, encoding="utf-8")
    try:
        W = int(cfg["Sequence"]["imWidth"])
        H = int(cfg["Sequence"]["imHeight"])
        return W, H
    except(KeyError, ValueError):
        return None


def best_frame_offset(label_df, img_dir, first_index):
    """Try offsets 0 and 1; choose the one that maps to most existing image files."""
    if not AUTO_FRAME_OFFSET:
        return 0
    if label_df.empty:
        return 0

    frames = []
    for _, row in label_df.iterrows():
        f = safe_int(row[0])
        if f is not None:
            frames.append(f)
    if not frames:
        return 0

    def hit_count(offset):
        hits = 0
        for f in frames[:500]:  # limit cost
            img_index = (first_index or 0) + f + offset
            img_name = f"{img_index:06d}.jpg"
            img_path = os.path.join(img_dir, img_name)
            if os.path.exists(img_path):
                hits += 1
        return hits

    return 0 if hit_count(0) >= hit_count(1) else 1


# ================= Main =================
def main():
    os.makedirs(YOLO_ROOT, exist_ok=True)

    labels_by_image = {}  # key: original img_path, value: list of yolo lines
    image_sizes_cache = {}  # key: original img_path, value: (H,W)
    all_images = []  # unique original img paths that have at least one label
    seq_of_image = {}
    folders = sorted(glob.glob(os.path.join(GMOT_ROOT, "*")))
    print(f"Found {len(folders)} class/sequence folders under GMOT_ROOT")

    for folder in tqdm(folders, desc="Processing sequences"):
        folder_name = os.path.basename(folder)
        # Expect folder like 'car-02', class_name will be 'car'
        class_name = normalize_class_name(folder_name.split("-")[0])
        if class_name not in CLASS_MAP:
            print(f"[WARN] Skipping unknown class folder: {folder_name} (parsed '{class_name}')")
            continue
        class_id = CLASS_MAP[class_name]

        img_dir = os.path.join(folder, "img1")
        if not os.path.isdir(img_dir):
            continue
        imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        if not imgs:
            continue

        first_index = detect_first_index(imgs)

        # Read label file (assumes one text per sequence: '{folder_name}.txt')
        label_file_candidates = glob.glob(os.path.join(TRACK_LABEL_ROOT, f"{folder_name}.txt"))
        if not label_file_candidates:
            print(f"[INFO] No label file for {folder_name}, skipping...")
            continue
        label_file = label_file_candidates[0]

        try:
            df = pd.read_csv(label_file, header=None)
        except Exception as e:
            print(f"[WARN] Failed reading {label_file}: {e}")
            continue

        if df.shape[1] < 6:
            print(f"[WARN] Unexpected columns in {label_file} (got {df.shape[1]}), skipping...")
            continue
        df.columns = list(range(df.shape[1]))

        offset = best_frame_offset(df, img_dir, first_index)

        seq_dims = read_seq_dims(folder_name)

        # Iterate rows and convert to YOLO labels
        for _, row in df.iterrows():
            frame = safe_int(row[0])  # frame index within sequence
            try:
                x = float(row[2])
                y = float(row[3])
                w_box = float(row[4])
                h_box = float(row[5])
            except Exception:
                continue
            if frame is None or w_box <= 0 or h_box <= 0:
                continue

            # Map frame -> filename (handle index + offset)
            img_index = (first_index or 0) + frame + offset
            img_name = f"{img_index:06d}.jpg"
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                continue

            # Read image size once
            # determine image dimensions -fast path first seq dims
            # fast path uses the dimensions that has been already read from seqinfo.ini
            # all frames in one sequence share the same resolution
            if seq_dims is not None:
                W, H = seq_dims
            elif img_path in image_sizes_cache:
                H, W = image_sizes_cache[img_path]
            else:
                im = cv2.imread(img_path)
                if im is None:
                    continue
                H, W = im.shape[:2]
                image_sizes_cache[img_path] = (H, W)

            x_c, y_c, w_n, h_n = yolo_bbox(x, y, w_box, h_box, W, H)
            line = f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"

            labels_by_image.setdefault(img_path, []).append(line)
            all_images.append(img_path)
            seq_of_image[img_path] = folder_name

    # Deduplicate
    all_images = sorted(set(all_images))
    if not all_images:
        raise SystemExit("No labeled images found. Check CSV mapping and folder names!")

    # ================= Split train/val/test =================
    # fixed sequence-level split
    # splitting individual frames randomly caused data leakage.
    # frames from the same video that are 1-2 frames apart can end up in both train and val
    # now we collected the unique sequence names, shuffled and split them
    # then assign every frame to whatever split its parent sequence landed in
    # This will ensure that it is a proper split
    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-6:
        raise SystemExit("Train/Val/Test ratios must sum to 1.0")

    all_sequences_ = sorted(set(seq_of_image[p] for p in all_images))
    print(f"Found {len(all_sequences_)} sequences in {len(all_images)} images")

    random.seed(RANDOM_SEED)
    random.shuffle(all_sequences_)

    total = len(all_sequences_)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_seq = set(all_sequences_[:train_end])
    val_seq = set(all_sequences_[train_end:val_end])
    test_seq = set(all_sequences_[val_end:])

    train_list = [p for p in all_images if seq_of_image[p] in train_seq]
    val_list = [p for p in all_images if seq_of_image[p] in val_seq]
    test_list = [p for p in all_images if seq_of_image[p] in test_seq]

    assert len(train_list) + len(val_list) + len(test_list) == len(
        all_images), "Split counts do not sum up to total images!"

    splits = [("train", train_list), ("val", val_list), ("test", test_list)]

    # ================= Write images & labels =================
    for split, lst in splits:
        img_out = os.path.join(YOLO_ROOT, split, "images")
        lbl_out = os.path.join(YOLO_ROOT, split, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for img_path in tqdm(lst, desc=f"Writing {split}"):
            basename = os.path.basename(img_path)
            name_noext = os.path.splitext(basename)[0]
            # Avoid filename collisions across sequences: prefix with parent sequence folder
            parent_seq = seq_of_image[img_path]
            safe_prefix = parent_seq.replace(" ", "_")
            out_img_name = f"{safe_prefix}_{name_noext}.jpg"
            out_lbl_name = f"{safe_prefix}_{name_noext}.txt"

            dst_img = os.path.join(img_out, out_img_name)
            dst_lbl = os.path.join(lbl_out, out_lbl_name)

            shutil.copy2(img_path, dst_img)
            lines = labels_by_image.get(img_path, [])
            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

    # ================= Audit (ID consistency) =================
    allowed_ids = set(CLASS_NAMES.keys())
    violations = 0
    for split, lst in splits:
        lbl_out = os.path.join(YOLO_ROOT, split, "labels")
        for p in glob.glob(os.path.join(lbl_out, "*.txt")):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    cid = int(parts[0])
                    if cid not in allowed_ids:
                        violations += 1
                        print(f"[ID MISMATCH] {p}: {cid} not in {sorted(allowed_ids)}")
    if violations == 0:
        print("[AUDIT] All labels use GMOT IDs:", sorted(allowed_ids))
    else:
        print(f"[AUDIT] Found {violations} ID mismatches (see logs above).")

    # ================= Summary =================
    print("\n========== SUMMARY ==========")
    print(f"Total labeled images: {len(all_images)}")
    print(f"Train set: {len(train_list)} images")
    print(f"Val set:   {len(val_list)} images")
    print(f"Test set:  {len(test_list)} images")
    # yolo detect train data={yaml_path} model=yolov11n.pt epochs=50 imgsz=640 batch=16")


if __name__ == "__main__":
    main()
