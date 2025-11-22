import os
import glob
import shutil
import random
from tqdm import tqdm
import pandas as pd
import cv2

# ================= User paths =================
GMOT_ROOT = r"C:/Users/USER/PycharmProjects/Multi-Object-Tracking-in-Surveillance-Videos/GenericMOT_JPEG_Sequence"
TRACK_LABEL_ROOT = r"C:/Users/USER/PycharmProjects/Multi-Object-Tracking-in-Surveillance-Videos/track_label"
YOLO_ROOT = r"C:/Users/USER/PycharmProjects/Multi-Object-Tracking-in-Surveillance-Videos/datasets/gmot_yolo"

TRAIN_RATIO = 0.8  # 80/20 split

# Class mapping
CLASS_MAP = {
    "airplane": 0, "fish": 1, "ball": 2, "bird": 3,
    "boat": 4, "balloon": 5, "person": 6, "insect": 7,
    "stock": 8, "car": 9
}
CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}

# ================= Helpers =================
def safe_int(s):
    try:
        return int(s)
    except:
        return None

def yolo_bbox(x, y, w, h, img_w, img_h):
    x_c = (x + w / 2.0) / img_w
    y_c = (y + h / 2.0) / img_h
    w_n = w / img_w
    h_n = h / img_h
    return x_c, y_c, w_n, h_n

# ================= Main =================
labels_by_image = {}
all_images = []

folders = sorted(glob.glob(os.path.join(GMOT_ROOT, "*")))
print(f"Found {len(folders)} class folders")

for folder in tqdm(folders, desc="Processing class folders"):
    folder_name = os.path.basename(folder)
    class_name = folder_name.split("-")[0].lower()
    if class_name not in CLASS_MAP:
        print(f"Skipping unknown class folder: {folder_name}")
        continue
    class_id = CLASS_MAP[class_name]

    img_dir = os.path.join(folder, "img1")
    if not os.path.isdir(img_dir):
        continue
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not imgs:
        continue

    # Detect first image index
    first_basename = os.path.splitext(os.path.basename(imgs[0]))[0]
    first_index = safe_int(first_basename.lstrip("0")) if first_basename.lstrip("0") != "" else 0

    # Read label file
    label_file_candidates = glob.glob(os.path.join(TRACK_LABEL_ROOT, f"{folder_name}.txt"))
    if not label_file_candidates:
        print(f"No label file for {folder_name}, skipping...")
        continue
    label_file = label_file_candidates[0]

    try:
        df = pd.read_csv(label_file, header=None)
    except Exception as e:
        print(f"Failed reading {label_file}: {e}")
        continue

    if df.shape[1] < 7:
        print(f"Unexpected columns in {label_file}, skipping...")
        continue

    df.columns = list(range(df.shape[1]))

    # Iterate rows
    for idx, row in df.iterrows():
        frame = safe_int(row[0])
        x, y, w_box, h_box = float(row[2]), float(row[3]), float(row[4]), float(row[5])
        if frame is None or w_box <= 0 or h_box <= 0:
            continue

        # Map frame -> filename
        img_index = first_index + frame
        img_name = f"{img_index:06d}.jpg"
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            continue

        # Read image for size
        im = cv2.imread(img_path)
        if im is None:
            continue
        H, W = im.shape[:2]

        x_c, y_c, w_n, h_n = yolo_bbox(x, y, w_box, h_box, W, H)
        line = f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"

        labels_by_image.setdefault(img_path, []).append(line)
        all_images.append(img_path)

# Deduplicate
all_images = sorted(set(all_images))
if not all_images:
    raise SystemExit("No labeled images found. Check CSV mapping!")

# ================= Split train/val =================
random.shuffle(all_images)
split_idx = int(len(all_images) * TRAIN_RATIO)
train_list = all_images[:split_idx]
val_list = all_images[split_idx:]

# Create folders
for split, lst in [("train", train_list), ("val", val_list)]:
    img_out = os.path.join(YOLO_ROOT, split, "images")
    lbl_out = os.path.join(YOLO_ROOT, split, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    for img_path in tqdm(lst, desc=f"Writing {split}"):
        basename = os.path.basename(img_path)
        name_noext = os.path.splitext(basename)[0]
        dst_img = os.path.join(img_out, basename)
        dst_lbl = os.path.join(lbl_out, f"{name_noext}.txt")

        shutil.copy2(img_path, dst_img)
        lines = labels_by_image.get(img_path, [])
        with open(dst_lbl, "w") as f:
            f.write("\n".join(lines))

# ================= Write YAML =================
yaml_path = os.path.join(YOLO_ROOT, "gmot.yaml")
names_block = "\n".join([f"  {i}: {CLASS_NAMES[i]}" for i in sorted(CLASS_NAMES.keys())])
yaml_content = f"""# GMOT -> YOLO dataset config
path: {YOLO_ROOT}
train: {os.path.join(YOLO_ROOT, 'train', 'images')}
val: {os.path.join(YOLO_ROOT, 'val', 'images')}
nc: {len(CLASS_NAMES)}
names:
{names_block}
"""
with open(yaml_path, "w") as f:
    f.write(yaml_content)

# ================= Summary =================
print("\n========== SUMMARY ==========")
print(f"Total labeled images: {len(all_images)}")
print(f"Train set: {len(train_list)} images")
print(f"Val set:   {len(val_list)} images")
print(f"YAML saved to: {yaml_path}")
print("==============================")
print(f"You can now train YOLOv8 with:\n  yolo detect train data={yaml_path} model=yolov8n.pt epochs=50 imgsz=640")
