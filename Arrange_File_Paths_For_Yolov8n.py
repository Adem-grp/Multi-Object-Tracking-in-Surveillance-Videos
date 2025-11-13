import os
import shutil
import glob
import random
import pandas as pd
import cv2

gmot_path = "C:/Users/USER/PycharmProjects/Multi-Object-Tracking-in-Surveillance-Videos/GenericMOT_JPEG_Sequence"
yolo_path = "C:/Users/USER/PycharmProjects/Multi-Object-Tracking-in-Surveillance-Videos/gmot_yolo"
label_root = "C:/Users/USER/PycharmProjects/Multi-Object-Tracking-in-Surveillance-Videos/track_label"

# Create train/val folders
for split in ["train", "val"]:
    for sub in ["images", "labels"]:
        os.makedirs(os.path.join(yolo_path, split, sub), exist_ok=True)

# Split images
s_ratio = 0.8
all_images = glob.glob(os.path.join(gmot_path, "*", "*", "*.jpg"))
all_images.sort()
random.shuffle(all_images)
split_idx = int(len(all_images) * s_ratio)
train_images = all_images[:split_idx]
valid_images = all_images[split_idx:]

def copy_images(images, split):
    dst_folder = os.path.join(yolo_path, split, "images")
    for img in images:
        shutil.copy(img, dst_folder)

copy_images(train_images, "train")
copy_images(valid_images, "val")


def convert_gmot_to_yolo(img_root, label_root, out_root):
    os.makedirs(out_root, exist_ok=True)
    label_files = glob.glob(os.path.join(label_root, "*.txt"))

    for f in label_files:
        set_name = os.path.splitext(os.path.basename(f))[0]  # e.g., airplane-0
        class_name = set_name.split("-")[0]
        class_id = hash(class_name) % 1000  # numeric ID

        img_folder = os.path.join(img_root, set_name, "img1")
        if not os.path.exists(img_folder):
            continue

        df = pd.read_csv(f, header=None)
        df.columns = ["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis1", "vis2"]
        grouped = df.groupby("frame")
        images = sorted(glob.glob(os.path.join(img_folder, "*.jpg")))

        for img_path in images:
            img_name = os.path.basename(img_path)
            frame_no = int(os.path.splitext(img_name)[0])
            if frame_no not in grouped.groups:
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            H, W = img.shape[:2]
            rows = grouped.get_group(frame_no)
            yolo_lines = []

            for _, row in rows.iterrows():
                x_center = (row['x'] + row['w'] / 2) / W
                y_center = (row['y'] + row['h'] / 2) / H
                width = row['w'] / W
                height = row['h'] / H
                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            out_file = os.path.join(out_root, f"{set_name}_{os.path.splitext(img_name)[0]}.txt")

            with open(out_file, "w") as f_out:
                f_out.write("\n".join(yolo_lines))


# Convert all track labels to YOLO format
converted_labels_root = os.path.join(yolo_path, "labels_all")
convert_gmot_to_yolo(gmot_path, label_root, converted_labels_root)


def copy_labels(images, split):
    label_dest = os.path.join(yolo_path, split, "labels")
    os.makedirs(label_dest, exist_ok=True)

    for img_path in images:
        img_name = os.path.basename(img_path)
        set_name = img_path.split(os.sep)[-3]  # e.g., airplane-0
        label_file_name = f"{set_name}_{img_name.replace('.jpg','.txt')}"
        src_label_path = os.path.join(converted_labels_root, label_file_name)
        dst_label_path = os.path.join(label_dest, img_name.replace('.jpg','.txt'))

        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, dst_label_path)


copy_labels(train_images, "train")
copy_labels(valid_images, "val")
