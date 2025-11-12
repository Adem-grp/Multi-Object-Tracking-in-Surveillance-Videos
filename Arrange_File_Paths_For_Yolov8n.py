import os
import shutil
import glob
import random
import pandas as pd
import cv2
from scipy.sparse.csgraph import depth_first_tree

gmot_path = "C:/Users/USER/PycharmProjects/Multi-Object-Tracking-in-Surveillance-Videos/GenericMOT_JPEG_Sequence"
yolo_path = "C:/Users/USER/PycharmProjects/Multi-Object-Tracking-in-Surveillance-Videos/gmot_yolo"

# creating folders
for split in ["train", "val"]:
    for sub in ["images", "labels"]:
        os.makedirs(os.path.join(yolo_path, split, sub), exist_ok=True)

#split ratio
s_ratio = 0.8
all_images = all_images = glob.glob(r"GenericMOT_JPEG_Sequence/*/*/*.jpg")
all_images.sort()
# splitting
random.shuffle(all_images)
split_idx = int(len(all_images) * s_ratio)
train_images = all_images[:split_idx]
valid_images = all_images[split_idx:]


def copy_images(images, spl1t):
    for image in images:
        f_name = os.path.basename(image)
        dst = os.path.join(yolo_path, spl1t, "images", f_name)
        shutil.copy(image, dst)


copy_images(train_images, "train")
copy_images(valid_images, "val")


def convert_gmot_to_yolo(img_root, label_root, out_root):
    os.makedirs(out_root, exist_ok=True)
    label_files = glob.glob(os.path.join(label_root, "*.txt"))
    for f in label_files:
        set_name = os.path.splitext(os.path.basename(f))[0]
        img_folder = os.path.join(img_root, set_name, "img1")
        if not os.path.exists(img_folder):
            print("Skipping {set_name}, image folder not found")
            continue
        df = pd.read_csv(f, header=None)
        df.columns = ["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"]

        # Group by frame for easier processing
        grouped = df.groupby("frame")
        images = glob.glob(os.path.join(img_folder, "*.jpg"))
        for img in images:
            img_name = os.path.basename(img)
            frame_no = int(os.path.splitext(img_name)[0])
            if frame_no not in grouped.groups:
                continue
            rows = grouped.get_group(frame_no)
            img = cv2.imread(img)
            H, W = img.shape[:2]
            yolo_lines = []
            for _, row in rows.iterrows():
                x_center = (row['x'] + row['w'] / 2) / W
                y_center = (row['y'] + row['h'] / 2) / H
                width = row['w'] / W
                height = row['h'] / H
                class_id = int(row['cls'])
                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            out_file=os.path.join(out_root,f"{set_name}_{img_name}.txt")
            with open(out_file, "w") as f:
                f.write("\n".join(yolo_lines))
