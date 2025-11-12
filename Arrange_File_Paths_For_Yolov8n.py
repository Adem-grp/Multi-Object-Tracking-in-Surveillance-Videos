import os
import shutil
import glob
import random
import pandas as pd
import cv2

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


def convert_mot_to_yolo(seq_folder, out_folder):
    """ seq=GMOT sequence folder
        out= yolo labels folder
    """
    gt_file = os.path.join(seq_folder, "gt", "gt.txt")
    if not os.path.exists(gt_file):
        return
    df = pd.read_csv(gt_file, header=None)
    df.columns = ["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"]
    grouped = df.groupby("frame")
    img_folder = os.path.join(seq_folder, "imag1")
    images = glob.glob(os.path.join(img_folder, "*.jpg"))
    for img in images:
        f_name = os.path.basename(img)
        frame_no = int(os.path.splitext(f_name)[0])
        dst = os.path.join(out_folder, f_name)
        img = cv2.imread(img)
        H, W = img.shape[:2]

        if frame_no not in grouped.groups:
            continue
        rows = grouped.get_group(frame_no)

        yolo_lines = []
        for _, row in rows.iterrows():
            x_center = (row["x"] + row["w"] / 2) / W
            y_center = (row["y"] + row["h"] / 2) / H
            width = row["w"] / W
            height = row["h"] / H
            class_id=int(row["cls"])
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            out_path= os.path.join(out_folder,os.path.splitext(f_name)[0]+".txt")
            with open(out_path, "w") as f:
                f.write("\n".join(yolo_lines))
