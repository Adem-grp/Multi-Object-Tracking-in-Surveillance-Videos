import os
import shutil
import glob
import random
import pandas as pd

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

copy_images(train_images,"train")
copy_images(valid_images,"val")

