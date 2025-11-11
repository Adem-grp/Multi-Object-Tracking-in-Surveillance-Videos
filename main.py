from ultralytics import YOLO
from PIL import Image
import cv2
import os
import glob
from natsort import natsorted

# Load YOLO model
model = YOLO("yolov8n.pt")  # fastest version

# Collect all images
all_images = glob.glob(r"GenericMOT_JPEG_Sequence/*/*/*.jpg")

# Filter out corrupted images
valid_images = []
for img_path in all_images:
    try:
        Image.open(img_path)
        valid_images.append(img_path)
    except Exception:
        print(f"Skipping corrupted image: {img_path}")

print(f"Total valid images: {len(valid_images)}")

# Predict in batches to avoid "too many open files"
"""batch_size = 50  # adjust if needed
for i in range(0, len(valid_images), batch_size): # no need to eval more than once for now 
    batch = valid_images[i:i + batch_size]
    model.predict(source=batch, show=False, save=False)"""

base_path = "runs/detect"
folders = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.startswith("predict")]
latest_folder = max(folders, key=os.path.getctime)

print(f"Creating video from: {latest_folder}")

# Get all annotated images
images = [img for img in os.listdir(latest_folder) if img.endswith(".jpg")]
images = natsorted(images)

# Read first frame to get dimensions
first_frame = cv2.imread(os.path.join(latest_folder, images[0]))
height, width, _ = first_frame.shape

# Create video writer
output_path = os.path.join(latest_folder, "output_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

for img_name in images:
    frame = cv2.imread(os.path.join(latest_folder, img_name))
    out.write(frame)

out.release()
print(f"✅ Video saved at: {output_path}")
