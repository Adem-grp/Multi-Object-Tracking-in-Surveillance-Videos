import numpy as np
from ultralytics import YOLO
from PIL import Image
import cv2
import os
import glob
import torch
from natsort import natsorted
from deep_sort_pytorch.deep_sort import DeepSort
"""yolo detect train data="gmot.yaml" model=yolov8n.pt epochs=50 imgsz=640
"""
# --- Load models ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("runs/detect/train3/weights/best.pt")  # YOLOv8 small/faster version
deepsort = DeepSort("deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7")

# --- Collect images ---
all_images = glob.glob(r"GenericMOT_JPEG_Sequence/*/*/*.jpg")
valid_images = []
for img_path in all_images:
    try:
        Image.open(img_path)
        valid_images.append(img_path)
    except Exception:
        print(f"Skipping corrupted image: {img_path}")
valid_images = natsorted(valid_images)
print(f"Total valid images: {len(valid_images)}")

# --- Predict in batches to avoid too many open files ---
batch_size = 50
for i in range(0, len(valid_images), batch_size):
    batch = valid_images[i:i + batch_size]
    model.predict(source=batch, show=False, save=False)

# --- Get latest prediction folder ---
base_path = "runs/detect"
folders = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.startswith("predict")]
latest_folder = max(folders, key=os.path.getctime)
print(f"Creating video from: {latest_folder}")

# --- Video writer setup ---
images = natsorted([img for img in os.listdir(latest_folder) if img.endswith(".jpg")])
first_frame = cv2.imread(os.path.join(latest_folder, images[0]))
height, width, _ = first_frame.shape
output_path = os.path.join(latest_folder, "output_video_fine-tuned.mp4")
fourcc = 0x7634706d
fps = 30
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

for img_name in images:
    frame_path = os.path.join(latest_folder, img_name)
    frame = cv2.imread(frame_path)

    # --- YOLO predictions ---
    results = model.predict(source=frame, conf=0.2, show=False, save=False, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()

    bbox_xywh, confs, cls_ids = [], [], []
    for *xyxy, conf, cls in detections:
        # --- Filter by confidence threshold ---
        if conf < 0.3:  # change 0.5 to whatever threshold you want
            continue

        x1, y1, x2, y2 = xyxy
        w, h = x2 - x1, y2 - y1
        x_c, y_c = x1 + w / 2, y1 + h / 2
        bbox_xywh.append([x_c, y_c, w, h])
        confs.append(conf)
        cls_ids.append(int(cls))

    # --- Only run DeepSort if there are remaining high-confidence detections ---
    if len(bbox_xywh) > 0:
        bbox_xywh = np.array(bbox_xywh)
        confs = np.array(confs)
        cls_ids = np.array(cls_ids)

        # --- Run DeepSort ---
        outputs = deepsort.update(bbox_xywh, confs, cls_ids, frame)

        if outputs is not None and len(outputs) > 0:
            for output in outputs:
                output = np.array(output).flatten()
                if len(output) != 6:
                    print("Skipping invalid output:", output)
                    continue

                x1, y1, x2, y2, track_id, cls_id = [float(o) for o in output]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {int(track_id)}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            print(f"No tracks in frame {img_name}")
    else:
        print(f"No detections in frame {img_name}")

    # --- Write frame to video ---
    out.write(frame)


out.release()
print(f"Saved video to: {output_path}")
