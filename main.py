import os
import glob
import cv2
import numpy as np
from natsort import natsorted
from ultralytics import YOLO
from PIL import Image
import torch
import time

# ---------------- CONFIG ----------------
YOLO_WEIGHTS = "runs/detect/train3/weights/best.pt"
IMAGE_GLOB = "GenericMOT_JPEG_Sequence/**/*.jpg"
OUTPUT_VIDEO = "gmot_iou_tracker.mp4"

CONF_THRES = 0.03
IMG_SIZE = 1280
IOU_THRESHOLD = 0.3  # match objects across frames
# ----------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(YOLO_WEIGHTS)
print("Device:", device)
print("YOLO Classes:", model.names)

# Load images
all_images = natsorted(glob.glob(IMAGE_GLOB, recursive=True))
valid_images = []

for p in all_images:
    try:
        Image.open(p).verify()
        valid_images.append(p)
    except:
        print("Skipping:", p)

if len(valid_images) == 0:
    raise RuntimeError("No images found!")

print("Total images:", len(valid_images))

# Prepare video
first = cv2.imread(valid_images[0])
H, W = first.shape[:2]
out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (W, H)
)

# ---------------- SIMPLE IOU TRACKER ----------------
next_track_id = 0
tracks = {}  # track_id -> last bbox

def compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])
    return inter / (areaA + areaB - inter + 1e-6)

# ----------------------------------------------------

for idx, path in enumerate(valid_images):
    frame = cv2.imread(path)

    # YOLO detection
    results = model.predict(frame, conf=CONF_THRES, imgsz=IMG_SIZE, verbose=False)[0]
    dets = results.boxes.data.cpu().numpy()  # x1,y1,x2,y2,conf,cls

    bboxes = []
    clss = []

    for x1, y1, x2, y2, conf, cls in dets:
        bboxes.append([float(x1), float(y1), float(x2), float(y2)])
        clss.append(int(cls))

    assigned = {}  # det index -> track_id
    used_tracks = set()

    # MATCH EXISTING TRACKS BY IOU
    for ti, t_bbox in tracks.items():
        best_iou = 0
        best_det = -1

        for di, d_bbox in enumerate(bboxes):
            if di in assigned:
                continue
            iou = compute_iou(t_bbox, d_bbox)
            if iou > best_iou:
                best_iou = iou
                best_det = di

        if best_iou >= IOU_THRESHOLD:
            assigned[best_det] = ti
            used_tracks.add(ti)
            tracks[ti] = bboxes[best_det]  # update track

    # NEW TRACKS FOR UNMATCHED DETECTIONS
    for di, d_bbox in enumerate(bboxes):
        if di not in assigned:
            assigned[di] = next_track_id
            tracks[next_track_id] = d_bbox
            next_track_id += 1

    # DRAW RESULTS
    for di, tid in assigned.items():
        x1, y1, x2, y2 = map(int, bboxes[di])
        cls_name = model.names[clss[di]]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{cls_name} ID:{tid}",
                    (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    print(f"Frame {idx+1}/{len(valid_images)}: Detections={len(bboxes)} Tracks={len(assigned)}")
    out.write(frame)

out.release()
print("Saved:", OUTPUT_VIDEO)
