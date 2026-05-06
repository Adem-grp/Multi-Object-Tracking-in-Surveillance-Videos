import sys
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET

# =========================
# PATHS
# =========================
BASE = Path(r"/")

VIDEO = BASE / "cvat_inputs" / "avenue" / "avenue_01.mp4"
DET_DIR = BASE / "cvat_inputs" / "avenue_ocsort_detections"
OUT_XML = BASE / "cvat_inputs" / "avenue_gt" / "annotations.xml"

OUT_XML.parent.mkdir(parents=True, exist_ok=True)

# =========================
# IMPORT OC-SORT
# =========================
sys.path.append(str(BASE / "OC_SORT"))
from trackers.ocsort_tracker.ocsort import OCSort

# =========================
# VIDEO INFO
# =========================
cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    raise RuntimeError("Cannot open video")

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
NUM_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

img_info = (H, W)
img_size = (H, W)

# =========================
# TRACKER
# =========================
tracker = OCSort(
    det_thresh=0.5,   # strict threshold
    max_age=15,
    min_hits=5,
    iou_threshold=0.3
)

# =========================
# RUN TRACKER (EVERY FRAME)
# =========================
tracks_by_id = defaultdict(list)

for frame_idx in range(NUM_FRAMES):
    det_file = DET_DIR / f"{frame_idx:06d}.txt"
    detections = []

    if det_file.exists() and det_file.stat().st_size > 0:
        dets = np.loadtxt(det_file, delimiter=",")
        if dets.ndim == 1:
            dets = dets[None, :]

        for x1, y1, x2, y2, _ in dets:
            w = x2 - x1
            h = y2 - y1

            # ✅ CRITICAL FIX:
            # score MUST be strictly greater than det_thresh
            score = 0.9

            detections.append([x1, y1, w, h, score])

    dets_np = np.array(detections, dtype=float) if detections else np.empty((0, 5))
    tracks = tracker.update(dets_np, img_info, img_size)

    for x, y, w, h, tid in tracks:
        tracks_by_id[int(tid)].append(
            (frame_idx, x, y, x + w, y + h)
        )

print(f"[DEBUG] Number of tracks produced: {len(tracks_by_id)}")

# =========================
# BUILD CVAT 1.1 XML
# =========================
root = ET.Element("annotations")

meta = ET.SubElement(root, "meta")
task = ET.SubElement(meta, "task")

# ---- REQUIRED: segments ----
segments = ET.SubElement(task, "segments")
ET.SubElement(segments, "segment", {
    "start": "0",
    "stop": str(NUM_FRAMES - 1)
})

# ---- REQUIRED: labels (BY NAME, NOT ID) ----
labels = ET.SubElement(task, "labels")
label = ET.SubElement(labels, "label", {"name": "person"})
ET.SubElement(label, "attributes")

# ---- TRACKS ----
for tid in sorted(tracks_by_id.keys()):
    track_el = ET.SubElement(root, "track", {
        "id": str(tid),
        "label": "person"   # ✅ THIS IS WHAT YOUR CVAT EXPECTS
    })

    last_frame = None

    for frame, x1, y1, x2, y2 in tracks_by_id[tid]:

        # fill gaps
        if last_frame is not None:
            for gap in range(last_frame + 1, frame):
                ET.SubElement(track_el, "box", {
                    "frame": str(gap),
                    "xtl": "0",
                    "ytl": "0",
                    "xbr": "0",
                    "ybr": "0",
                    "outside": "1",
                    "occluded": "0",
                    "keyframe": "0",
                    "z_order": "0"
                })

        # real detection
        ET.SubElement(track_el, "box", {
            "frame": str(frame),
            "xtl": f"{x1:.2f}",
            "ytl": f"{y1:.2f}",
            "xbr": f"{x2:.2f}",
            "ybr": f"{y2:.2f}",
            "outside": "0",
            "occluded": "0",
            "keyframe": "1",
            "z_order": "0"
        })

        last_frame = frame

# =========================
# WRITE XML
# =========================
ET.ElementTree(root).write(
    OUT_XML,
    encoding="utf-8",
    xml_declaration=True
)

print("[OK] CVAT 1.1 XML written successfully:")
print(OUT_XML)
