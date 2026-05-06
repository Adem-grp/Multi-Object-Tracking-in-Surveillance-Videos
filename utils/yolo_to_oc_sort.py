import cv2
from pathlib import Path

BASE = Path(r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos")

VIDEO = BASE / "cvat_inputs" / "avenue" / "avenue_01.mp4"
YOLO_LABEL_DIR = BASE / "cvat_inputs" / "avenue_labels" / "avenue_01" / "labels"
OUT_DIR = BASE / "cvat_inputs" / "avenue_ocsort_detections"

OUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO}")

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

for label_file in sorted(YOLO_LABEL_DIR.glob("*.txt")):
    # avenue_01_123.txt -> frame 122
    frame_idx = int(label_file.stem.split("_")[-1]) - 1
    if frame_idx < 0:
        continue

    out_file = OUT_DIR / f"{frame_idx:06d}.txt"
    lines_out = []

    with open(label_file, "r") as f:
        for line in f:
            if not line.strip():
                continue

            cls, xc, yc, w, h = map(float, line.split())

            x1 = (xc - w / 2) * W
            y1 = (yc - h / 2) * H
            x2 = (xc + w / 2) * W
            y2 = (yc + h / 2) * H

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W - 1, x2)
            y2 = min(H - 1, y2)

            score = 0.5
            lines_out.append(f"{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f},{score}")

    out_file.write_text("\n".join(lines_out))

print(f"[OK] OC-SORT detections written to: {OUT_DIR}")
