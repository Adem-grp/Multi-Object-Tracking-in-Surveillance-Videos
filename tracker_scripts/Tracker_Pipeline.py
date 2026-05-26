# runs final all dataset detection model version is chosen to be best since it achieves near highest mAP and has less
# computational cost with 23 boxes while the best mAP GMOT+UCSD produces 70 boxes

import argparse
import time
import cv2
import torch
import numpy as np
import pandas as pd
import motmetrics as mm
from itertools import product
from pathlib import Path
from ultralytics import YOLO
from boxmot import DeepSORT, ByteTrack, OcSort

from OC_SORT.trackers.deepsort_tracker.deepsort import DeepSort

# TrackEval needed for HOTA computation
try:
    import trackeval

    TRACKEVAL = True
except ImportError:
    TRACKEVAL = False
    print("Trackeval not available")

from inspect import signature

DetectorWeights = r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\runs_final\detect\all_datasets\weights\best.pt"
OutDir = r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\tracker_outputs"
Conf, Iou, Max_det, Agnostic_nms, device, Imgsz = 0.01, 0.6, 500, False, 0, 640

# ReID is for deepsort it will be downloaded automatically after first run
ReID_weights = Path("osnet_x0_25_msmt17.pt")

# Datasets since there are multiple videos per dataset it is important to arrange the ground truths and video paths

DATASETS = {
    "avenue": [
        (r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\avenue_yolo\test\img1",
         r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\avenue_yolo\test\gt\gt.txt"), ],
    "UCSD": [
        (r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\ucsd_yolo\test\ped1_test1\img1",
         r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\ucsd_yolo\test\ped1_test1\gt\gt.txt"),
        (r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\ucsd_yolo\test\ped1_test2\img1",
         r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\ucsd_yolo\test\ped1_test2\gt\gt.txt"),
        (r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\ucsd_yolo\test\ped1_test3\img1",
         r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\ucsd_yolo\test\ped1_test3\gt\gt.txt"),
        (r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\ucsd_yolo\test\ped2_test1\img1",
         r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\ucsd_yolo\test\ped2_test1\gt\gt.txt"),
        (r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\ucsd_yolo\test\ped2_test2\img1",
         r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\ucsd_yolo\test\ped2_test2\gt\gt.txt"),
        (r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\ucsd_yolo\test\ped2_test3\img1",
         r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\ucsd_yolo\test\ped2_test3\gt\gt.txt"),
    ],
    "shanghai": [
        (r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\shanghaitech_yolo\test\shanghai_10\img1",
         r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\shanghaitech_yolo\test\shanghai_10\gt\gt.txt"),
        (r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\shanghaitech_yolo\test\shanghai_128\img1",
         r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\shanghaitech_yolo\test\shanghai_128\gt\gt.txt"),
        (r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\shanghaitech_yolo\test\shanghai_164\img1",
         r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\shanghaitech_yolo\test\shanghai_164\gt\gt.txt"),
    ],
    "gmot": [
        (r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\gmot_yolo\test\airplane-1\img1",
         r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\gmot_yolo\test\airplane-1\gt\gt.txt"),
        (r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\gmot_yolo\test\ball-1\img1",
         r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\gmot_yolo\test\ball-1\gt\gt.txt"),
        (r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\gmot_yolo\test\ball-2\img1",
         r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\gmot_yolo\test\ball-2\gt\gt.txt"),
        (r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\gmot_yolo\test\ball-3\img1",
         r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\gmot_yolo\test\ball-3\gt\gt.txt"),
        (r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\gmot_yolo\test\balloon-0\img1",
         r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\gmot_yolo\test\balloon-0\gt\gt.txt"),
        (r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\gmot_yolo\test\bird-2\img1",
         r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\gmot_yolo\test\bird-2\gt\gt.txt"),
        (r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\gmot_yolo\test\bird-3\img1",
         r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\gmot_yolo\test\bird-3\gt\gt.txt"),
        (r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\gmot_yolo\test\boat-1\img1",
         r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\gmot_yolo\test\boat-1\gt\gt.txt"), ],
    # turn gmot to video and add its gt as well
}

TrackerGrids = {  # will be extended
    "deepsort": {
        "max_dist": [0.1, 0.2, 0.3],
        "max_age": [30, 50, 70],
        "n_init": [1, 3, 5],
        "max_iou_dist": [0.5, 0.7, 0.9],
    },
    "bytetrack": {
        "track_high_thresh": [0.4, 0.5, 0.6],
        "track_low_thresh": [0.05, 0.1, 0.2],
        "track_buffer": [20, 30, 40],
        "match_thresh": [0.7, 0.8, 0.9],
    },
    "ocsort": {
        "det_thresh": [0.4, 0.5, 0.6],
        "max_age": [20, 30, 50],
        "min_hits": [1, 3, 5],
        "iou_threshold": [0.2, 0.3, 0.4],
    },
}

ConfGrid = {
    "bytetrack": [0.01,0.03,0.05],
    "deepsort": [0.3,0.4,0.5],
    "ocsort": [0.2,0.3,0.4],
}
IouGrid = [0.5, 0.6, 0.7]

# Default params per tracker (used in baseline)
TrackerDefaults = {
    "deepsort": {"max_dist": 0.2, "max_age": 30, "n_init": 3, "max_iou_dist": 0.7},
    "bytetrack": {"track_high_thresh": 0.5, "track_low_thresh": 0.1,
                  "track_buffer": 30, "match_thresh": 0.8},
    "ocsort": {"det_thresh": 0.5, "max_age": 30, "min_hits": 3, "iou_threshold": 0.3},
}

OutDir.mkdir(parents=True, exist_ok=True)


def build_tracker(tracker, params):
    if tracker == "deepsort":
        return DeepSort(
            max_dist=params["max_dist"],
            max_age=params["max_age"],
            n_init=params["n_init"],
            max_iou_distance=params["max_iou_dist"],
        )
    elif tracker == "bytetrack":
        return ByteTrack(
            track_high_thresh=params["track_high_thresh"],
            track_low_thresh=params["track_low_thresh"],
            track_buffer=params["track_buffer"],
            match_thresh=params["match_thresh"],
        )
    elif tracker == "ocsort":
        return OcSort(
            det_thresh=params["det_thresh"],
            max_age=params["max_age"],
            min_hits=params["min_hits"],
            iou_threshold=params["iou_threshold"],
        )
    else:
        raise ValueError(f"Tracker {tracker} is not supported.")


# run tracker on one sequence
def tracker_on_sequence(img_folder, tracker_name, tracker_params, output_path):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    img_folder = Path(img_folder)
    frame_paths = sorted(img_folder.glob("*.jpg"))
    if not frame_paths:
        frame_paths = sorted(img_folder.glob("*.png"))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {img_folder}")
    model = YOLO(DetectorWeights)
    tracker = build_tracker(tracker_name, tracker_params)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    mot_lines = []
    frame_times = []
    total_time = 0.0
    for frame_idx, img_path in enumerate(frame_paths):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Frame {frame_idx} cannot be read.")
            continue
        t_start = time.perf_counter()
        results = model.predict(
            frame, conf=Conf, iou=Iou, imgsz=Imgsz, verbose=False,
        )
        dets = []
        if results[0].boxes is not None and len(results[0].boxes):
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].confs.cpu().numpy()
            clss = results[0].cls.cpu().numpy()
            for box, conf, cls in zip(boxes, confs, clss):
                dets.append([*box, conf, cls])
        dets_np = np.array(dets) if dets else np.empty((0, 6))
        tracks = tracker.update(dets_np, frame)
        t_end = time.perf_counter()
        frame_ms = (t_end - t_start) * 1000
        total_time += (t_end - t_start)
        frame_times.append(frame_ms)

        for track in tracks:
            x1, y1, x2, y2, conf, tid = track[0], track[1], track[2], track[3], float(track[5]), int(track[4])
            w = x2 - x1
            h = y2 - y1
            mot_lines.append(
                f"{frame_idx},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1"
            )

    output_path.write_text("\n".join(mot_lines), encoding="utf-8")
    n_frames = len(frame_paths)
    fps = n_frames / total_time if total_time > 0 else 0.0  # check if this gpu_only fps fix it
    latency_mean = round(float(np.mean(frame_times)), 2)
    latency_p95 = round(float(np.percentile(frame_times, 95)), 2)
    peak_vram = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1) if torch.cuda.is_available() else 0.0
    print(f"{img_folder.parent.name}/{img_folder.name} | {n_frames} frames | {fps:.1f} FPS")
    print(f"latency: {latency_mean:.2f} ms | VRAM {peak_vram} MB")
    return {
        "fps": round(fps, 1),
        "frames": n_frames,
        "latency_mean_ms": latency_mean,
        "latency_p95_ms": latency_p95,
        "peak_vram": peak_vram,
    }

# evaluate one sequence
def load_mot(mot_path):
    path = Path(mot_path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["frame", "id", "x", "y", "w", "h"])
    df = pd.read_csv(mot_path,header=None, names=["frame", "id", "x", "y", "w", "h","conf","cx","cy","cz"])
    return df[["frame","id","x","y","w","h"]].copy()

def evaluate_sequence(gt_path,pred_path,iou_threshold=0.5):
    gt_df = load_mot(gt_path)
    pred_df = load_mot(pred_path)
    acc = mm.MOTAccumulator(auto_id=True)
    all_frames = sorted(set(gt["frame"] | set(pred_df["frame"])).union(set(gt_df["frame"])))
    for frame in all_frames:
        gt_frame = gt_df[gt_df["frame"] == frame]
        pred_frame = pred_df[pred_df["frame"] == frame]
        gt_boxes = gt_frame[["x","y","w","h"]].values.tolist()
        pred_boxes = pred_frame[["x","y","w","h"]].values.tolist()
        gt_ids= gt_frame["id"].tolist()
        pred_ids = pred_frame["id"].tolist()
        if gt_boxes and pred_boxes:
            distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=1-iou_threshold)
        else:
            distances = np.empty((len(gt_boxes), len(pred_boxes)))
        acc.update(gt_ids, pred_ids, distances)
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["mota", "idf1", "num_switches", "num_misses",
                 "num_false_positives", "mostly_tracked",
                 "mostly_lost", "num_fragmentations"],
        name="eval",
    )
    r = summary.to_dict(orient="records")[0]
    r["mota_pct"] = round(float(r["mota"]) * 100, 2)
    r["idf1_pct"] = round(float(r["idf1"]) * 100, 2)
    return r


# evaluate dataset average it across clips as there are multiple clips per dataset





if __name__ == "__main__":
    print(signature(DeepSort))
