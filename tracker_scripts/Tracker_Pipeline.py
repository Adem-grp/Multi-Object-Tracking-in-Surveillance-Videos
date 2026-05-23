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

# TrackEval needed for HOTA computation
try:
    import trackeval

    TRACKEVAL = True
except ImportError:
    TRACKEVAL = False
    print("Trackeval not available")

DetectorWeights = r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\runs_final\detect\all_datasets\weights\best.pt"
OutDir = r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\tracker_outputs"
Conf, Iou, Max_det, agnostic_nms, device, Imgsz = 0.01, 0.6, 500, False, 0, 640

# ReID is for deepsort it will be downloaded automatically after first run
ReID_weights = Path("osnet_x0_25_msmt17.pt")

# Datasets since there are multiple videos per dataset it is important to arrange the ground truths and video paths

DATASETS = {
    "avenue": [ (r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\cvat_inputs\avenue\avenue_01.mp4",r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\cvat_inputs\avenue\avenue_gt.txt"),],
    "UCSD": [
        (r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\cvat_inputs\ucsd\ped1_video\UCSD_Ped1_Test001.mp4",r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\cvat_inputs\ucsd\ped1_video\ucsd_ped1_test1_gt.txt"), # (video files, mot_gt.txt) files
        (r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\cvat_inputs\ucsd\ped1_video\UCSD_Ped1_Test002.mp4",r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\cvat_inputs\ucsd\ped1_video\ucsd_ped1_test2_gt.txt"),
        (r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\cvat_inputs\ucsd\ped1_video\UCSD_Ped1_Test003.mp4",r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\cvat_inputs\ucsd\ped1_video\ucsd_ped1_test3_gt.txt"),
        (r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\cvat_inputs\ucsd\ped2_video\UCSD_Ped2_Test001.mp4",r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\cvat_inputs\ucsd\ped2_video\ucsd_ped2_test1_gt.txt"),
        (r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\cvat_inputs\ucsd\ped2_video\UCSD_Ped2_Test002.mp4",r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\cvat_inputs\ucsd\ped2_video\ucsd_ped2_test2_gt.txt"),
        (r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\cvat_inputs\ucsd\ped2_video\UCSD_Ped2_Test003.mp4",r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\cvat_inputs\ucsd\ped2_video\ucsd_ped2_test3_gt.txt"),
             ],
    "shanghai": [
        (r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\cvat_inputs\shanghai\test_02_0128.mp4",r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\cvat_inputs\shanghai\shanghai_128_gt.txt"),
        (r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\cvat_inputs\shanghai\test_02_0164.mp4",r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\cvat_inputs\shanghai\shanghai_164_gt.txt"),
        (r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\cvat_inputs\shanghai\test_04_0010.mp4",r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\cvat_inputs\shanghai\shanghai_10_gt.txt"),
    ],
    "gmot": [
        (r"",r""),], # turn gmot to video and add its gt as well
}

TrackerGrids = {
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

# Default params per tracker (used in baseline)
TrackerDefaults = {
    "deepsort": {"max_dist": 0.2, "max_age": 30, "n_init": 3, "max_iou_dist": 0.7},
    "bytetrack": {"track_high_thresh": 0.5, "track_low_thresh": 0.1,
                  "track_buffer": 30, "match_thresh": 0.8},
    "ocsort": {"det_thresh": 0.5, "max_age": 30, "min_hits": 3, "iou_threshold": 0.3},
}

OutDir.mkdir(parents=True, exist_ok=True)
