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

}
