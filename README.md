# Multi-Object Tracking in Surveillance Videos

**MSc Dissertation Project | Kingston University London**

A YOLO11m-based multi-object tracking pipeline evaluated and optimised for deployment on consumer-grade GPU hardware. Includes a systematic comparison of DeepSORT, ByteTrack, and OC-SORT across four surveillance datasets, a detector compression study, and a real-time deployment demo.

---

## Overview

This project trains and tunes a YOLO11m detector, benchmarks three multi-object trackers against it, applies structured pruning and quantisation to the best-performing configuration, and deploys the result as a real-time tracking application — all evaluated under consumer-grade hardware constraints (NVIDIA RTX 4070).

**Selected deployment configuration:** YOLO11m (FP16-quantised) + ByteTrack

---

## Objectives

- Train and evaluate a YOLO11m detector across five dataset combinations
- Benchmark DeepSORT, ByteTrack, and OC-SORT with a two-stage hyperparameter search
- Create MOT-style ground truth annotations for three anomaly-detection datasets that had none
- Apply structured pruning and post-training quantisation to the best detector-tracker combination
- Deploy a real-time tracking application on the optimised pipeline

---

## Datasets

| Dataset | Type | Ground Truth |
|---|---|---|
| GMOT-40 | Multi-class (9 non-person + person) | Professional, pre-existing |
| CUHK Avenue | Pedestrian, anomaly detection | Manually annotated in CVAT (this project) |
| UCSD Anomaly Detection | Pedestrian, dense crowd | Manually annotated in CVAT (this project) |
| ShanghaiTech Campus | Pedestrian, anomaly detection | Manually annotated in CVAT (3 of 13 scenes) |

Avenue, UCSD, and ShanghaiTech were originally built for frame-level anomaly detection and had no MOT tracking labels. Ground truth was created manually in CVAT specifically for this project, following consistent full-body bounding box and occlusion-handling rules — see dissertation Section 3.3 for the full annotation methodology.

---

## Detector — YOLO11m

- Trained via transfer learning from COCO-pretrained weights (no layers frozen)
- Training hyperparameters (lr0, momentum, weight_decay) optimised via Ray Tune with an ASHA scheduler
- Selected from five dataset combinations (GMOT-only, GMOT+each anomaly dataset, all combined) based on highest baseline mAP50
- Confidence threshold tuned from the YOLO default (0.05) down to 0.01, substantially improving recall

| | Baseline (conf=0.05) | Tuned (conf=0.01) |
|---|---|---|
| mAP50 | 40.8% | 63.1% |
| mAP50-95 | 16.3% | 25.9% |
| Precision | 70.0% | 83.4% |
| Recall | 39.5% | 58.7% |

For reference, COCO-pretrained YOLO11m with no domain-specific training achieved only 0.17% mAP50 on this task.

---

## Tracker Benchmarking

Three trackers were evaluated with published default parameters (baseline), then tuned via a two-stage grid search (confidence/IoU thresholds, then tracker-specific parameters), selecting the configuration with the highest average HOTA across all four datasets.

**Final tuned results (averaged across all four datasets):**

| Tracker | HOTA | MOTA | IDF1 | FPS | Peak VRAM |
|---|---|---|---|---|---|
| **ByteTrack** (selected) | **45.28%** | 52.28% | 59.59% | **88.55** | 408.98 MB |
| OC-SORT | 44.02% | 50.98% | 58.51% | 67.43 | **399.13 MB** |
| DeepSORT | 41.64% | 31.11% | 54.53% | 21.48 | 1692.93 MB |

**ByteTrack was selected** for the best average HOTA across all datasets, the fastest inference, and VRAM close to the lowest — not the single highest score on every individual dataset, but the best overall balance of accuracy, speed, and stability for deployment. It also showed the strongest robustness to hyperparameter variation of the three trackers.

DeepSORT's appearance-based re-identification (OSNet) provides identity stability but at significant computational cost, making it unsuitable for real-time deployment on this hardware.

---

## Detector Compression

Structured L1-norm pruning (20/40/60% sparsity) and post-training quantisation (FP16, INT8) were applied to the selected detector.

**Key finding:** raw pruning without fine-tuning collapses model performance to 0% HOTA at every sparsity level — pruned filters desynchronise BatchNorm statistics from the remaining activation distributions. Fine-tuning is not optional; it is structurally required to recover functional accuracy after pruning.

After fine-tuning, pruned variants recover close to baseline accuracy but do **not** reduce VRAM or improve FPS as might be expected, since PyTorch's `torch.nn.utils.prune` zeroes filter weights without physically compacting tensor dimensions — genuine memory/speed gains would require further model restructuring beyond the pruning step itself.

**FP16 quantisation was the effective compression technique:** matching or exceeding baseline HOTA at roughly a third less VRAM, with no fine-tuning required.

| Variant | HOTA | FPS | Peak VRAM |
|---|---|---|---|
| Baseline (FP32) | 45.28% | 88.0 | 274 MB |
| Raw pruning (any sparsity, no fine-tune) | 0.00% | 100–104 | 236–352 MB |
| Pruned + fine-tuned (20/40/60%) | 43.7–45.3% | 79.9–85.5 | 391–428 MB |
| **FP16 only** | **45.67%** | 73.7 | **180 MB** |

INT8 quantisation via PyTorch's dynamic quantisation is limited to Linear layers (the CPU backend does not support Conv2d dynamic quantisation); full Conv2d INT8 would require TensorRT and was out of scope for this project.

---

## Real-Time Pipeline

A standalone real-time application (`Real-Time_Pipeline.py`) runs the FP16-quantised YOLO11m detector with ByteTrack on a live webcam or video file, displaying bounding boxes, track IDs, class labels, confidence scores, and live FPS/VRAM usage.

Note: the real-time pipeline does not compute HOTA/MOTA/IDF1, as these metrics require pre-existing ground truth to compare against, which a live feed does not have. All accuracy figures reported above come from the offline evaluation pipeline (`Tracker_Pipeline.py`), which does write MOT1.1-format output and evaluate it against annotated ground truth.

**Controls:** `Q` quit · `SPACE` pause/resume · `S` screenshot · `V` toggle recording

---

## Repository Structure
```
├── GMOT-to-YOLO_Dataset_Conversion.py    # Converts GMOT-40 annotations to YOLO format
├── Avenue_YOLO_Preparation.py            # Prepares Avenue dataset splits
├── ShanghaiTech_YOLO_Preparation.py      # Prepares ShanghaiTech dataset splits
├── UCSD_YOLO_Preparation.py              # Prepares UCSD dataset splits
├── Annotation_Index_Arranger.py          # Reindexes CVAT-exported annotations
├── Class_Find.py                         # Class ID auditing/remapping utility
├── Dataset_Bridge_Creation.py            # Silver-label generation for detector training
├── GMOT_yaml                             # YOLO training data configuration
├── RayTune_YOLO11m.py                    # Ray Tune training hyperparameter search
├── YOLO_Training_Pipeline.py             # Full detector training pipeline
├── Find_Best_Inference_Hyperparameters.py # Detector conf/IoU threshold search
├── Tracker_Pipeline.py                   # Offline tracker benchmarking & evaluation
├── Find_Best_Tracker_Params.py           # Per-tracker hyperparameter grid search
├── Find_Best_Tracker_Overall.py          # Best universal parameter selection
├── Compression_Pipeline.py               # Pruning and quantisation pipeline
└── Real-Time_Pipeline.py                 # Real-time webcam/video demo
```

---

## Evaluation Metrics

- **HOTA** (Higher Order Tracking Accuracy) — balances detection and association quality
- **MOTA** (Multi-Object Tracking Accuracy) — penalises misses, false positives, and ID switches
- **IDF1** — identity preservation across frames
- **FPS** / **Peak VRAM** — real-time deployment efficiency

---

## Dependencies
```
ultralytics
boxmot (ByteTrack)
deep-sort-realtime
ocsort
torch, torchvision
motmetrics
trackeval
ray[tune]
opencv-python
```

---

## References

- Redmon, J. et al. (2016) — YOLO
- Bewley, A. et al. (2016) — SORT
- Wojke, N. et al. (2017) — DeepSORT
- Zhang, Y. et al. (2022) — ByteTrack
- Cao, J. et al. (2023) — OC-SORT
- Luiten, J. et al. (2021) — HOTA

Full reference list with DOIs available in the accompanying dissertation.

---

---

## Author

Adem Garip — MSc Computer Science, Kingston University London
