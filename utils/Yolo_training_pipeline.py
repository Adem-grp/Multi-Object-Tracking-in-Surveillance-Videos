from ultralytics import YOLO
import torch
import time

import json
from pathlib import Path
import os
import ray

# Lock CUDA to a single GPU BEFORE torch is imported
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# yaml specific to lab pc change it for laptop it was related to yolo guessing yaml was in utils so
yamlPath = r"C:\Users\k2549603\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\gmot.yaml"
DEVICE = 0

"""
Model         #Training Data      Research Question 
1 (Baseline)  GMOT-40 Only        How does standard YOLO handle generic objects?
2             GMOT + Avenue       Does adding campus-style surveillance improve pedestrian/person detection?
3             GMOT + UCSD         Does adding low-resolution/overhead footage help with scale invariance?
4             GMOT + ShanghaiTech Does massive urban data help with crowded scene detection?
5             All Combined        Does more diverse data lead to better generalization across all object types and scenes?

"""

# each run do not forget to change these runs
train_run = "runs/detect/GMOT_only/train"  # change in each train val runs
runName = "full_finetune(GMOT_only)"  #  1 is for inital without hyperparameter tuning much
eval_results_path = r"C:\Users\K2549603\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\YOLO_inference_evaluations\eval_results_dataset_combinations.json"
# change in each evaluation
batchSize = 16  # for tuning change later
number_of_epochs = 100
tune_run = "runs/GMOT_only/tune"


def train(best_weights_path):
    model = YOLO(best_weights_path)
    results = model.train(
        data=str(Path(yamlPath).resolve()),
        epochs=number_of_epochs,
        imgsz=640,
        batch=batchSize,
        device=DEVICE,
        project=train_run,  # each run do not forget to change these runs
        name=runName,
        patience=3,
        optimizer='auto',
        seed=42,
        resume=True

    )


def validate_best(best_weights_path):
    model = YOLO(best_weights_path)
    _ = model.predict(
        source=r"D:\datasets\gmot_yolo\val\images\airplane-0_000037.jpg",
        device=DEVICE,
        verbose=False)
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    results = model.val(
        data=yamlPath,
        split="val",
        imgsz=640,
        conf=0.05,
        iou=0.6,
        max_det=500,
        batch=batchSize,
        device=DEVICE,
        seed=42,
        agnostic_nms=True,
        save=False,
        plots=False,
    )
    total_tm = time.time() - start_time
    # average inference time in ms pre-process+Inference+Post-process
    avg_inference_ms = results.speed['inference']
    fps = 1000 / avg_inference_ms if avg_inference_ms > 0 else 0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_vram = 0
    metrics = {
        "model": "All Datasets",
        "conf": 0.05,
        "iou": 0.6,
        "max_det": 500,
        "detection_metrics": {
            "precision": round(results.results_dict['metrics/precision(B)'], 4),
            "recall": round(results.results_dict['metrics/recall(B)'], 4),
            "mAP50": round(results.results_dict['metrics/mAP50(B)'], 4),
            "mAP50-95": round(results.results_dict['metrics/mAP50-95(B)'], 4)
        },
        "real_time_metrics": {
            "avg_latency_ms": round(avg_inference_ms, 2),
            "estimated_fps_gpu_only": round(fps, 1),
            "peak_vram_usage_mb": round(peak_vram, 2)
        },
        "total_validation_time_sec": round(total_tm, 2)
    }
    os.makedirs(os.path.dirname(eval_results_path), exist_ok=True)
    with open(eval_results_path, "a") as f:
        f.write(json.dumps(metrics) + "\n")

    print(f"--- Benchmark Complete: {best_weights_path} ---")
    print(
        f"GPU_only FPS: {metrics['real_time_metrics']['estimated_fps_gpu_only']} |"
        f" mAP50: {metrics['detection_metrics']['mAP50']} | mAP50-95: {metrics['detection_metrics']['mAP50-95']} |"
        f" Conf: {0.05} | IoU: {0.6} | Max Det: {500} | Time: {metrics['total_validation_time_sec']}s | "
        f"VRAM: {metrics['real_time_metrics']['peak_vram_usage_mb']}MB")


def validate(best_weights_path):
    model = YOLO(best_weights_path)
    confs = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    ious = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]
    dets = [100, 200, 300, 500, 750]
    agnostic = [True, False]
    _ = model.predict(
        source=r"D:\datasets\gmot_yolo\val\images\airplane-0_000037.jpg",
        device=DEVICE,
        verbose=False)
    for agno in agnostic:
        for conf in confs:
            for iou in ious:
                for det in dets:
                    start_time = time.time()
                    results = model.val(
                        data=yamlPath,
                        split="val",
                        imgsz=640,
                        conf=conf,
                        iou=iou,
                        max_det=det,
                        agnostic_nms=agno,
                        batch=batchSize,
                        device=DEVICE,
                        seed=42,
                        save=False,
                        plots=False,
                    )
                    total_tm = time.time() - start_time
                    # average inference time in ms pre-process+Inference+Post-process
                    avg_inference_ms = results.speed['inference']
                    fps = 1000 / avg_inference_ms if avg_inference_ms > 0 else 0
                    if torch.cuda.is_available():
                        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
                    else:
                        peak_vram = 0
                    metrics = {
                        "model": "YOLO11m",
                        "conf": conf,
                        "iou": iou,
                        "max_det": det,
                        "agnostic_nms": agno,
                        "detection_metrics": {
                            "precision": round(results.results_dict['metrics/precision(B)'], 4),
                            "recall": round(results.results_dict['metrics/recall(B)'], 4),
                            "mAP50": round(results.results_dict['metrics/mAP50(B)'], 4),
                            "mAP50-95": round(results.results_dict['metrics/mAP50-95(B)'], 4)
                        },
                        "real_time_metrics": {
                            "avg_latency_ms": round(avg_inference_ms, 2),
                            "estimated_fps_gpu_only": round(fps, 1),
                            "peak_vram_usage_mb": round(peak_vram, 2)
                        },
                        "total_validation_time_sec": round(total_tm, 2)
                    }
                    os.makedirs(os.path.dirname(eval_results_path), exist_ok=True)
                    with open(eval_results_path, "a") as f:
                        f.write(json.dumps(metrics) + "\n")

                    print(f"--- Benchmark Complete: {best_weights_path} ---")
                    print(
                        f"GPU_only FPS: {metrics['real_time_metrics']['estimated_fps_gpu_only']} | mAP50: {metrics['detection_metrics']['mAP50']} | mAP50-95: {metrics['detection_metrics']['mAP50-95']} | Conf: {conf} | IoU: {iou} | Max Det: {det} | Time: {metrics['total_validation_time_sec']}s | VRAM: {metrics['real_time_metrics']['peak_vram_usage_mb']}MB")


def tune():
    model = YOLO("yolo11m.pt")

    model.tune(
        data=str(Path(yamlPath).resolve()),
        epochs=100,
        iterations=5,
        patience=10,
        optimizer="auto",
        batch=batchSize,
        plots=True,
        save=True,
        project=tune_run,
        name="tune_ray_main_yolo11m",  # train yolo11m as well for 20 iterations to see the difference
        device=DEVICE,
        use_ray=True,
        resume=True,
        space={},  # REQUIRED to avoid NoneType crash
        exist_ok=True
    )


if __name__ == "__main__":
    print("CWD:", os.getcwd())
    print("JSON absolute path:", os.path.abspath(eval_results_path))

    print("Cuda available:", torch.cuda.is_available())
    # Ray resource cap (very important for stability)
    """ ray.init(
        num_cpus=8,
        num_gpus=1,
        include_dashboard=False,
        ignore_reinit_error=True
    )"""

    if torch.cuda.is_available():
        print("Cuda device:", torch.cuda.get_device_name(0))

    print("FINAL YAML PATH:", Path(yamlPath).resolve())
    #tune()
    #results = train()
    #best_weights = r"C:\Users\K2549603\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\runs\detect\runs\GMOT+ShanghaiTech\train\full_finetune(GMOT+ShanghaiTech)\weights\best.pt"
    best_weights = r"C:\Users\k2549603\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\runs\detect\runs\All_Datasets\train\full_finetune(All_Datasets)\weights\best.pt"
    #validate(best_weights_path=best_weights)
    validate_best(best_weights_path=best_weights)
    #train(best_weights)
