import cv2
from ultralytics import YOLO
import torch
import time
import json
from pathlib import Path
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Best_LR0 = 0.002038823487169829
Best_MOMENTUM = 0.731639442196722
Best_WEIGHT_DECAY = 1.4718411535261374e-05

yamlPath = r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\gmot.yaml"
DEVICE = 0

train_run = r"C:\Users\k2549603\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\runs\detect"
runName = "all_datasets"

eval_results_path = r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\YOLO_inference_evaluations\evaluation_results_dataset_combinations_final.json"

batchSize = 16
number_of_epochs = 50

def train(best_weights_path):
    model = YOLO(best_weights_path)

    model.train(
        data=str(Path(yamlPath).resolve()),
        epochs=number_of_epochs,
        imgsz=640,
        batch=batchSize,
        device=DEVICE,
        project=str(Path(train_run).resolve()),
        name=runName,
        exist_ok=True,
        patience=12,
        lr0=Best_LR0,
        momentum=Best_MOMENTUM,
        weight_decay=Best_WEIGHT_DECAY,
        optimizer="SGD",
        seed=42,
    )

def measure_fps(model,images_dir,max_frames=150):
    imageFiles = sorted([f for f in os.listdir(images_dir) if f.endswith((".jpg","png","jpeg"))])
    imageFiles = imageFiles[:max_frames]

    # warm up
    first_img = os.path.join(images_dir,imageFiles[0])
    res = model.predict(
        source=first_img,
        conf=0.01,
        iou=0.6,
        max_det=500,
        agnostic_nms=False,
        device=DEVICE,
        verbose=False
    )
    print("Boxes: ",len(res[0].boxes))
    frameCount = 0
    start_time = time.time()
    for img in imageFiles:
        img_path = os.path.join(images_dir, img)
        _ = model.predict(
            source=img_path,
            conf=0.01,
            iou=0.6,
            max_det=500,
            agnostic_nms=False,
            device=DEVICE,
            seed=42,
            verbose=False
        )
        frameCount += 1
    total_time = time.time() - start_time
    fps = frameCount / total_time if total_time > 0 else 0
    return round(fps, 2)

def validate_best(best_weights_path):
    model = YOLO(best_weights_path)
    _ = model.predict(
        source=r"D:\datasets\gmot_yolo\val\images\airplane-0_000037.jpg",
        device=DEVICE,
        verbose=False
    )

    start_time = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    results = model.val(
        data=yamlPath,
        split="val",
        imgsz=640,
        conf=0.01,
        iou=0.6,
        max_det=500,
        batch=batchSize,
        device=DEVICE,
        seed=42,
        agnostic_nms=False,
        save=False,
        plots=False
    )

    total_tm = time.time() - start_time
    avg_inference_ms = results.speed["inference"]
    fps = measure_fps(model,images_dir=r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\gmot_yolo\val\images", max_frames=150)

    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

    metrics = {
        "model": "All_datasets",
        "conf": 0.01,
        "iou": 0.6,
        "max_det": 500,
        "detection_metrics": {
            "precision": round(results.results_dict["metrics/precision(B)"], 4),
            "recall": round(results.results_dict["metrics/recall(B)"], 4),
            "mAP50": round(results.results_dict["metrics/mAP50(B)"], 4),
            "mAP50-95": round(results.results_dict["metrics/mAP50-95(B)"], 4)
        },
        "real_time_metrics": {
            "avg_latency_ms": round(avg_inference_ms, 2),
            "estimated_real_fps": round(fps, 1),
            "peak_vram_usage_mb": round(peak_vram, 2)
        },
        "total_validation_time_sec": round(total_tm, 2)
    }

    os.makedirs(os.path.dirname(eval_results_path), exist_ok=True)

    with open(eval_results_path, "a") as f:
        f.write(json.dumps(metrics) + "\n")

def validate(best_weights_path):
    model = YOLO(best_weights_path)

    confs = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    ious = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]
    dets = [100, 200, 300, 500, 750]
    agnostic = [True, False]

    _ = model.predict(
        source=r"D:\datasets\gmot_yolo\val\images\airplane-0_000037.jpg",
        device=DEVICE,
        verbose=False
    )

    for agno in agnostic:
        for conf in confs:
            for iou in ious:
                for det in dets:
                    start_time = time.time()
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
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
                        plots=False
                    )

                    total_tm = time.time() - start_time
                    avg_inference_ms = results.speed["inference"]
                    fps = 1000 / avg_inference_ms if avg_inference_ms > 0 else 0

                    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

                    metrics = {
                        "model": "YOLO11m",
                        "conf": conf,
                        "iou": iou,
                        "max_det": det,
                        "agnostic_nms": agno,
                        "detection_metrics": {
                            "precision": round(results.results_dict["metrics/precision(B)"], 4),
                            "recall": round(results.results_dict["metrics/recall(B)"], 4),
                            "mAP50": round(results.results_dict["metrics/mAP50(B)"], 4),
                            "mAP50-95": round(results.results_dict["metrics/mAP50-95(B)"], 4)
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

if __name__ == "__main__":
    print("CWD:", os.getcwd())
    print("FINAL YAML PATH:", Path(yamlPath).resolve())

    best_weights = r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\runs_final\detect\all_datasets\weights\best.pt"

    validate_best(best_weights)