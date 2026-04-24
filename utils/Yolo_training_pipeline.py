from ultralytics import YOLO
import torch
import time
import os
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
5             All Combined        Is there a "Limit" to how much data improves a pruned model?


"""

Model = "yolo11n.pt"
# each run do not forget to change these runs
train_run = "runs/GMOT_only_1/train"  # change in each train val runs
val_run = "runs/GMOT_only_1/val"  # so change these according to table
runName = "full_finetune"  #  1 is for inital without hyperparameter tuning much
eval_results_path = "YOLO_training_results/eval_GMOT_ONLY_results_1.json"  # change in each evaluation
batchSize = 16  # for tuning change later
number_of_epochs = 50
tune_run = "runs/GMOT_only_1/tune"


def train():
    model = YOLO(Model)
    results = model.train(
        data=yamlPath,
        epochs=number_of_epochs,
        imgsz=640,
        batch=batchSize,
        device=DEVICE,
        project=train_run,  # each run do not forget to change these runs
        name=runName,
        patience=4,
        # other parameters will be added after looking at ultralytics docs
        # I need to find the best parameters that will optimise the model
        # for my case specific in this one
        # set pretrained true for training the best model again
        optimizer='auto',
        seed=42,

    )
    return results


def validate(best_weights_path):
    model = YOLO(best_weights_path)
    confs = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
    ious = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    dets = [300,500,750,1000]

    _ = model.predict(
        source=r"D:\datasets\gmot_yolo\val\images\airplane-0_000037.jpg",
        device=DEVICE,
        verbose=False)
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
                    batch=batchSize,
                    max_det= det,
                    device=DEVICE,
                    project=val_run,
                    name=runName,
                    seed=42,
                    agnostic_nms=True,
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
                    "model_file": best_weights_path,
                    "conf": conf,
                    "iou": iou,
                    "max_det": det,
                    "accuracy_metrics": {
                        "precision": round(results.results_dict['metrics/precision(B)'], 4),
                        "recall": round(results.results_dict['metrics/recall(B)'], 4),
                        "mAP50": round(results.results_dict['metrics/mAP50(B)'], 4),
                        "mAP50-95": round(results.results_dict['metrics/mAP50-95(B)'], 4)
                    },
                    "real_time_metrics": {
                        "avg_latency_ms": round(avg_inference_ms, 2),
                        "estimated_fps": round(fps, 1),
                        "peak_vram_usage_mb": round(peak_vram, 2)
                    },
                    "total_validation_time_sec": round(total_tm, 2)
                }
                with open(eval_results_path, "a") as f:
                    f.write(json.dumps(metrics) + "\n")

                print(f"--- Benchmark Complete: {best_weights_path} ---")
                print(
                    f"FPS: {metrics['real_time_metrics']['estimated_fps']} | mAP50: {metrics['accuracy_metrics']['mAP50']}")


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
    print("Cuda available:", torch.cuda.is_available())

    # Ray resource cap (very important for stability)
    ray.init(
        num_cpus=8,
        num_gpus=1,
        include_dashboard=False,
        ignore_reinit_error=True
    )

    if torch.cuda.is_available():
        print("Cuda device:", torch.cuda.get_device_name(0))

    print("FINAL YAML PATH:", Path(yamlPath).resolve())
    #tune()
    #results = train()
    best_weights = r"C:\Users\K2549603\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\runs\detect\runs\GMOT_only_1\tune\tune_ray_main_yolo11m\weights\best.pt"
    #validate(best_weights_path=best_weights)
