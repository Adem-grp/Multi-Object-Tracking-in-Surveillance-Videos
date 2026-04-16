from ultralytics import YOLO
import torch
import time
import os
import json

"""
Model         #Training Data      Research Question 
1 (Baseline)  GMOT-40 Only        How does standard YOLO handle generic objects?
2             GMOT + Avenue       Does adding campus-style surveillance improve pedestrian/person detection?
3             GMOT + UCSD         Does adding low-resolution/overhead footage help with scale invariance?
4             GMOT + ShanghaiTech Does massive urban data help with crowded scene detection?
5             All Combined        Is there a "Limit" to how much data improves a pruned model?


"""


Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yamlPath = "gmot.yaml"
Model = "yolo11n.pt"
# each run do not forget to change these runs
train_run = "runs/GMOT_only_1/train"  # change in each train val runs
val_run = "runs/GMOT_only_1/val"  # so change these according to table
runName = "gmot_train"       #  1 is for inital without hyper parameter tuning much
eval_results_path = "YOLO_training_results/eval_GMOT_ONLY_results_1.json"  # change in each evaluation
batchSize = 32
number_of_epochs = 50


def train():
    model = YOLO(Model)
    results = model.train(
        data=yamlPath,
        epochs=number_of_epochs,
        imgsz=640,
        batchsz=batchSize,
        device=Device,
        project=train_run,  # each run do not forget to change these runs
        name=runName,
        patience=4,
        # other parameters will be added after looking at ultralytics docs
        # I need to find the best parameters that will optimise the model
        # for my case specific in this one
        # set pretrained true for training the best model again
        optimizer='auto',
        seed=42,
        # the ones below are explanation of hyper parameters
        # after the initial run check the results and apply the necessaryones
        # each I wrote their explanations if not enough check ultralytics again

        #cos_lr=True,
        #close_mosaic=3, # disable data augmentation in last 3 epochs for traning stabilisation
        # if I want to reduce memory usage I need to set amp=True
        # freeze (int): freezes first N layers or specified layers by index
        # it reduces the number of trainable parameters it is useful for
        # fine-tuning and transfer learning
        # lr0(float) = inital learning rate  adjusting this is important
        # lrf (float)= final learning rate as a fraction of the inital rate (lr0*lrf) used in conjunction
        # with schedulers to adjust the learning rate over time
        # momentum(float)= it is for SGD or beta1 for adam optmisors for influencing the
        # incorporation of past gradients in the current update
        # weight_decay(float): penalise large weights to prevent overfittin like 5e-5
        # warmup_epochs(float): e.g 3 . for learning rate warm up gradually increasing the learning rate from a low val
        # to the lr0 to stabilise training early on
        # warmup_momentum (float): test this but I guess using momentum won't be that much of a big deal so
        # warmup_bias_lr(float): learning rate for bias parameters during the warmup phase helping stablise model training
        # in the initial epochs
        # box (float) : weight of the box loss componenet in the loss function influencing how much emphasis is placed
        # on accuractely predicting the bounding box coordinates
        # cls (float): weight of the class loss component in the loss function influencing how much emphasis is placed
        # on accurately predicting the class labels
        # cls_pw (float): Power for class weighting to handle class imbalance using inverse class frequency.
        # 0.0 disables class weighting, 1.0 applies full inverse frequency weighting. Values between 0 and 1 provide partial weighting.
        # there are a few more but first deal with the ones above

    )
    return results


def validate(best_weights_path):
    model = YOLO(best_weights_path)
    _ = model.predict(
        source=r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\gmot_yolo\train\images\airplane-3_000000.jpg",
        device=Device, verbose=False)
    start_time = time.time()
    results = model.val(
        data=yamlPath,
        split="val",
        imgsz=640,
        batch=batchSize,
        device=Device,
        project=val_run,
        name=runName,
        seed=42,
    )
    total_tm = time.time() - start_time
    # average inference time in ms pre-process+Inference+Post-process
    avg_inference_ms = results.speed['inference']
    fps = 1000 / avg_inference_ms if avg_inference_ms > 0 else 0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if Device == "cuda" else 0
    metrics = {
        "model_file": best_weights_path,
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
        "hardware_context": {
            "device": torch.cuda.get_device_name(0) if Device == "cuda" else "CPU",
            "vram_total_gb": 8 if "4070" in torch.cuda.get_device_name(0) else "Unknown"
        }
    }
    with open(eval_results_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"--- Benchmark Complete: {best_weights_path} ---")
    print(f"FPS: {metrics['real_time_metrics']['estimated_fps']} | mAP50: {metrics['accuracy_metrics']['mAP50']}")


if __name__ == "__main__":
    results = train()
    best_weights = f"{train_run}/{runName}/weights/best.pt"
    validate(best_weights_path=best_weights)
