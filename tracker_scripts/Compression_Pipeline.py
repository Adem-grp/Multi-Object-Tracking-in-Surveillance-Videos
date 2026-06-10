import os
import time
import shutil
import argparse

import pylab as p
import yaml
import cv2
import torch
import torch.nn.utils.prune as torch_prune
import numpy as np
import pandas as pd
import motmetrics as mm
from pathlib import Path

from ultralytics import YOLO
from trackeval.metrics import HOTA
from boxmot.trackers.bytetrack.byte_tracker import BYTETracker
from deep_sort_realtime.deepsort_tracker import DeepSort
from ocsort.ocsort import OCSort
from Tracker_Pipeline import *  # instead of duplicate code I will use what is necessary from that script
from utils.cvat_labeling_init import output_path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DetectorWeights = r"D:\runs_final\detect\all_datasets\weights\best.pt"
TrainRunDir = r"D:\runs_final\detect\all_datasets"
OutDir = Path(r"C:\Users\k2549603\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\tracker_outputs")
CompressDir = OutDir / "compression"

# ReID is for deepsort it will be downloaded automatically after first run
##ReID_weights = Path("osnet_x0_25_msmt17.pt")
Imgsz = 640
DEVICE = 0
TrainDataYaml = str(CompressDir / "TrainData.yaml")
Prune_Grid = [0.2, 0.4, 0.6]

# Datasets since there are multiple videos per dataset it is important to arrange the ground truths and video paths
# change these paths after putting the datasets folder to the hard drive
DATASETS = {
    "avenue": [
        (r"D:\datasets\avenue_yolo\test\img1",
         r"D:\datasets\avenue_yolo\test\gt\gt.txt"), ],
    "UCSD": [
        (r"D:\datasets\ucsd_yolo\test\ped1_test1\img1",
         r"D:\datasets\ucsd_yolo\test\ped1_test1\gt\gt.txt"),
        (r"D:\datasets\ucsd_yolo\test\ped1_test2\img1",
         r"D:\datasets\ucsd_yolo\test\ped1_test2\gt\gt.txt"),
        (r"D:\datasets\ucsd_yolo\test\ped1_test3\img1",
         r"D:\datasets\ucsd_yolo\test\ped1_test3\gt\gt.txt"),
        (r"D:\datasets\ucsd_yolo\test\ped2_test1\img1",
         r"D:\datasets\ucsd_yolo\test\ped2_test1\gt\gt.txt"),
        (r"D:\datasets\ucsd_yolo\test\ped2_test2\img1",
         r"D:\datasets\ucsd_yolo\test\ped2_test2\gt\gt.txt"),
        (r"D:\datasets\ucsd_yolo\test\ped2_test3\img1",
         r"D:\datasets\ucsd_yolo\test\ped2_test3\gt\gt.txt"),
    ],
    "shanghai": [
        (r"D:\datasets\shanghaitech_yolo\test\shanghai_10\img1",
         r"D:\datasets\shanghaitech_yolo\test\shanghai_10\gt\gt.txt"),
        (r"D:\datasets\shanghaitech_yolo\test\shanghai_128\img1",
         r"D:\datasets\shanghaitech_yolo\test\shanghai_128\gt\gt.txt"),
        (r"D:\datasets\shanghaitech_yolo\test\shanghai_164\img1",
         r"D:\datasets\shanghaitech_yolo\test\shanghai_164\gt\gt.txt"),
    ],
    "gmot": [
        (r"D:\datasets\gmot_yolo\test\airplane-1\img1",
         r"D:\datasets\gmot_yolo\test\airplane-1\gt\gt.txt"),
        (r"D:\datasets\gmot_yolo\test\ball-1\img1",
         r"D:\datasets\gmot_yolo\test\ball-1\gt\gt.txt"),
        (r"D:\datasets\gmot_yolo\test\ball-2\img1",
         r"D:\datasets\gmot_yolo\test\ball-2\gt\gt.txt"),
        (r"D:\datasets\gmot_yolo\test\ball-3\img1",
         r"D:\datasets\gmot_yolo\test\ball-3\gt\gt.txt"),
        (r"D:\datasets\gmot_yolo\test\balloon-0\img1",
         r"D:\datasets\gmot_yolo\test\balloon-0\gt\gt.txt"),
        (r"D:\datasets\gmot_yolo\test\bird-2\img1",
         r"D:\datasets\gmot_yolo\test\bird-2\gt\gt.txt"),
        (r"D:\datasets\gmot_yolo\test\bird-3\img1",
         r"D:\datasets\gmot_yolo\test\bird-3\gt\gt.txt"),
        (r"D:\datasets\gmot_yolo\test\boat-1\img1",
         r"D:\datasets\gmot_yolo\test\boat-1\gt\gt.txt"), ],
    # turn gmot to video and add its gt as well
}

EvalTrackers = {  # when all combinations are finished just fill with the best tracker and its hyper parameters
    "deepsort": {
        "max_dist": [0.2, 0.3, 0.5, 0.6],
        "max_age": [20, 30, 50, 70, 100],
        "n_init": [1, 3, 5],
        "max_iou_dist": [0.2, 0.3, 0.5, 0.7, 0.9],
    },
    "bytetrack": {
        "track_high_thresh": [0.4, 0.5, 0.6, 0.7, 0.8],
        "track_buffer": [20, 30, 40, 60],
        "match_thresh": [0.7, 0.8, 0.9],
    },
    "ocsort": {
        "det_thresh": [0.4, 0.5, 0.6, 0.7],
        "max_age": [20, 30, 40, 50, 70],
        "min_hits": [1, 3],
        "iou_threshold": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    },
}


# Build TrainData.yaml
def build_train_yaml():
    CompressDir.mkdir(parents=True, exist_ok=True)
    yaml_out = Path(TrainDataYaml)
    if yaml_out.exists():
        print("Train data already exists")
        return str(yaml_out)
    run_dir = Path(TrainRunDir)
    found = list(run_dir.glob("*.yaml")) + list(run_dir.parent.glob("*.yaml"))
    if found:
        shutil.copy(found[0], yaml_out)
        print("Train data copied to {}".format(yaml_out))
        return str(yaml_out)
    project_root = Path(DetectorWeights).parents[
        3]  # double check before running  if the parents are correct and what not here
    datasets_dir = project_root / "datasets"
    train_imgs, val_imgs = [], []
    for d in datasets_dir.iterdir():
        if not d.is_dir():
            continue
        ti = d / "train" / "images"
        vi = d / "val" / "images"
        if ti.exists():
            train_imgs.append(str(ti))
        if vi.exists():
            val_imgs.append(str(vi))
    with open(yaml_out, "w") as f:
        yaml.dump(
            {"path": str(datasets_dir), "train": train_imgs, "val": val_imgs, "nc": 10,
             "names": {0: "airplane", 1: "fish", 2: "ball", 3: "bird", 4: "boat", 5: "balloon", 6: "person",
                       7: "insect",
                       8: "stock", 9: "car"}}, f)
        print("Yaml Created")
        return str(yaml_out)


def get_prunable_convs(nn_model):
    prunable_convs = []
    for name, module in nn_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if "cv3" not in name and "dfl" not in name:  # head output layers in YOLO are skipped
                prunable_convs.append((name, module))
    return prunable_convs


def apply_structured_pruning(nn_model, sparsity):
    # L1-norm filter pruning removes the weakest output filters
    # while dim=0 targets output channels, n=1 uses l1 norm to score them
    #prune.remove() bakes the mask permanently into the weight tensor
    prunable = get_prunable_convs(nn_model)
    for name, module in prunable:
        n_filters = module.weight.shape[0]
        n_prune = min(max(1, int(n_filters * sparsity)), n_filters - 1)
        torch_prune.ln_structured(module, name="weight", amount=n_prune, n=1, dim=0)
        torch_prune.remove(module, "weight")
    nonzero = sum(p.nonzero().shape[0] for p in nn_model.parameters())
    total = sum(p.numel() for p in nn_model.parameters())
    print(f"Pruned: {len(prunable)} layers | sparsity: {sparsity:.0%}")
    print(f"nonzero: {nonzero:,}/{total:,} ({nonzero / total:.2%})")
    return nn_model


def finetune(weights_path, out_dir, epochs):
    model = YOLO(str(weights_path))
    results = model.train(
        data=TrainDataYaml,
        epochs=epochs,
        imgsz=Imgsz,
        batch=16,
        device=DEVICE,
        project=str(out_dir),
        name="finetune",
        exist_ok=True,
        verbose=False,
        lr0=1e-4,
        lrf=1e-5,
        warmup_epochs=1,
        mosaic=0.5,
        mixup=0.0,
        patience=5
    )
    best = Path(model.trainer.best)
    if not best.exists():
        best = Path(model.trainer.last)
    return best


def quantise_fp16(weights_path, out_path):
    model = YOLO(str(weights_path))
    model.export(format="pt", imgsz=Imgsz, device=DEVICE, half=True)
    exported = weights_path.parent / (weights_path.stem + ".pt")
    if exported.exists():
        shutil.copy(exported, out_path)
        print("Exported to", out_path)
        return output_path
    print("Export not found, keeping fp32")
    shutil.copy(weights_path, out_path)
    return output_path


def quantisation_int8(weights_path, out_path):
    yolo = YOLO(str(weights_path))
    model = yolo.model.float().cpu()  # float seems to be deprecated
    model.eval()
    quantised = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear},
        dtype=torch.qint8,
    )
    torch.save({"model_state": quantised.state_dict(),
                "base_weights": str(weights_path)}, out_path)
    print(f"INT8 state dict saved to {out_path}")
    return out_path


def build_variants(epochs):
    variants = []  # basically we are taking the baseline and different versions of pruned and quantised versions and measure them
    # these are the all possible combinations I believe we go with sparsity meaning pruning degree
    # we arrange pruning with and without quantisation also we check the version where we fine tune the model after pruning it as well
    variants.append(
        {"name": "baseline", "sparsity": 0.0, "finetune": False, "quantisation": "none", "weights_path": None,
         "loadable": True})
    for sparsity in Prune_Grid:
        pct = int(sparsity * 100)
        variants.append({"name": f"prune{pct}", "sparsity": sparsity, "finetune": False, "quantisation": "none",
                         "weights_path": None, "loadable": True})
        variants.append(
            {"name": f"prune{pct}_ft{epochs}ep", "sparsity": sparsity, "finetune": True, "quantisation": "none",
             "weights_path": None, "loadable": True})
        variants.append(
            {"name": f"prune{pct}_ft{epochs}ep_fp16", "sparsity": sparsity, "finetune": True, "quantisation": "fp16",
             "weights_path": None, "loadable": True})
        variants.append(
            {"name": f"prune{pct}_ft{epochs}ep_int8", "sparsity": sparsity, "finetune": True, "quantisation": "int8",
             "weights_path": None, "loadable": False})
    variants.append(
        {"name": "fp16_only", "sparsity": 0.0, "finetune": False, "quantisation": "fp16", "weights_path": None,
         "loadable": True})
    variants.append(
        {"name": "int8_only", "sparsity": 0.0, "finetune": False, "quantisation": "int8", "weights_path": None,
         "loadable": False})
    return variants


def run_compression(epochs):
    CompressDir.mkdir(exist_ok=True, parents=True)
    build_train_yaml()
    variants = build_variants(epochs)
    for variant in variants:
        varDir = CompressDir / variant["name"]
        varDir.mkdir(exist_ok=True, parents=True)
        pruned_pt = varDir / "pruned.pt"
        ft_pt = varDir / "finetuned.pt"
        final_pt = varDir / "model_final.pt"
        print(f"\n Variant: {variant['name']}\n")
        # baseline
        if variant["sparsity"] == 0.0 and not variant["finetune"] and variant["quantisation"] == "none":
            shutil.copy(DetectorWeights, final_pt)
            variant["weights_path"] = final_pt
            print("Baseline Copied")
            continue
        working = Path(DetectorWeights)

        # pruning steps
        if variant["sparsity"] > 0.0:
            if pruned_pt.exists():
                print("Pruned weights already exists")
            else:
                yolo = YOLO(str(working))
                yolo.model = apply_structured_pruning(yolo.model, variant["sparsity"])
                yolo.save(str(pruned_pt))
                print("Pruned weights saved")
            working = pruned_pt
        # Fine tune step basically each variant enters its own category and the action is taken accordingly
        if variant["finetune"]:
            if ft_pt.exists():
                print("Finetuned weights already exists")
            else:
                best_ft = finetune(working, varDir, epochs)
                shutil.copy(best_ft, ft_pt)
            working = ft_pt
        # Quantisation for the ones which have both quantisation and prune step and fine tune step each will enter each if statement basically
        if variant["quantisation"] == "fp16":
            q_path = varDir / "model_f16.pt"
            if not q_path.exists():
                q_path = quantise_fp16(working, q_path)  # this was a bug there was no assignment double check it
            variant["weights_path"] = q_path
        elif variant["quantisation"] == "int8":
            q_path = varDir / "model_int8.pt"  # change all to .pt tho double check first
            if not q_path.exists():
                q_path = quantisation_int8(working, q_path)
            variant["weights_path"] = q_path
        else:
            shutil.copy(working, final_pt)
            variant["weights_path"] = final_pt
    print("All compression variants are created")
    manifest = pd.DataFrame([{"name": variant["name"], "sparsity": variant["sparsity"], "finetune": variant["finetune"],
                              "quantisation": variant["quantisation"],
                              "weights_path": str(variant["weights_path"]) if variant["weights_path"] else "",
                              "loadable": variant["loadable"]} for variant in variants])
    manifest.to_csv(CompressDir / "manifest_variants.csv", index=False)
    return variants


def run_eval(variants, tracker_name):
    cfg = EvalTrackers[tracker_name]
    tracker_params = cfg["params"]
    conf, iou = cfg["conf"], cfg["iou"]
    rows =[]
    for variant in variants:
        if not v.get("loadable",True):
            print(f"\nVariant: {variant['name']} skip int8 state dict needs tensorrt for conv inference\n")
            rows.append({
                "variant": variant["name"],
                "sparsity": variant["sparsity"],
                "quantisation": variant["quantisation"],
                "tracker" : tracker_name,
                "dataset": "N/A",
                "note": "INT8 skipped use Tensor RT for full Conv int8 inference",
                **{k: None for k in ["HOTA (%)", "MOTA (%)", "IDF1 (%)","FPS","latency mean (ms)", "Peak VRAM ( MB)"]},
            })
            continue
        wp = variant.get("weights_path")
        if wp is None or not Path(str(wp)).exists():
            print(f"\nVariant: {variant['name']} weights not found\n")
            continue
        print(f" Evaluating {variant['name']}\n tracker: {tracker_name}")
        for dataset_name in DATASETS:
            print(f" Dataset: {dataset_name}\n")
            tmp_dir = CompressDir / "_tmp" / dataset_name
            tmp_dir.mkdir(exist_ok=True, parents=True)
            avg = evaluate_dataset(dataset_name,wp,tracker_name,tracker_params,tmp_dir,conf,iou) # rewrite this and tracker_on_sequence

