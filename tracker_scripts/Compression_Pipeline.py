""" Compression Pipeline — Structured Pruning + Post-Training Quantisation
 Takes the trained YOLO detector, applies pruning and quantisation at various
 levels, fine-tunes after pruning to recover accuracy, then runs the full
 tracker pipeline on each compressed model to measure the quality/speed tradeoff.
"""
import os
import time
import shutil
import yaml  # for writing the training data yaml
import cv2
import torch
import torch.nn.utils.prune as torch_prune  # PyTorch's built-in pruning utilities
import numpy as np
import pandas as pd
import motmetrics as mm
from pathlib import Path
from ultralytics import YOLO
from trackeval.metrics import HOTA
from boxmot.trackers.bytetrack.byte_tracker import BYTETracker
from deep_sort_realtime.deepsort_tracker import DeepSort
from ocsort.ocsort import OCSort
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DetectorWeights = r"D:\runs_final\detect\all_datasets\weights\best.pt"
TrainRunDir = r"D:\runs_final\detect\all_datasets"  # folder of the original training run, used to find the yaml
OutDir = Path(r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\tracker_outputs")
CompressDir = OutDir / "compression"  # all compression outputs go here
Imgsz = 640
DEVICE = 0

# path where the training yaml will be written or found — used for fine-tuning after pruning
TrainDataYaml = r"C:\Users\k2549603\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\gmot.yaml"

# sparsity levels to test — 0.2 removes 20% of filters, 0.4 removes 40%, etc.
PruneLevels = [0.2, 0.4, 0.6]

# Datasets to evaluate on — each dataset is a list of (img_folder, gt_path) tuples
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
}

EvalTrackers = {  # best hyperparameters of the best performing tracker
    "bytetrack": {
        "params": {"track_high_thresh": 0.4, "track_buffer": 20, "match_thresh": 0.8},
        "conf": 0.01,
        "iou": 0.5,
    },
}


def build_train_yaml():
    CompressDir.mkdir(parents=True, exist_ok=True)
    yaml_out = Path(TrainDataYaml)

    if yaml_out.exists():  # already built from a previous run, reuse it
        print(f"[YAML] Using existing {yaml_out}")
        return str(yaml_out)

    # try to find the yaml used in the original training run
    run_dir = Path(TrainRunDir)
    found = list(run_dir.glob("*.yaml")) + list(run_dir.parent.glob("*.yaml"))
    if found:
        shutil.copy(found[0], yaml_out)  # copy it to CompressDir so it's self-contained
        print(f"[YAML] Copied from {found[0]}")
        return str(yaml_out)

    # fallback — scan the datasets folder and build a yaml from whatever train/val folders exist
    project_root = Path(DetectorWeights).parents[3]
    datasets_dir = project_root / "datasets"
    train_imgs, val_imgs = [], []
    for d in datasets_dir.iterdir():
        if not d.is_dir():
            continue
        ti = d / "train" / "images"
        vi = d / "val" / "images"
        if ti.exists(): train_imgs.append(str(ti))
        if vi.exists(): val_imgs.append(str(vi))
    with open(yaml_out, "w") as f:
        # write a yaml
        yaml.dump({"path": str(datasets_dir), "train": train_imgs,
                   "val": val_imgs, "nc": 1, "names": {0: "person"}}, f)
    print(f"[YAML] Built from dataset folders: {yaml_out}")
    return str(yaml_out)


def get_prunable_convs(nn_model):
    prunable = []
    for name, module in nn_model.named_modules():  # walk every layer in the network
        if isinstance(module, torch.nn.Conv2d):  # only conv layers are prunable
            # skip detection head output convs — cv3 and dfl produce the final predictions
            # removing filters from them would change the output shape and break the model
            if "cv3" not in name and "dfl" not in name:
                prunable.append((name, module))  # store (layer_name, layer_object) pairs
    return prunable


def apply_structured_pruning(nn_model, sparsity: float):
    prunable = get_prunable_convs(nn_model)  # get all layers safe to prune
    for name, module in prunable:
        n_filters = module.weight.shape[0]  # total number of output filters in this layer
        n_prune = min(max(1, int(n_filters * sparsity)), n_filters - 1)
        # ln_structured: score each filter by L1 norm (sum of absolute weight values)
        torch_prune.ln_structured(module, name="weight", amount=n_prune, n=1, dim=0)
        # remove() bakes the pruning mask permanently into the weight tensor
        # without this the mask is a separate hook — still applied but not saved in the weights
        torch_prune.remove(module, "weight")
    # count non-zero params across the whole network to show real achieved sparsity
    nonzero = sum(p.nonzero().shape[0] for p in nn_model.parameters())
    total = sum(p.numel() for p in nn_model.parameters())
    print(f"  Pruned {len(prunable)} layers | sparsity={sparsity:.0%} | "
          f"non-zero: {nonzero:,}/{total:,} ({nonzero / total:.1%})")
    return nn_model


def finetune(weights_path: Path, out_dir: Path, epochs: int):
    print(f"  Fine-tuning for {epochs} epochs ...")
    if torch.cuda.is_available():
        print("  Using GPU")
    else:
        print(" Not Using CPU")
    model = YOLO(str(weights_path))  # train the detector for a few epochs to recover accuracy after pruning
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
        optimizer="SGD",
        lr0=1e-4,
        lrf=1e-5,
        warmup_epochs=1,  # warmup for 1 epoch to avoid divergence at the start
        mosaic=0.5,
        mixup=0.0,
        patience=5,
    )
    if torch.cuda.is_available():
        print(f"Peak VRAM during training {torch.cuda.max_memory_allocated(DEVICE) / 1024 / 1024:.1f} MB")
    # model.trainer.best points directly to best.pt saved during training
    best = Path(model.trainer.best)
    if not best.exists():
        best = Path(model.trainer.last)  # fallback to last checkpoint if best wasn't saved
    print(f"  Fine-tune done: {best}")
    return best


def quantise_fp16(weights_path: Path, out_path: Path):
    print("  Exporting FP16 ...")
    model = YOLO(str(weights_path))

    model.export(format="torchscript", imgsz=Imgsz, half=True, device=DEVICE)

    exported = weights_path.parent / (weights_path.stem + ".torchscript")
    if exported.exists():
        shutil.copy(exported, out_path)  # copy to the variant's output folder
        print(f"  FP16 saved: {out_path}")
        return out_path
    # if export failed for some reason, fall back to keeping the FP32 weights
    print("  [WARN] FP16 export not found, keeping FP32")
    shutil.copy(weights_path, out_path)
    return out_path


def quantise_int8(weights_path: Path, out_path: Path):
    print("  Applying INT8 quantisation ...")
    yolo = YOLO(str(weights_path))
    model = yolo.model.float().cpu()
    model.eval()
    quantised = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8,
    )

    torch.save({"model_state": quantised.state_dict(),
                "base_weights": str(weights_path)}, out_path)
    print(f"  INT8 state dict saved: {out_path}")
    return out_path


# Variant list defines every model configuration to produce

def build_variants(epochs: int):
    variants = []

    # baseline — original weights unchanged, used as the reference point
    variants.append({"name": "baseline", "sparsity": 0.0, "finetune": False,
                     "quantisation": "none", "weights_path": None, "loadable": True})

    for sparsity in PruneLevels:
        pct = int(sparsity * 100)

        # prune only — no fine-tune, to measure raw pruning impact
        variants.append({"name": f"prune{pct}", "sparsity": sparsity, "finetune": False,
                         "quantisation": "none", "weights_path": None, "loadable": True})

        # prune + fine-tune — recover accuracy after pruning
        variants.append({"name": f"prune{pct}_ft{epochs}ep", "sparsity": sparsity, "finetune": True,
                         "quantisation": "none", "weights_path": None, "loadable": True})

        # prune + fine-tune + FP16 — further compress the fine-tuned model
        variants.append({"name": f"prune{pct}_ft{epochs}ep_fp16", "sparsity": sparsity, "finetune": True,
                         "quantisation": "fp16", "weights_path": None, "loadable": True})

        # prune + fine-tune + INT8 — maximum compression, but not directly loadable by YOLO()
        variants.append({"name": f"prune{pct}_ft{epochs}ep_int8", "sparsity": sparsity, "finetune": True,
                         "quantisation": "int8", "weights_path": None, "loadable": False})

    # quantisation only on original weights — no pruning, to isolate quantisation effect
    variants.append({"name": "fp16_only", "sparsity": 0.0, "finetune": False,
                     "quantisation": "fp16", "weights_path": None, "loadable": True})
    variants.append({"name": "int8_only", "sparsity": 0.0, "finetune": False,
                     "quantisation": "int8", "weights_path": None, "loadable": False})
    return variants


# Compression runner creates all model files

def run_compression(epochs: int):
    CompressDir.mkdir(parents=True, exist_ok=True)
    build_train_yaml()
    variants = build_variants(epochs)

    for v in variants:
        vdir = CompressDir / v["name"]
        vdir.mkdir(parents=True, exist_ok=True)
        pruned_pt = vdir / "pruned.pt"  # intermediate: pruned weights before fine-tune
        ft_pt = vdir / "finetuned.pt"  # intermediate: fine-tuned weights
        final_pt = vdir / "model.pt"  # final output: what evaluation loads
        print(f"\n[Variant] {v['name']}")

        
        if v["sparsity"] == 0.0 and not v["finetune"] and v["quantisation"] == "none":
            shutil.copy(DetectorWeights, final_pt)
            v["weights_path"] = final_pt  # record where the weights ended up
            print("  Baseline copied.")
            continue  # skip to next variant

        working = Path(DetectorWeights)  # start from original weights for every variant

        # --- pruning step ---
        if v["sparsity"] > 0.0:
            if pruned_pt.exists():  # skip if already done from a previous run
                print("  Pruned weights exist, skipping.")
            else:
                yolo = YOLO(str(working))  # load original model
                yolo.model = apply_structured_pruning(yolo.model, v["sparsity"])  # prune in place
                yolo.save(str(pruned_pt))  # save pruned weights
                print(f"  Pruned saved: {pruned_pt}")
            working = pruned_pt  # next step works from pruned weights

        # --- fine-tune step ---
        if v["finetune"]:
            if ft_pt.exists():  # skip if already done
                print("  Fine-tuned weights exist, skipping.")
            else:
                best_ft = finetune(working, vdir, epochs)  # run fine-tuning
                shutil.copy(best_ft, ft_pt)  # copy best checkpoint to ft_pt
            working = ft_pt  # next step works from fine-tuned weights

        # --- quantisation step ---
        if v["quantisation"] == "fp16":
            q_path = vdir / "model_fp16.torchscript"
            if not q_path.exists():
                q_path = quantise_fp16(working, q_path)  # export to FP16 TorchScript
            v["weights_path"] = q_path  # record path for evaluation
        elif v["quantisation"] == "int8":
            q_path = vdir / "model_int8.pt"
            if not q_path.exists():
                q_path = quantise_int8(working, q_path)  # apply dynamic INT8
            v["weights_path"] = q_path
        else:
            # no quantisation — copy working weights to final_pt
            shutil.copy(working, final_pt)
            v["weights_path"] = final_pt

    print("\n[Compression] All variants created.")
    # save manifest CSV so evaluate mode can reload the variant list without recompressing
    manifest = pd.DataFrame([{
        "name": v["name"],
        "sparsity": v["sparsity"],
        "finetune": v["finetune"],
        "quantisation": v["quantisation"],
        "weights_path": str(v["weights_path"]) if v["weights_path"] else "",
        "loadable": v["loadable"],  # False for INT8 — evaluation skips these
    } for v in variants])
    manifest.to_csv(CompressDir / "variants_manifest.csv", index=False)
    return variants


def build_tracker(tracker_name, params):
    if tracker_name == "deepsort":
        return DeepSort(max_cosine_distance=params["max_dist"], max_age=params["max_age"],
                        n_init=params["n_init"], max_iou_distance=params["max_iou_dist"],
                        embedder_gpu=True, half=False)
    elif tracker_name == "bytetrack":
        return BYTETracker(track_thresh=params["track_high_thresh"],
                           match_thresh=params["match_thresh"],
                           track_buffer=params["track_buffer"], frame_rate=30)
    elif tracker_name == "ocsort":
        return OCSort(det_thresh=params["det_thresh"], max_age=params["max_age"],
                      min_hits=params["min_hits"], iou_threshold=params["iou_threshold"],
                      use_byte=False)
    raise ValueError(f"Unknown tracker: {tracker_name}")


def tracker_on_sequence(img_folder, model, tracker_name, tracker_params,
                        output_path, conf, iou):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # collect all frame images in sorted order so tracking is temporally consistent
    frame_paths = sorted(Path(img_folder).glob("*.jpg")) or sorted(Path(img_folder).glob("*.png"))
    if not frame_paths:
        raise RuntimeError(f"No frames in {img_folder}")
    tracker = build_tracker(tracker_name, tracker_params)
    mot_lines, frame_times, total_time = [], [], 0.0

    for frame_idx, img_path in enumerate(frame_paths):
        frame = cv2.imread(str(img_path))  # read frame as BGR numpy array
        if frame is None:
            continue
        if frame_idx == 1 and torch.cuda.is_available():
            # reset after frame 0 so the warmup allocation doesn't inflate VRAM readings
            torch.cuda.reset_peak_memory_stats()

        t_start = time.perf_counter()
        results = model.predict(frame, conf=conf, iou=iou, imgsz=Imgsz, verbose=False)

        # extract detections from YOLO results into a flat list
        dets = []
        if results[0].boxes is not None and len(results[0].boxes):
            boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1,y1,x2,y2]
            confs = results[0].boxes.conf.cpu().numpy()  # confidence scores
            clss = results[0].boxes.cls.cpu().numpy()  # class indices
            for box, dc, cls in zip(boxes, confs, clss):
                dets.append([*box, dc, cls])  # combine into [x1,y1,x2,y2,conf,cls]
        dets_np = np.array(dets) if dets else np.empty((0, 6))

        # each tracker has a different update interface — handle separately
        active_tracks = []
        if tracker_name == "deepsort":
            # deep_sort_realtime expects list of ([x1,y1,w,h], conf, cls_id)
            ds_dets = [([d[0], d[1], d[2] - d[0], d[3] - d[1]], d[4], 0) for d in dets_np]
            for t in tracker.update_tracks(ds_dets, frame=frame):
                if t.is_confirmed():  # only output confirmed tracks, not tentative ones
                    x1, y1, x2, y2 = t.to_ltrb()  # get bounding box in xyxy format
                    active_tracks.append((x1, y1, x2, y2, t.track_id, t.get_det_conf() or 1.0))
        elif tracker_name == "bytetrack":
            # boxmot BYTETracker expects [x1,y1,x2,y2,conf,cls] numpy array
            bt_dets = dets_np[:, :6] if len(dets_np) else np.empty((0, 6))
            for t in tracker.update(bt_dets, frame):
                # returns numpy array rows [x1,y1,x2,y2,tid,conf,cls]
                active_tracks.append((t[0], t[1], t[2], t[3], int(t[4]), t[5]))
        elif tracker_name == "ocsort":
            # ocsort expects a torch tensor [x1,y1,x2,y2,conf,cls]
            oc_in = torch.from_numpy(dets_np[:, :6].astype(np.float32)) if len(dets_np) else torch.zeros((0, 6))
            for t in tracker.update(oc_in, frame):
                # returns [x1,y1,x2,y2,tid,cls,conf] — note conf is at index 6
                active_tracks.append((t[0], t[1], t[2], t[3], int(t[4]), t[6]))

        t_end = time.perf_counter()
        total_time += (t_end - t_start)
        frame_times.append((t_end - t_start) * 1000)  # store in ms

        # write each active track as a MOT1.1 format line
        for x1, y1, x2, y2, tid, tc in active_tracks:
            mot_lines.append(f"{frame_idx},{tid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{tc:.4f},-1,-1,-1")

    output_path.write_text("\n".join(mot_lines), encoding="utf-8")
    n = len(frame_paths)
    fps = n / total_time if total_time > 0 else 0.0
    lm = round(float(np.mean(frame_times)), 2)  # mean latency per frame in ms
    lp = round(float(np.percentile(frame_times, 95)), 2)  # 95th percentile latency
    vram = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1) if torch.cuda.is_available() else 0.0
    print(f"  {Path(img_folder).parent.name}/{Path(img_folder).name} | {n}f | "
          f"{fps:.1f} FPS | lat {lm:.1f}ms | VRAM {vram}MB")
    return {"fps": round(fps, 1), "frames": n, "latency_mean_ms": lm,
            "latency_p95_ms": lp, "peak_vram_mb": vram}


def load_mot(mot_path):
    path = Path(mot_path)
    if not path.exists() or path.stat().st_size == 0:  # handle missing or empty files
        return pd.DataFrame(columns=["frame", "id", "x", "y", "w", "h"])
    df = pd.read_csv(mot_path, header=None,
                     names=["frame", "id", "x", "y", "w", "h", "conf", "cx", "cy", "cz"])
    return df[["frame", "id", "x", "y", "w", "h"]].copy()  # only keep the columns we need


def compute_hota(gt_df, pred_df, iou_threshold=0.5):
    hota_metric = HOTA({"THRESHOLD": iou_threshold})
    all_frames = sorted(set(gt_df["frame"]) | set(pred_df["frame"]))
    all_gt_ids = sorted(gt_df["id"].unique().tolist())
    all_pred_ids = sorted(pred_df["id"].unique().tolist())
    # remap IDs to 0-based — trackeval uses IDs as array indices internally
    gt_id_map = {v: i for i, v in enumerate(all_gt_ids)}
    pred_id_map = {v: i for i, v in enumerate(all_pred_ids)}
    gt_ids_l, pred_ids_l, sim_l = [], [], []
    num_gt, num_pred = 0, 0
    for frame in all_frames:
        gf = gt_df[gt_df["frame"] == frame]
        pf = pred_df[pred_df["frame"] == frame]
        # remap each ID to its 0-based index
        gi = np.array([gt_id_map[i] for i in gf["id"].tolist()], dtype=np.int32)
        pi = np.array([pred_id_map[i] for i in pf["id"].tolist()], dtype=np.int32)
        gb = gf[["x", "y", "w", "h"]].to_numpy()
        pb = pf[["x", "y", "w", "h"]].to_numpy()
        # compute IoU similarity matrix — 1 - distance because iou_matrix returns distance
        sim = 1.0 - mm.distances.iou_matrix(gb, pb) if len(gb) and len(pb) \
            else np.zeros((len(gb), len(pb)))
        gt_ids_l.append(gi);
        pred_ids_l.append(pi);
        sim_l.append(sim)
        num_gt += len(gi);
        num_pred += len(pi)
    res = hota_metric.eval_sequence({
        "num_timesteps": len(all_frames),  # total frames in the sequence
        "num_gt_dets": num_gt,  # total GT detections across all frames
        "num_tracker_dets": num_pred,  # total predicted detections
        "num_gt_ids": len(all_gt_ids),  # unique GT track IDs
        "num_tracker_ids": len(all_pred_ids),  # unique predicted track IDs
        "gt_ids": gt_ids_l,  # per-frame GT ID arrays
        "tracker_ids": pred_ids_l,  # per-frame predicted ID arrays
        "similarity_scores": sim_l,  # per-frame IoU similarity matrices
    })
    # res["HOTA"] is an array of scores at different IoU thresholds — take the mean
    return float(np.mean(res["HOTA"])) * 100


def evaluate_sequence(gt_path, pred_path, iou_threshold=0.5):
    gt_df = load_mot(gt_path)
    pred_df = load_mot(pred_path)
    acc = mm.MOTAccumulator(auto_id=True)  # accumulates frame-by-frame matching results
    for frame in sorted(set(gt_df["frame"]) | set(pred_df["frame"])):
        gf = gt_df[gt_df["frame"] == frame]
        pf = pred_df[pred_df["frame"] == frame]
        gb = gf[["x", "y", "w", "h"]].values.tolist()
        pb = pf[["x", "y", "w", "h"]].values.tolist()
        # max_iou=1-threshold means detections further than threshold IoU distance are not matched
        dist = mm.distances.iou_matrix(gb, pb, max_iou=1 - iou_threshold) \
            if gb and pb else np.empty((len(gb), len(pb)))
        acc.update(gf["id"].tolist(), pf["id"].tolist(), dist)  # update accumulator for this frame
    mh = mm.metrics.create()
    s = mh.compute(acc, metrics=["mota", "idf1", "num_switches",
                                 "mostly_tracked", "mostly_lost"], name="e")
    r = s.to_dict(orient="records")[0]  # convert summary DataFrame to a single dict
    r["mota_pct"] = round(float(r["mota"]) * 100, 2)  # convert 0-1 to percentage
    r["idf1_pct"] = round(float(r["idf1"]) * 100, 2)
    r["hota_pct"] = round(compute_hota(gt_df, pred_df, iou_threshold), 2)
    return r


def evaluate_dataset(dataset_name, weights_path, tracker_name, tracker_params,
                     tmp_dir, conf, iou):
    model = YOLO(str(weights_path))  # load the compressed model for this variant
    clips = DATASETS[dataset_name]  # list of (img_folder, gt_path) tuples for this dataset
    clip_rows, timing_rows = [], []
    for i, (img_folder, gt_path) in enumerate(clips):
        seq_name = Path(img_folder).parent.name  # e.g. "ped1_test1"
        out_file = tmp_dir / f"{seq_name}_{tracker_name}.txt"  # temp MOT output file
        print(f"   Clip {i + 1}/{len(clips)}: {seq_name}")
        timing = tracker_on_sequence(img_folder, model, tracker_name,
                                     tracker_params, out_file, conf, iou)
        timing_rows.append(timing)
        clip_rows.append(evaluate_sequence(gt_path, str(out_file)))
    # average all metrics across clips for this dataset
    avg = {}
    for k in ["mota_pct", "idf1_pct", "hota_pct", "num_switches",
              "mostly_tracked", "mostly_lost"]:
        vals = [row[k] for row in clip_rows if k in row]
        avg[k] = round(float(np.mean(vals)), 3) if vals else 0.0
    avg["fps_mean"] = round(float(np.mean([t["fps"] for t in timing_rows])), 1)
    avg["latency_mean_ms"] = round(float(np.mean([t["latency_mean_ms"] for t in timing_rows])), 2)
    avg["latency_p95_ms"] = round(float(np.mean([t["latency_p95_ms"] for t in timing_rows])), 2)
    avg["peak_vram_mb"] = round(float(np.max([t["peak_vram_mb"] for t in timing_rows])),
                                1)  # max not mean — worst case VRAM
    return avg


# Evaluation runner: it runs every loadable variant through the tracker pipeline

def run_evaluation(variants, tracker_name):
    cfg = EvalTrackers[tracker_name]  # look up best params for this tracker
    tracker_params = cfg["params"]
    conf, iou = cfg["conf"], cfg["iou"]
    rows = []  # will hold one row per variant per dataset

    for v in variants:
        # INT8 state dicts can't be loaded by YOLO() — skip but note in CSV
        if not v.get("loadable", True):
            print(f"[SKIP] {v['name']} — INT8 state dict needs TensorRT for Conv inference")
            rows.append({"variant": v["name"], "sparsity": v["sparsity"],
                         "quantisation": v["quantisation"], "tracker": tracker_name,
                         "dataset": "N/A",
                         "note": "INT8 skipped — use TensorRT for full Conv INT8 inference",
                         **{k: None for k in ["HOTA (%)", "MOTA (%)", "IDF1 (%)",
                                              "FPS", "Latency mean (ms)", "Peak VRAM (MB)"]}})
            continue

        wp = v.get("weights_path")
        if wp is None or not Path(str(wp)).exists():  # safety check — skip missing files
            print(f"[SKIP] {v['name']} — weights not found")
            continue

        print(f"\n[Eval] {v['name']} | tracker={tracker_name}")
        for dataset_name in DATASETS:
            print(f"  Dataset: {dataset_name}")
            tmp_dir = CompressDir / "_tmp" / dataset_name  # reused temp folder per dataset
            tmp_dir.mkdir(parents=True, exist_ok=True)
            avg = evaluate_dataset(dataset_name, wp, tracker_name,
                                   tracker_params, tmp_dir, conf, iou)
            rows.append({
                "variant": v["name"],  # e.g. "prune20_ft15ep_fp16"
                "sparsity": v["sparsity"],  # 0.2, 0.4, 0.6 or 0.0
                "finetune": v["finetune"],  # True/False
                "quantisation": v["quantisation"],  # "none", "fp16", "int8"
                "tracker": tracker_name,
                "dataset": dataset_name,
                "HOTA (%)": avg["hota_pct"],
                "MOTA (%)": avg["mota_pct"],
                "IDF1 (%)": avg["idf1_pct"],
                "ID Switches": avg["num_switches"],
                "MostlyTracked": avg["mostly_tracked"],
                "MostlyLost": avg["mostly_lost"],
                "FPS": avg["fps_mean"],
                "Latency mean (ms)": avg["latency_mean_ms"],
                "Latency p95 (ms)": avg["latency_p95_ms"],
                "Peak VRAM (MB)": avg["peak_vram_mb"],
            })
            print(f"    HOTA:{avg['hota_pct']}%  MOTA:{avg['mota_pct']}%  "
                  f"FPS:{avg['fps_mean']}  VRAM:{avg['peak_vram_mb']}MB")
            print("Cooling GPU for more accurate FPS")
            time.sleep(90)

    df = pd.DataFrame(rows)
    csv_path = CompressDir / f"compression_results_{tracker_name}_90s_break.csv"  # one CSV per tracker
    df.to_csv(csv_path, index=False)
    print(f"\n[Done] {csv_path}")
    _print_summary(df, tracker_name)
    return df


def _print_summary(df, tracker_name):
    numeric = df[df["HOTA (%)"].notna()].copy()  # exclude INT8 skipped rows
    if numeric.empty:
        return
    # average metrics across all datasets for each variant
    summary = (numeric.groupby("variant")[["HOTA (%)", "MOTA (%)", "IDF1 (%)",
                                           "FPS", "Latency mean (ms)", "Peak VRAM (MB)"]]
               .mean().round(2).reset_index()
               .merge(df[["variant", "sparsity", "quantisation"]].drop_duplicates(),
                      on="variant", how="left"))  # join back sparsity and quantisation columns
    print(f"\n[Summary — {tracker_name} — averaged across datasets]")
    print(summary.to_string(index=False))

    # Pareto front: walk from slowest to fastest, keep only variants that improve HOTA
    # a variant is dominated if something faster also has equal or better HOTA
    print("\n[Pareto front — best HOTA per FPS tier]")
    pareto, best_hota = [], -1
    for _, row in summary.sort_values("FPS").iterrows():
        if row["HOTA (%)"] > best_hota:  # only keep if better than everything seen so far
            best_hota = row["HOTA (%)"]
            pareto.append(row)
    print(pd.DataFrame(pareto)[["variant", "sparsity", "quantisation",
                                "HOTA (%)", "FPS", "Latency mean (ms)"]].to_string(index=False))


if __name__ == "__main__":
    CompressDir.mkdir(parents=True, exist_ok=True)

    print("Compression Pipeline")
    print("1. Compress (pruning + quantisation)")
    print("2. Evaluate (run tracker on existing compressed models)")
    mode_sel = int(input("Selection (1/2): "))
    tracker_name = "bytetrack"

    if mode_sel == 1:
        epochs = int(input("Fine-tuning epochs after pruning (default 15): ") or 15)
        run_compression(epochs=epochs)

    elif mode_sel == 2:
        manifest_path = CompressDir / "variants_manifest.csv"
        if not manifest_path.exists():
            raise RuntimeError("No variants_manifest.csv — run compress first (option 1)")
        manifest = pd.read_csv(manifest_path)
        variants = manifest.to_dict(orient="records")
        for v in variants:
            v["weights_path"] = Path(v["weights_path"]) if v["weights_path"] else None
        run_evaluation(variants, tracker_name)
