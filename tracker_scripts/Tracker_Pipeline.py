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

from trackeval.metrics import HOTA
from ultralytics import YOLO
from boxmot.trackers.bytetrack.byte_tracker import BYTETracker
from deep_sort_realtime.deepsort_tracker import DeepSort
from ocsort.ocsort import OCSort

# TrackEval needed for HOTA computation
try:
    import trackeval

    TRACKEVAL = True
except ImportError:
    TRACKEVAL = False
    print("Trackeval not available")

# change this paths for lab computer before running
DetectorWeights = r"D:\runs_final\detect\all_datasets\weights\best.pt"
OutDir = Path(r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\tracker_outputs")

# ReID is for deepsort it will be downloaded automatically after first run
ReID_weights = Path("osnet_x0_25_msmt17.pt")
Imgsz = 640
Def_conf = 0.01
Def_iou = 0.6
Byte_conf= 0.01
Byte_iou = 0.5
Deep_conf= 0.3
Deep_iou = 0.5
OC_conf = 0.2
OC_iou = 0.5
# Datasets since there are multiple videos per dataset it is important to arrange the ground truths and video paths
# change these paaths after putting the datasets folder to the hard drive
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

TrackerGrids = {
    "deepsort": {
        "max_dist": [0.1, 0.2, 0.3,0.5],
        "max_age": [30, 50, 70, 100],
        "n_init": [1, 3, 5],
        "max_iou_dist": [0.5, 0.7, 0.9],
    },
    "bytetrack": {
        "track_high_thresh": [0.4, 0.5, 0.6, 0.7, 0.8],
        "track_buffer": [20, 30, 40, 60],
        "match_thresh": [0.7, 0.8, 0.9],
    },
    "ocsort": {
        "det_thresh": [0.3,0.4, 0.5, 0.6, 0.7],
        "max_age": [20, 30, 40, 50, 70,100],
        "min_hits": [1, 3],
        "iou_threshold": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    },
}

ConfGrid = {
    "bytetrack": [0.01, 0.03, 0.05],
    "deepsort": [0.3, 0.4, 0.5],
    "ocsort": [0.2, 0.3, 0.4],
}
IouGrid = [0.5, 0.6, 0.7]

TrackerDefaults = {
    #"deepsort": {"max_dist": 0.1, "max_age": 30, "n_init": 3, "max_iou_dist": 0.9},
    #"bytetrack": {"track_high_thresh": 0.4, "track_buffer": 20, "match_thresh": 0.8},
    "ocsort": {"det_thresh": 0.3, "max_age": 20, "min_hits": 1, "iou_threshold": 0.6},
}

"""
baseline values
TrackerDefaults = {
    "deepsort": {"max_dist": 0.2, "max_age": 30, "n_init": 3, "max_iou_dist": 0.7},
    "bytetrack": {"track_high_thresh": 0.5, "track_buffer": 30, "match_thresh": 0.8},
    "ocsort": {"det_thresh": 0.5, "max_age": 30, "min_hits": 3, "iou_threshold": 0.3},
}


"""

OutDir.mkdir(parents=True, exist_ok=True)


def build_tracker(tracker, params):
    if tracker == "deepsort":
        return DeepSort(
            max_cosine_distance=params["max_dist"],
            max_age=params["max_age"],
            n_init=params["n_init"],
            max_iou_distance=params["max_iou_dist"],
            embedder_gpu=True,
            half=False,
        )
    elif tracker == "bytetrack":
        return BYTETracker(
            track_thresh=params["track_high_thresh"],
            match_thresh=params["match_thresh"],
            track_buffer=params["track_buffer"],
            frame_rate=30,
        )
    elif tracker == "ocsort":
        return OCSort(
            det_thresh=params["det_thresh"],
            max_age=params["max_age"],
            min_hits=params["min_hits"],
            iou_threshold=params["iou_threshold"],
            use_byte=False,
        )
    else:
        raise ValueError(f"Tracker {tracker} is not supported.")


def tracker_on_sequence(img_folder, tracker_name, tracker_params, output_path, conf, iou):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img_folder = Path(img_folder)
    frame_paths = sorted(img_folder.glob("*.jpg"))
    if not frame_paths:
        frame_paths = sorted(img_folder.glob("*.png"))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {img_folder}")
    model = YOLO(DetectorWeights)
    tracker = build_tracker(tracker_name, tracker_params)
    mot_lines = []
    frame_times = []
    total_time = 0.0
    for frame_idx, img_path in enumerate(frame_paths):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Frame {frame_idx} cannot be read.")
            continue
        if frame_idx == 1 and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_start = time.perf_counter()
        results = model.predict(
            frame, conf=conf, iou=iou, imgsz=Imgsz, verbose=False,
        )
        dets = []
        if results[0].boxes is not None and len(results[0].boxes):
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()
            for box, det_conf, cls in zip(boxes, confs, clss):
                dets.append([*box, det_conf, cls])
        dets_np = np.array(dets) if dets else np.empty((0, 6))

        active_tracks = []
        if tracker_name == "deepsort":
            ds_dets = []
            for d in dets_np:
                x1, y1, x2, y2, c = d[0], d[1], d[2], d[3], d[4]
                ds_dets.append(([x1, y1, x2 - x1, y2 - y1], c, 0))
            ds_tracks = tracker.update_tracks(ds_dets, frame=frame)
            for t in ds_tracks:
                if not t.is_confirmed():
                    continue
                x1, y1, x2, y2 = t.to_ltrb()
                active_tracks.append((x1, y1, x2, y2, t.track_id, t.get_det_conf() or 1.0))
        elif tracker_name == "bytetrack":
            if len(dets_np):
                bt_dets = dets_np[:, :6]
            else:
                bt_dets = np.empty((0, 6))
            bt_tracks = tracker.update(bt_dets, frame)
            for t in bt_tracks:
                x1, y1, x2, y2, tid, conf_t = t[0], t[1], t[2], t[3], int(t[4]), t[5]
                active_tracks.append((x1, y1, x2, y2, tid, conf_t))
        elif tracker_name == "ocsort":
            if len(dets_np):
                oc_input = torch.from_numpy(dets_np[:, :6].astype(np.float32))
            else:
                oc_input = torch.zeros((0, 6))
            oc_tracks = tracker.update(oc_input, frame)
            for t in oc_tracks:
                active_tracks.append((t[0], t[1], t[2], t[3], int(t[4]), t[6]))

        t_end = time.perf_counter()
        frame_ms = (t_end - t_start) * 1000
        total_time += (t_end - t_start)
        frame_times.append(frame_ms)

        for x1, y1, x2, y2, tid, tconf in active_tracks:
            w = x2 - x1
            h = y2 - y1
            mot_lines.append(
                f"{frame_idx},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{tconf:.4f},-1,-1,-1"
            )

    output_path.write_text("\n".join(mot_lines), encoding="utf-8")
    n_frames = len(frame_paths)
    fps = n_frames / total_time if total_time > 0 else 0.0
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
        "peak_vram_mb": peak_vram,
    }


def load_mot(mot_path):
    path = Path(mot_path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["frame", "id", "x", "y", "w", "h"])
    df = pd.read_csv(mot_path, header=None, names=["frame", "id", "x", "y", "w", "h", "conf", "cx", "cy", "cz"])
    return df[["frame", "id", "x", "y", "w", "h"]].copy()


def evaluate_sequence(gt_path, pred_path, iou_threshold=0.5):
    gt_df = load_mot(gt_path)
    pred_df = load_mot(pred_path)
    acc = mm.MOTAccumulator(auto_id=True)
    all_frames = sorted(set(gt_df["frame"]) | set(pred_df["frame"]))
    for frame in all_frames:
        gt_frame = gt_df[gt_df["frame"] == frame]
        pred_frame = pred_df[pred_df["frame"] == frame]
        gt_boxes = gt_frame[["x", "y", "w", "h"]].values.tolist()
        pred_boxes = pred_frame[["x", "y", "w", "h"]].values.tolist()
        gt_ids = gt_frame["id"].tolist()
        pred_ids = pred_frame["id"].tolist()
        if gt_boxes and pred_boxes:
            distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=1 - iou_threshold)
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
    hota = compute_hota(gt_df, pred_df, iou_threshold=iou_threshold)
    r["mota_pct"] = round(float(r["mota"]) * 100, 2)
    r["idf1_pct"] = round(float(r["idf1"]) * 100, 2)
    r["hota_pct"] = round(hota, 2) if hota is not None else 0.0
    return r


def evaluate_dataset(dataset_name, tracker_name, tracker_params, output_subdir, conf, iou):
    clips = DATASETS[dataset_name]
    clip_rows = []
    timing_rows = []
    for i, (img_folder, gt_path) in enumerate(clips):
        seq_name = Path(img_folder).parent.name
        out_file = output_subdir / f"{seq_name}_{tracker_name}.txt"
        print(f" Clip {i + 1}/{len(clips)}: {seq_name}")
        timing = tracker_on_sequence(img_folder, tracker_name, tracker_params, out_file, conf, iou)
        timing_rows.append(timing)
        eval_seq = evaluate_sequence(gt_path, str(out_file))
        clip_rows.append(eval_seq)
    avg = {}
    for k in ["mota_pct", "idf1_pct", "hota_pct", "num_switches", "num_misses",
              "num_false_positives", "mostly_tracked", "mostly_lost", "num_fragmentations"]:
        vals = [row[k] for row in clip_rows if k in row]
        avg[k] = round(float(np.mean(vals)), 3) if vals else 0.0
    avg["fps_mean"] = round(float(np.mean([t["fps"] for t in timing_rows])), 1)
    avg["latency_mean_ms"] = round(float(np.mean([t["latency_mean_ms"] for t in timing_rows])), 2)
    avg["latency_p95_ms"] = round(float(np.mean([t["latency_p95_ms"] for t in timing_rows])), 2)
    avg["peak_vram_mb"] = round(float(np.max([t["peak_vram_mb"] for t in timing_rows])), 1)
    avg["num_clips"] = len(clips)
    return avg


def run_baseline():
    print("Baseline evaluation")
    rows = []
    for tracker_name, default_params in TrackerDefaults.items():
        print(f" Tracker {tracker_name}")
        for dataset_name in DATASETS:
            print(f" Dataset {dataset_name}")
            tmp_dir = OutDir / "_tmp" / dataset_name
            tmp_dir.mkdir(parents=True, exist_ok=True)
            avg = evaluate_dataset(dataset_name, tracker_name, default_params, tmp_dir, Def_conf, Def_iou)
            row = {
                "tracker_name": tracker_name,
                "dataset_name": dataset_name,
                "conf": OC_conf,
                "iou": OC_iou,
                "HOTA (%)": avg["hota_pct"],
                "MOTA (%)": avg["mota_pct"],
                "IDF1 (%)": avg["idf1_pct"],
                "ID Switches": avg["num_switches"],
                "MostlyTracked": avg["mostly_tracked"],
                "MostlyLost": avg["mostly_lost"],
                "FPS": avg["fps_mean"],
                "Latency mean (ms)": avg["latency_mean_ms"],
                "Latency 95 (ms)": avg["latency_p95_ms"],
                "Peak VRAM (MB)": avg["peak_vram_mb"],
            }
            rows.append(row)
            print(f"    HOTA:{row['HOTA (%)']}%  MOTA:{row['MOTA (%)']}%  IDF1:{row['IDF1 (%)']}%  FPS:{row['FPS']}")
    df = pd.DataFrame(rows)
    df.to_csv(OutDir / f"final_results_ocsort.csv", index=False)
    print(df.to_string(index=False))


def compute_hota(gt_df, pred_df, iou_threshold=0.5):
    hota_metric = HOTA({"THRESHOLD": iou_threshold})
    all_frames = sorted(set(gt_df["frame"]) | set(pred_df["frame"]))
    all_gt_ids = sorted(gt_df["id"].unique().tolist())
    all_pred_ids = sorted(pred_df["id"].unique().tolist())
    gt_id_map = {v: i for i, v in enumerate(all_gt_ids)}
    pred_id_map = {v: i for i, v in enumerate(all_pred_ids)}
    gt_ids_list, tracker_ids_list, sim_list = [], [], []
    num_gt_dets, num_tracker_dets = 0, 0
    for frame in all_frames:
        gt_frame = gt_df[gt_df["frame"] == frame]
        pred_frame = pred_df[pred_df["frame"] == frame]
        gt_ids = np.array([gt_id_map[i] for i in gt_frame["id"].tolist()], dtype=np.int32)
        pred_ids = np.array([pred_id_map[i] for i in pred_frame["id"].tolist()], dtype=np.int32)
        gt_boxes = gt_frame[["x", "y", "w", "h"]].to_numpy()
        pred_boxes = pred_frame[["x", "y", "w", "h"]].to_numpy()
        if len(gt_boxes) and len(pred_boxes):
            sim = 1.0 - mm.distances.iou_matrix(gt_boxes, pred_boxes)
        else:
            sim = np.zeros((len(gt_boxes), len(pred_boxes)))
        gt_ids_list.append(gt_ids)
        tracker_ids_list.append(pred_ids)
        sim_list.append(sim)
        num_gt_dets += len(gt_ids)
        num_tracker_dets += len(pred_ids)
    data = {
        "num_timesteps": len(all_frames),
        "num_gt_dets": num_gt_dets,
        "num_tracker_dets": num_tracker_dets,
        "num_gt_ids": len(all_gt_ids),
        "num_tracker_ids": len(all_pred_ids),
        "gt_ids": gt_ids_list,
        "tracker_ids": tracker_ids_list,
        "similarity_scores": sim_list,
    }
    res = hota_metric.eval_sequence(data)
    return float(np.mean(res["HOTA"])) * 100


def run_tuning(tracker_name):
    print("Tuning")
    print("Tuning conf and iou")
    conf_vals = ConfGrid[tracker_name]
    iou_vals = IouGrid
    s1_rows = []
    # load existing rows from a previous partial run so resuming doesn't overwrite them
    csv_path = OutDir / f"tuning_{tracker_name}.csv"
    if csv_path.exists():
        existing = pd.read_csv(csv_path).to_dict(orient="records")
        all_rows = existing
        s1_rows = [r for r in existing if r.get("stage") == 1]
        print(f"  [Resume] Loaded {len(existing)} existing rows from {csv_path.name}")
    else:
        all_rows = []
    for dataset_name in DATASETS:
        print(f" Dataset {dataset_name}")
        for conf, iou in product(conf_vals, iou_vals):
            run_id = f"conf:{conf},iou:{iou}"
            tmp_dir = OutDir / "_tmp" / dataset_name
            tmp_dir.mkdir(parents=True, exist_ok=True)
            avg = evaluate_dataset(dataset_name, tracker_name, TrackerDefaults[tracker_name], tmp_dir, conf, iou)
            row = {
                "tracker_name": tracker_name,
                "dataset_name": dataset_name,
                "stage": 1,
                "run_id": run_id,
                "conf": conf,
                "iou": iou,
                "HOTA (%)": avg["hota_pct"],
                "MOTA (%)": avg["mota_pct"],
                "IDF1 (%)": avg["idf1_pct"],
                "ID Switches": avg["num_switches"],
                "MostlyTracked": avg["mostly_tracked"],
                "MostlyLost": avg["mostly_lost"],
                "FPS": avg["fps_mean"],
                "Latency mean (ms)": avg["latency_mean_ms"],
                "Latency 95 (ms)": avg["latency_p95_ms"],
                "Peak VRAM (MB)": avg["peak_vram_mb"],
            }
            s1_rows.append(row)
            all_rows.append(row)
    # find the best conf/iou per dataset based on HOTA
    s1_df = pd.DataFrame(s1_rows)
    best_conf_iou = {}
    for ds in DATASETS:
        subset = s1_df[s1_df["dataset_name"] == ds]
        if subset.empty:
            best_conf_iou[ds] = (Def_conf, Def_iou)
            continue
        best = subset.loc[subset["HOTA (%)"].idxmax()]
        best_conf_iou[ds] = (float(best["conf"]), float(best["iou"]))
        print(f"  [Stage 1 best] {ds}: conf={best['conf']} iou={best['iou']} "
              f"IDF1={best['IDF1 (%)']}%, HOTA = {best['HOTA (%)']}")

    print("tracker parameters eval with best conf and iou")
    grid = TrackerGrids[tracker_name]
    keys = list(grid.keys())
    combos = list(product(*grid.values()))
    print(f"{len(combos)} combos per dataset")
    for dataset_name in DATASETS:
        best_conf, best_iou = best_conf_iou[dataset_name]
        print(f" Dataset {dataset_name}: conf={best_conf} iou={best_iou} ")
        best_hota, best_combo = -1, None
        for combo in combos:
            params = dict(zip(keys, combo))
            run_id = "_".join(f"{k}{v}" for k, v in params.items())
            tmp_dir = OutDir / "_tmp" / dataset_name
            tmp_dir.mkdir(parents=True, exist_ok=True)
            avg = evaluate_dataset(dataset_name, tracker_name,
                                   params, tmp_dir, best_conf, best_iou)
            row = {"tracker_name": tracker_name, "dataset_name": dataset_name,
                   "stage": 2, "run_id": run_id,
                   "conf": best_conf, "iou": best_iou,
                   **params,
                   "MOTA (%)": avg["mota_pct"], "IDF1 (%)": avg["idf1_pct"],
                   "HOTA (%)": avg["hota_pct"],
                   "ID Switches": avg["num_switches"],
                   "MostlyTracked": avg["mostly_tracked"], "MostlyLost": avg["mostly_lost"],
                   "FPS": avg["fps_mean"],
                   "Latency mean (ms)": avg["latency_mean_ms"],
                   "Latency p95 (ms)": avg["latency_p95_ms"],
                   "Peak VRAM (MB)": avg["peak_vram_mb"]}
            all_rows.append(row)
            if avg["hota_pct"] > best_hota:
                best_hota, best_combo = avg["hota_pct"], params
            print(f"  Best HOTA: {best_hota}%  Params: {best_combo}")
        # checkpoint after each dataset so progress is never lost on crash or cut-off
        csv_path = OutDir / f"tuning_{tracker_name}.csv"
        pd.DataFrame(all_rows).to_csv(csv_path, index=False)
        print(f"  [Checkpoint] {csv_path.name} saved ({len(all_rows)} rows)")
    print("Tuning complete")


def build_results_table():
    print("Results table creation")
    rows = []
    for tracker_name in TrackerDefaults:
        csv_path = OutDir / f"tuning_{tracker_name}.csv"
        if not csv_path.exists():
            print(f"[SKIP] {csv_path} not found — run tuning first")
            continue
        df = pd.read_csv(csv_path)
        stage2 = df[df["stage"] == 2]
        for dataset_name in DATASETS:
            subset = stage2[stage2["dataset_name"] == dataset_name]
            if subset.empty:
                continue
            best = subset.loc[subset["HOTA (%)"].idxmax()]
            rows.append({
                "Tracker": tracker_name,
                "Dataset": dataset_name,
                "conf": best["conf"],
                "iou": best["iou"],
                "HOTA (%)": best["HOTA (%)"],
                "MOTA (%)": best["MOTA (%)"],
                "IDF1 (%)": best["IDF1 (%)"],
                "ID Switches": int(best["ID Switches"]),
                "MostlyTracked": round(best["MostlyTracked"], 3),
                "MostlyLost": round(best["MostlyLost"], 3),
                "FPS": best["FPS"] if "FPS" in best.index else "-",
                "Latency mean (ms)": best["Latency mean (ms)"] if "Latency mean (ms)" in best.index else "-",
                "Latency p95 (ms)": best["Latency p95 (ms)"] if "Latency p95 (ms)" in best.index else "-",
                "Peak VRAM (MB)": best["Peak VRAM (MB)"] if "Peak VRAM (MB)" in best.index else "-",
            })
    if not rows:
        print("[WARN] No tuning results found.")
        return
    summary = pd.DataFrame(rows)
    summary.to_csv(OutDir / "results_summary.csv", index=False)
    with open(OutDir / "results_table.txt", "w") as f:
        f.write(summary.to_string(index=False))
    print(summary.to_string(index=False))
    print(f"\n[DONE] {OutDir / 'results_summary.csv'}")
    print("\n[BEST TRACKER PER DATASET]")
    best_per = summary.loc[summary.groupby("Dataset")["HOTA (%)"].idxmax()]
    print(best_per[["Dataset", "Tracker", "conf", "iou", "MOTA (%)", "IDF1 (%)"]].to_string(index=False))


if __name__ == "__main__":
    print("Modes")
    print("1.Baseline Calculate")
    print("2.Tracker Tuning")
    print("3. Results Table")
    selection_1 = int(input("Selection(1/2/3: "))
    if selection_1 == 1:
        run_baseline()
    if selection_1 == 2:
        print("Choose Tracker")
        print("1.DeepSort")
        print("2.ByteTrack")
        print("3. OCSort")
        selection_2 = int(input("Selection(1/2/3: "))
        if selection_2 == 1:
            run_tuning("deepsort")
        elif selection_2 == 2:
            run_tuning("bytetrack")
        elif selection_2 == 3:
            run_tuning("ocsort")
    if selection_1 == 3:
        build_results_table()
