
import glob
import json
import time
import gc
from pathlib import Path
from natsort import natsorted
from PIL import Image
from ultralytics import YOLO

# =========================
# Config
# =========================
BASE_DIR = Path(__file__).resolve().parent

# Point this to your latest unified model when ready
MODEL_WEIGHTS = str((BASE_DIR / "runs" / "detect" / "train5" / "weights" / "best.pt").resolve())

# YAML for detector validation (GMOT or combined YAML)
DETECTOR_DATA_YAML = str((BASE_DIR / "datasets" / "gmot_coco_bridge.yaml").resolve())

# Tracking/val params
TRACKER_YAML = "botsort.yaml"
CONF_THRES   = 0.30
IOU_THRES    = 0.50
IMG_SIZE     = 640        # If OOM, try 512 or 448
DEVICE       = 0          # GPU index for sequences that use GPU
BATCH_SIZE   = 16         # detector .val() batch
PLOTS        = False

# Outputs (flat files-first, append-only)
RUN_TAG    = "GMOT10"
OUT_ROOT   = (BASE_DIR / "gmot_tracking_results").resolve()
TRACKS_CSV      = OUT_ROOT / f"{RUN_TAG}_tracks.csv"          # per-frame rows
TOTALS_CSV      = OUT_ROOT / f"{RUN_TAG}_totals.csv"          # per-run totals
DET_VAL_JSONL   = OUT_ROOT / f"{RUN_TAG}_detector_val.jsonl"  # val metrics (JSONL)
PROCESS_SUMMARY = OUT_ROOT / f"{RUN_TAG}_processing_summary.json"  # JSON array

# Visuals
SAVE_VIS = True
VIS_ROOT = OUT_ROOT / f"{RUN_TAG}_vis"  # per-sequence subfolders

# Optional consolidated bridge visuals (off by default)
MIRROR_TO_BRIDGE_VIS = False
BRIDGE_VIS_ROOT = OUT_ROOT / "bridge_yolo_vis"

# OOM fallback ladder (used only when a sequence runs on GPU)
SAFE_RETRY_ON_OOM = True
RETRY_IMGSZS = [512, 448, 384, 320]

# =========================
# Sequences (strict .jpg only; case-insensitive)
# =========================
SEQUENCES = [
    ("gmot_yolo_val",
     (BASE_DIR / "datasets" / "gmot_yolo"   / "val" / "images").resolve(),
     ["*.jpg", "*.JPG"]),
    ("bridge_yolo_val",
     (BASE_DIR / "datasets" / "bridge_yolo" / "val" / "images").resolve(),
     ["*.jpg", "*.JPG"]),
]

# Per-sequence device override (VRAM-safe: GMOT on GPU, Bridge on CPU by default)
DEVICE_MAP = {
    "gmot_yolo_val": DEVICE,
    "bridge_yolo_val": DEVICE,
}

# =========================
# Helpers
# =========================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def append_jsonl(path: Path, obj: dict):
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def verify_images(image_paths):
    """Verify readable images and return sorted valid list."""
    valid = []
    for p in natsorted(image_paths):
        try:
            Image.open(p).verify()
            valid.append(p)
        except Exception:
            print("[WARN] Skipping unreadable image:", p)
    return valid

def cuda_cleanup():
    """Free VRAM between sequences to mitigate fragmentation and OOM."""
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        print(f"[WARN] CUDA cleanup failed: {e}")

def mirror_visuals(src_dir: Path, dst_root: Path, seq_name: str):
    """Copy visuals into a unified bridge_yolo_vis/<sequence>/ folder (output-only)."""
    try:
        from shutil import copy2
        ensure_dir(dst_root)
        dst_dir = dst_root / seq_name
        ensure_dir(dst_dir)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        count = 0
        for p in src_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                target = dst_dir / p.name
                copy2(p, target)
                count += 1
        print(f"[OK] Mirrored {count} visual(s) → {dst_dir}")
    except Exception as e:
        print(f"[WARN] Visual mirroring failed: {e}")

def collect_images_case_insensitive(dir_path: Path, patterns: list[str]) -> list[str]:
    """Collect images using multiple patterns (e.g., ['*.jpg', '*.JPG'])."""
    files = []
    for pat in patterns:
        glob_path = str(dir_path / pat)
        hits = glob.glob(glob_path)
        print(f"[DEBUG] Pattern {pat} matched {len(hits)} files in {dir_path}")
        files.extend(hits)
    # Deduplicate and sort naturally
    files = natsorted(list(set(files)))
    return files

# =========================
# Detector validation (append-only JSONL)
# =========================
def run_detector_val_and_save_metrics(model_path: str, data_yaml: str):
    """Run Ultralytics .val() and append metrics JSONL before tracking."""
    yaml_path = Path(data_yaml)
    weights_path = Path(model_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"[ERROR] Detector data YAML not found: {yaml_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"[ERROR] Model weights not found: {weights_path}")

    print("\n[STEP] Detector validation (mAP/P/R) — appending metrics JSONL...")
    ensure_dir(OUT_ROOT)

    model = YOLO(model_path)
    r = model.val(
        data=data_yaml,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        verbose=True,
        plots=PLOTS,
    )

    metrics_dict = getattr(r, "results_dict", None) or {}
    metrics_out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_tag": RUN_TAG,
        "imgsz": IMG_SIZE,
        "batch": BATCH_SIZE,
        "device": DEVICE,
        **metrics_dict
    }
    append_jsonl(DET_VAL_JSONL, metrics_out)
    print("[OK] Detector metrics appended →", DET_VAL_JSONL)

    for k in ["metrics/mAP50-95(B)", "metrics/mAP50(B)", "precision(B)", "recall(B)"]:
        if k in metrics_dict:
            try:
                print(f"  {k}: {float(metrics_dict[k]):.4f}")
            except Exception:
                print(f"  {k}: {metrics_dict[k]}")

# =========================
# Tracking: append frames + per-class totals (OOM-safe)
# =========================
def track_sequence(model: YOLO, image_files: list, seq_name: str, device_override=None):
    """
    Track a sequence:
    - Append per-frame rows to TRACKS_CSV (header once).
    - Append per-run totals to TOTALS_CSV (per-class aggregate in JSON).
    - Visuals optional via SAVE_VIS.
    - OOM-safe: reduces imgsz then CPU fallback if needed (when using GPU).
    """
    print(f"\n[STEP] Tracking sequence '{seq_name}' with {len(image_files)} frames")
    ensure_dir(OUT_ROOT)
    if SAVE_VIS:
        ensure_dir(VIS_ROOT)
        vis_dir = VIS_ROOT / seq_name
        ensure_dir(vis_dir)
    else:
        vis_dir = None

    names = getattr(model, "names", {}) or {}
    print("[DEBUG] model.names:", names)

    total_detections = 0
    class_counts = {n: 0 for n in names.values()}  # aggregate per-class

    print(f"[DEBUG] Starting tracking; SAVE_VIS={SAVE_VIS}")

    def run_track_attempt(imgsz_try: int, device_try, half_try: bool):
        dev = device_override if device_override is not None else device_try
        print(f"[DEBUG] track() attempt: imgsz={imgsz_try}, device={dev}, half={half_try}")
        return model.track(
            source=image_files,
            tracker=TRACKER_YAML,
            conf=CONF_THRES,
            iou=IOU_THRES,
            device=dev,
            imgsz=imgsz_try,
            half=half_try,
            stream=True,
            show=False,
            save=bool(SAVE_VIS),
            project=str(VIS_ROOT) if SAVE_VIS else None,
            name=seq_name if SAVE_VIS else None,
            exist_ok=True,
            verbose=True,
        )

    # Use OOM ladder only if running on GPU
    use_gpu = (device_override == DEVICE) or (device_override is None and isinstance(DEVICE, int))
    results = None
    try:
        half_flag = True if use_gpu else False
        results = run_track_attempt(IMG_SIZE, DEVICE, half_flag)
    except Exception as e:
        msg = str(e).lower()
        oom = ("out of memory" in msg) or ("cuda out of memory" in msg)
        if not SAFE_RETRY_ON_OOM or not oom or not use_gpu:
            raise
        print("[WARN] OOM at base settings; retrying with reduced imgsz on GPU...")
        cuda_cleanup()
        for sz in RETRY_IMGSZS:
            try:
                results = run_track_attempt(sz, DEVICE, True)
                print(f"[INFO] Recovered on GPU with imgsz={sz} half=True")
                break
            except Exception as e2:
                if ("out of memory" in str(e2).lower()) or ("cuda out of memory" in str(e2).lower()):
                    print(f"[WARN] OOM again at imgsz={sz}; continuing fallback...")
                    cuda_cleanup()
                    continue
                else:
                    raise
        if results is None:
            print("[WARN] Falling back to CPU for this sequence...")
            results = run_track_attempt(IMG_SIZE, "cpu", False)

    # Append frames
    import csv
    csv_exists = TRACKS_CSV.exists()
    with TRACKS_CSV.open("a", newline="") as f:  # append mode
        wr = csv.writer(f)
        if not csv_exists:
            wr.writerow(["frame", "id", "x", "y", "w", "h", "conf", "class", "vis", "sequence", "run_tag"])

        frame_idx = 0
        for r in results:
            frame_idx += 1
            boxes = r.boxes
            if boxes is None:
                continue

            xywh = boxes.xywh.cpu().numpy() if getattr(boxes, "xywh", None) is not None else []
            conf = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else []
            cls  = boxes.cls.cpu().numpy()  if getattr(boxes, "cls",  None) is not None else []
            ids  = boxes.id.cpu().numpy()   if getattr(boxes, "id",   None) is not None else []

            total_detections += len(xywh)

            for k in range(len(xywh)):
                x, y, w, h = xywh[k]
                c  = float(conf[k]) if len(conf) else 1.0
                cl = int(cls[k])    if len(cls)  else -1
                tid= int(ids[k])    if len(ids)  else -1

                label = names.get(cl, str(cl))
                if label in class_counts:
                    class_counts[label] += 1
                else:
                    class_counts[label] = class_counts.get(label, 0) + 1

                # center (x,y) → top-left (x,y)
                x_tl = int(x - w/2)
                y_tl = int(y - h/2)
                wr.writerow([frame_idx, tid, x_tl, y_tl, int(w), int(h), f"{c:.3f}", cl, -1, seq_name, RUN_TAG])


    if not TRACKS_CSV.exists():
        raise RuntimeError(f"[ERROR] Tracks CSV was not created: {TRACKS_CSV}")
    print(f"[OK] Tracks CSV appended → {TRACKS_CSV}")

    if SAVE_VIS:
        vis_seq_dir = VIS_ROOT / seq_name
        print(f"[OK] Visuals saved under → {vis_seq_dir}")
        if MIRROR_TO_BRIDGE_VIS:
            mirror_visuals(vis_seq_dir, BRIDGE_VIS_ROOT, seq_name)

    # Per-run totals CSV (append-only + blank line)
    totals_row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_tag": RUN_TAG,
        "sequence": seq_name,
        "frames": len(image_files),
        "total_detections": total_detections,
        "class_counts_json": json.dumps(class_counts, ensure_ascii=False),
        "tracks_csv": str(TRACKS_CSV),
        "visuals_dir": str(VIS_ROOT / seq_name) if SAVE_VIS else "",
    }
    totals_exists = TOTALS_CSV.exists()
    with TOTALS_CSV.open("a", newline="", encoding="utf-8") as f:
        import csv as _csv
        header = ["timestamp","run_tag","sequence","frames","total_detections","class_counts_json","tracks_csv","visuals_dir"]
        w = _csv.DictWriter(f, fieldnames=header)
        if not totals_exists:
            w.writeheader()
        w.writerow(totals_row)

    if not TOTALS_CSV.exists():
        raise RuntimeError(f"[ERROR] Totals CSV was not created: {TOTALS_CSV}")
    print(f"[OK] Totals CSV appended → {TOTALS_CSV}")
    print(f"[TOTALS] {seq_name}: total_detections={total_detections}")
    print(f"[DEBUG] Per-class counts: {class_counts}")

    print(f"[DEBUG] Finished tracking: wrote rows to {TRACKS_CSV}")

    # Free VRAM at end (good hygiene)
    cuda_cleanup()

    return totals_row

# =========================
# Main
# =========================
def main():
    print(f"Unified run: GMOT + Bridge | Weights: {MODEL_WEIGHTS}")
    print(f"Tracker: {TRACKER_YAML}")
    print(f"Output root: {OUT_ROOT} (append-only)")
    print(f"Batch: {BATCH_SIZE} | Img size: {IMG_SIZE} | Base device: {DEVICE} | SAVE_VIS={SAVE_VIS}")
    print(f"[DEBUG] DETECTOR_DATA_YAML: {DETECTOR_DATA_YAML}")
    print(f"[DEBUG] OUT_ROOT: {OUT_ROOT}")

    # 1) DETECTOR VALIDATION — metrics JSONL append (once per run)
    run_detector_val_and_save_metrics(MODEL_WEIGHTS, DETECTOR_DATA_YAML)
    if not DET_VAL_JSONL.exists():
        raise RuntimeError(f"[ERROR] Detector metrics JSONL was not created: {DET_VAL_JSONL}")

    # 2) Process each sequence (GMOT + Bridge)
    summary_entries = []
    for seq_name, seq_dir, patterns in SEQUENCES:
        print(f"\n[SEQ] {seq_name} → dir: {seq_dir}")
        if not seq_dir.exists():
            print(f"[WARN] Sequence dir does not exist: {seq_dir} — skipping.")
            continue

        all_images = collect_images_case_insensitive(seq_dir, patterns)
        print(f"[DEBUG] Total matched files: {len(all_images)}")
        for p in all_images[:3]:
            print(f"[DEBUG] Example image: {p}")

        if not all_images:
            print(f"[WARN] No images for {seq_name} (patterns: {patterns}) — skipping.")
            continue

        valid_images = verify_images(all_images)
        if not valid_images:
            print(f"[WARN] No valid images after verification for {seq_name} — skipping.")
            continue

        print(f"[INFO] Total images discovered for {seq_name}: {len(valid_images)}")

        # Re-instantiate model per sequence to avoid long-lived VRAM usage
        model = YOLO(MODEL_WEIGHTS)

        # Per-sequence device override (GPU for GMOT, CPU for Bridge by default)
        dev_override = DEVICE_MAP.get(seq_name, DEVICE)

        totals_row = track_sequence(model, valid_images, seq_name, device_override=dev_override)

        # Append processing summary entry for this sequence
        summary_entries.append({
            "timestamp": totals_row["timestamp"],
            "run_tag": RUN_TAG,
            "sequence": seq_name,
            "frames": totals_row["frames"],
            "total_detections": totals_row["total_detections"],
            "tracks_csv": totals_row["tracks_csv"],
            "visuals_dir": totals_row["visuals_dir"],
            "detector_val_jsonl": str(DET_VAL_JSONL),
            "class_counts": json.loads(totals_row["class_counts_json"]),
        })

    # 3) Processing summary (append to JSON array safely)
    try:
        ensure_dir(OUT_ROOT)
        if PROCESS_SUMMARY.exists():
            text = PROCESS_SUMMARY.read_text(encoding="utf-8")
            existing = json.loads(text)
            if isinstance(existing, list):
                existing.extend(summary_entries)
                PROCESS_SUMMARY.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            else:
                PROCESS_SUMMARY.write_text(json.dumps(summary_entries, indent=2), encoding="utf-8")
        else:
            PROCESS_SUMMARY.write_text(json.dumps(summary_entries, indent=2), encoding="utf-8")
    except Exception as e:
        PROCESS_SUMMARY.write_text(json.dumps(summary_entries, indent=2), encoding="utf-8")
        print(f"[WARN] Failed to append to processing summary ({PROCESS_SUMMARY}): {e}")

    print("\n========== SUMMARY ==========")
    print(f"Tracks CSV       : {TRACKS_CSV}")
    print(f"Totals CSV       : {TOTALS_CSV}")
    print(f"Detector JSONL   : {DET_VAL_JSONL}")
    print(f"Processing summary: {PROCESS_SUMMARY}")
    if SAVE_VIS:
        print(f"Visualizations   : {VIS_ROOT} \\ ({', '.join([s[0] for s in SEQUENCES])})")
        if MIRROR_TO_BRIDGE_VIS:
            print(f"Mirrors          : {BRIDGE_VIS_ROOT}")
    print("=============================")

if __name__ == "__main__":
    main()
