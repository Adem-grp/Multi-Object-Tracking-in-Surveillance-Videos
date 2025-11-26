
# avenue_arranger_fixed.py
"""
Avenue -> YOLO-style frames with EMPTY labels (MOT-ready).
- Combines ALL videos from Avenue dataset (training + testing).
- Splits by videos: train=0.7, val=0.1, test=0.2.
- Extracts frames, writes empty labels.
"""

import cv2
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# CONFIG
SRC_ROOT = Path(r"C:/Users/USER/PycharmProjects/Multi-Object-Tracking-in-Surveillance-Videos/Avenue Dataset")
DST_ROOT = Path(r"C:/Users/USER/PycharmProjects/Multi-Object-Tracking-in-Surveillance-Videos/datasets/avenue_yolo")
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.1, 0.2
FRAME_STRIDE = 1
JPEG_QUALITY = 95
SEED = 42
VALID_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".wmv"}

def list_all_videos(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS])

def split_videos(videos: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    random.seed(SEED)
    vids = videos.copy()
    random.shuffle(vids)
    n = len(vids)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train = vids[:n_train]
    val = vids[n_train:n_train + n_val]
    test = vids[n_train + n_val:]
    return train, val, test

def ensure_dirs(split: str):
    img_dir = DST_ROOT / split / "images"
    lbl_dir = DST_ROOT / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, lbl_dir

def save_frame(frame, img_dir, lbl_dir, stem):
    cv2.imwrite(str(img_dir / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    (lbl_dir / f"{stem}.txt").write_text("", encoding="utf-8")

def extract(video: Path, img_dir: Path, lbl_dir: Path, tag: str) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video}")
        return 0
    saved = 0
    idx = 0
    pbar = tqdm(desc=f"{tag}:{video.stem}", unit="f")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % FRAME_STRIDE == 0:
            stem = f"{video.stem}_{idx:06d}"
            save_frame(frame, img_dir, lbl_dir, stem)
            saved += 1
        idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    return saved

def main():
    if DST_ROOT.exists():
        shutil.rmtree(DST_ROOT)
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    all_videos = list_all_videos(SRC_ROOT)
    if not all_videos:
        raise SystemExit(f"No videos found under {SRC_ROOT}. Check path or extensions.")

    train_videos, val_videos, test_videos = split_videos(all_videos)
    splits = [("train", train_videos), ("val", val_videos), ("test", test_videos)]

    stats = {}
    for split, vids in splits:
        img_dir, lbl_dir = ensure_dirs(split)
        total = 0
        for v in vids:
            total += extract(v, img_dir, lbl_dir, split)
        stats[split] = {"videos": len(vids), "frames": total}

    print("\n========== SUMMARY ==========")
    for s, d in stats.items():
        print(f"{s}: {d['videos']} videos -> {d['frames']} frames")
    print(f"Output: {DST_ROOT}")
    print("=============================")

if __name__ == "__main__":
    main()

