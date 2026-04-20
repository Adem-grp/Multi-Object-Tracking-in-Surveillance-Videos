"""
shanghaitech_dataset_path_arrangement.py
=========================================
ShanghaiTech Campus Anomaly Dataset -> YOLO-style frames with EMPTY labels.

Respects the original training/testing split from the dataset authors.
Anomaly masks (test_frame_mask, test_pixel_mask) are completely ignored —
those are numpy array anomaly annotations, meaningless for object detection.
We generate our own pseudo-labels via the bridge script afterwards.

Dataset structure:
    shanghaitech/
        training/
            videos/
                01_001.avi   <- raw video files, we extract frames from these
                01_002.avi
                ...
        testing/
            frames/
                01_0014/     <- sequence folders with .jpg frames already extracted
                    000.jpg
                    001.jpg
                    ...
                01_0015/
                ...
            test_frame_mask/  <- .npy anomaly masks, ignored
            test_pixel_mask/  <- .npy anomaly masks, ignored

Output structure:
    datasets/shanghaitech_yolo/
        train/
            images/
            labels/   <- empty .txt files
        test/
            images/
            labels/   <- empty .txt files

Note:
    Both train and test outputs feed into gmot.yaml under 'train'.
    Neither has real object-level ground truth so neither serves
    as our evaluation benchmark. GMOT-40 test set is our benchmark.
"""

import cv2
import shutil
from pathlib import Path
from tqdm import tqdm

# ================= CONFIG =================
SRC_ROOT     = Path(r"C:/Users/USER/PycharmProjects/Multi-Object-Tracking-in-Surveillance-Videos/ShanghaiTech Campus dataset (Anomaly Detection)/shanghaitech")
DST_ROOT     = Path(r"C:/Users/USER/PycharmProjects/Multi-Object-Tracking-in-Surveillance-Videos/datasets/shanghaitech_yolo")

FRAME_STRIDE = 1      # 1 = every frame, 2 = every other frame
JPEG_QUALITY = 95

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}

# ================= HELPERS =================
def ensure_dirs(split: str) -> tuple[Path, Path]:
    img_dir = DST_ROOT / split / "images"
    lbl_dir = DST_ROOT / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, lbl_dir


def write_frame(img_dir: Path, lbl_dir: Path, stem: str, frame) -> bool:
    """Write a single frame as .jpg and create an empty label file alongside it."""
    img_path = img_dir / f"{stem}.jpg"
    lbl_path = lbl_dir / f"{stem}.txt"
    cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    lbl_path.write_text("", encoding="utf-8")
    return True


# ================= TRAINING SIDE (extract from .avi videos) =================
def process_training(img_dir: Path, lbl_dir: Path) -> int:
    """
    Training data comes as raw .avi video files.
    Same approach as avenue_dataset_path_arrangement.py —
    open each video with cv2.VideoCapture and extract frames.

    Filename format: train_{videoStem}_{frameIdx:06d}.jpg
    Example: train_01_001_000042.jpg
    """
    videos_dir = SRC_ROOT / "training" / "videos"
    if not videos_dir.exists():
        print(f"[WARN] Training videos directory not found: {videos_dir}")
        return 0

    video_files = sorted(
        p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    if not video_files:
        print(f"[WARN] No video files found in {videos_dir}")
        return 0

    print(f"[INFO] Found {len(video_files)} training videos")
    total_frames = 0

    for video_path in video_files:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[WARN] Could not open video: {video_path}, skipping.")
            continue

        # Use video stem as part of filename, replacing dots with underscores
        # e.g. '01_001' stays as '01_001'
        video_stem = video_path.stem
        idx        = 0
        saved      = 0

        pbar = tqdm(desc=f"train/{video_stem}", unit="f", leave=False)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % FRAME_STRIDE == 0:
                stem = f"train_{video_stem}_{idx:06d}"
                write_frame(img_dir, lbl_dir, stem, frame)
                saved += 1
            idx  += 1
            pbar.update(1)
        pbar.close()
        cap.release()
        total_frames += saved

    return total_frames


# ================= TESTING SIDE (frames already extracted) =================
def process_testing(img_dir: Path, lbl_dir: Path) -> int:
    """
    Testing data comes as pre-extracted .jpg frames inside sequence subfolders.
    We just copy them over and write empty label files.

    Only processes the 'frames' subfolder — ignores test_frame_mask and
    test_pixel_mask entirely since those are numpy anomaly annotations.

    Filename format: test_{sequenceName}_{frameIdx:06d}.jpg
    Example: test_01_0014_000000.jpg
    """
    frames_dir = SRC_ROOT / "testing" / "frames"
    if not frames_dir.exists():
        print(f"[WARN] Testing frames directory not found: {frames_dir}")
        return 0

    # Each subfolder is one test sequence e.g. 01_0014, 01_0015
    sequence_folders = sorted(
        p for p in frames_dir.iterdir() if p.is_dir()
    )
    if not sequence_folders:
        print(f"[WARN] No sequence folders found in {frames_dir}")
        return 0

    print(f"[INFO] Found {len(sequence_folders)} testing sequences")
    total_frames = 0

    for seq_folder in sequence_folders:
        seq_name = seq_folder.name  # e.g. 01_0014

        jpg_files = sorted(seq_folder.glob("*.jpg"))
        if not jpg_files:
            print(f"[WARN] No .jpg files in {seq_folder}, skipping.")
            continue

        for idx, jpg_path in enumerate(tqdm(jpg_files, desc=f"test/{seq_name}", leave=False)):
            if idx % FRAME_STRIDE != 0:
                continue

            frame = cv2.imread(str(jpg_path))
            if frame is None:
                print(f"[WARN] Could not read {jpg_path}, skipping.")
                continue

            stem = f"test_{seq_name}_{idx:06d}"
            write_frame(img_dir, lbl_dir, stem, frame)
            total_frames += 1

    return total_frames


# ================= MAIN =================
def main():
    if DST_ROOT.exists():
        shutil.rmtree(DST_ROOT)
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    # Training side — extract from videos
    train_img, train_lbl = ensure_dirs("train")
    print("\n[STEP 1/2] Processing training videos...")
    train_frames = process_training(train_img, train_lbl)

    # Testing side — copy pre-extracted frames
    test_img, test_lbl = ensure_dirs("test")
    print("\n[STEP 2/2] Processing testing frames...")
    test_frames = process_testing(test_img, test_lbl)

    print("\n========== SUMMARY ==========")
    print(f"Training frames extracted : {train_frames}")
    print(f"Testing frames copied     : {test_frames}")
    print(f"Total frames              : {train_frames + test_frames}")
    print(f"Output root               : {DST_ROOT}")
    print("==============================")
    print(
        "\nNext steps:"
        "\n  1. Run bridge.py with SOURCE_DATASET = 'shanghaitech' for both splits"
        "\n     Make sure SPLITS = ['train', 'test'] in bridge.py config"
        "\n  2. Add to gmot.yaml under train:"
        "\n       - datasets/bridge_yolo/shanghaitech/train/images"
        "\n       - datasets/bridge_yolo/shanghaitech/test/images"
    )


if __name__ == "__main__":
    main()