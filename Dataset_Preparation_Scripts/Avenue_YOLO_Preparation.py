"""
Avenue -> YOLO-style frames with EMPTY labels (MOT-ready).
- Combines ALL videos from Avenue dataset (training + testing).
- Splits by videos: train=0.7, val=0.1, test=0.2.
- Extracts frames, writes empty labels.
- Empty labels are written because there is no MOT annotation for the anomaly datasets
"""

import cv2
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# CONFIG
# path to avenue dataset change if needed
SRC_ROOT = Path(r"C:/Users/USER/PycharmProjects/Multi-Object-Tracking-in-Surveillance-Videos/Avenue Dataset")
DST_ROOT = Path(r"/datasets/avenue_yolo")  # output folder for YOLO-formatted avenue
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.1, 0.2
FRAME_STRIDE = 1  # extract each frame
JPEG_QUALITY = 95  # to achieve high quality while using jpeg compression
SEED = 42
VALID_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".wmv"}


# searches entire dataset with its subdirectories included
def list_all_videos(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS])


# arrange train-val-test splits
def split_videos(videos: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    random.seed(SEED)
    vids = videos.copy()  # get the copy of the videos to avoid modifying the original
    random.shuffle(vids)
    n = len(vids)
    n_train = int(n * TRAIN_RATIO)  # divide the videos
    n_val = int(n * VAL_RATIO)
    train = vids[:n_train]
    val = vids[n_train:n_train + n_val]
    test = vids[n_train + n_val:]
    return train, val, test


# create YOLO style directory structure
def ensure_dirs(split: str):
    img_dir = DST_ROOT / split / "images"
    lbl_dir = DST_ROOT / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, lbl_dir


# save one extracted video frame
def save_frame(frame, img_dir, lbl_dir, stem):
    # write the frame as a JPEG
    cv2.imwrite(str(img_dir / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    (lbl_dir / f"{stem}.txt").write_text("", encoding="utf-8")  # create empty text file with the same name


# conversion from video to frame function
def extract(video: Path, img_dir: Path, lbl_dir: Path, tag: str) -> int:
    cap = cv2.VideoCapture(str(video))  # open the video
    if not cap.isOpened():  # check if it is opened
        print(f"[ERROR] Cannot open {video}")
        return 0
    saved = 0  # count written frames
    idx = 0  # track frame index
    pbar = tqdm(desc=f"{tag}:{video.stem}", unit="f")
    while True:
        ret, frame = cap.read()  # read the frame
        if not ret:  # if frame couldn't be read or the video has ended, exits the loop
            break
        if idx % FRAME_STRIDE == 0:  # save each frame
            stem = f"{video.stem}_{idx:06d}"  # create a unique filename to avoid duplicates
            save_frame(frame, img_dir, lbl_dir, stem)  # save the frame
            saved += 1
        idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    return saved


def main():
    # if DST_ROOT.exists():
    #   shutil.rmtree(DST_ROOT) # in case the conversion was not completed remove the old folder

    DST_ROOT.mkdir(parents=True, exist_ok=True)

    all_videos = list_all_videos(SRC_ROOT) # fetch all videos
    if not all_videos:
        raise SystemExit(f"No videos found under {SRC_ROOT}. Check path or extensions.")
    # train-val-test split
    train_videos, val_videos, test_videos = split_videos(all_videos)
    splits = [("train", train_videos), ("val", val_videos), ("test", test_videos)]

    stats = {}
    for split, vids in splits:
        img_dir, lbl_dir = ensure_dirs(split) # create split directories
        total = 0
        for v in vids: # extract all frames from each video per split
            total += extract(v, img_dir, lbl_dir, split)
        stats[split] = {"videos": len(vids), "frames": total} # record what is processed

    print("\n========== SUMMARY ==========")
    for s, d in stats.items():
        print(f"{s}: {d['videos']} videos -> {d['frames']} frames")
    print(f"Output: {DST_ROOT}")
    print("=============================")


if __name__ == "__main__":
    main()
