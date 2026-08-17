import cv2
import shutil
from pathlib import Path
from tqdm import tqdm

# CONFIG
# path to shanghaiTech dataset arrange if necessary
SRC_ROOT = Path(
    r"C:/Users/USER/PycharmProjects/Multi-Object-Tracking-in-Surveillance-Videos/ShanghaiTech Campus dataset (Anomaly Detection)/shanghaitech")
DST_ROOT = Path(r"/datasets/shanghaitech_yolo") # output path

FRAME_STRIDE = 1  # read every frame
JPEG_QUALITY = 95 # same with avenu we arrange jpeg_quality

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}


# HELPERS
# creating folder structure according to the YOLO format
def ensure_dirs(split: str) -> tuple[Path, Path]:
    img_dir = DST_ROOT / split / "images"
    lbl_dir = DST_ROOT / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, lbl_dir


def write_frame(img_dir: Path, lbl_dir: Path, stem: str, frame) -> bool:
    """Write a single frame as .jpg and create an empty label file alongside it since ShanghaiTech does not have MOT annotations."""
    img_path = img_dir / f"{stem}.jpg"
    lbl_path = lbl_dir / f"{stem}.txt"
    cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    lbl_path.write_text("", encoding="utf-8")
    return True


# TRAINING SIDE (extract from .avi videos)
def process_training(img_dir: Path, lbl_dir: Path) -> int:
    # training data is raw .avi files
    # we use the same approach as we used in avenue
    # open each video and extract the frames
    # then give unique filenames

    videos_dir = SRC_ROOT / "training" / "videos"
    if not videos_dir.exists():
        print(f"[WARN] Training videos directory not found: {videos_dir}")
        return 0
    # get the video files
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
        cap = cv2.VideoCapture(str(video_path)) # open the video
        if not cap.isOpened():# check if it is opened safely
            print(f"[WARN] Could not open video: {video_path}, skipping.")
            continue

        # Use video stem as part of filename
        # e.g. '01_001' stays as '01_001'
        # this will avoid collisions between from different videos
        video_stem = video_path.stem
        idx = 0
        saved = 0
        # same logic as avenue extracting all frames
        pbar = tqdm(desc=f"train/{video_stem}", unit="f", leave=False)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % FRAME_STRIDE == 0:
                stem = f"train_{video_stem}_{idx:06d}"
                write_frame(img_dir, lbl_dir, stem, frame) # save frame and empty label
                saved += 1
            idx += 1
            pbar.update(1)
        pbar.close()
        cap.release()
        total_frames += saved

    return total_frames


#  TESTING SIDE (frames already extracted)
def process_testing(img_dir: Path, lbl_dir: Path) -> int:
    # Since testing data is already extracted jpeg frames, they are just copied
    # and empty label files are written

    frames_dir = SRC_ROOT / "testing" / "frames"
    if not frames_dir.exists():
        print(f"[WARN] Testing frames directory not found: {frames_dir}")
        return 0

    # Each subfolder is one test sequence
    sequence_folders = sorted(
        p for p in frames_dir.iterdir() if p.is_dir()
    )
    if not sequence_folders:
        print(f"[WARN] No sequence folders found in {frames_dir}")
        return 0

    print(f"[INFO] Found {len(sequence_folders)} testing sequences")
    total_frames = 0

    for seq_folder in sequence_folders:
        seq_name = seq_folder.name

        jpg_files = sorted(seq_folder.glob("*.jpg")) # read all jpeg frames per sequence
        if not jpg_files:
            print(f"[WARN] No .jpg files in {seq_folder}, skipping.")
            continue
        # give each frame an index
        for idx, jpg_path in enumerate(tqdm(jpg_files, desc=f"test/{seq_name}", leave=False)):
            if idx % FRAME_STRIDE != 0:
                continue

            frame = cv2.imread(str(jpg_path)) # read the image
            if frame is None:
                print(f"[WARN] Could not read {jpg_path}, skipping.")
                continue

            stem = f"test_{seq_name}_{idx:06d}" # give unique output name to the frame
            write_frame(img_dir, lbl_dir, stem, frame)
            total_frames += 1

    return total_frames


#  MAIN
def main():
    if DST_ROOT.exists(): # delete existing if it is incomplete comment it out if it is completed
        shutil.rmtree(DST_ROOT)
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    # Training side — extract from videos
    train_img, train_lbl = ensure_dirs("train") # create folder structure
    print("\n[STEP 1/2] Processing training videos...")
    train_frames = process_training(train_img, train_lbl) # extract all frames

    # Testing side — copy pre-extracted frames
    test_img, test_lbl = ensure_dirs("test") # create folder structure
    print("\n[STEP 2/2] Processing testing frames...")
    test_frames = process_testing(test_img, test_lbl) # extract all frames

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
