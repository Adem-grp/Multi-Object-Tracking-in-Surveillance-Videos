import cv2
import shutil
from pathlib import Path
from tqdm import tqdm

# CONFIG
# dataset path change if needed
SRC_ROOT = Path(r"C:/Users/USER/PycharmProjects/Multi-Object-Tracking-in-Surveillance-Videos/UCSD_Anomaly_Dataset.v1p2")
DST_ROOT = Path(r"/datasets/ucsd_yolo")  # output path

FRAME_STRIDE = 1  # take all frames
JPEG_QUALITY = 95  # arrange the image quality

# there are two subsets of this dataset
SUBSETS = ["UCSDped1", "UCSDped2"]

# prepare the split as test and train since val split will not be necessary
SPLIT_MAP = {
    "Train": "train",
    "Test": "test",
}


# arrange the folders according to YOLO format
def ensure_dirs(split: str) -> tuple[Path, Path]:
    img_dir = DST_ROOT / split / "images"
    lbl_dir = DST_ROOT / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, lbl_dir


def list_sequences(split_dir: Path) -> list[Path]:
    # Return all valid sequence folders inside a split directory.
    # Skips anything ending in '_gt' since those are anomaly mask folders.
    # Skips files (only processes directories).
    if not split_dir.exists():
        print(f"[WARN] Split directory not found: {split_dir}")
        return []

    sequences = []
    for item in sorted(split_dir.iterdir()):
        if not item.is_dir():
            continue
        if item.name.endswith("_gt"):
            # Binary anomaly mask folder — meaningless for object detection
            continue
        sequences.append(item)
    return sequences


def extract_sequence(seq_folder: Path, img_dir: Path, lbl_dir: Path) -> int:
    """
    Read all .tif frames from a sequence folder, convert to .jpg,
    and write an empty .txt label file alongside each one.

    Filename format: {subset}_{splitName}_{sequenceName}_{frameIdx:06d}.jpg
    Example: UCSDped1_Train_Train001_000000.jpg

    This verbose prefix guarantees no collisions between:
      - UCSDped1 and UCSDped2 (same sequence names in both)
      - Train and Test splits (both may have Test001 etc.)
    """
    subset_name = seq_folder.parent.parent.name  # UCSDped1 or UCSDped2
    split_name = seq_folder.parent.name  # Train or Test
    seq_name = seq_folder.name  # Train001, Test001 etc.
    prefix = f"{subset_name}_{split_name}_{seq_name}"

    tif_files = sorted(seq_folder.glob("*.tif"))  # get all tif files
    if not tif_files:
        print(f"[WARN] No .tif files in {seq_folder}, skipping.")
        return 0

    saved = 0
    # give each tif file an index and process them like in other preparation files
    for idx, tif_path in enumerate(tqdm(tif_files, desc=prefix, leave=False)):
        if idx % FRAME_STRIDE != 0:
            continue

        frame = cv2.imread(str(tif_path))  # read the frame
        if frame is None:
            print(f"[WARN] Could not read {tif_path}, skipping.")
            continue

        stem = f"{prefix}_{idx:06d}"
        img_path = img_dir / f"{stem}.jpg"
        lbl_path = lbl_dir / f"{stem}.txt"
        # save the image as JPEG for convenience
        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        lbl_path.write_text("", encoding="utf-8")  # create empty labels since this is also an anomaly dataset
        saved += 1

    return saved


def main():
    if DST_ROOT.exists():  # this removes the previous arranged dataset if it is incomplete so comment it out if necessary
        shutil.rmtree(DST_ROOT)
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    stats = {}
    # arrange the YOLO folder structure
    for subset in SUBSETS:
        for original_split, our_split in SPLIT_MAP.items():
            split_dir = SRC_ROOT / subset / original_split
            sequences = list_sequences(split_dir)  # find actual sequence directories

            if not sequences:
                print(f"[INFO] No sequences found in {split_dir}, skipping.")
                continue

            img_dir, lbl_dir = ensure_dirs(our_split)  # use mapped name and arrange output split

            total_frames = 0
            for seq in sequences:
                total_frames += extract_sequence(seq, img_dir, lbl_dir)

            key = f"{subset}/{original_split}"  # create identifiers
            stats[key] = {  # store values for the summary part
                "sequences": len(sequences),
                "frames": total_frames,
                "output": our_split
            }

    for key, d in stats.items():
        print(f"{key}: {d['sequences']} sequences -> {d['frames']} frames -> datasets/ucsd_yolo/{d['output']}")
    print(f"\nOutput root: {DST_ROOT}")


if __name__ == "__main__":
    main()
