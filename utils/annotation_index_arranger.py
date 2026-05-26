import shutil
from pathlib import Path

CVAT_GT_FILES = [
    r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\avenue_yolo\test\gt\gt.txt",
    r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\shanghaitech_yolo\test\shanghai_10\gt\gt.txt",
    r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\shanghaitech_yolo\test\shanghai_128\gt\gt.txt",
    r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\shanghaitech_yolo\test\shanghai_164\gt\gt.txt",
    r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\ucsd_yolo\test\ped1_test1\gt\gt.txt",
    r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\ucsd_yolo\test\ped1_test2\gt\gt.txt",
    r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\ucsd_yolo\test\ped1_test3\gt\gt.txt",
    r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\ucsd_yolo\test\ped2_test1\gt\gt.txt",
    r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\ucsd_yolo\test\ped2_test2\gt\gt.txt",
    r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\ucsd_yolo\test\ped2_test3\gt\gt.txt",
]

GMOT_GT_FILES = [
    r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\gmot_yolo\test\airplane-1\gt\gt.txt",
    r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\gmot_yolo\test\ball-1\gt\gt.txt",
    r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\gmot_yolo\test\ball-2\gt\gt.txt",
    r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\gmot_yolo\test\ball-3\gt\gt.txt",
    r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\gmot_yolo\test\balloon-0\gt\gt.txt",
    r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\gmot_yolo\test\bird-2\gt\gt.txt",
    r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\gmot_yolo\test\bird-3\gt\gt.txt",
    r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\datasets\gmot_yolo\test\boat-1\gt\gt.txt",
]

Person = 6


def backup(path):
    bak = path.with_suffix(".txt.bak")
    if not bak.exists():
        shutil.copy(path, bak)
        print(f"backup {bak.name}")


def parse_line(line):
    line = line.strip()
    if not line:
        return None
    parts = line.split(",")
    if len(parts) < 6:
        parts = line.split()
    return parts if len(parts) >= 6 else None


# cvat gt fix
def fix_cvat_gt(gt_path):
    path = Path(gt_path)
    if not path.exists():
        print(f"fix_cvat_gt {gt_path} is skipped not found ")
        return
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        print(f"fix_cvat_gt {gt_path} is empty")
        return
    backup(path)
    fixed = []
    frame_changed = 0
    class_changed = 0
    for raw in lines:
        parts = parse_line(raw)
        if parts is None:
            continue
        try:
            frame = int(float(parts[0]))
            if frame >= 1:
                parts[0] = str(frame - 1)
                frame_changed += 1
        except ValueError:
            pass
        if len(parts) > 7:
            if parts[7].strip != str(Person):
                parts[7] = str(Person)
                class_changed += 1
            fixed.append(",".join(parts))
        path.write_text("\n".join(fixed), encoding="utf-8")
        print(f"{path.name}-{frame_changed} frames shifted, {class_changed} class indices fixed")

        # gmot label sanity check
        def check_gmot_gt(gt_path):
            path = Path(gt_path)
            if not path.exists():
                print(f"fix_cvat_gt {gt_path} is skipped not found ")
                return
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            frames, ids = [], []
            for raw in lines:
                parts = parse_line(raw)
                if parts is None:
                    continue
                try:
                    frames.append(int(float(parts[0])))
                    ids.append(int(float(parts[1])))
                except ValueError:
                    pass
            if not frames:
                print(f"{gt_path} no valid rows")
                return
            print(f"{path.name} - frames {min(frames)}-{max(frames)}")
            print(f"{len(set(ids))} unique ids, {len(frames)} rows")

            if __name__ == "__main__":
                print("Apply FIX to CVAT_GT files frame re-indexing and class index change")
                for f in CVAT_GT_FILES:
                    fix_cvat_gt(f)
                if GMOT_GT_FILES:
                    print("GMOT sanity check")
                    for f in GMOT_GT_FILES:
                        check_gmot_gt(f)
                print("Processs is finished")
