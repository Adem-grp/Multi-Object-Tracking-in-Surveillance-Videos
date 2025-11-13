import glob
from collections import Counter
import os
label_files = glob.glob("gmot_yolo/**/**/*.txt", recursive=True)

class_counts = Counter()

for lf in label_files:
    with open(lf, "r") as f:
        for line in f:
            if line.strip():
                cls_id = int(line.split()[0])
                class_counts[cls_id] += 1

print("Unique class IDs found:", sorted(class_counts.keys()))
print("\nCounts per class:")
for cls, cnt in sorted(class_counts.items()):
    print(f"Class {cls}: {cnt} samples")


# Mapping from old class IDs to new 0-9 IDs
mapping = {
    42: 0,
    59: 1,
    130: 2,
    267: 3,
    348: 4,
    460: 5,
    772: 6,
    967: 7,
    973: 8,
    995: 9
}

label_files = glob.glob("gmot_yolo/**/labels/*.txt", recursive=True)

for lf in label_files:
    new_lines = []
    with open(lf, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            old_id = int(parts[0])
            if old_id not in mapping:
                print(f"[WARNING] Unknown class {old_id} in {lf}")
                continue

            parts[0] = str(mapping[old_id])
            new_lines.append(" ".join(parts))

    with open(lf, "w") as f:
        f.write("\n".join(new_lines))

print(" All labels remapped successfully!")
