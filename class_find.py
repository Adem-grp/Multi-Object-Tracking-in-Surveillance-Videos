

from collections import Counter
import glob

# Dataset root
ROOT = r"C:/Users/USER/PycharmProjects/Multi-Object-Tracking-in-Surveillance-Videos/datasets/gmot_yolo"


# Map the original GMOT class IDs to the 10 class IDs used in this project.

# Folder → class ID mapping
class_map = {
    "airplane": 0,
    "fish": 1,
    "ball": 2,
    "bird": 3,
    "boat": 4,
    "balloon": 5,
    "person": 6,
    "insect": 7,
    "stock": 8,
    "car": 9,
}
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


def audit_labels(root=ROOT):  # this only scans it how many labels exist per class ID and prints summary
    label_files = glob.glob(f"{root}/**/labels/*.txt", recursive=True)
    if not label_files:
        label_files = glob.glob(f"{root}/**/**/*.txt", recursive=True)
        # this is just a fail safe
    print(f"Found {len(label_files)} label files")
    class_counts = Counter()
    empty_files = 0
    for l in label_files:
        with open(l, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            empty_files += 1
            continue
        for line in lines:
            cls_id = int(line.split()[0])
            class_counts[cls_id] += 1
    print(f"Empty label files :{empty_files}")
    print(f"Unique Class IDs : {sorted(class_counts.keys())}")
    print("Counts per ID:")
    for key, val in sorted(class_counts.items()):
        print(f"ID: {key} : {val} bounding boxes")

    # if ID is not remapped :
    # change k>9 if  more classes are added
    if any(k > 9 for k in class_counts):
        print("Class IDs>9 fix it ")
    else:
        print("Class IDs are Ok continue")
    return class_counts

# rewrites IDs and replaces old ones within files
# soneeds to be run per dataset first time only
def remap_labels(root=ROOT,mapping = mapping):
    print("ID substitutions will be applied to all label files")
    for old_id, new_id in sorted(mapping.items()):
        print(f"old: {old_id} new: {new_id}")
    label_files = glob.glob(f"{root}/**/labels/*.txt", recursive=True)
    if not label_files:
        label_files = glob.glob(f"{root}/**/**/*.txt", recursive=True)
    print(f"Remapping starts..")
    warnings=0
    for lf in label_files:
        new_lines=[]
        with open(lf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.strip()
                old_id = int(parts[0])
                if old_id not in mapping:# detect unknown class ids
                    print(f"Unknown class ID {old_id} in {lf}")
                    warnings += 1
                    continue
                parts[0] = str(mapping[old_id])
                new_lines.append(" ".join(parts))
        with open(lf, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines))
    print(f"Remapping finished with {warnings} warnings")
    if warnings ==0:
        print("No warnings so continue no data loss")


if __name__ == "__main__":
    audit_labels()
    # auditlabels checks  if there are any problems with labels
    # remap labels directly remaps the labels accordingly
    #remap_labels()
