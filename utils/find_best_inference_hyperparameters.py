import json

file_path =r"C:\Users\k2549603\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\YOLO_inference_evaluations\eval_results_YOLOm.json"

configs = []

with open(file_path, "r") as f:
    for line in f:
        configs.append(json.loads(line.strip()))

# remove duplicates (ignore max_det)
seen = {}
for c in configs:
    key = (c["conf"], c["iou"], c["agnostic_nms"])
    if key not in seen:
        seen[key] = c

configs = list(seen.values())


top_map = sorted(
    configs,
    key=lambda c: c["detection_metrics"]["mAP50"],
    reverse=True
)[:10]

top_recall = sorted(
    configs,
    key=lambda c: c["detection_metrics"]["recall"],
    reverse=True
)[:10]


map_keys = set((c["conf"], c["iou"], c["agnostic_nms"]) for c in top_map)
recall_keys = set((c["conf"], c["iou"], c["agnostic_nms"]) for c in top_recall)

intersection_keys = map_keys & recall_keys


if intersection_keys:
    candidates = [c for c in configs if (c["conf"], c["iou"], c["agnostic_nms"]) in intersection_keys]

    best = max(
        candidates,
        key=lambda c: c["detection_metrics"]["mAP50"]
    )

    print(" Best (intersection):")
    print(best)

else:
    print(" No overlap → using combined score")

    best = max(
        configs,
        key=lambda c: (
                c["detection_metrics"]["mAP50"] + 1.5 * c["detection_metrics"]["recall"]
        )
    )

    print(" Best (combined):")
    print(best)
