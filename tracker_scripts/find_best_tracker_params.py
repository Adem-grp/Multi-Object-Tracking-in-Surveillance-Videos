# verify_best_params.py
# Reads tuning CSVs and finds the single best hyperparameter set per tracker
# averaged across all datasets. Run this to confirm the params before using them.

import pandas as pd
from collections import Counter
from pathlib import Path


CSV_PATHS = {
    "bytetrack": r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\tracker_outputs\tuning_bytetrack.csv",
    "deepsort": r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\tracker_outputs\tuning_deepsort.csv",
    "ocsort": r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\tracker_outputs\tuning_ocsort.csv",
}

TRACKER_PARAM_COLS = {
    "bytetrack": ["track_high_thresh", "track_buffer", "match_thresh"],
    "deepsort": ["max_dist", "max_age", "n_init", "max_iou_dist"],
    "ocsort": ["det_thresh", "max_age", "min_hits", "iou_threshold"],
}

EXPECTED_DATASETS = ["avenue", "UCSD", "shanghai", "gmot"]

# find the best conf/iou and tracker params averaged across all datasets for a given tracker
def verify_and_find_best(tracker, csv_path, param_cols):
    print(f"\n{'=' * 60}")
    print(f"TRACKER: {tracker.upper()}")
    print(f"{'=' * 60}")

    df = pd.read_csv(csv_path)

    # --- data completeness checks ---
    s1 = df[df["stage"] == 1]
    s2 = df[df["stage"] == 2]
    print(f"\n[Data check]")
    print(f"  Total rows: {len(df)} | Stage1: {len(s1)} | Stage2: {len(s2)}")
    datasets_found = sorted(s2["dataset_name"].unique())
    print(f"  Datasets in stage2: {datasets_found}")
    rows_per_ds = s2["dataset_name"].value_counts().to_dict()
    print(f"  Rows per dataset: {rows_per_ds}")
    nan_count = s2["HOTA (%)"].isna().sum()
    print(f"  NaN HOTA values: {nan_count}")
    if nan_count > 0:
        print(f"  [WARN] NaN values found — results may be incomplete")
    missing = [d for d in EXPECTED_DATASETS if d not in datasets_found]
    if missing:
        print(f"  [WARN] Missing datasets: {missing}")
    else:
        print(f"  All 4 datasets present — data is complete")

    # --- stage 1: find best conf/iou per dataset ---
    print(f"\n[Stage 1 — best conf/iou per dataset]")
    best_confs, best_ious = [], []
    for ds in EXPECTED_DATASETS:
        s1_ds = s1[s1["dataset_name"] == ds]
        if s1_ds.empty:
            print(f"  {ds}: missing")
            continue
        best = s1_ds.loc[s1_ds["HOTA (%)"].idxmax()]
        best_confs.append(float(best["conf"]))
        best_ious.append(float(best["iou"]))
        print(f"  {ds}: conf={best['conf']}  iou={best['iou']}  HOTA={best['HOTA (%)']:.3f}%")

    # most common conf and iou across datasets
    best_conf = Counter(best_confs).most_common(1)[0][0]
    best_iou = Counter(best_ious).most_common(1)[0][0]
    print(f"\n  Most common conf across datasets: {best_conf}")
    print(f"  Most common iou  across datasets: {best_iou}")

    # --- stage 2: find best tracker params averaged across all datasets ---
    print(f"\n[Stage 2 — best tracker params averaged across all datasets]")
    s2 = s2.copy()
    s2["param_key"] = s2[param_cols].astype(str).agg("_".join, axis=1)

    # only consider combos present in ALL datasets for fair averaging
    counts = s2.groupby("param_key")["dataset_name"].nunique()
    full_combos = counts[counts == len(datasets_found)].index
    print(f"  Combos present in all {len(datasets_found)} datasets: {len(full_combos)}")

    s2_full = s2[s2["param_key"].isin(full_combos)]
    avg = s2_full.groupby("param_key")["HOTA (%)"].mean().reset_index()
    avg.columns = ["param_key", "avg_hota"]
    avg = avg.sort_values("avg_hota", ascending=False).reset_index(drop=True)

    best_key = avg.loc[0, "param_key"]
    best_avg = avg.loc[0, "avg_hota"]
    best_row = s2_full[s2_full["param_key"] == best_key].iloc[0]

    print(f"\n  Best combo — avg HOTA: {best_avg:.4f}%")
    print(f"  Per-dataset HOTA for this combo:")
    for ds in EXPECTED_DATASETS:
        ds_rows = s2_full[(s2_full["param_key"] == best_key) & (s2_full["dataset_name"] == ds)]
        if not ds_rows.empty:
            print(f"    {ds}: {ds_rows['HOTA (%)'].values[0]:.3f}%")
        else:
            print(f"    {ds}: missing")

    print(f"\n[FINAL PARAMS — {tracker.upper()}]")
    print(f"  conf:            {best_conf}")
    print(f"  iou:             {best_iou}")
    for c in param_cols:
        print(f"  {c}: {best_row[c]}")
    print(f"  Average HOTA:    {best_avg:.4f}%")

    return {
        "tracker": tracker,
        "conf": best_conf,
        "iou": best_iou,
        "avg_hota": round(best_avg, 4),
        **{c: best_row[c] for c in param_cols},
    }


if __name__ == "__main__":
    results = []
    for tracker, csv_path in CSV_PATHS.items():
        if not Path(csv_path).exists():
            print(f"\n[SKIP] {tracker} — CSV not found at {csv_path}")
            continue
        param_cols = TRACKER_PARAM_COLS[tracker]
        result = verify_and_find_best(tracker, csv_path, param_cols)
        results.append(result)

    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY — USE THESE PARAMS")
    print(f"{'=' * 60}")
    for r in results:
        print(f"\n{r['tracker'].upper()} (avg HOTA: {r['avg_hota']}%)")
        for k, v in r.items():
            if k not in ["tracker", "avg_hota"]:
                print(f"  {k}: {v}")
