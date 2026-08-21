import pandas as pd
from pathlib import Path

csv_paths = {
    "bytetrack": r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\tracker_outputs\final_results_bytetrack.csv",
    "ocsort": r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\tracker_outputs\final_results_ocsort.csv",
    "deepsort": r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\tracker_outputs\final_results_deepsort.csv"

}
OutPath = r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\tracker_outputs\average_results_per_tracker.csv"

Metrics = ["HOTA (%)", "MOTA (%)", "IDF1 (%)",
           "ID Switches", "MostlyTracked", "MostlyLost", "FPS",
           "Latency mean (ms)", "Latency 95 (ms)", "Peak VRAM (MB)"
           ]
# average results across all datasets for each tracker and save to a new CSV file
def average_results():
    rows = []
    for tracker_name,csv_path in csv_paths.items():
        path = Path(csv_path)
        if not path.exists():
            print(f"{tracker_name} does not exist")
            continue
        df = pd.read_csv(path)
        print(f"\n{tracker_name.upper()} — {len(df)} dataset rows found: {df['dataset_name'].tolist()}")
        row = {"tracker": tracker_name,"n_datasets": len(df)}
        for metric in Metrics:
            if metric in df.columns:
                row[metric] = round(df[metric].mean(),3)
            else:
                row[metric]= None
                print(f"{metric} not found ")
        rows.append(row)
        # print per-dataset for sanity check
        for _, r in df.iterrows():
            print(f"    {r['dataset_name']:10} HOTA={r['HOTA (%)']:.2f}%  "
                  f"MOTA={r['MOTA (%)']:.2f}%  FPS={r['FPS']:.1f}")
        if not rows:
            print("No results found")
            return
        summary= pd.DataFrame(rows)
        summary= summary.sort_values("HOTA (%)", ascending=False).reset_index(drop=True)
        summary.to_csv(OutPath, index=False)
        print("AVERAGED RESULTS PER TRACKER (across all 4 datasets)")
        print(summary.to_string(index=False))
        print(f"\nSaved to {OutPath}")


if __name__ == "__main__":
    average_results()
