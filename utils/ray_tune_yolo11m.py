from ultralytics import YOLO
from pathlib import Path
import os
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
import json
import shutil


os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


BASE_DIR = r"C:\ray_runs"
os.makedirs(BASE_DIR, exist_ok=True)

yamlPath = r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\gmot.yaml"
results_json_path = os.path.join(BASE_DIR, "yolo11m_results.json")


def train_yolo(config):

    from filelock import FileLock

    trial_id = session.get_trial_id()


    trial_dir = os.path.join(BASE_DIR, f"trial_{trial_id}")
    os.makedirs(trial_dir, exist_ok=True)

    model = YOLO("yolo11m.pt") # trained on gmot using yaml and evaluated on gmot val 

    results = model.train(
        data=str(Path(yamlPath).resolve()),
        epochs=40,
        imgsz=640,
        batch=8,
        device=0,

        lr0=config["lr0"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],

        optimizer="SGD",
        patience=8,
        seed=42,
        verbose=False,

        project=trial_dir,
        name="run"
    )

    metrics = results.results_dict

    result_entry = {
        "trial_id": trial_id,
        "lr0": config["lr0"],
        "momentum": config["momentum"],
        "weight_decay": config["weight_decay"],
        "precision": float(metrics["metrics/precision(B)"]),
        "recall": float(metrics["metrics/recall(B)"]),
        "mAP50": float(metrics["metrics/mAP50(B)"]),
        "mAP50-95": float(metrics["metrics/mAP50-95(B)"])
    }


    lock = FileLock(results_json_path + ".lock")

    with lock:
        if os.path.exists(results_json_path):
            try:
                with open(results_json_path, "r") as f:
                    data = json.load(f)
            except:
                data = []
        else:
            data = []

        data.append(result_entry)

        with open(results_json_path, "w") as f:
            json.dump(data, f, indent=4)


    tune.report({
        "mAP50": result_entry["mAP50"],
        "precision": result_entry["precision"],
        "recall": result_entry["recall"],
        "trial_id": trial_id
    })



search_space = {
    "lr0": tune.loguniform(1e-4, 1e-2),
    "momentum": tune.uniform(0.7, 0.95),
    "weight_decay": tune.loguniform(1e-5, 5e-4)
}

scheduler = ASHAScheduler(
    max_t=40,
    grace_period=8,
    reduction_factor=2
)


if __name__ == "__main__":

    ray.init()

    trainable = tune.with_resources(
        train_yolo,
        resources={"cpu": 2, "gpu": 1}
    )

    tuner = tune.Tuner(
        trainable,
        param_space=search_space,

        tune_config=tune.TuneConfig(
            metric="mAP50",
            mode="max",
            scheduler=scheduler,
            num_samples=15,
            max_concurrent_trials=1
        ),

        run_config=ray.air.RunConfig(
            name="exp",
            storage_path=BASE_DIR,
            log_to_file=False
        )
    )

    results = tuner.fit()

    best_result = results.get_best_result(metric="mAP50", mode="max")

    print("\nBEST CONFIG:", best_result.config)
    print("\nBEST METRICS:", best_result.metrics)

    best_trial_id = best_result.metrics["trial_id"]
    best_trial_dir = os.path.join(BASE_DIR, f"trial_{best_trial_id}")

    best_weights_path = os.path.join(best_trial_dir, "run", "weights", "best.pt")
    final_model_path = os.path.join(BASE_DIR, "best_overall.pt")

    if os.path.exists(best_weights_path):
        shutil.copy(best_weights_path, final_model_path)
        print("\nBest model saved to:", final_model_path)
    else:
        print("\nCould not locate best.pt")