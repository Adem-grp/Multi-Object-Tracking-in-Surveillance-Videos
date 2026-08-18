from ultralytics import YOLO
from pathlib import Path
import os
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
import json
import shutil

# Configure Ray Tune logging and restrict the experiment to the CUDA.
os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BASE_DIR = r"C:\ray_runs"
os.makedirs(BASE_DIR, exist_ok=True)

# yaml for train test and val
yamlPath = r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\gmot.yaml"
results_json_path = os.path.join(BASE_DIR, "yolo11m_results.json")  # results stored in json file


def train_yolo(config):
    from filelock import FileLock

    trial_id = session.get_trial_id()

    trial_dir = os.path.join(BASE_DIR, f"trial_{trial_id}")  # each rayTune trial has its own id
    os.makedirs(trial_dir, exist_ok=True)

    model = YOLO("yolo11m.pt")  # load yolo11m

    results = model.train(
        # using GMOT-40 dataset train and validate to see which hyperparameter combination is the best
        data=str(Path(yamlPath).resolve()),
        epochs=40,
        imgsz=640,
        batch=8,
        device=0,

        lr0=config["lr0"],  # parameters that will be optimised
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],

        optimizer="SGD",
        patience=8,  # if training does not improve we stop at patience 8 to avoid overfitting
        seed=42,
        verbose=False,

        project=trial_dir,
        name="run"
    )

    metrics = results.results_dict  # get the detection metric results from validation

    result_entry = {  # results are arranged according to json format hyperparameters and results are stored together
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
    # lock is applied to make sure only one trial modifies or reads the results file

    with lock:  # prevents concurrent trials from corrupting the shared results file
        if os.path.exists(results_json_path):
            try:
                with open(results_json_path, "r") as f:  # read the file take the existing data
                    data = json.load(f)
            except:
                data = []
        else:
            data = []

        data.append(result_entry)  # add the new data

        with open(results_json_path, "w") as f:  # and write them all together
            json.dump(data, f, indent=4)

    tune.report({  # send metrics back to rayTune for trial comparison and scheduling
        "mAP50": result_entry["mAP50"],
        "precision": result_entry["precision"],
        "recall": result_entry["recall"],
        "trial_id": trial_id
    })


# Search ranges for the three SGD training hyperparameters.
# Log-uniform sampling is used for learning rate and weight decay.
search_space = {
    "lr0": tune.loguniform(1e-4, 1e-2),
    "momentum": tune.uniform(0.7, 0.95),
    "weight_decay": tune.loguniform(1e-5, 5e-4)
}

# Asha allows ray tune to stop trials that perform poorly rather than allowing every config to train for full 40 epochs

scheduler = ASHAScheduler(
    max_t=40,
    grace_period=8,
    reduction_factor=2
)

if __name__ == "__main__":

    ray.init()  # initialize ray

    trainable = tune.with_resources(
        train_yolo,
        resources={"cpu": 2, "gpu": 1}  # arranging the resources properly
    )

    tuner = tune.Tuner( # tuner connects training function with search space
        trainable,
        param_space=search_space,

        tune_config=tune.TuneConfig(
            metric="mAP50", # use mAP50 to evaluate performance
            mode="max", # higher mAP50 is better
            scheduler=scheduler,
            num_samples=15,
            max_concurrent_trials=1 # only one trial will run at a time
        ),

        run_config=ray.air.RunConfig(
            name="exp",
            storage_path=BASE_DIR,
            log_to_file=False
        )
    )

    results = tuner.fit()
    # get the highest mAP50 trial
    best_result = results.get_best_result(metric="mAP50", mode="max")

    print("\nBEST CONFIG:", best_result.config)
    print("\nBEST METRICS:", best_result.metrics)

    # locate the best trial to see the hyperparameters
    best_trial_id = best_result.metrics["trial_id"]
    best_trial_dir = os.path.join(BASE_DIR, f"trial_{best_trial_id}")
    # Copy the best trial's weights to a certain location for further evaluation.
    best_weights_path = os.path.join(best_trial_dir, "run", "weights", "best.pt")
    final_model_path = os.path.join(BASE_DIR, "best_baseline_before_dataset_comb.pt")

    if os.path.exists(best_weights_path):
        shutil.copy(best_weights_path, final_model_path)
        print("\nBest model saved to:", final_model_path)
    else:
        print("\nCould not locate best.pt")
