from ultralytics import YOLO
import torch

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yamlPath = "gmot.yaml"
Model = "yolo11n.pt"
# each run do not forget to change these runs
train_run = "runs/run_1/train"
val_run = "runs/run_1/val"
number_of_epochs = 10
runName = "gmot_train_v1"  # Do not forget to change this each run tho
batchSize = 32
number_of_epochs = 30


def train():
    model = YOLO(Model)
    results = model.train(
        data=yamlPath,
        epochs=number_of_epochs,
        imgsz=640,
        batchsz=batchSize,
        device=Device,
        project=train_run,  # each run do not forget to change these runs
        name=runName,
        # other parameters will be added after looking at ultralytics docs
        # I need to find the best parameters that will optimise the model
        # for my case specific in this one

    )
    return results


def validate(best_weights_path):
    model = YOLO(best_weights_path)
    results = model.val(
        data=yamlPath,
        split="val",
        epochs=number_of_epochs,
        imgsz=640,
        batchsz=batchSize,
        device=Device,
        project=val_run,
        name=runName,
    )

    if __name__ == "__main__":
        results = train()
        best_weights = f"{train_run}/{runName}/weights/best.pt"
        validate(best_weights_path=best_weights)
