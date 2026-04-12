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
        split="train",
        epochs=number_of_epochs,
        imgsz=640,
        batchsz=batchSize,
        device=Device,
        project=train_run,  # each run do not forget to change these runs
        name=runName,
        patience=4,
        # other parameters will be added after looking at ultralytics docs
        # I need to find the best parameters that will optimise the model
        # for my case specific in this one
        # set pretrained true for training the best model again
        optimizer='auto',
        seed=42,
        # the ones below are explanation of hyper parameters
        # after the initial run check the results and apply the necessaryones
        # each I wrote their explanations if not enough check ultralytics again

        #cos_lr=True,
        #close_mosaic=3, # disable data augmentation in last 3 epochs for traning stabilisation
        # if I want to reduce memory usage I need to set amp=True
        # freeze (int): freezes first N layers or specified layers by index
        # it reduces the number of trainable parameters it is useful for
        # fine-tuning and transfer learning
        # lr0(float) = inital learning rate  adjusting this is important
        # lrf (float)= final learning rate as a fraction of the inital rate (lr0*lrf) used in conjunction
        # with schedulers to adjust the learning rate over time
        # momentum(float)= it is for SGD or beta1 for adam optmisors for influencing the
        # incorporation of past gradients in the current update
        # weight_decay(float): penalise large weights to prevent overfittin like 5e-5
        # warmup_epochs(float): e.g 3 . for learning rate warm up gradually increasing the learning rate from a low val
        # to the lr0 to stabilise training early on
        # warmup_momentum (float): test this but I guess using momentum won't be that much of a big deal so
        # warmup_bias_lr(float): learning rate for bias parameters during the warmup phase helping stablise model training
        # in the initial epochs
        # box (float) : weight of the box loss componenet in the loss function influencing how much emphasis is placed
        # on accuractely predicting the bounding box coordinates
        # cls (float): weight of the class loss component in the loss function influencing how much emphasis is placed
        # on accurately predicting the class labels
        # cls_pw (float): Power for class weighting to handle class imbalance using inverse class frequency.
        # 0.0 disables class weighting, 1.0 applies full inverse frequency weighting. Values between 0 and 1 provide partial weighting.
        # there are a few more but first deal with the ones above











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
        seed=42,
    )

    if __name__ == "__main__":
        results = train()
        best_weights = f"{train_run}/{runName}/weights/best.pt"
        validate(best_weights_path=best_weights)
