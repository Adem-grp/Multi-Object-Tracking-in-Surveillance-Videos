from ultralytics import YOLO
best_weights_path = r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\runs\detect\runs\All_Datasets\train\full_finetune(All_Datasets)\weights\best.pt"
source =r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\cvat_inputs\avenue\avenue_01.mp4"
output_path =r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\cvat_inputs\avenue_labels"
model = YOLO(best_weights_path)
model.predict(source=source,
              imgsz=640,
              conf=0.45,
              iou=0.6,
              save=False,
              save_txt=True,
              project = output_path,
              name ="avenue_01",
              exist_ok= True
              )


