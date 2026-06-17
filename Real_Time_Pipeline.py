import argparse
import time
import cv2
import torch
import numpy as np
from pathlib import Path
from collections import deque
from ultralytics import YOLO
from boxmot.trackers.bytetrack.byte_tracker import BYTETracker

DetectorWeights = r"D:\runs_final\detect\all_datasets\weights\best.pt"
OutDir = Path(r"C:\Users\k2549603\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\realtime_outputs")
TrackerParams = {
    "track_high_thresh": 0.4,
    "track_buffer": 20,
    "match_thresh": 0.8,
}
Conf = 0.01
IOU = 0.5
Imgsz = 640
Device = 0

ClassNames = {
    0: "airplane",
    1: "fish",
    2: "ball",
    3: "bird",
    4: "boat",
    5: "balloon",
    6: "person",
    7: "insect",
    8: "stock",
    9: "car"
}
ClassColors = {
    0: (0, 165, 255),  # orange
    1: (255, 0, 0),  #blue
    2: (0, 255, 0),  #green
    3: (255, 255, 0),  # cyan
    4: (0, 0, 255),  # red
    5: (255, 0, 255),  # magenta
    6: (0, 255, 255),  # yellow
    7: (128, 0, 255),  #purple
    8: (255, 128, 0),  # light blue
    9: (0, 128, 255)  # orange-red
}


def draw_box(frame, x1, y1, x2, y2, tid, cls_id, conf):
    color = ClassColors.get(int(cls_id), (200, 200, 200))  # get the class colour
    label = f"ID: {tid} {ClassNames.get(int(cls_id), str(int(cls_id)))} {conf:.2f}"  # arrange the label that will be displayed
    # draw a rectangle
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    # draw label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(
        frame, (int(x1), int(y1) - th - 6),
        (int(x1) + tw + 4, int(y1)),
        color, -1
    )
    # draw label text in black
    cv2.putText(frame, label, (int(x1) + 2, int(y1) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def draw_overlay(frame, fps, vram_mb, n_tracks, paused=False):
    # display fps vram and track count in the top left corner
    h, w = frame.shape[:2]
    overlay = frame.copy()
    # create a semi-transparent background panel
    cv2.rectangle(overlay, (0, 0), (220, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    fpsColor = (0, 255, 0) if fps >= 25 else (0, 165, 255) if fps >= 15 else (0, 0, 255)
    vramColor = (0, 255, 0) if vram_mb < 2000 else (0, 165, 255) if vram_mb < 4000 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {fps:.2f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fpsColor, 1, cv2.LINE_AA)
    cv2.putText(frame, f"VRAM: {vram_mb:.0f} MB", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, vramColor, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Tracks: {n_tracks}", (8, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    if paused:
        cv2.putText(frame, "Paused", (8, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
    return frame


def run(source, save_video=False):
    OutDir.mkdir(parents=True, exist_ok=True)
    # load the detector model
    print(f"Loading detector from {DetectorWeights}")
    model = YOLO(DetectorWeights)
    # tracker initialisation
    tracker = BYTETracker(
        track_thresh=TrackerParams["track_thresh"],
        match_thresh=TrackerParams["match_thresh"],
        track_buffer=TrackerParams["track_buffer"],
        frame_rate=30,
    )
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source {source}")
    # get frame dimensions
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source opened: {frame_w}x{frame_h}")
    # optional video writer
    videoWriter = None
    recording = save_video
    if recording:
        ts = time.strftime("%Y%m%d_%H%M%S")
        videoPath = OutDir / f"tracking_{ts}.mp4"
        videoWriter = cv2.VideoWriter(str(videoPath), cv2.VideoWriter.fourcc(*"mp4v"), 30, (frame_w, frame_h))
        print(f"Recording video: {videoPath}")
    # fps smoothing for last 30 frames
    fps_buffer = deque(maxlen=30)
    paused = False
    frame_idx = 0
    # warm up running a dummy inference to avoid first frame being slow
    dummy = np.zeros((frame_h, frame_w, 3), np.uint8)
    model.predict(dummy, conf=Conf, iou=IOU, imgsz=Imgsz, verbose=False, device=Device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print("Press Q to quit S to screenshot and V to toggle recording.")
    while True:
        if paused:
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = False
            continue
        ret, frame = cap.read()
        if not ret:  # end of video file loop back to start or break for webcam
            if isinstance(source, str) and source != "0":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        t_start = time.perf_counter()
        # detection inference
        results = model.predict(frame, conf=Conf, iou=IOU, imgsz=Imgsz, verbose=False, device=Device)
        dets = []
        if results[0].boxes is not None and len(results[0].boxes):
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()
            for box, c, cls in zip(boxes, confs, clss):
                dets.append([*box, c, cls])
        dets_np = np.array(dets) if dets else np.empty((0, 6))

        # tracking
        tracks = tracker.update(dets_np, frame) if len(dets_np) else np.empty((0, 7))
        # drawing the tracks
        for t in tracks:
            x1, y1, x2, y2 = t[0], t[1], t[2], t[3]
            tid = int(t[4])
            conf_t = float(t[5])
            cls_id = int(t[6]) if len(t) > 6 else 6 # if class missing default to person class
            frame = draw_box(frame, x1, y1, x2, y2, tid, cls_id, conf_t)

        # compute soothed fps
        t_end = time.perf_counter()
        fps_buffer.append(1.0/(t_end - t_start) if(t_end - t_start)>0 else 0)
        fps = float(np.mean(fps_buffer))
        # measure VRAM
        vram_mb = 0.0
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated()/1024/1024
        frame = draw_overlay(frame,fps,vram_mb,len(tracks))
        # if it is recording create a video
        if recording and videoWriter is not None:
            videoWriter.write(frame)
        cv2.imshow("Real-Time Tracking", frame)
        # key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = True
        elif key == ord("s"):
            ts = time.strftime("%Y%m%d_%H%M%S")
            ss_path = OutDir / f"ss_{ts}.jpg"
            cv2.imwrite(str(ss_path), frame)
            print("Saved SS image")
        elif key == ord("v"):
            if not recording:
                ts=time.strftime("%Y%m%d_%H%M%S")
                videoPath = OutDir / f"tracking_{ts}.mp4"
                videoWriter = cv2.VideoWriter(
                    str(videoPath),
                    cv2.VideoWriter.fourcc(*"mp4v"), # fourcc might malfunction so double check that
                    30,(frame_w, frame_h),
                )
                recording = True
                print(f"Recording started:{videoPath}")
            else:
                if videoWriter:
                    videoWriter.release()
                recording = False
                print(f"Recording stopped:{videoPath}")
        frame_idx += 1

    # cleanup
    cap.release()
    if videoWriter:
        videoWriter.release()
    cv2.destroyAllWindows()
    print(f"Session ended. Processed {frame_idx} frames.")


# add main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Tracking Pipeline")
    parser.add_argument(
        "--source", default=0,
        help="Webcam index (0, 1, ...) or path to video file (default: 0)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Start recording immediately on launch"
    )
    args = parser.parse_args()

    # convert source to int if it's a digit string (webcam index)
    source = int(args.source) if str(args.source).isdigit() else args.source

    run(source=source, save_video=args.save)
