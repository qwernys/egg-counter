import cv2
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from types import SimpleNamespace
import torch
import os
import argparse
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Egg Counter")

    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    parser.add_argument("--data-dir", type=str, default="/app/data", 
                        help="Directory to store data files")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--camera-id", type=int, choices=[1,2,3,4], default=4,
                        help="Camera ID (1-4)")
    parser.add_argument("--line-offset", type=int, default=0,
                        help="Offset for the counting line position")

    return parser.parse_args()

def get_model(fuse = True, grad = False, half = True):
    model = YOLO("./weights/best.pt")
    if fuse:
        model.fuse()  # Fuse Conv2d and BatchNorm layers for faster inference
    if torch.cuda.is_available():
        model.cuda()
    torch.set_grad_enabled(grad)  # Disable gradients for inference
    if half:
        model = model.half()  # Use half precision for faster inference (Use if GPU supports it)

    return model

def write_to_file_safe(path, data):
    temp_path = path + ".tmp"
    with open(temp_path, "w") as f:
        f.write(data)
    os.replace(temp_path, path)

def update_date_file(date_path, today):
    with open(date_path, "r") as f:
        daily_data = f.read().strip()
        date_str, count_str = daily_data.split(',')
        last_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        daily_count = int(count_str)

    if today != last_date:
        daily_count = 0
        write_to_file_safe(date_path, f"{today},0")

    return daily_count

def main (args):
    cameraId = args.camera_id
    verbose = args.verbose

    count_xb_path = os.path.join(args.data_dir, f"count_{cameraId}b.txt")
    count_xa_path = os.path.join(args.data_dir, f"count_{cameraId}a.txt")

    daily_xa_path = os.path.join(args.data_dir, f"daily_{cameraId}a.txt")
    daily_xb_path = os.path.join(args.data_dir, f"daily_{cameraId}b.txt")

    # RTSP stream and resolution
    # rtsp://<username>:<password>@<camera_ip>:554/h264Preview_01_main
    RTSP_URL = f'rtsp://admin:Egg%21Camera1@192.168.140.5{cameraId}:554/h264Preview_01_main'
    width, height = 1920, 1080
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    today = datetime.now().date()

    if not cap.isOpened():
        print("Error: Cannot open stream")
        exit()

    if not os.path.exists(count_xb_path):
        write_to_file_safe(count_xb_path, "0")
    if not os.path.exists(daily_xb_path):
        write_to_file_safe(daily_xb_path, f"{today},0")
    if not os.path.exists(count_xa_path):
        write_to_file_safe(count_xa_path, "0")
    if not os.path.exists(daily_xa_path):
        write_to_file_safe(daily_xa_path, f"{today},0")
    
    with open(count_xb_path, "r") as f:
        total_count_xb = int(f.read().strip())

    with open(count_xa_path, "r") as f:
        total_count_xa = int(f.read().strip())

    daily_xb = update_date_file(daily_xb_path, today)
    daily_xa = update_date_file(daily_xa_path, today)

    # Load YOLOv8 model
    model = get_model(fuse=True, grad=False, half=False)

    # Setup tracker
    byte_args = SimpleNamespace(
        track_thresh=0.5,
        track_buffer=15,
        match_thresh=0.7,
        min_box_area=100,
        mot20=False,
        frame_rate=25
    )
    tracker = BYTETracker(byte_args)

    hor_line = height//2
    ver_line = width//2 + args.line_offset
    counted_ids = set()

    # Initialize error state (Temporary fix for log spam at stream read failure)
    error = False

    # Main loop
    while True:
        if datetime.now().date() != today:
            daily_xa = 0
            daily_xb = 0
            write_to_file_safe(daily_xb_path, f"{today},0")
            write_to_file_safe(daily_xa_path, f"{today},0")
            today = datetime.now().date()


        ret, frame = cap.read()
        if not ret and not error:
            print("Stream read failed")
            error = True
            continue
    
        error = False  # Reset error state on successful read

        frame = cv2.resize(frame, (width, height))

        results = model(frame, verbose=verbose)[0]
        detections = results.boxes

        dets = []
        for box in detections:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            if conf > 0.5:
                dets.append([x1, y1, x2, y2, conf])
        
        if dets:
            dets_tensor = torch.tensor(dets, dtype=torch.float32)
            tracks = tracker.update(dets_tensor, frame.shape[:2], frame.shape)
        else:
            tracks = []

        for track in tracks:
            track_id = int(track.track_id)
            x, y, w, h = track.tlwh

            if track_id not in counted_ids and y < hor_line < y + h:
                counted_ids.add(track_id)

                if x + w // 2 < ver_line:
                    total_count_xb += 1
                    daily_xb += 1
                else:
                    total_count_xa += 1
                    daily_xa += 1
        
                # Update Modbus register and file

                write_to_file_safe(count_xb_path, str(total_count_xb))
                write_to_file_safe(count_xa_path, str(total_count_xa))
                write_to_file_safe(daily_xb_path, f"{today},{daily_xb}")
                write_to_file_safe(daily_xa_path, f"{today},{daily_xa}")

def debug (args):
    SAVE_DIR = os.path.join(args.data_dir, "test_image.png")

    cameraId = args.camera_id
    verbose = args.verbose

    # RTSP stream and resolution
    RTSP_URL = f'rtsp://admin:Egg%21Camera1@192.168.140.5{cameraId}:554/h264Preview_01_main'
    width, height = 1920, 1080
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print("Error: Cannot open stream")
        exit()

    # Load YOLOv8 model
    model = get_model(fuse=True, grad=False, half=False)

    # Setup tracker
    byte_args = SimpleNamespace(
        track_thresh=0.5,
        track_buffer=15,
        match_thresh=0.7,
        min_box_area=100,
        mot20=False,
        frame_rate=25
    )
    tracker = BYTETracker(byte_args)

    hor_line = height//2
    ver_line = width//2 + args.line_offset
    total_count = 0
    counted_ids = set()

    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream read failed")
            continue

        frame = cv2.resize(frame, (width, height))

        results = model(frame, verbose=verbose)[0]

        detections = results.boxes

        dets = []
        for box in detections:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            if conf > 0.5:
                dets.append([x1, y1, x2, y2, conf])
        
        if dets:
            dets_tensor = torch.tensor(dets, dtype=torch.float32)
            tracks = tracker.update(dets_tensor, frame.shape[:2], frame.shape)
        else:
            tracks = []

        for track in tracks:
            track_id = int(track.track_id)
            x, y, w, h = track.tlwh
            center_x = x + w / 2
            center_y = y + h / 2

            if track_id not in counted_ids and y < hor_line < y + h:
                counted_ids.add(track_id)
                total_count += 1
            
            # Find the corresponding detection confidence (fallback = 0.0)
            conf = 0.0
            for det in dets:
                if abs(det[0] - x) < 5 and abs(det[1] - y) < 5:
                    conf = det[4]
                    break

            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)
            cv2.putText(frame, f"ID:{track_id} | {conf:.2f}", (int(x), int(y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)

        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(frame, f"FPS: {fps:.2f}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        cv2.line(frame, (0, hor_line), (frame.shape[1], hor_line), (0, 255, 0), 2)
        cv2.line(frame, (ver_line, 0), (ver_line, frame.shape[0]), (255, 0, 0), 2)
        cv2.putText(frame, f"Total Count: {total_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
        cv2.imwrite(SAVE_DIR, frame)
        break
    
    cap.release()

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        debug(args)
    else:
        main(args)