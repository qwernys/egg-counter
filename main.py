import threading
import cv2
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from types import SimpleNamespace
from pymodbus.server import StartTcpServer
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
from pymodbus.datastore import ModbusSequentialDataBlock
import torch
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Egg Counter")

    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    parser.add_argument("--data-dir", type=str, default="/app/data", 
                        help="Directory to store data files")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")

    return parser.parse_args()

def modbus_server(context):
    print(f"Starting Modbus server at 0.0.0.0:5020")
    StartTcpServer(context, address=("0.0.0.0", 5020))

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

def main (args):
    verbose = args.verbose
    path = os.path.join(args.data_dir, "total_count.txt")
    # RTSP stream and resolution
    RTSP_URL = 'rtsp://admin:Egg%21Camera1@192.168.140.51:554/h264Preview_01_main'
    width, height = 1920, 1080
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print("Error: Cannot open stream")
        exit()

    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("0")
    # Initialize total count from file
    with open(path, "r") as f:
        total_count = int(f.read().strip())

    # Modbus context setup
    store = ModbusSlaveContext(
        hr=ModbusSequentialDataBlock(0, [0]*10)  # 10 holding registers
    )
    context = ModbusServerContext(slaves=store, single=True)
    context[0].setValues(3, 0, [total_count])  # Initialize register 0 with total count
    threading.Thread(target=modbus_server, args=(context,), daemon=True).start()

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

    line_position = height//2
    counted_ids = set()

    # Initialize error state (Temporary fix for log spam at stream read failure)
    error = False

    # Main loop
    while True:
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

            if track_id not in counted_ids and y < line_position < y + h:
                counted_ids.add(track_id)
                total_count += 1
                # Update Modbus register and file
                with open(path, "w") as f:
                    f.write(str(total_count))
                context[0].setValues(3, 0, [total_count])

def debug (args):
    verbose = args.verbose
    path = os.path.join(args.data_dir, "total_count.txt")
    # RTSP stream and resolution
    RTSP_URL = 'rtsp://admin:Egg%21Camera1@192.168.140.51:554/h264Preview_01_main'
    width, height = 1920, 1080
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print("Error: Cannot open stream")
        exit()

    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("0")
    # Initialize total count from file
    with open(path, "r") as f:
        total_count = int(f.read().strip())
    print(f"Initial total count: {total_count}")

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

    line_position = height//2
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

            if track_id not in counted_ids and y < line_position < y + h:
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
        
        cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 255, 0), 2)
        cv2.putText(frame, f"Total Count: {total_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        cv2.imshow("Egg Counter", frame)
        if cv2.waitKey(1) == ord('q'):
            cv2.imwrite("test_frame.jpg", frame)
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        debug(args)
    else:
        main(args)