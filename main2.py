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
from datetime import datetime

from pymodbus.payload import BinaryPayloadBuilder
from pymodbus.constants import Endian

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

def update_date_file(date_path, today, row):
    with open(date_path, "r") as f:
        lines = f.read().strip().split('\n')
        date_str = lines[0]
        daily_data = lines[1:]
        last_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        daily_count = int(daily_data[row].split(',')[1])

    if today != last_date:
        daily_count = 0
        with open(date_path, "w") as f:
            # Total daily count for all cameras
            f.write(f"{today}\n") # Date
            f.write("0\n") # Total daily count for all cameras
            for i in range(0,4): # Daily count for each camera
                f.write(f"0\n") # a
                f.write(f"0\n") # b

    return daily_count

def get_register(value):
    """Convert a 32-bit integer to Modbus holding register format."""

    builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.BIG)
    builder.add_32bit_uint(value)
    return builder.to_registers()

def check_caps(caps):
    for cap in caps:
        if not cap.isOpened():
            print("Error: Cannot open one of the camera streams")
            exit()

def main (args):
    # Init
    verbose = args.verbose
    count_path = os.path.join(args.data_dir, "total_count.txt")
    date_path = os.path.join(args.data_dir, "last_date.txt")

    # RTSP stream and resolution
    CAM1_RTSP_URL = 'rtsp://admin:Egg%21Camera1@192.168.140.51:554/h264Preview_01_main'
    CAM2_RTSP_URL = 'rtsp://admin:Egg%21Camera1@192.168.140.51:554/h264Preview_01_main'
    CAM3_RTSP_URL = 'rtsp://admin:Egg%21Camera1@192.168.140.51:554/h264Preview_01_main'
    CAM4_RTSP_URL = 'rtsp://admin:Egg%21Camera1@192.168.140.51:554/h264Preview_01_main'

    width, height = 1920, 1080

    cap1 = cv2.VideoCapture(CAM1_RTSP_URL)
    cap2 = cv2.VideoCapture(CAM2_RTSP_URL)
    cap3 = cv2.VideoCapture(CAM3_RTSP_URL)
    cap4 = cv2.VideoCapture(CAM4_RTSP_URL)

    caps = [cap1, cap2, cap3, cap4]

    today = datetime.now().date()

    if not os.path.exists(date_path):
        with open(date_path, "w") as f:
            f.write(f"{today}\n") # Date
            f.write("0\n") # Total daily count for all cameras
            for i in range(0,4): # Daily count for each camera
                f.write("0\n") # a
                f.write("0\n") # b

    check_caps(caps)

    if not os.path.exists(count_path):
        with open(count_path, "w") as f:
            f.write("0\n") # Total count
            for i in range(0,4): # Total count for each camera
                f.write("0\n") # a
                f.write("0\n") # b

    # Initialize total counts from file
    with open(count_path, "r") as f:
        count_data = f.read().strip().split('\n')

    # Initialize daily counts from file
    with open(date_path, "r") as f:
        lines = f.read().strip().split('\n')
        today = lines[0]
        daily_data = lines[1:]

    dailies = [] # [total, 1a, 1b, 2a, 2b, 3a, 3b, 4a, 4b]
    counts = [] # [total, 1a, 1b, 2a, 2b, 3a, 3b, 4a, 4b] 

    for i in range(0,9):
        dailies.append(int(daily_data[i]))
        counts.append(int(count_data[i]))

    # Modbus context setup
    store = ModbusSlaveContext(
        hr=ModbusSequentialDataBlock(0, [0]*34)  # 34 holding registers
    )
    context = ModbusServerContext(slaves=store, single=True)
    for i in range(len(counts)):
        context[0].setValues(3, 0 + 4 * i, get_register(counts[i]))  # Initialize registers with total counts
        context[0].setValues(3, 2 + 4 * i, get_register(dailies[i]))  # Initialize registers with daily counts

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
    trackers = [BYTETracker(byte_args) for _ in range(4)]

    hor_line = height//2
    ver_line = width//2
    counters = [set() for _ in range(4)]  # Separate counted IDs for each camera

    # Initialize error state (Temporary fix for log spam at stream read failure)
    error = False

    # Main loop
    while True:
        if datetime.now().date() != today:
            # Reset daily counts at midnight
            today = datetime.now().date()
            dailies = [0]*9
            with open(date_path, "w") as f:
                f.write(f"{today}\n") # Date
                f.write("0\n") # Total daily count for all cameras
                for i in range(0,4): # Daily count for each camera
                    f.write("0\n") # a
                    f.write("0\n") # b
            
            for i in range(0,9):
                context[0].setValues(3, 2 + 4 * i, get_register(dailies[i]))  # Reset daily count registers to 0

        i = 0

        for cap in caps:
            ret, frame = cap.read()
            tracker = trackers[i]
            counted_ids = counters[i]

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
                
                    # update total over all time and total daily
                    counts[0] += 1
                    dailies[0] += 1

                    if x + w // 2 < ver_line:
                        counts[2 + 2*i] += 1 # b
                        dailies[2 + 2*i] += 1
                    else:
                        counts[1 + 2*i] += 1 # a
                        dailies[1 + 2*i] += 1

                    # Update Modbus register and file
                    with open(count_path, "w") as f:
                        f.write(str(counts[0]) + '\n') # Total count
                        for j in range(0,4): # Total count for each camera
                            f.write(str(counts[1 + 2*j]) + '\n') # a
                            f.write(str(counts[2 + 2*j]) + '\n') # b

                    with open(date_path, "w") as f:
                        f.write(f"{today}\n") # Date
                        f.write(f"{dailies[0]}\n") # Total daily count for all cameras
                        for j in range(0,4): # Daily count for each camera
                            f.write(f"{dailies[1 + 2*j]}\n") # a
                            f.write(f"{dailies[2 + 2*j]}\n") # b
                    
                    context[0].setValues(3, 0, get_register(counts[0]))  # Update total count register
                    context[0].setValues(3, 2, get_register(dailies[0]))  # Update daily count register

                    a_idx = 2 * i + 1
                    b_idx = 2 * i + 2

                    context[0].setValues(3, 4 + i * 8, get_register(counts[a_idx]))   # cam A total
                    context[0].setValues(3, 4 + i * 8 + 2, get_register(counts[b_idx]))   # cam B total
                    context[0].setValues(3, 4 + i * 8 + 4, get_register(dailies[a_idx]))  # cam A daily
                    context[0].setValues(3, 4 + i * 8 + 6, get_register(dailies[b_idx]))  # cam B daily

            i += 1

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
    trackers = [BYTETracker(byte_args) for _ in range(4)]

    hor_line = height//2
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