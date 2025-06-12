import subprocess
import numpy as np
import threading
import cv2
from queue import Queue
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from types import SimpleNamespace
from pymodbus.server import StartTcpServer
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
from pymodbus.datastore import ModbusSequentialDataBlock
import torch
import socket

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()

def modbus_server(context, ip):
    print(f"Starting Modbus server at {ip}:5020")
    StartTcpServer(context, address=(ip, 5020))

def read_frames(process, frame_queue, width=1920, height=1080):
    while True:
        raw_frame = process.stdout.read(width * height * 3)
        if not raw_frame:
            continue
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3)).copy()
        if not frame_queue.full():
            frame_queue.put(frame)

def get_model(fuse = True, grad = False, half = True):
    model = YOLO("./weights/best.pt")
    if fuse:
        model.fuse()  # Fuse Conv2d and BatchNorm layers for faster inference
    if torch.cuda.is_available():
        model.cuda()
    torch.set_grad_enabled(grad)  # Disable gradients for inference
    if half:
        model = model.half()  # Use half precision for faster inference (Use if GPU supports it)

def main ():
    # Get local IP
    ip = get_local_ip()

    # Modbus context setup
    store = ModbusSlaveContext(
        hr=ModbusSequentialDataBlock(0, [0]*10)  # 10 holding registers
    )
    context = ModbusServerContext(slaves=store, single=True)
    threading.Thread(target=modbus_server, args=(context, ip), daemon=True).start()

    # Load YOLOv8 model
    model = get_model(half = False)

    # RTSP stream and resolution
    RTSP_URL = 'rtsp://admin:Egg%21Camera1@192.168.140.10:554/h264Preview_01_main'
    width, height = 1920, 1080

    # FFmpeg command (change frame rate?)
    ffmpeg_cmd = [
        'ffmpeg',
        '-rtsp_transport', 'tcp',
        '-fflags', 'nobuffer',
        '-flags', 'low_delay',
        '-analyzeduration', '0',
        '-probesize', '32',
        '-i', RTSP_URL,
        '-f', 'image2pipe',
        '-pix_fmt', 'bgr24',
        '-vcodec', 'rawvideo',
        '-vf', f'scale={width}:{height}',
        '-r', '15',
        '-'
    ]

    # Setup tracker
    args = SimpleNamespace(
        track_thresh=0.5,
        track_buffer=15,
        match_thresh=0.7,
        min_box_area=100,
        mot20=False,
        frame_rate=15
    )
    tracker = BYTETracker(args)

    line_position = width//2
    counted_ids = set()
    total_count = 0
        
    # Start ffmpeg subprocess
    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

    frame_queue = Queue(maxsize=5)

    # Frame reader thread (non-blocking)
    threading.Thread(target=read_frames, args=(process, frame_queue, width, height), daemon=True).start()

    # Main loop
    while True:
        if frame_queue.empty():
            continue
        frame = frame_queue.get()

        results = model(frame)[0]
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

            # Find the corresponding detection confidence (fallback = 0.0)
            conf = 0.0
            for det in dets:
                if abs(det[0] - x) < 5 and abs(det[1] - y) < 5:
                    conf = det[4]
                    break

            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)
            cv2.putText(frame, f"ID:{track_id} | {conf:.2f}", (int(x), int(y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if track_id not in counted_ids and x < line_position < x + w:
                counted_ids.add(track_id)
                total_count += 1
                # Store total_count in register 0
                context[0].setValues(3, 0, [total_count])

            cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)

        cv2.line(frame, (line_position, 0), (line_position, frame.shape[1]), (0, 255, 0), 2)
        cv2.putText(frame, f"Total Count: {total_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        cv2.imshow("Egg Counter", frame)
        if cv2.waitKey(1) == ord('q'):
            cv2.imwrite("test_frame.jpg", frame)
            break

    process.terminate()
    cv2.destroyAllWindows()

main()