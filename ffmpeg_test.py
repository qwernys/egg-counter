import subprocess
import numpy as np
import cv2
import time

# Stream and resolution
RTSP_URL = 'rtsp://admin:Egg%21Camera1@192.168.140.51:554/h264Preview_01_main'
width, height = 1920, 1080
frame_size = width * height * 3

# FFmpeg command
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
    '-r', '25',  # Try matching camera FPS
    '-'
]

print("[INFO] Starting FFmpeg...")
process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

frame_count = 0
start_time = time.time()

try:
    while True:
        raw_frame = process.stdout.read(frame_size)
        if len(raw_frame) != frame_size:
            print("[WARNING] Incomplete frame received")
            continue

        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))

        frame_count += 1
        fps = frame_count / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        cv2.imshow("FFmpeg RTSP Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")

finally:
    print("[INFO] Cleaning up...")
    process.terminate()
    cv2.destroyAllWindows()
