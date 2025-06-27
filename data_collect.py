import cv2
import os
import time
from datetime import datetime

# === Settings ===
DATA_DIR = "/app/data"
SAVE_DIR = os.path.join(DATA_DIR, "images")
INTERVAL_SECONDS = 1
RTSP_URL = 'rtsp://admin:Egg%21Camera1@192.168.140.51:554/h264Preview_01_main'

# === Create save directory if it doesn't exist ===
os.makedirs(SAVE_DIR, exist_ok=True)

# === Open video source ===
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

print(f"Saving images to: {SAVE_DIR}")
print("Press Ctrl+C to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to capture frame.")
            continue

        # Create a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_DIR, f"{timestamp}.jpg")

        # Save the image
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")

        # Wait before capturing the next frame
        time.sleep(INTERVAL_SECONDS)

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    cap.release()
    print("Video capture released.")