import cv2

cap = cv2.VideoCapture("rtsp://admin:Egg%21Camera1@192.168.140.51:554/h264Preview_01_main")

if not cap.isOpened():
    print("Error: Cannot open stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream read failed")
        continue

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
