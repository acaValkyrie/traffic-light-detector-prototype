import cv2
import detect_traffic_light as dtl

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if ret == False: break
    if cv2.waitKey(1) & 0xFF == ord('q'): break

    signals = dtl.detect_from_buffer(frame)
    
    if len(signals) != 0:
        print(f"Detected {len(signals)} signals")

    for signal, (x1, y1, x2, y2) in signals:
        color = (0, 0, 255) if signal == "red" else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.imshow('Webcam Live', frame)

cap.release()
cv2.destroyAllWindows()