import cv2
import numpy as np
import detect_traffic_light as dtl

dtl.HSV_THRESHOLDS = {
    "B_red_on":     [((0, 255, 255), (0, 255, 255)), ((172, 180, 220), (179, 255, 255))],
    "B_red_off":    [((0, 255, 255), (0, 255, 255)), ((100, 50, 90), (179, 80, 150))],
    "B_blue_on":    [((80, 90, 130), (140, 250, 250))],
    "B_blue_off":   [((95, 100, 60), (105, 210, 140))]
}

def colorize_mask(mask, color):
    """
    グレースケールのマスクに色を付ける（color: (B, G, R)）
    0の部分は黒（透明）、1の部分は指定色
    """
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i in range(3):
        color_mask[:, :, i] = (mask / 255 * color[i]).astype(np.uint8)
    return color_mask

from camera_selecter import get_camera_index_by_name
cap = cv2.VideoCapture(get_camera_index_by_name("HD USB Camera"))

cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # 1で露光設定を手動にセット
cap.set(cv2.CAP_PROP_EXPOSURE, 0)

cap.set(cv2.CAP_PROP_AUTO_WB, 0) # 0でホワイトバランスを手動にセット
cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 3400)

import datetime
while cap.isOpened():
    ret, frame = cap.read()
    if ret == False: break
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    if cv2.waitKey(1) & 0xFF == ord('c'):
        # 現在時刻をつけてフレームを保存
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'frame_{timestamp}.jpg'
        cv2.imwrite(f'cap/{filename}', frame)
        print(f"Frame captured and saved as {filename}")

    signals, masks = dtl.detect_from_buffer(frame)
    mask_colors = [
        (0, 0, 255), (0, 0, 100),
        (255, 0, 0), (100, 0, 0) 
        ]
    masks = [colorize_mask(mask, color) for mask, color in zip(masks, mask_colors)]
    masks = np.vstack(masks)
    masks = cv2.resize(masks, ( masks.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    frame = np.hstack((frame, masks))

    if len(signals) != 0:
        print(f"Detected {len(signals)} signals")

    for signal, (x1, y1, x2, y2) in signals:
        color = (0, 0, 255) if signal == "red" else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.imshow('Webcam Live', frame)

cap.release()
cv2.destroyAllWindows()