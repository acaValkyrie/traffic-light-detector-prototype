import cv2
import numpy as np
import detect_traffic_light as dtl

dtl.HSV_THRESHOLDS = {
    "B_red_on":     [((172, 70, 100), (179, 90, 130))],
    "B_red_off":    [((100, 70, 100), (179, 90, 130))],
    "B_blue_on":    [((80, 90, 130), (140, 250, 250))],
    "B_blue_off":   [((90, 60, 70), (100, 70, 80))]
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
# cap = cv2.VideoCapture(get_camera_index_by_name("C270 HD WEBCAM"))
camera_index = get_camera_index_by_name("Intel(R) RealSense(TM) Depth Ca", 4)
print(f"Using camera index: {camera_index}")
cap = cv2.VideoCapture(camera_index)

cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
cap.set(cv2.CAP_PROP_EXPOSURE, 40)

cap.set(cv2.CAP_PROP_AUTO_WB, 1)
# cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4600)

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

    res = dtl.detect_from_buffer(frame)
    if not res:
        # insufficient masks/signals — skip this frame
        continue
    signals, masks = res
    mask_colors = [
        (0, 0, 255), (0, 0, 100),
        (255, 0, 0), (100, 0, 0)
    ]

    # detect_from_buffer creates masks at IMAGE_RESIZE (e.g. 200x200).
    # Resize them to the incoming frame size so we can horizontally stack for display.
    h, w = frame.shape[:2]
    masks_resized = [cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) for mask in masks]

    colored_masks = [colorize_mask(mask, color) for mask, color in zip(masks_resized, mask_colors)]

    # combine the colored masks into a single overlay image
    all_mask = colored_masks[0].copy()
    for mask in colored_masks[1:]:
        all_mask = cv2.add(all_mask, mask)
    masks = all_mask

    # attach masks to the right of the frame for visualization
    frame = np.hstack((frame, masks))

    if len(signals) != 0:
        print(f"Detected {len(signals)} signals")

    for signal, (x1, y1, x2, y2) in signals:
        color = (0, 0, 255) if signal == "red" else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.imshow('Webcam Live', frame)

cap.release()
cv2.destroyAllWindows()