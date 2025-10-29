import cv2
import numpy as np
import glob

# === 設定 ===
IMAGE_DIR = "./traffic-light/*.jpg"  # 読み込み対象
images = [cv2.imread(f) for f in sorted(glob.glob(IMAGE_DIR))]
if not images:
    raise FileNotFoundError("画像が見つかりません。")

# HSV閾値の初期値
h_min, s_min, v_min = 0, 0, 0
h_max, s_max, v_max = 179, 255, 255

def nothing(x):
    pass

cv2.namedWindow("Mask Control")
cv2.createTrackbar("H min", "Mask Control", h_min, 179, nothing)
cv2.createTrackbar("S min", "Mask Control", s_min, 255, nothing)
cv2.createTrackbar("V min", "Mask Control", v_min, 255, nothing)
cv2.createTrackbar("H max", "Mask Control", h_max, 179, nothing)
cv2.createTrackbar("S max", "Mask Control", s_max, 255, nothing)
cv2.createTrackbar("V max", "Mask Control", v_max, 255, nothing)

while True:
    # スライダー値取得
    h_min = cv2.getTrackbarPos("H min", "Mask Control")
    s_min = cv2.getTrackbarPos("S min", "Mask Control")
    v_min = cv2.getTrackbarPos("V min", "Mask Control")
    h_max = cv2.getTrackbarPos("H max", "Mask Control")
    s_max = cv2.getTrackbarPos("S max", "Mask Control")
    v_max = cv2.getTrackbarPos("V max", "Mask Control")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    processed = []
    for img in images:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        concat = np.hstack((img, mask_bgr))
        processed.append(concat)

    combined = np.vstack(processed)

    # でかすぎるのでリサイズ
    scale = 600 / combined.shape[1]
    combined = cv2.resize(combined, (0, 0), fx=scale, fy=scale)

    cv2.imshow("Images + Masks", combined)

    if cv2.waitKey(30) & 0xFF == 27:  # ESCキーで終了
        break

cv2.destroyAllWindows()
