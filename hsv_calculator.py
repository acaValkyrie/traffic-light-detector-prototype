import cv2
import numpy as np
from tkinter import Tk, filedialog

drawing = False
ix, iy = -1, -1
rect = None

def rgb_to_hsv(rgb):
    rgb_array = np.uint8([[rgb]])
    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)[0][0]
    return hsv

def mouse_event(event, x, y, flags, param):
    global ix, iy, drawing, rect

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        rect = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            rect = (ix, iy, x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect = (ix, iy, x, y)
        process_rect(param['image'], rect)

def process_rect(image, rect):
    x1, y1, x2, y2 = rect
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        print("選択範囲が無効です")
        return

    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    mean_rgb = np.mean(rgb_roi, axis=(0, 1)).astype(np.uint8)
    hsv = rgb_to_hsv(mean_rgb)

    print(f"平均RGB: {mean_rgb} → 平均HSV: {hsv} (H×2 = {hsv[0]*2}°)")

def main():
    # ファイル選択ダイアログ
    root = Tk()
    root.withdraw()  # GUIウィンドウを表示しない
    file_path = filedialog.askopenfilename(
        title="画像ファイルを選択",
        filetypes=[("画像ファイル", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
    )
    if not file_path:
        print("ファイルが選択されませんでした")
        return

    img = cv2.imread(file_path)
    if img is None:
        print("画像の読み込みに失敗しました")
        return

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_event, param={'image': img})

    while True:
        display_img = img.copy()
        if rect:
            x1, y1, x2, y2 = rect
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('image', display_img)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
