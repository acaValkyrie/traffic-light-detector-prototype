import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import math

plt.rcParams["font.family"] = "IPAexGothic"
IMAGE_RESIZE = (640, 480)

# HSV thresholds
HSV_THRESHOLDS = {
    "red_on":     ((165, 69, 61), (180, 153, 103)),
    "red_off":    ((114, 26, 43), (171, 93, 62)),
    "blue_on":    ((73, 51, 48), (106, 104, 95)),
    "blue_off":   ((74, 13, 34), (123, 93, 73))
}

LABELS = [
    "元画像＋ぼかし",
    "赤ON", "赤OFF",
    "青ON", "青OFF",
    "結果"
]

def circular_mean_hue(hue_values):
    radians = [math.radians(h * 2) for h in hue_values]
    sin_sum = sum(math.sin(r) for r in radians)
    cos_sum = sum(math.cos(r) for r in radians)
    avg_angle = math.atan2(sin_sum, cos_sum)
    if avg_angle < 0:
        avg_angle += 2 * math.pi
    avg_h = round(math.degrees(avg_angle) / 2)
    return avg_h

def hsv_to_rgb(hsv):
    hsv = np.uint8([[hsv]])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
    return rgb / 255.0

def generate_color_mask(hsv_img, ranges):
    # 指定されたHSV範囲でマスクを生成
    lower, upper = ranges
    mask = cv2.inRange(hsv_img, lower, upper)

    # 指定されたhsvの範囲の平均色を求めてマスクの色としたい
    lh, ls, lv = lower
    uh, us, uv = upper
    avg_h = (lh + uh) // 2
    avg_s = (ls + us) // 2
    avg_v = (lv + uv) // 2
    avg_color_hsv = (avg_h, avg_s, avg_v)
    # rgb色空間で返す
    color_rgb = hsv_to_rgb(avg_color_hsv)

    return mask, color_rgb

def draw_contours_on_mask(mask, color_rgb, min_area_ratio=0.003, max_area_ratio=0.5):
    h, w = mask.shape
    min_area = h * w * min_area_ratio
    max_area = h * w * max_area_ratio
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rgb_image = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(3):
        rgb_image[:, :, i] = mask / 255.0 * color_rgb[i]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area and area <= max_area:
            x, y, ww, hh = cv2.boundingRect(cnt)
            cv2.rectangle(rgb_image, (x, y), (x+ww, y+hh), (1, 1, 1)-color_rgb, 1)

    return rgb_image

def process_mask(mask, min_area=500, max_area=5000, iterations=3):
    # カーネルの定義（膨張・縮小処理用）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 膨張処理
    mask_dilated = cv2.dilate(mask, kernel, iterations=iterations)

    # 縮小処理
    mask_eroded = cv2.erode(mask_dilated, kernel, iterations=iterations)

    # 輪郭を検出
    contours, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 新しいマスクを作成
    filtered_mask = np.zeros_like(mask)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area and area <= max_area:
            # 面積が条件を満たす場合、輪郭を描画
            cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    return filtered_mask

def detect_traffic_signal(B_red_on, B_blue_off, B_red_off, B_blue_on):
    # 各マスクの輪郭を取得
    contours_red_on, _ = cv2.findContours(B_red_on, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue_off, _ = cv2.findContours(B_blue_off, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_red_off, _ = cv2.findContours(B_red_off, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue_on, _ = cv2.findContours(B_blue_on, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 信号の状態を格納するリスト
    signals = []
    horizontal_offset_threshold = 0.8  # 中点の左右方向のズレの許容範囲
    max_vertical_distance = 2  # 中点の上下方向のズレの許容範囲

    # 赤信号の判定
    for cnt_red in contours_red_on:
        x_red, y_red, w_red, h_red = cv2.boundingRect(cnt_red)
        center_red_x = x_red + w_red // 2
        center_red_y = y_red + h_red // 2

        for cnt_blue in contours_blue_off:
            x_blue, y_blue, w_blue, h_blue = cv2.boundingRect(cnt_blue)
            center_blue_x = x_blue + w_blue // 2
            center_blue_y = y_blue + h_blue // 2

            # 中点の左右方向のズレが一定範囲内か判定
            horizontal_offset = abs(center_red_x - center_blue_x)
            vertical_distance = center_blue_y - center_red_y

            if horizontal_offset <= max(w_red, w_blue) * horizontal_offset_threshold and vertical_distance <= max(h_red, h_blue) * max_vertical_distance:
                # 両方の輪郭を含む領域を囲む
                x1 = min(x_red, x_blue)
                y1 = min(y_red, y_blue)
                x2 = max(x_red + w_red, x_blue + w_blue)
                y2 = max(y_red + h_red, y_blue + h_blue)
                signals.append(("red", (x1, y1, x2, y2)))

    # 青信号の判定
    for cnt_red in contours_red_off:
        x_red, y_red, w_red, h_red = cv2.boundingRect(cnt_red)
        center_red_x = x_red + w_red // 2
        center_red_y = y_red + h_red // 2

        for cnt_blue in contours_blue_on:
            x_blue, y_blue, w_blue, h_blue = cv2.boundingRect(cnt_blue)
            center_blue_x = x_blue + w_blue // 2
            center_blue_y = y_blue + h_blue // 2

            # 中点の左右方向のズレが一定範囲内か判定
            horizontal_offset = abs(center_red_x - center_blue_x)
            vertical_distance = center_blue_y - center_red_y

            if horizontal_offset <= max(w_red, w_blue) * horizontal_offset_threshold and vertical_distance <= max(h_red, h_blue) * max_vertical_distance:
                # 両方の輪郭を含む領域を囲む
                x1 = min(x_red, x_blue)
                y1 = min(y_red, y_blue)
                x2 = max(x_red + w_red, x_blue + w_blue)
                y2 = max(y_red + h_red, y_blue + h_blue)
                signals.append(("blue", (x1, y1, x2, y2)))

    return signals

def process_and_plot_images(input_dir="traffic-light"):
    if not os.path.exists(input_dir):
        print(f"入力ディレクトリが存在しません: {input_dir}")
        return

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    num_images = len(image_files)
    cols = 6  # 元画像 + 4つのマスク + 信号機を囲った画像
    rows = num_images

    fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axs = axs.reshape(rows, cols)

    signals_detected = []  # 信号検出結果を格納するリスト

    for row, filename in enumerate(image_files):
        path = os.path.join(input_dir, filename)
        img = cv2.imread(path)
        if img is None:
            continue

        img_resized = cv2.resize(img, IMAGE_RESIZE)
        blur_size = 3
        img_blur = cv2.blur(img_resized, (blur_size, blur_size))
        hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

        axs[row, 0].imshow(cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB))
        axs[row, 0].set_title(f"{filename}", fontsize=9)
        axs[row, 0].axis("off")

        masks = []
        for col, key in enumerate(list(HSV_THRESHOLDS.keys()), start=1):
            mask, color_rgb = generate_color_mask(hsv, HSV_THRESHOLDS[key])
            mask = process_mask(mask, min_area=300, max_area=5000, iterations=7)
            rgb_image = draw_contours_on_mask(mask, color_rgb)
            axs[row, col].imshow(rgb_image)
            axs[row, col].set_title(LABELS[col], fontsize=9)
            axs[row, col].axis("off")
            masks.append(mask)

        # 信号を検出
        if len(masks) >= 4:  # 必要なマスクが揃っている場合
            B_red_on, B_red_off, B_blue_on, B_blue_off = masks
            detected_signals = detect_traffic_signal(B_red_on, B_blue_off, B_red_off, B_blue_on)
            signals_detected.append((filename, detected_signals))

            # 信号を囲む画像を作成
            img_with_signals = img_resized.copy()
            for signal, (x1, y1, x2, y2) in detected_signals:
                color = (0, 0, 255) if signal == "red" else (255, 0, 0)  # 赤信号は赤枠、青信号は青枠
                cv2.rectangle(img_with_signals, (x1, y1), (x2, y2), color, 2)

            # 信号を囲んだ画像を表示
            axs[row, 5].imshow(cv2.cvtColor(img_with_signals, cv2.COLOR_BGR2RGB))
            axs[row, 5].set_title("Detected Signals", fontsize=9)
            axs[row, 5].axis("off")

    # 結果を保存
    plt.tight_layout()
    output_dir = "output"
    output_path = os.path.join(output_dir, "traffic_signal_detection_results.png")
    plt.savefig(output_path)
    print(f"Results saved to {output_path}")

def detect_from_buffer(img: np.ndarray) -> list:
    img_resized = cv2.resize(img, IMAGE_RESIZE)
    blur_size = 3
    img_blur = cv2.blur(img_resized, (blur_size, blur_size))
    hsv_buffer = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

    masks = []
    for col, key in enumerate(list(HSV_THRESHOLDS.keys()), start=1):
        mask, color_rgb = generate_color_mask(hsv_buffer, HSV_THRESHOLDS[key])
        
        mask = process_mask(mask, min_area=300, max_area=5000, iterations=7)
        masks.append(mask)

    if len(masks) < 4: return
    plt.close()
    fig, axes = plt.subplots(1, 1 + len(masks))
    axes[0].imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    for ax, mask, label in zip(axes[1:], masks, LABELS[1:]):
        ax.imshow(mask, cmap='gray')
        ax.set_title(label)
    plt.tight_layout()
    plt.show()

    # 必要なマスクが揃っている場合
    B_red_on, B_red_off, B_blue_on, B_blue_off = masks
    detected_signals = detect_traffic_signal(B_red_on, B_blue_off, B_red_off, B_blue_on)

    # IMAGE_RESIZEに合わせて座標を調整
    h, w = img.shape[:2]
    scale_x = w / IMAGE_RESIZE[0]
    scale_y = h / IMAGE_RESIZE[1]
    detected_signals = [
        (signal, (int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)))
        for signal, (x1, y1, x2, y2) in detected_signals
    ]

    # return detected_signals
    return detected_signals, masks
    
def main():
    input_dir="traffic-light"
    if not os.path.exists(input_dir):
        print(f"入力ディレクトリが存在しません: {input_dir}")

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    num_images = len(image_files)
    cols = 2  # 元画像 + 信号機を囲った画像
    rows = num_images

    fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axs = axs.reshape(rows, cols)

    for row_i, filename in enumerate(image_files):
        path = os.path.join(input_dir, filename)
        img = cv2.imread(path)
        if img is None:
            continue

        signals, masks = detect_from_buffer(img)

        axs[row_i, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[row_i, 0].set_title(f"{filename}", fontsize=9)
        axs[row_i, 0].axis("off")
        img_with_signals = img.copy()
        if signals and len(signals) > 0:
            for signal, (x1, y1, x2, y2) in signals:
                color = (0, 0, 255) if signal == "red" else (255, 0, 0)
                cv2.rectangle(img_with_signals, (x1, y1), (x2, y2), color, 2)
        axs[row_i, 1].imshow(cv2.cvtColor(img_with_signals, cv2.COLOR_BGR2RGB))
        axs[row_i, 1].set_title("Detected Signals", fontsize=9)
        axs[row_i, 1].axis("off")

    plt.tight_layout()
    output_dir = "output"
    output_path = os.path.join(output_dir, "traffic_signal_detection_results.png")
    plt.savefig(output_path)

if __name__ == "__main__":
    main()