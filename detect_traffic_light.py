import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import math

plt.rcParams["font.family"] = "IPAexGothic"
IMAGE_RESIZE = (200, 200)

# HSV threshold definitions
HSV_THRESHOLDS_A = {
    "A_red": [((0, 170, 220), (1, 255, 255)), ((170, 170, 220), (179, 255, 255))],
    "A_blue": [((80, 100, 190), (90, 255, 255))],
}

HSV_THRESHOLDS_B = {
    "B_red_on":     [((0, 220, 200), (1, 255, 255)), ((170, 220, 200), (179, 255, 255))],
    "B_red_off":    [((0, 50, 50), (1, 240, 150)), ((145, 50, 50), (179, 240, 150))],
    "B_blue_on":    [((80, 150, 90), (100, 255, 180))],
    "B_blue_off":   [((95, 110, 60), (105, 220, 140))],
    "B_yellow_on":  [((15, 10, 170), (35, 135, 255))],
    "B_yellow_off": [((90, 30, 100), (179, 135, 150))],
}

ALL_THRESHOLDS = {**HSV_THRESHOLDS_A, **HSV_THRESHOLDS_B}

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
    combined_mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    hue_list, sat_list, val_list = [], [], []

    for lower, upper in ranges:
        mask = cv2.inRange(hsv_img, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
        lh, ls, lv = lower
        uh, us, uv = upper
        hue_list.append((lh + uh) // 2)
        sat_list.append((ls + us) // 2)
        val_list.append((lv + uv) // 2)

    avg_h = circular_mean_hue(hue_list)
    avg_s = int(np.mean(sat_list))
    avg_v = int(np.mean(val_list))
    avg_color_hsv = (avg_h, avg_s, avg_v)
    color_rgb = hsv_to_rgb(avg_color_hsv)
    return combined_mask, color_rgb

def process_and_plot_images(input_dir="traffic-light"):
    if not os.path.exists(input_dir):
        print(f"入力ディレクトリが見つかりません: {input_dir}")
        return

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not image_files:
        print("画像が見つかりません。")
        return

    num_images = len(image_files)
    cols = 9
    rows = num_images

    fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axs = axs.reshape(rows, cols)

    labels = [
        "元画像", "Aの赤", "Aの青",
        "B点灯赤", "B消灯赤",
        "B点灯青", "B消灯青",
        "B点灯黄", "B消灯黄"
    ]

    for row, filename in enumerate(image_files):
        path = os.path.join(input_dir, filename)
        img = cv2.imread(path)
        if img is None:
            continue

        img_resized = cv2.resize(img, IMAGE_RESIZE)
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

        axs[row, 0].imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
        axs[row, 0].set_title(f"{filename}", fontsize=9)
        axs[row, 0].axis("off")

        for col, key in enumerate(list(ALL_THRESHOLDS.keys()), start=1):
            mask, color_rgb = generate_color_mask(hsv, ALL_THRESHOLDS[key])
            rgb_image = np.zeros((*mask.shape, 3), dtype=np.float32)
            for i in range(3):
                rgb_image[:, :, i] = mask / 255.0 * color_rgb[i]
            axs[row, col].imshow(rgb_image)
            axs[row, col].set_title(labels[col], fontsize=9)
            axs[row, col].axis("off")

    plt.tight_layout()
    plt.savefig("signal_all_masks_colored_fixed.png", dpi=150)
    print("保存しました → signal_all_masks_colored_fixed.png")
    plt.show()

if __name__ == "__main__":
    process_and_plot_images()
