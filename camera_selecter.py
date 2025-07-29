import subprocess
import re

def get_camera_index_by_name(target_name: str) -> int | None:
    """
    指定されたカメラ名に対応する /dev/videoX の X（整数）を返す。
    見つからなければ None を返す。
    """
    try:
        output = subprocess.check_output(['v4l2-ctl', '--list-devices'], text=True)
    except FileNotFoundError:
        print("v4l2-ctl が見つかりません。sudo apt install v4l-utils でインストールしてください。")
        return None

    lines = output.strip().splitlines()
    devices = []
    current_name = None
    current_video_indices = []

    for line in lines:
        if not line.startswith('\t'):
            if current_name and current_video_indices:
                devices.append((current_name, current_video_indices))
            current_name = line.strip().split(':')[0]
            current_video_indices = []
        else:
            line = line.strip()
            match = re.match(r'/dev/video(\d+)', line)
            if match:
                current_video_indices.append(int(match.group(1)))

    # 最後のカメラも追加
    if current_name and current_video_indices:
        devices.append((current_name, current_video_indices))

    # 対象名が含まれていれば、その最初の /dev/videoX の X を返す
    for name, indices in devices:
        if target_name in name:
            return indices[0]

    return None
