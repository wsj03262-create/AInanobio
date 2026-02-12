import csv, time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2

# ===== 설정 =====
WIDTH, HEIGHT = 1920, 1080
INTERVAL_SEC = 1.0
OUTPUT_DIR = Path("data")
CSV_NAME = "rgb_points.csv"

SAVE_IMAGE = True
IMAGE_EXT = "jpg"
JPG_QUALITY = 95
SAVE_EVERY_N = 1

POINTS = [
    ("p1", 600, 300),
    ("p2", 1320, 300),
    ("p3", 600, 780),
    ("p4", 1320, 780),
    ("pc", 960, 540),
]

def now_iso():
    return datetime.now().isoformat(timespec="milliseconds")

def now_fname():
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")[:-3]

def unix_ms():
    return int(time.time() * 1000)

def safe_get_rgb(frame_bgr, x, y):
    h, w = frame_bgr.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return None
    b, g, r = frame_bgr[y, x]
    return int(r), int(g), int(b)

def save_frame(image_dir: Path, frame_bgr, fname_stem: str):
    image_dir.mkdir(parents=True, exist_ok=True)
    out_path = image_dir / f"{fname_stem}.{IMAGE_EXT}"
    if IMAGE_EXT.lower() in ["jpg", "jpeg"]:
        cv2.imwrite(str(out_path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY])
    else:
        cv2.imwrite(str(out_path), frame_bgr)
    return str(out_path)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / CSV_NAME
    image_dir = OUTPUT_DIR / "images"

    # Picamera2 설정 (RGB888로 받으면 OpenCV 변환이 단순)
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    write_header = not csv_path.exists()
    f = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["timestamp_iso", "unix_ms", "image_path", "point_id", "x", "y", "R", "G", "B"])

    sample_idx = 0

    try:
        while True:
            sample_idx += 1
            t_iso = now_iso()
            t_ms = unix_ms()
            fname_stem = f"{now_fname()}_{t_ms}"

            # Picamera2는 RGB로 줌 -> OpenCV는 BGR을 주로 쓰니까 변환
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            img_path_str = ""
            if SAVE_IMAGE and (sample_idx % SAVE_EVERY_N == 0):
                img_path_str = save_frame(image_dir, frame_bgr, fname_stem)

            for pid, x, y in POINTS:
                rgb = safe_get_rgb(frame_bgr, x, y)
                if rgb is None:
                    writer.writerow([t_iso, t_ms, img_path_str, pid, x, y, "", "", ""])
                else:
                    r, g, b = rgb
                    writer.writerow([t_iso, t_ms, img_path_str, pid, x, y, r, g, b])

            f.flush()
            time.sleep(INTERVAL_SEC)

    except KeyboardInterrupt:
        pass
    finally:
        f.close()
        picam2.stop()

if __name__ == "__main__":
    main()