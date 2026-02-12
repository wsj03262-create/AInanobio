import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
from picamera2 import Picamera2

# ================== 사용자 설정 ==================
# ✅ 해상도 낮추기 (추천: 1280x720부터)
WIDTH, HEIGHT = 1280, 720

# ✅ 저장 주기만 따로 (예: 1초마다 저장)
LOG_INTERVAL_SEC = 1.0

# 프리뷰
SHOW_PREVIEW = True
PREVIEW_MAX_W = 960   # 더 가볍게: 640

# 5포인트
POINTS = [
    ("p1", 300, 150),
    ("p2", 980, 150),
    ("p3", 300, 560),
    ("p4", 980, 560),
    ("pc", 640, 360),
]

# 이미지 저장
SAVE_IMAGE = True
IMAGE_EXT = "jpg"
JPG_QUALITY = 90
SAVE_EVERY_N = 1

BASE_DIR = Path("data")


# ================== 유틸 ==================
def ts_now_iso():
    return datetime.now().isoformat(timespec="milliseconds")

def ts_now_ms():
    return int(time.time() * 1000)

def session_stamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def safe_get_rgb_from_bgr(frame_bgr, x, y):
    h, w = frame_bgr.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return None
    b, g, r = frame_bgr[y, x]
    return int(r), int(g), int(b)

def draw_points_overlay(frame_bgr, points):
    out = frame_bgr.copy()
    for pid, x, y in points:
        cv2.circle(out, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(out, pid, (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return out

def resize_for_preview(frame_bgr, max_w):
    h, w = frame_bgr.shape[:2]
    if w <= max_w:
        return frame_bgr
    scale = max_w / w
    return cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def save_frame(image_dir: Path, frame_bgr, fname_stem: str):
    image_dir.mkdir(parents=True, exist_ok=True)
    out_path = image_dir / f"{fname_stem}.{IMAGE_EXT}"
    if IMAGE_EXT.lower() in ["jpg", "jpeg"]:
        cv2.imwrite(str(out_path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY])
    else:
        cv2.imwrite(str(out_path), frame_bgr)
    return str(out_path)


# ================== 메인 ==================
def main():
    sess = session_stamp()
    session_dir = BASE_DIR / sess
    images_dir = session_dir / "images"
    session_dir.mkdir(parents=True, exist_ok=True)

    csv_path = session_dir / f"rgb_points_{sess}.csv"

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["timestamp_iso", "unix_ms", "image_path", "point_id", "x", "y", "R", "G", "B"])

    sample_idx = 0
    next_log_time = time.time()  # ✅ 저장 타이머

    try:
        while True:
            # 프리뷰는 가능한 자주 갱신
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if SHOW_PREVIEW:
                preview = draw_points_overlay(frame_bgr, POINTS)
                preview = resize_for_preview(preview, PREVIEW_MAX_W)
                cv2.imshow("RGB Sensor Preview (press q to quit)", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # ✅ 저장은 LOG_INTERVAL_SEC마다만 수행
            now = time.time()
            if now >= next_log_time:
                sample_idx += 1
                t_iso = ts_now_iso()
                t_ms = ts_now_ms()
                fname_stem = f"{sess}_{t_ms}"

                img_path_str = ""
                if SAVE_IMAGE and (sample_idx % SAVE_EVERY_N == 0):
                    img_path_str = save_frame(images_dir, frame_bgr, fname_stem)

                for pid, x, y in POINTS:
                    rgb = safe_get_rgb_from_bgr(frame_bgr, x, y)
                    if rgb is None:
                        writer.writerow([t_iso, t_ms, img_path_str, pid, x, y, "", "", ""])
                    else:
                        r, g, b = rgb
                        writer.writerow([t_iso, t_ms, img_path_str, pid, x, y, r, g, b])

                f.flush()
                next_log_time = now + LOG_INTERVAL_SEC

    finally:
        f.close()
        picam2.stop()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()

    print(f"[DONE] CSV saved: {csv_path}")
    if SAVE_IMAGE:
        print(f"[DONE] Images dir: {images_dir}")


if __name__ == "__main__":
    main()