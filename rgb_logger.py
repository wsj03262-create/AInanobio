import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
from picamera2 import Picamera2

# ================== 사용자 설정 ==================
WIDTH, HEIGHT = 1920, 1080
INTERVAL_SEC = 1.0            # 몇 초마다 저장할지

# 5포인트 (X자 끝 4개 + 중심 1개) - 너가 좌표만 바꾸면 됨
POINTS = [
    ("p1", 600, 300),
    ("p2", 1320, 300),
    ("p3", 600, 780),
    ("p4", 1320, 780),
    ("pc", 960, 540),
]

# 프리뷰 설정
SHOW_PREVIEW = True
PREVIEW_MAX_W = 1280          # 프리뷰 창이 너무 크면 줄이기용 (성능/화면 크기)

# 이미지 저장 옵션
SAVE_IMAGE = True
IMAGE_EXT = "jpg"             # "jpg" 또는 "png"
JPG_QUALITY = 95
SAVE_EVERY_N = 1              # 1이면 매번 저장, 10이면 10번 중 1번 저장

# 저장 루트
BASE_DIR = Path("data")


# ================== 유틸 ==================
def ts_now_iso():
    return datetime.now().isoformat(timespec="milliseconds")

def ts_now_ms():
    return int(time.time() * 1000)

def session_stamp():
    # 프로그램 시작 시각 기준(세션 고정): 2026-02-12_14-03-55
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def safe_get_rgb_from_bgr(frame_bgr, x, y):
    """단일 픽셀 RGB (범위 밖이면 None). frame_bgr[y,x]는 BGR."""
    h, w = frame_bgr.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return None
    b, g, r = frame_bgr[y, x]
    return int(r), int(g), int(b)

def draw_points_overlay(frame_bgr, points):
    """프리뷰에 5포인트를 표시(점+라벨)."""
    out = frame_bgr.copy()
    for pid, x, y in points:
        cv2.circle(out, (x, y), 6, (0, 255, 0), -1)          # 초록 점
        cv2.circle(out, (x, y), 10, (0, 255, 0), 2)          # 링
        cv2.putText(out, pid, (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return out

def resize_for_preview(frame_bgr, max_w):
    h, w = frame_bgr.shape[:2]
    if w <= max_w:
        return frame_bgr
    scale = max_w / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

def save_frame(image_dir: Path, frame_bgr, fname_stem: str):
    image_dir.mkdir(parents=True, exist_ok=True)
    out_path = image_dir / f"{fname_stem}.{IMAGE_EXT}"

    if IMAGE_EXT.lower() in ["jpg", "jpeg"]:
        cv2.imwrite(str(out_path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY])
    elif IMAGE_EXT.lower() == "png":
        cv2.imwrite(str(out_path), frame_bgr)
    else:
        raise ValueError("IMAGE_EXT는 'jpg' 또는 'png'만 지원합니다.")

    return str(out_path)


# ================== 메인 ==================
def main():
    # ✅ (2)(3) 프로그램 시작 시각 기준으로 세션 폴더/CSV 생성
    sess = session_stamp()
    session_dir = BASE_DIR / sess
    images_dir = session_dir / "images"
    session_dir.mkdir(parents=True, exist_ok=True)

    csv_path = session_dir / f"rgb_points_{sess}.csv"

    # Picamera2 세팅
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    # CSV 생성(매 실행마다 새 파일)
    f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow([
        "timestamp_iso", "unix_ms",
        "image_path",
        "point_id", "x", "y", "R", "G", "B"
    ])

    sample_idx = 0

    try:
        while True:
            sample_idx += 1
            t_iso = ts_now_iso()
            t_ms = ts_now_ms()

            # 파일명: 세션시간 + 현재 ms (겹침 방지)
            fname_stem = f"{sess}_{t_ms}"

            # 프레임 캡처 (Picamera2는 RGB -> OpenCV용 BGR로 변환)
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # ✅ (3) 이미지도 세션 폴더 아래로 저장
            img_path_str = ""
            if SAVE_IMAGE and (sample_idx % SAVE_EVERY_N == 0):
                img_path_str = save_frame(images_dir, frame_bgr, fname_stem)

            # RGB 저장 (단일 픽셀)
            for pid, x, y in POINTS:
                rgb = safe_get_rgb_from_bgr(frame_bgr, x, y)
                if rgb is None:
                    writer.writerow([t_iso, t_ms, img_path_str, pid, x, y, "", "", ""])
                else:
                    r, g, b = rgb
                    writer.writerow([t_iso, t_ms, img_path_str, pid, x, y, r, g, b])

            f.flush()

            # ✅ (1) 프리뷰: 5포인트 표시해서 화면 띄우기
            if SHOW_PREVIEW:
                preview = draw_points_overlay(frame_bgr, POINTS)
                preview = resize_for_preview(preview, PREVIEW_MAX_W)
                cv2.imshow("RGB Sensor Preview (press q to quit)", preview)

                # q 누르면 종료
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time.sleep(INTERVAL_SEC)

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