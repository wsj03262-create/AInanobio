import cv2
import csv
import time
from datetime import datetime
from pathlib import Path

# ================== 사용자 설정 ==================
CAM_INDEX = 0
WIDTH = 1920
HEIGHT = 1080

INTERVAL_SEC = 1.0            # 몇 초마다 측정/저장할지
OUTPUT_DIR = Path("data")
CSV_NAME = "rgb_points.csv"

# 이미지 저장 옵션
SAVE_IMAGE = True             # True면 이미지 저장
IMAGE_EXT = "jpg"             # "jpg" 또는 "png"
JPG_QUALITY = 95              # jpg 품질 (0~100)
SAVE_EVERY_N = 1              # 1이면 매번 저장, 10이면 10번 중 1번 저장

# 5포인트 (너가 나중에 숫자만 바꾸면 됨)
POINTS = [
    ("p1", 600, 300),
    ("p2", 1320, 300),
    ("p3", 600, 780),
    ("p4", 1320, 780),
    ("pc", 960, 540),
]

# ================== 유틸 ==================
def now_iso():
    return datetime.now().isoformat(timespec="milliseconds")

def now_fname():
    # 파일명 안전하게: 2026-02-12T12-34-56-789
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")[:-3]

def unix_ms():
    return int(time.time() * 1000)

def safe_get_rgb(frame_bgr, x, y):
    """프레임 범위 밖이면 None. OpenCV는 BGR -> RGB로 바꿔 반환."""
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
    elif IMAGE_EXT.lower() == "png":
        cv2.imwrite(str(out_path), frame_bgr)  # png는 무손실(용량 큼)
    else:
        raise ValueError("IMAGE_EXT는 'jpg' 또는 'png'만 지원합니다.")

    return out_path

# ================== 메인 ==================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / CSV_NAME
    image_dir = OUTPUT_DIR / "images"

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다. CAM_INDEX를 확인하세요.")

    # 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    # CSV 헤더(파일이 없을 때만 작성)
    write_header = not csv_path.exists()
    f = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if write_header:
        writer.writerow([
            "timestamp_iso", "unix_ms",
            "image_path",
            "point_id", "x", "y", "R", "G", "B"
        ])

    sample_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽지 못했습니다. 재시도...")
                time.sleep(0.2)
                continue

            sample_idx += 1
            t_iso = now_iso()
            t_ms = unix_ms()
            fname_stem = f"{now_fname()}_{t_ms}"

            # (선택) 이미지 저장
            img_path_str = ""
            if SAVE_IMAGE and (sample_idx % SAVE_EVERY_N == 0):
                img_path = save_frame(image_dir, frame, fname_stem)
                img_path_str = str(img_path)

            # RGB 저장 (단일 픽셀)
            for pid, x, y in POINTS:
                rgb = safe_get_rgb(frame, x, y)
                if rgb is None:
                    writer.writerow([t_iso, t_ms, img_path_str, pid, x, y, "", "", ""])
                else:
                    r, g, b = rgb
                    writer.writerow([t_iso, t_ms, img_path_str, pid, x, y, r, g, b])

            f.flush()  # 장시간 실험 안전

            time.sleep(INTERVAL_SEC)

    except KeyboardInterrupt:
        print("종료합니다.")
    finally:
        f.close()
        cap.release()

if __name__ == "__main__":
    main()