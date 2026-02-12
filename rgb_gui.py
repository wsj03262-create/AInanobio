import sys
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
from picamera2 import Picamera2

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout


# ================== 고정 설정(필요 시 너만 수정) ==================
WIDTH, HEIGHT = 1280, 720              # 캡처 해상도(고정)
PREVIEW_MAX_W = 960                    # 프리뷰 표시 폭(고정, 낮추면 더 부드러움)
LOG_INTERVAL_SEC = 1.0                 # 데이터 수집 주기(초)
SAVE_IMAGE = True                      # True면 이미지도 저장
IMAGE_EXT = "jpg"
JPG_QUALITY = 90
DATA_ROOT = Path("data")               # 저장 루트

# 5포인트(고정) - 너가 좌표만 바꿔서 배포하면 됨
POINTS = [
    ("p1", 300, 150),
    ("p2", 980, 150),
    ("p3", 300, 560),
    ("p4", 980, 560),
    ("pc", 640, 360),
]
# ================================================================


def session_stamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def now_iso():
    return datetime.now().isoformat(timespec="milliseconds")


def now_ms():
    return int(time.time() * 1000)


def resize_for_preview(frame_bgr, max_w):
    h, w = frame_bgr.shape[:2]
    if w <= max_w:
        return frame_bgr
    scale = max_w / w
    return cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def draw_points(frame_bgr, points):
    out = frame_bgr.copy()
    for pid, x, y in points:
        cv2.circle(out, (x, y), 5, (0, 255, 0), -1)
        cv2.circle(out, (x, y), 10, (0, 255, 0), 2)
        cv2.putText(out, pid, (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def safe_rgb(frame_bgr, x, y):
    h, w = frame_bgr.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return None
    b, g, r = frame_bgr[y, x]
    return int(r), int(g), int(b)


class RGBApplianceGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI NanoBio RGB Sensor")

        # --- UI ---
        self.preview_label = QLabel("Camera preview…")
        self.preview_label.setAlignment(Qt.AlignCenter)

        # RGB 표시 라벨(5개)
        self.rgb_labels = {}
        rgb_box = QVBoxLayout()
        for pid, _, _ in POINTS:
            lab = QLabel(f"{pid}: R -, G -, B -")
            lab.setStyleSheet("font-size: 16px;")
            self.rgb_labels[pid] = lab
            rgb_box.addWidget(lab)

        self.status_label = QLabel("상태: 대기 (카메라 동작 중)")
        self.status_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.btn_start = QPushButton("실험 시작")
        self.btn_stop = QPushButton("실험 종료")
        self.btn_stop.setEnabled(False)

        self.btn_start.setStyleSheet("font-size: 18px; padding: 10px;")
        self.btn_stop.setStyleSheet("font-size: 18px; padding: 10px;")

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)

        layout = QVBoxLayout()
        layout.addWidget(self.preview_label, stretch=1)
        layout.addLayout(rgb_box)
        layout.addWidget(self.status_label)
        layout.addLayout(btn_row)
        self.setLayout(layout)

        self.btn_start.clicked.connect(self.start_experiment)
        self.btn_stop.clicked.connect(self.stop_experiment)

        # --- Camera ---
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()

        # --- Logging state ---
        self.running = False
        self.session_dir = None
        self.images_dir = None
        self.csv_path = None
        self.csv_file = None
        self.csv_writer = None
        self.next_log_time = time.time()

        # --- Preview timer (부드럽게) ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(30)  # ~33fps 목표(실제는 환경 따라)

    def start_experiment(self):
        if self.running:
            return

        sess = session_stamp()
        self.session_dir = DATA_ROOT / sess
        self.images_dir = self.session_dir / "images"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.session_dir / f"rgb_points_{sess}.csv"
        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["timestamp_iso", "unix_ms", "image_path", "point_id", "x", "y", "R", "G", "B"])

        self.running = True
        self.next_log_time = time.time()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_label.setText(f"상태: 수집 중…  (저장: {self.session_dir})")

    def stop_experiment(self):
        if not self.running:
            return

        self.running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_label.setText("상태: 대기 (카메라 동작 중)")

        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

    def tick(self):
        # 프레임 가져오기
        frame_rgb = self.picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # 실시간 RGB 표시 업데이트
        for pid, x, y in POINTS:
            rgb = safe_rgb(frame_bgr, x, y)
            if rgb is None:
                self.rgb_labels[pid].setText(f"{pid}: R -, G -, B - (out)")
            else:
                r, g, b = rgb
                self.rgb_labels[pid].setText(f"{pid}: R {r:3d}, G {g:3d}, B {b:3d}")

        # 프리뷰 오버레이 + 표시
        overlay = draw_points(frame_bgr, POINTS)
        disp = resize_for_preview(overlay, PREVIEW_MAX_W)
        disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w = disp_rgb.shape[:2]
        qimg = QImage(disp_rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self.preview_label.setPixmap(QPixmap.fromImage(qimg))

        # 데이터 수집(주기마다)
        if self.running and time.time() >= self.next_log_time and self.csv_writer is not None:
            t_iso = now_iso()
            t_ms = now_ms()

            img_path_str = ""
            if SAVE_IMAGE:
                self.images_dir.mkdir(parents=True, exist_ok=True)
                img_path = self.images_dir / f"{t_ms}.{IMAGE_EXT}"
                if IMAGE_EXT.lower() in ["jpg", "jpeg"]:
                    cv2.imwrite(str(img_path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY])
                else:
                    cv2.imwrite(str(img_path), frame_bgr)
                img_path_str = str(img_path)

            for pid, x, y in POINTS:
                rgb = safe_rgb(frame_bgr, x, y)
                if rgb is None:
                    self.csv_writer.writerow([t_iso, t_ms, img_path_str, pid, x, y, "", "", ""])
                else:
                    r, g, b = rgb
                    self.csv_writer.writerow([t_iso, t_ms, img_path_str, pid, x, y, r, g, b])

            self.csv_file.flush()
            self.next_log_time = time.time() + LOG_INTERVAL_SEC

    def closeEvent(self, event):
        # 종료 처리
        try:
            self.running = False
            if self.csv_file:
                self.csv_file.close()
            self.picam2.stop()
        except Exception:
            pass
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = RGBApplianceGUI()
    w.resize(1000, 900)
    w.show()
    sys.exit(app.exec_())