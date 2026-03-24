import sys
import csv
import time
import queue
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QCloseEvent
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QMessageBox
)

# =========================
# 설정
# =========================
WIDTH, HEIGHT = 1280, 720
PREVIEW_W = 640

UI_INTERVAL_MS = 100          # 프리뷰/분석 주기
CSV_INTERVAL_SEC = 60         # CSV 저장 주기
IMAGE_INTERVAL_SEC = 300      # 이미지 저장 주기

SAVE_IMAGE = True
IMAGE_EXT = "jpg"
JPG_QUALITY = 90

DATA_ROOT = Path.home() / "rgb_sessions"

# =========================
# 이미지 저장 쓰레드
# =========================
class ImageSaver(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.q = queue.Queue(maxsize=4)   # 큐 과적 방지
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            try:
                item = self.q.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                break

            path, frame = item
            try:
                cv2.imwrite(
                    str(path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY]
                )
            except Exception as e:
                print("[ImageSaver] save error:", e)
            finally:
                self.q.task_done()

    def save(self, path, frame):
        if self.stop_event.is_set():
            return
        try:
            # frame.copy()로 원본 수명 문제 차단
            self.q.put_nowait((path, frame.copy()))
        except queue.Full:
            print("[ImageSaver] queue full -> image skipped")

    def shutdown(self):
        self.stop_event.set()
        try:
            self.q.put_nowait(None)
        except:
            pass


class RGBLoggerGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("RGB Logger")
        self.resize(1000, 700)

        self.running = False
        self.closing = False

        self.session_dir = None
        self.images_dir = None
        self.csv_path = None
        self.csv_file = None
        self.csv_writer = None

        self.last_csv_time = 0.0
        self.last_img_time = 0.0

        self.latest_frame = None

        self.image_saver = ImageSaver()
        self.image_saver.start()

        # UI
        self.preview_label = QLabel("Preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(PREVIEW_W, int(PREVIEW_W * HEIGHT / WIDTH))

        self.status_label = QLabel("READY")
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")

        self.start_btn.clicked.connect(self.start_experiment)
        self.stop_btn.clicked.connect(self.stop_experiment)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.preview_label)
        main_layout.addWidget(self.status_label)
        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)

        # 카메라
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (WIDTH, HEIGHT), "format": "RGB888"},
            buffer_count=4,
            queue=False
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(1.0)

        # UI 타이머
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(UI_INTERVAL_MS)

    def create_session(self):
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = DATA_ROOT / now
        self.images_dir = self.session_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.session_dir / "rgb_log.csv"
        self.csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["timestamp", "r", "g", "b"])

    def start_experiment(self):
        if self.running or self.closing:
            return

        self.create_session()
        self.running = True
        self.last_csv_time = 0.0
        self.last_img_time = 0.0
        self.status_label.setText("RECORDING")

    def stop_experiment(self):
        if not self.running:
            return
        self.running = False
        self.status_label.setText("READY")
        self.flush_and_close_csv()

    def flush_and_close_csv(self):
        try:
            if self.csv_file:
                self.csv_file.flush()
                self.csv_file.close()
        except Exception as e:
            print("[CSV] close error:", e)
        finally:
            self.csv_file = None
            self.csv_writer = None

    def extract_rgb(self, frame):
        """
        예시: 중앙 1픽셀 대신 작은 ROI 평균 사용
        """
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        roi = frame[max(0, cy-2):cy+3, max(0, cx-2):cx+3]
        mean_rgb = roi.mean(axis=(0, 1))
        r, g, b = int(mean_rgb[0]), int(mean_rgb[1]), int(mean_rgb[2])
        return r, g, b

    def update_frame(self):
        if self.closing:
            return

        try:
            frame = self.picam2.capture_array("main")
        except Exception as e:
            print("[Camera] capture error:", e)
            return

        if frame is None:
            return

        self.latest_frame = frame

        # 프리뷰 표시용 축소
        preview = cv2.resize(frame, (PREVIEW_W, int(PREVIEW_W * HEIGHT / WIDTH)))
        qimg = QImage(
            preview.data,
            preview.shape[1],
            preview.shape[0],
            preview.strides[0],
            QImage.Format_RGB888
        ).copy()  # 원본 배열 수명 분리
        self.preview_label.setPixmap(QPixmap.fromImage(qimg))

        if not self.running:
            return

        now = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 분석
        r, g, b = self.extract_rgb(frame)

        # CSV 저장
        if now - self.last_csv_time >= CSV_INTERVAL_SEC:
            try:
                self.csv_writer.writerow([timestamp, r, g, b])
                self.csv_file.flush()
            except Exception as e:
                print("[CSV] write error:", e)
            self.last_csv_time = now

        # 이미지 저장
        if SAVE_IMAGE and (now - self.last_img_time >= IMAGE_INTERVAL_SEC):
            img_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f".{IMAGE_EXT}"
            img_path = self.images_dir / img_name
            self.image_saver.save(img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self.last_img_time = now

    def keyPressEvent(self, event):
        # ESC 같은 키 커스텀 필요하면 여기서 처리
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent):
        if self.closing:
            event.accept()
            return

        reply = QMessageBox.question(
            self,
            "종료 확인",
            "실험 중이면 카메라와 저장 작업을 정리한 뒤 종료합니다.\n종료할까요?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            event.ignore()
            return

        self.closing = True
        self.status_label.setText("CLOSING...")
        self.setEnabled(False)

        # 1) 타이머 정지
        try:
            self.timer.stop()
        except Exception as e:
            print("[Close] timer stop error:", e)

        # 2) 실험 중지 및 CSV 종료
        self.running = False
        self.flush_and_close_csv()

        # 3) 이미지 저장 쓰레드 종료
        try:
            self.image_saver.shutdown()
            self.image_saver.join(timeout=3.0)
        except Exception as e:
            print("[Close] image saver shutdown error:", e)

        # 4) 카메라 정지/해제
        try:
            self.picam2.stop()
        except Exception as e:
            print("[Close] picam2 stop error:", e)

        try:
            self.picam2.close()
        except Exception as e:
            print("[Close] picam2 close error:", e)

        # 5) 참조 해제
        self.latest_frame = None

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = RGBLoggerGUI()
    w.show()
    sys.exit(app.exec_())