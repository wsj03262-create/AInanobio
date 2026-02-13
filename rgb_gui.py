import sys
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
from picamera2 import Picamera2

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame
)

# ================== 고정 설정(필요 시 너만 수정) ==================
WIDTH, HEIGHT = 1280, 720
PREVIEW_MAX_W = 960
LOG_INTERVAL_SEC = 1.0

SAVE_IMAGE = True
IMAGE_EXT = "jpg"
JPG_QUALITY = 90
DATA_ROOT = Path("data")

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
        self.setStyleSheet(self._qss())

        # ===== 상단 타이틀 / 상태 =====
        self.title_label = QLabel("AI NanoBio RGB Sensor")
        self.title_label.setObjectName("TitleLabel")

        self.status_pill = QLabel("READY  •  Camera ON")
        self.status_pill.setObjectName("StatusPill")
        self.status_pill.setAlignment(Qt.AlignCenter)

        top_row = QHBoxLayout()
        top_row.addWidget(self.title_label, stretch=1)
        top_row.addWidget(self.status_pill)

        # ===== 프리뷰 =====
        self.preview_label = QLabel()
        self.preview_label.setObjectName("Preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setText("Camera preview…")

        # ===== RGB 테이블 =====
        self.table = QTableWidget(len(POINTS), 6)
        self.table.setObjectName("RGBTable")
        self.table.setHorizontalHeaderLabels(["Point", "X", "Y", "R", "G", "B"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # 테이블 초기값 채우기
        for row, (pid, x, y) in enumerate(POINTS):
            self._set_table_item(row, 0, pid, align=Qt.AlignCenter)
            self._set_table_item(row, 1, str(x), align=Qt.AlignCenter)
            self._set_table_item(row, 2, str(y), align=Qt.AlignCenter)
            self._set_table_item(row, 3, "-", align=Qt.AlignCenter)
            self._set_table_item(row, 4, "-", align=Qt.AlignCenter)
            self._set_table_item(row, 5, "-", align=Qt.AlignCenter)

        # ===== 버튼 =====
        self.btn_start = QPushButton("실험 시작")
        self.btn_start.setObjectName("StartBtn")
        self.btn_stop = QPushButton("실험 종료")
        self.btn_stop.setObjectName("StopBtn")
        self.btn_stop.setEnabled(False)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)

        # ===== 레이아웃 =====
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setObjectName("Separator")

        layout = QVBoxLayout()
        layout.addLayout(top_row)
        layout.addWidget(separator)
        layout.addWidget(self.preview_label, stretch=1)
        layout.addWidget(QLabel("Live RGB (5 Points)"), alignment=Qt.AlignLeft)
        layout.addWidget(self.table)
        layout.addLayout(btn_row)
        self.setLayout(layout)

        self.btn_start.clicked.connect(self.start_experiment)
        self.btn_stop.clicked.connect(self.stop_experiment)

        # ===== Camera =====
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()

        # ===== Logging state =====
        self.running = False
        self.session_dir = None
        self.images_dir = None
        self.csv_path = None
        self.csv_file = None
        self.csv_writer = None
        self.next_log_time = time.time()

        # ===== Timer =====
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(30)  # 프리뷰 부드럽게

    def _qss(self):
        # "멋있게" 보이도록: 카드 느낌, pill 상태, 버튼 컬러
        return """
        QWidget { background: #0f172a; color: #e2e8f0; }
        #TitleLabel {
            font-size: 22px; font-weight: 700; padding: 6px 2px;
        }
        #StatusPill {
            background: #1f2937;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 6px 12px;
            font-size: 13px;
            color: #cbd5e1;
        }
        #Preview {
            background: #0b1220;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 6px;
        }
        #Separator {
            color: #334155;
            border: 1px solid #334155;
        }
        QLabel { font-size: 14px; }

        #RGBTable {
            background: #0b1220;
            border: 1px solid #334155;
            border-radius: 12px;
            gridline-color: #334155;
            font-size: 14px;
        }
        QHeaderView::section {
            background: #111827;
            color: #e2e8f0;
            border: none;
            padding: 8px;
            font-weight: 600;
        }
        QTableWidget::item {
            padding: 10px;
        }

        QPushButton {
            border: none;
            border-radius: 12px;
            padding: 14px 16px;
            font-size: 18px;
            font-weight: 700;
        }
        QPushButton:disabled {
            background: #334155;
            color: #94a3b8;
        }
        #StartBtn {
            background: #ef4444;
            color: white;
        }
        #StartBtn:hover { background: #dc2626; }
        #StopBtn {
            background: #3b82f6;
            color: white;
        }
        #StopBtn:hover { background: #2563eb; }
        """

    def _set_table_item(self, row, col, text, align=Qt.AlignLeft):
        item = QTableWidgetItem(text)
        item.setTextAlignment(align)
        self.table.setItem(row, col, item)

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
        self.status_pill.setText(f"RECORDING  •  {sess}")

    def stop_experiment(self):
        if not self.running:
            return

        self.running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_pill.setText("READY  •  Camera ON")

        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

    def tick(self):
        frame_rgb = self.picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # 테이블 업데이트 (RGB + 좌표는 고정이지만 보기 좋게 계속 유지)
        for row, (pid, x, y) in enumerate(POINTS):
            rgb = safe_rgb(frame_bgr, x, y)
            if rgb is None:
                self.table.item(row, 3).setText("-")
                self.table.item(row, 4).setText("-")
                self.table.item(row, 5).setText("-")
            else:
                r, g, b = rgb
                self.table.item(row, 3).setText(str(r))
                self.table.item(row, 4).setText(str(g))
                self.table.item(row, 5).setText(str(b))

        # 프리뷰 표시
        overlay = draw_points(frame_bgr, POINTS)
        disp = resize_for_preview(overlay, PREVIEW_MAX_W)
        disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w = disp_rgb.shape[:2]
        qimg = QImage(disp_rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self.preview_label.setPixmap(QPixmap.fromImage(qimg))

        # 로깅
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
    w.resize(1100, 900)
    w.show()
    sys.exit(app.exec_())