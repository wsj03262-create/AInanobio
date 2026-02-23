import sys
import csv
import time
import shutil
from datetime import datetime
from pathlib import Path

import cv2
from picamera2 import Picamera2

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame, QSizePolicy
)

# ================== 고정 설정(필요 시 너만 수정) ==================
WIDTH, HEIGHT = 1280, 720

# ✅ 프리뷰 더 작게
PREVIEW_MAX_W = 640

LOG_INTERVAL_SEC = 300.0

SAVE_IMAGE = True
IMAGE_EXT = "jpg"
JPG_QUALITY = 90
DATA_ROOT = Path("data")

POINTS = [
    ("p1", 500, 213),
    ("p2", 740, 213),
    ("p3", 500, 427),
    ("p4", 740, 427),
    ("pc", 620, 320),
]

# ✅ 디스크 용량 업데이트 주기(너무 자주 하면 쓸데없이 부담)
DISK_UPDATE_SEC = 2.0
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

def fmt_hms(seconds: int) -> str:
    if seconds < 0:
        seconds = 0
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def fmt_bytes(n: int) -> str:
    # 보기 좋은 단위로 (GiB 기준)
    gib = n / (1024**3)
    return f"{gib:.2f} GB"

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

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setObjectName("Separator")

        # ===== 왼쪽: 프리뷰 =====
        self.preview_label = QLabel()
        self.preview_label.setObjectName("Preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setText("Camera preview…")
        self.preview_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.preview_label.setFixedWidth(PREVIEW_MAX_W + 40)  # 프리뷰가 왼쪽에서 과하게 커지지 않게 고정

        # ===== 오른쪽: 상태 패널 =====
        self.info_panel = QFrame()
        self.info_panel.setObjectName("InfoPanel")
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(14, 14, 14, 14)
        info_layout.setSpacing(10)

        self.info_title = QLabel("현재 상태")
        self.info_title.setObjectName("InfoTitle")

        self.lab_state = QLabel("실험 상태: 대기")
        self.lab_start = QLabel("실험 시작 시간: -")
        self.lab_elapsed = QLabel("실험 경과 시간: 00:00:00")
        self.lab_disk = QLabel("남은 용량: -")

        for w in [self.lab_state, self.lab_start, self.lab_elapsed, self.lab_disk]:
            w.setObjectName("InfoLine")

        info_layout.addWidget(self.info_title)
        info_layout.addWidget(self.lab_state)
        info_layout.addWidget(self.lab_start)
        info_layout.addWidget(self.lab_elapsed)
        info_layout.addWidget(self.lab_disk)
        self.info_panel.setLayout(info_layout)

        # ===== RGB 테이블 =====
        self.table = QTableWidget(len(POINTS), 6)
        self.table.setObjectName("RGBTable")
        self.table.setHorizontalHeaderLabels(["Point", "X", "Y", "R", "G", "B"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

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

        # ===== 메인 바디 레이아웃 (왼쪽 프리뷰 / 오른쪽 상태+테이블+버튼) =====
        right_col = QVBoxLayout()
        right_col.setSpacing(12)
        right_col.addWidget(self.info_panel)
        right_col.addWidget(QLabel("Live RGB (5 Points)"), alignment=Qt.AlignLeft)
        right_col.addWidget(self.table, stretch=1)
        right_col.addLayout(btn_row)

        body_row = QHBoxLayout()
        body_row.setSpacing(14)
        body_row.addWidget(self.preview_label, stretch=0)
        body_row.addLayout(right_col, stretch=1)

        # ===== 전체 레이아웃 =====
        layout = QVBoxLayout()
        layout.addLayout(top_row)
        layout.addWidget(separator)
        layout.addLayout(body_row, stretch=1)
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

        self.experiment_start_dt = None  # ✅ 시작 시간 표시용

        # 디스크 정보 갱신 타이머
        self._last_disk_check = 0.0
        self._update_disk_label(force=True)

        # ===== Timer =====
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(30)  # 지금 “아주 잘 됨”이라고 했으니 일단 유지

    def _qss(self):
        return """
        QWidget { background: #0f172a; color: #e2e8f0; }
        #TitleLabel { font-size: 22px; font-weight: 700; padding: 6px 2px; }

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
        #Separator { color: #334155; border: 1px solid #334155; }
        QLabel { font-size: 14px; }

        #InfoPanel{
            background: #0b1220;
            border: 1px solid #334155;
            border-radius: 12px;
        }
        #InfoTitle{
            font-size: 16px;
            font-weight: 700;
            margin-bottom: 6px;
        }
        #InfoLine{
            font-size: 14px;
            color: #cbd5e1;
        }

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
        QTableWidget::item { padding: 10px; }

        QPushButton {
            border: none;
            border-radius: 12px;
            padding: 14px 16px;
            font-size: 18px;
            font-weight: 700;
        }
        QPushButton:disabled { background: #334155; color: #94a3b8; }
        #StartBtn { background: #ef4444; color: white; }
        #StartBtn:hover { background: #dc2626; }
        #StopBtn { background: #3b82f6; color: white; }
        #StopBtn:hover { background: #2563eb; }
        """

    def _set_table_item(self, row, col, text, align=Qt.AlignLeft):
        item = QTableWidgetItem(text)
        item.setTextAlignment(align)
        self.table.setItem(row, col, item)

    def _update_disk_label(self, force=False):
        now_t = time.time()
        if (not force) and (now_t - self._last_disk_check < DISK_UPDATE_SEC):
            return
        self._last_disk_check = now_t

        # DATA_ROOT가 있는 파티션의 남은 용량 기준
        try:
            DATA_ROOT.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(str(DATA_ROOT.resolve()))
            self.lab_disk.setText(f"남은 용량: {fmt_bytes(usage.free)} (총 {fmt_bytes(usage.total)})")
        except Exception:
            self.lab_disk.setText("남은 용량: 확인 실패")

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

        # ✅ 상태 패널 값 갱신
        self.experiment_start_dt = datetime.now()
        self.lab_state.setText("실험 상태: 실험중")
        self.lab_start.setText(f"실험 시작 시간: {self.experiment_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        self.lab_elapsed.setText("실험 경과 시간: 00:00:00")

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

        # ✅ 상태 패널 갱신
        self.lab_state.setText("실험 상태: 대기")
        # 시작시간은 마지막 기록으로 남겨두고 싶으면 유지해도 됨
        # self.lab_start.setText("실험 시작 시간: -")
        # self.lab_elapsed.setText("실험 경과 시간: 00:00:00")

        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

        self._update_disk_label(force=True)

    def tick(self):
        frame_rgb = self.picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # ✅ 디스크 용량은 일정 주기만 갱신
        self._update_disk_label()

        # ✅ 경과 시간 표시(실험중일 때만)
        if self.running and self.experiment_start_dt is not None:
            elapsed = int((datetime.now() - self.experiment_start_dt).total_seconds())
            self.lab_elapsed.setText(f"실험 경과 시간: {fmt_hms(elapsed)}")

        # 테이블 업데이트
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

        # 프리뷰 표시 (왼쪽, 더 작게)
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
    w.showFullScreen()
    sys.exit(app.exec_())