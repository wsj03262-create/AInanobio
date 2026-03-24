import sys
import csv
import time
import shutil
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from collections import deque

import cv2
import numpy as np
from picamera2 import Picamera2

from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame,
    QComboBox, QMessageBox, QInputDialog, QLineEdit
)

# ================== 고정 설정 ==================
WIDTH, HEIGHT = 1280, 720
PREVIEW_MAX_W = 480

RGB_LOG_INTERVAL_SEC = 60.0
IMAGE_LOG_INTERVAL_SEC = 300.0

SAVE_IMAGE = True
IMAGE_EXT = "jpg"
JPG_QUALITY = 95
DATA_ROOT = Path("/home/pi/Ainanobio_data")

# 안정화용
PREVIEW_INTERVAL_MS = 150     # 기존 100 -> 150
INFO_INTERVAL_MS = 1000       # 기존 500 -> 1000
USB_INTERVAL_MS = 8000        # 기존 5000 -> 8000
DISK_UPDATE_SEC = 3.0
CSV_FLUSH_INTERVAL_SEC = 10.0
SHOW_GRID_POINTS_ON_PREVIEW = True
SHOW_POINT_LABELS = False
PLOT_MAX_POINTS = 240

DEV_PASSWORD = "2472"

# 기존 3x3 기준점
POINTS = [
    ("p1", 530, 230),
    ("p2", 650, 230),
    ("p3", 770, 230),

    ("p4", 530, 308),
    ("p5", 650, 308),
    ("p6", 770, 308),

    ("p7", 530, 386),
    ("p8", 650, 386),
    ("p9", 770, 386),
]

# ROI 내부 99개 샘플링 점
GRID_ROWS = 9
GRID_COLS = 11
# ===============================================


def session_stamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def now_ms():
    return int(time.time() * 1000)


def resize_for_preview(frame_bgr, max_w):
    h, w = frame_bgr.shape[:2]
    if w <= max_w:
        return frame_bgr
    scale = max_w / w
    return cv2.resize(
        frame_bgr,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA
    )


def get_roi_from_points(points):
    xs = [x for _, x, _ in points]
    ys = [y for _, _, y in points]
    left = min(xs)
    right = max(xs)
    top = min(ys)
    bottom = max(ys)
    return left, top, right, bottom


def generate_grid_points_from_roi(left, top, right, bottom, rows, cols):
    if rows < 1 or cols < 1:
        return []

    xs = np.linspace(left, right, cols)
    ys = np.linspace(top, bottom, rows)

    pts = []
    idx = 1
    for y in ys:
        for x in xs:
            pts.append((f"g{idx:02d}", int(round(x)), int(round(y))))
            idx += 1
    return pts


def draw_roi_and_grid(frame_bgr, roi, grid_points=None):
    out = frame_bgr.copy()
    left, top, right, bottom = roi

    cv2.rectangle(out, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(
        out, "ROI", (left, max(20, top - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
    )

    if grid_points:
        for pid, x, y in grid_points:
            cv2.circle(out, (x, y), 2, (0, 255, 255), -1)
            if SHOW_POINT_LABELS:
                cv2.putText(
                    out, pid, (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA
                )
    return out


def safe_rgb(frame_bgr, x, y):
    h, w = frame_bgr.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return None
    b, g, r = frame_bgr[y, x]
    return int(r), int(g), int(b)


def get_roi_mean_rgb(frame_bgr, roi):
    left, top, right, bottom = roi
    h, w = frame_bgr.shape[:2]

    left = max(0, min(left, w - 1))
    right = max(0, min(right, w - 1))
    top = max(0, min(top, h - 1))
    bottom = max(0, min(bottom, h - 1))

    if left > right or top > bottom:
        return None

    roi_img = frame_bgr[top:bottom + 1, left:right + 1]
    if roi_img.size == 0:
        return None

    mean_b, mean_g, mean_r = roi_img.mean(axis=(0, 1))
    return round(float(mean_r), 1), round(float(mean_g), 1), round(float(mean_b), 1)


def sample_grid_rgb(frame_bgr, grid_points):
    samples = []
    valid_rs = []
    valid_gs = []
    valid_bs = []

    for pid, x, y in grid_points:
        rgb = safe_rgb(frame_bgr, x, y)
        if rgb is None:
            samples.append((pid, x, y, "", "", ""))
        else:
            r, g, b = rgb
            samples.append((pid, x, y, r, g, b))
            valid_rs.append(r)
            valid_gs.append(g)
            valid_bs.append(b)

    if valid_rs:
        avg_r = round(sum(valid_rs) / len(valid_rs), 1)
        avg_g = round(sum(valid_gs) / len(valid_gs), 1)
        avg_b = round(sum(valid_bs) / len(valid_bs), 1)
    else:
        avg_r, avg_g, avg_b = "", "", ""

    return samples, (avg_r, avg_g, avg_b)


def fmt_hms(seconds: int) -> str:
    if seconds < 0:
        seconds = 0
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_bytes(n: int) -> str:
    gib = n / (1024**3)
    return f"{gib:.2f} GB"


# ===================== USB / Session helpers =====================

def find_usb_mounts():
    mounts = []
    try:
        with open("/proc/mounts", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                mnt = parts[1]

                if not (mnt.startswith("/media/") or mnt.startswith("/run/media/")):
                    continue

                if mnt in ("/media", "/media/pi", "/run/media", "/run/media/pi"):
                    continue

                if os.path.isdir(mnt):
                    mounts.append(mnt)
    except Exception:
        pass

    return sorted(list(dict.fromkeys(mounts)))


def is_session_dir_name(name: str) -> bool:
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", name))


def list_sessions(data_root: Path):
    out = []
    if not data_root.exists():
        return out

    for p in data_root.iterdir():
        if p.is_dir() and is_session_dir_name(p.name):
            out.append(p)

    return sorted(out, key=lambda x: x.name, reverse=True)


def recent_sessions(data_root: Path, limit=3):
    return list_sessions(data_root)[:limit]


def session_display_name(session_name: str):
    try:
        dt = datetime.strptime(session_name, "%Y-%m-%d_%H-%M-%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return session_name


def get_device_from_mount(mount_point: str):
    try:
        result = subprocess.run(
            ["findmnt", "-no", "SOURCE", "--target", mount_point],
            capture_output=True, text=True, check=True
        )
        dev = result.stdout.strip()
        return dev if dev else None
    except Exception:
        return None


def get_parent_block_device(partition_dev: str):
    try:
        result = subprocess.run(
            ["lsblk", "-no", "PKNAME", partition_dev],
            capture_output=True, text=True, check=True
        )
        parent = result.stdout.strip()
        if parent:
            return f"/dev/{parent}"
    except Exception:
        pass
    return None


def sync_filesystem():
    try:
        os.sync()
    except Exception:
        pass


class CopyWorker(QThread):
    log = pyqtSignal(str)
    done = pyqtSignal(bool, str)

    def __init__(self, src_dirs, dst_root):
        super().__init__()
        self.src_dirs = [Path(p) for p in src_dirs]
        self.dst_root = Path(dst_root)

    def run(self):
        try:
            self.dst_root.mkdir(parents=True, exist_ok=True)

            for src in self.src_dirs:
                dst = self.dst_root / src.name
                self.log.emit(f"복사 중: {src.name}")

                if dst.exists():
                    shutil.rmtree(dst)

                shutil.copytree(src, dst)

            sync_filesystem()
            self.done.emit(True, f"완료: {len(self.src_dirs)}개 세션 복사됨")
        except Exception as e:
            self.done.emit(False, f"실패: {e}")


class ImageSaveWorker(QThread):
    done = pyqtSignal(bool, str)

    def __init__(self, frame_bgr, img_path, image_ext, jpg_quality):
        super().__init__()
        self.frame_bgr = frame_bgr.copy()
        self.img_path = str(img_path)
        self.image_ext = image_ext
        self.jpg_quality = jpg_quality

    def run(self):
        try:
            if self.image_ext.lower() in ["jpg", "jpeg"]:
                ok = cv2.imwrite(
                    self.img_path,
                    self.frame_bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpg_quality]
                )
            else:
                ok = cv2.imwrite(self.img_path, self.frame_bgr)

            self.done.emit(ok, self.img_path)
        except Exception as e:
            self.done.emit(False, str(e))


class RGBPlotWidget(QFrame):
    def __init__(self, title="RGB Plot", parent=None):
        super().__init__(parent)
        self.setObjectName("PlotPanel")
        self.title = title

        self.x = deque(maxlen=PLOT_MAX_POINTS)
        self.r = deque(maxlen=PLOT_MAX_POINTS)
        self.g = deque(maxlen=PLOT_MAX_POINTS)
        self.b = deque(maxlen=PLOT_MAX_POINTS)

        self._bg = QColor("#0b1220")
        self._grid = QColor("#334155")
        self._text = QColor("#cbd5e1")

        self._pen_r = QPen(QColor(239, 68, 68), 2)
        self._pen_g = QPen(QColor(34, 197, 94), 2)
        self._pen_b = QPen(QColor(59, 130, 246), 2)

    def reset(self):
        self.x.clear()
        self.r.clear()
        self.g.clear()
        self.b.clear()
        self.update()

    def append(self, x_idx: int, rgb_tuple):
        rr, gg, bb = rgb_tuple
        self.x.append(x_idx)
        self.r.append(rr)
        self.g.append(gg)
        self.b.append(bb)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), self._bg)

        w = self.width()
        h = self.height()

        left = 46
        right = 14
        top = 36
        bottom = 28

        plot_w = max(1, w - left - right)
        plot_h = max(1, h - top - bottom)

        p.setPen(self._text)
        f = QFont()
        f.setPointSize(12)
        f.setBold(True)
        p.setFont(f)
        p.drawText(12, 22, self.title)

        p.setPen(QPen(self._grid, 1))
        for i in range(5):
            y = top + int(plot_h * i / 4)
            p.drawLine(left, y, left + plot_w, y)

        p.setPen(self._text)
        f2 = QFont()
        f2.setPointSize(9)
        p.setFont(f2)
        y_ticks = [255, 192, 128, 64, 0]
        for i, val in enumerate(y_ticks):
            y = top + int(plot_h * i / 4)
            p.drawText(8, y + 4, f"{val:3d}")

        if len(self.x) < 2:
            p.end()
            return

        n = len(self.x)

        def x_to_px(i):
            return left + int(plot_w * i / (n - 1))

        def y_to_px(v):
            v = max(0, min(255, float(v)))
            return top + int(plot_h * (1.0 - (v / 255.0)))

        def draw_line(vals, pen):
            p.setPen(pen)
            prev_x = x_to_px(0)
            prev_y = y_to_px(vals[0])
            for i in range(1, n):
                cx = x_to_px(i)
                cy = y_to_px(vals[i])
                p.drawLine(prev_x, prev_y, cx, cy)
                prev_x, prev_y = cx, cy

        draw_line(list(self.r), self._pen_r)
        draw_line(list(self.g), self._pen_g)
        draw_line(list(self.b), self._pen_b)

        legend_x = left + plot_w - 110
        legend_y = 18
        p.setPen(self._pen_r)
        p.drawText(legend_x, legend_y, "R")
        p.setPen(self._pen_g)
        p.drawText(legend_x + 22, legend_y, "G")
        p.setPen(self._pen_b)
        p.drawText(legend_x + 44, legend_y, "B")

        p.end()


class RGBApplianceGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI NanoBio RGB Sensor")
        self.setStyleSheet(self._qss())

        self.allow_close = False
        self.is_shutting_down = False

        self.roi = get_roi_from_points(POINTS)
        self.grid_points = generate_grid_points_from_roi(*self.roi, GRID_ROWS, GRID_COLS)

        self.latest_frame_bgr = None
        self.last_preview_qpixmap = None

        self.running = False
        self.session_dir = None
        self.images_dir = None
        self.csv_path = None
        self.csv_file = None
        self.csv_writer = None

        self.next_rgb_log_time = time.time()
        self.next_img_log_time = time.time()
        self.last_csv_flush_time = time.time()

        self.experiment_start_dt = None
        self.sample_count = 0
        self.image_count = 0
        self.data_log_count = 0

        self.usb_mounts = []
        self.usb_mount = None
        self.copy_worker = None
        self.image_save_worker = None

        self.usb_removed_mode = False
        self.last_usb_signature = tuple()
        self._last_disk_check = 0.0
        self.last_session_names = []

        self._build_ui()

        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (WIDTH, HEIGHT), "format": "BGR888"}
        )
        self.picam2.configure(config)
        self.picam2.start()

        self._set_state_badge(False)
        self._update_counts_ui()
        self._update_button_states()
        self._update_roi_info_label()
        self._update_disk_label(force=True)

        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self.update_preview_and_capture)
        self.preview_timer.start(PREVIEW_INTERVAL_MS)

        self.info_timer = QTimer(self)
        self.info_timer.timeout.connect(self.update_info_and_table)
        self.info_timer.start(INFO_INTERVAL_MS)

        self.usb_timer = QTimer(self)
        self.usb_timer.timeout.connect(self.refresh_usb_ui)
        self.usb_timer.start(USB_INTERVAL_MS)

        self.refresh_usb_ui()

    def _build_ui(self):
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

        self.preview_label = QLabel("Camera preview…")
        self.preview_label.setObjectName("Preview")
        self.preview_label.setAlignment(Qt.AlignCenter)

        self.info_panel = QFrame()
        self.info_panel.setObjectName("InfoPanel")
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(14, 14, 14, 14)
        info_layout.setSpacing(10)

        self.status_frame = QFrame()
        self.status_frame.setObjectName("StatusFrame")
        status_layout = QVBoxLayout()
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(8)

        title_row = QHBoxLayout()
        self.info_title = QLabel("현재 상태")
        self.info_title.setObjectName("InfoTitle")

        self.state_badge = QLabel("")
        self.state_badge.setObjectName("StateBadge")
        self.state_badge.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        title_row.addWidget(self.info_title, stretch=1)
        title_row.addWidget(self.state_badge, stretch=0)
        status_layout.addLayout(title_row)

        self.lab_state = QLabel("실험 상태: 대기")
        self.lab_start = QLabel("실험 시작 시간: -")
        self.lab_elapsed = QLabel("실험 경과 시간: 00:00:00")
        self.lab_disk = QLabel("남은 용량: -")
        self.lab_img_count = QLabel("이미지 수집: 0")
        self.lab_data_count = QLabel("데이터 수집: 0")
        self.lab_roi_size = QLabel("ROI 영역: -")
        self.lab_roi_avg = QLabel("ROI 평균 RGB: -")
        self.lab_grid_avg = QLabel("99포인트 평균 RGB: -")

        for w in [
            self.lab_state, self.lab_start, self.lab_elapsed, self.lab_disk,
            self.lab_img_count, self.lab_data_count, self.lab_roi_size,
            self.lab_roi_avg, self.lab_grid_avg
        ]:
            w.setObjectName("InfoLine")
            status_layout.addWidget(w)

        status_layout.addStretch(1)
        self.status_frame.setLayout(status_layout)

        self.usb_frame = QFrame()
        self.usb_frame.setObjectName("UsbFrame")
        usb_layout = QVBoxLayout()
        usb_layout.setContentsMargins(0, 0, 0, 0)
        usb_layout.setSpacing(8)

        usb_sep = QFrame()
        usb_sep.setFrameShape(QFrame.HLine)
        usb_sep.setObjectName("Separator")

        self.usb_title = QLabel("USB 데이터 전송")
        self.usb_title.setObjectName("InfoTitle")

        self.usb_status = QLabel("USB: 감지 중...")
        self.usb_status.setObjectName("InfoLine")

        self.recent_title = QLabel("최근 실험 내역")
        self.recent_title.setObjectName("SubTitle")

        self.recent_labels = []
        for _ in range(3):
            lbl = QLabel("-")
            lbl.setObjectName("RecentLine")
            self.recent_labels.append(lbl)

        self.session_combo = QComboBox()
        self.session_combo.setObjectName("UsbCombo")

        self.btn_refresh_usb = QPushButton("새로고침")
        self.btn_refresh_usb.setObjectName("UsbBtn")

        self.btn_copy_usb = QPushButton("USB로 복사")
        self.btn_copy_usb.setObjectName("UsbBtnPrimary")
        self.btn_copy_usb.setEnabled(False)

        self.btn_eject_usb = QPushButton("USB 제거")
        self.btn_eject_usb.setObjectName("UsbBtn")
        self.btn_eject_usb.setEnabled(False)

        self.usb_progress = QLabel("대기 중")
        self.usb_progress.setObjectName("InfoLine")

        usb_row = QHBoxLayout()
        usb_row.addWidget(self.session_combo, stretch=1)
        usb_row.addWidget(self.btn_refresh_usb)
        usb_row.addWidget(self.btn_copy_usb)
        usb_row.addWidget(self.btn_eject_usb)

        usb_layout.addWidget(usb_sep)
        usb_layout.addWidget(self.usb_title)
        usb_layout.addWidget(self.usb_status)
        usb_layout.addWidget(self.recent_title)
        for lbl in self.recent_labels:
            usb_layout.addWidget(lbl)
        usb_layout.addLayout(usb_row)
        usb_layout.addWidget(self.usb_progress)
        usb_layout.addStretch(1)
        self.usb_frame.setLayout(usb_layout)

        info_layout.addWidget(self.status_frame, stretch=1)
        info_layout.addWidget(self.usb_frame, stretch=1)
        self.info_panel.setLayout(info_layout)

        self.table = QTableWidget(2, 4)
        self.table.setObjectName("RGBTable")
        self.table.setHorizontalHeaderLabels(["Type", "R", "G", "B"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self._set_table_item(0, 0, "ROI 평균", align=Qt.AlignCenter)
        self._set_table_item(0, 1, "-", align=Qt.AlignCenter)
        self._set_table_item(0, 2, "-", align=Qt.AlignCenter)
        self._set_table_item(0, 3, "-", align=Qt.AlignCenter)

        self._set_table_item(1, 0, f"{GRID_ROWS}x{GRID_COLS} 평균", align=Qt.AlignCenter)
        self._set_table_item(1, 1, "-", align=Qt.AlignCenter)
        self._set_table_item(1, 2, "-", align=Qt.AlignCenter)
        self._set_table_item(1, 3, "-", align=Qt.AlignCenter)

        self.plot = RGBPlotWidget(title="ROI 평균 RGB 그래프")

        self.btn_start = QPushButton("실험 시작")
        self.btn_start.setObjectName("StartBtn")

        self.btn_stop = QPushButton("실험 종료")
        self.btn_stop.setObjectName("StopBtn")
        self.btn_stop.setEnabled(False)

        self.btn_dev = QPushButton("개발자 모드")
        self.btn_dev.setObjectName("DevBtn")

        self.btn_power = QPushButton("Power Off")
        self.btn_power.setObjectName("PowerBtn")

        self.btn_start.clicked.connect(self.start_experiment)
        self.btn_stop.clicked.connect(self.stop_experiment)
        self.btn_dev.clicked.connect(self.open_developer_mode)
        self.btn_power.clicked.connect(self.confirm_power_off)

        self.btn_refresh_usb.clicked.connect(self.refresh_usb_ui)
        self.btn_copy_usb.clicked.connect(self.copy_selected_session_to_usb)
        self.btn_eject_usb.clicked.connect(self.confirm_eject_usb)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(12)

        grid.addWidget(self.preview_label, 0, 0)
        grid.addWidget(self.info_panel, 0, 1)

        grid.addWidget(self.table, 1, 0)
        grid.addWidget(self.plot, 1, 1)

        button_row = QHBoxLayout()
        button_row.setSpacing(12)
        button_row.addWidget(self.btn_start)
        button_row.addWidget(self.btn_stop)
        button_row.addWidget(self.btn_dev)
        button_row.addWidget(self.btn_power)

        grid.addLayout(button_row, 2, 0, 1, 2)

        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 2)
        grid.setRowStretch(0, 4)
        grid.setRowStretch(1, 3)
        grid.setRowStretch(2, 0)

        layout = QVBoxLayout()
        layout.addLayout(top_row)
        layout.addWidget(separator)
        layout.addLayout(grid, stretch=1)
        self.setLayout(layout)

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

        #Separator { color: #334155; border: 1px solid #334155; }

        #Preview {
            background: #0b1220;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 6px;
            min-height: 260px;
        }

        #InfoPanel {
            background: #0b1220;
            border: 1px solid #334155;
            border-radius: 12px;
        }

        #InfoTitle {
            font-size: 16px;
            font-weight: 800;
            margin-bottom: 2px;
        }

        #InfoLine {
            font-size: 14px;
            color: #cbd5e1;
        }

        #StateBadge {
            font-size: 14px;
            font-weight: 900;
            color: #e2e8f0;
        }

        #SubTitle {
            font-size: 14px;
            font-weight: 800;
            color: #e2e8f0;
            margin-top: 4px;
        }

        #RecentLine {
            font-size: 13px;
            color: #94a3b8;
            padding-left: 2px;
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
            font-weight: 700;
        }

        QTableWidget::item { padding: 10px; }

        #PlotPanel {
            background: #0b1220;
            border: 1px solid #334155;
            border-radius: 12px;
        }

        QPushButton {
            border: none;
            border-radius: 12px;
            padding: 18px 16px;
            font-size: 20px;
            font-weight: 900;
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

        #DevBtn {
            background: #8b5cf6;
            color: white;
        }
        #DevBtn:hover { background: #7c3aed; }

        #PowerBtn {
            background: #f59e0b;
            color: #0b1220;
        }
        #PowerBtn:hover { background: #d97706; }

        #UsbCombo {
            background: #111827;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 8px;
            color: #e2e8f0;
            font-size: 14px;
        }

        #UsbBtn {
            background: #334155;
            color: #e2e8f0;
            padding: 10px 12px;
            font-size: 14px;
            font-weight: 800;
            border-radius: 10px;
        }
        #UsbBtn:hover { background: #475569; }

        #UsbBtnPrimary {
            background: #22c55e;
            color: #0b1220;
            padding: 10px 12px;
            font-size: 14px;
            font-weight: 900;
            border-radius: 10px;
        }
        #UsbBtnPrimary:hover { background: #16a34a; }
        """

    def _set_table_item(self, row, col, text, align=Qt.AlignLeft):
        item = QTableWidgetItem(text)
        item.setTextAlignment(align)
        self.table.setItem(row, col, item)

    def _set_state_badge(self, running: bool):
        if running:
            self.state_badge.setText('<span style="color:#ef4444;">●</span> 실험 중')
        else:
            self.state_badge.setText('<span style="color:#3b82f6;">●</span> 실험 대기')

    def _update_counts_ui(self):
        self.lab_img_count.setText(f"이미지 수집: {self.image_count}")
        self.lab_data_count.setText(f"데이터 수집: {self.data_log_count}")

    def _update_roi_info_label(self):
        left, top, right, bottom = self.roi
        self.lab_roi_size.setText(
            f"ROI 영역: ({left}, {top}) ~ ({right}, {bottom})  /  {right-left+1}x{bottom-top+1}px"
        )

    def _update_button_states(self):
        self.btn_start.setEnabled(not self.running and not self.is_shutting_down)
        self.btn_stop.setEnabled(self.running and not self.is_shutting_down)
        self.btn_dev.setEnabled(not self.is_shutting_down)
        self.btn_power.setEnabled(not self.is_shutting_down)

    def _update_disk_label(self, force=False):
        now_t = time.time()
        if (not force) and (now_t - self._last_disk_check < DISK_UPDATE_SEC):
            return

        self._last_disk_check = now_t
        try:
            DATA_ROOT.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(str(DATA_ROOT.resolve()))
            self.lab_disk.setText(f"남은 용량: {fmt_bytes(usage.free)} (총 {fmt_bytes(usage.total)})")
        except Exception:
            self.lab_disk.setText("남은 용량: 확인 실패")

    def _update_recent_sessions_ui(self):
        recents = recent_sessions(DATA_ROOT, limit=3)
        for i in range(3):
            if i < len(recents):
                name = recents[i].name
                self.recent_labels[i].setText(f"• {session_display_name(name)}")
            else:
                self.recent_labels[i].setText("• -")

    def _update_usb_state_transition(self, mounts):
        current_sig = tuple(mounts)
        if current_sig != self.last_usb_signature and len(current_sig) > 0:
            self.usb_removed_mode = False
        if len(current_sig) == 0:
            self.usb_removed_mode = False
        self.last_usb_signature = current_sig

    # =================== USB ===================

    def refresh_usb_ui(self):
        mounts = find_usb_mounts()
        self._update_usb_state_transition(mounts)

        self.usb_mounts = mounts
        self.usb_mount = mounts[0] if mounts else None

        if self.usb_removed_mode:
            self.usb_status.setText("USB: 안전 제거 완료 (이제 뽑아도 됨)")
        elif self.usb_mount:
            self.usb_status.setText(f"USB: 연결됨  •  {self.usb_mount}")
        else:
            self.usb_status.setText("USB: 연결 안됨")

        self._update_recent_sessions_ui()

        sessions = list_sessions(DATA_ROOT)
        session_names = [s.name for s in sessions]
        current = self.session_combo.currentText() if self.session_combo.count() else ""

        if session_names != self.last_session_names:
            self.session_combo.blockSignals(True)
            self.session_combo.clear()
            for s in sessions:
                self.session_combo.addItem(s.name)
            if current and current in session_names:
                self.session_combo.setCurrentText(current)
            self.session_combo.blockSignals(False)
            self.last_session_names = session_names

        can_copy = (self.usb_mount is not None) and (self.session_combo.count() > 0) and (not self.usb_removed_mode)
        can_eject = (self.usb_mount is not None) and (not self.usb_removed_mode)

        if self.copy_worker is not None and self.copy_worker.isRunning():
            can_copy = False
            can_eject = False

        self.btn_copy_usb.setEnabled(can_copy and not self.is_shutting_down)
        self.btn_eject_usb.setEnabled(can_eject and not self.is_shutting_down)

        if self.usb_removed_mode:
            self.usb_progress.setText("안전 제거 완료 (USB를 분리해도 됨)")
        elif self.session_combo.count() == 0:
            self.usb_progress.setText("대기 중 (복사할 세션 데이터 없음)")
        elif not self.usb_mount:
            self.usb_progress.setText("대기 중 (USB 연결 필요)")
        else:
            s = self.session_combo.currentText().strip()
            self.usb_progress.setText(f"대기 중 (선택 세션: {session_display_name(s)})")

    def copy_selected_session_to_usb(self):
        if self.usb_removed_mode:
            self.usb_progress.setText("안전 제거된 USB임. 다시 꽂은 뒤 사용해줘.")
            return

        if not self.usb_mount:
            self.usb_progress.setText("USB가 연결되지 않음")
            return

        session_name = self.session_combo.currentText().strip()
        if not session_name:
            self.usb_progress.setText("복사할 세션을 선택해줘")
            return

        src_dir = DATA_ROOT / session_name
        if not src_dir.exists() or not src_dir.is_dir():
            self.usb_progress.setText(f"선택한 세션 폴더가 없음: {session_name}")
            return

        dst_root = Path(self.usb_mount) / "Ainanobio_export"

        self.btn_copy_usb.setEnabled(False)
        self.btn_eject_usb.setEnabled(False)
        self.usb_progress.setText("복사 준비 중...")

        self.copy_worker = CopyWorker([src_dir], dst_root)
        self.copy_worker.log.connect(self._on_copy_log)
        self.copy_worker.done.connect(self._on_copy_done)
        self.copy_worker.start()

    def _on_copy_log(self, msg: str):
        self.usb_progress.setText(msg)

    def _on_copy_done(self, ok: bool, msg: str):
        self.usb_progress.setText(msg)
        self.copy_worker = None
        self.refresh_usb_ui()

    def confirm_eject_usb(self):
        if self.copy_worker is not None and self.copy_worker.isRunning():
            QMessageBox.warning(self, "USB 제거 불가", "USB 복사 작업이 진행 중입니다.")
            return

        if not self.usb_mount:
            QMessageBox.information(self, "USB 제거", "현재 연결된 USB가 없어.")
            return

        msg = QMessageBox(self)
        msg.setWindowTitle("USB 제거")
        msg.setText("USB를 안전하게 제거합니다.")
        msg.setInformativeText("제거 완료 메시지가 뜬 뒤 USB를 뽑아줘.")
        msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)

        yes_btn = msg.button(QMessageBox.Yes)
        no_btn = msg.button(QMessageBox.No)
        if yes_btn:
            yes_btn.setText("예")
        if no_btn:
            no_btn.setText("아니요")

        if msg.exec_() == QMessageBox.Yes:
            self.eject_usb()

    def eject_usb(self):
        if not self.usb_mount:
            self.usb_progress.setText("제거할 USB가 없음")
            return

        mount_to_eject = self.usb_mount
        partition_dev = get_device_from_mount(mount_to_eject)
        parent_dev = get_parent_block_device(partition_dev) if partition_dev else None

        self.btn_copy_usb.setEnabled(False)
        self.btn_eject_usb.setEnabled(False)
        self.btn_refresh_usb.setEnabled(False)
        self.session_combo.setEnabled(False)
        self.usb_progress.setText("USB 안전 제거 중...")

        try:
            sync_filesystem()

            umount_ok = False
            umount_err = ""

            try:
                result = subprocess.run(
                    ["udisksctl", "unmount", "-b", partition_dev],
                    capture_output=True, text=True, check=False
                )
                if result.returncode == 0:
                    umount_ok = True
                else:
                    umount_err = result.stderr.strip() or result.stdout.strip()
            except Exception as e:
                umount_err = str(e)

            if not umount_ok:
                try:
                    result = subprocess.run(
                        ["umount", mount_to_eject],
                        capture_output=True, text=True, check=False
                    )
                    if result.returncode == 0:
                        umount_ok = True
                    else:
                        umount_err = result.stderr.strip() or result.stdout.strip()
                except Exception as e:
                    umount_err = str(e)

            if not umount_ok:
                raise RuntimeError(f"언마운트 실패: {umount_err or '원인 불명'}")

            sync_filesystem()

            poweroff_msg = ""
            if parent_dev:
                try:
                    result = subprocess.run(
                        ["udisksctl", "power-off", "-b", parent_dev],
                        capture_output=True, text=True, check=False
                    )
                    if result.returncode == 0:
                        poweroff_msg = " / 전원 차단 완료"
                    else:
                        poweroff_msg = " / 전원 차단은 생략됨"
                except Exception:
                    poweroff_msg = " / 전원 차단은 생략됨"

            self.usb_removed_mode = True
            self.usb_mount = None
            self.usb_mounts = []
            self.usb_status.setText("USB: 안전 제거 완료 (이제 뽑아도 됨)")
            self.usb_progress.setText(f"안전 제거 완료{poweroff_msg}")

        except Exception as e:
            self.usb_progress.setText(f"USB 제거 실패: {e}")

        finally:
            self.btn_refresh_usb.setEnabled(True)
            self.session_combo.setEnabled(True)
            self.refresh_usb_ui()

    # =================== Power Off ===================

    def confirm_power_off(self):
        if self.copy_worker is not None and self.copy_worker.isRunning():
            QMessageBox.warning(self, "전원 종료 불가", "USB 복사 작업이 진행 중입니다.")
            return

        msg = QMessageBox(self)
        msg.setWindowTitle("전원 종료")
        msg.setText("라즈베리파이 전원을 종료합니다.")
        msg.setInformativeText("진행 중인 작업이 있으면 저장 후 종료됩니다.")
        msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)

        yes_btn = msg.button(QMessageBox.Yes)
        no_btn = msg.button(QMessageBox.No)
        if yes_btn:
            yes_btn.setText("예")
        if no_btn:
            no_btn.setText("아니요")

        if msg.exec_() == QMessageBox.Yes:
            self.power_off_system()

    def power_off_system(self):
        try:
            self.is_shutting_down = True
            self.status_pill.setText("POWER OFF  •  시스템 종료 중...")
            self.usb_progress.setText("시스템 종료 준비 중...")
            self._update_button_states()
            self.btn_copy_usb.setEnabled(False)
            self.btn_eject_usb.setEnabled(False)
            self.btn_refresh_usb.setEnabled(False)
            self.session_combo.setEnabled(False)

            self._stop_all_timers()

            if self.running:
                self.running = False

            self._finalize_logging()

            if self.image_save_worker is not None and self.image_save_worker.isRunning():
                self.image_save_worker.wait(2000)

            try:
                self.picam2.stop()
            except Exception:
                pass

            sync_filesystem()
            QApplication.processEvents()
            subprocess.Popen(["sudo", "shutdown", "-h", "now"])

        except Exception as e:
            self.usb_progress.setText(f"종료 실패: {e}")
            self.is_shutting_down = False
            self._update_button_states()

    # =================== Developer Mode / Close ===================

    def open_developer_mode(self):
        if self.is_shutting_down:
            return

        pwd, ok = QInputDialog.getText(
            self,
            "개발자 모드",
            "비밀번호를 입력하세요:",
            QLineEdit.Password
        )

        if not ok:
            return

        if pwd != DEV_PASSWORD:
            QMessageBox.warning(self, "개발자 모드", "비밀번호가 틀렸습니다.")
            return

        msg = QMessageBox(self)
        msg.setWindowTitle("개발자 모드")
        msg.setText("GUI를 종료합니다.")
        msg.setInformativeText("저장 정리 후 프로그램을 닫습니다.")
        msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)

        yes_btn = msg.button(QMessageBox.Yes)
        no_btn = msg.button(QMessageBox.No)
        if yes_btn:
            yes_btn.setText("종료")
        if no_btn:
            no_btn.setText("취소")

        if msg.exec_() == QMessageBox.Yes:
            self.developer_close()

    def developer_close(self):
        try:
            self.is_shutting_down = True
            self.status_pill.setText("DEVELOPER MODE  •  종료 중...")
            self._update_button_states()
            self.btn_copy_usb.setEnabled(False)
            self.btn_eject_usb.setEnabled(False)
            self.btn_refresh_usb.setEnabled(False)
            self.session_combo.setEnabled(False)

            self._stop_all_timers()

            if self.running:
                self.running = False

            self._finalize_logging()

            if self.copy_worker is not None and self.copy_worker.isRunning():
                self.copy_worker.wait(2000)

            if self.image_save_worker is not None and self.image_save_worker.isRunning():
                self.image_save_worker.wait(2000)

            try:
                self.picam2.stop()
            except Exception:
                pass

            sync_filesystem()

            self.allow_close = True
            self.close()

        except Exception as e:
            QMessageBox.warning(self, "종료 실패", str(e))
            self.is_shutting_down = False
            self.allow_close = False
            self._update_button_states()

    def _stop_all_timers(self):
        try:
            self.preview_timer.stop()
        except Exception:
            pass
        try:
            self.info_timer.stop()
        except Exception:
            pass
        try:
            self.usb_timer.stop()
        except Exception:
            pass

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F4 and (event.modifiers() & Qt.AltModifier):
            event.ignore()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        if not self.allow_close:
            event.ignore()
            return
        event.accept()

    # =================== Experiment ===================

    def _build_csv_header(self):
        header = [
            "Timestamp",
            "ROI_LEFT", "ROI_TOP", "ROI_RIGHT", "ROI_BOTTOM",
            "ROI_AVG_R", "ROI_AVG_G", "ROI_AVG_B",
            "GRID_AVG_R", "GRID_AVG_G", "GRID_AVG_B"
        ]

        for pid, x, y in self.grid_points:
            header.extend([f"{pid}_X", f"{pid}_Y", f"{pid}_R", f"{pid}_G", f"{pid}_B"])

        return header

    def start_experiment(self):
        if self.running or self.is_shutting_down:
            return

        sess = session_stamp()
        self.session_dir = DATA_ROOT / sess
        self.images_dir = self.session_dir / "images"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.session_dir / f"rgb_roi_grid99_{sess}.csv"
        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(self._build_csv_header())

        self.running = True

        now_t = time.time()
        self.next_rgb_log_time = now_t
        self.next_img_log_time = now_t
        self.last_csv_flush_time = now_t

        self.experiment_start_dt = datetime.now()
        self.sample_count = 0
        self.image_count = 0
        self.data_log_count = 0
        self._update_counts_ui()

        self.lab_state.setText("실험 상태: 실험중")
        self.lab_start.setText(f"실험 시작 시간: {self.experiment_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        self.lab_elapsed.setText("실험 경과 시간: 00:00:00")
        self.status_pill.setText(f"RECORDING  •  {sess}")

        self._set_state_badge(True)
        self._update_button_states()
        self.plot.reset()
        self.refresh_usb_ui()

    def stop_experiment(self):
        if not self.running:
            return

        self.running = False
        self.status_pill.setText("READY  •  Camera ON")
        self.lab_state.setText("실험 상태: 대기")
        self._set_state_badge(False)
        self._update_disk_label(force=True)
        self._update_button_states()
        self._finalize_logging()
        self.refresh_usb_ui()

    def _finalize_logging(self):
        try:
            if self.csv_file:
                self.csv_file.flush()
                self.csv_file.close()
                self.csv_file = None
                self.csv_writer = None
        except Exception:
            pass

        try:
            sync_filesystem()
        except Exception:
            pass

    # =================== Core Update ===================

    def update_preview_and_capture(self):
        if self.is_shutting_down:
            return

        try:
            frame_bgr = self.picam2.capture_array()
        except Exception:
            return

        if frame_bgr is None:
            return

        self.latest_frame_bgr = frame_bgr

        overlay = draw_roi_and_grid(
            frame_bgr,
            self.roi,
            self.grid_points if SHOW_GRID_POINTS_ON_PREVIEW else None
        )

        disp = resize_for_preview(overlay, PREVIEW_MAX_W)
        disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w = disp_rgb.shape[:2]
        qimg = QImage(disp_rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self.last_preview_qpixmap = QPixmap.fromImage(qimg)
        self.preview_label.setPixmap(self.last_preview_qpixmap)

        if self.running and self.csv_writer is not None:
            self._handle_logging(frame_bgr)

    def update_info_and_table(self):
        self._update_disk_label()

        if self.running and self.experiment_start_dt is not None:
            elapsed = int((datetime.now() - self.experiment_start_dt).total_seconds())
            self.lab_elapsed.setText(f"실험 경과 시간: {fmt_hms(elapsed)}")

        frame_bgr = self.latest_frame_bgr
        if frame_bgr is None:
            return

        roi_avg = get_roi_mean_rgb(frame_bgr, self.roi)
        _, grid_avg = sample_grid_rgb(frame_bgr, self.grid_points)

        if roi_avg is None:
            self.table.item(0, 1).setText("-")
            self.table.item(0, 2).setText("-")
            self.table.item(0, 3).setText("-")
            self.lab_roi_avg.setText("ROI 평균 RGB: -")
        else:
            r, g, b = roi_avg
            self.table.item(0, 1).setText(str(r))
            self.table.item(0, 2).setText(str(g))
            self.table.item(0, 3).setText(str(b))
            self.lab_roi_avg.setText(f"ROI 평균 RGB: R={r}, G={g}, B={b}")

        gr, gg, gb = grid_avg
        if gr == "":
            self.table.item(1, 1).setText("-")
            self.table.item(1, 2).setText("-")
            self.table.item(1, 3).setText("-")
            self.lab_grid_avg.setText("99포인트 평균 RGB: -")
        else:
            self.table.item(1, 1).setText(str(gr))
            self.table.item(1, 2).setText(str(gg))
            self.table.item(1, 3).setText(str(gb))
            self.lab_grid_avg.setText(f"99포인트 평균 RGB: R={gr}, G={gg}, B={gb}")

    def _start_async_image_save(self, frame_bgr):
        if self.image_save_worker is not None and self.image_save_worker.isRunning():
            return False

        self.images_dir.mkdir(parents=True, exist_ok=True)
        t_ms_img = now_ms()
        img_path = self.images_dir / f"{t_ms_img}.{IMAGE_EXT}"

        self.image_save_worker = ImageSaveWorker(
            frame_bgr=frame_bgr,
            img_path=img_path,
            image_ext=IMAGE_EXT,
            jpg_quality=JPG_QUALITY
        )
        self.image_save_worker.done.connect(self._on_image_saved)
        self.image_save_worker.start()
        return True

    def _on_image_saved(self, ok: bool, msg: str):
        if ok:
            self.image_count += 1
            self._update_counts_ui()
        self.image_save_worker = None

    def _handle_logging(self, frame_bgr):
        now_t = time.time()

        if SAVE_IMAGE and now_t >= self.next_img_log_time:
            started = self._start_async_image_save(frame_bgr)
            if started:
                self.next_img_log_time = now_t + IMAGE_LOG_INTERVAL_SEC

        if now_t >= self.next_rgb_log_time:
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            left, top, right, bottom = self.roi

            roi_avg = get_roi_mean_rgb(frame_bgr, self.roi)
            grid_samples, grid_avg = sample_grid_rgb(frame_bgr, self.grid_points)

            if roi_avg is None:
                roi_r, roi_g, roi_b = "", "", ""
            else:
                roi_r, roi_g, roi_b = roi_avg

            grid_r, grid_g, grid_b = grid_avg

            row = [
                timestamp_str,
                left, top, right, bottom,
                roi_r, roi_g, roi_b,
                grid_r, grid_g, grid_b,
            ]

            for pid, x, y, r, g, b in grid_samples:
                row.extend([x, y, r, g, b])

            try:
                self.csv_writer.writerow(row)
                if now_t - self.last_csv_flush_time >= CSV_FLUSH_INTERVAL_SEC:
                    self.csv_file.flush()
                    self.last_csv_flush_time = now_t
            except Exception:
                pass

            self.data_log_count += 1
            self._update_counts_ui()

            if roi_avg is not None:
                self.sample_count += 1
                self.plot.append(self.sample_count, roi_avg)

            self.next_rgb_log_time = now_t + RGB_LOG_INTERVAL_SEC


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = RGBApplianceGUI()
    w.showFullScreen()
    sys.exit(app.exec_())