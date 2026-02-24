import sys
import csv
import time
import shutil
import os
import re
from datetime import datetime
from pathlib import Path
from collections import deque

import cv2
from picamera2 import Picamera2

from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame,
    QComboBox
)

# ================== 고정 설정(필요 시 너만 수정) ==================
WIDTH, HEIGHT = 1280, 720
PREVIEW_MAX_W = 640

# ✅ 저장 주기 분리
RGB_LOG_INTERVAL_SEC = 60.0      # RGB(CSV) 1분마다
IMAGE_LOG_INTERVAL_SEC = 300.0   # 이미지 5분마다

SAVE_IMAGE = True
IMAGE_EXT = "jpg"
JPG_QUALITY = 90
DATA_ROOT = Path("/home/pi/Ainanobio_data")

POINTS = [
    ("p1", 506, 253),
    ("p2", 746, 253),
    ("p3", 506, 387),
    ("p4", 746, 387),
    ("pc", 626, 320),
]

# ✅ 그래프는 기본으로 pc만 그리기
PLOT_POINT_ID = "pc"

# ✅ 그래프에 유지할 최대 샘플 수
PLOT_MAX_POINTS = 240

# ✅ 디스크 용량 표시 갱신 주기
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
    gib = n / (1024**3)
    return f"{gib:.2f} GB"


# ===================== USB helpers =====================

def find_usb_mounts():
    """
    USB 자동 마운트 지점 탐색(대개 /media/* 또는 /run/media/*).
    """
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

    mounts = sorted(list(dict.fromkeys(mounts)))
    return mounts


def list_session_dates(data_root: Path):
    """
    DATA_ROOT 아래 세션 폴더(YYYY-MM-DD_HH-MM-SS)를 스캔해서 날짜 목록만 뽑음.
    """
    dates = set()
    if not data_root.exists():
        return []
    for p in data_root.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", name):
            dates.add(name[:10])
    return sorted(dates, reverse=True)


def sessions_for_date(data_root: Path, yyyy_mm_dd: str):
    """
    특정 날짜의 세션 폴더들 반환.
    """
    out = []
    if not data_root.exists():
        return out
    prefix = f"{yyyy_mm_dd}_"
    for p in data_root.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            out.append(p)
    return sorted(out)


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

                # 이미 존재하면 덮어쓰기
                if dst.exists():
                    shutil.rmtree(dst)

                shutil.copytree(src, dst)

            self.done.emit(True, f"완료: {len(self.src_dirs)}개 세션 복사됨")
        except Exception as e:
            self.done.emit(False, f"실패: {e}")


class RGBPlotWidget(QFrame):
    """
    가벼운 실시간 그래프 위젯(QPainter로 직접 그림).
    - y축: 0~255
    - x축: 샘플 인덱스(최근 PLOT_MAX_POINTS)
    """
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

        self._pen_r = QPen(QColor(239, 68, 68), 2)   # Red-ish
        self._pen_g = QPen(QColor(34, 197, 94), 2)   # Green-ish
        self._pen_b = QPen(QColor(59, 130, 246), 2)  # Blue-ish

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
        f2.setBold(False)
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
            v = max(0, min(255, v))
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

        p.setFont(f2)
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

        # ===== 상단 타이틀 / 상태 pill =====
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

        # ===== 좌상단: 프리뷰 =====
        self.preview_label = QLabel("Camera preview…")
        self.preview_label.setObjectName("Preview")
        self.preview_label.setAlignment(Qt.AlignCenter)

        # ===== 우상단: 현재 상태 + USB 전송 패널(반반) =====
        self.info_panel = QFrame()
        self.info_panel.setObjectName("InfoPanel")
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(14, 14, 14, 14)
        info_layout.setSpacing(10)

        # --- (A) 상태 영역 ---
        self.status_frame = QFrame()
        self.status_frame.setObjectName("StatusFrame")
        status_layout = QVBoxLayout()
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(8)

        self.info_title = QLabel("현재 상태")
        self.info_title.setObjectName("InfoTitle")

        self.lab_state = QLabel("실험 상태: 대기")
        self.lab_start = QLabel("실험 시작 시간: -")
        self.lab_elapsed = QLabel("실험 경과 시간: 00:00:00")
        self.lab_disk = QLabel("남은 용량: -")

        for w in [self.lab_state, self.lab_start, self.lab_elapsed, self.lab_disk]:
            w.setObjectName("InfoLine")

        status_layout.addWidget(self.info_title)
        status_layout.addWidget(self.lab_state)
        status_layout.addWidget(self.lab_start)
        status_layout.addWidget(self.lab_elapsed)
        status_layout.addWidget(self.lab_disk)
        status_layout.addStretch(1)
        self.status_frame.setLayout(status_layout)

        # --- (B) USB 전송 영역 ---
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

        self.date_combo = QComboBox()
        self.date_combo.setObjectName("UsbCombo")

        self.btn_refresh_usb = QPushButton("새로고침")
        self.btn_refresh_usb.setObjectName("UsbBtn")

        self.btn_copy_usb = QPushButton("USB로 복사")
        self.btn_copy_usb.setObjectName("UsbBtnPrimary")
        self.btn_copy_usb.setEnabled(False)

        self.usb_progress = QLabel("대기 중")
        self.usb_progress.setObjectName("InfoLine")

        usb_row = QHBoxLayout()
        usb_row.addWidget(self.date_combo, stretch=1)
        usb_row.addWidget(self.btn_refresh_usb)
        usb_row.addWidget(self.btn_copy_usb)

        usb_layout.addWidget(usb_sep)
        usb_layout.addWidget(self.usb_title)
        usb_layout.addWidget(self.usb_status)
        usb_layout.addLayout(usb_row)
        usb_layout.addWidget(self.usb_progress)
        usb_layout.addStretch(1)
        self.usb_frame.setLayout(usb_layout)

        # 반반 배치(Stretch)
        info_layout.addWidget(self.status_frame, stretch=1)
        info_layout.addWidget(self.usb_frame, stretch=1)
        self.info_panel.setLayout(info_layout)

        # ===== 좌중단: Live RGB 테이블 (Point + RGB만) =====
        self.table = QTableWidget(len(POINTS), 4)
        self.table.setObjectName("RGBTable")
        self.table.setHorizontalHeaderLabels(["Point", "R", "G", "B"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        for row, (pid, _, _) in enumerate(POINTS):
            self._set_table_item(row, 0, pid, align=Qt.AlignCenter)
            self._set_table_item(row, 1, "-", align=Qt.AlignCenter)
            self._set_table_item(row, 2, "-", align=Qt.AlignCenter)
            self._set_table_item(row, 3, "-", align=Qt.AlignCenter)

        # ===== 우중단: 실험 저장값 그래프 =====
        self.plot = RGBPlotWidget(title=f"실험 RGB 그래프 ({PLOT_POINT_ID})")

        # ===== 좌하단/우하단: 버튼 =====
        self.btn_start = QPushButton("실험 시작")
        self.btn_start.setObjectName("StartBtn")
        self.btn_stop = QPushButton("실험 종료")
        self.btn_stop.setObjectName("StopBtn")
        self.btn_stop.setEnabled(False)

        self.btn_start.clicked.connect(self.start_experiment)
        self.btn_stop.clicked.connect(self.stop_experiment)

        # ===== 2열×3행 레이아웃 =====
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(12)

        grid.addWidget(self.preview_label, 0, 0)
        grid.addWidget(self.info_panel,   0, 1)

        grid.addWidget(self.table,        1, 0)
        grid.addWidget(self.plot,         1, 1)

        grid.addWidget(self.btn_start,    2, 0)
        grid.addWidget(self.btn_stop,     2, 1)

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

        # ===== Camera =====
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (WIDTH, HEIGHT), "format": "BGR888"}
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

        self.next_rgb_log_time = time.time()
        self.next_img_log_time = time.time()

        self.experiment_start_dt = None
        self.sample_count = 0

        self._last_disk_check = 0.0
        self._update_disk_label(force=True)

        # ===== USB state =====
        self.usb_mounts = []
        self.usb_mount = None
        self.copy_worker = None

        self.btn_refresh_usb.clicked.connect(self.refresh_usb_ui)
        self.btn_copy_usb.clicked.connect(self.copy_selected_date_to_usb)

        self.usb_timer = QTimer()
        self.usb_timer.timeout.connect(self.refresh_usb_ui)
        self.usb_timer.start(1000)  # 1초마다 감지/갱신
        self.refresh_usb_ui()

        # ===== Main Timer =====
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(40)

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

        #InfoPanel{
            background: #0b1220;
            border: 1px solid #334155;
            border-radius: 12px;
        }
        #InfoTitle{
            font-size: 16px;
            font-weight: 800;
            margin-bottom: 2px;
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
        QPushButton:disabled { background: #334155; color: #94a3b8; }
        #StartBtn { background: #ef4444; color: white; }
        #StartBtn:hover { background: #dc2626; }
        #StopBtn { background: #3b82f6; color: white; }
        #StopBtn:hover { background: #2563eb; }

        /* USB area */
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

    # =================== USB UI/Logic ===================

    def refresh_usb_ui(self):
        mounts = find_usb_mounts()
        self.usb_mounts = mounts
        self.usb_mount = mounts[0] if mounts else None

        if self.usb_mount:
            self.usb_status.setText(f"USB: 연결됨  •  {self.usb_mount}")
        else:
            self.usb_status.setText("USB: 연결 안됨")

        dates = list_session_dates(DATA_ROOT)
        current = self.date_combo.currentText() if self.date_combo.count() else ""

        self.date_combo.blockSignals(True)
        self.date_combo.clear()
        for d in dates:
            self.date_combo.addItem(d)
        if current and current in dates:
            self.date_combo.setCurrentText(current)
        self.date_combo.blockSignals(False)

        can_copy = (self.usb_mount is not None) and (self.date_combo.count() > 0)
        if self.copy_worker is not None and self.copy_worker.isRunning():
            can_copy = False

        self.btn_copy_usb.setEnabled(can_copy)

        if self.date_combo.count() == 0:
            self.usb_progress.setText("대기 중 (복사할 날짜 데이터 없음)")
        elif not self.usb_mount:
            self.usb_progress.setText("대기 중 (USB 연결 필요)")
        else:
            d = self.date_combo.currentText()
            n = len(sessions_for_date(DATA_ROOT, d)) if d else 0
            self.usb_progress.setText(f"대기 중 (선택 날짜: {d}, 세션 {n}개)")

    def copy_selected_date_to_usb(self):
        if not self.usb_mount:
            self.usb_progress.setText("USB가 연결되지 않음")
            return

        date = self.date_combo.currentText().strip()
        if not date:
            self.usb_progress.setText("복사할 날짜를 선택해줘")
            return

        src_dirs = sessions_for_date(DATA_ROOT, date)
        if not src_dirs:
            self.usb_progress.setText(f"{date} 날짜에 복사할 세션 폴더가 없음")
            return

        dst_root = Path(self.usb_mount) / "Ainanobio_export" / date

        self.btn_copy_usb.setEnabled(False)
        self.usb_progress.setText("복사 준비 중...")

        self.copy_worker = CopyWorker(src_dirs, dst_root)
        self.copy_worker.log.connect(self._on_copy_log)
        self.copy_worker.done.connect(self._on_copy_done)
        self.copy_worker.start()

    def _on_copy_log(self, msg: str):
        self.usb_progress.setText(msg)

    def _on_copy_done(self, ok: bool, msg: str):
        self.usb_progress.setText(msg)
        self.copy_worker = None
        self.refresh_usb_ui()

    # =================== Experiment ===================

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

        now_t = time.time()
        self.next_rgb_log_time = now_t
        self.next_img_log_time = now_t

        self.experiment_start_dt = datetime.now()
        self.sample_count = 0

        self.lab_state.setText("실험 상태: 실험중")
        self.lab_start.setText(f"실험 시작 시간: {self.experiment_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        self.lab_elapsed.setText("실험 경과 시간: 00:00:00")

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_pill.setText(f"RECORDING  •  {sess}")

        self.plot.reset()
        self.refresh_usb_ui()

    def stop_experiment(self):
        if not self.running:
            return

        self.running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_pill.setText("READY  •  Camera ON")

        self.lab_state.setText("실험 상태: 대기")
        self._update_disk_label(force=True)

        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

        self.refresh_usb_ui()

    def tick(self):
        # ⚠️ 네 환경에서 이 조합이 색이 맞는 상태(유지)
        frame_rgb = self.picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        self._update_disk_label()

        if self.running and self.experiment_start_dt is not None:
            elapsed = int((datetime.now() - self.experiment_start_dt).total_seconds())
            self.lab_elapsed.setText(f"실험 경과 시간: {fmt_hms(elapsed)}")

        # Live RGB 테이블 갱신(현재 프레임 기준)
        for row, (pid, x, y) in enumerate(POINTS):
            rgb = safe_rgb(frame_bgr, x, y)
            if rgb is None:
                self.table.item(row, 1).setText("-")
                self.table.item(row, 2).setText("-")
                self.table.item(row, 3).setText("-")
            else:
                r, g, b = rgb
                self.table.item(row, 1).setText(str(r))
                self.table.item(row, 2).setText(str(g))
                self.table.item(row, 3).setText(str(b))

        # 프리뷰(좌상단)
        overlay = draw_points(frame_bgr, POINTS)
        disp = resize_for_preview(overlay, PREVIEW_MAX_W)
        disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w = disp_rgb.shape[:2]
        qimg = QImage(disp_rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self.preview_label.setPixmap(QPixmap.fromImage(qimg))

        # ================= 저장 로직 분리 =================
        if self.running and self.csv_writer is not None:
            now_t = time.time()

            # (1) 이미지 저장: 5분마다
            img_path_str = ""
            if SAVE_IMAGE and now_t >= self.next_img_log_time:
                self.images_dir.mkdir(parents=True, exist_ok=True)
                t_ms_img = now_ms()
                img_path = self.images_dir / f"{t_ms_img}.{IMAGE_EXT}"
                if IMAGE_EXT.lower() in ["jpg", "jpeg"]:
                    cv2.imwrite(str(img_path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY])
                else:
                    cv2.imwrite(str(img_path), frame_bgr)
                img_path_str = str(img_path)

                self.next_img_log_time = now_t + IMAGE_LOG_INTERVAL_SEC

            # (2) RGB 저장: 1분마다 (그래프 갱신도 여기서)
            if now_t >= self.next_rgb_log_time:
                t_iso = now_iso()
                t_ms = now_ms()

                plot_rgb = None
                for pid, x, y in POINTS:
                    rgb = safe_rgb(frame_bgr, x, y)
                    if pid == PLOT_POINT_ID:
                        plot_rgb = rgb

                    if rgb is None:
                        self.csv_writer.writerow([t_iso, t_ms, img_path_str, pid, x, y, "", "", ""])
                    else:
                        r, g, b = rgb
                        self.csv_writer.writerow([t_iso, t_ms, img_path_str, pid, x, y, r, g, b])

                self.csv_file.flush()

                if plot_rgb is not None:
                    self.sample_count += 1
                    self.plot.append(self.sample_count, plot_rgb)

                self.next_rgb_log_time = now_t + RGB_LOG_INTERVAL_SEC
        # ===================================================

    def closeEvent(self, event):
        try:
            self.running = False
            if self.csv_file:
                self.csv_file.close()

            if self.copy_worker is not None and self.copy_worker.isRunning():
                # 복사중이면 짧게 정리(강제종료는 데이터 손상 위험)
                self.usb_progress.setText("종료 중... (USB 복사 작업 정리)")
                self.copy_worker.wait(1500)

            self.picam2.stop()
        except Exception:
            pass
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = RGBApplianceGUI()
    w.showFullScreen()
    sys.exit(app.exec_())