import sys, json, time, csv
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import cv2
from picamera2 import Picamera2

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFormLayout, QDoubleSpinBox, QSpinBox, QCheckBox, QMessageBox, QFileDialog
)

# =========================
# 설정 모델 + 저장/로드
# =========================
CONFIG_PATH = Path("config.json")
DATA_DIR = Path("data")

@dataclass
class AppConfig:
    width: int = 1280
    height: int = 720
    preview_max_w: int = 960
    log_interval_sec: float = 1.0

    save_image: bool = True
    save_every_n: int = 1
    jpg_quality: int = 90

    # 5포인트 (p1..p4, pc)
    p1x: int = 300; p1y: int = 150
    p2x: int = 980; p2y: int = 150
    p3x: int = 300; p3y: int = 560
    p4x: int = 980; p4y: int = 560
    pcx: int = 640; pcy: int = 360

def load_config() -> AppConfig:
    if CONFIG_PATH.exists():
        d = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return AppConfig(**d)
    return AppConfig()

def save_config(cfg: AppConfig):
    CONFIG_PATH.write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")

def session_stamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def now_iso():
    return datetime.now().isoformat(timespec="milliseconds")

def now_ms():
    return int(time.time() * 1000)

# =========================
# 메인 GUI
# =========================
class ClickableLabel(QLabel):
    """프리뷰 QLabel에 마우스 클릭 이벤트를 받기 위한 라벨"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._click_cb = None

    def set_click_callback(self, cb):
        self._click_cb = cb

    def mousePressEvent(self, ev):
        if self._click_cb is not None and ev.button() == Qt.LeftButton:
            self._click_cb(ev.pos())

class RGBGui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI NanoBio RGB Logger (Pi5)")

        self.cfg = load_config()

        # Picamera2
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (self.cfg.width, self.cfg.height), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()

        # 상태
        self.running = False
        self.edit_mode = False
        self.edit_order = ["p1", "p2", "p3", "p4", "pc"]
        self.edit_idx = 0

        self.session_dir = None
        self.images_dir = None
        self.csv_path = None
        self.csv_file = None
        self.csv_writer = None
        self.sample_idx = 0
        self.next_log_time = time.time()

        # UI 구성
        self.preview = ClickableLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.set_click_callback(self.on_preview_click)

        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)

        self.btn_edit = QPushButton("Edit Points: OFF")
        self.btn_save_cfg = QPushButton("Save Config")

        self.btn_start.clicked.connect(self.start_logging)
        self.btn_stop.clicked.connect(self.stop_logging)
        self.btn_edit.clicked.connect(self.toggle_edit_mode)
        self.btn_save_cfg.clicked.connect(self.on_save_config)

        # 설정 폼
        self.spin_w = QSpinBox(); self.spin_w.setRange(320, 4056); self.spin_w.setValue(self.cfg.width)
        self.spin_h = QSpinBox(); self.spin_h.setRange(240, 3040); self.spin_h.setValue(self.cfg.height)
        self.spin_prevw = QSpinBox(); self.spin_prevw.setRange(320, 1920); self.spin_prevw.setValue(self.cfg.preview_max_w)

        self.spin_interval = QDoubleSpinBox(); self.spin_interval.setDecimals(3)
        self.spin_interval.setRange(0.05, 3600.0); self.spin_interval.setValue(self.cfg.log_interval_sec)

        self.chk_save_img = QCheckBox("Save Image")
        self.chk_save_img.setChecked(self.cfg.save_image)

        self.spin_every = QSpinBox(); self.spin_every.setRange(1, 100000); self.spin_every.setValue(self.cfg.save_every_n)
        self.spin_jpgq = QSpinBox(); self.spin_jpgq.setRange(10, 100); self.spin_jpgq.setValue(self.cfg.jpg_quality)

        form = QFormLayout()
        form.addRow("Width", self.spin_w)
        form.addRow("Height", self.spin_h)
        form.addRow("Preview max width", self.spin_prevw)
        form.addRow("Log interval (sec)", self.spin_interval)
        form.addRow(self.chk_save_img)
        form.addRow("Save every N", self.spin_every)
        form.addRow("JPG quality", self.spin_jpgq)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_edit)
        btn_row.addWidget(self.btn_save_cfg)

        layout = QVBoxLayout()
        layout.addWidget(self.preview, stretch=1)
        layout.addLayout(form)
        layout.addLayout(btn_row)
        self.setLayout(layout)

        # 프레임 업데이트 타이머 (프리뷰 부드럽게)
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_tick)
        self.timer.start(30)  # 약 33fps 느낌(상황에 따라 실제 fps는 달라짐)

    # ----- 포인트 관련 -----
    def points(self):
        return [
            ("p1", self.cfg.p1x, self.cfg.p1y),
            ("p2", self.cfg.p2x, self.cfg.p2y),
            ("p3", self.cfg.p3x, self.cfg.p3y),
            ("p4", self.cfg.p4x, self.cfg.p4y),
            ("pc", self.cfg.pcx, self.cfg.pcy),
        ]

    def set_point(self, pid, x, y):
        if pid == "p1": self.cfg.p1x, self.cfg.p1y = x, y
        elif pid == "p2": self.cfg.p2x, self.cfg.p2y = x, y
        elif pid == "p3": self.cfg.p3x, self.cfg.p3y = x, y
        elif pid == "p4": self.cfg.p4x, self.cfg.p4y = x, y
        elif pid == "pc": self.cfg.pcx, self.cfg.pcy = x, y

    def toggle_edit_mode(self):
        self.edit_mode = not self.edit_mode
        self.edit_idx = 0
        self.btn_edit.setText(f"Edit Points: {'ON' if self.edit_mode else 'OFF'}")
        if self.edit_mode:
            QMessageBox.information(self, "Edit Mode",
                                    "프리뷰를 클릭하면 p1→p2→p3→p4→pc 순서로 좌표가 설정됩니다.\n"
                                    "다시 버튼을 누르면 편집 모드가 꺼집니다.")

    def on_preview_click(self, pos):
        if not self.edit_mode:
            return

        # QLabel 좌표(pos)를 실제 프레임 좌표로 환산해야 함
        # 현재는 프리뷰가 리사이즈되어 표시되므로 스케일 계산 필요
        if self._last_display_w is None or self._last_display_h is None:
            return

        disp_w, disp_h = self._last_display_w, self._last_display_h
        frame_w, frame_h = self._last_frame_w, self._last_frame_h

        x = int(pos.x() * frame_w / disp_w)
        y = int(pos.y() * frame_h / disp_h)

        pid = self.edit_order[self.edit_idx]
        self.set_point(pid, x, y)

        self.edit_idx = (self.edit_idx + 1) % len(self.edit_order)
        QMessageBox.information(self, "Point Set", f"{pid} = ({x}, {y}) 설정됨")

    # ----- 로깅 -----
    def start_logging(self):
        # 현재 UI 값 -> cfg 반영
        self.apply_ui_to_cfg()

        # 세션 폴더 생성 (실행 시작 시각 기준)
        sess = session_stamp()
        self.session_dir = DATA_DIR / sess
        self.images_dir = self.session_dir / "images"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.session_dir / f"rgb_points_{sess}.csv"
        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["timestamp_iso", "unix_ms", "image_path", "point_id", "x", "y", "R", "G", "B"])

        self.sample_idx = 0
        self.next_log_time = time.time()
        self.running = True

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        QMessageBox.information(self, "Started", f"로깅 시작!\nCSV: {self.csv_path}")

    def stop_logging(self):
        self.running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

        QMessageBox.information(self, "Stopped", "로깅 종료!")

    # ----- 프레임 루프 -----
    def on_tick(self):
        # 프레임 캡처
        frame_rgb = self.picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # 오버레이
        overlay = frame_bgr.copy()
        for pid, x, y in self.points():
            cv2.circle(overlay, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(overlay, pid, (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # 프리뷰 리사이즈
        h, w = overlay.shape[:2]
        disp = overlay
        if w > self.cfg.preview_max_w:
            scale = self.cfg.preview_max_w / w
            disp = cv2.resize(overlay, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # QLabel에 표시 (BGR->RGB)
        disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        dh, dw = disp_rgb.shape[:2]

        # 클릭 환산을 위해 저장
        self._last_display_w, self._last_display_h = dw, dh
        self._last_frame_w, self._last_frame_h = w, h

        qimg = QImage(disp_rgb.data, dw, dh, dw * 3, QImage.Format_RGB888)
        self.preview.setPixmap(QPixmap.fromImage(qimg))

        # 로깅 타이밍
        if self.running and time.time() >= self.next_log_time:
            self.sample_idx += 1
            t_iso = now_iso()
            t_ms = now_ms()

            # 이미지 저장
            img_path_str = ""
            if self.cfg.save_image and (self.sample_idx % self.cfg.save_every_n == 0):
                self.images_dir.mkdir(parents=True, exist_ok=True)
                img_path = self.images_dir / f"{t_ms}.jpg"
                cv2.imwrite(str(img_path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.jpg_quality)])
                img_path_str = str(img_path)

            # RGB 저장 (단일 픽셀)
            for pid, x, y in self.points():
                if 0 <= x < w and 0 <= y < h:
                    b, g, r = frame_bgr[y, x]
                    self.csv_writer.writerow([t_iso, t_ms, img_path_str, pid, x, y, int(r), int(g), int(b)])
                else:
                    self.csv_writer.writerow([t_iso, t_ms, img_path_str, pid, x, y, "", "", ""])

            self.csv_file.flush()
            self.next_log_time = time.time() + self.cfg.log_interval_sec

    # ----- 설정 -----
    def apply_ui_to_cfg(self):
        self.cfg.width = int(self.spin_w.value())
        self.cfg.height = int(self.spin_h.value())
        self.cfg.preview_max_w = int(self.spin_prevw.value())
        self.cfg.log_interval_sec = float(self.spin_interval.value())
        self.cfg.save_image = bool(self.chk_save_img.isChecked())
        self.cfg.save_every_n = int(self.spin_every.value())
        self.cfg.jpg_quality = int(self.spin_jpgq.value())

    def on_save_config(self):
        self.apply_ui_to_cfg()
        save_config(self.cfg)
        QMessageBox.information(self, "Saved", f"config.json 저장 완료")

    def closeEvent(self, event):
        try:
            self.running = False
            if self.csv_file:
                self.csv_file.close()
            self.picam2.stop()
        except:
            pass
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = RGBGui()
    w.resize(1000, 800)
    w.show()
    sys.exit(app.exec_())