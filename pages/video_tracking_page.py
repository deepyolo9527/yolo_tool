
# -*- coding: utf-8 -*-

"""SAM2视频追踪标注页：首帧框选目标 -> 逐帧自动分割追踪 -> 导出帧图片与YOLO标签"""

import json
import os
import sys
import threading
from dataclasses import dataclass
from typing import List, Tuple

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont
from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
                             QComboBox, QListWidget, QListWidgetItem,
                             QSpinBox, QProgressBar, QFileDialog, QMessageBox,
                             QInputDialog, QFrame, QScrollArea)

from pages.data_annotation_page import scan_sam_models, find_sam_model_path
from utils.sam_config import load_sam_settings, resolve_sam_imgsz
from utils import theme

try:
    import cv2
    import numpy as np
    cv2_available = True
except ImportError:
    cv2_available = False

TARGET_COLORS = [(231, 76, 60), (46, 204, 113), (52, 152, 219),
                 (241, 196, 15), (155, 89, 182), (26, 188, 156)]


@dataclass
class TrackTarget:
    """追踪目标"""
    target_id: int
    class_name: str
    class_index: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) 像素坐标
    status: str = "active"           # active / lost / ended
    frame_started: int = 0
    color: tuple = (231, 76, 60)
    missing: int = 0                 # 连续分割失败帧计数（容错用）


class VideoTrackWorker(QThread):
    """SAM2逐帧追踪线程：用目标框中心点作为前景提示，每帧分割并更新bbox"""
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    frame_signal = pyqtSignal(object, int, float)   # (RGB frame, idx, seconds)
    target_update_signal = pyqtSignal(list)         # [{'id','bbox','status'}]
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, video_path, output_dir, sam_model_path, targets,
                 save_every_frames, imgsz):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.sam_model_path = sam_model_path
        self.targets = targets
        self.save_every_frames = max(1, int(save_every_frames))
        self.imgsz = imgsz
        self.max_missing = 15            # 连续失败多少帧才判定目标丢失
        self._reframes = []              # [(target_id, (x,y,w,h))] UI线程投递
        self._new_targets = []           # UI线程在追踪中途新增的目标
        self._reframe_lock = threading.Lock()
        self.is_running = True
        self.is_paused = False

    def run(self):
        cap = None
        try:
            if not cv2_available:
                self.finished_signal.emit(False, "需要安装 opencv-python 处理视频")
                return
            from ultralytics import SAM
            self.log_signal.emit(f"加载SAM2模型: {os.path.basename(self.sam_model_path)}")
            model = SAM(self.sam_model_path)

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.finished_signal.emit(False, "无法打开视频文件")
                return
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.log_signal.emit(f"视频: {total_frames} 帧, {fps:.1f} FPS")

            labels_dir = os.path.join(self.output_dir, 'labels')
            images_dir = os.path.join(self.output_dir, 'images')
            os.makedirs(labels_dir, exist_ok=True)
            os.makedirs(images_dir, exist_ok=True)

            active_targets = [t for t in self.targets if t.status == "active"]
            frame_idx = 0
            last_saved_idx = -1
            w = h = 0
            frame = None
            frame_rgb = None

            while self.is_running and cap.isOpened():
                if self.is_paused:
                    self.msleep(50)
                    continue
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame.shape[:2]

                # 处理 UI 线程投递的"重新框选"与"追踪中新增目标"
                pending, pending_new = [], []
                with self._reframe_lock:
                    if self._reframes:
                        pending = self._reframes
                        self._reframes = []
                    if self._new_targets:
                        pending_new = self._new_targets
                        self._new_targets = []
                for t in pending_new:
                    if t not in self.targets:
                        self.targets.append(t)
                    if t not in active_targets:
                        active_targets.append(t)
                    self.log_signal.emit(
                        f"目标 #{t.target_id} ({t.class_name}) 已加入追踪，帧 {frame_idx}")
                if pending:
                    by_id = {t.target_id: t for t in self.targets}
                    active_ids = {t.target_id for t in active_targets}
                    for tid, bbox in pending:
                        t = by_id.get(tid)
                        if t is None:
                            continue
                        t.bbox = bbox
                        t.status = "active"
                        t.missing = 0
                        if tid not in active_ids:
                            active_targets.append(t)
                            active_ids.add(tid)
                        self.log_signal.emit(f"目标 #{tid} 已在帧 {frame_idx} 重新框选")

                contours_map = {}
                if active_targets:
                    # 用上一帧的 bbox 作为框提示（比单点稳定得多），逐目标重新分割
                    boxes = []
                    for t in active_targets:
                        x, y, bw, bh = t.bbox
                        boxes.append([x, y, x + bw, y + bh])  # xyxy
                    try:
                        results = model.predict(frame_rgb, bboxes=boxes,
                                                imgsz=self.imgsz, verbose=False)
                        masks = None
                        if (results and len(results) > 0
                                and getattr(results[0], 'masks', None) is not None):
                            masks = results[0].masks.data
                        new_active = []
                        for i, t in enumerate(active_targets):
                            bbox, contours = (None, None)
                            if masks is not None and i < len(masks):
                                bbox, contours = self._mask_to_bbox(masks[i])
                            if bbox:
                                t.bbox = bbox
                                t.missing = 0
                                if contours:
                                    contours_map[t.target_id] = contours
                                new_active.append(t)
                            else:
                                # 单帧失败：保留上一帧 bbox，连续失败超限才判丢失
                                t.missing += 1
                                if t.missing > self.max_missing:
                                    t.status = "lost"
                                    self.log_signal.emit(
                                        f"目标 #{t.target_id} ({t.class_name}) 在帧 {frame_idx} 丢失")
                                else:
                                    new_active.append(t)
                        active_targets = new_active
                    except Exception as e:
                        self.log_signal.emit(f"追踪异常: {e}")

                annotated = self._draw(frame_rgb, active_targets, contours_map)
                self.frame_signal.emit(annotated, frame_idx, frame_idx / fps)
                self.target_update_signal.emit(
                    [{'id': t.target_id, 'bbox': t.bbox, 'status': t.status}
                     for t in self.targets])
                self.progress_signal.emit(
                    int(frame_idx / total_frames * 100) if total_frames else 0)

                if frame_idx % self.save_every_frames == 0:
                    self._save_frame(frame_idx, active_targets, w, h,
                                     labels_dir, images_dir, frame)
                    last_saved_idx = frame_idx

                frame_idx += 1

            if frame_idx > 0 and frame is not None and last_saved_idx != frame_idx - 1:
                self._save_frame(frame_idx - 1, active_targets, w, h,
                                 labels_dir, images_dir, frame)
            self._write_session(total_frames, fps, frame_idx)
            self.progress_signal.emit(100)

            if self.is_running:
                self.finished_signal.emit(True, f"追踪完成: 共处理 {frame_idx} 帧")
            else:
                self.finished_signal.emit(True, f"已停止: 处理 {frame_idx} 帧，结果已保存")
        except Exception as e:
            self.finished_signal.emit(False, str(e))
        finally:
            if cap is not None:
                cap.release()

    @staticmethod
    def _mask_to_bbox(mask):
        """mask -> ((x, y, w, h) 像素bbox, 外轮廓列表)；失败返回 (None, None)"""
        try:
            if hasattr(mask, 'cpu'):
                mask = mask.cpu().numpy()
            m = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None, None
            largest = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(largest)
            if bw < 2 or bh < 2:
                return None, None
            return (int(x), int(y), int(bw), int(bh)), [largest]
        except Exception:
            return None, None

    @staticmethod
    def _draw(frame_rgb, targets, contours_map=None):
        annotated = frame_rgb.copy()
        if contours_map:
            overlay = annotated.copy()
            for t in targets:
                cs = contours_map.get(t.target_id)
                if cs:
                    cv2.fillPoly(overlay, cs, t.color)
            cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)
        for t in targets:
            x, y, bw, bh = t.bbox
            cv2.rectangle(annotated, (int(x), int(y)), (int(x + bw), int(y + bh)),
                          t.color, 2)
            cv2.putText(annotated, f"#{t.target_id} {t.class_name}",
                        (int(x), max(12, int(y) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, t.color, 1)
        return annotated

    def _save_frame(self, frame_idx, targets, w, h, labels_dir, images_dir, frame_bgr):
        lines = []
        for t in targets:
            if t.status == "ended":
                continue
            x, y, bw, bh = t.bbox
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            lines.append(f"{t.class_index} {cx:.6f} {cy:.6f} "
                         f"{bw / w:.6f} {bh / h:.6f}")
        with open(os.path.join(labels_dir, f"frame_{frame_idx:06d}.txt"),
                  'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        cv2.imwrite(os.path.join(images_dir, f"frame_{frame_idx:06d}.jpg"), frame_bgr)

    def _write_session(self, total_frames, fps, processed):
        session = {
            'video_path': self.video_path,
            'output_dir': self.output_dir,
            'sam_model': self.sam_model_path,
            'total_frames': total_frames,
            'fps': fps,
            'frames_processed': processed,
            'targets': [{'id': t.target_id, 'class_name': t.class_name,
                         'class_index': t.class_index, 'bbox': list(t.bbox),
                         'status': t.status, 'frame_started': t.frame_started}
                        for t in self.targets],
        }
        with open(os.path.join(self.output_dir, 'session.json'),
                  'w', encoding='utf-8') as f:
            json.dump(session, f, indent=2, ensure_ascii=False)

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def reframe(self, target_id, bbox):
        """UI线程投递：为某目标重新框选并重新激活（bbox=(x,y,w,h) 像素）"""
        with self._reframe_lock:
            self._reframes.append((target_id, tuple(int(v) for v in bbox)))

    def add_target(self, target):
        """UI线程投递：追踪进行中新增追踪目标，下一帧起参与分割追踪"""
        with self._reframe_lock:
            self._new_targets.append(target)

    def stop(self):
        self.is_running = False
        self.is_paused = False


class VideoCanvas(QLabel):
    """视频画布：显示首帧/追踪帧，支持拖拽框选目标"""
    box_selected = pyqtSignal(tuple)  # (x1, y1, x2, y2) 图像坐标

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(400)
        self.base_pixmap = None      # 首帧原图
        self.targets: List[TrackTarget] = []
        self._drawing = False
        self._start = None
        self._end = None

    def set_first_frame(self, pixmap):
        self.base_pixmap = pixmap
        self._redraw()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.base_pixmap:
            self._redraw()

    def _scaled(self):
        return self.base_pixmap.scaled(self.size(), Qt.KeepAspectRatio,
                                       Qt.SmoothTransformation)

    def _redraw(self):
        if not self.base_pixmap:
            return
        scaled = self._scaled()
        sx = scaled.width() / self.base_pixmap.width()
        sy = scaled.height() / self.base_pixmap.height()
        painter = QPainter(scaled)
        painter.setRenderHint(QPainter.Antialiasing)
        for t in self.targets:
            if t.status == "ended":
                continue
            color = QColor(*t.color)
            pen = QPen(color, 3 if t.status == "active" else 1,
                       Qt.SolidLine if t.status == "active" else Qt.DashLine)
            painter.setPen(pen)
            fill = QColor(color)
            fill.setAlpha(50)
            painter.setBrush(fill)
            x, y, w, h = t.bbox
            painter.drawRect(QRectF(x * sx, y * sy, w * sx, h * sy))
            painter.setFont(QFont("微软雅黑", 9))
            painter.setPen(QPen(QColor("white")))
            painter.setBrush(Qt.NoBrush)
            painter.drawText(int(x * sx) + 3, int(y * sy) + 14,
                             f"#{t.target_id} {t.class_name}")
        if self._drawing and self._start and self._end:
            painter.setPen(QPen(QColor("#ecf0f1"), 1, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            x1, y1 = self._scaled_to_image(self._start)
            x2, y2 = self._scaled_to_image(self._end)
            # 图像坐标 -> 缩放图绘制坐标
            painter.drawRect(QRectF(min(x1, x2) * sx, min(y1, y2) * sy,
                                    abs(x2 - x1) * sx, abs(y2 - y1) * sy))
        painter.end()
        self.setPixmap(scaled)

    def _widget_to_scaled_pos(self, pos):
        scaled = self._scaled()
        ox = (self.width() - scaled.width()) / 2
        oy = (self.height() - scaled.height()) / 2
        return pos.x() - ox, pos.y() - oy

    def _in_image(self, pos):
        if not self.base_pixmap:
            return False
        sx, sy = self._widget_to_scaled_pos(pos)
        scaled = self._scaled()
        return 0 <= sx <= scaled.width() and 0 <= sy <= scaled.height()

    def _scaled_to_image(self, pos):
        sx, sy = self._widget_to_scaled_pos(pos)
        scaled = self._scaled()
        kx = self.base_pixmap.width() / scaled.width()
        ky = self.base_pixmap.height() / scaled.height()
        return sx * kx, sy * ky

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.base_pixmap and self._in_image(event.pos()):
            self._drawing = True
            self._start = event.pos()
            self._end = event.pos()

    def mouseMoveEvent(self, event):
        if self._drawing:
            self._end = event.pos()
            self._redraw()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or not self._drawing:
            return
        self._drawing = False
        if self._start and self._end:
            x1, y1 = self._scaled_to_image(self._start)
            x2, y2 = self._scaled_to_image(self._end)
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                self.box_selected.emit((min(x1, x2), min(y1, y2),
                                        max(x1, x2), max(y1, y2)))
        self._start = self._end = None
        self._redraw()


class VideoTrackingPage(QWidget):
    """SAM2视频追踪标注页"""

    def __init__(self):
        super().__init__()
        self.video_path = ""
        self.output_dir = ""
        self.targets: List[TrackTarget] = []
        self.target_counter = 0
        self.worker = None
        self._reframe_target = None   # 待重新框选的目标（追踪中交互）
        self._init_ui()
        self._bind_events()
        self._scan_models()

    def _section_row(self, icon, title):
        """卡片分节标题行（图标 + 加粗标题），与 YOLO训练 页一致"""
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel(icon))
        row.addWidget(theme.section(title))
        row.addStretch()
        return row

    def _init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(15, 15, 15, 15)
        outer.setSpacing(12)
        outer.addWidget(theme.header("🎬", "SAM视频追踪",
                                     "首帧画框一次，SAM2 逐帧自动追踪，按帧导出图片与 YOLO 标签"))

        # 顶部运行控制行（彩色按钮，参照 YOLO训练 页）
        run_row = QHBoxLayout()
        run_row.setSpacing(12)
        self.start_btn = QPushButton("🚀 开始追踪")
        theme.set_role(self.start_btn, "success")
        self.pause_btn = QPushButton("⏸ 暂停")
        theme.set_role(self.pause_btn, "primary")
        self.pause_btn.setEnabled(False)
        self.stop_btn = QPushButton("⏹ 停止")
        theme.set_role(self.stop_btn, "danger")
        self.stop_btn.setEnabled(False)
        run_row.addWidget(self.start_btn)
        run_row.addWidget(self.pause_btn)
        run_row.addWidget(self.stop_btn)
        run_row.addStretch()
        self.frame_info_label = QLabel("帧号: -  时间: -")
        theme.set_role(self.frame_info_label, "hint")
        run_row.addWidget(self.frame_info_label)
        outer.addLayout(run_row)

        main_layout = QHBoxLayout()
        main_layout.setSpacing(12)
        outer.addLayout(main_layout, 1)

        # 左侧设置面板（卡片 + 分节，滚动容器防裁切）
        panel = QFrame()
        theme.set_role(panel, "card")
        panel.setMinimumWidth(284)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(14, 14, 14, 14)
        pl.setSpacing(10)
        panel_scroll = QScrollArea()
        panel_scroll.setWidget(panel)
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setFixedWidth(300)
        panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        def add_btn(layout, text, role="primary"):
            btn = QPushButton(text)
            if role:
                theme.set_role(btn, role)
            layout.addWidget(btn)
            return btn

        pl.addLayout(self._section_row("🎬", "视频与输出"))
        self.video_btn = add_btn(pl, "🎬 选择视频")
        self.video_name_label = QLabel("未选择")
        theme.set_role(self.video_name_label, "hint")
        self.video_name_label.setWordWrap(True)
        pl.addWidget(self.video_name_label)
        self.output_btn = add_btn(pl, "📂 选择输出目录")
        self.output_name_label = QLabel("默认: 视频同目录 <视频名>_tracking")
        theme.set_role(self.output_name_label, "hint")
        self.output_name_label.setWordWrap(True)
        pl.addWidget(self.output_name_label)
        self.load_frame_btn = add_btn(pl, "🖼 重新读取视频")

        pl.addLayout(self._section_row("🤖", "模型与参数"))
        pl.addWidget(QLabel("SAM2模型:"))
        self.model_combo = QComboBox()
        pl.addWidget(self.model_combo)
        interval_row = QHBoxLayout()
        interval_row.addWidget(QLabel("保存间隔: 每"))
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 100000)
        self.interval_spin.setValue(30)
        interval_row.addWidget(self.interval_spin)
        interval_row.addWidget(QLabel("帧"))
        pl.addLayout(interval_row)

        pl.addLayout(self._section_row("🎯", "追踪目标"))
        target_hint = QLabel("在画面上拖拽画框添加目标，选中目标后可在下方面板管理")
        theme.set_role(target_hint, "hint")
        target_hint.setWordWrap(True)
        pl.addWidget(target_hint)
        self.object_list = QListWidget()
        self.object_list.setMinimumHeight(100)
        pl.addWidget(self.object_list)
        self.edit_class_btn = add_btn(pl, "✏️ 修改选中类别")
        self.reframe_btn = add_btn(pl, "🎯 重新框选选中目标(需暂停)")
        self.end_btn = add_btn(pl, "⏹ 结束选中目标", role=None)
        self.clear_targets_btn = add_btn(pl, "🧹 清空所有目标", role=None)

        pl.addStretch()
        self.open_output_btn = add_btn(pl, "📂 打开输出目录", role="amber")
        main_layout.addWidget(panel_scroll)

        # 右侧内容区：状态 + 进度 + 画布 + 时间轴
        content = QVBoxLayout()
        content.setSpacing(8)
        status_row = QHBoxLayout()
        self.status_label = QLabel("请选择视频，将自动读取首帧")
        theme.set_role(self.status_label, "status")
        status_row.addWidget(self.status_label)
        status_row.addStretch()
        content.addLayout(status_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        content.addWidget(self.progress_bar)

        self.canvas = VideoCanvas()
        theme.set_role(self.canvas, "canvas")
        content.addWidget(self.canvas, 1)

        self.timeline_list = QListWidget()
        self.timeline_list.setAlternatingRowColors(True)
        self.timeline_list.setMaximumHeight(120)
        content.addWidget(self.timeline_list)
        main_layout.addLayout(content, 1)

    def _bind_events(self):
        self.video_btn.clicked.connect(self.browse_video)
        self.output_btn.clicked.connect(self.browse_output)
        self.load_frame_btn.clicked.connect(self.load_first_frame)
        self.start_btn.clicked.connect(self.start_tracking)
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.stop_btn.clicked.connect(self.stop_tracking)
        self.open_output_btn.clicked.connect(self.open_output)
        self.edit_class_btn.clicked.connect(self.edit_target_class)
        self.reframe_btn.clicked.connect(self.start_reframe)
        self.end_btn.clicked.connect(lambda: self._set_target_status("ended"))
        self.clear_targets_btn.clicked.connect(self.clear_targets)
        self.canvas.box_selected.connect(self.on_box_drawn)

    def _scan_models(self):
        models = scan_sam_models()
        self.model_combo.clear()
        self.model_combo.addItems(models)

    # ---------- 文件与首帧 ----------
    def browse_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "",
                                              "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv)")
        if not path:
            return
        self.video_path = path
        self.video_name_label.setText(os.path.basename(path))
        if not self.output_dir:
            self.output_dir = os.path.splitext(path)[0] + "_tracking"
        self.output_name_label.setText(self.output_dir)
        self.load_first_frame()

    def browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if d:
            self.output_dir = d
            self.output_name_label.setText(d)

    def load_first_frame(self):
        if not self.video_path:
            QMessageBox.warning(self, "提示", "请先选择视频")
            return
        if not cv2_available:
            QMessageBox.warning(self, "提示", "需要安装 opencv-python")
            return
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            QMessageBox.warning(self, "错误", "无法读取视频首帧")
            return
        self.canvas.targets = self.targets
        self.canvas.set_first_frame(self._frame_to_pixmap(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self._set_status("请在右侧画面上拖拽画框添加追踪目标")

    @staticmethod
    def _frame_to_pixmap(frame_rgb):
        h, w = frame_rgb.shape[:2]
        qimg = QImage(np.ascontiguousarray(frame_rgb).data, w, h, 3 * w,
                      QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    # ---------- 目标管理 ----------
    def on_box_drawn(self, box):
        x1, y1, x2, y2 = box
        if self._reframe_target is not None:
            t = self._reframe_target
            self._reframe_target = None
            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            t.bbox = bbox
            t.missing = 0
            if self.worker and self.worker.isRunning():
                self.worker.reframe(t.target_id, bbox)
                if self.worker.is_paused:
                    self.worker.resume()
                    self.pause_btn.setText("⏸ 暂停")
            else:
                t.status = "active"
            self._refresh_object_list()
            self.canvas.targets = self.targets
            self.canvas._redraw()
            self._set_status(f"目标 #{t.target_id} 已重新框选，继续追踪")
            return
        name, ok = QInputDialog.getText(self, "设置类别", "类别名称:")
        if not ok or not name.strip():
            return
        name = name.strip()
        existing = next((t.class_index for t in self.targets
                         if t.class_name == name), None)
        class_index = existing if existing is not None else len(
            {t.class_name for t in self.targets})
        self.target_counter += 1
        target = TrackTarget(
            target_id=self.target_counter,
            class_name=name,
            class_index=class_index,
            bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
            color=TARGET_COLORS[(self.target_counter - 1) % len(TARGET_COLORS)],
        )
        self.targets.append(target)
        if self.worker and self.worker.isRunning():
            self.worker.add_target(target)
            self._set_status(f"目标 #{target.target_id} ({name}) 已加入追踪，下一帧生效")
        self._refresh_object_list()
        self.canvas.targets = self.targets
        self.canvas._redraw()

    def _refresh_object_list(self):
        self.object_list.clear()
        for t in self.targets:
            status_text = {"active": "追踪中", "lost": "已丢失",
                           "ended": "已结束"}.get(t.status, t.status)
            self.object_list.addItem(
                QListWidgetItem(f"#{t.target_id} {t.class_name} [{status_text}]"))

    def edit_target_class(self):
        row = self.object_list.currentRow()
        if not (0 <= row < len(self.targets)):
            QMessageBox.warning(self, "提示", "请先选择目标")
            return
        new_name, ok = QInputDialog.getText(self, "修改类别", "新类别名称:",
                                            text=self.targets[row].class_name)
        if ok and new_name.strip():
            self.targets[row].class_name = new_name.strip()
            self._refresh_object_list()
            self.canvas._redraw()

    def start_reframe(self):
        """进入重新框选模式：需追踪运行中且已暂停，选中目标后到画布上拖框"""
        if not (self.worker and self.worker.isRunning()):
            QMessageBox.warning(self, "提示", "请先开始追踪（此功能用于追踪过程中目标丢失后补框）")
            return
        if not self.worker.is_paused:
            QMessageBox.information(self, "提示", "请先点击【暂停】，在暂停画面上重新框选目标")
            return
        row = self.object_list.currentRow()
        if not (0 <= row < len(self.targets)):
            QMessageBox.warning(self, "提示", "请先在列表中选中要补框的目标")
            return
        t = self.targets[row]
        self._reframe_target = t
        self._set_status(f"请在画面上拖框重新框选目标 #{t.target_id} ({t.class_name})")

    def _set_target_status(self, status):
        row = self.object_list.currentRow()
        if 0 <= row < len(self.targets):
            self.targets[row].status = status
            self._refresh_object_list()
            self.canvas.targets = self.targets
            self.canvas._redraw()

    def clear_targets(self):
        """一键清空所有追踪目标（追踪运行中需先停止）"""
        if not self.targets:
            self._set_status("当前没有可清空的目标")
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "提示", "请先停止追踪后再清空目标")
            return
        if QMessageBox.question(self, "确认清空",
                                f"确定要清空全部 {len(self.targets)} 个追踪目标吗？"
                                ) != QMessageBox.Yes:
            return
        self.targets = []
        self._reframe_target = None
        self.target_counter = 0
        self._refresh_object_list()
        self.canvas.targets = self.targets
        self.canvas._redraw()
        self._set_status("已清空所有追踪目标")

    # ---------- 追踪控制 ----------
    def start_tracking(self):
        if not self.video_path:
            QMessageBox.warning(self, "提示", "请先选择视频")
            return
        if not self.output_dir:
            QMessageBox.warning(self, "提示", "请先设置输出目录")
            return
        if not self.targets:
            QMessageBox.warning(self, "提示", "请先在首帧上框选追踪目标")
            return
        model_name = self.model_combo.currentText()
        model_path = find_sam_model_path(model_name) if model_name else ""
        if not model_path:
            QMessageBox.warning(self, "提示", "未找到SAM2模型，请将 sam2_*.pt 放入 sam/ 目录")
            return

        settings = load_sam_settings()
        imgsz = resolve_sam_imgsz(settings.get('video', {}).get('imgsz'))
        for t in self.targets:
            if t.status == "lost":
                t.status = "active"

        self.worker = VideoTrackWorker(self.video_path, self.output_dir,
                                       model_path, list(self.targets),
                                       self.interval_spin.value(), imgsz)
        self.worker.log_signal.connect(self._log_event)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.frame_signal.connect(self._on_worker_frame)
        self.worker.target_update_signal.connect(self._on_target_update)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.pause_btn.setText("⏸ 暂停")
        # 追踪帧由工作线程直接绘制标注，画布不再叠加目标框
        self.canvas.targets = []
        self._log_event("开始追踪...")

    def _on_worker_frame(self, frame_rgb, idx, ts):
        self.canvas.set_first_frame(self._frame_to_pixmap(frame_rgb))
        self.frame_info_label.setText(f"帧号: {idx}  时间: {ts:.2f}s")

    def _on_target_update(self, dicts):
        changed = {d['id']: d for d in dicts}
        for t in self.targets:
            d = changed.get(t.target_id)
            if d:
                t.bbox = d['bbox']
                t.status = d['status']
        self._refresh_object_list()

    def toggle_pause(self):
        if not (self.worker and self.worker.isRunning()):
            return
        if self.worker.is_paused:
            self.worker.resume()
            self.pause_btn.setText("⏸ 暂停")
        else:
            self.worker.pause()
            self.pause_btn.setText("▶ 继续")

    def stop_tracking(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()

    def _on_finished(self, success, msg):
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setText("⏸ 暂停")
        self._log_event(msg)
        self._set_status(("✅ " if success else "❌ ") + msg)

    def _log_event(self, text):
        self.timeline_list.addItem(str(text))
        self.timeline_list.scrollToBottom()

    def open_output(self):
        if self.output_dir and os.path.isdir(self.output_dir):
            if sys.platform == 'win32':
                os.startfile(self.output_dir)
            else:
                import subprocess
                subprocess.Popen(['xdg-open', self.output_dir])

    def _set_status(self, msg):
        self.status_label.setText(str(msg))

    def log(self, msg):
        self._set_status(msg)
