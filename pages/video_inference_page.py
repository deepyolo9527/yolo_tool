
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit, QPushButton,
                             QHBoxLayout, QVBoxLayout, QTextEdit,
                             QMessageBox, QFileDialog, QDoubleSpinBox, QSpinBox,
                             QProgressBar, QCheckBox)
from PyQt5.QtCore import Qt, QMetaObject, Q_ARG
from pages.base_page import BasePage
from utils import theme
from utils.preview import PreviewLabel
from utils.path_helpers import get_app_dir, get_weights_dir
from workers.inference_worker import VideoInferenceWorker
import os
import sys


project_root = get_app_dir()


class VideoInferencePage(BasePage):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.output_video = ""
        self.init_ui()

    def init_ui(self):
        layout = self.get_content_layout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)

        layout.addWidget(theme.header("🎬", "视频推理", "使用训练好的YOLO模型对视频进行目标检测"))

        layout.addWidget(self._build_params_card())
        layout.addWidget(self._build_run_card())
        layout.addWidget(self._build_view_card(), 3)
        layout.addWidget(self._build_log_card(), 1)

    # ---------- 参数区（两行紧凑排布） ----------
    def _build_params_card(self):
        card, card_layout = theme.card()
        card_layout.addLayout(theme.section_row("⚙️", "参数配置"))

        grid = theme.grid(8)

        # 第一行：三个路径
        self.model_edit = QLineEdit(os.path.join(get_weights_dir(), "yolo26n.pt"))
        grid.addWidget(theme.field("模型路径:", theme.path_row(
            self.model_edit, "",
            lambda: self._select_file(self.model_edit, "模型文件 (*.pt)"))),
            0, 0, 1, 3)

        self.video_edit = QLineEdit()
        grid.addWidget(theme.field("视频文件:", theme.path_row(
            self.video_edit, "请选择视频文件",
            lambda: self._select_file(self.video_edit, "视频文件 (*.mp4 *.avi *.mov *.mkv)"),
            "🎬")),
            0, 3, 1, 3)

        self.output_edit = QLineEdit(os.path.join(project_root, "runs", "detect"))
        grid.addWidget(theme.field("输出目录:", theme.path_row(
            self.output_edit, "", self.browse_output_dir, "📁", amber=True)),
            0, 6, 1, 2)

        # 第二行：阈值 / 帧间隔 / 设备 / 类别 / 保存选项
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setValue(0.25)
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setDecimals(2)
        grid.addWidget(theme.field("置信度:", self.conf_spin), 1, 0)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setValue(0.7)
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setDecimals(2)
        grid.addWidget(theme.field("IoU:", self.iou_spin), 1, 1)

        self.frame_spin = QSpinBox()
        self.frame_spin.setValue(1)
        self.frame_spin.setRange(1, 30)
        grid.addWidget(theme.field("帧间隔:", self.frame_spin), 1, 2)

        self.device_edit = QLineEdit("0")
        grid.addWidget(theme.field("设备:", self.device_edit), 1, 3)

        self.classes_edit = QLineEdit()
        self.classes_edit.setPlaceholderText("如: 0,1 或 person,car")
        grid.addWidget(theme.field("类别:", self.classes_edit), 1, 4, 1, 2)

        opts = QWidget()
        opts_layout = QHBoxLayout(opts)
        opts_layout.setContentsMargins(0, 0, 0, 0)
        opts_layout.setSpacing(14)
        self.save_txt_check = QCheckBox("保存TXT")
        opts_layout.addWidget(self.save_txt_check)
        self.save_image_check = QCheckBox("保存视频")
        self.save_image_check.setChecked(True)
        opts_layout.addWidget(self.save_image_check)
        opts_layout.addStretch()
        grid.addWidget(opts, 1, 6, 1, 2)

        card_layout.addLayout(grid)
        return card

    # ---------- 运行控制 + 进度 ----------
    def _build_run_card(self):
        card, card_layout = theme.card()
        row = QHBoxLayout()
        row.setSpacing(12)

        self.infer_btn = QPushButton("🚀 开始推理")
        theme.set_role(self.infer_btn, "success")
        self.infer_btn.clicked.connect(self.start_inference)
        row.addWidget(self.infer_btn)

        self.stop_btn = QPushButton("⏹️ 停止")
        theme.set_role(self.stop_btn, "danger")
        self.stop_btn.clicked.connect(self.stop_inference)
        self.stop_btn.setEnabled(False)
        row.addWidget(self.stop_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        row.addWidget(self.progress_bar, 1)

        self.frame_label = QLabel("当前帧: 0")
        theme.set_role(self.frame_label, "status")
        row.addWidget(self.frame_label)

        self.open_btn = QPushButton("📂 打开输出视频")
        theme.set_role(self.open_btn, "amber")
        self.open_btn.clicked.connect(self.open_output_video)
        self.open_btn.setEnabled(False)
        row.addWidget(self.open_btn)

        card_layout.addLayout(row)
        return card

    # ---------- 可视化（左原视频 / 右检测结果） ----------
    def _build_view_card(self):
        card, card_layout = theme.card()
        card_layout.addLayout(theme.section_row("👁️", "推理可视化"))

        views = QHBoxLayout()
        views.setSpacing(12)

        left = QVBoxLayout()
        left_title = QLabel("原始画面")
        theme.set_role(left_title, "hint")
        left.addWidget(left_title)
        self.orig_view = PreviewLabel("尚未推理，原始视频帧将显示在此处")
        left.addWidget(self.orig_view, 1)
        views.addLayout(left, 1)

        right = QVBoxLayout()
        right_title = QLabel("检测结果")
        theme.set_role(right_title, "hint")
        right.addWidget(right_title)
        self.result_view = PreviewLabel("推理过程中实时显示带检测框的画面")
        right.addWidget(self.result_view, 1)
        views.addLayout(right, 1)

        card_layout.addLayout(views, 1)
        return card

    def _build_log_card(self):
        card, card_layout = theme.card()
        card_layout.addLayout(theme.section_row("📋", "推理日志"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(140)
        theme.set_role(self.log_text, "log")
        card_layout.addWidget(self.log_text)
        return card

    # ---------- 实时预览 ----------
    def _on_preview(self, orig_bgr, ann_bgr):
        self.orig_view.set_frame(orig_bgr)
        self.result_view.set_frame(ann_bgr)

    def open_output_video(self):
        if self.output_video and os.path.exists(self.output_video):
            if sys.platform == 'win32':
                os.startfile(self.output_video)
            else:
                import subprocess
                subprocess.Popen(['xdg-open', self.output_video])
        else:
            QMessageBox.information(self, "提示", "输出视频尚未生成")

    # ---------- 文件选择 ----------
    def _select_file(self, line_edit, filter_str):
        file_path = QFileDialog.getOpenFileName(self, "选择文件", "", filter_str)[0]
        if file_path:
            line_edit.setText(file_path)

    def browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.output_edit.setText(dir_path)

    def log(self, msg):
        QMetaObject.invokeMethod(self.log_text, "append", Qt.QueuedConnection,
                                Q_ARG(str, msg))

    def _set_progress(self, value, total):
        QMetaObject.invokeMethod(self.progress_bar, "setValue", Qt.QueuedConnection,
                                Q_ARG(int, value))

    def _set_frame(self, frame_num):
        QMetaObject.invokeMethod(self.frame_label, "setText", Qt.QueuedConnection,
                                Q_ARG(str, f"当前帧: {frame_num}"))

    def start_inference(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "提示", "推理任务正在运行中")
            return

        model_path = self.model_edit.text().strip()
        if not model_path:
            QMessageBox.warning(self, "提示", "请选择模型文件")
            return

        video_path = self.video_edit.text().strip()
        if not video_path or not os.path.exists(video_path):
            QMessageBox.warning(self, "提示", "请选择有效的视频文件")
            return

        classes_text = self.classes_edit.text().strip()
        classes = []
        if classes_text:
            try:
                classes = [int(c.strip()) for c in classes_text.split(',')]
            except ValueError:
                classes = [c.strip() for c in classes_text.split(',')]

        output_dir = self.output_edit.text().strip()
        params = {
            'model': model_path,
            'video': video_path,
            'output_dir': output_dir,
            'conf': self.conf_spin.value(),
            'iou': self.iou_spin.value(),
            'device': self.device_edit.text().strip(),
            'frame_interval': self.frame_spin.value(),
            'classes': classes if classes else None,
            'save_txt': self.save_txt_check.isChecked(),
            'save_image': self.save_image_check.isChecked(),
        }

        self.progress_bar.setValue(0)
        self.frame_label.setText("当前帧: 0")
        self.output_video = os.path.join(output_dir, 'video_output.mp4')
        self.open_btn.setEnabled(False)
        self.orig_view.clear_image()
        self.result_view.clear_image()

        self.worker = VideoInferenceWorker(params)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_inference_finished)
        self.worker.progress_signal.connect(self._set_progress)
        self.worker.frame_signal.connect(self._set_frame)
        self.worker.preview_signal.connect(self._on_preview)
        self.worker.start()
        self.infer_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log("开始视频推理...")

    def stop_inference(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.log("正在停止推理...")

    def on_inference_finished(self, success, msg):
        self.infer_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100 if success else 0)
        if success:
            self.open_btn.setEnabled(os.path.exists(self.output_video))
            self.log(f"[完成] {msg}")
        else:
            self.log(f"[错误] {msg}")
