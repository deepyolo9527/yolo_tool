
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit, QPushButton,
                             QHBoxLayout, QVBoxLayout, QTextEdit,
                             QMessageBox, QFileDialog, QDoubleSpinBox, QProgressBar,
                             QCheckBox)
from PyQt5.QtCore import Qt, QMetaObject, Q_ARG
from pages.base_page import BasePage
from utils import theme
from utils.preview import PreviewLabel, bgr_to_pixmap
from utils.path_helpers import get_app_dir, get_weights_dir
from workers.inference_worker import ImageInferenceWorker
import os


project_root = get_app_dir()


class ImageInferencePage(BasePage):
    MAX_PAIRS = 50   # 预览保留的原图/结果对上限，避免批量推理时内存无限增长

    def __init__(self):
        super().__init__()
        self.worker = None
        self.pairs = []      # [(图片路径, 原图QPixmap, 结果QPixmap)]
        self.pair_index = -1
        self.init_ui()

    def init_ui(self):
        layout = self.get_content_layout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)

        layout.addWidget(theme.header("🖼️", "图片推理", "使用训练好的YOLO模型对图片进行目标检测"))

        layout.addWidget(self._build_params_card())
        layout.addWidget(self._build_run_card())
        layout.addWidget(self._build_view_card(), 3)
        layout.addWidget(self._build_log_card(), 1)

    # ---------- 参数区（两行紧凑排布） ----------
    def _build_params_card(self):
        card, card_layout = theme.card()
        card_layout.addLayout(theme.section_row("⚙️", "参数配置"))

        grid = theme.grid(6)

        # 第一行：三个路径
        self.model_edit = QLineEdit(os.path.join(get_weights_dir(), "yolo26n.pt"))
        grid.addWidget(theme.field("模型路径:", theme.path_row(
            self.model_edit, "",
            lambda: self._select_file(self.model_edit, "模型文件 (*.pt)"))),
            0, 0, 1, 2)

        self.image_edit = QLineEdit()
        grid.addWidget(theme.field("图片/目录:", theme.path_row(
            self.image_edit, "请选择图片文件或目录", self.browse_images, "🖼️")),
            0, 2, 1, 2)

        self.output_edit = QLineEdit(os.path.join(project_root, "runs", "detect"))
        grid.addWidget(theme.field("输出目录:", theme.path_row(
            self.output_edit, "", self.browse_output_dir, "📁", amber=True)),
            0, 4, 1, 2)

        # 第二行：阈值 / 设备 / 类别 / 保存选项
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

        self.device_edit = QLineEdit("0")
        grid.addWidget(theme.field("设备:", self.device_edit), 1, 2)

        self.classes_edit = QLineEdit()
        self.classes_edit.setPlaceholderText("如: 0,1 或 person,car")
        grid.addWidget(theme.field("类别:", self.classes_edit), 1, 3)

        opts = QWidget()
        opts_layout = QHBoxLayout(opts)
        opts_layout.setContentsMargins(0, 0, 0, 0)
        opts_layout.setSpacing(14)
        self.save_txt_check = QCheckBox("保存TXT")
        opts_layout.addWidget(self.save_txt_check)
        self.save_image_check = QCheckBox("保存图像")
        self.save_image_check.setChecked(True)
        opts_layout.addWidget(self.save_image_check)
        opts_layout.addStretch()
        grid.addWidget(opts, 1, 4, 1, 2)

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
        card_layout.addLayout(row)
        return card

    # ---------- 可视化（左原图 / 右检测结果） ----------
    def _build_view_card(self):
        card, card_layout = theme.card()

        header = QHBoxLayout()
        header.addWidget(QLabel("👁️"))
        header.addWidget(theme.section("推理可视化"))
        header.addStretch()

        self.prev_btn = QPushButton("◀")
        theme.set_role(self.prev_btn, "primary")
        self.prev_btn.setFixedWidth(46)
        self.prev_btn.setToolTip("上一张")
        self.prev_btn.clicked.connect(self._show_prev)
        header.addWidget(self.prev_btn)

        self.view_info = QLabel("0 / 0")
        theme.set_role(self.view_info, "status")
        header.addWidget(self.view_info)

        self.next_btn = QPushButton("▶")
        theme.set_role(self.next_btn, "primary")
        self.next_btn.setFixedWidth(46)
        self.next_btn.setToolTip("下一张")
        self.next_btn.clicked.connect(self._show_next)
        header.addWidget(self.next_btn)

        card_layout.addLayout(header)

        views = QHBoxLayout()
        views.setSpacing(12)

        left = QVBoxLayout()
        left_title = QLabel("原图")
        theme.set_role(left_title, "hint")
        left.addWidget(left_title)
        self.orig_view = PreviewLabel("尚未推理，画面将显示在此处")
        left.addWidget(self.orig_view, 1)
        views.addLayout(left, 1)

        right = QVBoxLayout()
        right_title = QLabel("检测结果")
        theme.set_role(right_title, "hint")
        right.addWidget(right_title)
        self.result_view = PreviewLabel("推理完成后显示带检测框的画面")
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

    # ---------- 预览切换 ----------
    def _on_preview(self, img_path, orig_bgr, ann_bgr):
        self.pairs.append((img_path, bgr_to_pixmap(orig_bgr), bgr_to_pixmap(ann_bgr)))
        if len(self.pairs) > self.MAX_PAIRS:
            self.pairs.pop(0)
        self._show_index(len(self.pairs) - 1)

    def _show_index(self, idx):
        if not self.pairs:
            return
        self.pair_index = max(0, min(idx, len(self.pairs) - 1))
        img_path, orig_pix, ann_pix = self.pairs[self.pair_index]
        self.orig_view.set_pixmap(orig_pix)
        self.result_view.set_pixmap(ann_pix)
        self.view_info.setText(f"{self.pair_index + 1} / {len(self.pairs)}")
        self.view_info.setToolTip(os.path.basename(img_path))

    def _show_prev(self):
        self._show_index(self.pair_index - 1)

    def _show_next(self):
        self._show_index(self.pair_index + 1)

    def _clear_views(self):
        self.pairs = []
        self.pair_index = -1
        self.orig_view.clear_image()
        self.result_view.clear_image()
        self.view_info.setText("0 / 0")

    # ---------- 文件选择 ----------
    def _select_file(self, line_edit, filter_str):
        file_path = QFileDialog.getOpenFileName(self, "选择文件", "", filter_str)[0]
        if file_path:
            line_edit.setText(file_path)

    def browse_images(self):
        file_paths = QFileDialog.getOpenFileNames(self, "选择图片文件", "", "图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff)")[0]
        if file_paths:
            if len(file_paths) == 1:
                self.image_edit.setText(file_paths[0])
            else:
                self.image_edit.setText(";".join(file_paths))

    def browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.output_edit.setText(dir_path)

    def log(self, msg):
        QMetaObject.invokeMethod(self.log_text, "append", Qt.QueuedConnection,
                                Q_ARG(str, msg))

    def _set_progress(self, current, total):
        if total > 0:
            progress = int((current / total) * 100)
            QMetaObject.invokeMethod(self.progress_bar, "setValue", Qt.QueuedConnection,
                                    Q_ARG(int, progress))

    def start_inference(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "提示", "推理任务正在运行中")
            return

        model_path = self.model_edit.text().strip()
        if not model_path:
            QMessageBox.warning(self, "提示", "请选择模型文件")
            return

        image_input = self.image_edit.text().strip()
        if not image_input:
            QMessageBox.warning(self, "提示", "请选择图片文件")
            return

        image_paths = []
        if ';' in image_input:
            for p in image_input.split(';'):
                p = p.strip()
                if p and os.path.exists(p):
                    image_paths.append(p)
        elif os.path.isfile(image_input):
            image_paths.append(image_input)
        elif os.path.isdir(image_input):
            img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            for f in os.listdir(image_input):
                ext = os.path.splitext(f)[1].lower()
                if ext in img_exts:
                    image_paths.append(os.path.join(image_input, f))

        if not image_paths:
            QMessageBox.warning(self, "提示", "未找到有效的图片文件")
            return

        classes_text = self.classes_edit.text().strip()
        classes = []
        if classes_text:
            try:
                classes = [int(c.strip()) for c in classes_text.split(',')]
            except ValueError:
                classes = [c.strip() for c in classes_text.split(',')]

        params = {
            'model': model_path,
            'images': image_paths,
            'output_dir': self.output_edit.text().strip(),
            'conf': self.conf_spin.value(),
            'iou': self.iou_spin.value(),
            'device': self.device_edit.text().strip(),
            'classes': classes if classes else None,
            'save_txt': self.save_txt_check.isChecked(),
            'save_image': self.save_image_check.isChecked(),
        }

        self.progress_bar.setValue(0)
        self._clear_views()
        self.worker = ImageInferenceWorker(params)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_inference_finished)
        self.worker.progress_signal.connect(self._set_progress)
        self.worker.preview_signal.connect(self._on_preview)
        self.worker.start()
        self.infer_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log(f"开始推理 {len(image_paths)} 张图片...")

    def stop_inference(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.log("正在停止推理...")

    def on_inference_finished(self, success, msg):
        self.infer_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100 if success else 0)
        if success:
            self.log(f"[完成] {msg}")
        else:
            self.log(f"[错误] {msg}")
