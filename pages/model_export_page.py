
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (QLabel, QLineEdit, QPushButton, QHBoxLayout,
                             QTextEdit, QMessageBox, QFileDialog,
                             QComboBox, QSpinBox)
from PyQt5.QtCore import Qt, QMetaObject, Q_ARG
from pages.base_page import BasePage
from utils import theme
from utils.path_helpers import get_app_dir
from workers.export_worker import ModelExportWorker
import os


project_root = get_app_dir()


class ModelExportPage(BasePage):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.init_ui()

    def init_ui(self):
        layout = self.get_content_layout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)

        layout.addWidget(theme.header("📦", "模型导出", "将训练好的YOLO模型导出为不同格式，便于部署和推理"))

        layout.addWidget(self._build_params_card())
        layout.addWidget(self._build_run_card())
        layout.addWidget(self._build_info_card())
        layout.addWidget(self._build_log_card(), 2)

    # ---------- 参数区（两行紧凑排布） ----------
    def _build_params_card(self):
        card, card_layout = theme.card()
        card_layout.addLayout(theme.section_row("⚙️", "导出参数"))

        grid = theme.grid(6)

        self.model_edit = QLineEdit()
        grid.addWidget(theme.field("模型路径:", theme.path_row(
            self.model_edit, "请选择模型文件",
            lambda: self._select_file(self.model_edit, "模型文件 (*.pt)"))),
            0, 0, 1, 4)

        self.format_combo = QComboBox()
        self.format_combo.addItems(['onnx', 'torchscript', 'tflite', 'tensorrt',
                                    'openvino', 'coreml', 'engine', 'pt'])
        grid.addWidget(theme.field("导出格式:", self.format_combo), 0, 4, 1, 2)

        self.output_edit = QLineEdit(os.path.join(project_root, "runs", "export"))
        grid.addWidget(theme.field("输出目录:", theme.path_row(
            self.output_edit, "", self.browse_output_dir, amber=True)),
            1, 0, 1, 4)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 4096)
        self.imgsz_spin.setValue(640)
        self.imgsz_spin.setToolTip("imgsz")
        grid.addWidget(theme.field("图像尺寸:", self.imgsz_spin), 1, 4)

        self.device_edit = QLineEdit("cpu")
        self.device_edit.setToolTip("device")
        grid.addWidget(theme.field("设备:", self.device_edit), 1, 5)

        card_layout.addLayout(grid)
        return card

    # ---------- 运行控制 ----------
    def _build_run_card(self):
        card, card_layout = theme.card()
        row = QHBoxLayout()
        row.setSpacing(12)

        self.export_btn = QPushButton("🚀 开始导出")
        theme.set_role(self.export_btn, "success")
        self.export_btn.clicked.connect(self.start_export)
        row.addWidget(self.export_btn)

        self.stop_btn = QPushButton("⏹️ 停止导出")
        theme.set_role(self.stop_btn, "danger")
        self.stop_btn.clicked.connect(self.stop_export)
        self.stop_btn.setEnabled(False)
        row.addWidget(self.stop_btn)

        row.addStretch()
        card_layout.addLayout(row)
        return card

    def _build_info_card(self):
        card, card_layout = theme.card()
        card_layout.addLayout(theme.section_row("📋", "格式说明"))

        format_info = QLabel("""
<ul style="margin: 0; padding-left: 20px;">
<li><b>ONNX</b>: 开放神经网络交换格式，适用于跨平台部署</li>
<li><b>TorchScript</b>: PyTorch脚本格式，适合PyTorch环境部署</li>
<li><b>TensorFlow Lite</b>: 适用于移动端和嵌入式设备</li>
<li><b>TensorRT</b>: NVIDIA高性能推理引擎</li>
<li><b>OpenVINO</b>: Intel推理优化工具包</li>
<li><b>Core ML</b>: Apple设备专用格式</li>
<li><b>Engine</b>: TensorRT Engine格式，已优化的推理引擎</li>
<li><b>PT</b>: PyTorch标准格式</li>
</ul>
""")
        theme.set_role(format_info, "hint")
        card_layout.addWidget(format_info)
        return card

    def _build_log_card(self):
        card, card_layout = theme.card()
        card_layout.addLayout(theme.section_row("📋", "导出日志"))

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        theme.set_role(self.log_text, "log")
        card_layout.addWidget(self.log_text)
        return card

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

    def start_export(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "提示", "导出任务正在运行中")
            return

        model_path = self.model_edit.text().strip()
        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "提示", "请选择有效的模型文件")
            return

        params = {
            'model': model_path,
            'format': self.format_combo.currentText(),
            'output_dir': self.output_edit.text().strip(),
            'imgsz': self.imgsz_spin.value(),
            'device': self.device_edit.text().strip(),
        }

        self.worker = ModelExportWorker(params)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_export_finished)
        self.worker.start()
        self.export_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log(f"开始导出模型为 {params['format']} 格式...")

    def stop_export(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.log("正在停止导出...")

    def on_export_finished(self, success, msg):
        self.export_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if success:
            self.log(f"[完成] {msg}")
        else:
            self.log(f"[错误] {msg}")
