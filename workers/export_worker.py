
# -*- coding: utf-8 -*-

from PyQt5.QtCore import QThread, pyqtSignal
import os
import sys


class ModelExportWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self._stop_flag = False
        self.model = None

    def stop(self):
        self._stop_flag = True

    def run(self):
        try:
            try:
                from ultralytics import YOLO
            except ImportError:
                import importlib
                ultralytics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ultralytics')
                sys.path.insert(0, ultralytics_path)
                if 'ultralytics' in sys.modules:
                    del sys.modules['ultralytics']
                from ultralytics import YOLO

            model_path = self.params.get('model', '')
            export_format = self.params.get('format', 'onnx')
            output_dir = self.params.get('output_dir', 'runs/export')
            imgsz = self.params.get('imgsz', 640)
            device = self.params.get('device', 'cpu')

            if not model_path or not os.path.exists(model_path):
                raise Exception(f"模型文件不存在: {model_path}")

            self.model = YOLO(model_path)
            self._log(f"加载模型: {model_path}")

            os.makedirs(output_dir, exist_ok=True)

            self._log(f"开始导出模型为 {export_format} 格式...")

            formats = {
                'onnx': 'ONNX',
                'torchscript': 'TorchScript',
                'tflite': 'TensorFlow Lite',
                'tensorrt': 'TensorRT',
                'openvino': 'OpenVINO',
                'coreml': 'Core ML',
                'engine': 'TensorRT Engine',
                'pt': 'PyTorch',
            }

            format_name = formats.get(export_format, export_format)
            self._log(f"目标格式: {format_name}")

            results = self.model.export(
                format=export_format,
                imgsz=imgsz,
                device=device,
                project=output_dir,
                name=f'{os.path.splitext(os.path.basename(model_path))[0]}_{export_format}',
                exist_ok=True
            )

            if results:
                self._log(f"模型导出成功！")
                self._log(f"导出文件: {results}")
                self.finished_signal.emit(True, f"模型已导出为 {format_name} 格式: {results}")
            else:
                self._log("模型导出完成")
                self.finished_signal.emit(True, f"模型已导出为 {format_name} 格式")

        except Exception as e:
            if self._stop_flag:
                self.finished_signal.emit(False, "导出已停止")
            else:
                import traceback
                self._log(f"错误详情: {traceback.format_exc()}")
                self.finished_signal.emit(False, str(e))

    def _log(self, msg):
        self.log_signal.emit(msg)
