
# -*- coding: utf-8 -*-

from PyQt5.QtCore import QThread, pyqtSignal
import os
import sys


class YOLOTrainWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    progress_signal = pyqtSignal(int, int)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self._stop_flag = False
        self.model = None

    def stop(self):
        self._stop_flag = True
        if self.model and hasattr(self.model, 'trainer'):
            self.model.trainer.stop()

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

            model_path = self.params.get('model', 'yolo11n.pt')
            yaml_path = self.params.get('yaml', '')

            if yaml_path and os.path.exists(yaml_path):
                self.model = YOLO(yaml_path).load(model_path)
                self._log(f"从YAML构建模型并加载权重: {yaml_path} + {model_path}")
            else:
                self.model = YOLO(model_path)
                self._log(f"加载预训练模型: {model_path}")

            data_yaml = self.params.get('data', '')
            if not data_yaml or not os.path.exists(data_yaml):
                raise Exception(f"数据集配置文件不存在: {data_yaml}")

            train_params = {
                'data': data_yaml,
                'epochs': self.params.get('epochs', 100),
                'batch': self.params.get('batch', 16),
                'imgsz': self.params.get('imgsz', 640),
                'device': self.params.get('device', '0'),
                'project': self.params.get('project', 'runs/train'),
                'name': self.params.get('name', 'exp'),
                'patience': self.params.get('patience', 50),
                'save': self.params.get('save', True),
                'save_period': self.params.get('save_period', -1),
                'cache': self.params.get('cache', False),
                'exist_ok': self.params.get('exist_ok', False),
                'pretrained': self.params.get('pretrained', False),
                'optimizer': self.params.get('optimizer', 'SGD'),
                'verbose': self.params.get('verbose', True),
                'seed': self.params.get('seed', 0),
                'deterministic': self.params.get('deterministic', True),
                'single_cls': self.params.get('single_cls', False),
                'rect': self.params.get('rect', False),
                'cos_lr': self.params.get('cos_lr', False),
                'close_mosaic': self.params.get('close_mosaic', 10),
                'resume': self.params.get('resume', False),
                'amp': self.params.get('amp', True),
                'fraction': self.params.get('fraction', 1.0),
                'val': self.params.get('val', True),
                'split': self.params.get('split', 'val'),
                'save_json': self.params.get('save_json', False),
                'save_hybrid': self.params.get('save_hybrid', False),
                'iou': self.params.get('iou', 0.7),
                'max_det': self.params.get('max_det', 300),
                'half': self.params.get('half', False),
                'dnn': self.params.get('dnn', False),
                'plots': self.params.get('plots', True),
                'lr0': self.params.get('lr0', 0.01),
                'lrf': self.params.get('lrf', 0.01),
                'momentum': self.params.get('momentum', 0.937),
                'hsv_v': self.params.get('hsv_v', 0.4),
                'degrees': self.params.get('degrees', 0.1),
                'translate': self.params.get('translate', 0.0),
                'scale': self.params.get('scale', 0.5),
                'shear': self.params.get('shear', 0.0),
                'perspective': self.params.get('perspective', 0.0),
                'flipud': self.params.get('flipud', 0.0),
                'fliplr': self.params.get('fliplr', 0.5),
                'mosaic': self.params.get('mosaic', 1.0),
                'mixup': self.params.get('mixup', 0.0),
                'copy_paste': self.params.get('copy_paste', 0.2),
                'erasing': self.params.get('erasing', 0.4),
                'crop_fraction': self.params.get('crop_fraction', 1.0),
            }

            if self.params.get('auto_augment', ''):
                train_params['auto_augment'] = self.params['auto_augment']

            self._log("开始训练...")
            self._log(f"数据集: {data_yaml}")
            self._log(f"轮数: {train_params['epochs']}, 批次: {train_params['batch']}, 图像尺寸: {train_params['imgsz']}")

            if self._stop_flag:
                self._log("训练已取消")
                self.finished_signal.emit(False, "训练已取消")
                return

            results = self.model.train(**train_params)

            if not self._stop_flag:
                self._log("训练完成！")
                self.finished_signal.emit(True, "训练完成")
            else:
                self._log("训练已停止")
                self.finished_signal.emit(False, "训练已停止")
        except Exception as e:
            if self._stop_flag:
                self._log("训练已停止")
                self.finished_signal.emit(False, "训练已停止")
            else:
                self._log(f"错误详情: {traceback.format_exc()}")
                self.finished_signal.emit(False, str(e))

    def _log(self, msg):
        self.log_signal.emit(msg)
