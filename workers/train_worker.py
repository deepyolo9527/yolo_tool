
# -*- coding: utf-8 -*-

import contextlib
import io
import logging
import os
import re
import sys
import traceback

from PyQt5.QtCore import QThread, pyqtSignal


# 终端色码（ultralytics 用 \x1b[1m 加粗、\x1b[0m 复位，tqdm 用 \x1b[2K 清行）。
# 终端会解释它们，但日志框是纯文本，原样透传会显示成 ^[[1m 这样的乱码
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


class _LinePublisher:
    """把终端文本流按行整理后转发出去。

    tqdm 每刷新一次就写一段带 \\r 的内容，逐条转发的话日志会被进度条中间态刷成上千行；
    这里把 \\r 当作“原地重写当前行”，只在换行时提交，日志里看到的就是终端的最终画面。
    """

    def __init__(self, forward):
        self._forward = forward
        self._buf = ""

    def write(self, text):
        if not text:
            return 0
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._commit(line)
        return len(text)

    def flush(self):
        if self._buf:
            line, self._buf = self._buf, ""
            self._commit(line)

    def write_line(self, text):
        """写入一条独立日志：先提交尚未换行的残行，免得和进度条拼成一行"""
        self.flush()
        self._commit(text)

    def _commit(self, line):
        line = _ANSI_RE.sub("", line.rsplit("\r", 1)[-1]).rstrip()
        if line.strip():
            self._forward(line)


class _TeeTextStream(io.TextIOBase):
    """替换 sys.stdout / sys.stderr：原样写回终端，同时交给发布器整理成日志行"""

    def __init__(self, original, publisher):
        super().__init__()
        self._original = original
        self._publisher = publisher
        self._encoding = getattr(original, "encoding", None) or "utf-8"

    @property
    def encoding(self):
        return self._encoding

    def writable(self):
        return True

    def isatty(self):
        return False

    def write(self, text):
        try:
            if self._original is not None:
                self._original.write(text)
        except Exception:
            pass
        self._publisher.write(text)
        return len(text) if text else 0

    def flush(self):
        try:
            if self._original is not None:
                self._original.flush()
        except Exception:
            pass


class _LogBridge(logging.Handler):
    """接住 logging 输出的训练日志。

    ultralytics 的 StreamHandler 在模块导入时就绑定了当时的 stdout，之后再重定向
    sys.stdout 也抓不到它的输出，只能另挂一个 handler。
    """

    def __init__(self, publisher):
        super().__init__()
        self.setFormatter(logging.Formatter("%(message)s"))
        self._publisher = publisher

    def emit(self, record):
        try:
            self._publisher.write_line(self.format(record))
        except Exception:
            pass


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
        # 两条流各自一个行缓冲：共用一个缓冲区时，stdout 的半行会和 stderr 的内容拼成一行
        out_pub = _LinePublisher(self._log)
        err_pub = _LinePublisher(self._log)
        stdout_tee = _TeeTextStream(sys.stdout, out_pub)
        stderr_tee = _TeeTextStream(sys.stderr, err_pub)
        root_logger = logging.getLogger()
        bridge = _LogBridge(out_pub)
        root_logger.addHandler(bridge)

        try:
            with contextlib.ExitStack() as stack:
                stack.enter_context(contextlib.redirect_stdout(stdout_tee))
                stack.enter_context(contextlib.redirect_stderr(stderr_tee))
                self._train()
        except Exception as e:
            if self._stop_flag:
                self._log("训练已停止")
                self.finished_signal.emit(False, "训练已停止")
            else:
                self._log(f"错误详情: {traceback.format_exc()}")
                self.finished_signal.emit(False, str(e))
        finally:
            root_logger.removeHandler(bridge)
            stdout_tee.flush()
            stderr_tee.flush()
            out_pub.flush()
            err_pub.flush()

    def _train(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            import importlib
            ultralytics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ultralytics')
            sys.path.insert(0, ultralytics_path)
            if 'ultralytics' in sys.modules:
                del sys.modules['ultralytics']
            from ultralytics import YOLO

        self._log_environment()

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
            # 打包版(onefile)里每个 DataLoader 子进程都要重新解压一份程序包，
            # 8 个进程会吃掉数倍内存并残留 _MEIxxxx 目录，因此默认单进程加载
            'workers': self.params.get('workers', 0 if getattr(sys, 'frozen', False) else 8),
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
        self._log(f"轮数: {train_params['epochs']}, 批次: {train_params['batch']}, "
                  f"图像尺寸: {train_params['imgsz']}, 设备: {train_params['device']}")
        self._log("训练参数: " + ", ".join(f"{k}={v}" for k, v in sorted(train_params.items())))

        if self._stop_flag:
            self._log("训练已取消")
            self.finished_signal.emit(False, "训练已取消")
            return

        self.model.train(**train_params)

        if not self._stop_flag:
            self._log("训练完成！")
            self.finished_signal.emit(True, "训练完成")
        else:
            self._log("训练已停止")
            self.finished_signal.emit(False, "训练已停止")

    def _log_environment(self):
        """训练前打印一次运行环境，便于事后从日志定位版本/CUDA 问题"""
        self._log("=" * 60)
        self._log(f"Python {sys.version.split()[0]} | 平台 {sys.platform}")
        try:
            import torch
            if torch.cuda.is_available():
                names = ", ".join(torch.cuda.get_device_name(i)
                                  for i in range(torch.cuda.device_count()))
                self._log(f"PyTorch {torch.__version__} | GPU 可用: {names}")
            else:
                self._log(f"PyTorch {torch.__version__} | GPU 不可用，本次训练将使用 CPU")
        except Exception as e:
            self._log(f"PyTorch 导入失败: {e}")
        try:
            import ultralytics
            self._log(f"ultralytics {getattr(ultralytics, '__version__', '未知')}")
        except Exception as e:
            self._log(f"ultralytics 版本读取失败: {e}")
        self._log("=" * 60)

    def _log(self, msg):
        self.log_signal.emit(msg)
