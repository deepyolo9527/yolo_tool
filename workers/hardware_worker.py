# -*- coding: utf-8 -*-
"""硬件占用采集线程：nvidia-smi 子进程、psutil 采样和 torch 导入都要几十到几百毫秒，
放在主线程会让界面在刷新瞬间掉帧。"""

from PyQt5.QtCore import QThread, pyqtSignal

from utils import hardware


class HardwareWorker(QThread):
    result_signal = pyqtSignal(dict)

    def run(self):
        try:
            self.result_signal.emit(hardware.snapshot())
        except Exception:
            self.result_signal.emit({})
