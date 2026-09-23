
# -*- coding: utf-8 -*-

from PyQt5.QtCore import QThread, pyqtSignal
import traceback


class WorkerThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, object)
    progress_signal = pyqtSignal(int, int)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        try:
            self.kwargs['_stop_event'] = self
            result = self.func(*self.args, **self.kwargs)
            if not self._stop_flag:
                self.finished_signal.emit(True, result if result else "完成")
            else:
                self.finished_signal.emit(False, "任务已停止")
        except Exception as e:
            if self._stop_flag:
                self.finished_signal.emit(False, "任务已停止")
            else:
                self.finished_signal.emit(False, str(e))
