
# -*- coding: utf-8 -*-

"""页面基类：统一滚动容器与后台任务管理，样式由 utils.theme 全局提供"""

from PyQt5.QtWidgets import QWidget, QScrollArea, QVBoxLayout, QMessageBox


class BasePage(QWidget):
    def __init__(self):
        super().__init__()
        self.thread = None
        self._content_layout = None
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.content_widget = QWidget()
        self.scroll_area.setWidget(self.content_widget)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.scroll_area)

    def get_content_layout(self):
        if self._content_layout is None:
            self._content_layout = QVBoxLayout(self.content_widget)
        return self._content_layout

    def run_task(self, func, *args, **kwargs):
        if self.thread and self.thread.isRunning():
            QMessageBox.warning(self, "提示", "任务正在运行中")
            return
        from workers.base_worker import WorkerThread
        self.thread = WorkerThread(func, *args, **kwargs)
        self.thread.log_signal.connect(self.log)
        self.thread.finished_signal.connect(self.on_task_finished)
        self.thread.start()

    def on_task_finished(self, success, msg):
        if success:
            self.log(f"[完成] {msg}")
        else:
            self.log(f"[错误] {msg}")

    def log(self, msg):
        pass
