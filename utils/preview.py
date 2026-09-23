# -*- coding: utf-8 -*-
"""推理可视化预览控件：双列（原图 / 检测结果）显示共用"""

import numpy as np
import cv2
from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from utils.theme import C


def bgr_to_pixmap(bgr):
    """OpenCV BGR ndarray -> QPixmap"""
    rgb = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class PreviewLabel(QLabel):
    """等比自适应缩放的画面预览标签，无图时显示占位文字"""

    def __init__(self, placeholder="暂无画面", min_height=280, parent=None):
        super().__init__(placeholder, parent)
        self._placeholder = placeholder
        self._pixmap = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(min_height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {C['canvas_bg']};
                border: 1px solid {C['border']};
                border-radius: 12px;
                color: #94a3b8;
            }}
        """)
        self.setText(placeholder)

    def set_frame(self, bgr):
        self.set_pixmap(bgr_to_pixmap(bgr) if bgr is not None else None)

    def set_pixmap(self, pixmap):
        self._pixmap = pixmap
        self._refit()

    def clear_image(self):
        self._pixmap = None
        self._refit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refit()

    def _refit(self):
        if self._pixmap is None or self._pixmap.isNull():
            self.setPixmap(QPixmap())
            self.setText(self._placeholder)
            return
        self.setText("")
        self.setPixmap(self._pixmap.scaled(self.size(), Qt.KeepAspectRatio,
                                           Qt.SmoothTransformation))
