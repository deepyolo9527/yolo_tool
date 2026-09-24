# -*- coding: utf-8 -*-
"""预览画布控件：推理画面预览与图表预览共用"""

import numpy as np
import cv2
from PyQt5.QtWidgets import (QLabel, QSizePolicy, QWidget, QVBoxLayout,
                             QScrollArea, QFrame)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QSize, QEvent

from utils.theme import C, set_role


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

    def sizeHint(self):
        """不按 pixmap 计算：否则左栏有图、右栏只有占位文字时，等分拉伸会被拉成一宽一窄"""
        return QSize(0, self.minimumHeight())

    def minimumSizeHint(self):
        return QSize(0, self.minimumHeight())

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


class ChartLabel(QWidget):
    """图表预览画布。

    默认固定尺寸：图片按原始像素显示，画布放不下才等比缩小，绝不放大；
    容器尺寸策略用 Ignored 断开反馈环，否则外层 QScrollArea 会按标签 sizeHint 拉长
    内容区，每次 Resize 又按更大的画布重画，图片就停不住地一直变大。

    双击切换放大：小图放大铺满画布，大图按 100% 原始像素显示，超出部分用滚动条查看。
    """

    def __init__(self, placeholder="暂无图表", min_height=300, parent=None):
        super().__init__(parent)
        self._placeholder = placeholder
        self._pixmap = None
        self._zoomed = False
        self.setMinimumHeight(min_height)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._scroll = QScrollArea()
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setWidgetResizable(True)
        self._view = QLabel(placeholder)
        self._view.setAlignment(Qt.AlignCenter)
        self._view.setCursor(Qt.PointingHandCursor)
        self._view.setToolTip("双击：在「固定尺寸」与「放大」之间切换")
        set_role(self._view, "canvas")
        self._view.installEventFilter(self)
        self._scroll.setWidget(self._view)
        lay.addWidget(self._scroll)

    @property
    def source_pixmap(self):
        """未缩放的原图，保存时用它"""
        return self._pixmap

    @property
    def displayed_pixmap(self):
        return self._view.pixmap()

    @property
    def is_zoomed(self):
        return self._zoomed

    def set_image(self, pixmap):
        self._pixmap = None if (pixmap is None or pixmap.isNull()) else pixmap
        self._refit()

    def clear_image(self, text=None):
        self._pixmap = None
        self._view.setPixmap(QPixmap())
        self._view.setText(text or self._placeholder)

    def eventFilter(self, obj, event):
        if obj is self._view and event.type() == QEvent.MouseButtonDblClick \
                and self._pixmap is not None:
            self._zoomed = not self._zoomed
            self._refit()
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refit()

    def _refit(self):
        if self._pixmap is None:
            self._view.setPixmap(QPixmap())
            self._view.setText(self._placeholder)
            return
        self._view.setText("")
        natural = self._pixmap.size()
        avail = self.size()
        if avail.width() <= 0 or avail.height() <= 0:
            return
        # 等比缩放到画布所需的比例：固定模式只许缩小，放大模式只许放大
        fit = min(avail.width() / natural.width(), avail.height() / natural.height())
        factor = max(fit, 1.0) if self._zoomed else min(fit, 1.0)
        target = QSize(max(1, round(natural.width() * factor)),
                       max(1, round(natural.height() * factor)))
        scaled = self._pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        current = self._view.pixmap()
        if current is None or current.size() != scaled.size():
            self._view.setPixmap(scaled)
