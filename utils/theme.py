# -*- coding: utf-8 -*-
"""
统一视觉主题
- 设计令牌（颜色/字号/圆角/间距）
- 通过动态属性 tc="role" 选择组件角色，生成全局响应式 QSS
- 提供 header()/card() 等轻量构建器，保证各页排版一致
所有页面共用一份 app 级样式表，字号随主窗口宽度整体缩放。
"""

from PyQt5.QtWidgets import (QWidget, QLabel, QFrame, QPushButton, QHBoxLayout,
                             QVBoxLayout, QGridLayout, QSizePolicy)
from PyQt5.QtCore import Qt

FONT_FAMILY = "Microsoft YaHei"

# 调色板（浅色 slate 卡片主题，与主窗口深色顶栏/侧栏协调）
C = {
    "bg": "#f8fafc",
    "card": "#ffffff",
    "border": "#e2e8f0",
    "border_strong": "#cbd5e1",
    "text": "#334155",
    "text_strong": "#1e293b",
    "text_muted": "#64748b",
    "primary": "#3b82f6",
    "primary_hover": "#2563eb",
    "primary_pressed": "#1d4ed8",
    "primary_soft": "#eff6ff",
    "primary_softer": "#dbeafe",
    "success": "#22c55e",
    "success_hover": "#16a34a",
    "danger": "#ef4444",
    "danger_hover": "#dc2626",
    "amber": "#f59e0b",
    "amber_hover": "#d97706",
    "canvas_bg": "#1a1a2e",
}

# 基准字号/尺寸（scale=1.0 时）
S = {
    "body": 14, "h1": 21, "h1sub": 13, "h2": 16, "hint": 12, "status": 15,
    "input_pad": 10, "btn_pad_v": 10, "btn_pad_h": 20,
    "card_radius": 12, "ctrl_radius": 8, "card_pad": 20, "card_gap": 12,
}


def _i(v, scale):
    return int(round(v * scale))


def set_role(widget, role):
    """给组件打角色标签，供全局 QSS 选择"""
    widget.setProperty("tc", role)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    return widget


def qss(scale=1.0):
    """生成全局样式表，scale 控制整体字号/间距"""
    body = _i(S["body"], scale)
    h1 = _i(S["h1"], scale)
    h1sub = _i(S["h1sub"], scale)
    h2 = _i(S["h2"], scale)
    hint = _i(S["hint"], scale)
    status = _i(S["status"], scale)
    ipad = _i(S["input_pad"], scale)
    bpv = _i(S["btn_pad_v"], scale)
    bph = _i(S["btn_pad_h"], scale)
    crad = _i(S["card_radius"], scale)
    trad = _i(S["ctrl_radius"], scale)
    cpad = _i(S["card_pad"], scale)

    return f"""
    QWidget {{
        font-family: {FONT_FAMILY};
        font-size: {body}px;
        color: {C['text']};
    }}
    QMenu {{
        background-color: {C['card']};
        border: 1px solid {C['border']};
        border-radius: {trad}px;
        padding: {_i(4, scale)}px;
    }}
    QMenu::item {{
        padding: {_i(6, scale)}px {_i(20, scale)}px;
        border-radius: {_i(5, scale)}px;
    }}
    QMenu::item:selected {{ background-color: {C['primary_soft']}; color: {C['text_strong']}; }}
    QToolTip {{
        background-color: {C['text_strong']}; color: #ffffff;
        border: none; border-radius: {_i(5, scale)}px; padding: {_i(6, scale)}px;
    }}

    /* ===== 卡片容器 ===== */
    QFrame[tc="card"] {{
        background-color: {C['card']};
        border: 1px solid {C['border']};
        border-radius: {crad}px;
    }}
    QScrollArea {{ border: none; background-color: {C['bg']}; }}
    QMainWindow, QDialog {{ background-color: {C['bg']}; }}

    /* ===== 标题 / 说明 ===== */
    QLabel[tc="h1"] {{ font-size: {h1}px; font-weight: bold; color: {C['text_strong']}; }}
    QLabel[tc="h1sub"] {{ font-size: {h1sub}px; color: {C['text_muted']}; }}
    QLabel[tc="h2"] {{ font-size: {h2}px; font-weight: bold; color: {C['text_strong']}; }}
    QLabel[tc="status"] {{ font-size: {status}px; font-weight: bold; color: {C['text_strong']}; }}
    QLabel[tc="hint"] {{ font-size: {hint}px; color: {C['text_muted']}; }}
    QLabel[tc="icon"] {{ font-size: {_i(30, scale)}px; }}

    /* ===== 通用输入控件 ===== */
    QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        padding: {ipad}px {_i(14, scale)}px;
        border: 1px solid {C['border_strong']};
        border-radius: {trad}px;
        background-color: {C['bg']};
        color: {C['text']};
        selection-background-color: {C['primary']};
    }}
    QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus,
    QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {C['primary']};
        background-color: {C['card']};
    }}
    QComboBox::drop-down {{ border: none; width: {_i(22, scale)}px; }}
    QComboBox QAbstractItemView {{
        border: 1px solid {C['border']};
        border-radius: {trad}px;
        background-color: {C['card']};
        selection-background-color: {C['primary_soft']};
        selection-color: {C['text_strong']};
    }}

    /* ===== 按钮 ===== */
    QPushButton {{
        padding: {bpv}px {bph}px;
        border: 1px solid {C['border_strong']};
        border-radius: {trad}px;
        background-color: {C['bg']};
        color: {C['text']};
        font-weight: 500;
    }}
    QPushButton:hover {{ background-color: {C['primary_soft']}; border-color: {C['primary']}; }}
    QPushButton:pressed {{ background-color: {C['primary_softer']}; }}
    QPushButton:checked {{ background-color: {C['primary_softer']}; border-color: {C['primary']}; color: {C['primary_pressed']}; }}
    QPushButton:disabled {{ color: {C['text_muted']}; background-color: {C['border']}; border-color: {C['border']}; }}

    QPushButton[tc="primary"] {{
        background-color: {C['primary']}; color: #ffffff; border: none; font-weight: bold;
    }}
    QPushButton[tc="primary"]:hover {{ background-color: {C['primary_hover']}; }}
    QPushButton[tc="primary"]:pressed {{ background-color: {C['primary_pressed']}; }}
    QPushButton[tc="primary"]:disabled {{ background-color: {C['border_strong']}; color: #ffffff; }}

    QPushButton[tc="success"] {{
        background-color: {C['success']}; color: #ffffff; border: none; font-weight: bold;
    }}
    QPushButton[tc="success"]:hover {{ background-color: {C['success_hover']}; }}
    QPushButton[tc="success"]:disabled {{ background-color: {C['border_strong']}; color: #ffffff; }}

    QPushButton[tc="danger"] {{
        background-color: {C['danger']}; color: #ffffff; border: none; font-weight: bold;
    }}
    QPushButton[tc="danger"]:hover {{ background-color: {C['danger_hover']}; }}

    QPushButton[tc="amber"] {{
        background-color: {C['amber']}; color: #ffffff; border: none; font-weight: bold;
    }}
    QPushButton[tc="amber"]:hover {{ background-color: {C['amber_hover']}; }}

    QPushButton[tc="tool"] {{
        text-align: left; padding: {_i(9, scale)}px {_i(13, scale)}px;
        background-color: {C['bg']};
    }}

    /* ===== 表格 ===== */
    QTableWidget {{
        border: 1px solid {C['border']};
        border-radius: {trad}px;
        background-color: {C['card']};
        gridline-color: {C['border']};
    }}
    QTableWidget::item {{ padding: {_i(8, scale)}px; border-bottom: 1px solid {C['border']}; }}
    QTableWidget::item:selected {{ background-color: {C['primary_soft']}; color: {C['text_strong']}; }}
    QHeaderView::section {{
        background-color: {C['bg']}; padding: {_i(9, scale)}px;
        font-weight: bold; color: {C['text_strong']};
        border: none; border-bottom: 2px solid {C['border_strong']};
    }}

    /* ===== 列表 ===== */
    QListWidget {{
        border: 1px solid {C['border']};
        border-radius: {trad}px;
        background-color: {C['card']};
        padding: {_i(4, scale)}px;
    }}
    QListWidget::item {{ padding: {_i(6, scale)}px; border-radius: {_i(5, scale)}px; }}
    QListWidget::item:hover {{ background-color: {C['bg']}; }}
    QListWidget::item:selected {{
        background-color: {C['primary_softer']};
        border: 1px solid {C['primary']};
        color: {C['primary_pressed']};
        font-weight: bold;
    }}
    /* 勾选框放大，便于鼠标点选多项 */
    QListWidget::indicator {{
        width: {_i(17, scale)}px;
        height: {_i(17, scale)}px;
    }}

    /* ===== 进度条 ===== */
    QProgressBar {{
        border: 1px solid {C['border']};
        border-radius: {trad}px;
        background-color: {C['bg']};
        text-align: center; color: {C['text_strong']};
        height: {_i(18, scale)}px;
    }}
    QProgressBar::chunk {{
        border-radius: {_i(6, scale)}px;
        background-color: {C['primary']};
    }}

    /* ===== 日志文本框 ===== */
    QTextEdit[tc="log"] {{
        background-color: {C['bg']};
        border: 1px solid {C['border']};
        border-radius: {trad}px;
        padding: {_i(10, scale)}px;
        font-family: Consolas, monospace;
        font-size: {_i(13, scale)}px;
        color: {C['text']};
    }}

    /* ===== 画布 ===== */
    QLabel[tc="canvas"] {{
        background-color: {C['canvas_bg']};
        border-radius: {crad}px;
    }}

    /* ===== 复选框 ===== */
    QCheckBox {{
        spacing: {_i(7, scale)}px;
        background-color: transparent;
    }}
    QCheckBox::indicator {{
        width: {_i(16, scale)}px;
        height: {_i(16, scale)}px;
    }}

    /* ===== 分组框 ===== */
    QGroupBox {{
        border: 1px solid {C['border']};
        border-radius: {trad}px;
        margin-top: {_i(10, scale)}px;
        padding-top: {_i(6, scale)}px;
        background-color: {C['card']};
        font-weight: bold; color: {C['text_strong']};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: {_i(12, scale)}px;
        padding: 0 {_i(5, scale)}px;
        background-color: transparent;
    }}

    /* ===== 滚动条 ===== */
    QScrollBar:vertical {{
        width: 8px; background-color: {C['border']}; border-radius: 4px; margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background-color: {C['text_muted']}; border-radius: 4px; min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{ background-color: {C['text']}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar:horizontal {{
        height: 8px; background-color: {C['border']}; border-radius: 4px; margin: 0;
    }}
    QScrollBar::handle:horizontal {{
        background-color: {C['text_muted']}; border-radius: 4px; min-width: 30px;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
    """


def scale_for_width(width, base=1400, lo=0.85, hi=1.15):
    """按窗口宽度计算缩放系数"""
    if width <= 0:
        return 1.0
    return max(lo, min(hi, width / base))


# ---------- 轻量构建器 ----------

def header(icon, title, subtitle=None):
    """统一的页面头部：图标 + 标题 + 可选副标题

    纵向尺寸策略固定为 Fixed，避免窗口最大化时副标题的 wordWrap
    在滚动容器内把头部撑高、导致图标与标题上下散开。
    """
    bar = QWidget()
    bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    lay = QHBoxLayout(bar)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(S["card_gap"])
    lay.setAlignment(Qt.AlignVCenter)

    if icon:
        ic = QLabel(icon)
        ic.setProperty("tc", "icon")
        ic.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        lay.addWidget(ic, 0, Qt.AlignVCenter)

    col = QVBoxLayout()
    col.setSpacing(_i(2, 1.0))
    t = QLabel(title)
    t.setProperty("tc", "h1")
    t.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
    col.addWidget(t)
    if subtitle:
        s = QLabel(subtitle)
        s.setProperty("tc", "h1sub")
        s.setWordWrap(True)
        s.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        col.addWidget(s)
    lay.addLayout(col)
    lay.addStretch()
    return bar


def card(margin=None, spacing=None):
    """白色圆角卡片容器，返回 (frame, layout)"""
    frame = QFrame()
    frame.setProperty("tc", "card")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(margin if margin is not None else _i(S["card_pad"], 1.0),
                           margin if margin is not None else _i(S["card_pad"], 1.0),
                           margin if margin is not None else _i(S["card_pad"], 1.0),
                           margin if margin is not None else _i(S["card_pad"], 1.0))
    lay.setSpacing(spacing if spacing is not None else S["card_gap"])
    return frame, lay


def section(title):
    """卡片内的二级标题"""
    lab = QLabel(title)
    lab.setProperty("tc", "h2")
    return lab


def section_row(icon, title):
    """卡片内「图标 + 二级标题」行。

    图标与标题共用 h2 字号，两者字体度量一致，
    因此主窗口最大化、全局字号缩放后仍保持对齐。
    """
    row = QHBoxLayout()
    if icon:
        ic = QLabel(icon)
        ic.setProperty("tc", "h2")
        row.addWidget(ic)
    row.addWidget(section(title))
    row.addStretch()
    return row


def grid(columns=6, h_spacing=10, v_spacing=10):
    """等宽多列网格，用于卡片内的紧凑参数排布"""
    g = QGridLayout()
    g.setContentsMargins(0, 0, 0, 0)
    g.setHorizontalSpacing(h_spacing)
    g.setVerticalSpacing(v_spacing)
    for col in range(columns):
        g.setColumnStretch(col, 1)
    return g


def field(label_text, control, tip=None):
    """「标签 + 控件」同行字段，供 grid 多列摆布使用"""
    wrapper = QWidget()
    h = QHBoxLayout(wrapper)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(6)
    lab = QLabel(label_text)
    lab.setProperty("tc", "hint")
    h.addWidget(lab)
    h.addWidget(control, 1)
    if tip:
        control.setToolTip(tip)
    return wrapper


def path_row(edit, placeholder, on_click, icon="📁", amber=False):
    """「输入框 + 选择按钮」行"""
    if placeholder:
        edit.setPlaceholderText(placeholder)
    row = QWidget()
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(6)
    h.addWidget(edit, 1)
    btn = QPushButton(f"{icon} 选择")
    set_role(btn, "amber" if amber else "primary")
    btn.clicked.connect(on_click)
    h.addWidget(btn)
    return row
