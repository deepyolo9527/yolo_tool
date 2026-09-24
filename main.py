
# -*- coding: utf-8 -*-

import sys
import os
import re
import io


def _ensure_console_streams():
    """无控制台打包(--windowed)时 sys.stdout/stderr 是 None，必须在建窗之前补上真实流。

    ultralytics 在导入时执行 logging.StreamHandler(sys.stdout)，把当时的 None 永久绑进了
    handler，之后它每写一条日志都会抛 AttributeError 并打印一整段 '--- Logging error ---'。
    这里换成丢弃输出的文件对象：界面日志仍由训练线程重定向抓取，互不影响。
    """
    for name in ("stdout", "stderr"):
        if hasattr(getattr(sys, name, None), "write"):
            continue
        try:
            fallback = open(os.devnull, "w", encoding="utf-8", errors="replace")
        except OSError:
            fallback = io.StringIO()
        setattr(sys, name, fallback)


_ensure_console_streams()

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QHBoxLayout, QVBoxLayout, QPushButton, QStackedWidget,
                             QDialog, QMessageBox, QTextBrowser, QListWidget,
                             QListWidgetItem)
from PyQt5.QtCore import Qt, QTimer, QDateTime, QUrl
from PyQt5.QtGui import QFont, QPixmap, QDesktopServices, QIcon

from utils import theme
from pages.env_setup_page import EnvSetupPage
from pages.data_annotation_page import DataAnnotationPage
from pages.video_tracking_page import VideoTrackingPage
from pages.count_labels_page import CountLabelsPage
from pages.training_page import TrainingPage
from pages.result_preview_page import ResultPreviewPage
from pages.image_inference_page import ImageInferencePage
from pages.video_inference_page import VideoInferencePage
from pages.model_export_page import ModelExportPage


if getattr(sys, 'frozen', False):
    # 打包运行：onefile 的工作目录是临时解压目录(_MEIxxxx)，切到 exe 所在目录，
    # 使 weights/datasets/runs 等相对路径落在 exe 旁
    _app_dir = os.path.dirname(os.path.abspath(sys.executable))
    os.chdir(_app_dir)
    # ultralytics 的 settings.json 单独放在 exe 旁：否则它与开发环境共用
    # %APPDATA%\Ultralytics，会把临时解压目录等错误路径写给对方
    os.environ.setdefault("YOLO_CONFIG_DIR",
                          os.path.join(_app_dir, "configs", "ultralytics_settings"))


def _resource_root():
    """资源根目录：兼容 PyInstaller 打包(assets 随 exe 解压)与源码运行"""
    if getattr(sys, 'frozen', False):
        return getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(sys.argv[0])))
    return os.path.dirname(os.path.abspath(__file__))


def _app_icon():
    """程序图标：assets/app.ico 内含 16~256 多尺寸，缺失时退到 PNG 图标"""
    for name in ("app.ico", "app.png", "app_1024.png", "logo.png"):
        path = os.path.join(_resource_root(), "assets", name)
        if os.path.exists(path):
            return QIcon(path)
    return QIcon()


# 帮助文档章节（锚点 id, 侧栏标题），与 assets/help.html 中的 <h2 id="..."> 对应
HELP_SECTIONS = [
    ("s1", "1. 系统总览"),
    ("s2", "2. 快速上手路线"),
    ("s3", "3. 环境配置"),
    ("s4", "4. 数据标注"),
    ("s5", "5. SAM视频追踪"),
    ("s6", "6. 标签统计"),
    ("s7", "7. 模型训练"),
    ("s8", "8. 图片推理"),
    ("s9", "9. 视频推理"),
    ("s10", "10. 模型导出"),
    ("s11", "11. 结果预览"),
    ("s12", "12. 目录结构与配置文件"),
    ("s13", "13. 常见问题 FAQ"),
]


class HelpDialog(QDialog):
    """内置使用说明：左侧章节导航 + 右侧文档，滚动与导航双向联动"""

    def __init__(self, help_path, parent=None):
        super().__init__(parent)
        self.help_path = help_path
        self._anchor_pos = []
        self._sync = False
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("YOLO AI 训练系统 · 使用说明")
        self.resize(1120, 800)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        bar = QHBoxLayout()
        bar.setSpacing(6)
        title = QLabel("📖 操作使用说明")
        title.setStyleSheet("color:#0f172a; font-size:15px; font-weight:bold;")
        bar.addWidget(title)
        bar.addStretch()

        def tool_btn(text, tip="", width=46):
            btn = QPushButton(text)
            btn.setFixedWidth(width)
            if tip:
                btn.setToolTip(tip)
            bar.addWidget(btn)
            return btn

        btn_out = tool_btn("－", "缩小字号")
        btn_in = tool_btn("＋", "放大字号")
        btn_web = tool_btn("🌐 浏览器打开", "用系统默认浏览器查看完整排版", 130)
        btn_close = tool_btn("✕ 关闭", "关闭帮助窗口", 90)
        lay.addLayout(bar)

        body = QHBoxLayout()
        body.setSpacing(8)

        self.nav = QListWidget()
        self.nav.setFixedWidth(190)
        for anchor, label in HELP_SECTIONS:
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, anchor)
            self.nav.addItem(item)
        self.nav.currentRowChanged.connect(self.jump_row)
        body.addWidget(self.nav)

        self.browser = QTextBrowser()
        self.browser.setStyleSheet("""
            QTextBrowser {
                background-color: #f8fafc;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        self.browser.setSearchPaths([os.path.dirname(self.help_path)])
        self.browser.setHtml(self._read_html())
        self.browser.anchorClicked.connect(self._on_anchor_clicked)
        self.browser.verticalScrollBar().valueChanged.connect(self.on_scroll)
        body.addWidget(self.browser, 1)
        lay.addLayout(body, 1)

        btn_out.clicked.connect(self.browser.zoomOut)
        btn_in.clicked.connect(self.browser.zoomIn)
        btn_web.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.help_path)))
        btn_close.clicked.connect(self.close)

        self.nav.setCurrentRow(0)
        QTimer.singleShot(0, self.measure_anchors)

    def _read_html(self):
        """Qt 富文本不支持固定侧栏，去掉 HTML 目录与缩进，改用窗口左侧导航列表"""
        try:
            with open(self.help_path, "r", encoding="utf-8", errors="ignore") as f:
                html = f.read()
        except OSError:
            return "<h2>帮助文档读取失败</h2>"
        html = re.sub(r'<nav class="menu">.*?</nav>', "", html, flags=re.S)
        html = re.sub(r'<script[^>]*>.*?</script>', "", html, flags=re.S)
        return re.sub(r"\.doc\s*\{[^}]*\}", "", html)

    def measure_anchors(self):
        """记录各章节的滚动位置，供滚动反向高亮导航"""
        bar = self.browser.verticalScrollBar()
        self._anchor_pos = []
        for row, (anchor, _) in enumerate(HELP_SECTIONS):
            self.browser.scrollToAnchor(anchor)
            self._anchor_pos.append((bar.value(), row))
        bar.setValue(0)
        self._anchor_pos.sort()

    def jump_row(self, row):
        if self._sync or row < 0 or row >= len(HELP_SECTIONS):
            return
        self._sync = True
        self.browser.scrollToAnchor(HELP_SECTIONS[row][0])
        self._sync = False

    def on_scroll(self, value):
        if self._sync or not self._anchor_pos:
            return
        row = 0
        for pos, r in self._anchor_pos:
            if value >= pos - 6:
                row = r
            else:
                break
        if row != self.nav.currentRow():
            self._sync = True
            self.nav.setCurrentRow(row)
            self._sync = False

    def _on_anchor_clicked(self, url):
        # 正文内的章节引用只在窗口内跳转，不重新加载文档
        frag = url.fragment()
        if frag:
            self.browser.scrollToAnchor(frag)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("YOLO AI 训练系统")
        self.setWindowIcon(_app_icon())
        self.setGeometry(50, 50, 1280, 720)
        self.setMinimumSize(1024, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self._create_header(main_layout)

        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self._create_sidebar(content_layout)

        self.pages = QStackedWidget()
        self.pages.setStyleSheet("""
            QStackedWidget {
                background-color: #f8fafc;
                border-radius: 0;
            }
        """)

        menu_items = [
            ("⚙️ 环境配置", EnvSetupPage),
            ("✏️ 数据标注", DataAnnotationPage),
            ("🎬 SAM视频追踪", VideoTrackingPage),
            ("📊 标签统计", CountLabelsPage),
            ("🚀 YOLO训练", TrainingPage),
            ("🖼️ 图片推理", ImageInferencePage),
            ("🎬 视频推理", VideoInferencePage),
            ("📦 模型导出", ModelExportPage),
            ("📈 结果预览", ResultPreviewPage),
        ]

        for name, page_class in menu_items:
            page = page_class()
            self.pages.addWidget(page)

        anno_page = next((self.pages.widget(i) for i in range(self.pages.count())
                          if isinstance(self.pages.widget(i), DataAnnotationPage)), None)
        video_idx = next((i for i in range(self.pages.count())
                          if isinstance(self.pages.widget(i), VideoTrackingPage)), -1)
        if anno_page is not None and video_idx >= 0:
            anno_page.open_video_tracking.connect(lambda: self.switch_page(video_idx))

        content_layout.addWidget(self.pages, 1)
        main_layout.addLayout(content_layout)

        self.switch_page(0)

    def _create_header(self, parent_layout):
        header = QWidget()
        header.setFixedHeight(70)
        header.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #0f172a, stop:0.5 #1e293b, stop:1 #0f172a);
                border-bottom: 2px solid #3b82f6;
            }
        """)

        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(25, 0, 25, 0)
        header_layout.setSpacing(20)

        logo_label = QLabel("🤖")
        logo_label.setStyleSheet("font-size: 36px;")
        header_layout.addWidget(logo_label)

        title_label = QLabel("YOLO AI 训练系统")
        title_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 22px;
                font-weight: bold;
                font-family: Microsoft YaHei;
            }
        """)
        header_layout.addWidget(title_label)

        self.tagline_label = QLabel("一站式从数据智能标注到模型一键训练、推理")
        self.tagline_label.setStyleSheet("""
            QLabel {
                color: #94a3b8;
                font-size: 13px;
                font-family: Microsoft YaHei;
            }
        """)
        header_layout.addWidget(self.tagline_label, 0, Qt.AlignVCenter)

        header_layout.addStretch()

        shop_widget = QWidget()
        shop_layout = QVBoxLayout(shop_widget)
        shop_layout.setContentsMargins(0, 0, 0, 0)
        shop_layout.setSpacing(2)

        shop_label = QLabel("💬 微信公众号: YOLO深度学习")
        shop_label.setStyleSheet("""
            QLabel {
                color: #fbbf24;
                font-size: 14px;
                font-weight: bold;
                font-family: Microsoft YaHei;
            }
        """)
        shop_label.setCursor(Qt.PointingHandCursor)
        shop_label.setToolTip("点击访问公众号文章")
        shop_label.mousePressEvent = self._open_taobao
        shop_layout.addWidget(shop_label)

        author_label = QLabel("© YOLO深度学习 出品")
        author_label.setStyleSheet("""
            QLabel {
                color: #64748b;
                font-size: 11px;
            }
        """)
        author_label.setAlignment(Qt.AlignCenter)
        author_label.setCursor(Qt.PointingHandCursor)
        author_label.setToolTip("点击查看微信公众号二维码")
        author_label.mousePressEvent = self._show_qrcode
        shop_layout.addWidget(author_label)

        header_layout.addWidget(shop_widget)
        header_layout.addSpacing(20)

        status_label = QLabel("● 就绪")
        status_label.setStyleSheet("""
            QLabel {
                color: #22c55e;
                font-size: 16px;
                padding-right: 10px;
            }
        """)
        header_layout.addWidget(status_label)

        about_label = QLabel("ℹ️ 关于系统")
        about_label.setStyleSheet("""
            QLabel {
                color: #93c5fd;
                font-size: 14px;
                font-family: Microsoft YaHei;
                padding-right: 10px;
            }
            QLabel:hover {
                color: #ffffff;
            }
        """)
        about_label.setCursor(Qt.PointingHandCursor)
        about_label.setToolTip("点击查看软件使用帮助说明")
        about_label.mousePressEvent = self._open_help
        header_layout.addWidget(about_label)

        time_label = QLabel()
        time_label.setStyleSheet("""
            QLabel {
                color: #94a3b8;
                font-size: 16px;
            }
        """)
        header_layout.addWidget(time_label)

        def update_time():
            current_time = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
            time_label.setText(current_time)
        timer = QTimer(self)
        timer.timeout.connect(update_time)
        timer.start(1000)
        update_time()

        parent_layout.addWidget(header)

    # ---------- 公众号文章跳转 ----------
    def _open_taobao(self, event=None):
        QDesktopServices.openUrl(QUrl("https://mp.weixin.qq.com/s/_mgm-EhrPkegM7FuPFgfEg"))

    # ---------- 关于系统 / 使用帮助 ----------
    def _find_help_file(self):
        for rel in (("assets", "help.html"), ("docs", "help.html")):
            path = os.path.join(_resource_root(), *rel)
            if os.path.exists(path):
                return path
        return ""

    def _open_help(self, event=None):
        help_path = self._find_help_file()
        if not help_path:
            box = QMessageBox(self)
            box.setWindowTitle("使用帮助")
            box.setIcon(QMessageBox.Warning)
            box.setText("未找到帮助文档。\n\n请确认以下文件存在:\n"
                        f"{os.path.join(_resource_root(), 'assets', 'help.html')}")
            box.exec_()
            return
        HelpDialog(help_path, self).exec_()

    # ---------- 公众号二维码 ----------
    def _find_qr_image(self):
        """在 assets/ 下查找二维码图片，优先匹配含关键字的文件名"""
        assets = os.path.join(_resource_root(), "assets")
        if not os.path.isdir(assets):
            return ""
        exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        names = [n for n in sorted(os.listdir(assets))
                 if n.lower().endswith(exts)]
        if not names:
            return ""
        keys = ("qr", "weixin", "wechat", "gzh", "公众号", "二维码")
        preferred = [n for n in names
                     if any(k in n.lower() for k in keys)]
        return os.path.join(assets, (preferred or names)[0])

    def _show_qrcode(self, event=None):
        img_path = self._find_qr_image()
        if not img_path:
            box = QMessageBox(self)
            box.setWindowTitle("微信公众号")
            box.setIcon(QMessageBox.Information)
            box.setText("未找到二维码图片。\n\n请将公众号图片放到:\n"
                        f"{os.path.join(_resource_root(), 'assets')}\n"
                        "（如 qrcode.png），然后点击本行即可显示。")
            box.exec_()
            return
        pixmap = QPixmap(img_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "错误", f"无法读取图片:\n{img_path}")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("微信公众号© YOLO深度学习")
        dlg.setWindowFlags(dlg.windowFlags() | Qt.FramelessWindowHint)
        dlg.setStyleSheet("QDialog { background: #ffffff; }")

        shown = pixmap
        max_w = min(pixmap.width(), 360)
        if pixmap.width() > max_w:
            shown = pixmap.scaledToWidth(max_w, Qt.SmoothTransformation)

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        pic = QLabel()
        pic.setPixmap(shown)
        pic.setAlignment(Qt.AlignCenter)
        layout.addWidget(pic)

        tip = QLabel("扫码关注 · 获取更多源码好工具")
        tip.setAlignment(Qt.AlignCenter)
        tip.setStyleSheet("color:#475569; font-size:13px; font-family:Microsoft YaHei;")
        layout.addWidget(tip)

        def close_dialog(_=None):
            dlg.close()
        dlg.mouseReleaseEvent = close_dialog
        pic.mouseReleaseEvent = close_dialog
        tip.mouseReleaseEvent = close_dialog

        dlg.exec_()

    def _create_sidebar(self, parent_layout):
        sidebar = QWidget()
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #1e293b, stop:1 #0f172a);
                border-right: 1px solid #334155;
            }
        """)

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(12, 20, 12, 25)
        sidebar_layout.setSpacing(6)

        top_qss = """
            QPushButton {
                text-align: left;
                padding: 15px 18px;
                border: none;
                font-size: 16px;
                font-weight: 500;
                color: #cbd5e1;
                background-color: transparent;
                border-radius: 10px;
                font-family: Microsoft YaHei;
            }
            QPushButton:hover { background-color: #334155; color: #ffffff; }
            QPushButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #3b82f6, stop:1 #2563eb);
                color: #ffffff;
                border-left: 3px solid #60a5fa;
            }
        """
        child_qss = """
            QPushButton {
                text-align: left;
                padding: 11px 14px 11px 30px;
                border: none;
                font-size: 15px;
                color: #94a3b8;
                background-color: transparent;
                border-radius: 8px;
                font-family: Microsoft YaHei;
            }
            QPushButton:hover { background-color: #334155; color: #ffffff; }
            QPushButton:checked {
                background-color: #3b82f6;
                color: #ffffff;
                font-weight: 500;
                border-left: 3px solid #60a5fa;
            }
        """
        group_qss = """
            QPushButton {
                text-align: left;
                padding: 12px 14px;
                border: none;
                font-size: 16px;
                font-weight: bold;
                color: #e2e8f0;
                background-color: transparent;
                border-radius: 8px;
                font-family: Microsoft YaHei;
            }
            QPushButton:hover { background-color: #334155; color: #ffffff; }
        """

        self.menu_buttons = {}    # 页索引 -> 按钮
        self._page_group = {}     # 页索引 -> (容器, 组标题按钮)

        def make_page_btn(text, idx, qss, tooltip=""):
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setStyleSheet(qss)
            if tooltip:
                btn.setToolTip(tooltip)
            btn.clicked.connect(lambda checked, i=idx: self.switch_page(i))
            self.menu_buttons[idx] = btn
            return btn

        def make_group(title, items, icon=""):
            label = f"{icon} {title}".strip()
            header = QPushButton(f"▸  {label}")
            header.setStyleSheet(group_qss)
            header.setProperty("group_title", label)
            container = QWidget()
            cl = QVBoxLayout(container)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(4)
            for text, idx, tip in items:
                btn = make_page_btn(text, idx, child_qss, tip)
                cl.addWidget(btn)
                self._page_group[idx] = (container, header)
            container.setVisible(False)
            header.clicked.connect(lambda: self._toggle_group(container, header))
            sidebar_layout.addWidget(header)
            sidebar_layout.addWidget(container)

        sidebar_layout.addWidget(make_page_btn("⚙️ 环境配置", 0, top_qss,
                                               "一键配置Python训练环境"))
        make_group("数据处理", [
            ("✏️ 数据标注", 1, "矩形框、SAM分割与AI预标注"),
            ("🎬 SAM视频追踪", 2, "SAM2视频追踪标注：框选一次，自动逐帧标注"),
            ("📊 标签统计", 3, "统计并可视化标签分布"),
        ], icon="🗂️")
        make_group("YOLO训练", [
            ("🚀 模型训练", 4, "配置参数并启动YOLO训练"),
            ("📦 模型导出", 7, "导出为ONNX等多种格式"),
        ], icon="🔥")
        make_group("YOLO推理", [
            ("🖼️ 图片推理", 5, "对图片批量推理"),
            ("🎬 视频推理", 6, "对视频逐帧推理"),
            ("📈 结果预览", 8, "查看训练结果与指标曲线"),
        ], icon="⚡")

        sidebar_layout.addStretch()

        version_label = QLabel("v1.0.0")
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setStyleSheet("""
            QLabel {
                color: #64748b;
                font-size: 14px;
            }
        """)
        sidebar_layout.addWidget(version_label)

        parent_layout.addWidget(sidebar)

    def _toggle_group(self, container, header, force_open=None):
        open_it = (not container.isVisible()) if force_open is None else force_open
        container.setVisible(open_it)
        title = header.property("group_title")
        header.setText(("▾  " if open_it else "▸  ") + title)

    def switch_page(self, index):
        for btn in self.menu_buttons.values():
            btn.setChecked(False)
        btn = self.menu_buttons.get(index)
        if btn is not None:
            btn.setChecked(True)
        group = self._page_group.get(index)
        if group is not None:
            container, header = group
            if not container.isVisible():
                self._toggle_group(container, header, force_open=True)
        self.pages.setCurrentIndex(index)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        scale = theme.scale_for_width(self.width())
        if abs(scale - getattr(self, "_qss_scale", 1.0)) >= 0.03:
            self._qss_scale = scale
            QApplication.instance().setStyleSheet(theme.qss(scale))
        # 窗口过窄时收起标语，避免与右侧状态区挤压被裁剪
        self.tagline_label.setVisible(
            self.width() - self.tagline_label.sizeHint().width() >= 1000)


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    font = QFont("Microsoft YaHei", 13)
    app.setFont(font)

    # 应用级图标：任务栏、Alt+Tab 与所有对话框统一使用
    app.setWindowIcon(_app_icon())

    window = MainWindow()
    app.setStyleSheet(theme.qss(1.0))
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    # 打包版里 ultralytics 训练的 DataLoader 用 spawn 启动子进程，子进程会再次执行本入口；
    # freeze_support() 让子进程只跑数据加载任务，不再弹出 GUI 与残留临时解压目录
    import multiprocessing

    multiprocessing.freeze_support()
    main()
