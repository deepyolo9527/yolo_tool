
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (QLabel, QLineEdit, QPushButton,
                             QHBoxLayout, QVBoxLayout, QFormLayout, QGridLayout,
                             QTextEdit, QMessageBox, QCheckBox, QProgressBar,
                             QSizePolicy)
from PyQt5.QtCore import Qt, QMetaObject, Q_ARG, QTimer
from pages.base_page import BasePage
from workers.base_worker import WorkerThread
from workers.hardware_worker import HardwareWorker
from utils import theme
import os
import re
import shutil
import subprocess


class EnvSetupPage(BasePage):
    def __init__(self):
        super().__init__()
        self.target_env = ""
        self.target_prefix = ""
        self._hw_timer = None
        self._hw_worker = None
        self.init_ui()

    def init_ui(self):
        layout = self.get_content_layout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)

        layout.addWidget(theme.header("⚙️", "一键环境配置",
                                      "自动配置YOLO训练所需的Python环境和依赖包"))

        info_card, info_vbox = theme.card()
        info_layout = QFormLayout()
        info_layout.setSpacing(18)
        info_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        info_vbox.addLayout(info_layout)

        env_name_layout = QHBoxLayout()
        self.env_name_edit = QLineEdit("yolo_env")
        env_name_layout.addWidget(self.env_name_edit)
        info_layout.addRow(QLabel("📁 环境名称:"), env_name_layout)

        python_layout = QHBoxLayout()
        self.python_ver_edit = QLineEdit("3.10")
        python_layout.addWidget(self.python_ver_edit)
        info_layout.addRow(QLabel("🐍 Python版本:"), python_layout)

        conda_layout = QHBoxLayout()
        self.conda_path_edit = QLineEdit()
        self.conda_path_edit.setPlaceholderText("留空自动检测")
        conda_layout.addWidget(self.conda_path_edit)
        info_layout.addRow(QLabel("📦 Conda路径:"), conda_layout)

        gpu_layout = QHBoxLayout()
        self.gpu_checkbox = QCheckBox("使用GPU训练")
        self.gpu_checkbox.setChecked(False)
        gpu_layout.addWidget(self.gpu_checkbox)
        info_layout.addRow(QLabel("💻 硬件加速:"), gpu_layout)

        layout.addWidget(info_card)

        layout.addWidget(self._build_hardware_card())

        status_card, status_layout = theme.card()
        status_layout.addWidget(theme.section("📋 配置日志"))

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        # 用最小高度而不是最大高度：卡片被拉高时多出的空间归日志框，
        # 否则空间会分给标题标签，全屏时标题被顶到卡片中间、四周全是留白
        self.status_text.setMinimumHeight(160)
        self.status_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        theme.set_role(self.status_text, "log")
        status_layout.addWidget(self.status_text)
        layout.addWidget(status_card)

        progress_card, progress_layout = theme.card()
        progress_layout.addWidget(theme.section("📊 安装进度"))

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)
        layout.addWidget(progress_card)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(20)

        self.setup_btn = QPushButton("⚙️ 一键配置环境")
        theme.set_role(self.setup_btn, "primary")
        self.setup_btn.clicked.connect(self.start_setup)
        btn_layout.addWidget(self.setup_btn)

        self.stop_btn = QPushButton("⏹️ 停止配置")
        theme.set_role(self.stop_btn, "danger")
        self.stop_btn.clicked.connect(self.stop_setup)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    # ---------- 系统硬件 ----------
    _HW_ROWS = (("cpu", "CPU"), ("memory", "内存"), ("gpu", "显卡"))

    def _build_hardware_card(self):
        card, card_layout = theme.card()
        card_layout.addLayout(theme.section_row("🖥️", "系统硬件"))

        hint = QLabel("GPU 硬件检测结果与 CPU / 内存 / 显卡实时占用，本页面可见时每 3 秒自动刷新")
        theme.set_role(hint, "hint")
        card_layout.addWidget(hint)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(12)
        grid.setColumnMinimumWidth(0, 52)
        grid.setColumnStretch(1, 1)

        self._hw_views = {}
        for row, (key, title) in enumerate(self._HW_ROWS):
            name_label = QLabel(title)
            theme.set_role(name_label, "h2")
            grid.addWidget(name_label, row, 0, Qt.AlignVCenter | Qt.AlignLeft)

            value_label = QLabel("采集中…")
            value_label.setWordWrap(True)
            note_label = QLabel()
            note_label.setWordWrap(True)
            theme.set_role(note_label, "hint")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(False)
            percent_label = QLabel("--")
            theme.set_role(percent_label, "status")
            percent_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            percent_label.setMinimumWidth(52)

            bar_row = QHBoxLayout()
            bar_row.setContentsMargins(0, 0, 0, 0)
            bar_row.setSpacing(10)
            bar_row.addWidget(bar, 1)
            bar_row.addWidget(percent_label)

            column = QVBoxLayout()
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(4)
            column.addWidget(value_label)
            column.addWidget(note_label)
            column.addLayout(bar_row)
            grid.addLayout(column, row, 1)

            self._hw_views[key] = (value_label, note_label, bar, percent_label)

        card_layout.addLayout(grid)
        return card

    def _show_hardware(self, data):
        if not data:
            for value, note, bar, percent in self._hw_views.values():
                self._set_hw_row(value, note, bar, percent, "硬件信息采集失败", "", None, True)
            return

        cpu = data.get("cpu") or {}
        value, note, bar, percent = self._hw_views["cpu"]
        self._set_hw_row(value, note, bar, percent,
                         " · ".join(x for x in (cpu.get("name"), cpu.get("cores")) if x),
                         cpu.get("detail", ""), cpu.get("percent"), False)

        mem = data.get("memory") or {}
        value, note, bar, percent = self._hw_views["memory"]
        self._set_hw_row(value, note, bar, percent,
                         mem.get("detail") or "无法读取内存占用", "", mem.get("percent"), False)

        gpu = data.get("gpu") or {}
        value, note, bar, percent = self._hw_views["gpu"]
        available = bool(gpu.get("available"))
        # 没有占用可显示时收起进度条，空着的进度条会被误读成「占用 0%」
        self._set_hw_row(value, note, bar, percent,
                         gpu.get("name") or "未检测到显卡", gpu.get("detail", ""),
                         gpu.get("percent") if available else None,
                         warn=not available)

    @staticmethod
    def _set_hw_row(value, note, bar, percent_label, text, note_text, percent, warn):
        value.setText(text or "--")
        note.setVisible(bool(note_text))
        note.setText(note_text)
        # GPU 不可用是用户最需要看清的一条，整行改用警示色
        theme.set_role(note, "warn" if warn else "hint")
        if percent is None:
            bar.hide()
            percent_label.hide()
        else:
            bar.show()
            percent_label.show()
            bar.setValue(int(percent))
            percent_label.setText(f"{percent:.0f}%")

    def showEvent(self, event):
        super().showEvent(event)
        if self._hw_timer is None:
            self._hw_timer = QTimer(self)
            self._hw_timer.setInterval(3000)
            self._hw_timer.timeout.connect(self._probe_hardware)
        if not self._hw_timer.isActive():
            self._probe_hardware()
            self._hw_timer.start()

    def hideEvent(self, event):
        super().hideEvent(event)
        if self._hw_timer is not None:
            self._hw_timer.stop()
        if self._hw_worker is not None and self._hw_worker.isRunning():
            self._hw_worker.wait(1500)

    def _probe_hardware(self):
        if self._hw_worker is not None and self._hw_worker.isRunning():
            return
        self._hw_worker = HardwareWorker(self)
        self._hw_worker.result_signal.connect(self._show_hardware)
        self._hw_worker.start()

    def log(self, msg):
        QMetaObject.invokeMethod(self.status_text, "append", Qt.QueuedConnection,
                                Q_ARG(str, msg))

    def _set_progress(self, value):
        QMetaObject.invokeMethod(self.progress_bar, "setValue", Qt.QueuedConnection,
                                Q_ARG(int, value))

    def _find_conda(self):
        candidates = [os.environ.get('CONDA_EXE', '')]
        for root in (os.environ.get('USERPROFILE', ''),
                     'C:\\', 'D:\\', 'E:\\', 'F:\\'):
            if not root:
                continue
            for name in ('miniconda3', 'anaconda3', 'Miniconda3', 'Anaconda3'):
                candidates.append(os.path.join(root, name, 'Scripts', 'conda.exe'))

        for path in candidates:
            if path and os.path.exists(path):
                return path

        return shutil.which('conda')

    def _get_env_prefix(self, conda_path, env_name):
        """从 `conda env list` 解析环境绝对路径（名称后可能带 * / + 标记，路径可能含空格）"""
        try:
            result = subprocess.run([conda_path, 'env', 'list'],
                                    capture_output=True, text=True)
        except OSError:
            return None

        if result.returncode != 0:
            return None

        for line in result.stdout.splitlines():
            m = re.match(r'^(\S+)\s+([*+])?\s+(.+)$', line.strip())
            if m and m.group(1) == env_name:
                prefix = m.group(3).strip()
                if os.path.isdir(prefix):
                    return prefix

        # 兜底：conda 同级 envs 目录（应对 conda 输出格式变化）
        if os.path.isabs(conda_path):
            candidate = os.path.join(os.path.dirname(os.path.dirname(conda_path)),
                                     'envs', env_name)
            if os.path.isdir(candidate):
                return candidate
        return None

    def _get_env_python(self, prefix):
        # conda 在 Windows 下把 python.exe 放在环境根目录，pip.exe 才在 Scripts 下
        for rel in (('python.exe',), ('Scripts', 'python.exe'),
                    ('python',), ('bin', 'python'), ('bin', 'python3')):
            exe = os.path.join(prefix, *rel)
            if os.path.exists(exe):
                return exe
        return None

    def _install_conda(self):
        self.log("正在下载Miniconda...")
        import urllib.request
        import tempfile

        miniconda_url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
        install_dir = os.path.join(os.environ.get('USERPROFILE', ''), 'miniconda3')

        if os.path.exists(install_dir):
            self.log(f"Miniconda已安装: {install_dir}")
            return os.path.join(install_dir, 'Scripts', 'conda.exe')

        try:
            with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as f:
                temp_path = f.name

            self.log(f"下载地址: {miniconda_url}")
            self.log(f"临时文件: {temp_path}")

            urllib.request.urlretrieve(miniconda_url, temp_path)

            self.log("正在安装Miniconda...")
            result = subprocess.run(
                [temp_path, '/InstallationType=JustMe', '/RegisterPython=0', '/S', f'/D={install_dir}'],
                capture_output=True, text=True, timeout=600
            )

            os.unlink(temp_path)

            if result.returncode == 0:
                conda_path = os.path.join(install_dir, 'Scripts', 'conda.exe')
                self.log(f"Miniconda安装成功: {conda_path}")
                return conda_path
            else:
                self.log(f"Miniconda安装失败: {result.stderr}")
                return None
        except Exception as e:
            self.log(f"下载/安装Miniconda失败: {str(e)}")
            return None

    def _create_conda_env(self, conda_path, env_name, python_version):
        self.log(f"检查环境: {env_name}")

        if self._get_env_prefix(conda_path, env_name):
            self.log(f"环境 {env_name} 已存在")
            return True

        self.log(f"创建环境 {env_name} (Python {python_version})...")
        try:
            result = subprocess.run(
                [conda_path, 'create', '-n', env_name,
                 f'python={python_version}', 'pip', '-y'],
                capture_output=True, text=True
            )
        except OSError as e:
            self.log(f"创建环境失败: {e}")
            return False

        if result.returncode == 0:
            self.log(f"环境 {env_name} 创建成功")
            return True
        else:
            self.log(f"创建环境失败: {result.stderr}")
            return False

    def _pip_install(self, python_exe, args):
        """统一走 `python -m pip`，避免依赖 pip.exe 的路径推断"""
        try:
            result = subprocess.run([python_exe, '-m', 'pip', 'install'] + args,
                                    capture_output=True, text=True)
        except OSError as e:
            return False, str(e)

        if result.returncode == 0:
            return True, ''
        return False, (result.stderr or result.stdout or '').strip()

    def _has_pip(self, python_exe):
        try:
            return subprocess.run([python_exe, '-m', 'pip', '--version'],
                                  capture_output=True, text=True).returncode == 0
        except OSError:
            return False

    def _ensure_pip(self, conda_path, env_name, python_exe):
        if self._has_pip(python_exe):
            return True

        self.log("环境中缺少 pip，正在通过 conda 安装...")
        try:
            result = subprocess.run([conda_path, 'install', '-n', env_name, 'pip', '-y'],
                                    capture_output=True, text=True)
        except OSError as e:
            self.log(f"pip 安装失败: {e}")
            return False

        if result.returncode != 0 or not self._has_pip(python_exe):
            self.log(f"pip 安装失败: {(result.stderr or result.stdout or '').strip()}")
            return False

        self.log("pip 安装成功")
        return True

    def _install_dependencies(self, conda_path, env_name, use_gpu=False, _stop_event=None):
        """返回失败的包名列表；无法继续时返回 None"""
        self.log("安装依赖包...")
        failed = []

        def check_stop():
            return _stop_event and _stop_event._stop_flag

        prefix = self._get_env_prefix(conda_path, env_name)
        if not prefix:
            self.log(f"未找到环境 {env_name} 的安装路径，无法安装依赖")
            return None

        python_exe = self._get_env_python(prefix)
        if not python_exe:
            self.log(f"环境 {env_name} 中未找到 Python 解释器: {prefix}")
            return None

        self.log(f"环境路径: {prefix}")
        self.target_prefix = prefix
        if not self._ensure_pip(conda_path, env_name, python_exe):
            return None

        pypi_mirror = "https://pypi.tuna.tsinghua.edu.cn/simple"
        mirror_args = ['-i', pypi_mirror,
                       '--trusted-host', 'pypi.tuna.tsinghua.edu.cn']

        dependencies = [
            'ultralytics',
            'pyqt5',
            'opencv-python',
            'pyyaml',
            # 系统硬件分区的 CPU/内存占用采集；缺失时界面会降级显示，不影响训练
            'psutil',
        ]

        base_progress = 50
        progress_step = 40 // len(dependencies)

        for i, dep in enumerate(dependencies):
            if check_stop():
                return failed
            self.log(f"安装 {dep}...")
            ok, err = self._pip_install(python_exe, [dep] + mirror_args)
            if ok:
                self.log(f"{dep} 安装成功")
            else:
                self.log(f"{dep} 安装失败: {err}")
                failed.append(dep)
            self._set_progress(base_progress + (i + 1) * progress_step)

        if check_stop():
            return failed

        self._set_progress(90)
        if use_gpu:
            self.log("安装GPU版本PyTorch...")
            ok, err = self._pip_install(
                python_exe,
                ['torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu121'])
            if ok:
                self.log("GPU版本PyTorch安装成功")
                return failed
            self.log(f"GPU版本PyTorch安装失败: {err}")
            self.log("尝试安装CPU版本...")
            ok, err = self._pip_install(python_exe, ['torch', 'torchvision'] + mirror_args)
            if ok:
                self.log("CPU版本PyTorch安装成功")
            else:
                self.log(f"PyTorch安装失败: {err}")
                failed.append('torch')
        else:
            self.log("安装CPU版本PyTorch...")
            ok, err = self._pip_install(python_exe, ['torch', 'torchvision'] + mirror_args)
            if ok:
                self.log("CPU版本PyTorch安装成功")
            else:
                self.log(f"PyTorch安装失败: {err}")
                failed.append('torch')
        self.log("依赖安装完成" if not failed else "依赖安装结束，部分包失败")
        return failed

    def start_setup(self):
        if self.thread and self.thread.isRunning():
            QMessageBox.warning(self, "提示", "任务正在运行中")
            return

        env_name = self.env_name_edit.text().strip()
        python_version = self.python_ver_edit.text().strip()
        conda_path = self.conda_path_edit.text().strip()
        use_gpu = self.gpu_checkbox.isChecked()

        if not env_name:
            QMessageBox.warning(self, "提示", "请输入环境名称")
            return

        self.target_env = env_name
        self.target_prefix = ""
        self.progress_bar.setValue(0)
        self.thread = WorkerThread(self._setup_task, env_name, python_version,
                                   conda_path, use_gpu)
        self.thread.log_signal.connect(self.log)
        self.thread.finished_signal.connect(self.on_setup_finished)
        self.thread.start()
        self.stop_btn.setEnabled(True)

    def stop_setup(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.log("正在停止配置...")

    def _setup_task(self, env_name, python_version, conda_path='', use_gpu=False,
                    _stop_event=None):
        self.log("=" * 50)
        self.log("开始一键环境配置")
        self.log(f"硬件加速: {'GPU' if use_gpu else 'CPU'}")
        self.log("=" * 50)

        def check_stop():
            return _stop_event and _stop_event._stop_flag

        self._set_progress(5)
        if check_stop():
            return "任务已停止"

        if not conda_path:
            self.log("检测Conda...")
            conda_path = self._find_conda()
            self._set_progress(10)

        if check_stop():
            return "任务已停止"

        if not conda_path:
            self.log("未找到Conda，开始安装Miniconda...")
            conda_path = self._install_conda()
            self._set_progress(30)

        if check_stop():
            return "任务已停止"

        if not conda_path:
            return "Conda安装失败，请手动安装"

        self.log(f"Conda路径: {conda_path}")

        self._set_progress(40)
        if check_stop():
            return "任务已停止"

        if not self._create_conda_env(conda_path, env_name, python_version):
            return "创建环境失败"

        self._set_progress(50)
        if check_stop():
            return "任务已停止"

        failed = self._install_dependencies(conda_path, env_name, use_gpu, _stop_event)

        if check_stop():
            return "任务已停止"

        if failed is None:
            return "依赖安装未能开始，请查看日志"

        self._set_progress(100)
        if failed:
            return "环境已就绪，但以下依赖安装失败: " + ", ".join(failed)
        return f"环境 {env_name} 配置完成"

    def on_setup_finished(self, success, msg):
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100 if success else 0)
        if not success:
            self.log(f"[错误] {msg}")
            return

        self.log(f"[完成] {msg}")
        if not self.target_env:
            return

        cmd = f"conda activate {self.target_env}"
        steps = ["在命令行(Anaconda Prompt)中执行: " + cmd]
        python_exe = self._get_env_python(self.target_prefix) if self.target_prefix else None
        if python_exe:
            steps.append("该环境的 Python 解释器: " + python_exe)
        steps.append("在激活后的环境中重新启动本程序，训练/推理才会使用新依赖")

        lines = ["✅ " + str(msg), ""]
        lines += [f"{i}) {s}" for i, s in enumerate(steps, 1)]
        tip = "\n".join(lines)
        self.log("[提示] 请执行 " + cmd + " 后重新启动程序")
        QMessageBox.information(self, "环境配置完成", tip)
