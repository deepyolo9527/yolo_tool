
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (QLineEdit, QPushButton, QHBoxLayout, QTextEdit,
                             QMessageBox, QFileDialog, QSpinBox, QDoubleSpinBox,
                             QComboBox, QCheckBox)
from PyQt5.QtCore import Qt, QMetaObject, Q_ARG
from pages.base_page import BasePage
from utils import theme
from utils.path_helpers import get_app_dir, get_ultralytics_cfg_dir, get_weights_dir
from workers.train_worker import YOLOTrainWorker
import os
import yaml


project_root = get_app_dir()
cfg_root = get_ultralytics_cfg_dir()


class TrainingPage(BasePage):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.init_ui()

    def init_ui(self):
        layout = self.get_content_layout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)

        layout.addWidget(theme.header("🚀", "YOLO训练", "配置训练参数，启动YOLO模型训练"))

        # 运行控制与日志置顶，训练时无需滚动即可看到进度
        layout.addWidget(self._build_run_card())
        layout.addWidget(self._build_log_card())

        layout.addWidget(self._build_model_card())
        layout.addWidget(self._build_hyp_card())
        layout.addWidget(self._build_aug_card())
        layout.addWidget(self._build_options_card())

    # ---------- 参数控件工厂 ----------
    def _ispin(self, value, tip, lo=None, hi=None):
        spin = QSpinBox()
        if lo is not None:
            spin.setRange(lo, hi)
        spin.setValue(value)
        spin.setToolTip(tip)
        return spin

    def _dspin(self, value, tip, decimals=None):
        spin = QDoubleSpinBox()
        if decimals is not None:
            # 必须先定小数位再赋值，否则默认 2 位会把 0.937 之类的值四舍五入
            spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setToolTip(tip)
        return spin

    # ---------- 模型与数据路径 ----------
    def _build_model_card(self):
        card, card_layout = theme.card()
        card_layout.addLayout(theme.section_row("📦", "模型与数据"))

        grid = theme.grid(6)

        self.model_path = QLineEdit(os.path.join(get_weights_dir(), "yolo26n.pt"))
        grid.addWidget(theme.field("模型路径:", theme.path_row(
            self.model_path, "",
            lambda: self._select_file(self.model_path, "模型文件 (*.pt)"), "📁")),
            0, 0, 1, 3)

        yaml_path = os.path.join(cfg_root, "models", "26", "yolo26.yaml")
        self.yaml_edit = QLineEdit(yaml_path)
        grid.addWidget(theme.field("模型YAML:", theme.path_row(
            self.yaml_edit, "",
            lambda: self._select_file(self.yaml_edit, "YAML文件 (*.yaml)"), "📄")),
            0, 3, 1, 3)

        data_path = os.path.join(project_root, "datasets", "coco8", "coco8.yaml")
        self.data_edit = QLineEdit(data_path)
        grid.addWidget(theme.field("数据集配置:", theme.path_row(
            self.data_edit, "",
            lambda: self._select_file(self.data_edit, "YAML文件 (*.yaml)"), "📂")),
            1, 0, 1, 3)

        hyp_path = os.path.join(cfg_root, "default.yaml")
        self.hyp_yaml_edit = QLineEdit(hyp_path)
        grid.addWidget(theme.field("超参数YAML:", theme.path_row(
            self.hyp_yaml_edit, "",
            lambda: self._select_file(self.hyp_yaml_edit, "YAML文件 (*.yaml)"), "📄")),
            1, 3, 1, 3)

        project_path = os.path.join(project_root, "runs", "train")
        self.project = QLineEdit(project_path)
        self.project.setToolTip("project")
        grid.addWidget(theme.field("项目目录:", self.project), 2, 0, 1, 3)

        self.name = QLineEdit("exp")
        self.name.setToolTip("name")
        grid.addWidget(theme.field("实验名称:", self.name), 2, 3, 1, 2)

        btn_load_hyp = QPushButton("🔄 加载超参数")
        theme.set_role(btn_load_hyp, "amber")
        btn_load_hyp.setToolTip("从上方 YAML 文件导入超参数（不覆盖模型/数据集路径）")
        btn_load_hyp.clicked.connect(self._load_hyp_from_yaml)
        grid.addWidget(btn_load_hyp, 2, 5)

        card_layout.addLayout(grid)
        return card

    # ---------- 训练超参数 ----------
    def _build_hyp_card(self):
        card, card_layout = theme.card()
        card_layout.addLayout(theme.section_row("⚙️", "训练参数"))

        grid = theme.grid(6)

        self.epochs = self._ispin(100, "epochs", 1, 10000)
        grid.addWidget(theme.field("训练轮数:", self.epochs), 0, 0)

        self.batch_size = self._ispin(16, "batch")
        grid.addWidget(theme.field("批次大小:", self.batch_size), 0, 1)

        self.imgsz = self._ispin(640, "imgsz", 32, 4096)
        grid.addWidget(theme.field("图像尺寸:", self.imgsz), 0, 2)

        self.device = QLineEdit("0")
        self.device.setToolTip("device")
        grid.addWidget(theme.field("设备:", self.device), 0, 3)

        self.optimizer = QComboBox()
        self.optimizer.addItems(["SGD", "Adam", "AdamW", "NAdam"])
        self.optimizer.setToolTip("optimizer")
        grid.addWidget(theme.field("优化器:", self.optimizer), 0, 4)

        self.seed = self._ispin(0, "seed")
        grid.addWidget(theme.field("随机种子:", self.seed), 0, 5)

        self.lr0 = self._dspin(0.01, "lr0", 6)
        grid.addWidget(theme.field("初始学习率:", self.lr0), 1, 0)

        self.lrf = self._dspin(0.01, "lrf", 6)
        grid.addWidget(theme.field("最终学习率:", self.lrf), 1, 1)

        self.momentum = self._dspin(0.937, "momentum", 6)
        grid.addWidget(theme.field("动量:", self.momentum), 1, 2)

        self.patience = self._ispin(100, "patience", 0, 10000)
        grid.addWidget(theme.field("早停耐心:", self.patience), 1, 3)

        self.save_period = self._ispin(50, "save_period")
        grid.addWidget(theme.field("保存周期:", self.save_period), 1, 4)

        self.close_mosaic = self._ispin(10, "close_mosaic")
        grid.addWidget(theme.field("关闭马赛克:", self.close_mosaic), 1, 5)

        card_layout.addLayout(grid)
        return card

    # ---------- 数据增强 ----------
    def _build_aug_card(self):
        card, card_layout = theme.card()
        card_layout.addLayout(theme.section_row("🎨", "数据增强"))

        grid = theme.grid(6)

        self.hsv_v = self._dspin(0.4, "hsv_v")
        grid.addWidget(theme.field("色调:", self.hsv_v), 0, 0)

        self.degrees = self._dspin(0.0, "degrees")
        grid.addWidget(theme.field("旋转:", self.degrees), 0, 1)

        self.translate = self._dspin(0.1, "translate")
        grid.addWidget(theme.field("平移:", self.translate), 0, 2)

        self.scale = self._dspin(0.5, "scale")
        grid.addWidget(theme.field("缩放:", self.scale), 0, 3)

        self.shear = self._dspin(0.0, "shear")
        grid.addWidget(theme.field("剪切:", self.shear), 0, 4)

        self.perspective = self._dspin(0.0, "perspective")
        grid.addWidget(theme.field("透视:", self.perspective), 0, 5)

        self.flipud = self._dspin(0.0, "flipud")
        grid.addWidget(theme.field("上下翻转:", self.flipud), 1, 0)

        self.fliplr = self._dspin(0.5, "fliplr")
        grid.addWidget(theme.field("左右翻转:", self.fliplr), 1, 1)

        self.mosaic = self._dspin(1.0, "mosaic")
        grid.addWidget(theme.field("马赛克:", self.mosaic), 1, 2)

        self.mixup = self._dspin(0.0, "mixup")
        grid.addWidget(theme.field("混合:", self.mixup), 1, 3)

        self.copy_paste = self._dspin(0.0, "copy_paste")
        grid.addWidget(theme.field("复制粘贴:", self.copy_paste), 1, 4)

        self.erasing = self._dspin(0.4, "erasing")
        grid.addWidget(theme.field("擦除:", self.erasing), 1, 5)

        self.crop_fraction = self._dspin(1.0, "crop_fraction")
        grid.addWidget(theme.field("裁剪比例:", self.crop_fraction), 2, 0)

        self.auto_augment = QLineEdit("randaugment")
        self.auto_augment.setToolTip("auto_augment")
        grid.addWidget(theme.field("自动增强:", self.auto_augment), 2, 1, 1, 2)

        self.fraction = self._dspin(1.0, "fraction")
        grid.addWidget(theme.field("训练集比例:", self.fraction), 2, 3)

        self.iou = self._dspin(0.7, "iou")
        grid.addWidget(theme.field("IoU阈值:", self.iou), 2, 4)

        self.max_det = self._ispin(300, "max_det", 1, 10000)
        grid.addWidget(theme.field("最大检测数:", self.max_det), 2, 5)

        card_layout.addLayout(grid)
        return card

    # ---------- 开关选项 ----------
    def _build_options_card(self):
        card, card_layout = theme.card()
        card_layout.addLayout(theme.section_row("🔧", "其他选项"))

        grid = theme.grid(6)
        specs = [
            ('cos_lr', '余弦学习率', True), ('amp', '混合精度训练', True),
            ('save', '保存权重', True), ('save_txt', '保存标签', False),
            ('save_conf', '保存置信度', False), ('save_crop', '保存裁剪图', False),
            ('show_labels', '显示标签', True), ('show_conf', '显示置信度', True),
            ('plot', '绘制图表', True), ('val', '验证集评估', True),
            ('visualize', '可视化', False), ('verbose', '详细输出', True),
        ]
        for index, (key, text, checked) in enumerate(specs):
            cb = QCheckBox(text)
            cb.setToolTip(key)
            cb.setChecked(checked)
            setattr(self, key, cb)
            grid.addWidget(cb, index // 6, index % 6)

        card_layout.addLayout(grid)
        return card

    # ---------- 运行控制 ----------
    def _build_run_card(self):
        card, card_layout = theme.card()
        row = QHBoxLayout()
        row.setSpacing(12)

        self.train_btn = QPushButton("🚀 开始训练")
        theme.set_role(self.train_btn, "success")
        self.train_btn.clicked.connect(self.start_train)
        row.addWidget(self.train_btn)

        self.stop_btn = QPushButton("⏹️ 停止训练")
        theme.set_role(self.stop_btn, "danger")
        self.stop_btn.clicked.connect(self.stop_train)
        self.stop_btn.setEnabled(False)
        row.addWidget(self.stop_btn)

        self.save_config_btn = QPushButton("💾 保存配置")
        theme.set_role(self.save_config_btn, "primary")
        self.save_config_btn.setToolTip("将当前所有参数和路径保存为YAML文件，便于下次导入")
        self.save_config_btn.clicked.connect(self._save_params_to_file)
        row.addWidget(self.save_config_btn)

        self.import_config_btn = QPushButton("📥 导入配置")
        theme.set_role(self.import_config_btn, "primary")
        self.import_config_btn.setToolTip("从YAML文件一键导入所有参数和路径")
        self.import_config_btn.clicked.connect(self._import_params_from_file)
        row.addWidget(self.import_config_btn)

        row.addStretch()
        card_layout.addLayout(row)
        return card

    def _build_log_card(self):
        card, card_layout = theme.card()
        card_layout.addLayout(theme.section_row("📋", "训练日志"))

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        theme.set_role(self.log_text, "log")
        card_layout.addWidget(self.log_text)
        return card

    def _select_file(self, line_edit, filter_str):
        file_path = QFileDialog.getOpenFileName(self, "选择文件", "", filter_str)[0]
        if file_path:
            line_edit.setText(file_path)

    # ---------- 参数保存 / 导入 ----------
    # 控件名 -> YAML 键名 的映射，统一管理保存与导入逻辑
    _PATH_FIELDS = [
        ('model_path', 'model'),
        ('yaml_edit', 'yaml'),
        ('data_edit', 'data'),
        ('hyp_yaml_edit', 'hyp_yaml'),
        ('device', 'device'),
        ('project', 'project'),
        ('name', 'name'),
        ('auto_augment', 'auto_augment'),
    ]
    _INT_SPIN_FIELDS = [
        'epochs', 'batch_size', 'imgsz', 'patience', 'save_period',
        'seed', 'close_mosaic', 'max_det',
    ]
    _DOUBLE_SPIN_FIELDS = [
        'lr0', 'lrf', 'momentum',
        'hsv_v', 'degrees', 'translate', 'scale', 'shear', 'perspective',
        'flipud', 'fliplr', 'mosaic', 'mixup', 'copy_paste',
        'erasing', 'crop_fraction', 'fraction', 'iou',
    ]
    _COMBO_FIELDS = [('optimizer', 'optimizer')]
    _CHECKBOX_FIELDS = [
        'cos_lr', 'amp', 'save', 'save_txt', 'save_conf', 'save_crop',
        'show_labels', 'show_conf', 'plot', 'val', 'visualize', 'verbose',
    ]
    # batch_size / save_period 在 YAML 中使用 batch / save_period 键名以兼容 ultralytics 默认配置
    _SPIN_KEY_ALIAS = {'batch_size': 'batch'}

    def _collect_all_params(self):
        """收集所有训练参数和路径，返回字典（可用于保存或直接训练）"""
        params = {}
        for widget_name, key in self._PATH_FIELDS:
            params[key] = getattr(self, widget_name).text()
        for widget_name in self._INT_SPIN_FIELDS:
            key = self._SPIN_KEY_ALIAS.get(widget_name, widget_name)
            params[key] = getattr(self, widget_name).value()
        for widget_name in self._DOUBLE_SPIN_FIELDS:
            params[widget_name] = getattr(self, widget_name).value()
        for widget_name, key in self._COMBO_FIELDS:
            params[key] = getattr(self, widget_name).currentText()
        for widget_name in self._CHECKBOX_FIELDS:
            params[widget_name] = getattr(self, widget_name).isChecked()
        return params

    def _apply_params(self, data, include_paths=True):
        """将字典中的参数应用到所有控件（仅应用存在的键）

        include_paths=False 时跳过 model/yaml/data/hyp_yaml 等路径字段，
        以兼容 ultralytics 默认配置 YAML 的“仅加载超参数”场景。
        """
        # 路径 / 文本
        for widget_name, key in self._PATH_FIELDS:
            if not include_paths and key in ('model', 'yaml', 'data', 'hyp_yaml'):
                continue
            if key in data:
                getattr(self, widget_name).setText(str(data[key]))
        # 整数 SpinBox
        for widget_name in self._INT_SPIN_FIELDS:
            key = self._SPIN_KEY_ALIAS.get(widget_name, widget_name)
            if key in data and data[key] is not None:
                getattr(self, widget_name).setValue(int(data[key]))
        # 浮点 SpinBox
        for widget_name in self._DOUBLE_SPIN_FIELDS:
            if widget_name in data and data[widget_name] is not None:
                getattr(self, widget_name).setValue(float(data[widget_name]))
        # 下拉框
        for widget_name, key in self._COMBO_FIELDS:
            if key in data:
                index = getattr(self, widget_name).findText(str(data[key]), Qt.MatchFixedString)
                if index >= 0:
                    getattr(self, widget_name).setCurrentIndex(index)
        # 复选框
        for widget_name in self._CHECKBOX_FIELDS:
            if widget_name in data:
                getattr(self, widget_name).setChecked(bool(data[widget_name]))

    def _get_default_config_dir(self):
        default_dir = os.path.join(project_root, 'configs', 'train_configs')
        try:
            os.makedirs(default_dir, exist_ok=True)
        except Exception:
            pass
        return default_dir

    def _save_params_to_file(self):
        """保存当前所有参数和路径到 YAML 文件，便于下次一键导入"""
        from datetime import datetime
        default_dir = self._get_default_config_dir()
        default_name = f"train_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        default_path = os.path.join(default_dir, default_name)

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存训练参数配置", default_path, "YAML文件 (*.yaml *.yml)"
        )
        if not file_path:
            return

        try:
            params = self._collect_all_params()
            # 附加保存时间戳，便于追溯
            params['__saved_at__'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(params, f, allow_unicode=True,
                          default_flow_style=False, sort_keys=False)
            self.log(f"已保存参数配置到: {file_path}")
            QMessageBox.information(self, "保存成功",
                f"参数配置已保存到:\n{file_path}\n\n下次可通过「导入配置」一键载入。")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存参数配置失败: {str(e)}")

    def _import_params_from_file(self):
        """从 YAML 文件一键导入所有参数和路径"""
        default_dir = self._get_default_config_dir()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "导入训练参数配置", default_dir, "YAML文件 (*.yaml *.yml)"
        )
        if not file_path:
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                QMessageBox.warning(self, "错误", "配置文件格式无效，应为键值对形式的 YAML")
                return
            self._apply_params(data)
            self.log(f"已从 {file_path} 导入参数配置")
            QMessageBox.information(self, "导入成功",
                f"参数配置已从以下文件导入:\n{file_path}")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"导入参数配置失败: {str(e)}")

    def _load_hyp_from_yaml(self):
        hyp_path = self.hyp_yaml_edit.text().strip()
        if not hyp_path or not os.path.exists(hyp_path):
            QMessageBox.warning(self, "提示", "请选择有效的YAML文件")
            return

        try:
            with open(hyp_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                QMessageBox.warning(self, "错误", "YAML文件格式无效")
                return
            # 仅加载超参数，不覆盖 model/yaml/data/hyp_yaml 等路径字段
            self._apply_params(data, include_paths=False)
            self.log(f"已从 {hyp_path} 加载超参数")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载超参数失败: {str(e)}")

    def log(self, msg):
        QMetaObject.invokeMethod(self.log_text, "append", Qt.QueuedConnection,
                                Q_ARG(str, msg))

    def start_train(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "提示", "任务正在运行中")
            return

        data_yaml = self.data_edit.text().strip()
        if not data_yaml or not os.path.exists(data_yaml):
            QMessageBox.warning(self, "提示", "请选择有效的数据集配置文件")
            return

        params = {
            'model': self.model_path.text(),
            'yaml': self.yaml_edit.text(),
            'data': data_yaml,
            'epochs': self.epochs.value(),
            'batch': self.batch_size.value(),
            'imgsz': self.imgsz.value(),
            'device': self.device.text(),
            'patience': self.patience.value(),
            'save_period': self.save_period.value(),
            'project': self.project.text(),
            'name': self.name.text(),
            'optimizer': self.optimizer.currentText(),
            'lr0': self.lr0.value(),
            'lrf': self.lrf.value(),
            'momentum': self.momentum.value(),
            'seed': self.seed.value(),
            'hsv_v': self.hsv_v.value(),
            'degrees': self.degrees.value(),
            'translate': self.translate.value(),
            'scale': self.scale.value(),
            'shear': self.shear.value(),
            'perspective': self.perspective.value(),
            'flipud': self.flipud.value(),
            'fliplr': self.fliplr.value(),
            'mosaic': self.mosaic.value(),
            'mixup': self.mixup.value(),
            'copy_paste': self.copy_paste.value(),
            'auto_augment': self.auto_augment.text(),
            'erasing': self.erasing.value(),
            'crop_fraction': self.crop_fraction.value(),
            'close_mosaic': self.close_mosaic.value(),
            'fraction': self.fraction.value(),
            'iou': self.iou.value(),
            'max_det': self.max_det.value(),
            'cos_lr': self.cos_lr.isChecked(),
            'amp': self.amp.isChecked(),
            'save': self.save.isChecked(),
            'save_txt': self.save_txt.isChecked(),
            'save_conf': self.save_conf.isChecked(),
            'save_crop': self.save_crop.isChecked(),
            'plot': self.plot.isChecked(),
            'val': self.val.isChecked(),
            'visualize': self.visualize.isChecked(),
            'verbose': self.verbose.isChecked(),
        }

        self.worker = YOLOTrainWorker(params)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_train_finished)
        self.worker.start()
        self.stop_btn.setEnabled(True)

    def stop_train(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.log("正在停止训练...")

    def on_train_finished(self, success, msg):
        self.stop_btn.setEnabled(False)
        if success:
            self.log(f"[完成] {msg}")
        else:
            self.log(f"[错误] {msg}")
