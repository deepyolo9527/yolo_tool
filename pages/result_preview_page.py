
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit, QPushButton,
                             QHBoxLayout, QVBoxLayout, QTextEdit,
                             QMessageBox, QFileDialog, QListWidget, QListWidgetItem,
                             QComboBox, QSplitter, QTabWidget, QCheckBox)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from pages.base_page import BasePage
from utils import theme
from utils.path_helpers import get_app_dir
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import tempfile


project_root = get_app_dir()


class ResultPreviewPage(BasePage):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = self.get_content_layout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        layout.addWidget(theme.header("📈", "结果预览", "查看训练实验结果，对比不同训练配置的效果"))

        top_card, top_card_layout = theme.card()
        top_layout = QHBoxLayout()
        top_layout.setSpacing(10)

        self.project_edit = QLineEdit(os.path.join(project_root, "runs", "train"))
        top_layout.addWidget(self.project_edit)

        browse_btn = QPushButton("📁 浏览")
        theme.set_role(browse_btn, "primary")
        browse_btn.clicked.connect(lambda: self.browse_dir(self.project_edit))
        top_layout.addWidget(browse_btn)

        refresh_btn = QPushButton("🔄 刷新")
        theme.set_role(refresh_btn, "success")
        refresh_btn.clicked.connect(self.load_experiments)
        top_layout.addWidget(refresh_btn)

        top_card_layout.addLayout(top_layout)
        layout.addWidget(top_card)

        main_splitter = QSplitter(Qt.Horizontal)

        left_card, left_layout = theme.card()
        left_layout.setSpacing(15)

        left_layout.addLayout(theme.section_row("📋", "实验列表"))

        self.exp_list = QListWidget()
        self.exp_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.exp_list.currentItemChanged.connect(self.on_exp_selected)
        self.exp_list.itemChanged.connect(self._update_check_info)
        self.exp_list.itemSelectionChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self.exp_list, 1)

        self.check_info = QLabel("已勾选 0 个")
        theme.set_role(self.check_info, "hint")
        left_layout.addWidget(self.check_info)

        pick_row = QHBoxLayout()
        pick_row.setSpacing(8)
        check_all_btn = QPushButton("☑ 全选")
        theme.set_role(check_all_btn, "primary")
        check_all_btn.clicked.connect(lambda: self._set_all_checked(True))
        pick_row.addWidget(check_all_btn)

        uncheck_btn = QPushButton("☐ 清空勾选")
        theme.set_role(uncheck_btn, "primary")
        uncheck_btn.clicked.connect(lambda: self._set_all_checked(False))
        pick_row.addWidget(uncheck_btn)
        left_layout.addLayout(pick_row)

        self.compare_btn = QPushButton("📊 对比勾选实验")
        theme.set_role(self.compare_btn, "amber")
        self.compare_btn.clicked.connect(self.on_compare)
        left_layout.addWidget(self.compare_btn)

        main_splitter.addWidget(left_card)

        content_card, content_layout = theme.card()

        self.tabs = QTabWidget()

        config_tab = QWidget()
        config_layout = QVBoxLayout(config_tab)
        config_layout.addLayout(theme.section_row("⚙️", "配置信息"))

        self.config_text = QTextEdit()
        self.config_text.setReadOnly(True)
        theme.set_role(self.config_text, "log")
        config_layout.addWidget(self.config_text)
        self.tabs.addTab(config_tab, "⚙️ 配置信息")

        chart_tab = QWidget()
        chart_layout = QVBoxLayout(chart_tab)
        chart_layout.addLayout(theme.section_row("📈", "结果图表"))

        self.chart_combo = QComboBox()
        chart_layout.addWidget(self.chart_combo)

        self.chart_label = QLabel("图表预览区域")
        self.chart_label.setAlignment(Qt.AlignCenter)
        self.chart_label.setMinimumHeight(360)
        theme.set_role(self.chart_label, "canvas")
        chart_layout.addWidget(self.chart_label)
        self.tabs.addTab(chart_tab, "📈 结果图表")

        compare_tab = QWidget()
        compare_layout = QVBoxLayout(compare_tab)
        compare_layout.addLayout(theme.section_row("🔍", "曲线对比"))

        compare_top_layout = QHBoxLayout()
        compare_top_layout.setSpacing(10)

        metric_label = QLabel("选择对比指标:")
        compare_top_layout.addWidget(metric_label)

        self.metric_combo = QComboBox()
        self.metric_combo.addItems(['box_loss', 'cls_loss', 'dfl_loss', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss', 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)'])
        self.metric_combo.setMinimumWidth(200)
        self.metric_combo.currentIndexChanged.connect(self.on_metric_change)
        compare_top_layout.addWidget(self.metric_combo)
        compare_top_layout.addStretch()
        compare_layout.addLayout(compare_top_layout)

        self.compare_label = QLabel("曲线对比预览区域")
        self.compare_label.setAlignment(Qt.AlignCenter)
        self.compare_label.setMinimumHeight(360)
        theme.set_role(self.compare_label, "canvas")
        compare_layout.addWidget(self.compare_label)

        save_frame, save_layout = theme.card(margin=14, spacing=10)

        auto_save_checkbox = QCheckBox("自动保存对比图到本地（每次生成对比图时自动保存）")
        auto_save_checkbox.stateChanged.connect(self.on_auto_save_toggled)
        save_layout.addWidget(auto_save_checkbox)
        self.auto_save_checkbox = auto_save_checkbox

        path_layout = QHBoxLayout()
        path_layout.setSpacing(8)

        path_label = QLabel("保存路径:")
        path_layout.addWidget(path_label)

        default_save_dir = os.path.join(project_root, "compare_results")
        self.save_path_edit = QLineEdit(default_save_dir)
        path_layout.addWidget(self.save_path_edit, 1)

        save_browse_btn = QPushButton("📁 选择")
        theme.set_role(save_browse_btn, "primary")
        save_browse_btn.clicked.connect(lambda: self.browse_dir(self.save_path_edit))
        path_layout.addWidget(save_browse_btn)

        save_layout.addLayout(path_layout)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self.save_btn = QPushButton("💾 另存为图片")
        theme.set_role(self.save_btn, "primary")
        self.save_btn.clicked.connect(self.save_comparison_image)
        self.save_btn.setEnabled(False)
        btn_layout.addWidget(self.save_btn)

        self.save_all_btn = QPushButton("📦 批量保存全部对比图")
        theme.set_role(self.save_all_btn, "success")
        self.save_all_btn.clicked.connect(self.save_all_comparisons)
        btn_layout.addWidget(self.save_all_btn)

        btn_layout.addStretch()
        save_layout.addLayout(btn_layout)

        compare_layout.addWidget(save_frame)
        self.tabs.addTab(compare_tab, "🔍 曲线对比")

        content_layout.addWidget(self.tabs)
        main_splitter.addWidget(content_card)

        layout.addWidget(main_splitter, 1)

        self.load_experiments()

    def browse_dir(self, edit_widget):
        dir_path = QFileDialog.getExistingDirectory(self, "选择项目目录")
        if dir_path:
            edit_widget.setText(dir_path)

    def load_experiments(self):
        project_dir = self.project_edit.text().strip()
        self.exp_list.clear()

        if not os.path.exists(project_dir):
            self.log(f"项目目录不存在: {project_dir}")
            return

        exp_dirs = []
        for item in os.listdir(project_dir):
            item_path = os.path.join(project_dir, item)
            if os.path.isdir(item_path):
                args_file = os.path.join(item_path, "args.yaml")
                if os.path.exists(args_file):
                    exp_dirs.append((item, item_path))

        exp_dirs.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)

        for name, path in exp_dirs:
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, path)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.exp_list.addItem(item)

        self._update_check_info()
        self.log(f"加载了 {len(exp_dirs)} 个实验")

    # ---------- 勾选管理 ----------
    def _update_check_info(self, *_):
        n = len(self._checked_experiments())
        self.check_info.setText(f"已勾选 {n} 个")
        self.compare_btn.setText(f"📊 对比勾选实验 ({n})")

    def _checked_experiments(self):
        return [(self.exp_list.item(i).text(), self.exp_list.item(i).data(Qt.UserRole))
                for i in range(self.exp_list.count())
                if self.exp_list.item(i).checkState() == Qt.Checked]

    def _set_all_checked(self, checked):
        state = Qt.Checked if checked else Qt.Unchecked
        for i in range(self.exp_list.count()):
            self.exp_list.item(i).setCheckState(state)
        self._update_check_info()

    def _on_selection_changed(self):
        """鼠标框选（一次选中多行）即视为加入勾选，单选浏览配置不影响勾选"""
        selected = self.exp_list.selectedItems()
        if len(selected) < 2:
            return
        for it in selected:
            it.setCheckState(Qt.Checked)
        self._update_check_info()

    def on_exp_selected(self, current, previous):
        if not current:
            return

        exp_path = current.data(Qt.UserRole)
        self.load_exp_details(exp_path)

    def load_exp_details(self, exp_path):
        self.load_config(exp_path)
        self.load_charts(exp_path)

    def load_config(self, exp_path):
        import yaml
        args_file = os.path.join(exp_path, "args.yaml")
        if os.path.exists(args_file):
            try:
                with open(args_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.config_text.setText(content)
            except Exception as e:
                self.config_text.setText(f"读取配置失败: {str(e)}")
        else:
            self.config_text.setText("配置文件不存在")

    def load_charts(self, exp_path):
        chart_files = []
        img_exts = ['.png', '.jpg', '.jpeg']
        for f in os.listdir(exp_path):
            name, ext = os.path.splitext(f)
            if ext.lower() in img_exts:
                chart_files.append((name, os.path.join(exp_path, f)))

        self.chart_combo.clear()
        self.chart_combo.addItem("选择图表")
        for name, path in chart_files:
            self.chart_combo.addItem(name, path)

        if chart_files:
            self.chart_combo.currentIndexChanged.connect(self.on_chart_selected)
            self.on_chart_selected(1)

    def on_chart_selected(self, index):
        if index == 0:
            self.chart_label.setText("图表预览区域")
            return

        chart_path = self.chart_combo.itemData(index)
        if chart_path and os.path.exists(chart_path):
            from PyQt5.QtGui import QPixmap
            pixmap = QPixmap(chart_path)
            if not pixmap.isNull():
                self.chart_label.setPixmap(pixmap.scaled(self.chart_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.chart_label.setText("无法加载图片")
        else:
            self.chart_label.setText("图表文件不存在")

    def on_compare(self):
        exp_paths = self._checked_experiments()
        if len(exp_paths) < 2:
            QMessageBox.warning(self, "提示", "请至少勾选两个实验进行对比")
            return

        self.plot_comparison(exp_paths, self.metric_combo.currentText())

    def on_metric_change(self, index):
        exp_paths = self._checked_experiments()
        if len(exp_paths) >= 2:
            self.plot_comparison(exp_paths, self.metric_combo.currentText())

    def plot_comparison(self, exp_paths, metric):
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        for font_name in ['Microsoft YaHei', 'SimHei', 'SimSun']:
            try:
                fm.findfont(font_name, fallback_to_default=False)
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
                break
            except Exception:
                continue

        plt.figure(figsize=(10, 6))

        for i, (exp_name, exp_path) in enumerate(exp_paths):
            csv_file = os.path.join(exp_path, "results.csv")
            if not os.path.exists(csv_file):
                continue

            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                if not lines:
                    continue

                headers = lines[0].strip().split(',')
                if metric not in headers:
                    continue

                metric_idx = headers.index(metric)
                x_data = []
                y_data = []

                for line_num, line in enumerate(lines[1:], start=1):
                    line = line.strip()
                    if line:
                        parts = line.split(',')
                        if len(parts) > metric_idx:
                            try:
                                y_data.append(float(parts[metric_idx]))
                                x_data.append(line_num)
                            except:
                                pass

                if x_data:
                    plt.plot(x_data, y_data, label=exp_name, color=colors[i % len(colors)], linewidth=2)
            except Exception as e:
                self.log(f"读取 {exp_name} 的指标失败: {str(e)}")

        plt.title(f'{metric} 对比曲线')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, alpha=0.3)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name

        plt.savefig(temp_path, dpi=100, bbox_inches='tight')
        plt.close()

        if os.path.exists(temp_path):
            self.current_comparison_path = temp_path
            self.current_metric = metric
            self.current_exp_paths = exp_paths

            pixmap = QPixmap(temp_path)
            if not pixmap.isNull():
                self.compare_label.setPixmap(pixmap.scaled(self.compare_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.save_btn.setEnabled(True)
            else:
                self.compare_label.setText("无法生成对比图")
        else:
            self.compare_label.setText("生成对比图失败")
            self.save_btn.setEnabled(False)

        if hasattr(self, 'auto_save_checkbox') and self.auto_save_checkbox.isChecked():
            self.auto_save_comparison(temp_path, metric, exp_paths)

    def on_auto_save_toggled(self, state):
        if state == Qt.Checked and hasattr(self, 'current_comparison_path') and self.current_comparison_path:
            self.auto_save_comparison(
                self.current_comparison_path,
                getattr(self, 'current_metric', 'unknown'),
                getattr(self, 'current_exp_paths', [])
            )

    def generate_save_filename(self, metric, exp_paths):
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_names = '_'.join([name.replace(' ', '') for name, _ in exp_paths[:3]])
        if len(exp_paths) > 3:
            exp_names += f'_等{len(exp_paths)}组'
        safe_metric = metric.replace('/', '_').replace(' ', '_').replace('(B)', '').replace('-', '_')
        filename = f'comparison_{safe_metric}_{exp_names}_{timestamp}.png'
        return filename

    def auto_save_comparison(self, source_path, metric, exp_paths):
        save_dir = self.save_path_edit.text().strip()
        if not save_dir:
            save_dir = os.path.join(project_root, "compare_results")
            self.save_path_edit.setText(save_dir)

        os.makedirs(save_dir, exist_ok=True)

        filename = self.generate_save_filename(metric, exp_paths)
        target_path = os.path.join(save_dir, filename)

        try:
            pixmap = QPixmap(source_path)
            if not pixmap.isNull():
                pixmap.save(target_path, 'PNG')
                self.log(f"自动保存对比图: {target_path}")
        except Exception as e:
            self.log(f"自动保存失败: {str(e)}")

    def save_comparison_image(self):
        if not hasattr(self, 'current_comparison_path') or not self.current_comparison_path:
            QMessageBox.warning(self, "提示", "没有可保存的对比图")
            return

        save_dir = self.save_path_edit.text().strip()
        if not save_dir:
            save_dir = os.path.join(project_root, "compare_results")

        default_name = self.generate_save_filename(
            getattr(self, 'current_metric', 'unknown'),
            getattr(self, 'current_exp_paths', [])
        )

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存对比图",
            os.path.join(save_dir, default_name),
            "PNG图片 (*.png);;所有文件 (*)"
        )

        if file_path:
            try:
                pixmap = QPixmap(self.current_comparison_path)
                if not pixmap.isNull():
                    pixmap.save(file_path, 'PNG')
                    self.log(f"对比图已保存: {file_path}")
                    QMessageBox.information(self, "保存成功", f"对比图已保存到:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存对比图时出错:\n{str(e)}")

    def save_all_comparisons(self):
        exp_paths = self._checked_experiments()
        if len(exp_paths) < 2:
            QMessageBox.warning(self, "提示", "请至少勾选两个实验进行对比")
            return

        save_dir = self.save_path_edit.text().strip()
        if not save_dir:
            save_dir = os.path.join(project_root, "compare_results")
            self.save_path_edit.setText(save_dir)

        os.makedirs(save_dir, exist_ok=True)

        metrics = ['box_loss', 'val/box_loss', 'metrics/mAP50(B)', 'metrics/precision(B)']
        success_count = 0
        failed_metrics = []

        for metric in metrics:
            try:
                self.plot_comparison(exp_paths, metric)
                if hasattr(self, 'current_comparison_path') and self.current_comparison_path:
                    filename = self.generate_save_filename(metric, exp_paths)
                    target_path = os.path.join(save_dir, filename)
                    pixmap = QPixmap(self.current_comparison_path)
                    if not pixmap.isNull():
                        pixmap.save(target_path, 'PNG')
                        self.log(f"已保存: {target_path}")
                        success_count += 1
            except Exception as e:
                failed_metrics.append(metric)
                self.log(f"保存 {metric} 失败: {str(e)}")

        msg = f"批量保存完成！\n成功: {success_count} 张对比图\n保存位置: {save_dir}"
        if failed_metrics:
            msg += f"\n\n失败指标: {', '.join(failed_metrics)}"
        QMessageBox.information(self, "批量保存完成", msg)

    def log(self, msg):
        print(msg)
