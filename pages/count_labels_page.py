
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (QLabel, QLineEdit, QPushButton,
                             QHBoxLayout, QFormLayout, QTableWidget,
                             QTableWidgetItem, QHeaderView, QTextEdit, QMessageBox,
                             QFileDialog, QComboBox, QSizePolicy)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QMetaObject, Q_ARG
from pages.base_page import BasePage
from utils import theme
from pathlib import Path
from collections import defaultdict
import os
import tempfile
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


# 宽高比形态分档（宽/高）。名称里不能出现 < > ，否则会当成 HTML 标签被吃掉
AR_BUCKETS = [(0.0, 1 / 3, "横长≤0.33"),
              (1 / 3, 0.8, "偏宽0.33~0.8"),
              (0.8, 1.25, "近方0.8~1.25"),
              (1.25, 3.0, "偏高1.25~3"),
              (3.0, float('inf'), "竖长≥3")]

# 面积占比分档（归一化面积 w*h）
AREA_BUCKETS = [(0.0, 0.01, "小目标≤1%"),
                (0.01, 0.05, "中小1~5%"),
                (0.05, 0.15, "中大5~15%"),
                (0.15, 1.01, "大目标≥15%")]

CHART_TYPES = [("类别数量分布", "count"),
               ("宽高比分布直方图", "ar_hist"),
               ("宽/高分布直方图", "wh_hist"),
               ("按类别 宽-高 散点", "wh_scatter"),
               ("按类别宽高比箱线", "ar_box")]

SPLIT_NAMES = {'train': '训练集', 'valid': '验证集', 'val': '验证集', 'test': '测试集'}


class CountLabelsPage(BasePage):
    def __init__(self):
        super().__init__()
        self._geo = {}            # {'all': {cls: arr}, 'splits': {name: {cls: arr}}}
        self._class_names = {}
        self._chart_pixmap = None
        self.init_ui()

    def init_ui(self):
        layout = self.get_content_layout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)

        layout.addWidget(theme.header("📊", "标签统计", "统计数据集标签数量与目标框宽高分布，支持train/valid/test分割统计"))

        input_card, input_card_layout = theme.card()
        input_layout = QFormLayout()
        input_layout.setSpacing(14)
        input_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        input_card_layout.addLayout(input_layout)

        dir_layout = QHBoxLayout()
        dir_layout.setSpacing(10)
        self.input_dir_edit = QLineEdit()
        self.input_dir_edit.setPlaceholderText("请选择标签目录")
        browse_btn = QPushButton("📁 浏览")
        theme.set_role(browse_btn, "primary")
        browse_btn.clicked.connect(self.browse_input_dir)
        dir_layout.addWidget(self.input_dir_edit)
        dir_layout.addWidget(browse_btn)
        input_layout.addRow("标签目录:", dir_layout)

        yaml_layout = QHBoxLayout()
        yaml_layout.setSpacing(10)
        self.yaml_edit = QLineEdit()
        self.yaml_edit.setPlaceholderText("可选：数据集YAML文件")
        yaml_browse_btn = QPushButton("📄 选择")
        theme.set_role(yaml_browse_btn, "primary")
        yaml_browse_btn.clicked.connect(self.browse_yaml)
        yaml_layout.addWidget(self.yaml_edit)
        yaml_layout.addWidget(yaml_browse_btn)
        input_layout.addRow("数据集YAML:", yaml_layout)

        layout.addWidget(input_card)

        btn_layout = QHBoxLayout()
        self.action_btn = QPushButton("🚀 开始统计")
        theme.set_role(self.action_btn, "success")
        self.action_btn.clicked.connect(self.start_count)
        btn_layout.addWidget(self.action_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        result_card, result_layout = theme.card()
        result_layout.addLayout(theme.section_row("📈", "类别数量统计"))

        self.result_table = QTableWidget()
        self.result_table.setColumnCount(3)
        self.result_table.setHorizontalHeaderLabels(["类别ID", "类别名称", "数量"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        result_layout.addWidget(self.result_table)
        layout.addWidget(result_card, 3)

        geo_card, geo_layout = theme.card()
        geo_layout.addLayout(theme.section_row("📐", "宽高比与尺寸分布"))

        self.summary_label = QLabel("统计后显示每个类别的宽、高与宽高比分布范围")
        self.summary_label.setTextFormat(Qt.RichText)
        self.summary_label.setWordWrap(True)
        # 固定纵向策略：滚动区内自动换行的标签会把高度撑到整页宽度之上
        self.summary_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.summary_label.setMinimumHeight(6 * 22)
        theme.set_role(self.summary_label, "hint")
        geo_layout.addWidget(self.summary_label)

        self.geo_table = QTableWidget()
        self.geo_table.setColumnCount(8)
        self.geo_table.setHorizontalHeaderLabels(
            ["类别ID", "类别名称", "数量", "宽 均值/中位/P5~P95", "高 均值/中位/P5~P95",
             "宽高比 均值/中位/P5~P95", "宽范围", "高范围"])
        # 数值列按内容取宽，避免 8 列等宽时数字被省略号截断
        self.geo_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.geo_table.horizontalHeader().setStretchLastSection(True)
        geo_layout.addWidget(self.geo_table)
        layout.addWidget(geo_card, 3)

        chart_card, chart_layout = theme.card()
        chart_layout.addLayout(theme.section_row("🖼️", "可视化图表"))

        chart_row = QHBoxLayout()
        chart_row.setSpacing(10)
        chart_row.addWidget(QLabel("数据范围:"))
        self.split_combo = QComboBox()
        self.split_combo.setMinimumWidth(140)
        chart_row.addWidget(self.split_combo)
        chart_row.addWidget(QLabel("图表类型:"))
        self.chart_combo = QComboBox()
        self.chart_combo.setMinimumWidth(180)
        for text, key in CHART_TYPES:
            self.chart_combo.addItem(text, key)
        chart_row.addWidget(self.chart_combo)
        chart_row.addStretch()
        self.save_chart_btn = QPushButton("💾 保存图表")
        theme.set_role(self.save_chart_btn, "amber")
        self.save_chart_btn.setEnabled(False)
        self.save_chart_btn.clicked.connect(self.save_chart)
        chart_row.addWidget(self.save_chart_btn)
        chart_layout.addLayout(chart_row)

        self.chart_label = QLabel("统计后点击图表类型即可查看分布")
        self.chart_label.setAlignment(Qt.AlignCenter)
        self.chart_label.setMinimumHeight(380)
        theme.set_role(self.chart_label, "canvas")
        chart_layout.addWidget(self.chart_label)
        layout.addWidget(chart_card, 4)

        self.split_combo.currentIndexChanged.connect(self._refresh_chart)
        self.chart_combo.currentIndexChanged.connect(self._refresh_chart)

        log_card, log_layout = theme.card()

        log_header = QHBoxLayout()
        log_header.addWidget(QLabel("📋"))
        log_header.addWidget(theme.section("操作日志"))
        log_header.addStretch()
        log_layout.addLayout(log_header)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        theme.set_role(self.log_text, "log")
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_card, 1)

    def browse_input_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择标签目录")
        if dir_path:
            self.input_dir_edit.setText(dir_path)

    def browse_yaml(self):
        file_path = QFileDialog.getOpenFileName(self, "选择数据集YAML文件", "", "YAML文件 (*.yaml)")[0]
        if file_path:
            self.yaml_edit.setText(file_path)

    def load_class_names(self):
        yaml_path = self.yaml_edit.text().strip()
        if not yaml_path or not os.path.exists(yaml_path):
            return {}
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            if 'names' in data:
                names = data['names']
                if isinstance(names, list):
                    return {i: names[i] for i in range(len(names))}
                elif isinstance(names, dict):
                    return {int(k): v for k, v in names.items()}
        except Exception as e:
            self.log(f"读取YAML文件失败: {str(e)}")
        return {}

    def log(self, msg):
        QMetaObject.invokeMethod(self.log_text, "append", Qt.QueuedConnection,
                                Q_ARG(str, msg))

    def start_count(self):
        labels_dir = self.input_dir_edit.text().strip()
        if not labels_dir or not os.path.exists(labels_dir):
            QMessageBox.warning(self, "提示", "请选择有效的标签目录")
            return
        self.run_task(self.count_labels_task, labels_dir)
        self.log("开始统计标签数量...")

    def count_labels_task(self, labels_dir, _stop_event=None):
        labels_path = Path(labels_dir)
        all_counts = defaultdict(int)
        all_dims = defaultdict(list)
        split_results = {}
        split_dims = {}

        def stopped():
            return _stop_event and _stop_event._stop_flag

        def scan(directory):
            """返回 (类别计数字典, 类别->[[w,h],...] 字典)"""
            counts = defaultdict(int)
            dims = defaultdict(list)
            for label_file in Path(directory).glob('**/*.txt'):
                try:
                    with open(label_file, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            parts = line.split()
                            if len(parts) < 5:
                                continue
                            try:
                                label_id = int(float(parts[0]))
                                w = float(parts[3])
                                h = float(parts[4])
                            except ValueError:
                                continue
                            if w <= 0 or h <= 0:
                                continue
                            counts[label_id] += 1
                            dims[label_id].append((w, h))
                except Exception:
                    continue
            return counts, dims

        subdirs = sorted([d for d in labels_path.iterdir() if d.is_dir()])
        targets = subdirs if subdirs else [labels_path]

        for subdir in targets:
            if stopped():
                break
            counts, dims = scan(subdir)
            if counts:
                split_results[subdir.name] = counts
                split_dims[subdir.name] = {k: np.asarray(v, dtype=np.float64)
                                           for k, v in dims.items()}
                for k, v in counts.items():
                    all_counts[k] += v
                for k, v in dims.items():
                    all_dims[k].extend(v)

        if subdirs and not all_counts:
            # 子目录里没有标签时，回退到根目录整体统计
            counts, dims = scan(labels_path)
            split_results = {labels_path.name: counts}
            split_dims = {labels_path.name: {k: np.asarray(v, dtype=np.float64)
                                             for k, v in dims.items()}}
            all_counts = counts
            all_dims = dims

        merged = {k: np.asarray(v, dtype=np.float64) for k, v in all_dims.items()}
        return (split_results, dict(all_counts), {'all': merged, 'splits': split_dims})

    def _display_results(self, split_results, all_counts, geo=None):
        class_names = self.load_class_names()
        self._class_names = class_names
        self._geo = geo or {'all': {}, 'splits': {}}
        all_labels = sorted(all_counts.keys())

        priority = ['train', 'valid', 'val', 'test']
        ordered_splits = [s for s in priority if s in split_results]
        ordered_splits += sorted(s for s in split_results if s not in priority)

        headers = ["类别ID", "类别名称"]
        for split_name in ordered_splits:
            headers.append(SPLIT_NAMES.get(split_name, split_name))
        headers.append("总数")

        self.result_table.setColumnCount(len(headers))
        self.result_table.setHorizontalHeaderLabels(headers)
        self.result_table.setRowCount(len(all_labels))
        self.result_table.horizontalHeader().show()

        for i, label in enumerate(all_labels):
            class_name = class_names.get(label, f"类别 {label}")
            self.result_table.setItem(i, 0, QTableWidgetItem(str(label)))
            self.result_table.setItem(i, 1, QTableWidgetItem(class_name))

            col_idx = 2
            for split_name in ordered_splits:
                split_counts = split_results[split_name]
                count = split_counts.get(label, 0)
                self.result_table.setItem(i, col_idx, QTableWidgetItem(str(count)))
                col_idx += 1

            self.result_table.setItem(i, col_idx, QTableWidgetItem(str(all_counts[label])))

        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.log(f"共统计 {sum(all_counts.values())} 个标签")
        for split_name in ordered_splits:
            counts = split_results[split_name]
            display_name = SPLIT_NAMES.get(split_name, split_name)
            self.log(f"  - {display_name}: {sum(counts.values())} 个标签")
        if class_names:
            self.log(f"类别映射: {class_names}")

        self._display_geo(all_labels, split_results, ordered_splits)

    def _display_geo(self, all_labels, split_results, ordered_splits):
        """填充宽高比表格、总体摘要与图表下拉框"""
        rows = []
        for label in all_labels:
            arr = self._geo.get('all', {}).get(label)
            if arr is None or len(arr) == 0:
                continue
            s = self._stat_block(arr)
            rows.append((label, s))

        self.geo_table.setRowCount(len(rows))
        for i, (label, s) in enumerate(rows):
            class_name = self._class_names.get(label, f"类别 {label}")
            cells = [str(label), class_name, f"{s['count']}",
                     self._fmt_trio(s['w']), self._fmt_trio(s['h']), self._fmt_trio(s['ar']),
                     f"{s['w']['mn']:.3f}~{s['w']['mx']:.3f}",
                     f"{s['h']['mn']:.3f}~{s['h']['mx']:.3f}"]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col >= 2:
                    item.setTextAlignment(Qt.AlignCenter)
                self.geo_table.setItem(i, col, item)

        all_arr = self._concat(self._geo.get('all', {}))
        if all_arr is None:
            self.summary_label.setText("未找到有效的 YOLO 标签框（需要 5 列：class cx cy w h）")
            self.log("未解析到有效的目标框尺寸数据")
        else:
            text, plain = self._summary_html(all_arr)
            self.summary_label.setText(text)
            self.log(plain)

        # 数据范围下拉：全部 + 各分割
        self.split_combo.blockSignals(True)
        self.split_combo.clear()
        self.split_combo.addItem("全部数据", "__all__")
        for name in ordered_splits:
            if split_results.get(name):
                self.split_combo.addItem(SPLIT_NAMES.get(name, name), name)
        self.split_combo.blockSignals(False)

        self._refresh_chart()

    @staticmethod
    def _concat(dims):
        arrays = [a for a in dims.values() if a is not None and len(a)]
        if not arrays:
            return None
        return np.concatenate(arrays, axis=0)

    @staticmethod
    def _current_dims(geo, split_key):
        if split_key == '__all__':
            return geo.get('all', {})
        return geo.get('splits', {}).get(split_key, {})

    @staticmethod
    def _block(a):
        return {'mean': float(np.mean(a)), 'med': float(np.median(a)),
                'p5': float(np.percentile(a, 5)), 'p95': float(np.percentile(a, 95)),
                'mn': float(np.min(a)), 'mx': float(np.max(a))}

    @classmethod
    def _stat_block(cls, arr):
        w, h = arr[:, 0], arr[:, 1]
        return {'count': int(arr.shape[0]), 'w': cls._block(w), 'h': cls._block(h),
                'ar': cls._block(w / h), 'area': cls._block(w * h)}

    @staticmethod
    def _fmt_trio(b):
        return f"{b['mean']:.3f}/{b['med']:.3f}/{b['p5']:.2f}~{b['p95']:.2f}"

    def _summary_html(self, arr):
        s = self._stat_block(arr)
        total = s['count']
        ar = arr[:, 0] / arr[:, 1]
        area = arr[:, 0] * arr[:, 1]

        def bucket_lines(values, buckets):
            out = []
            for lo, hi, name in buckets:
                n = int(np.count_nonzero((values >= lo) & (values < hi)))
                out.append(f"{name} {n}({n / total * 100:.1f}%)")
            return " · ".join(out)

        ar_line = bucket_lines(ar, AR_BUCKETS)
        area_line = bucket_lines(area, AREA_BUCKETS)

        text = (
            f"<b>共 {total} 个目标框</b>（归一化尺寸）<br>"
            f"宽 均值 {s['w']['mean']:.3f} · 中位 {s['w']['med']:.3f} · "
            f"P5~P95 {s['w']['p5']:.3f}~{s['w']['p95']:.3f} · 范围 {s['w']['mn']:.3f}~{s['w']['mx']:.3f}<br>"
            f"高 均值 {s['h']['mean']:.3f} · 中位 {s['h']['med']:.3f} · "
            f"P5~P95 {s['h']['p5']:.3f}~{s['h']['p95']:.3f} · 范围 {s['h']['mn']:.3f}~{s['h']['mx']:.3f}<br>"
            f"宽高比 均值 {s['ar']['mean']:.3f} · 中位 {s['ar']['med']:.3f} · "
            f"P5~P95 {s['ar']['p5']:.3f}~{s['ar']['p95']:.3f} · 范围 {s['ar']['mn']:.3f}~{s['ar']['mx']:.3f}<br>"
            f"形态分布 — {ar_line}<br>"
            f"面积分布 — {area_line}"
        )
        plain = (f"宽高总体: 宽 {s['w']['mean']:.3f}/{s['w']['mn']:.3f}~{s['w']['mx']:.3f}, "
                 f"高 {s['h']['mean']:.3f}/{s['h']['mn']:.3f}~{s['h']['mx']:.3f}, "
                 f"宽高比 {s['ar']['mean']:.3f}/{s['ar']['mn']:.3f}~{s['ar']['mx']:.3f}")
        return text, plain

    def _selected_split(self):
        return self.split_combo.currentData() or '__all__'

    def _refresh_chart(self, *_):
        key = self.chart_combo.currentData()
        if not key or not self._geo:
            return
        try:
            path = self._plot(key)
        except Exception as e:
            self.log(f"生成图表失败: {e}")
            return
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.chart_label.setText("图表生成失败")
            return
        self._chart_pixmap = pixmap
        self._scale_chart()
        self.save_chart_btn.setEnabled(True)

    def _scale_chart(self):
        if self._chart_pixmap is None:
            return
        self.chart_label.setPixmap(
            self._chart_pixmap.scaled(self.chart_label.size(),
                                      Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale_chart()

    @staticmethod
    def _use_cjk_font():
        for font_name in ['Microsoft YaHei', 'SimHei', 'SimSun']:
            try:
                fm.findfont(font_name, fallback_to_default=False)
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
                return
            except Exception:
                continue

    def _plot(self, key):
        split_key = self._selected_split()
        dims = self._current_dims(self._geo, split_key)
        arr = self._concat(dims)
        if arr is None:
            raise ValueError("所选范围内没有可用的目标框数据")

        self._use_cjk_font()
        scope = "全部数据" if split_key == '__all__' else split_key

        if key == 'wh_hist':
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            self._plot_wh_hist(axes, arr, scope)
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            if key == 'count':
                self._plot_count(ax, dims, scope)
            elif key == 'ar_hist':
                self._plot_ar_hist(ax, arr, scope)
            elif key == 'wh_scatter':
                self._plot_scatter(ax, dims, scope)
            elif key == 'ar_box':
                self._plot_ar_box(ax, dims, scope)

        fig.tight_layout()
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        fig.savefig(temp_path, dpi=110, bbox_inches='tight')
        plt.close(fig)
        return temp_path

    def _label_of(self, cls_id):
        return str(self._class_names.get(cls_id, f"类别 {cls_id}"))

    def _plot_count(self, ax, dims, scope):
        counts = sorted(((self._label_of(k), int(len(v)), k)
                         for k, v in dims.items() if len(v)), key=lambda t: -t[1])[:30]
        counts = counts[::-1]
        names = [c[0] for c in counts]
        values = [c[1] for c in counts]
        ax.barh(names, values, color='#3b82f6')
        for y, v in enumerate(values):
            ax.text(v, y, f" {v}", va='center', fontsize=8)
        ax.set_title(f'各类别目标框数量 (Top {len(counts)} · {scope})')
        ax.set_xlabel('数量')
        ax.grid(axis='x', alpha=0.3)

    def _plot_ar_hist(self, ax, arr, scope):
        ar = arr[:, 0] / arr[:, 1]
        bins = np.logspace(np.log10(max(ar.min(), 1e-3)), np.log10(max(ar.max(), 1e-2)), 40)
        ax.hist(ar, bins=bins, color='#22c55e', edgecolor='white')
        ax.set_xscale('log')
        ax.axvline(1.0, color='#ef4444', linestyle='--', label='宽高比 = 1')
        for r in (1 / 3, 3.0):
            ax.axvline(r, color='#f59e0b', linestyle=':', linewidth=1)
        ax.set_title(f'宽高比(W/H)分布直方图 · {scope}')
        ax.set_xlabel('宽高比 (对数刻度)')
        ax.set_ylabel('目标框数量')
        ax.legend()
        ax.grid(alpha=0.3)

    def _plot_wh_hist(self, axes, arr, scope):
        for ax, col, name, color in ((axes[0], 0, '宽 W', '#3b82f6'),
                                     (axes[1], 1, '高 H', '#8b5cf6')):
            values = arr[:, col]
            ax.hist(values, bins=50, range=(0, 1), color=color, edgecolor='white')
            mean, med = float(np.mean(values)), float(np.median(values))
            p5, p95 = np.percentile(values, [5, 95])
            ax.axvline(med, color='#ef4444', linestyle='--',
                       label=f'中位 {med:.3f}')
            ax.axvline(p5, color='#f59e0b', linestyle=':', label=f'P5 {p5:.3f}')
            ax.axvline(p95, color='#f59e0b', linestyle=':')
            ax.set_title(f'{name} 分布 (均值 {mean:.3f}) · {scope}')
            ax.set_xlabel('归一化尺寸')
            ax.set_ylabel('数量')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

    def _plot_scatter(self, ax, dims, scope):
        cmap = matplotlib.colormaps['tab20']
        keys = sorted((k for k, v in dims.items() if len(v)), key=lambda k: -len(dims[k]))[:12]
        if not keys:
            raise ValueError("所选范围内没有可用的目标框数据")
        for i, k in enumerate(keys):
            a = dims[k]
            ax.scatter(a[:, 0], a[:, 1], s=8, alpha=0.5,
                       color=cmap(i % 20), label=self._label_of(k))
        lim_lo = min(1e-3, min(float(dims[k].min()) for k in keys))
        lim_hi = 1.0
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        diag = np.array([lim_lo, lim_hi])
        ax.plot(diag, diag, color='#64748b', linewidth=1, linestyle='--', label='W = H')
        for r in (1 / 3, 3.0):
            ax.plot(diag, diag * r, color='#f59e0b', linewidth=0.8, linestyle=':')
        ax.set_title(f'目标框 宽-高 散点 (Top {len(keys)} 类 · {scope})')
        ax.set_xlabel('宽 W (对数)')
        ax.set_ylabel('高 H (对数)')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3, which='both')

    def _plot_ar_box(self, ax, dims, scope):
        items = sorted(((k, v) for k, v in dims.items() if len(v) >= 20),
                       key=lambda kv: -len(kv[1]))[:15]
        if not items:
            items = sorted(((k, v) for k, v in dims.items() if len(v)),
                           key=lambda kv: -len(kv[1]))[:15]
        data = [v[:, 0] / v[:, 1] for _, v in items]
        labels = [self._label_of(k) for k, _ in items]
        ax.boxplot(data, vert=True, showfliers=False)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_yscale('log')
        ax.axhline(1.0, color='#ef4444', linestyle='--', linewidth=1)
        ax.set_title(f'各类别宽高比分布箱线 (Top {len(items)} 类 · {scope})')
        ax.set_ylabel('宽高比 W/H (对数)')
        ax.grid(axis='y', alpha=0.3)
        for tick in ax.get_xticklabels():
            tick.set_rotation(35)
            tick.set_ha('right')

    def save_chart(self):
        if self._chart_pixmap is None:
            QMessageBox.warning(self, "提示", "请先生成图表")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存图表", "", "PNG图片 (*.png)")
        if path and self._chart_pixmap.save(path):
            self.log(f"图表已保存: {path}")

    def on_task_finished(self, success, msg):
        if success:
            if isinstance(msg, tuple) and len(msg) == 3:
                split_results, all_counts, geo = msg
                self._display_results(split_results, all_counts, geo)
            elif isinstance(msg, tuple) and len(msg) == 2:
                split_results, all_counts = msg
                self._display_results(split_results, all_counts)
            else:
                self.log(f"[完成] {msg}")
        else:
            self.log(f"[错误] {msg}")
