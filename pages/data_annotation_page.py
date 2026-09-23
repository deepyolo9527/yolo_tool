
# -*- coding: utf-8 -*-

"""数据标注页面：画框标注、SAM交互分割、AI预标注、类别管理、标注检查"""

import os
from dataclasses import dataclass

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QBrush, QFont, QKeySequence
from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QHBoxLayout,
                             QVBoxLayout, QFormLayout, QComboBox, QListWidget,
                             QListWidgetItem, QFileDialog, QMessageBox, QDialog,
                             QInputDialog, QGroupBox, QDoubleSpinBox,
                             QRadioButton, QCheckBox, QProgressBar, QShortcut,
                             QTextEdit, QScrollArea)

from utils.label_constants import CLASS_COLORS, IMAGE_EXTENSIONS, COMMON_MAP
from utils.sort_utils import natural_sort_key
from utils.path_helpers import get_sam_dir, get_weights_dir
from utils import theme
from utils.sam_config import (load_sam_settings, save_sam_settings, resolve_sam_imgsz,
                              default_sam_settings, SAM_VIDEO_PRESETS, SAM_IMAGE_SIZES)


class BoundingBox:
    """YOLO归一化标注框"""

    def __init__(self, x_center, y_center, width, height, class_name="", class_index=-1):
        self.x_center = float(x_center)
        self.y_center = float(y_center)
        self.width = float(width)
        self.height = float(height)
        self.class_name = class_name
        self.class_index = class_index
        self.selected = False
        self.is_ai_generated = False

    def get_rect(self, img_width, img_height):
        x = (self.x_center - self.width / 2) * img_width
        y = (self.y_center - self.height / 2) * img_height
        return QRectF(x, y, self.width * img_width, self.height * img_height)

    def to_yolo_line(self):
        return (f"{self.class_index} {self.x_center:.6f} {self.y_center:.6f} "
                f"{self.width:.6f} {self.height:.6f}")


class ImageLabel(QLabel):
    """交互式标注画布：绘制图片、标注框，处理画框和SAM点击/框选事件"""

    TOOL_IDLE = 0
    TOOL_DRAW = 1
    TOOL_SAM = 2

    def __init__(self, page=None, parent=None):
        super().__init__(parent)
        self.page = page
        self.setAlignment(Qt.AlignCenter)
        theme.set_role(self, "canvas")
        self.setMinimumSize(600, 400)
        self.setMouseTracking(True)
        self.tool = self.TOOL_IDLE
        self.draw_start = None
        self.draw_end = None
        self.sam_start = None
        self.sam_end = None
        self.boxes = []
        self.current_pixmap = None

    def set_image(self, pixmap):
        self.current_pixmap = pixmap
        self.update_display()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display()

    def _scaled_pixmap(self):
        return self.current_pixmap.scaled(self.size(), Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation)

    def update_display(self):
        if not self.current_pixmap:
            return
        scaled = self._scaled_pixmap()
        painter = QPainter(scaled)
        painter.setRenderHint(QPainter.Antialiasing)

        for i, box in enumerate(self.boxes):
            color = QColor(CLASS_COLORS[box.class_index % len(CLASS_COLORS)]
                           if box.class_index >= 0
                           else CLASS_COLORS[i % len(CLASS_COLORS)])
            if box.selected:
                pen = QPen(color, 4)
            elif box.is_ai_generated:
                pen = QPen(color, 2, Qt.DashLine)
            else:
                pen = QPen(color, 2)
            painter.setPen(pen)
            fill = QColor(color)
            fill.setAlpha(50)
            painter.setBrush(fill)
            rect = box.get_rect(scaled.width(), scaled.height())
            painter.drawRect(rect)

            label_text = box.class_name
            if box.is_ai_generated:
                label_text += " [AI]"
            painter.setFont(QFont("微软雅黑", 8))
            chip = QColor(color)
            chip.setAlpha(200)
            painter.setBrush(chip)
            painter.setPen(Qt.NoPen)
            painter.drawRect(int(rect.x()), int(rect.y()) - 14,
                             max(40, len(label_text) * 8), 14)
            painter.setPen(QPen(QColor("white")))
            painter.setBrush(Qt.NoBrush)
            painter.drawText(int(rect.x()) + 2, int(rect.y()) - 2, label_text)

        # 预览框：鼠标位置是控件坐标，需减去图片居中显示的留白偏移才是图像绘制坐标
        ox = (self.width() - scaled.width()) / 2
        oy = (self.height() - scaled.height()) / 2
        if self.tool == self.TOOL_DRAW and self.draw_start and self.draw_end:
            painter.setPen(QPen(QColor("#e74c3c"), 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            x = min(self.draw_start.x(), self.draw_end.x()) - ox
            y = min(self.draw_start.y(), self.draw_end.y()) - oy
            w = abs(self.draw_end.x() - self.draw_start.x())
            h = abs(self.draw_end.y() - self.draw_start.y())
            painter.drawRect(int(x), int(y), int(w), int(h))

        if self.tool == self.TOOL_SAM and self.sam_start and self.sam_end \
                and self.sam_start != self.sam_end:
            painter.setPen(QPen(QColor("#3498db"), 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            x = min(self.sam_start.x(), self.sam_end.x()) - ox
            y = min(self.sam_start.y(), self.sam_end.y()) - oy
            w = abs(self.sam_end.x() - self.sam_start.x())
            h = abs(self.sam_end.y() - self.sam_start.y())
            painter.drawRect(int(x), int(y), int(w), int(h))

        if self.tool == self.TOOL_SAM:
            painter.setPen(QPen(QColor("#3498db"), 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(0, 0, scaled.width(), scaled.height())
            painter.setPen(QPen(QColor("#3498db")))
            painter.setFont(QFont("微软雅黑", 10))
            painter.drawText(10, 20, "✂️ SAM 分割模式 — 选择类别后点击/框选目标区域，按 Esc 退出")

        painter.end()
        self.setPixmap(scaled)

    def _widget_to_image_pos(self, pos):
        if not self.current_pixmap:
            return 0.0, 0.0
        scaled = self._scaled_pixmap()
        img_x = (pos.x() - (self.width() - scaled.width()) / 2) / scaled.width() * self.current_pixmap.width()
        img_y = (pos.y() - (self.height() - scaled.height()) / 2) / scaled.height() * self.current_pixmap.height()
        return img_x, img_y

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.current_pixmap and self.page:
            if self.tool == self.TOOL_SAM:
                self.sam_start = event.pos()
                self.sam_end = event.pos()
            elif self.tool == self.TOOL_DRAW:
                self.draw_start = event.pos()
                self.draw_end = event.pos()

    def mouseMoveEvent(self, event):
        if self.tool == self.TOOL_DRAW and self.draw_start:
            self.draw_end = event.pos()
            self.update_display()
        elif self.tool == self.TOOL_SAM and self.sam_start is not None:
            self.sam_end = event.pos()
            self.update_display()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or not self.page or not self.current_pixmap:
            return
        if self.tool == self.TOOL_SAM and self.sam_start is not None:
            start_x, start_y = self._widget_to_image_pos(self.sam_start)
            end_x, end_y = self._widget_to_image_pos(event.pos())
            is_click = self.sam_start == event.pos()
            self.sam_start = None
            self.sam_end = None
            self.update_display()
            if is_click:
                self.page.on_sam_click(start_x, start_y)
            else:
                self.page.on_sam_box(start_x, start_y, end_x, end_y)
        elif self.tool == self.TOOL_DRAW and self.draw_start and self.draw_end:
            self._add_box_from_draw()

    def _add_box_from_draw(self):
        img_w = self.current_pixmap.width()
        img_h = self.current_pixmap.height()
        x1, y1 = self._widget_to_image_pos(self.draw_start)
        x2, y2 = self._widget_to_image_pos(self.draw_end)
        xc = (x1 + x2) / 2 / img_w
        yc = (y1 + y2) / 2 / img_h
        w = abs(x2 - x1) / img_w
        h = abs(y2 - y1) / img_h
        self.draw_start = None
        self.draw_end = None
        if w > 0.01 and h > 0.01:
            box = BoundingBox(xc, yc, w, h,
                              self.page.get_current_class_name(),
                              self.page.get_current_class_index())
            self.page.on_box_added(box)
        self.update_display()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.tool == self.TOOL_SAM:
            self.page.set_sam_mode(False)
        else:
            super().keyPressEvent(event)


class SAMModelLoadWorker(QThread):
    """SAM模型后台加载线程"""
    finished_signal = pyqtSignal(object, str, str)  # (model, error, model_name)

    def __init__(self, model_name, full_path):
        super().__init__()
        self.model_name = model_name
        self.full_path = full_path

    def run(self):
        try:
            from ultralytics import SAM
            model = SAM(self.full_path)
            self.finished_signal.emit(model, "", self.model_name)
        except Exception as e:
            self.finished_signal.emit(None, str(e), self.model_name)


class SAMSegmentWorker(QThread):
    """SAM分割推理线程"""
    finished_signal = pyqtSignal(object, str, int)  # (masks, error, request_id)

    def __init__(self, model, image_path, point=None, bbox=None, request_id=0, imgsz=1024):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.point = point
        self.bbox = bbox
        self.request_id = request_id
        self.imgsz = imgsz

    def run(self):
        try:
            if self.bbox:
                result = self.model(self.image_path, bboxes=[self.bbox],
                                    verbose=False, imgsz=self.imgsz)
            elif self.point:
                result = self.model(self.image_path, points=[self.point], labels=[1],
                                    verbose=False, imgsz=self.imgsz)
            else:
                result = self.model(self.image_path, verbose=False, imgsz=self.imgsz)
            masks = None
            if result and len(result) > 0:
                if hasattr(result[0], 'masks') and result[0].masks is not None:
                    masks = result[0].masks.data
            self.finished_signal.emit(masks, "", self.request_id)
        except Exception as e:
            self.finished_signal.emit(None, str(e), self.request_id)


class AIPrelabelWorker(QThread):
    """AI批量预标注线程：模型类别名与项目类别自动匹配(COCO英文名->中文)"""
    progress_signal = pyqtSignal(int, int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, model_path, image_paths, label_dir, classes, conf, iou, mode,
                 skip_class_check=False):
        super().__init__()
        self.model_path = model_path
        self.image_paths = image_paths
        self.label_dir = label_dir
        self.classes = classes
        self.conf = conf
        self.iou = iou
        self.mode = mode
        self.skip_class_check = skip_class_check
        self.is_running = True

    def _build_class_map(self, model):
        """模型类别id -> 项目类别id 映射，未匹配的忽略"""
        class_map = {}
        skipped = []
        for idx, name in model.names.items():
            target = None
            if name in self.classes:
                target = name
            elif COMMON_MAP.get(name) in self.classes:
                target = COMMON_MAP.get(name)
            if target is not None:
                class_map[int(idx)] = self.classes.index(target)
            else:
                skipped.append(str(name))
        return class_map, skipped

    def _sync_classes_file(self, model):
        """按模型类别ID把类别名写入 classes.txt(第i行=ID i，从0开始)，
        保证后续显示与标注txt的序号一致"""
        names = {int(i): str(n) for i, n in model.names.items()}
        if not names:
            return
        os.makedirs(self.label_dir, exist_ok=True)
        path = os.path.join(self.label_dir, 'classes.txt')
        rows = []
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    rows = [l.strip() for l in f]
            except UnicodeDecodeError:
                with open(path, 'r', encoding='gbk') as f:
                    rows = [l.strip() for l in f]
        rows = [r for r in rows if r]  # 与 _load_classes 的过滤方式保持一致
        while len(rows) <= max(names):
            rows.append("")
        for i, name in names.items():
            rows[i] = name
        # 模型ID不连续时中间行填占位名，保证行号=类别ID
        rows = [r if r else f"class_{i}" for i, r in enumerate(rows)]
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(rows) + '\n')
        self.log_signal.emit(f"已按模型类别更新类别文件: {path}")

    def run(self):
        try:
            from ultralytics import YOLO
            self.log_signal.emit(f"加载模型: {self.model_path}")
            model = YOLO(self.model_path)
            if self.skip_class_check:
                # 跳过校验：模型类别ID直接写入标签（不做名称匹配与过滤）
                class_map = {int(idx): int(idx) for idx in model.names}
                self.log_signal.emit("已跳过类别校验，按模型类别ID直接写入")
                self._sync_classes_file(model)
            else:
                class_map, skipped = self._build_class_map(model)
                if skipped:
                    self.log_signal.emit(f"未匹配到的模型类别(将忽略): {skipped}")
                if not class_map:
                    self.finished_signal.emit(False, "模型类别与项目类别无匹配，请检查类别名称")
                    return

            total = len(self.image_paths)
            box_count = 0
            for i, img_path in enumerate(self.image_paths):
                if not self.is_running:
                    self.log_signal.emit("⚠️ 已被用户中断")
                    break
                result = model(img_path, conf=self.conf, iou=self.iou, verbose=False)
                new_lines = []
                if result and len(result) > 0:
                    res = result[0]
                    if hasattr(res, 'boxes') and res.boxes is not None:
                        for j in range(len(res.boxes.cls)):
                            cls_id = int(res.boxes.cls[j].item())
                            if cls_id not in class_map:
                                continue
                            xyxyn = res.boxes.xyxyn[j].cpu().numpy()
                            xc = (xyxyn[0] + xyxyn[2]) / 2
                            yc = (xyxyn[1] + xyxyn[3]) / 2
                            w = xyxyn[2] - xyxyn[0]
                            h = xyxyn[3] - xyxyn[1]
                            new_lines.append(
                                f"{class_map[cls_id]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                box_count += len(new_lines)
                self._write_label(img_path, new_lines)
                self.progress_signal.emit(i + 1, total)

            self.finished_signal.emit(True, f"完成: {i + 1 if total else 0} 张图片, "
                                           f"新增 {box_count} 个标注框")
        except Exception as e:
            self.finished_signal.emit(False, str(e))

    def _write_label(self, img_path, new_lines):
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, base + '.txt')
        mark_path = os.path.join(self.label_dir, base + '.mark')
        lines, marks = [], []
        if self.mode == 'append' and os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f if l.strip()]
            if os.path.exists(mark_path):
                with open(mark_path, 'r', encoding='utf-8') as f:
                    marks = [l.strip() for l in f]
        lines.extend(new_lines)
        marks.extend(["AI"] * len(new_lines))
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))
        with open(mark_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(marks) + ('\n' if marks else ''))

    def stop(self):
        self.is_running = False


@dataclass
class SamRequestContext:
    request_id: int
    image_path: str
    class_index: int
    class_name: str


def scan_sam_models():
    """扫描 sam/ 与 weights/ 目录下的SAM模型"""
    models = []
    for folder in (get_sam_dir(), get_weights_dir()):
        if os.path.isdir(folder):
            for f in os.listdir(folder):
                if f.lower().endswith('.pt') and 'sam' in f.lower():
                    models.append(f)
    return sorted(set(models))


def find_sam_model_path(model_name):
    for folder in (get_sam_dir(), get_weights_dir()):
        full = os.path.join(folder, model_name)
        if os.path.exists(full):
            return full
    return ""


class ClassManagerDialog(QDialog):
    """类别管理对话框"""

    def __init__(self, classes, parent=None):
        super().__init__(parent)
        self.classes = list(classes)
        self.setWindowTitle("🏷️ 标注类别管理")
        self.resize(400, 300)
        layout = QVBoxLayout(self)

        self.list_widget = QListWidget()
        self._reload_list()
        layout.addWidget(self.list_widget)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("➕ 添加类别")
        add_btn.clicked.connect(self.add_class)
        btn_row.addWidget(add_btn)
        import_btn = QPushButton("📥 导入类别文件")
        import_btn.setToolTip("从 classes.txt 等文本文件导入，每行一个类别；可选择合并或替换")
        import_btn.clicked.connect(self.import_classes)
        btn_row.addWidget(import_btn)
        del_btn = QPushButton("🗑 删除选中")
        del_btn.clicked.connect(self.del_class)
        btn_row.addWidget(del_btn)
        ok_btn = QPushButton("✅ 确定")
        ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

    def _reload_list(self):
        """列表项显示为 'ID. 类别名'，ID 与标注txt的 class_id 一致(从0开始)"""
        self.list_widget.clear()
        for i, c in enumerate(self.classes):
            self.list_widget.addItem(QListWidgetItem(f"{i}. {c}"))

    def add_class(self):
        name, ok = QInputDialog.getText(self, "添加类别", "类别名称:")
        if ok and name.strip():
            self.classes.append(name.strip())
            self._reload_list()

    def import_classes(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "导入类别文件", "", "类别文本文件 (*.txt);;所有文件 (*)")
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                names = [l.strip() for l in f if l.strip()]
        except UnicodeDecodeError:
            with open(path, 'r', encoding='gbk') as f:
                names = [l.strip() for l in f if l.strip()]
        if not names:
            QMessageBox.warning(self, "提示", "文件中没有读取到类别")
            return
        merge = QMessageBox.question(
            self, "导入方式",
            f"读取到 {len(names)} 个类别。\nYes=合并到现有列表(去重)，No=替换现有列表",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        if merge == QMessageBox.Cancel:
            return
        if merge == QMessageBox.Yes:
            for n in names:
                if n not in self.classes:
                    self.classes.append(n)
        else:
            self.classes = list(dict.fromkeys(names))
        self._reload_list()

    def del_class(self):
        row = self.list_widget.currentRow()
        if row >= 0:
            self.classes.pop(row)
            self._reload_list()

    def get_classes(self):
        return list(self.classes)


class SamSettingsDialog(QDialog):
    """SAM设置对话框（原系统设置中的SAM设置，图像/视频两组参数）"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("✂️ SAM设置")
        self.resize(420, 480)
        self._loading = True
        layout = QVBoxLayout(self)

        img_group = QGroupBox("SAM图像分割设置")
        img_form = QFormLayout(img_group)
        self.image_imgsz_combo = QComboBox()
        self.image_imgsz_combo.addItems([str(s) for s in SAM_IMAGE_SIZES])
        img_form.addRow("推理图像尺寸(imgsz)：", self.image_imgsz_combo)
        layout.addWidget(img_group)

        vid_group = QGroupBox("SAM视频追踪设置")
        vid_form = QFormLayout(vid_group)
        self.video_imgsz_combo = QComboBox()
        self.video_imgsz_combo.addItems([str(s) for s in SAM_IMAGE_SIZES])
        vid_form.addRow("推理图像尺寸(imgsz)：", self.video_imgsz_combo)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["均衡", "严格防漂移", "快速运动", "自定义"])
        vid_form.addRow("质量预设：", self.preset_combo)
        self.quality_spins = {}
        for key, label, maximum, single in [
            ('min_area_ratio', '最小面积比：', 1.0, 0.01),
            ('max_area_ratio', '最大面积比：', 10.0, 0.05),
            ('min_previous_iou', '最小前后IoU：', 1.0, 0.05),
            ('max_center_distance_ratio', '最大中心距离比：', 10.0, 0.05),
            ('max_frame_area_ratio', '最大单帧面积比：', 10.0, 0.05),
        ]:
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(0.0, maximum)
            spin.setSingleStep(single)
            self.quality_spins[key] = spin
            vid_form.addRow(label, spin)
        layout.addWidget(vid_group)

        hint = QLabel("配置保存在 configs/sam_settings.json，图像设置立即用于SAM分割推理。")
        hint.setWordWrap(True)
        theme.set_role(hint, "hint")
        layout.addWidget(hint)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("💾 保存")
        save_btn.clicked.connect(self.on_save)
        btn_row.addWidget(save_btn)
        default_btn = QPushButton("↩️ 恢复默认")
        default_btn.clicked.connect(self.on_restore_defaults)
        btn_row.addWidget(default_btn)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        for spin in self.quality_spins.values():
            spin.valueChanged.connect(self._on_quality_edited)

        self._apply_to_ui(load_sam_settings())
        self._loading = False

    def _apply_to_ui(self, settings):
        img_imgsz = str(settings['image']['imgsz'])
        idx = self.image_imgsz_combo.findText(img_imgsz)
        if idx >= 0:
            self.image_imgsz_combo.setCurrentIndex(idx)
        vid = settings['video']
        idx = self.video_imgsz_combo.findText(str(vid['imgsz']))
        if idx >= 0:
            self.video_imgsz_combo.setCurrentIndex(idx)
        idx = self.preset_combo.findText(vid['preset'])
        self.preset_combo.setCurrentIndex(idx if idx >= 0 else 0)
        for key, spin in self.quality_spins.items():
            spin.setValue(vid['quality'].get(key, 0.0))

    def _collect(self):
        return {
            'image': {'imgsz': int(self.image_imgsz_combo.currentText())},
            'video': {
                'imgsz': int(self.video_imgsz_combo.currentText()),
                'preset': self.preset_combo.currentText(),
                'quality': {k: s.value() for k, s in self.quality_spins.items()},
            },
        }

    def _on_preset_changed(self, preset_name):
        if self._loading:
            return
        preset = SAM_VIDEO_PRESETS.get(preset_name)
        if preset:
            for key, spin in self.quality_spins.items():
                if key in preset:
                    spin.setValue(preset[key])

    def _on_quality_edited(self, _value):
        if self._loading:
            return
        idx = self.preset_combo.findText("自定义")
        if idx >= 0 and self.preset_combo.currentText() != "自定义":
            self.preset_combo.setCurrentIndex(idx)

    def on_save(self):
        save_sam_settings(self._collect())
        QMessageBox.information(self, "成功", "SAM参数已保存")

    def on_restore_defaults(self):
        self._apply_to_ui(default_sam_settings())


class AIPrelabelDialog(QDialog):
    """AI预标注对话框：选择模型/置信度/范围/模式后批量生成标注"""

    def __init__(self, page, parent=None):
        super().__init__(parent)
        self.page = page
        self.worker = None
        self.setWindowTitle("🤖 AI预标注")
        self.resize(460, 400)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        weights_dir = get_weights_dir()
        if os.path.isdir(weights_dir):
            for f in sorted(os.listdir(weights_dir)):
                if f.endswith('.pt') and 'sam' not in f.lower():
                    self.model_combo.addItem(f)
        browse_btn = QPushButton("📁")
        browse_btn.setFixedWidth(36)
        browse_btn.clicked.connect(self.browse_model)
        model_row.addWidget(self.model_combo)
        model_row.addWidget(browse_btn)
        form.addRow("模型：", model_row)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        form.addRow("置信度阈值：", self.conf_spin)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.1, 0.95)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.7)
        form.addRow("NMS IoU：", self.iou_spin)
        layout.addLayout(form)

        scope_group = QGroupBox("标注范围")
        scope_layout = QVBoxLayout(scope_group)
        self.scope_current_radio = QRadioButton("仅当前图片")
        self.scope_unannotated_radio = QRadioButton("所有未标注图片")
        self.scope_all_radio = QRadioButton("所有图片")
        self.scope_current_radio.setChecked(True)
        scope_layout.addWidget(self.scope_current_radio)
        scope_layout.addWidget(self.scope_unannotated_radio)
        scope_layout.addWidget(self.scope_all_radio)
        layout.addWidget(scope_group)

        mode_group = QGroupBox("写入模式")
        mode_layout = QHBoxLayout(mode_group)
        self.mode_append_radio = QRadioButton("追加")
        self.mode_replace_radio = QRadioButton("覆盖")
        self.mode_append_radio.setChecked(True)
        mode_layout.addWidget(self.mode_append_radio)
        mode_layout.addWidget(self.mode_replace_radio)
        layout.addWidget(mode_group)

        self.skip_check_cb = QCheckBox("跳过类别校验（模型类别ID直接写入，类别名取模型输出并更新classes.txt）")
        self.skip_check_cb.setToolTip(
            "勾选后不做名称匹配：标签首列直接写模型类别ID，\n"
            "并用模型类别名按 ID 顺序(从0开始)更新标签目录下的 classes.txt，\n"
            "保证显示名称与标注txt序号一致")
        layout.addWidget(self.skip_check_cb)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        self.log_label = QLabel("模型类别将与项目类别按名称自动匹配(支持COCO中英映射)，未匹配的类别将被忽略。")
        self.log_label.setWordWrap(True)
        theme.set_role(self.log_label, "hint")
        layout.addWidget(self.log_label)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("🚀 开始标注")
        self.start_btn.clicked.connect(self.on_start)
        btn_row.addWidget(self.start_btn)
        self.stop_btn = QPushButton("⏹ 停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.on_stop)
        btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "",
                                              "PyTorch模型 (*.pt *.onnx)")
        if path:
            self.model_combo.setCurrentText(path)

    def _resolve_model_path(self):
        name = self.model_combo.currentText().strip()
        if not name:
            return ""
        if os.path.isabs(name):
            return name if os.path.exists(name) else ""
        full = os.path.join(get_weights_dir(), name)
        return full if os.path.exists(full) else ""

    def _target_images(self):
        if self.scope_current_radio.isChecked():
            current = self.page.current_image_path
            return [current] if current else []
        images = list(self.page.image_files)
        if self.scope_unannotated_radio.isChecked():
            label_dir = self.page.label_folder
            images = [img for img in images
                      if not os.path.exists(os.path.join(
                          label_dir, os.path.splitext(os.path.basename(img))[0] + '.txt'))]
        return images

    def on_start(self):
        if not self.page.label_folder:
            QMessageBox.warning(self, "提示", "请先选择标签文件夹")
            return
        if not self.page.classes and not self.skip_check_cb.isChecked():
            QMessageBox.warning(self, "提示", "请先在标注页标注类别管理")
            return
        if self.skip_check_cb.isChecked():
            if QMessageBox.question(
                    self, "确认跳过校验",
                    "跳过校验将按模型类别ID直接写入标签，并用模型类别名更新 classes.txt"
                    "（第 i 行 = 类别ID i，从0开始）。\n是否继续?") != QMessageBox.Yes:
                return
        model_path = self._resolve_model_path()
        if not model_path:
            QMessageBox.warning(self, "提示", "模型文件不存在，请选择或浏览模型")
            return
        images = self._target_images()
        if not images:
            QMessageBox.warning(self, "提示", "没有需要标注的图片")
            return

        mode = 'append' if self.mode_append_radio.isChecked() else 'replace'
        self.worker = AIPrelabelWorker(
            model_path, images, self.page.label_folder,
            list(self.page.classes), self.conf_spin.value(), self.iou_spin.value(), mode,
            skip_class_check=self.skip_check_cb.isChecked())
        self.worker.log_signal.connect(lambda m: self.log_label.setText(m))
        self.worker.progress_signal.connect(self.on_progress)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def on_stop(self):
        if self.worker:
            self.worker.stop()

    def on_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def on_finished(self, success, msg):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_label.setText(("✅ " if success else "❌ ") + msg)
        if success:
            self.page._load_classes()
            self.page.reload_current_image()
            self.page.refresh_annotation_stats()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(5000)
        super().closeEvent(event)


class DataAnnotationPage(QWidget):
    """数据标注页面"""

    open_video_tracking = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.image_files = []
        self.current_image_index = -1
        self.current_image_path = ""
        self.current_boxes = []
        self.classes = []
        self.classes_file = ""
        self.image_folder = ""
        self.label_folder = ""
        self.sam_model = None
        self.sam_model_name = None
        self._sam_request_counter = 0
        self._sam_worker = None
        self._sam_load_worker = None
        self._loading_file_list = False
        self.ai_dialog = None
        self._init_ui()
        self._bind_events()
        self._populate_sam_models()
        self._setup_shortcuts()

    # ---------- UI ----------
    def _section_row(self, icon, title):
        """卡片分节标题行（图标 + 加粗标题），与 YOLO训练 页一致"""
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel(icon))
        row.addWidget(theme.section(title))
        row.addStretch()
        return row

    def _init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(15, 15, 15, 15)
        outer.setSpacing(12)
        outer.addWidget(theme.header("✏️", "数据标注",
                                     "手动框标注 / SAM 图片分割 / AI 预标注，支持类别管理与标注统计"))

        # 顶部操作按钮行（彩色按钮，参照 YOLO训练 页）
        row1 = QHBoxLayout()
        row1.setSpacing(12)
        self.open_image_btn = QPushButton("📁 打开图片文件夹")
        theme.set_role(self.open_image_btn, "primary")
        self.open_label_btn = QPushButton("📂 打开标签文件夹")
        theme.set_role(self.open_label_btn, "primary")
        self.ai_btn = QPushButton("🤖 AI预标注")
        theme.set_role(self.ai_btn, "primary")
        self.check_btn = QPushButton("🔍 检查标注")
        theme.set_role(self.check_btn, "primary")
        self.video_track_btn = QPushButton("🎬 SAM视频追踪")
        theme.set_role(self.video_track_btn, "amber")
        self.sam_settings_btn = QPushButton("⚙️ SAM设置")
        for btn in (self.open_image_btn, self.open_label_btn, self.ai_btn,
                    self.check_btn, self.video_track_btn, self.sam_settings_btn):
            row1.addWidget(btn)
        row1.addStretch()
        outer.addLayout(row1)

        # 标注操作行：翻页 / 绘制模式 / 状态
        row2 = QHBoxLayout()
        row2.setSpacing(12)
        self.prev_btn = QPushButton("⬅ 上一张(A)")
        theme.set_role(self.prev_btn, "primary")
        self.next_btn = QPushButton("➡ 下一张(D)")
        theme.set_role(self.next_btn, "primary")
        self.draw_btn = QPushButton("✏️ 画标注框(W)")
        self.draw_btn.setCheckable(True)
        self.draw_btn.setToolTip("进入/退出画框模式")
        self.sam_btn = QPushButton("✂️ SAM图片分割")
        self.sam_btn.setCheckable(True)
        self.sam_btn.setToolTip("进入/退出 SAM 分割模式")
        for btn in (self.prev_btn, self.next_btn, self.draw_btn, self.sam_btn):
            row2.addWidget(btn)
        self.status_label = QLabel("请选择图片文件夹")
        self.status_label.setWordWrap(True)
        theme.set_role(self.status_label, "hint")
        row2.addStretch()
        row2.addWidget(self.status_label)
        outer.addLayout(row2)

        # 主体：左侧画布 + 右侧信息面板
        main_layout = QHBoxLayout()
        main_layout.setSpacing(12)
        outer.addLayout(main_layout, 1)

        self.canvas = ImageLabel(self)
        main_layout.addWidget(self.canvas, 1)

        panel, pl = theme.card(margin=14, spacing=10)
        panel.setMinimumWidth(304)
        panel_scroll = QScrollArea()
        panel_scroll.setWidget(panel)
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setFixedWidth(330)
        panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        pl.addLayout(self._section_row("🏷️", "类别与模型"))
        pl.addWidget(QLabel("SAM模型:"))
        self.sam_model_combo = QComboBox()
        pl.addWidget(self.sam_model_combo)
        pl.addWidget(QLabel("类别查看/选择:"))
        self.class_combo = QComboBox()
        pl.addWidget(self.class_combo)
        self.manage_class_btn = QPushButton("🏷️ 标注类别管理")
        theme.set_role(self.manage_class_btn, "primary")
        pl.addWidget(self.manage_class_btn)

        pl.addLayout(self._section_row("🖼️", "标注框列表"))
        self.box_list = QListWidget()
        self.box_list.setMinimumHeight(110)
        pl.addWidget(self.box_list)
        self.del_box_btn = QPushButton("🗑️ 删除选中框(Del)")
        theme.set_role(self.del_box_btn, "danger")
        pl.addWidget(self.del_box_btn)
        self.del_all_boxes_btn = QPushButton("🗑️ 删除所有框")
        theme.set_role(self.del_all_boxes_btn, "danger")
        pl.addWidget(self.del_all_boxes_btn)

        pl.addLayout(self._section_row("📁", "文件列表"))
        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(130)
        pl.addWidget(self.file_list)
        self.del_file_btn = QPushButton("🗑️ 删除选中文件")
        theme.set_role(self.del_file_btn, "danger")
        pl.addWidget(self.del_file_btn)

        pl.addLayout(self._section_row("📊", "标注统计"))
        self.stats_label = QLabel("选择图片与标签文件夹后自动统计")
        self.stats_label.setWordWrap(True)
        theme.set_role(self.stats_label, "hint")
        pl.addWidget(self.stats_label)
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMinimumHeight(130)
        theme.set_role(self.stats_text, "log")
        pl.addWidget(self.stats_text)
        self.refresh_stats_btn = QPushButton("🔄 刷新统计")
        theme.set_role(self.refresh_stats_btn, "primary")
        pl.addWidget(self.refresh_stats_btn)

        main_layout.addWidget(panel_scroll)

    def _bind_events(self):
        self.open_image_btn.clicked.connect(self.open_image_folder)
        self.open_label_btn.clicked.connect(self.open_label_folder)
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.draw_btn.clicked.connect(self.toggle_draw_mode)
        self.sam_btn.clicked.connect(self.toggle_sam_mode)
        self.ai_btn.clicked.connect(self.open_ai_prelabel)
        self.video_track_btn.clicked.connect(self.open_video_tracking.emit)
        self.check_btn.clicked.connect(self.check_annotations)
        self.sam_settings_btn.clicked.connect(self.open_sam_settings)
        self.manage_class_btn.clicked.connect(self.manage_classes)
        self.del_box_btn.clicked.connect(self.delete_selected_box)
        self.del_all_boxes_btn.clicked.connect(self.delete_all_boxes)
        self.refresh_stats_btn.clicked.connect(self.refresh_annotation_stats)
        self.del_file_btn.clicked.connect(self.delete_file)
        self.sam_model_combo.currentTextChanged.connect(self.on_sam_model_changed)
        self.file_list.currentRowChanged.connect(self.on_file_selected)
        self.box_list.currentRowChanged.connect(self.on_box_selected)

    def _setup_shortcuts(self):
        for seq, slot in [("A", self.prev_image), ("D", self.next_image),
                          ("W", self.toggle_draw_mode),
                          ("Delete", self.delete_selected_box),
                          ("Esc", lambda: self.set_sam_mode(False))]:
            shortcut = QShortcut(QKeySequence(seq), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)

    # ---------- 文件夹与文件 ----------
    def open_image_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder:
            return
        self.image_folder = folder
        if not self.label_folder:
            default_label = folder.rstrip('/\\') + '_labels'
            if os.path.isdir(default_label) or not os.path.isdir(folder + '_labels'):
                self.label_folder = default_label
        self._load_image_files()
        self._load_classes()
        self._set_status(f"图片 {len(self.image_files)} 张，标签目录: "
                         f"{self.label_folder or '未选择'}")
        self.refresh_annotation_stats()

    def open_label_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择标签文件夹")
        if folder:
            self.label_folder = folder
            self._load_classes()
            self.reload_current_image()
            self._set_status(f"标签目录: {folder}")
            self.refresh_annotation_stats()

    def _load_image_files(self):
        if not self.image_folder or not os.path.isdir(self.image_folder):
            return
        self.image_files = []
        for f in os.listdir(self.image_folder):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                self.image_files.append(os.path.join(self.image_folder, f))
        self.image_files.sort(key=natural_sort_key)
        self._loading_file_list = True
        self.file_list.clear()
        for fp in self.image_files:
            self.file_list.addItem(QListWidgetItem(os.path.basename(fp)))
        self._loading_file_list = False
        if self.image_files:
            self._load_image(0)

    # ---------- 类别 ----------
    def _load_classes(self):
        candidates = []
        if self.label_folder:
            candidates.append(os.path.join(self.label_folder, 'classes.txt'))
        if self.image_folder:
            candidates.append(os.path.join(self.image_folder, 'classes.txt'))
        self.classes_file = ""
        for path in candidates:
            if os.path.exists(path):
                self.classes_file = path
                break
        self.classes = []
        if self.classes_file:
            try:
                with open(self.classes_file, 'r', encoding='utf-8') as f:
                    self.classes = [l.strip() for l in f if l.strip()]
            except UnicodeDecodeError:
                with open(self.classes_file, 'r', encoding='gbk') as f:
                    self.classes = [l.strip() for l in f if l.strip()]
        current = self.class_combo.currentIndex()
        self._refresh_class_combo(keep_index=current)

    def _refresh_class_combo(self, keep_index=None):
        """下拉项显示为 'ID. 类别名'，ID 与标注txt首列 class_id 一致(从0开始)"""
        if keep_index is None:
            keep_index = self.class_combo.currentIndex()
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        self.class_combo.addItems([f"{i}. {n}" for i, n in enumerate(self.classes)])
        if 0 <= keep_index < len(self.classes):
            self.class_combo.setCurrentIndex(keep_index)
        self.class_combo.blockSignals(False)

    def manage_classes(self):
        if not self.label_folder:
            QMessageBox.warning(self, "提示", "请先选择标签文件夹（类别将保存到该目录 classes.txt）")
            return
        dialog = ClassManagerDialog(self.classes, self)
        if dialog.exec_() == QDialog.Accepted:
            self.classes = dialog.get_classes()
            self.classes_file = os.path.join(self.label_folder, 'classes.txt')
            os.makedirs(self.label_folder, exist_ok=True)
            with open(self.classes_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.classes) + ('\n' if self.classes else ''))
            self._refresh_class_combo(keep_index=0 if self.classes else -1)
            self._set_status(f"已更新类别: {len(self.classes)} 个")

    def get_current_class_name(self):
        i = self.class_combo.currentIndex()
        return self.classes[i] if 0 <= i < len(self.classes) else ""

    def get_current_class_index(self):
        return self.class_combo.currentIndex()

    def _ensure_class_selected(self):
        if self.get_current_class_index() < 0:
            QMessageBox.warning(self, "提示", "请先添加并选择类别")
            return False
        return True

    # ---------- 图片与标注 ----------
    def _load_image(self, index):
        if index < 0 or index >= len(self.image_files):
            return
        self.current_image_index = index
        img_path = self.image_files[index]
        pixmap = QPixmap(img_path)
        if pixmap.isNull():
            self._set_status(f"无法读取图片: {os.path.basename(img_path)}")
            return
        self.current_image_path = img_path
        self.current_boxes = []
        self._load_labels_for_image(img_path)
        self.canvas.boxes = self.current_boxes
        self.canvas.set_image(pixmap)
        self._loading_file_list = True
        self.file_list.setCurrentRow(index)
        self._loading_file_list = False
        self._set_status(f"[{index + 1}/{len(self.image_files)}] "
                         f"{os.path.basename(img_path)}，{len(self.current_boxes)} 个框")

    def _load_labels_for_image(self, img_path):
        if not self.label_folder:
            return
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_folder, base + '.txt')
        if not os.path.exists(label_path):
            self._update_box_list()
            return
        marks = []
        mark_path = os.path.join(self.label_folder, base + '.mark')
        if os.path.exists(mark_path):
            with open(mark_path, 'r', encoding='utf-8') as f:
                marks = [l.strip() for l in f.readlines()]
        with open(label_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls_id = int(parts[0])
                    xc, yc, w, h = (float(parts[1]), float(parts[2]),
                                    float(parts[3]), float(parts[4]))
                except ValueError:
                    continue
                cls_name = self.classes[cls_id] if 0 <= cls_id < len(self.classes) \
                    else f"class_{cls_id}"
                box = BoundingBox(xc, yc, w, h, cls_name, cls_id)
                if line_idx < len(marks) and 'ai' in marks[line_idx].lower():
                    box.is_ai_generated = True
                self.current_boxes.append(box)
        self._update_box_list()

    def _update_box_list(self):
        self.box_list.clear()
        for i, box in enumerate(self.current_boxes):
            prefix = "[AI] " if box.is_ai_generated else ""
            self.box_list.addItem(QListWidgetItem(f"{i + 1}. {prefix}{box.class_name}"))

    def _save_labels(self):
        if self.current_image_index < 0 or not self.current_image_path \
                or not self.label_folder:
            return
        base = os.path.splitext(os.path.basename(self.current_image_path))[0]
        os.makedirs(self.label_folder, exist_ok=True)
        label_path = os.path.join(self.label_folder, base + '.txt')
        mark_path = os.path.join(self.label_folder, base + '.mark')
        with open(label_path, 'w', encoding='utf-8') as f:
            for box in self.current_boxes:
                f.write(box.to_yolo_line() + '\n')
        with open(mark_path, 'w', encoding='utf-8') as f:
            for box in self.current_boxes:
                f.write("AI\n" if box.is_ai_generated else "人工\n")

    def reload_current_image(self):
        if self.current_image_index >= 0:
            self._load_image(self.current_image_index)

    def prev_image(self):
        if self.current_image_index > 0:
            self._save_labels()
            self._load_image(self.current_image_index - 1)

    def next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self._save_labels()
            self._load_image(self.current_image_index + 1)

    def on_file_selected(self, row):
        if self._loading_file_list:
            return
        if 0 <= row < len(self.image_files) and row != self.current_image_index:
            self._save_labels()
            self._load_image(row)

    def on_box_added(self, box):
        if box.class_index < 0:
            QMessageBox.warning(self, "提示", "请先添加并选择类别")
            return
        self.current_boxes.append(box)
        self._update_box_list()
        self._save_labels()
        self.canvas.update_display()

    def on_box_selected(self, row):
        for i, box in enumerate(self.current_boxes):
            box.selected = (i == row)
        self.canvas.update_display()

    def delete_selected_box(self):
        row = self.box_list.currentRow()
        if 0 <= row < len(self.current_boxes):
            self.current_boxes.pop(row)
            self._update_box_list()
            self._save_labels()
            self.canvas.update_display()

    def delete_all_boxes(self):
        if not self.current_boxes:
            self._set_status("当前图片没有标注框")
            return
        reply = QMessageBox.question(
            self, "确认", f"确定删除当前图片的全部 {len(self.current_boxes)} 个标注框吗？")
        if reply == QMessageBox.Yes:
            self.current_boxes = []
            self.canvas.boxes = self.current_boxes
            self._update_box_list()
            self._save_labels()
            self.canvas.update_display()
            self._set_status("已删除当前图片的所有标注框")

    def delete_file(self):
        row = self.file_list.currentRow()
        if 0 <= row < len(self.image_files):
            reply = QMessageBox.question(self, "确认",
                                         f"确定要删除图片 {os.path.basename(self.image_files[row])} 吗？")
            if reply == QMessageBox.Yes:
                os.remove(self.image_files[row])
                self.current_image_index = -1
                self._load_image_files()

    def check_annotations(self):
        if not self.image_files:
            QMessageBox.warning(self, "提示", "请先选择图片文件夹")
            return
        annotated = unannotated = 0
        for img_path in self.image_files:
            base = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(self.label_folder, base + '.txt') if self.label_folder else ""
            if label_path and os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                annotated += 1
            else:
                unannotated += 1
        QMessageBox.information(self, "标注检查",
                                f"图片总数: {len(self.image_files)}\n"
                                f"已标注: {annotated}\n未标注: {unannotated}")

    # ---------- 标注统计 ----------
    def refresh_annotation_stats(self):
        if not self.image_files or not self.label_folder:
            self.stats_label.setText("选择图片与标签文件夹后自动统计")
            self.stats_text.clear()
            return
        counts = {}
        annotated = 0
        total_boxes = 0
        for img_path in self.image_files:
            base = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(self.label_folder, base + '.txt')
            if not (os.path.exists(label_path) and os.path.getsize(label_path) > 0):
                continue
            annotated += 1
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        try:
                            cid = int(parts[0])
                        except ValueError:
                            continue
                        counts[cid] = counts.get(cid, 0) + 1
                        total_boxes += 1
            except OSError:
                continue
        self.stats_label.setText(
            f"图片 {annotated}/{len(self.image_files)} 已标注，框总数 {total_boxes}")
        lines = [f"已标注: {annotated}/{len(self.image_files)}",
                 f"标注框总数: {total_boxes}",
                 "---- 按类别 ----"]
        for cid in sorted(counts, key=lambda k: -counts[k]):
            name = self.classes[cid] if 0 <= cid < len(self.classes) else f"class_{cid}"
            lines.append(f"{name}: {counts[cid]}")
        if not counts:
            lines.append("(暂无标注)")
        self.stats_text.setPlainText('\n'.join(lines))

    # ---------- 工具模式 ----------
    def toggle_draw_mode(self):
        checked = self.draw_btn.isChecked()
        if checked and self.sam_btn.isChecked():
            self.sam_btn.setChecked(False)
            self.canvas.tool = ImageLabel.TOOL_IDLE
        self.canvas.tool = ImageLabel.TOOL_DRAW if checked else ImageLabel.TOOL_IDLE
        self.canvas.update_display()

    def toggle_sam_mode(self):
        self.set_sam_mode(self.sam_btn.isChecked())

    def set_sam_mode(self, checked):
        if checked and not self._ensure_class_selected():
            self.sam_btn.setChecked(False)
            return
        self.sam_btn.setChecked(checked)
        if checked and self.draw_btn.isChecked():
            self.draw_btn.setChecked(False)
            self.canvas.tool = ImageLabel.TOOL_IDLE
        if checked:
            if not self.sam_model:
                self._preload_sam_model()
            self.canvas.tool = ImageLabel.TOOL_SAM
        else:
            self.canvas.tool = ImageLabel.TOOL_IDLE
        self.canvas.update_display()

    # ---------- SAM分割 ----------
    def _populate_sam_models(self):
        models = scan_sam_models()
        self.sam_model_combo.clear()
        self.sam_model_combo.addItems(models)
        if models:
            self._set_status(f"发现 {len(models)} 个SAM模型")

    def _find_sam_model_path(self, model_name):
        return find_sam_model_path(model_name)

    def _preload_sam_model(self):
        model_name = self.sam_model_combo.currentText()
        if not model_name:
            self._set_status("未找到SAM模型，请将 sam2_*.pt 放入 sam/ 或 weights/ 目录")
            return
        full_path = self._find_sam_model_path(model_name)
        if not full_path:
            return
        self.sam_model = None
        self._sam_load_worker = SAMModelLoadWorker(model_name, full_path)
        self._sam_load_worker.finished_signal.connect(self._on_sam_model_loaded)
        self._sam_load_worker.start()
        self._set_status(f"正在加载SAM模型 {model_name} ...")

    def _on_sam_model_loaded(self, model, error, model_name):
        if error:
            self._set_status(f"SAM模型加载失败: {error}")
            return
        if model_name == self.sam_model_combo.currentText():
            self.sam_model = model
            self.sam_model_name = model_name
            self._set_status(f"SAM模型已就绪: {model_name}")

    def on_sam_model_changed(self, model_name):
        if model_name and model_name != self.sam_model_name:
            self.sam_model = None
            self._preload_sam_model()

    def _get_sam_imgsz(self):
        settings = load_sam_settings()
        return resolve_sam_imgsz(settings.get('image', {}).get('imgsz'))

    def on_sam_click(self, img_x, img_y):
        self._run_sam_segment(point=[float(img_x), float(img_y)])

    def on_sam_box(self, x1, y1, x2, y2):
        bbox = [float(min(x1, x2)), float(min(y1, y2)), float(max(x1, x2)), float(max(y1, y2))]
        self._run_sam_segment(bbox=bbox)

    def _run_sam_segment(self, point=None, bbox=None):
        if not self.current_image_path:
            return
        if not self._ensure_class_selected():
            return
        if not self.sam_model:
            self._preload_sam_model()
            self._set_status("SAM模型尚未加载完成，加载后请重新点击目标")
            return
        self._sam_request_counter += 1
        ctx = SamRequestContext(
            request_id=self._sam_request_counter,
            image_path=self.current_image_path,
            class_index=self.get_current_class_index(),
            class_name=self.get_current_class_name(),
        )
        self._sam_worker = SAMSegmentWorker(self.sam_model, self.current_image_path,
                                            point=point, bbox=bbox,
                                            request_id=ctx.request_id,
                                            imgsz=self._get_sam_imgsz())
        self._sam_worker.finished_signal.connect(
            lambda masks, error, rid, c=ctx: self._on_sam_segment_finished(masks, error, rid, c))
        self._sam_worker.start()
        self._set_status("SAM分割中...")

    def _on_sam_segment_finished(self, mask_data, error_msg, request_id, ctx):
        if error_msg:
            self._set_status(f"SAM分割失败: {error_msg}")
            return
        if request_id != self._sam_request_counter or ctx.image_path != self.current_image_path:
            return
        if mask_data is None or len(mask_data) == 0:
            self._set_status("SAM未检测到目标")
            return
        pixmap = QPixmap(ctx.image_path)
        if pixmap.isNull():
            return
        bbox = self._mask_to_bbox(mask_data[0])
        if not bbox:
            self._set_status("SAM结果解析失败")
            return
        box = BoundingBox(bbox['x_center'], bbox['y_center'], bbox['width'], bbox['height'],
                          ctx.class_name, ctx.class_index)
        box.is_ai_generated = True
        self.on_box_added(box)
        self._set_status(f"SAM分割完成: {ctx.class_name}")

    @staticmethod
    def _mask_to_bbox(mask):
        """mask -> 归一化bbox（mask本身即推理分辨率下的分割结果，直接归一化）"""
        try:
            if hasattr(mask, 'cpu'):
                import numpy as np
                mask = mask.cpu().numpy()
            if mask.size == 0:
                return None
            rows = mask.any(axis=1)
            cols = mask.any(axis=0)
            if not rows.any() or not cols.any():
                return None
            y_min, y_max = rows.argmax(), len(rows) - rows[::-1].argmax() - 1
            x_min, x_max = cols.argmax(), len(cols) - cols[::-1].argmax() - 1
            mask_h, mask_w = mask.shape
            return {
                'x_center': (x_min + x_max) / 2 / mask_w,
                'y_center': (y_min + y_max) / 2 / mask_h,
                'width': (x_max - x_min + 1) / mask_w,
                'height': (y_max - y_min + 1) / mask_h,
            }
        except Exception:
            return None

    # ---------- 对话框入口 ----------
    def open_ai_prelabel(self):
        if not self.image_files:
            QMessageBox.warning(self, "提示", "请先选择图片文件夹")
            return
        if self.ai_dialog is None:
            self.ai_dialog = AIPrelabelDialog(self, self)
        self.ai_dialog.show()
        self.ai_dialog.raise_()

    def open_sam_settings(self):
        dialog = SamSettingsDialog(self)
        dialog.exec_()

    # ---------- 其他 ----------
    def _set_status(self, msg):
        self.status_label.setText(msg)

    def log(self, msg):
        self._set_status(str(msg))

    def hideEvent(self, event):
        self._save_labels()
        super().hideEvent(event)
