
# -*- coding: utf-8 -*-

from PyQt5.QtCore import QThread, pyqtSignal
import os
import sys
import time
import cv2
from pathlib import Path


class ImageInferenceWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    progress_signal = pyqtSignal(int, int)
    result_signal = pyqtSignal(str, str)
    preview_signal = pyqtSignal(str, object, object)   # 原图路径, 原始BGR, 标注BGR

    def __init__(self, params):
        super().__init__()
        self.params = params
        self._stop_flag = False
        self.model = None

    def stop(self):
        self._stop_flag = True

    def run(self):
        try:
            try:
                from ultralytics import YOLO
            except ImportError:
                import importlib
                ultralytics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ultralytics')
                sys.path.insert(0, ultralytics_path)
                if 'ultralytics' in sys.modules:
                    del sys.modules['ultralytics']
                from ultralytics import YOLO

            model_path = self.params.get('model', 'weights/yolo26n.pt')
            image_paths = self.params.get('images', [])
            output_dir = self.params.get('output_dir', 'runs/detect')
            conf_threshold = self.params.get('conf', 0.25)
            iou_threshold = self.params.get('iou', 0.7)
            device = self.params.get('device', '0')
            classes = self.params.get('classes', None)
            save_txt = self.params.get('save_txt', False)
            save_image = self.params.get('save_image', False)

            self.model = YOLO(model_path)
            self._log(f"加载模型: {model_path}")

            os.makedirs(output_dir, exist_ok=True)
            predict_dir = os.path.join(output_dir, 'predict')
            os.makedirs(predict_dir, exist_ok=True)

            total = len(image_paths)
            for idx, img_path in enumerate(image_paths):
                if self._stop_flag:
                    self._log("推理已取消")
                    self.finished_signal.emit(False, "推理已取消")
                    return

                if not os.path.exists(img_path):
                    self._log(f"跳过不存在的文件: {img_path}")
                    continue

                self._log(f"处理 [{idx+1}/{total}]: {img_path}")
                self.progress_signal.emit(idx + 1, total)

                results = self.model.predict(
                    img_path,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    device=device,
                    save=False,
                    classes=classes,
                )

                result = results[0]
                base_name = os.path.splitext(os.path.basename(img_path))[0]

                annotated_frame = result.plot()
                self.preview_signal.emit(img_path, result.orig_img, annotated_frame)

                if save_image:
                    save_path = os.path.join(predict_dir, os.path.basename(img_path))
                    cv2.imwrite(save_path, annotated_frame)
                    self.result_signal.emit(img_path, save_path)
                    self._log(f"图像已保存: {save_path}")

                if save_txt:
                    txt_path = os.path.join(predict_dir, f"{base_name}.txt")
                    with open(txt_path, 'w') as f:
                        for box in result.boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            x_center, y_center, width, height = box.xywhn[0].tolist()
                            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")
                    self._log(f"TXT已保存: {txt_path}")

            self._log("图片推理完成！")
            self.finished_signal.emit(True, f"成功处理 {total} 张图片")
        except Exception as e:
            if self._stop_flag:
                self.finished_signal.emit(False, "推理已停止")
            else:
                import traceback
                self._log(f"错误详情: {traceback.format_exc()}")
                self.finished_signal.emit(False, str(e))

    def _log(self, msg):
        self.log_signal.emit(msg)


class VideoInferenceWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    progress_signal = pyqtSignal(int, int)
    frame_signal = pyqtSignal(int)
    preview_signal = pyqtSignal(object, object)   # 原始BGR帧, 标注BGR帧

    def __init__(self, params):
        super().__init__()
        self.params = params
        self._stop_flag = False
        self.model = None

    def stop(self):
        self._stop_flag = True

    def run(self):
        try:
            try:
                from ultralytics import YOLO
            except ImportError:
                import importlib
                ultralytics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ultralytics')
                sys.path.insert(0, ultralytics_path)
                if 'ultralytics' in sys.modules:
                    del sys.modules['ultralytics']
                from ultralytics import YOLO

            model_path = self.params.get('model', 'weights/yolo26n.pt')
            video_path = self.params.get('video', '')
            output_dir = self.params.get('output_dir', 'runs/detect')
            conf_threshold = self.params.get('conf', 0.25)
            iou_threshold = self.params.get('iou', 0.7)
            device = self.params.get('device', '0')
            frame_interval = self.params.get('frame_interval', 1)
            classes = self.params.get('classes', None)
            save_txt = self.params.get('save_txt', False)
            save_image = self.params.get('save_image', True)

            self.model = YOLO(model_path)
            self._log(f"加载模型: {model_path}")

            if not os.path.exists(video_path):
                raise Exception(f"视频文件不存在: {video_path}")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"无法打开视频文件: {video_path}")

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            self._log(f"视频信息: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")

            os.makedirs(output_dir, exist_ok=True)
            
            txt_dir = None
            if save_txt:
                txt_dir = os.path.join(output_dir, 'labels')
                os.makedirs(txt_dir, exist_ok=True)

            output_path = os.path.join(output_dir, 'video_output.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0
            processed_count = 0
            preview_interval = 1.0 / 15
            last_preview = 0.0

            while cap.isOpened():
                if self._stop_flag:
                    self._log("视频推理已取消")
                    cap.release()
                    out.release()
                    self.finished_signal.emit(False, "推理已取消")
                    return

                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                if frame_count % frame_interval != 0:
                    out.write(frame)
                    continue

                self.frame_signal.emit(frame_count)

                results = self.model.predict(
                    frame,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    device=device,
                    verbose=False,
                    classes=classes,
                )

                result = results[0]

                if save_txt:
                    txt_path = os.path.join(txt_dir, f"{frame_count:06d}.txt")
                    with open(txt_path, 'w') as f:
                        for box in result.boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            x_center, y_center, width_box, height_box = box.xywhn[0].tolist()
                            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width_box:.6f} {height_box:.6f} {conf:.6f}\n")

                annotated_frame = result.plot()

                # 限制预览推送频率，避免高速推理时事件队列堆积
                now = time.time()
                if now - last_preview >= preview_interval:
                    last_preview = now
                    self.preview_signal.emit(frame, annotated_frame)

                if save_image:
                    out.write(annotated_frame)
                else:
                    out.write(frame)

                processed_count += 1

                progress = int((frame_count / total_frames) * 100)
                self.progress_signal.emit(progress, 100)

            cap.release()
            out.release()

            self._log(f"视频推理完成！处理 {processed_count} 帧")
            self._log(f"输出视频: {output_path}")
            if save_txt:
                self._log(f"TXT标签已保存到: {txt_dir}")
            self.finished_signal.emit(True, f"视频推理完成，输出文件: {output_path}")
        except Exception as e:
            if self._stop_flag:
                self.finished_signal.emit(False, "推理已停止")
            else:
                import traceback
                self._log(f"错误详情: {traceback.format_exc()}")
                self.finished_signal.emit(False, str(e))

    def _log(self, msg):
        self.log_signal.emit(msg)
