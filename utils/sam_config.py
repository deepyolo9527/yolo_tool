# -*- coding: utf-8 -*-
"""SAM配置管理：全局配置文件 configs/sam_settings.json"""

import copy
import json
import os
import tempfile

from utils.path_helpers import get_configs_dir

SAM_DEFAULT_IMAGE_SIZE = 1024
SAM_MIN_IMAGE_SIZE = 32
SAM_MAX_IMAGE_SIZE = 4096
SAM_IMAGE_SIZES = [256, 512, 768, 1024, 1280, 1536, 2048]

_QUALITY_LIMITS = {
    'min_area_ratio': (0.0, 1.0, 0.45),
    'max_area_ratio': (0.0, 10.0, 1.8),
    'min_previous_iou': (0.0, 1.0, 0.02),
    'max_center_distance_ratio': (0.0, 10.0, 1.25),
    'max_frame_area_ratio': (0.0, 10.0, 0.85),
}

SAM_VIDEO_PRESETS = {
    '严格防漂移': {
        'min_area_ratio': 0.08,
        'max_area_ratio': 0.8,
        'min_previous_iou': 0.7,
        'max_center_distance_ratio': 0.25,
        'max_frame_area_ratio': 0.9,
    },
    '均衡': {
        'min_area_ratio': 0.02,
        'max_area_ratio': 1.25,
        'min_previous_iou': 0.6,
        'max_center_distance_ratio': 1.5,
        'max_frame_area_ratio': 0.85,
    },
    '快速运动': {
        'min_area_ratio': 0.005,
        'max_area_ratio': 2.5,
        'min_previous_iou': 0.02,
        'max_center_distance_ratio': 1.25,
        'max_frame_area_ratio': 0.7,
    },
}

DEFAULT_SAM_SETTINGS = {
    'image': {
        'imgsz': SAM_DEFAULT_IMAGE_SIZE,
    },
    'video': {
        'imgsz': SAM_DEFAULT_IMAGE_SIZE,
        'preset': '均衡',
        'quality': {
            'min_area_ratio': 0.02,
            'max_area_ratio': 1.25,
            'min_previous_iou': 0.6,
            'max_center_distance_ratio': 1.5,
            'max_frame_area_ratio': 0.85,
        },
    },
}


def default_sam_settings():
    return copy.deepcopy(DEFAULT_SAM_SETTINGS)


def is_valid_sam_imgsz(value):
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    if value < SAM_MIN_IMAGE_SIZE or value > SAM_MAX_IMAGE_SIZE:
        return False
    return value % 32 == 0


def resolve_sam_imgsz(value, default=SAM_DEFAULT_IMAGE_SIZE):
    if is_valid_sam_imgsz(value):
        return value
    return default


def normalize_sam_quality(values, fallback=None):
    if fallback is None:
        fallback = DEFAULT_SAM_SETTINGS['video']['quality']
    result = {}
    for key, (min_v, max_v, default_v) in _QUALITY_LIMITS.items():
        raw = values.get(key, fallback.get(key, default_v))
        try:
            v = float(raw)
            if v < min_v or v > max_v:
                v = fallback.get(key, default_v)
        except (ValueError, TypeError):
            v = fallback.get(key, default_v)
        result[key] = v
    return result


def normalize_sam_settings(values, fallback=None):
    if fallback is None:
        fallback = default_sam_settings()
    result = default_sam_settings()

    img_cfg = values.get('image', {})
    result['image']['imgsz'] = resolve_sam_imgsz(
        img_cfg.get('imgsz'), fallback.get('image', {}).get('imgsz'))

    vid_cfg = values.get('video', {})
    result['video']['imgsz'] = resolve_sam_imgsz(
        vid_cfg.get('imgsz'), fallback.get('video', {}).get('imgsz'))
    result['video']['preset'] = vid_cfg.get('preset', fallback.get('video', {}).get('preset', '均衡'))
    result['video']['quality'] = normalize_sam_quality(
        vid_cfg.get('quality', {}), fallback=fallback.get('video', {}).get('quality', {}))
    return result


def get_sam_config_file():
    return os.path.join(get_configs_dir(), "sam_settings.json")


def load_sam_settings():
    config_file = get_sam_config_file()
    if not os.path.exists(config_file):
        return default_sam_settings()
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return default_sam_settings()
    return normalize_sam_settings(data.get('sam', {}))


def save_sam_settings(settings):
    config_file = get_sam_config_file()
    data = {}
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}
    data['sam'] = normalize_sam_settings(settings)

    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(config_file), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, config_file)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
