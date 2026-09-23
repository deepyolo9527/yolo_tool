# -*- coding: utf-8 -*-
"""路径辅助：兼容开发环境与打包(exe)环境"""

import os
import sys


def get_app_dir():
    """可读写的应用目录：打包后为 exe 所在目录，源码运行为项目根目录"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_bundled_dir():
    """只读打包资源目录(onefile 解压到临时 _MEIxxxx)；源码运行为项目根目录"""
    if getattr(sys, 'frozen', False):
        return getattr(sys, '_MEIPASS', get_app_dir())
    return get_app_dir()


def get_weights_dir():
    return os.path.join(get_app_dir(), "weights")


def get_sam_dir():
    return os.path.join(get_app_dir(), "sam")


def get_configs_dir():
    return os.path.join(get_app_dir(), "configs")


def get_ultralytics_cfg_dir():
    """模型结构/超参数 YAML 所在的 cfg 目录：打包后在解压目录内，源码运行为仓库副本"""
    if getattr(sys, 'frozen', False):
        return os.path.join(get_bundled_dir(), "ultralytics", "cfg")
    return os.path.join(get_app_dir(), "ultralytics", "ultralytics", "cfg")
