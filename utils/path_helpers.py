# -*- coding: utf-8 -*-
"""路径辅助：兼容开发环境与打包(exe)环境"""

import os
import shutil
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
    """界面上可编辑的 ultralytics 配置目录（模型结构 / 超参数 YAML）

    源码运行直接用仓库内的副本；打包运行时从临时解压目录复制到 exe 旁的
    configs/ultralytics_cfg，否则界面上会显示 _MEIxxxx 临时路径，用户也改不了这些 YAML。
    """
    if not getattr(sys, 'frozen', False):
        return os.path.join(get_app_dir(), "ultralytics", "ultralytics", "cfg")

    src = os.path.join(get_bundled_dir(), "ultralytics", "cfg")
    target = os.path.join(get_configs_dir(), "ultralytics_cfg")
    try:
        for name in ("default.yaml", "models"):
            s, d = os.path.join(src, name), os.path.join(target, name)
            if os.path.isdir(s):
                if not os.path.isdir(d):
                    shutil.copytree(s, d)
            elif os.path.isfile(s) and not os.path.isfile(d):
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copy2(s, d)
    except OSError:
        return src    # exe 所在目录只读等情况下，退回解压目录
    return target
