# -*- coding: utf-8 -*-
"""系统硬件信息与实时占用采集，供环境配置页的「系统硬件」分区使用。

GPU 状态走 nvidia-smi 子进程而不是 pynvml：本机实测 pynvml 读取利用率会触发访问违例
（access violation）直接崩掉整个进程，Python 层捕获不到，因此不引入 pynvml。
"""

import os
import shutil
import subprocess
import sys

_SMI_COLUMNS = "index,name,driver_version,memory.total,memory.used,utilization.gpu"

# 型号/驱动等静态信息不必每次刷新都重新问系统
_static_cache = {}


def _to_float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _fmt_gb(nbytes):
    return round(nbytes / 1024 ** 3, 1)


def _nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    candidate = os.path.join(system_root, "System32", "nvidia-smi.exe")
    return candidate if os.path.exists(candidate) else None


def nvidia_gpus():
    """显卡清单，每项含 name/driver/mem_total_mb/mem_used_mb/util；读不到时返回空列表"""
    exe = _nvidia_smi()
    if not exe:
        return []

    cmd = [exe, f"--query-gpu={_SMI_COLUMNS}", "--format=csv,noheader,nounits"]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=5,
            **({"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}),
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []

    gpus = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6 or not parts[1]:
            continue
        gpus.append({
            "index": parts[0],
            "name": parts[1],
            "driver": parts[2],
            "mem_total_mb": _to_float(parts[3]),
            "mem_used_mb": _to_float(parts[4]),
            "util": _to_float(parts[5]),
        })
    return gpus


def cuda_ready():
    """当前 PyTorch 是否能用 GPU（CPU 版 torch 即使插着显卡也是 False）"""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def cpu_model():
    if "cpu_name" not in _static_cache:
        _static_cache["cpu_name"] = _read_cpu_model()
    return _static_cache["cpu_name"]


def _read_cpu_model():
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            with key:
                name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return " ".join(str(name).split())
        except OSError:
            pass
    import platform
    return platform.processor() or "未知型号"


def snapshot():
    """一次完整采集，返回 {"cpu":…, "memory":…, "gpu":…}，供界面直接渲染"""
    return {"cpu": _cpu_snapshot(), "memory": _memory_snapshot(), "gpu": _gpu_snapshot()}


def _cpu_snapshot():
    cores = os.cpu_count() or 0
    info = {"name": cpu_model(), "cores": f"{cores} 线程", "percent": None, "detail": ""}
    try:
        import psutil
    except ImportError:
        return info

    physical = psutil.cpu_count(logical=False) or cores
    info["cores"] = f"{physical} 核 / {cores} 线程"
    # interval=None 的首次采样恒为 0，这里用短间隔实测，采集在后台线程执行不卡界面
    info["percent"] = round(psutil.cpu_percent(interval=0.15), 1)
    freq = psutil.cpu_freq()
    if freq and freq.current:
        info["detail"] = f"当前主频 {freq.current / 1000:.2f} GHz"
    return info


def _memory_snapshot():
    info = {"percent": None, "detail": "未安装 psutil，无法读取内存占用"}
    try:
        import psutil
    except ImportError:
        return info
    vm = psutil.virtual_memory()
    info["percent"] = round(vm.percent, 1)
    info["detail"] = (f"已用 {_fmt_gb(vm.used)} / {_fmt_gb(vm.total)} GB"
                      f"（可用 {_fmt_gb(vm.available)} GB）")
    return info


def _gpu_snapshot():
    gpus = nvidia_gpus()
    ready = cuda_ready()

    if not gpus:
        return {"available": False, "name": "未检测到 NVIDIA 显卡",
                "detail": "当前环境 GPU 不可用，训练与推理将使用 CPU 运行", "percent": None}

    gpu = gpus[0]
    name = gpu["name"]
    if gpu["driver"]:
        name += f" · 驱动 {gpu['driver']}"
    if len(gpus) > 1:
        name += f" 等 {len(gpus)} 张显卡"

    if not ready:
        return {"available": False, "name": name,
                "detail": "已检测到显卡，但当前 PyTorch 不支持 GPU（多为 CPU 版 torch），"
                          "当前环境 GPU 不可用，安装 CUDA 版 PyTorch 后即可启用",
                "percent": None}

    percent = None
    detail = []
    if gpu["util"] is not None:
        percent = gpu["util"]
        detail.append(f"核心利用率 {gpu['util']:.0f}%")
    if gpu["mem_total_mb"]:
        used = gpu["mem_used_mb"] or 0
        detail.append(f"显存 {used / 1024:.1f} / {gpu['mem_total_mb'] / 1024:.1f} GB")
        if percent is None and gpu["mem_total_mb"] > 0:
            percent = round(used / gpu["mem_total_mb"] * 100, 1)
    return {"available": True, "name": name, "detail": " · ".join(detail),
            "percent": percent}
