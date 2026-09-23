#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 assets/logo.png 生成 assets/app.ico（16~256 多尺寸程序图标）

用法: python tools/make_icon.py
换 logo 后重跑一次即可；程序内图标由 main._app_icon() 读取 assets/app.ico。
"""

import os

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "assets", "logo.png")
OUT = os.path.join(ROOT, "assets", "app.ico")
SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
BG = np.array([236, 241, 244], dtype=np.float32)   # logo 的浅灰底，用于分离图标与背景


def locate_tile(im):
    """在 logo 中定位上方圆角方块（下方 'YOLO 深度学习' 文字不进入图标）"""
    w, h = im.size
    small = im.resize((400, int(400 * h / w)), Image.LANCZOS)
    arr = np.asarray(small)[:, :, :3].astype(np.float32)
    fg = np.sqrt(((arr - BG) ** 2).sum(axis=2)) > 60

    rows = fg.mean(axis=1) > 0.5
    runs, start = [], None
    for y, on in enumerate(rows):
        if on and start is None:
            start = y
        elif not on and start is not None:
            runs.append((start, y - 1))
            start = None
    if start is not None:
        runs.append((start, len(rows) - 1))
    r0, r1 = max(runs, key=lambda t: t[1] - t[0])          # 方块远高于一行文字

    cols = np.where(fg[r0:r1 + 1].mean(axis=0) > 0.05)[0]
    sy, sx = h / fg.shape[0], w / fg.shape[1]
    return cols.min() * sx, (cols.max() + 1) * sx, r0 * sy, (r1 + 1) * sy


def build():
    im = Image.open(SRC).convert("RGBA")
    W, H = im.size
    x0, x1, y0, y1 = locate_tile(im)
    side = max(x1 - x0, y1 - y0) * 1.02
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = min(max(0, cx - side / 2), W - side)
    top = min(max(0, cy - side / 2), H - side)
    tile = im.crop((int(left), int(top), int(left + side), int(top + side)))
    print(f"logo {W}x{H} -> 图标区 {int(side)}x{int(side)}，宽高比 "
          f"{(x1 - x0) / (y1 - y0):.3f}")

    tile = tile.resize((1024, 1024), Image.LANCZOS)
    mask = Image.new("L", (4096, 4096), 0)                 # 4x 超采样圆角，边缘无锯齿
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, 4095, 4095], radius=819, fill=255)
    tile.putalpha(mask.resize((1024, 1024), Image.LANCZOS))
    tile.save(OUT, sizes=SIZES)
    print(f"已生成 {os.path.relpath(OUT, ROOT)}（{os.path.getsize(OUT) // 1024} KB, "
          f"含 {len(SIZES)} 个尺寸）")


if __name__ == "__main__":
    build()
