#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
一键打包脚本 - build.py
使用方法: python build.py
          python build.py --with-deps   # 显式允许安装依赖(会改动当前环境)
"""

import os
import sys
import subprocess
import shutil
import tempfile
from pathlib import Path

# 打包时需要随包收集的数据/子模块的第三方包
COLLECT_ALL = ["ultralytics"]


def check_python():
    """检查Python环境"""
    print("[1/5] 检查Python环境...")
    try:
        version = sys.version_info
        if version.major >= 3 and version.minor >= 7:
            print(f"✓ Python {version.major}.{version.minor}.{version.micro} 符合要求")
            return True
        else:
            print("✗ Python版本过低，需要Python 3.7+")
            return False
    except:
        print("✗ 未检测到Python")
        return False


def _installed_version(module):
    """读取已安装包版本，不导入该包（避免 torch/PyQt5 的 DLL 加载顺序问题）"""
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(module)
    except PackageNotFoundError:
        return None


def _numpy_constraints_file():
    """生成 pip 约束文件，把 numpy 钉在当前版本

    requirements.txt 里是 `numpy>=1.20.0` 加 `ultralytics>=8.0.0`，直接安装会按
    最新解析结果升级 numpy，进而让已编译的 opencv/torch 与 numpy ABI 不匹配。
    """
    ver = _installed_version("numpy")
    if not ver:
        return None
    path = os.path.join(tempfile.gettempdir(), "yolo_tool_build_contracts.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"numpy=={ver}\n")
    print(f"! numpy 版本锁定为 {ver}（约束文件: {path}）")
    return path


def check_dependencies():
    """检查打包所需依赖，默认不改动当前环境

    `pip install -r requirements.txt` 会按最新解析版本连带升级 numpy 等已经装好的
    包，因此只在显式传入 --with-deps 时才执行安装，且安装时锁定 numpy 版本。
    """
    print("\n[2/5] 检查依赖...")

    if "--with-deps" in sys.argv:
        if os.path.exists("requirements.txt"):
            print("按参数要求，正在安装 requirements.txt 中的依赖...")
            cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
            constraints = _numpy_constraints_file()
            if constraints:
                cmd.extend(["-c", constraints])
            cmd.extend(["-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])
            subprocess.check_call(cmd)
            print("✓ 依赖安装完成")
        else:
            print("未找到 requirements.txt，跳过")
        return True

    missing = []
    for module, dist in (("PyInstaller", "pyinstaller"), ("torch", "torch"),
                         ("torchvision", "torchvision"), ("ultralytics", "ultralytics"),
                         ("PyQt5", "PyQt5"), ("cv2", "opencv-python"), ("numpy", "numpy"),
                         ("yaml", "pyyaml"), ("matplotlib", "matplotlib")):
        ver = _installed_version(dist)
        if ver:
            print(f"✓ {module} {ver}")
        else:
            try:
                __import__(module)
                print(f"✓ {module}")
            except ImportError:
                missing.append(module)
                print(f"! 缺少 {module}")

    if missing:
        print("\n✗ 以下依赖缺失，打包结果不可用: " + ", ".join(missing))
        print("  请先安装，或改用 python build.py --with-deps（会改动当前环境）")
        return False

    return True


def check_files():
    """检查必要的文件"""
    print("\n[3/5] 检查项目文件...")

    if not os.path.exists("main.py"):
        print("✗ 错误：未找到main.py文件！")
        print("  请确保main.py与build.py在同一目录下")
        return False

    print("✓ main.py 存在")

    # 检查图标文件（assets/app.ico 由 assets/logo.png 生成，含 16~256 多尺寸）
    global icon_file
    icon_file = None
    for icon_name in ["assets/app.ico", "icon.ico", "app.ico"]:
        if os.path.exists(icon_name):
            icon_file = icon_name
            print(f"✓ 图标文件: {icon_name}")
            break

    if not icon_file:
        print("! 未找到图标文件，将使用默认图标")

    # 检查assets文件夹
    if os.path.exists("assets"):
        print("✓ assets文件夹存在")
    else:
        print("! assets文件夹不存在，将创建空文件夹")
        os.makedirs("assets", exist_ok=True)

    return True


def clean_build():
    """清理之前的构建文件"""
    print("\n[4/5] 清理旧的构建文件...")

    folders_to_remove = ["build", "__pycache__"]
    files_to_remove = ["*.spec"]

    for folder in folders_to_remove:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"✓ 删除 {folder}/")

    # 清理spec文件
    for spec_file in Path(".").glob("*.spec"):
        spec_file.unlink()
        print(f"✓ 删除 {spec_file}")


def build_exe():
    """执行打包"""
    print("\n[5/5] 开始打包...")
    print("=" * 50)

    # 构建命令（用 python -m PyInstaller，避免 pyinstaller.exe 不在 PATH）
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",           # 单文件模式
        "--windowed",          # 无控制台窗口（GUI应用）
        "--name=MyApp",        # 输出文件名
        "--clean",             # 清理临时文件
        "--noconfirm",         # 覆盖确认
    ]

    # 添加图标
    if icon_file:
        cmd.extend(["--icon", icon_file])

    # 添加数据文件（如果有）
    if os.path.exists("assets") and os.listdir("assets"):
        cmd.append("--add-data=assets;assets")

    # ultralytics 的模型/超参数 YAML、trackers 配置等数据文件与全部子模块
    for pkg in COLLECT_ALL:
        cmd.extend(["--collect-all", pkg])
        cmd.extend(["--copy-metadata", pkg])

    # 添加隐藏导入（根据你的项目需求修改）
    hidden_imports = [
        "requests",
        "json",
        "os",
        "sys",
        "datetime",
    ]
    for imp in hidden_imports:
        cmd.extend(["--hidden-import", imp])

    # 添加主文件
    cmd.append("main.py")

    # 执行打包（输出直接透传，便于看到进度与告警）
    try:
        result = subprocess.run(cmd)

        if result.returncode == 0:
            print("\n" + "=" * 30)
            print("      🎉 打包成功！")
            print("=" * 30)

            # 显示生成的exe文件
            exe_path = Path("dist") / "MyApp.exe"
            if exe_path.exists():
                size_mb = exe_path.stat().st_size / (1024 * 1024)
                print(f"\n📁 输出路径: {exe_path}")
                print(f"📦 文件大小: {size_mb:.2f} MB")
                print("\n提示：exe 读取的 weights/、datasets/、runs/、configs/、sam/ 均在 exe 同级目录，")
                print("      请把模型和数据集放在 dist/weights、dist/datasets 下。")
        else:
            print("\n❌ 打包失败！退出码: " + str(result.returncode))

    except Exception as e:
        print(f"\n❌ 打包出错: {str(e)}")


def _pause():
    """仅在交互式终端下等待回车，避免管道/重定向时抛 EOFError"""
    if sys.stdin and sys.stdin.isatty():
        input("\n按Enter键退出...")


def main():
    """主函数"""
    print("=" * 40)
    print("   Python一键打包工具 v1.0")
    print("=" * 40)
    print()

    # 检查环境
    if not check_python():
        _pause()
        return

    # 检查依赖
    if not check_dependencies():
        _pause()
        return

    # 检查文件
    if not check_files():
        _pause()
        return

    # 清理旧文件
    clean_build()

    # 打包
    build_exe()

    print("\n" + "=" * 40)
    print("   操作完成！")
    print("=" * 40)
    _pause()


if __name__ == "__main__":
    main()
