"""Deprecated VS Code wrapper for one-step synthetic dataset generation.

New experiments should use workflows/static_simulation/step1 through step5.
This wrapper remains available to reproduce the earlier generation-only flow.
"""

from pathlib import Path
import subprocess
import sys


# ======================== 只需要修改这里 ========================

# 输入的静态物体图片
# 示例占位路径；请改为本机输入图。真实实验路径不要提交到仓库。
INPUT_IMAGE = "data/example_input.tif"

# 数据集保存位置
OUTPUT_DIR = "data/vscode_dataset"

# 完整方形区域都有光；默认不需要填写 aperture height
SIZE = 256
NUM_FRAMES = 100

# SLM 使用 Noll 1-15，系数独立高斯分布
SLM_SIGMA = 5.0

# 数据集像差独立随机生成，不使用单图展示程序的指定系数
# 可选 "zernike" 或 "complex-gaussian"
ABERRATION_MODE = "zernike"
ABERRATION_SIGMA = 1.0

NOISE_STD = 0.0
SEED = 0
BATCH_SIZE = 8
DEVICE = "cuda"

# 仅当需要复现论文 144×256 几何时改成 144；当前项目保持 None
APERTURE_HEIGHT = None

# ================================================================


PROJECT_ROOT = Path(__file__).resolve().parent


def build_command() -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "generate_neuws_data.py"),
        "simulate",
        "--input-image",
        INPUT_IMAGE,
        "--output-dir",
        str(PROJECT_ROOT / OUTPUT_DIR),
        "--size",
        str(SIZE),
        "--num-frames",
        str(NUM_FRAMES),
        "--slm-sigma",
        str(SLM_SIGMA),
        "--aberration-mode",
        ABERRATION_MODE,
        "--aberration-sigma",
        str(ABERRATION_SIGMA),
        "--noise-std",
        str(NOISE_STD),
        "--seed",
        str(SEED),
        "--batch-size",
        str(BATCH_SIZE),
        "--device",
        DEVICE,
    ]
    if APERTURE_HEIGHT is not None:
        command.extend(["--aperture-height", str(APERTURE_HEIGHT)])
    return command


def main() -> None:
    print(
        "提示：run_dataset.py 已 deprecated；完整新实验请使用 "
        "workflows/static_simulation/step1 到 step5。",
        flush=True,
    )
    print(f"正在生成 {NUM_FRAMES} 帧 NeuWS 数据集……", flush=True)
    subprocess.run(build_command(), cwd=PROJECT_ROOT, check=True)
    print(f"完成。请查看：{PROJECT_ROOT / OUTPUT_DIR}")


if __name__ == "__main__":
    main()
