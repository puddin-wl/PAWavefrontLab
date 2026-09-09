"""VS Code entry point: edit the settings below, then click Run Python File."""

from pathlib import Path
import subprocess
import sys


# ======================== 只需要修改这里 ========================

# 示例占位路径；请改为本机输入图。真实实验路径不要提交到仓库。
INPUT_IMAGE = "data/example_input.tif"

# 结果保存位置
OUTPUT_DIR = "outputs/single_aberration"

# 输出图片尺寸，必须是正偶数
SIZE = 256

# AOtools Noll 编号: 系数，单位 rad
# 例如：4 是 Defocus；7、8 是两个方向的 Coma
ZERNIKE_COEFFICIENTS = {
    4: 1.5,
    7: -0.25,
}

# "cuda"、"cpu" 或 "auto"
DEVICE = "cuda"

# 展示时一般保持 0；需要时可添加高斯噪声
NOISE_STD = 0.0
NOISE_SEED = 0

# ================================================================


PROJECT_ROOT = Path(__file__).resolve().parent


def build_command() -> list[str]:
    if not ZERNIKE_COEFFICIENTS:
        raise ValueError("ZERNIKE_COEFFICIENTS 至少需要填写一个 Noll 系数。")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "apply_aberration.py"),
        "--input-image",
        INPUT_IMAGE,
        "--output-dir",
        str(PROJECT_ROOT / OUTPUT_DIR),
        "--size",
        str(SIZE),
        "--device",
        DEVICE,
        "--noise-std",
        str(NOISE_STD),
        "--seed",
        str(NOISE_SEED),
    ]
    for noll_index, coefficient in sorted(ZERNIKE_COEFFICIENTS.items()):
        command.extend(["--coefficient", f"{noll_index}={coefficient}"])
    return command


def main() -> None:
    print("正在生成单张像差展示图……", flush=True)
    subprocess.run(build_command(), cwd=PROJECT_ROOT, check=True)
    print(f"完成。请查看：{PROJECT_ROOT / OUTPUT_DIR}")


if __name__ == "__main__":
    main()
