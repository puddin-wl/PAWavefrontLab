# 2026-07-28 NeuWS 环境配置与 AOtools 闭环迁移记录

## 1. 今日目标与当前结论

今天完成了 NeuWS 项目在当前 WSL + RTX 5070 Ti 环境中的本地化、AOtools 光学流程迁移、数据生成、静态重建、评估和 VS Code 直接运行入口。

当前项目已经具备两条相互独立的使用流程：

1. **单张图片像差展示**：由用户明确指定 Noll Zernike 系数，生成并保存像差相位、复场、PSF 和加像差后的图片。
2. **NeuWS 数据集生成**：随机生成一个静态未知像差，保持物体和像差不变，只让 SLM 调制逐帧变化，输出可直接供 NeuWS 加载和训练的数据。

项目默认使用**完整方形照明区域**，不再默认使用论文中的 `144×256` 居中活动区域。论文几何仍可通过显式设置 `--aperture-height 144` 使用。

## 2. 当前环境

### 硬件与 WSL

- GPU：NVIDIA GeForce RTX 5070 Ti，约 16 GB 显存
- Windows 驱动：581.80，通过 WSL 映射给 Linux 使用
- CUDA Toolkit：13.0，位置 `/usr/local/cuda`
- 系统 cuDNN：9.25 CUDA 13 软件包
- PyTorch 运行时报告 cuDNN：92000（9.2）

Windows NVIDIA 驱动不需要、也不应该在 WSL 内重复安装。WSL 内安装的 Linux CUDA Toolkit 和 cuDNN 与 Windows 驱动可以共存。

### Python 环境

- Miniconda：`/home/xiangwan/miniconda3`
- Conda 环境：`neuws`
- Python：3.10
- PyTorch：`2.13.0+cu130`
- PyTorch CUDA：13.0
- AOtools：1.0.7
- NumPy：1.26.4
- SciPy：1.15.3

激活方式：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neuws
```

VS Code 应选择解释器：

```text
/home/xiangwan/miniconda3/envs/neuws/bin/python
```

## 3. 项目与版本管理状态

- 项目位置：`/home/xiangwan/program/NeuWS`
- 分支：`main`
- 本地仓库无远程地址，不会上传原作者代码
- 原始上游基线标签：`upstream-baseline`
- 旧 Windows 项目参考位置：`/mnt/e/mlp_code`，对应 Windows 的 `E:\mlp_code`

关键本地提交：

| 提交 | 内容 |
| --- | --- |
| `c56b9bb` | 原始上游基线，标签 `upstream-baseline` |
| `245790c` | AOtools 光学核心与数据生成器 |
| `fde5dc7` | 通用懒加载、静态训练和输出修复 |
| `59ac6f3` | 图像/相位评估、测试、文档和忽略规则 |
| `d2cc537` | 单张图片指定像差展示程序、完整方形照明默认值 |
| `5236fbc` | VS Code 硬编码直接运行入口 |

## 4. 关键技术决策

### AOtools 与 Zernike

- 不逐行迁移旧 MATLAB 的 `zernfun.m` 和 `zernidx2nm.m`。
- 直接使用 AOtools 的 Noll 编号和归一化实现。
- SLM 图案使用 Noll 1–15，系数独立高斯分布，默认标准差 `5 rad`。
- NeuWS 未知像差网络继续使用原始的 28 项 AOtools Zernike 输入特征。
- 没有迁移旧实验中“网络降到 15 项并清零低阶项”的修改。

### 全区域照明

- 当前项目整个方形区域都有光。
- 默认 `aperture_height=size`，因此掩膜为全 1，不裁剪、不上下补零。
- `size=256` 时，SLM 相位和相机测量均为 `256×256`。
- 论文的 `144×256` 几何只作为兼容选项保留。

### 两个程序保持独立

- 单图展示程序必须由用户指定像差系数。
- 数据集程序不接收单图展示程序的指定系数。
- 数据集像差由 `seed`、像差模式和标准差独立随机产生，并保存到 `ground_truth.mat`。

## 5. 程序一：单张图片指定像差展示

### VS Code 直接运行入口

打开项目根目录下：

```text
run_single_image.py
```

只修改文件顶部配置区，然后点击 VS Code 右上角 **Run Python File**。

当前示例配置：

```python
INPUT_IMAGE = "/mnt/e/mlp_code/3.tif"
OUTPUT_DIR = "outputs/single_aberration"
SIZE = 256

ZERNIKE_COEFFICIENTS = {
    4: 1.5,     # Noll 4：Defocus
    7: -0.25,   # Noll 7：Coma
}

DEVICE = "cuda"
NOISE_STD = 0.0
```

系数单位是弧度。字典键是 Noll 编号，未填写的 1–28 项自动设为零。

### 命令行入口（保留备用）

```bash
python tools/apply_aberration.py \
  --input-image /mnt/e/mlp_code/3.tif \
  --output-dir outputs/single_aberration \
  --size 256 \
  --coefficient 4=1.5 \
  --coefficient 7=-0.25 \
  --device cuda
```

### 输出内容

```text
outputs/single_aberration/
├── comparison.png
├── input_image.png
├── aberrated_image.png
├── aberrated_image.npy
├── aberration_phase_wrapped.png
├── aberration_phase.npy
├── aberration_field.npy
├── zernike_coefficients.npy
├── psf.npy
├── aberration_result.mat
└── manifest.json
```

其中：

- `comparison.png`：输入图、像差相位、PSF、加像差结果的四联展示图。
- `aberrated_image.png`：16 位加像差图片。
- `aberration_result.mat`：保存输入图、结果图、28 项系数、相位、复场、振幅和 PSF。
- `.npy` 文件保存精确浮点或复数数值，避免展示图片量化造成信息损失。

## 6. 程序二：NeuWS 数据集生成

### VS Code 直接运行入口

打开项目根目录下：

```text
run_dataset.py
```

修改顶部配置区后点击 **Run Python File**。

主要配置：

```python
INPUT_IMAGE = "/mnt/e/mlp_code/3.tif"
OUTPUT_DIR = "data/vscode_dataset"

SIZE = 256
NUM_FRAMES = 100
SLM_SIGMA = 5.0

ABERRATION_MODE = "zernike"
ABERRATION_SIGMA = 1.0

NOISE_STD = 0.0
SEED = 0
BATCH_SIZE = 8
DEVICE = "cuda"

APERTURE_HEIGHT = None
```

`APERTURE_HEIGHT=None` 表示完整方形区域都有光。

像差模式：

- `zernike`：随机生成 28 项 Zernike 系数，适合验证。
- `complex-gaussian`：生成孔径内复圆高斯场，适合论文式散射测试。

数据集保持：

- 静态物体；
- 静态未知像差；
- 逐帧改变 SLM 相位；
- 相同 `seed` 得到完全相同的系数、图案、像差和测量。

### 数据集输出

```text
data/vscode_dataset/
├── SLM_sim1.mat ...       # proj_sim，SLM 相位
├── SLM_raw1.mat ...       # imsdata，相机测量
├── slm_patterns.npy
├── slm_coefficients.npy
├── ground_truth.mat
├── manifest.json
└── slm_png/
```

`ground_truth.mat` 中保存真实物体、随机未知像差场、振幅、相位及随机系数。

## 7. 加载、重建与评估

### 数据加载器

- 从 `imsdata` 自动推断偶数方形尺寸。
- 从 `proj_sim` 自动推断有效高度。
- 新数据优先读取 `manifest.json` 中的相位符号。
- 没有 manifest 的论文数据默认使用 `exp(-1j * proj_sim)`。
- 严格检查编号连续性、变量名、尺寸、有限值和归一化范围。
- 按批次从磁盘读取，不再把任意大数据一次性全部放入 GPU。

### 静态重建

入口：`recon_exp_data.py`

项目使用时必须显式保持静态像差：

```bash
python recon_exp_data.py \
  --static_phase \
  --data_dir data/vscode_dataset \
  --scene_name vscode_dataset \
  --num_epochs 1000 \
  --phs_layers 4
```

训练结果保存在：

```text
vis/vscode_dataset/final/
```

包括 `final_I_est.mat`、`final_aberration.mat`、显示图片和 `training_summary.json`。

### 评估

- 图像评估：原始 PSNR/SSIM，可选平移配准后的 PSNR/SSIM、偏移和有效重叠区域。
- 相位评估：孔径内包裹相位误差；主指标只消除 Piston、Tip、Tilt。
- Defocus/Coma 剥离仅作为诊断结果，不进入主指标。
- 入口：`tools/evaluate_neuws.py`。

## 8. 今日验证结果

- 完整 CPU 自动测试：14 项通过，1 项大尺寸 CUDA 测试在普通测试中按设计跳过。
- `1000×1000` CUDA 单批次前向、反向验收：通过。
- RTX 5070 Ti 上 `256×256` 单图指定像差程序：通过。
- 单图输出 PSF 总能量：`1.0`。
- RTX 5070 Ti 上完整方形数据集生成：通过。
- 数据集默认 SLM 相位与测量尺寸：均为完整方形。
- 未开启 `save_per_frame` 时静态训练完整结束：通过。
- 单图程序具有指定系数入口；数据集程序确认不包含该入口。
- VS Code 根目录运行脚本的默认配置：已实际执行验证。

测试命令：

```bash
python -m unittest discover -s tests -v
```

大尺寸 CUDA 测试：

```bash
NEUWS_RUN_LARGE_CUDA=1 python -m unittest \
  tests.test_optics_and_evaluation.LargeCudaTests -v
```

## 9. 重要文件索引

| 文件 | 作用 |
| --- | --- |
| `run_single_image.py` | VS Code 单图指定像差直接运行入口 |
| `run_dataset.py` | VS Code 数据集生成直接运行入口 |
| `tools/apply_aberration.py` | 单图指定像差命令行核心 |
| `tools/generate_neuws_data.py` | SLM 图案与 NeuWS 数据集生成器 |
| `optics.py` | AOtools Zernike、复场、PSF 和卷积共享核心 |
| `dataset.py` | 严格、懒加载 NeuWS 数据集 |
| `recon_exp_data.py` | 静态/动态重建训练入口 |
| `tools/evaluate_neuws.py` | 图像与相位评估入口 |
| `README.md` | 当前完整使用说明 |
| `MIGRATION.md` | MATLAB 到 Python 的功能对应与迁移边界 |

## 10. 后续继续工作时的建议顺序

1. 先在 `run_single_image.py` 中修改 `ZERNIKE_COEFFICIENTS`，确认展示效果和系数约定符合预期。
2. 在 `run_dataset.py` 中确定正式目标图、帧数、随机种子和像差模式。
3. 生成一个较小数据集进行训练烟雾测试。
4. 确认损失下降和最终输出正常后，再生成正式 100 帧或更多帧数据。
5. 使用图像和相位评估工具记录正式实验指标。

本记录对应 2026-07-28 的工作状态。继续开发前先查看本文件、`README.md` 和最新 Git 提交历史。
