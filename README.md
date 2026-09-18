# PAWavefrontLab

PAWavefrontLab 是围绕 NeuWS 构建的光声波前研究仓库，覆盖光学仿真、真实 PA
采集预处理、NeuWS 重建、像差分析、SLM 校正与实验验证。仓库保留 NeuWS 上游
网络和数据契约，并在其上增加可复现的科研 workflow 与测试。

上游来源、扩展边界和再分发约束见 [NOTICE.md](NOTICE.md) 与原始
[LICENSE.txt](LICENSE.txt)；本地 Git tag `upstream-baseline` 标记上游基线。

## 从哪里开始

| 目标 | 正式入口 |
| --- | --- |
| **真实 PA 实验：BIN → 去相关建集 → NeuWS 训练 → SLM 校正相位 → Zernike 分析** | **`workflows/real_experiment/run_pipeline.py`** |
| 完整静态仿真 | 按顺序运行 `workflows/static_simulation/step1_prepare_ground_truth.py` 到 `step5_evaluate.py` |
| 单张图添加指定 Zernike 像差 | 编辑并运行 `run_single_image.py`（VS Code 演示 wrapper） |
| 单个 BIN 测试电机串扰去噪 | `motor_crosstalk_denoise/run_denoise.py` |
| 查看历史实验、失败记录和构想 | [docs/experiments/README.md](docs/experiments/README.md) |

真实实验现在推荐使用统一总控入口。它不会把所有算法写进一个巨大程序，而是按顺序
启动已经验证的独立工具；训练完成后会**优先立即导出可加载的 SLM 校正相位**，再继续
做 Zernike 分析，避免动物实验现场额外等待。

第一次使用先复制配置：

```bash
cp configs/examples/real_experiment_pipeline.json configs/real_experiment.json
```

修改本机路径后可先检查：

```bash
python workflows/real_experiment/run_pipeline.py --dry-run
```

正式运行只需：

```bash
python workflows/real_experiment/run_pipeline.py
```

中断后继续：

```bash
python workflows/real_experiment/run_pipeline.py --resume
```

详细说明见 [workflows/real_experiment/README.md](workflows/real_experiment/README.md)。
具体任务该运行哪个程序、输入输出是什么，统一查阅
[PROJECT_ENTRYPOINTS.md](PROJECT_ENTRYPOINTS.md)。系统来源、数据流和模块边界见
[docs/architecture.md](docs/architecture.md)。静态仿真的参数与逐步教程见
[workflows/static_simulation/README.md](workflows/static_simulation/README.md)。

> `run_dataset.py` 是保留给旧用法的一步式数据生成 wrapper，不包含当前真实实验
> reference-guided 主流程、NeuWS 训练和 SLM 导出，已经 deprecated。

## 真实实验主链

统一入口内部依次调用：

```text
raw *.bin + SLM_simN.mat
        ↓
tools/prepare_reference_guided_real_dataset.py
        ↓
NeuWS dataset (SLM_rawN.mat + SLM_simN.mat)
        ↓
recon_exp_data.py
        ↓
vis/<scene>/final/final_aberration.mat
        ↓
tools/export_slm_correction.py
        ↓
SLM_final_correction_1080.png   ← 可立即加载到 SLM
        ↓
tools/fit_recovered_zernike.py
```

硬件实际加载的 `SLM_final_correction_1080.png` 是 grayscale `uint8`、范围 0–255；
preview PNG 只是给人查看的循环相位图，不是硬件灰度命令。Zernike 分析拟合 Noll
1–28，生成硬件 candidate 时仍去除 Noll 1–3。

当前生产验证配置为 reference-guided CUDA 后端、`chunk_rows=600`。具体实验参数仍以
`configs/real_experiment.json` 为准。

## 安装与测试

PAWavefrontLab 已在 Python 3.10 环境验证。PyTorch 请先按机器和 CUDA/CPU 环境
安装，再安装运行依赖；不要为整理仓库而升级已经验证的 NumPy、CuPy、PyTorch
或 AOtools。

```bash
python -m pip install -r requirements.txt
```

开发和测试环境再安装：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

需要 CuPy CUDA 去噪时使用：

```bash
python -m pip install -r requirements-cuda.txt
```

CUDA/CuPy 不可用时，GPU 专项测试会自动跳过；CPU 测试仍应全部通过。

## 数据与输出

原始采集和生成结果不进入 Git：`data/`、`outputs/`、`vis/`、BIN、MAT、TIFF、
NPY 等都由 `.gitignore` 排除。请为每次实验使用新的 `scene_name` 和输出目录，
不要覆盖历史结果，也不要修改原始 PA BIN。

NeuWS 重建所需的基本数据契约是连续编号的配对文件：

```text
SLM_sim1.mat  -> proj_sim
SLM_raw1.mat  -> imsdata
SLM_sim2.mat
SLM_raw2.mat
...
```

完整 MATLAB-to-Python 历史映射见 [docs/migration.md](docs/migration.md)。
