# PAWavefrontLab

PAWavefrontLab 是围绕 NeuWS 构建的光声波前研究仓库，覆盖光学仿真、真实 PA
采集预处理、NeuWS 重建、像差分析、SLM 校正与实验验证。仓库保留 NeuWS 上游
网络和数据契约，并在其上增加可复现的科研 workflow 与测试。

上游来源、扩展边界和再分发约束见 [NOTICE.md](NOTICE.md) 与原始
[LICENSE.txt](LICENSE.txt)；本地 Git tag `upstream-baseline` 标记上游基线。

## 从哪里开始

| 目标 | 正式入口 |
| --- | --- |
| 完整静态仿真 | 按顺序运行 `workflows/static_simulation/step1_prepare_ground_truth.py` 到 `step5_evaluate.py` |
| 真实 PA 主流程 | 从 `tools/generate_dual_grid_slm.py` 开始，随后按 [PROJECT_ENTRYPOINTS.md](PROJECT_ENTRYPOINTS.md) 操作 |
| 单张图添加指定 Zernike 像差 | 编辑并运行 `run_single_image.py`（VS Code 演示 wrapper） |
| 单个 BIN 测试电机串扰去噪 | `motor_crosstalk_denoise/run_denoise.py` |
| 查看历史实验、失败记录和构想 | [docs/experiments/README.md](docs/experiments/README.md) |

具体任务该运行哪个程序、输入输出是什么，统一查阅
[PROJECT_ENTRYPOINTS.md](PROJECT_ENTRYPOINTS.md)。系统来源、数据流和模块边界见
[docs/architecture.md](docs/architecture.md)。静态仿真的参数与逐步教程见
[workflows/static_simulation/README.md](workflows/static_simulation/README.md)。

> `run_dataset.py` 是保留给旧用法的一步式数据生成 wrapper，不包含五步流程中的
> 重建和评价，已经 deprecated。新实验不要从它开始。

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
