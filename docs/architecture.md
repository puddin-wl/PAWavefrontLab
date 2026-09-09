# PAWavefrontLab 架构与数据流

本文回答“整个系统为什么这样工作”。具体运行入口见
[`PROJECT_ENTRYPOINTS.md`](../PROJECT_ENTRYPOINTS.md)，历史实验见
[`experiments/README.md`](experiments/README.md)。

## 1. 项目来源与边界

PAWavefrontLab 由 NeuWS 上游仓库扩展而来：

```text
NeuWS upstream
  ├─ 神经重建网络与隐式表示
  ├─ SLM/测量 MAT 数据契约
  └─ 静态与动态散射重建方法
        ↓
PAWavefrontLab extensions
  ├─ Python/AOtools 光学与 Zernike 闭环仿真
  ├─ packed-12 PA 解码、MIP 和电机串扰去噪
  ├─ 真实点扫描数据集构建与质量控制
  ├─ 重建评价、Zernike 拟合和 SLM 校正导出
  └─ 分阶段 workflow、实验记录与回归测试
```

`networks.py`、`dataset.py`、`utils.py` 和 `recon_exp_data.py` 以 NeuWS 模型与
数据约定为核心，并包含为当前环境与数据规模所做的兼容扩展。`optics.py`、
`evaluation.py`、`preprocessing/`、`tools/` 和 `workflows/` 主要是
PAWavefrontLab 新增能力。精确代码边界以 `git diff upstream-baseline` 为准。

项目保留原始 `LICENSE.txt`、`NOTICE.md` 和 `upstream-baseline` tag。目录整理不
改变 NeuWS 数学模型、网络行为、Fourier/phase/Zernike 约定或上游归属。

## 2. 真实实验数据流

```text
SLM phase generation
  → PA acquisition
  → packed-12 decode
  → motor-crosstalk denoising
  → baseline subtraction
  → depth MIP
  → NeuWS dataset
  → reconstruction
  → object / aberration recovery
  → Zernike fitting
  → SLM correction
  → experimental validation
```

1. `tools/generate_dual_grid_slm.py` 生成网络计算网格和硬件网格的已知 SLM 相位。
   输入是网格、Noll 模式、强度与随机种子；输出包含 `SLM_simN.mat` 和硬件相位图。
2. 实验设备加载相位并采集原始 PA BIN。原始 BIN 始终只读且不进入 Git。
3. `tools/prepare_decorrelated_real_dataset.py` 流式读取 packed unsigned 12-bit
   A-line。它调用 `preprocessing/pa_denoising.py` 或
   `preprocessing/pa_denoising_gpu.py`，先在完整深度波形上做模板 NCC 和幅值拟合，
   再扣除被接受的电机串扰。不能先做 MIP，因为投影会丢失模板匹配所需的深度信息。
4. 去噪波形减去实验零点、负值截零并沿 depth 取最大值，得到二维 PA MIP。批处理
   工具将 MIP 写为 `SLM_rawN.mat:imsdata`，复制对应
   `SLM_simN.mat:proj_sim`，并产生 `measurements.npy`、manifest 与质量报告。
5. `recon_exp_data.py` 通过 `dataset.py` 读取配对数据。网络只看到测量与已知 SLM
   调制，不读取真实系统像差；`networks.py` 恢复非负物体和未知复场/相位。
6. 有 ORIGIN 时，`tools/evaluate_real_reconstruction.py` 用 `evaluation.py` 计算
   同坐标和配准后的指标。没有 ORIGIN 时只整理结果，不制造无参考 PSNR/SSIM。
7. `tools/fit_recovered_zernike.py` 将恢复相位拟合为明确 Noll 约定的 Zernike
   系数；`tools/export_slm_correction.py` 按现有相位符号导出硬件校正相位。
8. 重新加载校正相位并采集后，`tools/evaluate_optical_restoration.py` 比较真实光学
   校正结果。这一步才验证恢复相位能否在实验系统中产生物理改善。

不需要串扰去噪时，`tools/prepare_real_point_scan_dataset.py` 直接执行权威
packed-12 解码、baseline subtraction 与 MIP，再构建相同 NeuWS 数据契约。

## 3. 静态仿真数据流

```text
ground truth
  → system aberration
  → SLM modulation
  → simulated measurement
  → NeuWS reconstruction
  → evaluation
```

`workflows/static_simulation/` 把流程拆成五个可检查、可恢复的阶段：

1. Step 1 读取清晰图片，规范化得到 ground-truth object，并用固定配置和 seed 生成
   一份静态 system aberration、pupil field、PSF 与未调制基线测量。
2. Step 2 生成逐帧变化且对网络已知的 SLM modulation；系统像差在所有帧保持不变。
3. Step 3 把清晰物体通过“系统像差 + 当帧 SLM 相位”的组合 pupil 直接前向传播，
   生成 simulated measurement。它不会在已模糊图上再次卷积。真实三维 TIFF 可通过
   专用 Step 3 导入，保持后续数据契约不变。
4. Step 4 只向 NeuWS 提供测量和 SLM 相位，由 `recon_exp_data.py` 恢复物体与像差。
5. Step 5 用仿真 ground truth 评价恢复物体和 aperture 内的 wrapped phase error。

每一步读取前一步保存的产物，避免在后续阶段隐式重新抽样实验参数。配置中的随机
种子、相位符号、光学 convention 与验证过的训练参数不因目录整理而改变。

## 4. 模块职责与上下游关系

### `networks.py`

NeuWS 网络核心。输入为已知 SLM 复场、时间坐标/帧索引和网络配置；内部联合表示
物体与未知散射/像差场，并通过前向成像产生模拟测量；输出包含模拟测量、PSF、
恢复复场/相位和物体估计。上游是 `dataset.py` 提供的数据，下游是
`recon_exp_data.py` 的优化、保存和评价工具。第一轮保持其结构与 import 不动。

### `dataset.py`

NeuWS 数据契约适配层。输入为连续编号的 `SLM_simN.mat:proj_sim` 和
`SLM_rawN.mat:imsdata`；验证编号、形状、数值、相位符号与归一化，并逐批返回 SLM
复场、测量和索引。上游数据来自仿真或真实数据准备工具，下游是 PyTorch
`DataLoader` 和网络重建。

### `optics.py`

共享光学原语，包括 aperture、AOtools Noll Zernike basis、相位到复场、PSF 与
卷积前向模型。输入为图像、相位/系数和几何；输出为 pupil、PSF 和模拟测量。
它被仿真 workflow、SLM 工具、网络一致性测试和 phase 评价共同使用，因此 Fourier
与 phase sign convention 必须集中保持稳定。

### `evaluation.py`

共享评价算法。输入为真值与估计图像/复场；输出 raw/translation-registered
PSNR、SSIM、位移，以及去除指定低阶 Zernike 后的 wrapped phase error。它不训练
网络，也不改变估计结果，只为 `tools/evaluate_*.py` 和 Step 5 提供度量。

### `preprocessing/`

正式 PA 预处理算法层：

- `packed12.py`：权威 little-endian unsigned-12 解码；TIFF、普通 MIP、CPU/GPU
  串扰流程共同调用。
- `photoacoustic.py`：baseline subtraction、截零、指定轴 MIP 和 TIFF volume 读取。
- `pa_denoising.py`：CPU 参考实现及模板匹配/投影算法。
- `pa_denoising_gpu.py`：与 CPU 数学行为一致的 CuPy 加速实现。

输入是原始字节、PA volume 或模板，输出是二维投影与诊断数组。这里不负责实验
目录命名、复制 SLM 文件或画最终报告；这些属于 tools/workflows。

### `workflows/`

完整科研流程的分阶段编排。输入是统一 settings 与前一阶段产物；输出是可检查的
数据集、重建和评价目录。workflow 可以调用多个核心模块和工具，但不复制底层算法。

### `tools/`

单功能 command。它们负责 CLI、路径验证、输出保护、manifest、质量报告和格式转换，
调用核心算法完成一个清晰步骤。真实主流程由多个 tool 串联；工具不应含私人机器
默认路径。

### `tests/`

行为契约。覆盖光学/评价、闭环、静态五步、PA 预处理、packed BIN/TIFF、真实数据
准备、电机串扰 CPU 与 CPU/GPU 一致性。测试使用小型合成数据；CUDA/CuPy 不可用时
GPU 测试 skip，而不是使 CPU CI 失败。

## 5. 研究记录、探索代码与生成物

`log/` 的名称是历史遗留，它保存实验研究记录而非运行日志。第一轮为避免破坏大量
历史交叉引用而保留原位，通过 `docs/experiments/` 提供文档入口。失败实验和未实施
构想与成功实验同样保留。

`noise_cause_review/` 是从噪声假设到正式模板方法的探索与溯源层；其中脚本不是正式
入口。未来可在单独、可审查的迁移中整体移到 `archive/noise_cause_review/`，但必须
保留研究证据并修复相对 import 和文档链接。

`data/`、`outputs/`、`vis/` 和大体积 BIN/MAT/TIFF/NPY 是本地输入/生成物，不进入
Git。`motor_crosstalk_denoise/ppt_assets/` 当前 9 张图是已有汇报的 canonical
figures，来源与生成方式由同目录文档记录；普通新运行结果继续写入被忽略的目录。
