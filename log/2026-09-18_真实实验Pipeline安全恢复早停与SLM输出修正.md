# 2026-09-18 真实实验 Pipeline 安全恢复、早停与 SLM 输出修正

## 结论与实验状态

本次完成的是 2026-09-18 真实实验代码链路的整理和修正：建立统一 pipeline，
增加预处理输入指纹，修正 NeuWS early stopping，统一硬件 SLM PNG 编码，并将
Zernike 分析扩展到 Noll 1–28。

**这不表示 2026-09-18 这一轮真实实验已经成功完成。** 该批数据此前曾被错误命名，
并按 2026-09-16 的路径和流程处理；同时存在相位/配准问题。由错误输入关系得到的旧
09/18 处理和重建结果不能作为有效实验结果，后续必须在确认采集相位、文件配对和方向
约定后重新实验。

代码修正最终提交为：

```text
181b6bc feat: finalize safe real-experiment pipeline and SLM outputs
```

已推送到：

```text
origin/agent/pytest-test-log
```

## 真实数据来源与配对关系

本轮真实数据的正确来源为：

- 原始目录：`/mnt/c/neuws_data/raw/2026-09-18`；
- ORIGIN：`origin_20260918-031909_1_600_600_512_PA1.bin`；
- 测量：连续的 `s01`–`s50`；
- 模型相位：`SLM_sim1.mat`–`SLM_sim50.mat`；
- 正确配对：`s01 ↔ SLM_sim1`，依次到 `s50 ↔ SLM_sim50`；
- acquisition pattern：Noll 4–28，Noll 1–3 固定为零。

此前把这批数据绑定到 09-16 命名、目录或旧 processed 输出的做法是错误的。不能因为
目录中已经存在 `manifest.json`、`SLM_raw1.mat` 或网络结果，就把它们认定为当前
09/18 数据的有效产物。

## 统一真实实验 Pipeline

新增统一入口 `workflows/real_experiment/run_pipeline.py`，固定顺序为：

```text
01 preprocess
   原始 BIN + SLM_simN.mat -> reference-guided NeuWS 数据集

02 train/reconstruction
   NeuWS 训练 -> final_aberration.mat + training_summary.json

03 export SLM correction
   立即导出可供硬件使用的 full-phase SLM 校正文件

04 Zernike analysis
   对恢复相位做 Noll 1–28 拟合和分析
```

第 03 步先于 Zernike 分析执行，目的是训练完成后尽快得到硬件校正图。四个阶段仍由
独立程序完成；pipeline 只负责参数、顺序、状态和恢复逻辑。

## Preprocess safe resume

旧的 `--resume` 只看少量输出是否存在，可能把上一批实验的数据错误复用到当前实验。
新实现为 preprocess 增加 `pipeline_preprocess_fingerprint.json`。指纹记录并比较：

- raw/source 目录；
- origin BIN 的绝对路径、大小和纳秒级修改时间；
- 全部 measurement BIN 的编号、绝对路径、大小和纳秒级修改时间；
- SLM phase 目录及 `SLM_sim1...N.mat` 的文件身份；
- teacher template；
- backend、chunk rows、grid size、seed、相关阈值和最低拟合峰值等关键预处理参数；
- `scene_name` 与 `dataset_output_dir`。

上述规范化 payload 计算 SHA-256 digest。只有指纹一致，并且完整的
`manifest.json`、`SLM_raw1...N.mat`、`SLM_sim1...N.mat` 全部存在时，
`--resume` 才能安全跳过 preprocess。输入变化或旧目录不完整时直接拒绝继续。

没有指纹的历史数据不会被自动认领。`--adopt-existing-preprocess` 只能与
`--resume` 联用，并且只允许用于人工确认过 raw/origin/SLM 配对的旧数据；来源不明的
processed 目录禁止使用该选项。

## Early stopping 修正

旧实现把“真正最优状态”和“达到 `min_delta` 的 patience 参考状态”混在一个变量中，
可能导致训练结束恢复的并非全部历史 smoothed loss 中的最低状态。新逻辑明确分离：

1. `best_smoothed_loss`、`best_epoch` 和 best network state 表示 true historical best；
2. 任意更低的 smoothed loss 都立即更新 best state，不受 `min_delta` 限制；
3. loss window 一旦完整，就开始跟踪 true best；
4. warmup 只控制 `patience_reference_loss` 何时开始计数，不阻止 best 跟踪；
5. 无论触发 early stop 还是跑满 `num_epochs`，训练结束都恢复 true best state。

`patience_reference_loss` 单独处理 `min_delta` 和连续无显著改善 epoch 计数。当前建议：

```text
num_epochs = 1000
early_stop_warmup = 100
early_stop_window = 20
early_stop_patience = 60
early_stop_min_delta = 5e-6
```

## Zernike 模式范围

- NeuWS 网络 `zernike_features=28` 表示特征为 Noll 1–28；
- 本轮 acquisition pattern 的实际调制项为 Noll 4–28；
- `tools/fit_recovered_zernike.py` 的默认拟合由 10 项改为 Noll 1–28；
- pipeline 的 Zernike analysis 同样使用 `num_modes=28`；
- full-phase SLM export 和 Zernike hardware candidate 均去除 Noll 1–3；
- Piston 没有校正意义，Tilt 与图像平移存在歧义，因此不直接下发到硬件。

网络输入特征范围、采集调制模式范围和最终硬件候选范围是三个不同概念，记录和配置中
不能再把它们都简称为“10-mode”或“低阶相位”。

## SLM PNG 与连续相位数据

真正加载到 SLM 的硬件 PNG 已统一为：

- grayscale；
- `uint8`；
- 数值范围 0–255；
- 将 `[0, 2π)` 循环相位按现有 `rint(... × 255 ...)` 公式线性编码；
- `device_lut_applied=false`，即没有应用设备专用 LUT。

受此约束的文件包括 full-phase `SLM_final_correction_1080.png`、Zernike fitted
hardware candidate，以及双网格生成器的 `SLM_hwN.png`。

`SLM_final_correction_preview.png` 是给人查看的循环相位预览，不是硬件灰度命令，
不能加载到 SLM。model-side MAT 和相位数据继续保存连续弧度值；双网格生成器的
model MAT/PNG 高精度表示也没有因硬件 uint8 输出而降级。

## 分径向阶 Zernike 采样

`tools/generate_dual_grid_slm.py` 新增按径向阶设置 Gaussian sigma 和单项 absolute
limit 的功能，用于 09-17 记录中的 Noll 4–28 分阶衰减设计。相同径向阶的 Noll 模式
共享一组 sigma/limit，禁用项随后置零；固定 seed 时结果可复现。

同时增加参数一致性保护：单独提供
`--coefficient-limit-by-radial-order` 而没有
`--sigma-by-radial-order` 时直接报错，不再静默忽略。对应测试覆盖分阶限制、随机结果
复现、hardware uint8、model uint16/连续 MAT 和 manifest 范围。

## 验证结果

本次只验证代码和轻量输出，没有重新处理真实 BIN，也没有进行耗时 NeuWS 训练。

- `python -m pytest -q tests/test_dual_grid_slm.py`：`7 passed`；
- `py_compile`：相关入口和工具全部通过；
- pipeline config：四阶段命令及 28-mode/early-stop 参数检查通过；
- safe resume：无指纹拒绝、指纹不一致拒绝、完全匹配接受；
- uint8 output：full-phase export 和 Zernike candidate 均验证为 grayscale uint8；
- `git diff --check` 与 `git diff --cached --check`：通过；
- AOtools/NumPy/SciPy 的 DeprecationWarning 属第三方兼容提示，本次未修改第三方代码。

## 后续真实实验要求

下一轮实验必须从正确的 2026-09-18 raw/origin 和经过确认的 Noll 4–28 相位配对重新
开始，使用新的 `scene_name` 和 `dataset_output_dir`。正式运行前先做 pipeline
`--dry-run`，确认 `s01...s50 ↔ SLM_sim1...SLM_sim50`、硬件 PNG 灰度编码、方向和
设备设置。旧的错误 09/18 processed/vis 输出只保留作问题溯源，不得写成有效实验结果。
