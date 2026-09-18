# PAWavefrontLab 项目入口

本文件只回答：**为了完成某个任务，应运行哪个程序。** 系统设计见
[`docs/architecture.md`](docs/architecture.md)，实验记录见
[`docs/experiments/README.md`](docs/experiments/README.md)。所有命令均从项目根目录
运行。

## 入口身份

- **正式 workflow**：完成一条端到端科研流程，位于 `workflows/`。
- **正式 command/tool**：完成一个明确步骤，通常位于 `tools/`。
- **direct-run wrapper**：便于 VS Code 点击运行，内部调用正式工具；不是另一套算法。
- **deprecated/历史入口**：仅为复现旧用法保留，不建议新实验采用。

## 真实实验推荐入口

当前真实实验的第一推荐入口是：

```bash
python workflows/real_experiment/run_pipeline.py
```

第一次使用先复制并编辑本机配置：

```bash
cp configs/examples/real_experiment_pipeline.json configs/real_experiment.json
python workflows/real_experiment/run_pipeline.py --dry-run
```

中断恢复使用：

```bash
python workflows/real_experiment/run_pipeline.py --resume
```

总控程序只负责调度，内部步骤仍是独立程序：

| 顺序 | 任务 | 正式入口 | 主要输出 |
| --- | --- | --- | --- |
| 1 | reference-guided 自适应 A-line 去相关并建集 | `tools/prepare_reference_guided_real_dataset.py` | excess-RMS、自动标定/QC、`SLM_rawN.mat` |
| 2 | NeuWS 网络训练与像差恢复 | `recon_exp_data.py` | `vis/<scene>/final/final_aberration.mat`、`training_summary.json` |
| 3 | **立即导出硬件 SLM 校正** | `tools/export_slm_correction.py` | **`SLM_final_correction_1080.png`** / MAT / NPY |
| 4 | Zernike 拟合与分析 | `tools/fit_recovered_zernike.py` | 系数、拟合相位、报告 |
| 5 | 重新采集后的光学验证 | `tools/evaluate_optical_restoration.py` | 最终验证报告（独立后续步骤） |

第 3 步优先于 Zernike 分析：训练一结束就生成可加载的 SLM 相位，避免动物实验现场
等待分析图和报告。`fit_recovered_zernike.py` 产生的 Zernike SLM candidate 仅用于
分析，不替代 `export_slm_correction.py` 的 full-phase correction。

### 数据准备入口不要混用

| 场景 | 入口 |
| --- | --- |
| **新实验：历史 teacher → 当前实验模板 + 自动信号/噪声窗 + excess-RMS + CUDA 建集** | **`tools/prepare_reference_guided_real_dataset.py`（当前推荐）** |
| 早期实验级自适应流程复现 | `tools/prepare_adaptive_real_dataset.py` |
| 复现旧的固定模板去相关 + 全深度 MIP | `tools/prepare_decorrelated_real_dataset.py` |
| 不去串扰的普通 packed-12 BIN + 建集 | `tools/prepare_real_point_scan_dataset.py` |
| 旧的 excess-RMS 方法复现 | `tools/prepare_denoised_real_dataset.py`（兼容/研究入口） |

reference-guided 流程使用历史固定模板作为 teacher，在当前实验高可信事件上自动重标定
实验模板；最终去相关继续使用已验证的 NCC 匹配、LS 系数拟合和非循环模板减法。
生产验证配置为 CUDA、`chunk_rows=600`，整批实验复用 persistent GPU workspace。

## 静态仿真 workflow

配置位于 `workflows/static_simulation/config.py`。依次运行：

```text
step1_prepare_ground_truth.py
step2_generate_slm_patterns.py
step3_simulate_measurements.py
step4_reconstruct.py
step5_evaluate.py
```

若 Step 3 使用采集的三维 PA TIFF，改运行
`step3_import_photoacoustic_measurements.py`。底层
`workflows/static_simulation/workflow.py` 不作为直接入口。

`workflows/static_simulation/run_radial15_simulation.py` 是已验证的径向 15 阶专用
workflow，会要求 CUDA；它不是通用默认入口。

## 单功能入口

| 任务 | 入口 |
| --- | --- |
| reference-guided 整批 BIN 去相关并建集 | `tools/prepare_reference_guided_real_dataset.py` |
| 单图添加指定像差 | `tools/apply_aberration.py` |
| 生成仿真数据/SLM pattern | `tools/generate_neuws_data.py` |
| 单纯 BIN → 多页 TIFF 或 MIP | `savetif/bin_to_tiff.py` |
| 单 BIN 串扰去噪诊断 | `motor_crosstalk_denoise/run_denoise.py` |
| NeuWS 网络训练/重建 | `recon_exp_data.py` |
| full-phase SLM 校正导出 | `tools/export_slm_correction.py` |
| Zernike 拟合 | `tools/fit_recovered_zernike.py` |
| 通用图像/相位评价 | `tools/evaluate_neuws.py` |
| PPT 代表性素材整理 | `tools/export_ppt_assets.py` |

## VS Code direct-run wrapper

配置好 `configs/real_experiment.json` 后，`workflows/real_experiment/run_pipeline.py` 无需
额外参数即可运行，因此也可直接在 VS Code 中点击 Run Python File。

`run_single_image.py` 是受支持的单图演示 wrapper：修改顶部输入图、输出目录和
Zernike 系数后点击运行。其真正实现是 `tools/apply_aberration.py`。

`run_dataset.py` **已 deprecated**。它仍可复现旧的一步式随机数据生成，但不包含
reference-guided 预处理、NeuWS 训练、SLM 校正导出和阶段恢复。

## 不直接运行的模块

```text
networks.py                 NeuWS 网络
dataset.py                  NeuWS MAT 数据加载与归一化
optics.py                   光学与 Zernike 前向模型
evaluation.py               图像/相位评价
preprocessing/              正式 PA 预处理算法
workflows/.../workflow.py   workflow 底层编排
```

`noise_cause_review/` 保存噪声成因、PD、A-line、固定窗和模板 NCC 的研究溯源。
处理新数据不要从其中的脚本开始。

## 分阶段命令示例

如果不使用总控入口，可手动分别运行：

```bash
python tools/prepare_reference_guided_real_dataset.py \
  --source-dir /path/to/raw_experiment \
  --origin-source /path/to/origin_PA1.bin \
  --phase-dir /path/to/SLM_sim_mat \
  --output-dir /path/to/processed_dataset \
  --scene-name new_scene \
  --backend cuda \
  --chunk-rows 600

python recon_exp_data.py \
  --static_phase \
  --data_dir /path/to/processed_dataset \
  --scene_name new_scene \
  --num_epochs 1000 \
  --phs_layers 4

python tools/export_slm_correction.py \
  --input-phase vis/new_scene/final/final_aberration.mat \
  --output-dir vis/new_scene/slm_correction \
  --hardware-size 1080

python tools/fit_recovered_zernike.py \
  --input-phase vis/new_scene/final/final_aberration.mat \
  --output-dir vis/new_scene/zernike_fit \
  --num-modes 28 \
  --hardware-size 1080
```

真实主流程的数据形状、强度归一化、相位符号、Zernike 约定和去噪阈值属于实验
参数。除非正在设计新实验，不要为目录整理而修改它们。
