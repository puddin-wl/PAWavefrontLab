# PAWavefrontLab 项目入口

本文件只回答：**为了完成某个任务，应运行哪个程序。** 系统设计见
[`docs/architecture.md`](docs/architecture.md)，实验记录见
[`docs/experiments/README.md`](docs/experiments/README.md)。所有命令均从项目根目录
运行，路径示例使用 `/path/to/...`，请替换为本机路径。

## 入口身份

- **正式 workflow**：完成一条端到端科研流程，位于 `workflows/`，或由下面列出的
  多个正式工具组成。
- **正式 command/tool**：完成一个明确步骤，通常位于 `tools/`。
- **direct-run wrapper**：便于 VS Code 点击运行，内部调用正式工具；不是另一套算法。
- **deprecated/历史入口**：仅为复现旧用法保留，不建议新实验采用。

## 真实实验主流程

| 顺序 | 任务 | 正式入口 | 主要输出 |
| --- | --- | --- | --- |
| 1 | 生成真实实验双网格 SLM | `tools/generate_dual_grid_slm.py` | `SLM_simN.mat` 与硬件相位图 |
| 2 | 采集 PA 数据 | 实验设备 | 只读原始 PA BIN |
| 3 | 去串扰并构建 NeuWS 数据集 | `tools/prepare_decorrelated_real_dataset.py` | MIP、`SLM_rawN.mat`、manifest、质量报告 |
| 4 | NeuWS 重建 | `recon_exp_data.py` | `vis/<scene>/final/` |
| 5 | 有 ORIGIN 的重建评价 | `tools/evaluate_real_reconstruction.py` | 指标与对照图 |
| 6 | 拟合恢复像差 | `tools/fit_recovered_zernike.py` | Zernike 系数与拟合报告 |
| 7 | 导出 SLM 校正 | `tools/export_slm_correction.py` | 校正相位 |
| 8 | 重新采集后的光学验证 | `tools/evaluate_optical_restoration.py` | 最终验证报告 |

没有 ORIGIN 时，第 5 步改用 `tools/finalize_samples_only_reconstruction.py`，不要
强行计算 PSNR/SSIM。

### 数据准备的三个入口不要混用

| 场景 | 入口 |
| --- | --- |
| 正式整批去串扰 + 建集 | `tools/prepare_decorrelated_real_dataset.py` |
| 不去串扰的普通 packed-12 BIN + 建集 | `tools/prepare_real_point_scan_dataset.py` |
| 旧的 excess-RMS 方法复现 | `tools/prepare_denoised_real_dataset.py`（兼容/研究入口） |

正式批处理支持 `--backend cuda|cpu`。CUDA 后端调用
`preprocessing/pa_denoising_gpu.py`，CPU 参考实现位于
`preprocessing/pa_denoising.py`；两者输出必须保持一致。

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
| 单图添加指定像差 | `tools/apply_aberration.py` |
| 生成仿真数据/SLM pattern | `tools/generate_neuws_data.py` |
| 单纯 BIN → 多页 TIFF 或 MIP | `savetif/bin_to_tiff.py` |
| 单 BIN 串扰去噪诊断 | `motor_crosstalk_denoise/run_denoise.py` |
| 通用图像/相位评价 | `tools/evaluate_neuws.py` |
| PPT 代表性素材整理 | `tools/export_ppt_assets.py` |

## VS Code direct-run wrapper

`run_single_image.py` 是受支持的演示 wrapper：修改顶部输入图、输出目录和 Zernike
系数后点击运行。其真正实现是 `tools/apply_aberration.py`。

`motor_crosstalk_denoise/run_denoise.py` 同时提供 CLI 和 direct-run 配置。仓库默认
只放 placeholder 路径；本机真实数据路径不要提交。

`run_dataset.py` **已 deprecated**。它仍可复现旧的一步式随机数据生成，但不包含
五步 static workflow 的分阶段保护、NeuWS 重建和评价。新实验使用上面的五步流程。

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
处理新数据不要从其中的脚本开始；正式算法和入口分别位于 `preprocessing/` 与
`motor_crosstalk_denoise/run_denoise.py`。

## 常用命令示例

```bash
# 重建已整理的数据集
python recon_exp_data.py \
  --static_phase \
  --data_dir /path/to/neuws_dataset \
  --scene_name new_scene \
  --num_epochs 1000 \
  --phs_layers 4

# 单个 BIN 去串扰（所有路径均为示例）
python motor_crosstalk_denoise/run_denoise.py \
  --input /path/to/capture_PA1.bin \
  --output-dir outputs/denoise_example \
  --height 600 --width 600 --depth 512 \
  --backend cpu

# 全部测试；CUDA 不可用时 GPU 测试自动跳过
python -m pytest -q
```

真实主流程的数据形状、强度归一化、相位符号、Zernike 约定和去噪阈值属于实验
参数。除非正在设计新实验，不要为目录整理而修改它们。
