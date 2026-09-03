# PAWavefrontLab 项目入口速查

> 目的：减少 Codex 每次重新理解整个仓库的 token 消耗。
>
> 默认规则：**先读本文件确定入口，只读取对应入口脚本、直接依赖和相关测试；除非定位错误或修改底层算法，否则不要全仓库扫描。**

## 0. 固定环境

项目根目录：

```text
/home/xiangwan/program/PAWavefrontLab
```

Conda 环境：

```text
neuws
```

推荐运行：

```bash
conda run -n neuws python <入口脚本> [参数]
```

当前 GPU 环境：

- NVIDIA GeForce RTX 5070 Ti
- PyTorch CUDA 可用
- CuPy 13.6.0 可用
- NumPy 1.26.4
- `motor_crosstalk_denoise/run_denoise.py` 已支持 `--backend cuda|cpu`
- `tools/prepare_decorrelated_real_dataset.py` 已支持 `--backend cuda|cpu`，默认 CUDA
- CUDA 去噪核心：`preprocessing/pa_denoising_gpu.py`
- CPU 参考实现：`preprocessing/pa_denoising.py`

---

# 1. 真实实验主流程

```text
tools/generate_dual_grid_slm.py
    ↓
[真实实验采集 PA BIN]
    ↓
tools/prepare_decorrelated_real_dataset.py
    ↓
recon_exp_data.py
    ↓
tools/evaluate_real_reconstruction.py
    ↓
tools/fit_recovered_zernike.py
    ↓
tools/export_slm_correction.py
    ↓
[加载校正相位并重新采集]
    ↓
tools/evaluate_optical_restoration.py
```

含义：

```text
生成 SLM 相位
→ 采集
→ 电机串扰去噪 + MIP + NeuWS 数据集整理
→ NeuWS 重建
→ 与 ORIGIN 评价
→ 分析恢复系统像差
→ 生成 SLM 校正相位
→ 实际光学校正
→ 最终验证
```

---

# 2. 日常入口总表

| 任务 | 直接运行入口 |
|---|---|
| 单张图添加指定 Zernike 像差 | `run_single_image.py` |
| 完整静态仿真 | `workflows/static_simulation/step1_prepare_ground_truth.py` → `step5_evaluate.py` |
| 生成真实实验双网格 SLM | `tools/generate_dual_grid_slm.py` |
| 原始 BIN 构建普通 NeuWS 数据集，不去串扰 | `tools/prepare_real_point_scan_dataset.py` |
| 单个 BIN 测试电机串扰去噪 | `motor_crosstalk_denoise/run_denoise.py` |
| **整批真实数据去串扰并构建 NeuWS 数据集** | **`tools/prepare_decorrelated_real_dataset.py`** |
| NeuWS 网络重建 | `recon_exp_data.py` |
| 有 ORIGIN 时评价真实重建 | `tools/evaluate_real_reconstruction.py` |
| 无 ORIGIN 时整理重建结果 | `tools/finalize_samples_only_reconstruction.py` |
| 拟合恢复相位的 Zernike 系数 | `tools/fit_recovered_zernike.py` |
| 导出 SLM 校正相位 | `tools/export_slm_correction.py` |
| 校正后真实采集的最终评价 | `tools/evaluate_optical_restoration.py` |
| 单纯 BIN → TIFF/MIP | `savetif/bin_to_tiff.py` |
| PPT 素材整理 | `tools/export_ppt_assets.py` |

---

# 3. 电机串扰去噪

## 3.1 单文件测试

入口：

```text
motor_crosstalk_denoise/run_denoise.py
```

CUDA：

```bash
conda run -n neuws python motor_crosstalk_denoise/run_denoise.py --backend cuda
```

调用链：

```text
run_denoise.py
    ↓
preprocessing/pa_denoising_gpu.py
```

处理顺序：

```text
packed-12 PA BIN
→ 解码完整 A-line
→ 每条 A-line 中位数基线
→ 模板 NCC
→ 最小二乘幅度拟合
→ 满足阈值时扣除模板
→ cleaned A-line
→ max(cleaned - 2048, 0)
→ 沿 depth 做 MIP
→ 二维去噪投影
```

**必须先去噪再 MIP。** 一旦先投影，完整 A-line 深度波形消失，不能再做模板匹配。

## 3.2 整批真实数据

正式入口：

```text
tools/prepare_decorrelated_real_dataset.py
```

它应一次完成：

```text
origin + S1-S50 / D1-D50
→ 逐文件去串扰
→ 全深度 MIP
→ measurements.npy
→ SLM_rawN.mat
→ 复制 SLM_simN.mat
→ quality_control
→ manifest.json
```

不要手工循环 `run_denoise.py` 50 次。

正式推荐结构：

```text
tools/prepare_decorrelated_real_dataset.py
        ↓
--backend cuda
        ↓
preprocessing/pa_denoising_gpu.py
```

同时保留：

```text
--backend cpu
        ↓
preprocessing/pa_denoising.py
```

---

# 4. NeuWS 数据整理

不做电机串扰去噪时：

```text
tools/prepare_real_point_scan_dataset.py
```

处理：

```text
packed-12 BIN
→ decode
→ max(raw - 2048, 0)
→ depth MIP
→ measurements.npy
→ SLM_rawN.mat
→ SLM_simN.mat
→ manifest.json
→ quality_control
```

适用于 600×600×512、900×900×512 等尺寸，并支持 `s`、`d` 等帧名前缀。

---

# 5. NeuWS 网络重建

统一入口：

```text
recon_exp_data.py
```

无论数据来自：

- 仿真
- 600×600 真实数据
- 900×900 真实数据
- 去噪数据
- 未去噪数据

只要已整理成 NeuWS 格式，都从这里进入。

调用链：

```text
recon_exp_data.py
→ dataset.py
→ networks.py
→ StaticDiffuseNet / MovingDiffuse
→ vis/<scene_name>/final/
```

常用参数：

```text
--data_dir
--scene_name
--num_epochs
--batch_size
--static_phase
--phs_layers
--zernike_features
--normalization
--device
--seed
```

---

# 6. 重建后评价

有 ORIGIN：

```text
tools/evaluate_real_reconstruction.py
```

输出主要包括：

```text
reconstruction_report.json
reconstruction_comparison.png
reconstructed_object_normalized.*
reconstructed_aberration_phase.*
```

没有 ORIGIN：

```text
tools/finalize_samples_only_reconstruction.py
```

无参考时不要强行计算 PSNR/SSIM。

---

# 7. 系统像差分析与 SLM 校正

Zernike 拟合：

```text
tools/fit_recovered_zernike.py
```

SLM 校正相位导出：

```text
tools/export_slm_correction.py
```

相位约定：

```text
estimated system field = exp(+i φ)
SLM field              = exp(-i Γ)

Γ ≈ φ 时可抵消非 Piston/Tip/Tilt 系统像差。
```

真实加载校正相位重新采集后：

```text
tools/evaluate_optical_restoration.py
```

这一步才是对恢复相位物理有效性的最终验证。

---

# 8. 仿真流程

配置统一在：

```text
workflows/static_simulation/config.py
```

依次运行：

```text
step1_prepare_ground_truth.py
step2_generate_slm_patterns.py
step3_simulate_measurements.py
step4_reconstruct.py
step5_evaluate.py
```

真实三维 PA TIFF 替代仿真 Step 3 时使用：

```text
step3_import_photoacoustic_measurements.py
```

底层实现：

```text
workflows/static_simulation/workflow.py
```

不要直接运行 `workflow.py`。

---

# 9. 不直接运行的底层模块

一般只在修改算法时阅读：

```text
optics.py
networks.py
dataset.py
evaluation.py
image_utils.py
utils.py
preprocessing/photoacoustic.py
preprocessing/pa_denoising.py
preprocessing/pa_denoising_gpu.py
workflows/static_simulation/workflow.py
```

理解：

```text
入口脚本 = 方向盘
底层模块 = 发动机
```

---

# 10. 历史研究目录

```text
noise_cause_review/
```

主要用于噪声成因、PD 排查、A-line 分析、固定窗和模板 NCC 方法溯源。

正式日常去噪：

```text
motor_crosstalk_denoise/
```

处理新数据时不要从 `noise_cause_review/scripts/` 开始。

---

# 11. Codex 最小读取规则

以后默认要求 Codex：

1. 先读 `PROJECT_ENTRYPOINTS.md`。
2. 根据任务确定唯一入口。
3. 只读取当前入口、直接依赖、相关测试。
4. 不默认扫描整个 `log/`、`tests/`、`noise_cause_review/` 或所有工具。
5. 如果只是换数据运行，不修改代码。
6. 如果只是运行某个入口，不重新设计项目架构。
7. 修改 GPU 去噪时保留 CPU 参考实现和现有输出格式。
8. GPU 修改必须保持 CPU/GPU 数值结果一致。
9. 原始 PA BIN 永远只读。
10. 新实验使用新的输出目录/scene name，避免覆盖历史结果。

---

# 12. 给 Codex 的短指令模板

## 单个 BIN 去噪

```text
先读 PROJECT_ENTRYPOINTS.md。
这次只处理单个 PA BIN 的电机串扰去噪。
入口固定为 motor_crosstalk_denoise/run_denoise.py，
使用 --backend cuda。
不要扫描整个仓库，不要修改无关文件。
完成后只汇报耗时、输出目录、matched fraction、NCC 统计和异常情况。
```

## 批量真实数据去噪并构建 NeuWS 数据集

```text
先读 PROJECT_ENTRYPOINTS.md。
任务是批量真实 PA 数据去串扰并构建 NeuWS 数据集。
入口固定为 tools/prepare_decorrelated_real_dataset.py。
优先使用 CUDA 后端。
只检查这个入口及直接依赖，不要全仓库扫描。
原始 BIN 只读，不覆盖已有结果。
完成后汇报数据形状、shared max、matched fraction、质量报告路径和耗时。
```

## NeuWS 重建

```text
先读 PROJECT_ENTRYPOINTS.md。
数据已经准备完成，不需要重新检查预处理算法。
入口固定为 recon_exp_data.py。
只检查数据目录 manifest 和本次网络参数，然后开始训练。
不要扫描历史 log 或其他实验脚本。
```

## 重建评价

```text
先读 PROJECT_ENTRYPOINTS.md。
重建已经完成。
有 ORIGIN 时入口固定为 tools/evaluate_real_reconstruction.py；
没有 ORIGIN 时使用 tools/finalize_samples_only_reconstruction.py。
不要重新训练，不要修改预处理。
```

---

# 13. CuPy 批处理正式接入检查

当前状态（2026-09-03）：单文件及正式批处理入口均已接入 CUDA，并用真实
`900×900×512` PA BIN 完成 CPU/GPU 一致性与完整耗时验证。以下项目作为后续
修改时的回归清单：

1. `tools/prepare_decorrelated_real_dataset.py` 支持 `--backend cuda|cpu`。
2. CUDA 模式真正调用 `preprocessing/pa_denoising_gpu.py`，不是仍走 CPU。
3. 同一输入下 CPU/GPU 的：
   - projection
   - original projection
   - correlation map
   - center map
   - coefficient map
   - matched map
   在合理浮点误差内一致。
4. 至少验证一个真实 `900×900×512` 文件或相应真实 chunk。
5. 批量 50 帧时避免把完整 cleaned 3-D 数据回传 CPU，只回传 MIP 和必要诊断量。
6. 模板和固定量应复用，避免在每个 chunk 重复无必要计算。
7. `requirements-cuda.txt` 能在干净环境中复现安装。
8. 修改后重新运行 GPU 专项测试和全项目 pytest。

完成这些后，真实 PA 主流程即可默认使用 CUDA。
