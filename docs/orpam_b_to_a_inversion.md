# OR-PAM 模型 B→A 失配反演实验

本实验只进行 **B→A**：用物理前向模型 B 离线生成 50 帧观测，再用原 NeuWS 模型 A 的 `object_mmes` 流程联合恢复固定像差和物体。没有运行 A→A、B→B、A→B，也没有把模型 B 接入训练前向。

## 结论摘要

模型 A 可以很好地拟合模型 B 生成的观测，但没有准确恢复生成观测时使用的固定像差：50 帧最终评估 MSE 为 `4.7508e-5`，而去 piston 后的相位 RMSE 为 `0.2474 rad`，是真值 RMS (`0.2977 rad`) 的 `83.1%`，直接相关系数只有 `0.5712`。因此本次实验出现了“观测 Loss 很低，但固定像差明显错误”的情况。

恢复物体仍与代理物体高度相关（`0.9375`），但直接尺度 RMSE 为 `0.03282`，归一化 L2 差异为 `0.2847`。模型 A 实际参与前向计算的有符号复瞳振幅并非常数，其绝对值空间变异系数为 `23.4%`。这说明物体和振幅自由度都提供了补偿通道；当前单次联合反演能够确认补偿现象存在，但不能把残差定量拆分为“振幅补偿”和“物体补偿”各自的比例。

## 数据与物理设置

输入 TIFF 的存储形状为 `1×1000×1000`、单通道 `uint16`，数值范围为 `[0, 39620]`。程序先去除单例通道，再严格使用 `image[200:800, 200:800]` 得到 `600×600` 裁剪。没有去噪、平滑、锐化或二值化；唯一的强度处理是转换为 `float32` 后除以裁剪最大值 `39620`。原始 TIFF 和裁剪图均未提交到 Git。

模型 B 使用：

- 波长 `532 nm`，焦距 `50 mm`；
- `1080×1080` SLM，像元间距 `6.4 µm`，方形均匀照明；
- `NFFT=2160`，焦平面采样间距 `1.92419 µm`；
- 扫描步长 `5 µm`，`12` 阶 Gauss–Legendre 区域积分；
- 等效核半径 `32` 个扫描像素，`F.conv2d` 互相关且不翻转核；
- 真实实验对应的 50 张 `1080×1080` 硬件相位；模型 A 输入的是同一组随机系数生成并随库保存的 `600×600` 相位；
- 瞳函数相位约定为 `exp(+i Phi_true) exp(-i Gamma_i)`。

固定真值为

```text
Phi_true = 0.45*(x^2+y^2) + 0.35*(x^2-y^2) + 0.30*x*(x^2+y^2)
x,y = (index-(N-1)/2)/(N/2)
```

在 `600×600` 方形瞳面上，真值去 piston 后 RMS 为 `0.297714 rad`。50 个已知相位文件按编号逐一复制到 NeuWS 数据目录，复制内容与相位库源文件逐字节一致；`Phi_true` 只进入模型 B 的未知系统像差项，没有混入模型 A 的已知相位输入。

模型 B 在 CPU 上以 `torch.no_grad()` 逐帧运行一次，耗时 `38.87 s`。50 帧保留原相对强度，不做逐帧 min-max；模型 A 数据加载时仅除以全局最大值 `0.8524581`。等效核保留能量均值为 `0.995201`，范围为 `[0.994035, 0.996072]`，相应截断误差均值为 `0.004799`。

`NFFT=2160` 是本轮已验证的第一阶段配置。既有模型 B 验证中代表帧从 2160 到 4320 的图像差异约为 `0.0033–0.0097`；因此本结果包含该数值采样误差，不能视为 4320 极限结果。

## 反演设置

训练直接调用 `workflows/reconstruction/reconstruct_neuws.py`，前向仍是未经替换的模型 A。物体使用 `object_mmes`，秩为 `(512, 16, 512)`；固定像差使用原 NeuWS 的 4 层相位 MLP 和 28 个 Zernike 特征，没有使用 `dual_mmes`。批量大小为 16，随机种子为 0，静态相位、`shared-max` 归一化，最多 400 轮。早停参数为 warmup 100、window 20、patience 60、min-delta `5e-6`。

本次到第 400 轮仍有缓慢改善，因此未提前停止；恢复并保存的最佳平滑 Loss 状态也在第 400 轮。CUDA 训练耗时 `419.3 s`，设备为 NVIDIA GeForce RTX 5070 Ti。模型 A 的优化目标保持原流程：观测 MSE 加 `object_mmes` 自编码器项。

## 定量结果

### 观测拟合

| 指标 | 数值 |
|---|---:|
| 首轮训练观测 MSE | `1.4119e-2` |
| 最后一轮训练观测 MSE | `4.0004e-5` |
| 训练过程中最小观测 MSE | `3.9756e-5` |
| 最终全 50 帧评估 MSE | `4.7508e-5` |
| 最终总 Loss（观测 MSE + object AE） | `6.3614e-5` |
| 逐帧 MSE 中位数 | `4.1983e-5` |
| 逐帧 MSE 最小值（第 12 帧） | `2.5112e-5` |
| 逐帧 MSE 最大值（第 19 帧） | `1.0422e-4` |
| 逐帧相对 L2 均值 | `0.05410` |

训练日志按 batch 求均值，而最终评估由全部像素和 50 帧重新统计，所以最后一轮训练值与最终评估值并不要求完全相同。

### 固定像差

比较使用相同的 `600×600` 归一化方形瞳坐标和相同的 `exp(+i Phi)` 正号。只分别减去整个方形瞳面的均值作为 piston；没有去 tilt、没有平移、旋转、符号选择或其他配准。

| 指标 | 数值 |
|---|---:|
| 相位 RMSE | `0.247382 rad` |
| 相位 MAE | `0.191477 rad` |
| 相位相关系数 | `0.571217` |
| RMSE / 真值 RMS | `0.830937` |
| 估计相位 RMS | `0.208585 rad` |
| 真值相位 RMS | `0.297714 rad` |
| 28 项 Zernike 系数 RMSE | `0.106518 rad` |
| 28 项 Zernike 系数相关系数（去 piston 项） | `0.542509` |

相位残差小于 `pi`，所以 wrapped 与未 wrapped RMSE 相同。主要 Zernike 错差包括 Noll 14 的虚假 `0.2753 rad`、Noll 4 高估 `0.2312 rad`、Noll 2 低估 `-0.2165 rad`，并非单一 piston 或符号歧义。

### 复瞳振幅和物体

这里从 `StaticDiffuseNet.g_g` 直接提取实际参与模型 A 前向的有符号振幅，而不是使用旧导出接口中的单位振幅占位值。

| 指标 | 数值 |
|---|---:|
| 有符号振幅最小 / 最大 | `-0.01242 / 0.18259` |
| 有符号振幅均值 / 标准差 | `0.12911 / 0.03032` |
| `abs(amplitude)` 空间变异系数 | `0.23437` |
| 负振幅像素比例 | `0.001244` |
| 物体直接尺度 RMSE / MAE | `0.03282 / 0.01481` |
| 物体相关系数 | `0.93747` |
| 物体归一化 L2 差异 | `0.28474` |
| 物体负值比例 | `0.003253` |
| 仅形状拟合的最佳增益 | `0.97518` |

物体主指标不拟合任意增益：只撤销 NeuWS 加载器的 shared-max 标量，使恢复物体回到输入代理物体的 `[0,1]` 尺度。最佳增益只作为附加形状诊断。

## 结果文件

提交到 Git 的非敏感数值摘要：

- [summary.json](data/orpam_b_to_a_inversion/summary.json)
- [frame_residuals.csv](data/orpam_b_to_a_inversion/frame_residuals.csv)
- [zernike_coefficients.csv](data/orpam_b_to_a_inversion/zernike_coefficients.csv)

含血管结构或大数组的结果只保存在本地，没有提交：

```text
outputs/orpam_b_to_a/dataset_nfft2160/
outputs/orpam_b_to_a/analysis/phase_and_amplitude.png
outputs/orpam_b_to_a/analysis/object_comparison.png
outputs/orpam_b_to_a/analysis/loss_and_zernike.png
outputs/orpam_b_to_a/analysis/observation_fits.png
outputs/orpam_b_to_a/analysis/recovered_arrays.npz
vis/orpam_b_to_a_inversion/final/model_final.pt
vis/orpam_b_to_a_inversion/final/training_summary.json
```

## 复现命令

在项目根目录并激活 `neuws` 环境后：

```bash
python tools/orpam_b_to_a_experiment.py prepare \
  --vessel-image /path/to/1.tif \
  --output-dir outputs/orpam_b_to_a/dataset_nfft2160 \
  --fft-size 2160 --kernel-radius 32 --quadrature-order 12 --device cpu

python workflows/reconstruction/reconstruct_neuws.py \
  --root_dir . --data_dir outputs/orpam_b_to_a/dataset_nfft2160 \
  --scene_name orpam_b_to_a_inversion --num_epochs 400 --num_t 50 \
  --batch_size 16 --vis_freq 0 --log_freq 10 \
  --early_stop_warmup 100 --early_stop_window 20 \
  --early_stop_patience 60 --early_stop_min_delta 5e-6 \
  --static_phase --normalization shared-max --phs_layers 4 \
  --zernike_features 28 --device cuda --model_mode object_mmes \
  --object_mmes_rank1 512 --object_mmes_rank2 16 \
  --object_mmes_rank3 512 --object_mmes_seed 0 --seed 0 --silence_tqdm

python tools/orpam_b_to_a_experiment.py analyze \
  --data-dir outputs/orpam_b_to_a/dataset_nfft2160 \
  --reconstruction-dir vis/orpam_b_to_a_inversion \
  --output-dir outputs/orpam_b_to_a/analysis --device cuda
```

## 尚未确认的问题

1. 本轮只有一组初始化和一组联合优化超参数，不能判断失配偏差对随机初始化的敏感性。
2. 模型 A 同时允许物体、相位和复瞳振幅变化；在不增加受控约束实验的前提下，不能唯一分离各补偿通道的贡献。
3. 本轮按任务要求使用 `NFFT=2160`，没有重新运行 50 帧的 4320 数据生成与反演。
4. 真值多项式定义在方形瞳面上；Zernike 表仅是使用项目现有圆域 Noll 基底的诊断投影，不是生成真值的参数化定义。

下一阶段若要提高像差可辨识性，应优先在保持模型 B 物理定义不变的条件下限制或正则化模型 A 的振幅自由度，并用固定已知物体或分阶段优化区分物体补偿；这些建议不属于本次只做 B→A 的执行范围。

## 自动化测试

- `pytest -q tests/test_orpam_model_b.py tests/test_orpam_b_to_a_experiment.py`：`16 passed`。
- `pytest -q`：`143 passed, 2 skipped`（GPU 可用环境）。

警告来自 AOtools/NumPy、scikit-image 和 CuPy 的既有弃用或实验接口提示，本次没有新增测试失败。
