# 2026-07-30 NeuWS 完整静态仿真与 CUDA 重建记录

## 1. 今日目标与结论

今天围绕 `data/test.tif` 建立并实际跑通了一套可复现、可分步执行的 NeuWS
静态系统像差闭环：

1. 保存清晰物体真值，生成一份固定系统像差及平坦 SLM 下的基准模糊图。
2. 独立生成 50 张已知随机 SLM 调制相位。
3. 直接由清晰物体、固定系统像差和对应 SLM 相位生成 50 张调制测量。
4. 网络只读取调制测量和已知 SLM 相位，恢复清晰物体及固定系统像差。
5. 将恢复结果与物体真值和像差真值做定量、可视化对比。

正式的 `256×256`、50 帧、1000 轮 CUDA 训练已经完成。重建图像达到
`44.07 dB` PSNR 和 `0.9797` SSIM；去除不可辨识的 Piston、Tip、Tilt 后，
系统像差相位 RMSE 为 `0.0242 rad`，约 `1.39°`。

## 2. 明确的物理模型

本流程采用一个静态物体和一个静态系统像差。设：

- `O`：清晰物体 `clear_object`；
- `G0 = exp(+i·p0)`：固定系统像差复场；
- `Sn = exp(-i·pn)`：第 n 张已知 SLM 调制；
- `hn = |FFT(G0·Sn)|² / sum(|FFT(G0·Sn)|²)`：归一化非相干 PSF；
- `In = O * hn`：第 n 张调制测量。

平坦 SLM 时 `pn=0`，所得图像仅作为
`baseline_aberrated_measurement` 展示和比较。50 张训练测量全部从清晰物体
直接生成，绝不把基准模糊图作为新物体再次模糊。相应集成测试会显式比较
这两种计算并拒绝级联模糊实现。

系统像差真值只用于仿真生成和最终评估，不传给网络。网络输入仅为
`SLM_rawN.mat:imsdata` 与对应的 `SLM_simN.mat:proj_sim`。

## 3. CUDA 验证过程

Codex 默认隔离环境中没有 GPU 设备权限，因此最初看到
`torch.cuda.is_available() == False`。在得到明确授权后，使用正常 WSL GPU
权限进行了验证：

- GPU：NVIDIA GeForce RTX 5070 Ti，16303 MiB；
- Windows 驱动：581.80；
- WSL NVIDIA-SMI：580.105.07；
- CUDA Toolkit：13.0，`nvcc 13.0.88`；
- PyTorch：`2.13.0+cu130`；
- PyTorch 编译 CUDA：13.0；
- cuDNN：92000；
- `torch.cuda.is_available()`：True；
- 实际在 `cuda:0` 完成了 `2048×2048` 矩阵乘法，结果全部有限。

这说明之前的 False 是 Codex 沙箱隔离现象，不是本机 CUDA 安装故障。

## 4. 代码结构整理

今天新增的完整流程集中在：

```text
workflows/static_simulation/
├── config.py
├── workflow.py
├── step1_prepare_ground_truth.py
├── step2_generate_slm_patterns.py
├── step3_simulate_measurements.py
├── step4_reconstruct.py
└── step5_evaluate.py
```

- `config.py` 是唯一需要日常编辑的中文配置。
- `workflow.py` 保存五个阶段共用的实现、校验、清单和语义化输出。
- 五个 `step` 文件可直接在 VS Code 中逐个运行。
- 根目录的 `networks.py` 仍是论文上游反演网络核心。
- `utils.py` 中的 Zernike 与 FFT 卷积接口仍被网络调用，没有在本次做无关重写。
- `optics.py`、`dataset.py` 继续承担共享光学模型和严格数据加载。

## 5. 正式仿真参数

- 输入物体：`data/test.tif`，256×256、16 位灰度；
- 图像预处理：归一化到 `[0,1]`；
- 孔径：完整 256×256 方形区域；
- 固定系统像差：Noll 4–15；
- 系统像差分布：`N(0, 0.6²) rad`；
- 系统像差种子：20260730；
- 实际非零项：Noll 4–15，共 12 项；
- 实际系数范围：约 `[-0.798, 0.928] rad`；
- SLM：Noll 1–15，标准差 `5 rad`；
- SLM 种子：20260731；
- 帧数：50；
- 相位符号：`exp(-i·proj_sim)`；
- 相机噪声：0；
- 网络：静态像差、28 项 Zernike 特征、4 层相位网络；
- 训练：1000 轮、batch size 8、学习率 `1e-3`；
- 训练设备：CUDA。

生成结果经过检查：

- `SLM_simN.mat`：50 个；
- `SLM_rawN.mat`：50 个；
- SLM 相位预览：50 张；
- 调制测量预览：50 张；
- SLM 相位数组：`(50, 256, 256)`，全部有限；
- 调制测量数组：`(50, 256, 256)`，范围约 `[0.00226, 0.64155]`；
- 数据加载器正确识别 50 帧、256×256、全孔径和相位符号 `-1`。

## 6. 训练中发现并修正的归一化问题

第一次正式训练本身已收敛，但初次图像评估只有：

- PSNR：`20.87 dB`；
- SSIM：`0.8436`。

检查后确认这不是反演失败，而是单位不一致：数据加载器使用数据集全局
测量最大值 `0.6415498257` 归一化所有相机图像，因此网络物体输出处于
“归一化测量单位”；最初评估却直接把它与 `[0,1]` 物体真值比较。此外，
旧重建代码保存数值前把网络输出截断到 1，丢失了亮区的可逆信息。

修正内容：

1. 重建阶段额外保存未截断、非负的 `final_I_est_network_units.mat`；
2. 语义化输出使用清单中的 `measurement_max` 恢复物体辐射比例；
3. 保存 `reconstructed_object_network_units.npy` 便于审计；
4. `reconstructed_object` 与真值比较前已处于同一数值单位；
5. 训练摘要记录 `measurement_normalization_max`；
6. 新增测试验证 `reconstructed_object = clip(network_units × measurement_max)`。

随后重新完成了 1000 轮正式 CUDA 训练和评估。

## 7. 正式训练与最终指标

第二次正式训练：

- 用时：252.97 秒；
- 初始平均 MSE：`5.2607×10⁻²`；
- 最终平均 MSE：`2.4950×10⁻⁷`；
- 最低平均 MSE：`8.4096×10⁻⁸`；
- 未出现 NaN、CUDA 错误或显存不足。

图像指标：

| 结果 | PSNR | SSIM |
| --- | ---: | ---: |
| 平坦 SLM 基准模糊图 | 25.96 dB | 0.8814 |
| NeuWS 恢复清晰图 | 44.07 dB | 0.9797 |

恢复图像无需平移校正，估计偏移为 `[0,0]`。

相位指标：

| 指标 | RMSE | MAE |
| --- | ---: | ---: |
| 原始包裹相位误差 | 0.7651 rad | 0.7646 rad |
| 去除 Piston/Tip/Tilt | 0.0242 rad | 0.0170 rad |
| 诊断性去除 PTT/Defocus/Coma | 0.0235 rad | 0.0163 rad |

原始相位误差主要来自约 `0.765 rad` 的全局 Piston。非相干 PSF 对全局相位
常数不敏感，因此这部分在物理上不可辨识；主结果应使用去除 Piston、Tip、
Tilt 后的指标，而不是把原始 RMSE 解释为恢复失败。

## 8. 输出位置

正式数据：

```text
data/test_static_zernike_50/
```

包含 `reference/`、50 对 MATLAB 文件、精确 NumPy 数组、SLM/测量预览和
`manifest.json`。

网络恢复：

```text
vis/test_static_zernike_50/final/
```

最终评估：

```text
outputs/test_static_zernike_50/evaluation/
```

关键文件：

- `reconstruction_report.json`：完整定量指标；
- `reconstruction_comparison.png`：清晰物体、基准模糊图、恢复图及相位对比；
- `training_loss.png`：1000 轮损失曲线；
- `phase_errors.mat`：孔径掩膜和校准后的相位误差数组。

## 9. 测试状态

完整测试命令：

```bash
python -m unittest discover -s tests -v
```

当前共运行 17 项测试：16 项通过，1 项可选的 `1000×1000` CUDA 验收测试
按设计跳过。新增测试覆盖：

- 固定系统像差按种子复现；
- 只有 Noll 4–15 非零；
- SLM 与系统像差相位符号正确；
- 每帧由清晰物体直接生成，不使用级联模糊；
- 五步小尺寸 CPU 闭环；
- 网络结果与指标全部有限；
- 物体归一化比例正确还原。

## 10. 今后如何分步运行

第一次接触项目时，先阅读完整中文教程：

```text
workflows/static_simulation/README.md
```

先编辑：

```text
workflows/static_simulation/config.py
```

然后依次在 VS Code 中运行同目录下五个 `step` 文件。默认
`overwrite=False`，重复运行已有步骤会明确报错，避免覆盖正式数据。要进行
新实验，优先修改 `scene_name` 和 `data_dir` 创建新的运行目录。

## 11. 转向真实实验时的接口

真实实验保留步骤二、四、五：

1. 步骤二生成精确 SLM 相位；
2. 使用设备专用 LUT、gamma 和电压标定把相位送到真实 SLM；
3. 相机逐帧采集图像，替代仿真步骤三；
4. 将图像保存为 `SLM_rawN.mat`，变量名仍为 `imsdata`；
5. 对应相位保持 `SLM_simN.mat`，变量名 `proj_sim`；
6. 步骤四可直接加载并重建。

真实实验中系统像差没有真值，因此相位评估部分可以省略；物体真值是否可用
取决于是否另外采集了无像差参考图。当前 16 位相位 PNG 只是通用包裹相位
预览，不等于硬件标定后的 SLM 灰度图。

## 12. 今日结束状态

- 完整仿真、正式 CUDA 重建和评估均已完成；
- 生成数据和恢复结果由 `.gitignore` 排除，不会进入源码提交；
- 源码修改和本日志目前仍在工作区，尚未创建 Git 提交；
- 后续优先任务应是为具体 SLM 建立 LUT，并实现真实相机采集到
  `SLM_rawN.mat:imsdata` 的适配层。
