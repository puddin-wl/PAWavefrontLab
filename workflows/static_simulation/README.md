# PAWavefrontLab 静态像差完整仿真：中文新手教程

这份文档面向第一次接触 NeuWS、AOtools 和本项目代码的人。按顺序操作后，
你应该能够：

1. 给一张清晰图片生成固定系统像差相位图和基准模糊图；
2. 生成 50 张已知 SLM 调制相位；
3. 模拟得到对应的 50 张相机测量图；
4. 用 NeuWS 网络恢复清晰物体和固定系统像差；
5. 查看图像 PSNR/SSIM、相位 RMSE、训练曲线和最终对比图。

如果只想给单张图片添加指定像差，不需要训练网络，请直接看本文的
“任务 A：只生成单张像差相位图和模糊图”。

## 1. 先理解五个名词

| 名词 | 本项目中的含义 | 典型文件 |
| --- | --- | --- |
| 清晰物体 | 仿真的原始清晰图片，也是图像恢复真值 | `reference/clear_object.png` |
| 系统像差相位图 | 整个实验期间固定不变、网络事先不知道的像差 | `reference/system_aberration_phase.png` |
| SLM 调制相位图 | 每帧主动加载到 SLM、网络已知的随机相位 | `slm_png/slm_phase_0001.png` |
| 调制测量图 | 清晰物体经过固定系统像差和对应 SLM 相位后形成的图像 | `measurement_png/modulated_measurement_0001.png` |
| 恢复结果 | 网络估计的清晰物体和固定系统像差 | `reconstructed_object.png` 等 |

这里不使用 `o1`、`i0`、`p1` 等临时简称，避免不同人对编号产生不同理解。

每一帧测量都满足：

```text
清晰物体 + 固定系统像差 + 第 n 张 SLM 相位 -> 第 n 张调制测量
```

基准模糊图是 SLM 相位为零时的结果，只用于展示和比较。程序不会在基准
模糊图上继续加模糊。

## 2. 项目位置和输入图片

项目根目录：

```text
/home/xiangwan/program/PAWavefrontLab
```

当前示例输入图片：

```text
/home/xiangwan/program/PAWavefrontLab/data/test.tif
```

这是 Linux/WSL 路径。不要写成 `U:\home\...` 这样的 Windows 路径。
输入可以是普通灰度/RGB 图片，也可以是三维光声 TIFF。

普通图片会自动：

1. 转换为灰度；
2. 中心裁剪成正方形；
3. 缩放到配置尺寸；
4. 归一化到 `[0,1]`。

常量全黑、全白或包含 NaN/Inf 的图片会被拒绝。

三维光声 TIFF 按旧采集代码的约定自动执行：

```text
原始三维数据 -> max(原始值 - 2048, 0) -> 沿第 0 维最大投影 -> 二维清晰物体
```

这里的“置零”指减去 2048 后所有负值设为 0。投影维度始终是第 0 维，
但第 0 维可以是任意正层数，不再强制要求 512 层。程序先转成浮点数再减，
不会发生 16 位无符号整数下溢。

## 3. 首次运行前检查环境

### 3.1 在 VS Code 选择解释器

按 `Ctrl+Shift+P`，选择 **Python: Select Interpreter**，再选择：

```text
/home/xiangwan/miniconda3/envs/neuws/bin/python
```

### 3.2 在终端激活现有环境（可选）

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neuws
```

### 3.3 检查 PyTorch 和 CUDA

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

当前机器正常输出应包含：

```text
2.13.0+cu130
True
NVIDIA GeForce RTX 5070 Ti
```

如果只想验证 CPU 小流程，可以把配置中的 `generation_device` 和
`training_device` 改成 `"cpu"`；正式 256×256、50 帧、1000 轮训练建议使用
CUDA。

## 4. 任务 A：只生成单张像差相位图和模糊图

这种方式不运行 NeuWS 网络，适合先认识 Zernike 系数和像差效果。

### 4.1 修改配置

打开项目根目录下的：

```text
run_single_image.py
```

主要修改：

```python
INPUT_IMAGE = "/home/xiangwan/program/PAWavefrontLab/data/test.tif"
OUTPUT_DIR = "outputs/my_single_aberration"
SIZE = 256

ZERNIKE_COEFFICIENTS = {
    4: 1.5,     # Defocus，离焦
    7: -0.25,   # 一个方向的 Coma，彗差
}

DEVICE = "cuda"
```

常用 Noll 项：

- 4：Defocus，离焦；
- 5、6：两个方向的 Astigmatism，散光；
- 7、8：两个方向的 Coma，彗差。

系数单位为弧度。没有填写的 Noll 1–28 项自动设为零。

### 4.2 运行

打开 `run_single_image.py`，点击 VS Code 右上角 **Run Python File**。

### 4.3 查看结果

在配置的输出目录中查看：

| 文件 | 含义 |
| --- | --- |
| `comparison.png` | 输入图、系统像差相位、PSF 和模糊图四联图 |
| `aberration_phase_wrapped.png` | 包裹到 `[0,2π)` 的 16 位相位预览 |
| `aberration_phase.npy` | 未量化的精确相位数组，单位 rad |
| `aberration_field.npy` | 精确复数像差场 |
| `psf.npy` | 归一化 PSF |
| `aberrated_image.png/.npy` | 模糊图预览和精确数值 |
| `aberration_result.mat` | 供 MATLAB 使用的汇总文件 |
| `manifest.json` | 系数、符号、单位和运行配置 |

“相位图”最好明确说成“系统像差相位图”。PNG 是显示文件，正式数值计算应
使用 `.npy` 或 `.mat`。

## 5. 任务 B：完整五步静态仿真

所有入口和配置都在：

```text
workflows/static_simulation/
```

### 5.1 第一次必须修改运行名称

打开：

```text
workflows/static_simulation/config.py
```

当前默认运行 `test_static_zernike_50` 已经完成并保存。新实验不要覆盖它，
请至少修改下面三项：

```python
input_image=PROJECT_ROOT / "data" / "test.tif",
data_dir=PROJECT_ROOT / "data" / "my_first_simulation",
scene_name="my_first_simulation",
```

建议让 `data_dir` 最后一段与 `scene_name` 完全相同。这样数据、恢复结果和
评价报告会使用同一个名字，最容易寻找。

### 5.2 配置参数表

| 参数 | 默认值 | 含义 | 新手是否需要修改 |
| --- | --- | --- | --- |
| `input_image` | `data/test.tif` | 清晰输入图片 | 换图片时修改 |
| `data_dir` | `data/test_static_zernike_50` | 相位和测量数据目录 | 新实验必须修改 |
| `scene_name` | `test_static_zernike_50` | `vis/` 和 `outputs/` 下的结果名 | 新实验必须修改 |
| `input_mode` | `auto` | 自动区分普通图片和三维光声 TIFF | 通常不改 |
| `photoacoustic_baseline` | 2048 | 光声交流信号零点 | 采集零点改变时才改 |
| `photoacoustic_projection_axis` | 0 | 固定沿第 0 维最大投影 | 不改 |
| `raw_measurement_dir` | `data/raw_photoacoustic_measurements` | 真实三维 TIFF 测量目录 | 导入真实数据时修改 |
| `size` | 256 | 正方形计算尺寸，必须为正偶数 | 一般不改 |
| `aperture_height` | `None` | `None` 表示整个方形区域都有光 | 一般不改 |
| `num_frames` | 50 | SLM 相位和调制测量数量 | 正式实验一般保持 50 |
| `system_noll_start/end` | 4/15 | 固定系统像差使用的 Noll 范围 | 一般不改 |
| `system_sigma` | 0.6 rad | 固定系统像差强度 | 想改变模糊强度时修改 |
| `system_seed` | 20260730 | 固定系统像差随机种子 | 换一组像差时修改 |
| `slm_num_modes` | 15 | SLM 使用 Noll 1–15 | 一般不改 |
| `slm_sigma` | 5 rad | 随机 SLM 调制强度 | 一般不改 |
| `slm_seed` | 20260731 | SLM 随机种子 | 换一组 SLM 相位时修改 |
| `phase_sign` | -1 | 对应 `exp(-1j*proj_sim)` | 不要随意修改 |
| `noise_std` | 0 | 高斯相机噪声标准差 | 首轮保持 0 |
| `simulation_batch_size` | 8 | 前向仿真批量 | 显存不足时减小 |
| `generation_device` | `cuda` | 步骤一、三使用的设备 | 无 GPU 时改成 `cpu` |
| `training_device` | `cuda` | 网络训练设备 | 正式训练保持 `cuda` |
| `training_epochs` | 1000 | 网络训练轮数 | 快速检查时可暂时减小 |
| `training_batch_size` | 8 | 网络训练批量 | 显存不足时减小 |
| `phase_layers` | 4 | 相位网络隐藏层数 | 一般不改 |
| `visualization_frequency` | 1000 | 过程图保存间隔，0 表示关闭 | 一般不改 |
| `overwrite` | `False` | 防止覆盖已有输出 | 建议保持 `False` |

修改随机种子会得到一组新的、但仍可复现的数据。系统像差只在步骤一抽样
一次，后续步骤读取保存结果，不会偷偷重新抽样。

### 5.3 径向 15 阶完整闭环入口

项目保留上面的 Noll 1–15 默认实验，同时提供独立的径向 15 阶配置。这里的
“径向 15 阶”表示完整使用到 Noll 136，而不是只使用 Noll 15：

```bash
python workflows/static_simulation/run_radial15_simulation.py
```

该入口固定使用 `data/test.tif`、256×256、50 帧和 1000 轮 CUDA 训练。SLM
使用 Noll 1、4–136（Noll 2、3 置零），单项标准差为 1.22 rad；系统像差使用
Noll 4–136，网络空间特征也使用 136 项。运行前会实际执行一次 CUDA 前向、
反向和优化器步进；batch=8 显存不足时依次尝试 4、2、1。所有数据写入新的
`test_static_radial15_both_50` 目录，不会覆盖原来的 `test_static_zernike_50`。

一般实验也可以在 `SimulationSettings` 中分别配置：

- `system_num_modes` 和 `system_disabled_noll_indices`；
- `slm_num_modes` 和 `slm_disabled_noll_indices`；
- `network_zernike_features`。

这些参数默认仍为原来的 28 项系统/网络基底和 15 项 SLM，因此旧入口无需修改。

## 6. 五个步骤怎么运行

必须按顺序运行。每个文件都可以在 VS Code 中打开后点击右上角
**Run Python File**。

### 步骤一：准备清晰物体和固定系统像差

运行：

```text
step1_prepare_ground_truth.py
```

成功提示类似：

```text
步骤一完成：固定系统像差和基准模糊图已保存到 .../reference
```

重点查看：

```text
data/SCENE_NAME/reference/clear_object.png
data/SCENE_NAME/reference/system_aberration_phase.png
data/SCENE_NAME/reference/baseline_aberrated_measurement.png
data/SCENE_NAME/reference/baseline_psf.png
```

步骤一还会保存 `ground_truth.mat` 和精确 `.npy` 数组。

### 步骤二：生成已知 SLM 相位

运行：

```text
step2_generate_slm_patterns.py
```

成功后应有：

```text
data/SCENE_NAME/SLM_sim1.mat ... SLM_sim50.mat
data/SCENE_NAME/slm_patterns.npy
data/SCENE_NAME/slm_coefficients.npy
data/SCENE_NAME/slm_png/slm_phase_0001.png ... 0050.png
```

`SLM_simN.mat` 中的变量名必须是 `proj_sim`。相位 PNG 只是包裹相位预览，
不是某台真实 SLM 已标定的灰度图。

### 步骤三 A：模拟 50 张调制测量

运行：

```text
step3_simulate_measurements.py
```

成功后应有：

```text
data/SCENE_NAME/SLM_raw1.mat ... SLM_raw50.mat
data/SCENE_NAME/measurements.npy
data/SCENE_NAME/measurement_png/modulated_measurement_0001.png ... 0050.png
```

`SLM_rawN.mat` 中的变量名必须是 `imsdata`。每张图都由清晰物体直接生成，
不是在基准模糊图上继续卷积。

### 步骤三 B：导入真实三维光声测量（与三 A 二选一）

如果已经用步骤二的相位完成真实采集，就不要运行仿真版步骤三。先把正好
`num_frames` 个三维 TIFF 放进 `raw_measurement_dir`，再运行：

```text
step3_import_photoacoustic_measurements.py
```

文件会按名称中的数字自然排序，例如 `capture_2.tif` 会排在
`capture_10.tif` 前。第 n 个 TIFF 必须对应 `SLM_simN.mat` 的第 n 张相位。
每帧执行：

```text
corrected = max(raw_volume - 2048, 0)
imsdata = max(corrected, axis=0)
```

重要约定：

- 不要求三维 TIFF 有 512 层，只要求确实是非空三维数组；
- 投影后尺寸必须与 `size×size` 一致，真实测量不会被静默裁剪或缩放；
- `SLM_rawN.mat:imsdata` 保存未单独归一化的二维投影；
- 50 帧只使用一个数据集全局最大值供网络归一化；
- PNG 预览也共享这个最大值，不会逐帧拉伸亮度；
- 这样能保留不同 SLM 帧之间真实的相对强度，网络才能正确使用它们。

### 步骤四：运行 NeuWS 网络重建

运行：

```text
step4_reconstruct.py
```

当前 RTX 5070 Ti 上，256×256、50 帧、1000 轮大约需要 4–5 分钟。训练
过程中可以观察 MSE 是否总体下降。

成功后重点查看：

```text
vis/SCENE_NAME/final/reconstructed_object.png
vis/SCENE_NAME/final/reconstructed_object.npy
vis/SCENE_NAME/final/reconstructed_aberration_phase.png
vis/SCENE_NAME/final/reconstructed_aberration_phase.npy
vis/SCENE_NAME/final/training_summary.json
```

同时保留原 NeuWS 兼容文件 `final_I_est.mat` 和 `final_aberration.mat`。

### 步骤五：生成最终评价

运行：

```text
step5_evaluate.py
```

重点查看：

```text
outputs/SCENE_NAME/evaluation/reconstruction_comparison.png
outputs/SCENE_NAME/evaluation/reconstruction_report.json
outputs/SCENE_NAME/evaluation/training_loss.png
outputs/SCENE_NAME/evaluation/phase_errors.mat
```

`reconstruction_comparison.png` 一次显示：清晰物体、基准模糊图、网络恢复图、
系统像差真值、网络恢复像差以及相位误差。

## 7. 怎样判断仿真是否成功

至少检查以下项目：

- 五个步骤都输出“完成”，没有异常退出；
- `SLM_simN.mat` 和 `SLM_rawN.mat` 数量都等于 `num_frames`；
- `training_summary.json` 中损失总体下降且全部有限；
- 恢复图没有全黑、全白、NaN 或明显错位；
- 恢复图 PSNR/SSIM 高于基准模糊图；
- 相位主指标查看“去除 Piston/Tip/Tilt 后的 RMSE”。

图像指标：

- PSNR 越高越好；
- SSIM 越接近 1 越好；
- registered shift 应接近 `[0,0]`。

相位的全局常数 Piston 不影响非相干 PSF，因此网络无法从测量中唯一确定。
不要只看原始相位 RMSE，应主要看：

```text
primary_piston_tip_tilt_removed.rmse_rad
```

## 8. 输出目录总览

```text
data/SCENE_NAME/
├── reference/                     # 清晰物体、系统像差和基准模糊图
├── slm_png/                       # SLM 相位预览
├── measurement_png/               # 调制测量预览
├── SLM_simN.mat                   # 网络已知相位，变量 proj_sim
├── SLM_rawN.mat                   # 相机测量，变量 imsdata
├── slm_patterns.npy
├── measurements.npy
├── ground_truth.mat
└── manifest.json                  # 整个数据集的参数和完成状态

vis/SCENE_NAME/final/              # 网络恢复结果
outputs/SCENE_NAME/evaluation/     # 指标、误差和最终图
```

`.png` 用于查看，`.npy` 用于 Python 精确计算，`.mat` 用于 MATLAB 或网络
数据交换，`manifest.json` 用于记录本次运行的配置和数据约定。

## 9. 每个主要代码文件是干什么的

### 项目根目录

| 文件 | 作用 | 新手是否需要修改 |
| --- | --- | --- |
| `README.md` | 项目总入口和原始命令行说明 | 阅读，不必修改 |
| `requirements.txt` | Python 依赖 | 新环境安装时使用 |
| `run_single_image.py` | 单张图片指定像差的 VS Code 入口 | 做任务 A 时修改配置区 |
| `run_dataset.py` | 较早的随机数据集 VS Code 入口 | 完整五步仿真不使用 |
| `recon_exp_data.py` | NeuWS 训练和最终结果保存 | 一般不直接修改 |
| `optics.py` | Zernike、孔径、复场、PSF 和卷积物理模型 | 核心代码，不随意修改 |
| `dataset.py` | 加载并严格检查 `SLM_simN/SLM_rawN` | 一般不修改 |
| `networks.py` | 原论文 NeuWS 物体/像差反演网络 | 核心网络，不随意修改 |
| `utils.py` | 网络使用的 FFT 卷积、Zernike 等旧共享函数 | 一般不修改 |
| `evaluation.py` | PSNR、SSIM、配准和相位误差算法 | 一般不修改 |
| `image_utils.py` | 图片读取、归一化和 16 位 PNG 保存 | 一般不修改 |
| `preprocessing/photoacoustic.py` | 三维光声减 2048、负值置零和第 0 维投影 | 一般不修改 |
| `aberration_config.py` | 解析显式 Zernike 系数 | 任务 A 的底层支持 |
| `MIGRATION.md` | MATLAB 到 Python 的迁移边界 | 需要理解迁移时阅读 |

### `workflows/static_simulation/`

| 文件 | 作用 |
| --- | --- |
| `config.py` | 完整仿真的唯一用户配置文件 |
| `workflow.py` | 五步实现、校验、清单、语义化输出和评价汇总 |
| `step1_prepare_ground_truth.py` | 准备清晰物体、固定像差和基准图 |
| `step2_generate_slm_patterns.py` | 生成已知 SLM 相位 |
| `step3_simulate_measurements.py` | 直接生成调制测量 |
| `step3_import_photoacoustic_measurements.py` | 把真实三维光声 TIFF 转成网络测量文件 |
| `step4_reconstruct.py` | 调用 NeuWS 网络恢复 |
| `step5_evaluate.py` | 生成图像/相位指标和总览图 |

### `tools/`

| 文件 | 作用 |
| --- | --- |
| `apply_aberration.py` | 单图像差命令行核心 |
| `generate_neuws_data.py` | 旧的通用 SLM/数据集命令行生成器 |
| `evaluate_neuws.py` | 单独评价已有图像或相位结果 |

### `tests/`、`log/` 和生成目录

- `tests/`：自动测试，不是正式运行入口；
- `log/`：按日期记录环境、开发过程和正式实验结果；
- `data/`：输入图片及生成的数据集；
- `vis/`：网络恢复结果；
- `outputs/`：单图结果和最终评价报告。

## 10. 常见问题

### 10.1 提示输出已经存在

典型信息：

```text
FileExistsError: 步骤一的输出已经存在...
```

原因是 `overwrite=False` 正在保护已有结果。推荐做法：同时修改 `data_dir`
和 `scene_name`，为新实验使用新名称。只有明确需要覆盖时才设置
`overwrite=True`。

### 10.2 提示 CUDA 不可用

先确认 VS Code 选择的是 `neuws` 环境，再运行第三节中的 CUDA 检查命令。
如果 `torch.cuda.is_available()` 为 False，可以先用 CPU 做小尺寸检查，但正式
1000 轮训练不建议使用 CPU。

### 10.3 找不到输入图片

确认 `input_image` 是 WSL/Linux 路径，并且文件真实存在。例如：

```text
PROJECT_ROOT / "data" / "test.tif"
```

不要把 Windows 盘符和 Linux 路径混写。

### 10.4 步骤二、三或四提示前置步骤未完成

必须严格按照 step1 → step2 → step3 → step4 → step5 顺序运行。程序会读取
`manifest.json` 判断前置状态和配置是否一致。

### 10.5 相位 PNG 看起来颜色奇怪或有跳变

相位 PNG 把相位包裹到 `[0,2π)`，在 `0/2π` 边界出现跳变是正常的。需要
精确连续数值时读取 `.npy` 或 `.mat`，不要从 PNG 反推相位。

### 10.6 出现 AOtools 的 `np.math` DeprecationWarning

这是 AOtools 1.0.7 在新 NumPy 上的弃用警告，当前测试中不影响结果。

### 10.7 出现 Matplotlib 缓存目录警告

这只影响首次绘图缓存速度，不影响仿真或数值结果。必要时可把
`MPLCONFIGDIR` 指向有写权限的临时目录。

### 10.8 显存不足

依次尝试减小：

```python
simulation_batch_size=4
training_batch_size=4
```

不要先改变相位符号、物理模型或数据变量名。

### 10.9 三维 TIFF 提示尺寸不匹配

真实测量投影后必须恰好是配置中的 `size×size`。这项检查防止相机图与 SLM
坐标关系被自动缩放破坏。请核对采集尺寸，或在开始步骤一前把 `size` 设置成
真实投影尺寸并重新生成该运行目录。

### 10.10 三维 TIFF 层数不是 512

这是允许的。当前代码不检查固定层数；无论是几十层、512 层还是其他正层数，
都始终沿第 0 维做最大投影。

## 11. 怎样替换成真实实验

真实实验时保留步骤二生成的 SLM 相位，用真实 SLM 和相机替代步骤三：

1. 从 `SLM_simN.mat:proj_sim` 读取精确相位；
2. 使用设备专用 LUT、gamma 和电压标定转换成真实 SLM 灰度；
3. 逐帧显示相位并采集三维 TIFF；
4. 将所有 TIFF 放入 `raw_measurement_dir`，名称顺序必须与 SLM 顺序对应；
5. 运行 `step3_import_photoacoustic_measurements.py`，自动完成减 2048、置零、
   第 0 维投影和 `SLM_rawN.mat:imsdata` 写入；
6. 继续运行步骤四进行恢复。

当前输出的 16 位相位 PNG 不是硬件标定结果，不能直接假设它与某台 SLM
灰度、电压或相位响应线性对应。

真实实验通常没有系统像差真值，因此步骤五的相位真值评价可以省略；如果有
单独采集的清晰参考图，仍可评价恢复图像。

## 12. 当前已验证的正式结果

默认 `test_static_zernike_50` 已实际完成：

- 256×256；
- 50 帧；
- 1000 轮 CUDA 训练；
- RTX 5070 Ti 训练约 253 秒；
- 损失从 `5.26×10⁻²` 降到 `2.50×10⁻⁷`；
- 恢复图像 PSNR：`44.07 dB`；
- 恢复图像 SSIM：`0.9797`；
- 去除 Piston/Tip/Tilt 后相位 RMSE：`0.0242 rad`，约 `1.39°`。

详细过程见：

```text
log/2026-07-30_NeuWS完整静态仿真与CUDA重建.md
```

## 13. 新手完成检查表

- [ ] VS Code 选择了 `neuws` Python 环境；
- [ ] `input_image` 指向真实图片；
- [ ] `data_dir` 和 `scene_name` 使用新的相同名称；
- [ ] 按顺序完成 step1 到 step5；
- [ ] 找到了系统像差相位图和 50 张 SLM 相位图；
- [ ] 找到了 50 张调制测量图；
- [ ] 若使用真实三维光声数据，确认运行的是导入版步骤三且 TIFF 顺序正确；
- [ ] 找到了 `reconstructed_object.png`；
- [ ] 找到了 `reconstructed_aberration_phase.png`；
- [ ] 打开了 `reconstruction_comparison.png`；
- [ ] 阅读了 `reconstruction_report.json` 中的图像和相位指标。
