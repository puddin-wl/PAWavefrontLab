# Motor Crosstalk Denoise

这是 PA 图像中电机串扰噪点的正式工具目录。它把原先分散在
`noise_cause_review/` 中的程序、固定模板、代表性结果和 PPT 素材集中到项目
根目录下，便于直接找到和复用。

## 两种使用方式

### 1. 单独处理一份 PA 数据

打开 `run_denoise.py`，修改文件顶部的 `INPUT_BIN` 和 `OUTPUT_DIR`，然后在
VS Code 中点击 **Run Python File**。也可以使用命令行：

```bash
python \
  motor_crosstalk_denoise/run_denoise.py \
  --input /path/to/600_600_512_PA1.bin \
  --output-dir motor_crosstalk_denoise/runs/my_result
```

输入必须是包含每个像素完整 A-line 的 packed unsigned 12-bit PA BIN，默认形状
为 `600×600×512`。模板匹配需要深度曲线，已经完成投影的二维 PNG/TIFF 无法再
使用这套方法恢复性去噪。

每次运行会输出：

- 扣除前全深度 MIP；
- 去噪后的 float32 MIP；
- 最大绝对 NCC、匹配中心、拟合系数和扣除掩膜；
- 去噪前后对照图和诊断图；
- `metrics.json` 定量指标。

输出目录必须为空，程序不会覆盖已有结果，也不会修改原始 BIN。

### 2. 新实验的自适应 NeuWS 批处理（推荐）

新实验不应直接复用这里的跨日期固定模板。使用：

```bash
python tools/prepare_adaptive_real_dataset.py \
  --template-data-dir data/OLD_DATASET \
  --source-dir /path/to/S1-S50 \
  --origin-source /path/to/origin_PA1.bin \
  --output-dir data/NEW_ADAPTIVE_DATASET \
  --scene-name NEW_ADAPTIVE_DATASET
```

该入口会从本次实验的测量 A-line 自动确定信号窗、左右噪声窗、重复瞬态模板和
匹配阈值，再将同一份冻结标定应用到 origin 与全部测量帧。正式投影使用噪声功率
校正的 excess-RMS，不做空间平滑；自动质控失败时不会发布半成品训练数据集。

### 3. 固定模板流程（历史复现）

固定模板算法实现在 `preprocessing/pa_denoising.py`，批量数据集入口为：

```bash
python \
  tools/prepare_decorrelated_real_dataset.py \
  --template-data-dir data/OLD_DATASET \
  --source-dir /path/to/D1-D50 \
  --origin-source /path/to/origin_PA1.bin \
  --output-dir data/NEW_DATASET \
  --scene-name NEW_DATASET
```

批量入口默认调用本目录下的固定模板，无需再输入难找的旧研究路径。它会对
origin 与 D1-D50 执行完全相同的去相关，然后生成 `measurements.npy`、
`SLM_rawN.mat`、质量报告和 shared-max 预览；SLM 相位保持原样复制。

两个入口共用同一套核心实现，不维护两份容易漂移的算法代码。

## 固定方法参数

- 模板：`template/motor_crosstalk_template_v1.csv`；
- 模板长度：361 点，范围 `-180:180`；
- 每条 A-line 的匹配基线：中位数；
- 接受条件：`|NCC| >= 0.70` 且拟合模板峰值 `>= 80 ADC`；
- 幅度：有效非循环重叠范围内的最小二乘系数；
- 每条 A-line 最多扣除一个最强模板；
- 投影：`max(cleaned - 2048, 0)` 后做全 512 点 MIP；
- 不使用 PD，不做时间门控或空间平滑。

## 结果和 PPT 材料

- `ppt_assets/`：已经按汇报顺序编号的 9 张 canonical figures；新增普通运行图
  默认忽略，保留规则见其中的 `README.md`；
- `PPT_NOTES.md`：建议的逐页标题、图和结论；
- `metrics_summary.json`：可直接引用的关键数字；
- `results_archive/`：按“噪点观察、PD 排查、固定窗、模板去相关、50 帧验证”
  分类保存的结果，不包含网络训练或网络矫正输出。

## 适用范围和限制

该模板来自 2026-08-21 的六条代表性电机串扰 A-line，并已在 2026-08-20 的
50 帧数据上复核。更换电机、驱动器、采样率、触发方式或模拟前端后，应先检查
噪声波形与模板 NCC，再决定是否继续使用相同模板和阈值。
