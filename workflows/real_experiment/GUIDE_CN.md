# 真实实验一键流程：中文操作说明

本文档用于 `workflows/real_experiment/run_pipeline.py`。目标是把一次真实 PA 实验从原始 BIN 一直跑到可加载到 SLM 的校正图，并尽量避免误用上一批实验留下的旧文件。

## 1. 每次新实验先改哪里

只需要编辑本机配置：

```text
configs/real_experiment.json
```

最重要的 5 个字段：

```json
{
  "scene_name": "YYYY-MM-DD_reference_guided",
  "source_dir": "/home/USER/data/PAWavefrontLab/raw/YYYY-MM-DD",
  "origin_source": "/home/USER/data/PAWavefrontLab/raw/YYYY-MM-DD/origin_xxx_600_600_512_PA1.bin",
  "phase_dir": "/home/USER/program/PAWavefrontLab/data/slm50_noll4_28_decay4x_600_1080/model_600/mat",
  "dataset_output_dir": "/home/USER/data/PAWavefrontLab/processed/YYYY-MM-DD_reference_guided"
}
```

含义：

- `scene_name`：本次实验名称，同时决定 `vis/<scene_name>/` 的结果目录。
- `source_dir`：本次实验 `s01...sNN` 的 PA1 BIN 所在文件夹。
- `origin_source`：本次实验 origin PA1 BIN 的完整路径。
- `phase_dir`：与本次采集一一对应的 `SLM_sim1.mat ... SLM_simN.mat` 所在目录。这里不是 SLM 硬件直接加载的 PNG 目录。
- `dataset_output_dir`：reference-guided 去相关后生成 NeuWS 数据集的位置。不同实验建议使用不同目录。

## 2. 正常的新实验怎么跑

先检查命令和路径：

```bash
python workflows/real_experiment/run_pipeline.py --dry-run
```

确认无误后正式运行，并保存终端日志：

```bash
python workflows/real_experiment/run_pipeline.py 2>&1 | tee log/pipeline_YYYY-MM-DD.log
```

流程顺序固定为：

```text
01_preprocess
    原始 BIN -> reference-guided 去相关 -> NeuWS 数据集

02_train
    NeuWS 训练 -> final_aberration.mat

03_export_slm
    立即导出 SLM_final_correction_1080.png

04_zernike_analysis
    最后做 Zernike 拟合和分析
```

动物实验现场最关键的是第 03 步。看到终端打印：

```text
SLM 校正相位已生成，可以立即加载到 SLM：
.../SLM_final_correction_1080.png
```

即可先加载 SLM，不需要等待 Zernike 分析。

## 3. 结果都在哪里

### reference-guided 预处理数据集

```text
<dataset_output_dir>/
```

主要包含：

```text
manifest.json
SLM_raw1.mat ... SLM_rawN.mat
SLM_sim1.mat ... SLM_simN.mat
pipeline_preprocess_fingerprint.json
```

其中 `pipeline_preprocess_fingerprint.json` 是新的安全指纹文件，不要手工复制到另一批实验目录冒充旧数据。

### NeuWS 最终恢复结果

```text
vis/<scene_name>/final/
```

重点：

```text
final_aberration.mat
training_summary.json
```

### 真正加载到 SLM 的校正图

```text
vis/<scene_name>/slm_correction/SLM_final_correction_1080.png
```

该文件是 grayscale `uint8`、范围 0–255 的硬件灰度命令图，使用通用线性循环相位
编码；`device_lut_applied=false` 表示没有应用设备专用 LUT。相同目录中的 preview PNG
只用于人工查看循环相位，不可与硬件灰度图混用。

### Zernike 分析

```text
vis/<scene_name>/zernike_fit/
```

### pipeline 状态

```text
vis/<scene_name>/pipeline_state.json
```

## 4. `--resume` 现在如何保证不会误用旧 BIN 结果

新的 `--resume` 不再只检查 `manifest.json`、`SLM_raw1.mat` 等几个文件是否存在。

预处理阶段会记录并比较：

- `source_dir`；
- origin BIN 的绝对路径、文件大小、纳秒级修改时间；
- 所有 `s01...sNN` BIN 的绝对路径、文件大小、纳秒级修改时间；
- `phase_dir`；
- 所有对应 `SLM_simN.mat` 的绝对路径、文件大小、纳秒级修改时间；
- teacher template；
- reference-guided 预处理参数；
- `scene_name` 和 `dataset_output_dir`。

只有“输入指纹一致”并且 `SLM_raw1...N`、`SLM_sim1...N`、`manifest.json` 全部存在时，才会显示：

```text
[01_preprocess] 输入指纹一致且数据集完整，--resume 安全跳过。
```

如果换了 BIN、换了 origin、换了 SLM MAT、覆盖了原文件或者改了关键预处理参数，会直接拒绝 resume，而不是静默使用旧结果。

## 5. 中断后继续

正常情况下直接：

```bash
python workflows/real_experiment/run_pipeline.py --resume 2>&1 | tee -a log/pipeline_YYYY-MM-DD.log
```

不要为了让 `--resume` 通过而手工复制别的实验的 `pipeline_preprocess_fingerprint.json`。

## 6. 加入指纹机制之前的旧数据怎么办

旧预处理数据如果目录完整但缺少输入指纹文件，pipeline 会拒绝直接 resume。只有在人工
确认 raw BIN、origin、SLM MAT 与 processed 数据确实一一对应后，才可**仅首次**执行：

```bash
python workflows/real_experiment/run_pipeline.py --resume --adopt-existing-preprocess --dry-run
```

确认路径正确后：

```bash
python workflows/real_experiment/run_pipeline.py --resume --adopt-existing-preprocess
```

这不会重新处理 BIN，只会在当前配置的：

```text
<dataset_output_dir>/
```

写入：

```text
pipeline_preprocess_fingerprint.json
```

登记完成后再运行只需要：

```bash
python workflows/real_experiment/run_pipeline.py --resume
```

**`--adopt-existing-preprocess` 只用于已经人工确认来源的旧数据。新实验不需要，也不要随便使用。**

## 7. 如果提示指纹不一致怎么办

不要直接强行跳过。

最常见原因是：

- `configs/real_experiment.json` 仍指向上一批实验；
- 新的 BIN 放进了旧的 `source_dir`；
- origin 被替换；
- `SLM_simN.mat` 换了一套；
- 仍在使用上一批实验的 `dataset_output_dir`；
- 预处理参数发生变化。

新实验推荐同时更换：

```text
scene_name
dataset_output_dir
source_dir
origin_source
phase_dir
```

确认确实要重跑“同一批实验”时，再人工清理旧的 processed 输出目录后重新运行，不要让 pipeline 自动覆盖来源不明的数据。

## 8. 当前推荐训练与分析参数

配置模板使用：

```text
batch_size = 16
num_epochs = 1000
phs_layers = 4
zernike_features = 28
early_stop_warmup = 100
early_stop_window = 20
early_stop_patience = 60
early_stop_min_delta = 5e-6
normalization = shared-max
static_phase = true
device = cuda
zernike.num_modes = 28
zernike.hardware_size = 1080
```

early stopping 的 `best_smoothed_loss`、`best_epoch` 和最终恢复状态始终对应真正历史最优；
`patience_reference_loss` 独立负责 `min_delta` 与 patience 计数。无论提前停止还是跑满
epoch，训练结束都会恢复真正 best state。
