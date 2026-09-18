# Real Experiment Workflow

`run_pipeline.py` 是真实 PA 实验的统一总控入口。它不重复实现算法，而是按顺序启动现有独立程序：

1. `tools/prepare_reference_guided_real_dataset.py`：reference-guided A-line 去相关、excess-RMS 与 NeuWS 建集；
2. `recon_exp_data.py`：NeuWS 网络训练与系统像差恢复；
3. `tools/export_slm_correction.py`：**训练完成后立即**导出可加载的 1080×1080 SLM 校正相位；
4. `tools/fit_recovered_zernike.py`：随后进行 Noll 1–28 Zernike 拟合与分析。

前处理和网络训练使用独立 Python 子进程，因此 CuPy/CUDA 前处理退出后再启动 PyTorch，可避免两套 allocator/context 长时间共存。

## 第一次使用

复制配置模板：

```bash
cp configs/examples/real_experiment_pipeline.json configs/real_experiment.json
```

修改 `configs/real_experiment.json` 中的实验路径和训练参数。该本机配置建议加入 `.gitignore`，不要提交真实数据路径。

先检查命令：

```bash
python workflows/real_experiment/run_pipeline.py --dry-run
```

确认无误后正式运行：

```bash
python workflows/real_experiment/run_pipeline.py
```

如果中途失败或机器重启：

```bash
python workflows/real_experiment/run_pipeline.py --resume
```

`--resume` 对预处理阶段不仅检查输出完整性，还会校验原始 BIN、origin、SLM MAT、
teacher template 和预处理参数的输入指纹。没有指纹的历史数据不会自动认领；只有人工
确认来源后，才可首次联用 `--resume --adopt-existing-preprocess`。其余阶段根据关键输出
跳过已经完成的步骤。

## 最关键的实验输出

训练完成后 pipeline 会优先运行 SLM 导出，并在终端醒目打印：

```text
SLM 校正相位已生成，可以立即加载到 SLM：
.../vis/<scene_name>/slm_correction/SLM_final_correction_1080.png
```

此时即可加载 SLM 并继续动物实验；Zernike 分析在其后运行，不阻塞校正相位的生成。

`SLM_final_correction_1080.png` 是真正给硬件使用的 grayscale `uint8` 文件，数值范围
为 0–255，采用通用线性循环相位编码；manifest 中
`device_lut_applied=false`，表示尚未应用设备专用 LUT。`*_preview.png` 只用于人工查看，
不要当作硬件灰度图。

每次运行状态保存在：

```text
vis/<scene_name>/pipeline_state.json
```
