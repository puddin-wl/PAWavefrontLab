# Noise Cause Review

> 正式、易找的去噪入口已经整理到项目根目录
> [`motor_crosstalk_denoise/`](../motor_crosstalk_denoise/)。本目录保留为原始研究
> 记录和溯源材料，日常处理新数据请使用新目录中的 `run_denoise.py`。

这是后续噪点成因研究的独立工作目录。相关分析脚本、参数、记录和新结果统一放在这里。

原始采集数据保留在原位置，只通过路径读取，避免复制大体积数据或意外修改原始数据。此前已有的分析结果已复制到本目录作为研究起点。

## Contents

- `scripts/`：后续噪点检测、A-line 提取、统计和绘图脚本。
- `results/`：后续实验生成的图像、表格和机器可读结果。
- `notes/`：实验假设、参数、结论和待验证问题。
- `aline_analysis/`：噪点、真实结构和背景位置的 512 点 PA A-line 曲线，以及统计图和坐标数据。
- `pd_comparison/`：使用 PD 归一化与不使用 PD 的图像对比。
- `time_gate_test/`：按 500 MS/s、触发延时 12.3 us 检查 12–14 us 数据的结果。

## Current finding

噪点位置的 A-line 多表现为窄脉冲或非目标深度的异常峰；真实结构的峰主要集中在稳定的深度范围内，并且脉冲更宽。PD 幅值在噪点与真实结构位置之间差异很小，因此现有数据暂不支持“PD 能量抖动是主要噪点来源”的判断。

建议后续重点检查采集链路中的瞬态尖峰、电子干扰，以及全深度最大值投影对随机异常峰的放大作用。

## Active experiment

`scripts/fixed_window_denoise.py` 是第一版固定响应窗去噪：不使用 PD、不做空间平滑，仅将每条 A-line 的最大值搜索范围从全部 512 点限制到第 250–320 点。结果写入 `results/fixed_window_v1/`。

`scripts/plot_random_pa_pd_alines.py` 从已保存的噪点和连续结构候选中可复现地随机选择一对位置，读取并绘制对应的 PA 与 PD A-line。结果写入 `results/random_pa_pd_alines_v1/`。

`scripts/plot_multiple_noise_alines.py` 将 6 个代表性噪点从上到下逐栏绘制，每栏的 PA（红）与同位置 PD（蓝）使用独立坐标轴，不进行曲线叠加。结果写入 `results/multiple_noise_alines_v1/`。

`scripts/template_xcorr_denoise.py` 将 6 条噪声 A-line 对齐并拟合固定模板，使用归一化互相关定位模板、使用最小二乘估计幅值，再对高相关且幅值足够的匹配进行减法。第一轮只生成投影和诊断结果，不修改原始数据。结果写入 `results/template_xcorr_v1/`。

完整的方法总结和阶段结论记录在 `notes/motor_crosstalk_decorrelation.log`。
