# Results Archive

这里只保存噪点成因和去噪相关结果，不包含网络训练或 SLM 校正输出。

- `noise_characterization/`：噪点、真实结构和背景 A-line 的位置与统计；
- `pd_check/`：使用与不使用 PD 归一化的对比，说明 PD 抖动不是主要原因；
- `fixed_window/`：早期固定响应窗去噪结果，用于说明时间门控有效但不够通用；
- `template_xcorr/`：最终固定模板互相关去噪的图片、float32 MIP 和指标；
- `2026_08_20_batch/`：相同模板在 D1-D50 与 origin 上的批量去噪质量报告。

做汇报时优先使用上一级 `ppt_assets/` 中按 01–09 编号的图片；本目录用于查找
原始结果和机器可读指标。
