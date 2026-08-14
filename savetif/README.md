# BIN 转多页 TIFF

2026-08-11 实际采集数据从 `600×600×512` BIN 到 Windows 兼容单层最大值
投影的完整处理、问题排查和验证记录见：

**[`../log/2026-08-12_光声BIN最大值投影与Windows兼容TIFF.md`](../log/2026-08-12_光声BIN最大值投影与Windows兼容TIFF.md)**

`bin_to_tiff.py` 是 `BinTif.m` 与 `saveastiff.m` 的 Python 版本。默认保持原始
数据约定：

- BIN 是连续打包的 12 位无符号整数，采用 MATLAB/Windows 原生小端位序；
- 体数据形状为 `1000 × 1000 × 512`，深度维在 BIN 中变化最快；
- 输出为 512 页 `1000 × 1000` 的灰度 `float32` TIFF，不压缩；
- 批量转换时沿用 MATLAB 的命名方式，例如 `fish.bin` 输出为 `fish.bin.tif`。

## 批量转换一个目录

在项目根目录运行：

```bash
python savetif/bin_to_tiff.py /path/to/bin_directory /path/to/tiff_directory
```

已有文件默认不会被覆盖。确认要覆盖时加入：

```bash
python savetif/bin_to_tiff.py /path/to/bin_directory /path/to/tiff_directory --overwrite
```

## 转换单个文件

```bash
python savetif/bin_to_tiff.py input.bin output.tif
```

其他尺寸可显式指定：

```bash
python savetif/bin_to_tiff.py input.bin output.tif \
  --height 1000 --width 1000 --depth 512
```

默认尺寸下，输入 BIN 应为 768,000,000 字节。转换期间会在输出目录创建约
0.95 GiB 的临时解码文件，成功或失败后都会自动清理。若输出盘空间不足，可用
`--temp-dir /path/to/large_disk` 将它放到其他磁盘。最终 TIFF 约为 1.91 GiB。

## 直接生成单层最大值投影

光声数据可跳过中间的三维 TIFF，直接执行 `max(raw - 2048, 0)` 后沿 512 层
取最大值。批量模式默认只处理文件名以 `_PA1.bin` 结尾的数据：

```bash
python savetif/bin_to_tiff.py /path/to/bin_directory /path/to/mip_directory \
  --height 600 --width 600 --depth 512 --mip
```

每个输入会输出为 `原文件名_mip.tif`，图像是单层 `uint16` TIFF。文件使用
Windows 兼容性较好的 PackBits、小条带和 72 DPI 标签。12 位原始值减去默认
基线 2048 后范围为 `0–2047`，不指定显示映射时可无损保存。

若需要在普通图片查看器中获得合适亮度，可将同一组数据的共同显示上限映射到
65535。组内必须使用同一个值，才能保持帧之间的相对强度：

```bash
python savetif/bin_to_tiff.py /path/to/bin_directory /path/to/mip_directory \
  --height 600 --width 600 --depth 512 --mip --display-max 417
```

需要处理其他命名的 BIN 时，可通过 `--filename-suffix` 修改批量匹配后缀。
