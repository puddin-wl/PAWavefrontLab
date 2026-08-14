# 2026-08-12 光声 BIN 最大值投影与 Windows 兼容 TIFF 处理记录

## 1. 目标与最终结果

本次将 `E:\neuws_data\raw\2026-08-11` 中的光声采集 BIN 直接处理为
Windows 普通图片查看器可识别的单层 16 位 TIFF。完整计算流程为：

```text
packed ubit12 BIN
→ 恢复 600×600×512 光声体
→ 每个体素减去交流零点 2048
→ 负值置零
→ 沿 512 层深度维做最大值投影
→ 全组统一映射到 0–65535
→ Windows 兼容的 uint16 PackBits TIFF
```

程序实现位于：

```text
savetif/bin_to_tiff.py
```

最终生成 6 张 `600×600`、单页、16 位灰度最大值投影，保存到：

```text
E:\neuws_data\raw\2026-08-11\mip
```

原始 BIN 未修改，`PD1` 配对通道未参与本次光声最大值投影。

## 2. 输入数据核验

输入目录中实际存在 12 个 BIN，即 6 组 `PA1/PD1` 配对文件：

- 6 个文件名以 `_PA1.bin` 结尾；
- 6 个文件名以 `_PD1.bin` 结尾；
- 每个文件大小均为 `276,480,000` 字节。

指定体数据尺寸为：

```text
600 × 600 × 512 = 184,320,000 个 12 位采样值
184,320,000 × 12 / 8 = 276,480,000 字节
```

实际文件大小与 packed 12 位格式严格吻合。批量入口默认只选择 `_PA1.bin`，
避免把 `PD1` 参考通道误当成三维光声体处理。

## 3. 12 位 BIN 解码与体数据排列

原 MATLAB 程序使用：

```matlab
data = fread(fid, 'ubit12');
```

在 Windows/x86 原生小端位序下，每 3 字节连续打包两个 12 位无符号值。设
三个字节为 `b0, b1, b2`，Python 解码关系为：

```text
value0 = b0 | ((b1 & 0x0F) << 8)
value1 = (b1 >> 4) | (b2 << 4)
```

线性数据恢复成 `(height, width, depth)`，其中 `depth` 变化最快。本次形状为：

```text
(600, 600, 512)
```

这与原 MATLAB 三重循环中对 `sample(l, w, h)` 的填充顺序一致。

## 4. 基线校正和最大值投影

本项目已有光声预处理约定为：

```python
corrected = max(raw - 2048, 0)
projection = max(corrected, axis=depth)
```

由于减去常数和截零都是单调运算，下面两种写法数学上等价：

```text
max(max(raw - 2048, 0), depth)
= max(max(raw, depth) - 2048, 0)
```

程序因此按若干行分块解码，在内存中直接沿 512 层取最大值，然后减 2048
并截零。这样不需要先生成约 700 MiB 的 uint16 三维数组或约 1.4 GiB 的
float32 三维 TIFF，也不会产生数值差异。

直接投影实现已用已知 packed-12 小样本与完整三维处理结果逐像素比较，结果
完全一致。

## 5. Windows 显示问题及原因

第一次输出采用单页 `float32` TIFF，实际像素并非全白：

- 每张有 237–320 个不同灰度值；
- 最小值约 18–20；
- 全组最大值为 417；
- 图像中存在清晰的线状空间结构。

但很多 Windows 普通图片查看器将浮点 TIFF 的有效显示范围当作 `[0,1]`，
所有大于 1 的像素会被截成白色。随后改成无损 `uint16` 后，ImageJ 可以读取，
部分 Windows 查看器仍不能识别原先由 tifffile 写出的单大条带科学 TIFF。

对用户提供、可被 Windows 正常识别的 `1.tif` 进行标签对比后，确认它采用：

- 16 位无符号灰度；
- PackBits 压缩；
- 小条带布局，`RowsPerStrip=15`；
- `PhotometricInterpretation=MinIsBlack`；
- `PlanarConfiguration=Contiguous`；
- `Orientation=TopLeft`；
- 72 DPI，分辨率单位为英寸；
- 经典 TIFF，而不是 BigTIFF、ImageJ TIFF 或 OME-TIFF。

最终程序使用 Pillow 写出同类标签组合，从而提高 Windows 原生查看器兼容性。

## 6. 16 位显示映射

减去基线后的理论范围只有 `0–2047`，使用 uint16 足够。若原值直接写入
uint16，数据本身无损，但本次实际范围只有 `18–417`，在固定显示范围
`0–65535` 的普通查看器中会显得很暗。

本次没有逐张自动拉伸，而是使用整组共同最大值 `417` 做统一线性映射：

```text
windows_value = round(clip(projection / 417, 0, 1) × 65535)
```

这样不会截掉任何一张图的亮部，并保持 6 张图之间的相对强度关系。最终文件
范围如下：

| 文件标识 | 16 位显示范围 |
| --- | ---: |
| `w_05` | 3143–43376 |
| `w_1` | 2829–45576 |
| `w_f05` | 3143–65535 |
| `w_f1` | 2986–42904 |
| `wn_1` | 2986–55477 |
| `wn_2` | 3143–64435 |

重要：当前 `mip` 目录中的 TIFF 是显示编码值，不再直接等于“减 2048 后的
原始投影强度”。需要近似恢复时使用：

```text
projection ≈ windows_value × 417 / 65535
```

如果后续算法需要严格的定量输入，应重新运行但不要传入 `--display-max`；此时
输出仍是 Windows 兼容的 uint16 TIFF，但像素值直接保存原始基线校正投影。

## 7. 本次实际命令

在 WSL 项目根目录 `/home/xiangwan/program/PAWavefrontLab` 运行：

```bash
/home/xiangwan/miniconda3/envs/neuws/bin/python \
  savetif/bin_to_tiff.py \
  /mnt/e/neuws_data/raw/2026-08-11 \
  /mnt/e/neuws_data/raw/2026-08-11/mip \
  --height 600 \
  --width 600 \
  --depth 512 \
  --mip \
  --display-max 417 \
  --overwrite
```

Windows 路径与 WSL 路径对应关系为：

```text
E:\neuws_data\raw\2026-08-11
↔ /mnt/e/neuws_data/raw/2026-08-11
```

## 8. 后续复用方式

### 定量无损的 uint16 最大值投影

```bash
python savetif/bin_to_tiff.py INPUT_DIR OUTPUT_DIR \
  --height 600 --width 600 --depth 512 --mip
```

此模式不缩放投影值，适合后续计算和 NeuWS 数据导入。

### Windows 直接查看的组内统一显示版

先统计同组所有定量投影的共同显示上限，再对整组使用同一个值：

```bash
python savetif/bin_to_tiff.py INPUT_DIR OUTPUT_DIR \
  --height 600 --width 600 --depth 512 --mip \
  --display-max GROUP_MAX
```

不要对需要互相比较的图像逐张选用不同 `display-max`，否则会破坏帧间相对
强度关系。

## 9. 最终验证

6 张输出均重新读取并通过以下检查：

- 数量：6；
- 尺寸：`600×600`；
- TIFF 页数：1；
- 数据类型：`uint16`；
- 位深标签：16；
- 压缩：PackBits；
- `RowsPerStrip=15`；
- 图像方向：TopLeft；
- 分辨率：72 DPI；
- 所有像素值有效且位于 16 位范围内。
