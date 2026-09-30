# NeuWS：MMES 物体 + MMES 像差

本包在上一版“MMES 物体＋原始像差 MLP”基础上，进一步将像差分支替换成独立的 MMES。物体、像差分别拥有可学习张量 Z 和图像块自编码器，两支不共享网络参数。原始 SLM 调制符号、PSF 计算/归一化和 FFT 前向模型被继承。

这是将 MMES 扩展到物体与复场联合反演的实现，不是 MMES 论文中已验证的原始任务，也不保证比原始 NeuWS 更好。代码包含独立对照开关。

## 运行

解压整个文件夹，使用已有的 PyTorch/NeuWS 环境。打开 `recon_dual_mmes.py` 修改：

```python
DATA_DIR = r'D:/your_data/scene'
SCENE_NAME = 'scene_dual_mmes_01'
NUM_T = 100
MODEL_MODE = 'dual_mmes'
STATIC_PHASE = True
```

然后在本目录运行：

```bash
python recon_dual_mmes.py
```

也提供 `recon_dual_mmes.ipynb`。在解压目录打开 Notebook，修改配置后执行。

数据格式与上一版相同：`SLM_sim1.mat` 内 `proj_sim` 为 144×256 的弧度相位；`SLM_raw1.mat` 内 `imsdata` 为 256×256 测量。连续编号至 NUM_T；入口要求 WIDTH=256。SLM 上下补零至 256×256，实际受照孔径为中央 144×256；没有把实验孔径改为全正方形。MAX_INTENSITY=0 时使用所有测量的全局最大值归一化。

## 两个分支

| 分支 | 可学习张量 | 自编码器输入/输出 | 物理输出 |
| --- | --- | --- | --- |
| 物体 g_im | `[1,1,H,W]` | `tau_object²` 维图像块向量 | 单幅物体，线性输出 |
| 静态像差 g_g | `[1,2,H,W]` | `2*tau_aberration²` 维联合图像块向量 | 振幅残差、相位 |
| 时变像差 g_g | `[NUM_T,2,H,W]` | 同上；所有帧共享像差自编码器 | 每个测量时刻的振幅残差、相位 |

每一路都执行：REFLECT 填充 → 固定 one-hot 图像块提取 → 共享 MLP 自编码器 → 固定转置卷积重叠相加 → 裁剪并除以 tau²。多通道像差先按通道拼接块向量，经过同一个像差自编码器，再逐通道还原；不是用物体网络处理像差。

自编码器为四个 Linear 层，三个隐藏层使用 LeakyReLU(0.2)，最后线性输出。Xavier uniform 权重、零偏置。物体仍无 Sigmoid/Softplus/ReLU 输出限制；允许输出负数，这延续此前线性参数化，并不赋予负强度物理意义。

像差构造为：

```python
amplitude = AMPLITUDE_OFFSET + decoded_fields[:, 0:1]
phase = decoded_fields[:, 1:2]  # rad
complex_field = amplitude * torch.exp(1j * phase)
```

**AMPLITUDE_OFFSET 默认 1.0**，所以网络学习振幅残差，使初始复场接近单位振幅，避免从接近零的振幅开始进行 PSF 归一化。这是相对原 MLP 的明确初始化/参数化调整。振幅没有 clamp 或 Softplus，仍允许为负；相位不做 sigmoid、tanh、取模或去均值。固定 offset 不是可训练参数，AE 约束施加在残差/相位编码上，而非偏移后的振幅。

像差 MMES 只用可学习空间张量及图像块，不再输入 Zernike 基函数。原始像差 MLP 的学习参数完全替换；双 MMES 模型也移除了不用的原坐标特征。有效孔径外的像差没有测量约束，主要受网络和 AE 项影响，请仅在实际孔径内分析波前。

## 静态和时变像差

默认 STATIC_PHASE=True：全部测量共用同一幅物体和同一个像差复场；训练时一份物体噪声、一份像差噪声，各自生成一次后供当前测量 batch 共用。

STATIC_PHASE=False：物体仍为静态。像差为每个测量帧设置独立 Z 切片，所有帧共享一个像差 AE。利用原训练入口的 `t=i/(NUM_T-1)-0.5` 找回对应帧，支持打乱批次和最后一批大小为 1。只接受已采集帧对应的离散时刻，**不提供连续时间插值，也未加入时间平滑先验**；这与原始时间条件 MLP 不同。帧数越多，可学习张量和优化器状态占用越大。

## 配置与损失

```python
OBJECT_AE_WEIGHT = 1.0
ABERRATION_AE_WEIGHT = 1.0
AMPLITUDE_OFFSET = 1.0

OBJECT_MMES_CONFIG = dict(
    tau=4, ranks=[512,16,512], noise_std=0.01, seed=0,
    init_scale=0.1, chunk_size=4096, checkpoint_chunks=True,
)
ABERRATION_MMES_CONFIG = dict(
    tau=4, ranks=[512,16,512], noise_std=0.01, seed=1,
    init_scale=0.1, chunk_size=4096, checkpoint_chunks=True,
)
```

两路 Z 均初始化为 [0,0.1) 随机值，是参与优化的 nn.Parameter；不是 DIP 的固定输入。不同种子使两路初始化独立。

- 物体默认块维度 D=16、中间维度 r=16；像差默认 D=32、r=16。像差的两个通道在同一个瓶颈联合编码。ranks 始终指定三个隐藏层，输出维度由通道数与 tau 自动决定。
- noise_std 是训练时加在嵌入图像块上的高斯噪声标准差；物体与像差每次分别重新采样，设为 0 可做无噪声消融。
- chunk_size 只对图像块 MLP 计算分块，不改变全部图像块参与损失；它不是测量 BATCH_SIZE。checkpoint_chunks=True 通过反向重算激活降低显存，增加计算量。
- 测量损失和各路 AE 损失使用同一次相应分支的带噪解码。AE 目标是干净的 H(Z)，不 detach；Z 与 AE 权重均收到梯度。

```python
loss = measurement_mse + OBJECT_AE_WEIGHT*object_ae_mse + ABERRATION_AE_WEIGHT*aberration_ae_mse
```

全部使用均值 MSE。像差 AE 对振幅残差、弧度相位按当前数值尺度等权平均；时变模式对当前批的像差帧平均。两个 AE_WEIGHT 是独立的待调起点，**不等同于论文非归一化平方和下的 lambda**。本版未采用论文的自适应 lambda，也没有加 TV 或交替优化。调整两路噪声、ranks 和权重时建议逐项改变。

IM_LR 管物体 Z 与物体 AE，PH_LR 管像差 Z 与像差 AE；默认均 1e-3。IM_FINAL_LR/PH_FINAL_LR 同样默认 1e-3，即默认恒定学习率。所有最终评估与预览在 eval 模式下关闭噪声，中间预览结束后恢复训练状态。

## 三种模式

```python
MODEL_MODE = 'dual_mmes'   # MMES 物体 + MMES 像差（默认）
MODEL_MODE = 'object_mmes' # MMES 物体 + 原始像差幅相 MLP（上一版对照）
MODEL_MODE = 'original'    # 原始物体 Tensor+MLP + 原始像差幅相 MLP
```

实际运行只选择一个。object_mmes 的像差 AE 项为 0；original 的两个 AE 项均为 0。PHS_LAYERS 只影响含原像差 MLP 的对照模式，对 dual_mmes 的 AE 深度无影响。

相同 SEED 的批次顺序通过独立随机生成器保持一致。建议保持数据、帧数、batch 和测量归一化相同，同时看测量误差、实际耗时与重构质量。表达能力提高不等于联合反演结果更准确。

## 保存与恢复

输出为 `vis/<SCENE_NAME>_<MODEL_MODE>/`。相同场景同模式重复运行会覆盖；更改 SCENE_NAME 保留不同实验。

- `loss.csv`：epoch、measurement_mse、object_ae_mse、aberration_ae_mse、total。每轮按批内实际测量数加权；训练损失包含噪声。
- `config.json`：完整配置。
- `final/final_reconstruction.mat`：浮点物体、复场实部/虚部、振幅参数、绝对幅度、相位参数、实际 angle(g)、孔径、测量尺度、无噪声最终各项损失。最终时变像差 AE 损失平均所有测量帧，不仅是第一帧。
- `final/model_final.pt`：所有可学习 Z、自编码器、配置、优化器、调度器、损失历史与随机状态；提供推理恢复，没有自动断点续训入口。
- `final/summary.png`：测量、预测、物体、幅度、相位与 PSF。物体自动显示自身值域；跨实验比较请统一范围。
- `final/final_I_est.png`：截断至 [0,1] 的 8-bit 预览。定量分析用 MAT。
- VIS_FREQ 控制中间物体 MAT 和预览；时变模式且 SAVE_PER_FRAME=True 时额外导出逐帧像差。

注意 signed amplitude 可以为负，所以 `phase_parameter_t0` 不一定等于 `complex_phase_t0`（angle(g)）。两者都保存，不把神经网络相位参数直接当作唯一物理相位。

```python
import torch
from networks_DualMMES import model_from_checkpoint
state = torch.load('vis/scene_dual_mmes_01_dual_mmes/final/model_final.pt',
                   map_location='cpu', weights_only=True)
net = model_from_checkpoint(state)
with torch.no_grad():
    object_est, complex_field, phase_parameter = net.get_estimates(torch.tensor([-0.5]))
```

恢复静态图像及任意已采集帧像差无需原 MAT 输入；生成该帧预测测量仍需对应的 SLM 图案。

## 文件与验证

- networks_MMES.py：固定 H/Hinv 和通用多通道/多帧 MMES；也保留上一版物体单分支模型。
- networks_DualMMES.py：两路 MMES、离散帧索引、三种模型构建与 checkpoint 恢复。
- recon_dual_mmes.py / .ipynb：训练入口。
- networks.py、dataset.py、utils.py、LICENSE.txt：原 SA 文件，原样保留。

```bash
python validate_mmes.py
python validate_dual_mmes.py
python validate_end_to_end.py
```

验证包括：one-hot 块顺序、反射边界、重叠还原与梯度；多通道解码；时变帧选择；两路参数独立；仅用测量 loss 时梯度能到达物体及像差的两个通道；PSF 归一化；batch=1；完整训练、日志、保存和恢复。完整入口用三帧合成 MAT 数据验证双 MMES 静态/时变及两种对照模式，结束后删除临时数据。

开发验证为 CPU、PyTorch 2.5.1+cpu、torchvision 0.20.1+cpu、NumPy 1.26.4；未验证 CUDA 或真实实验质量。依赖沿用原 NeuWS 环境（torch、torchvision、numpy、scipy、aotools、h5py、imageio、matplotlib、Pillow、tqdm），默认梯度检查点使用 PyTorch 2.x 接口。requirements_original.txt 仅作历史参考，未列 torch/torchvision，无需为此降级已有可用环境。

MMES 参考：Yokota et al., *Manifold Modeling in Embedded Space: An Interpretable Alternative to Deep Image Prior*, IEEE TNNLS 33(3), 2022. https://doi.org/10.1109/TNNLS.2020.3037923
