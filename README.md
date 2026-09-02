# PAWavefrontLab

**Photoacoustic wavefront simulation, measurement preprocessing, and neural reconstruction.**

PAWavefrontLab is a research workflow built around the NeuWS reconstruction
method. It adds Python/AOtools closed-loop simulation, photoacoustic TIFF/BIN
preprocessing, real-measurement import, reconstruction evaluation, and
reproducible experiment entry points.

NeuWS remains the name of the upstream algorithm and its compatible data
contract; it is not the name of this extended project. See [NOTICE.md](NOTICE.md)
for upstream attribution, the extension boundary, and redistribution constraints.

## 中文快速入口（第一次使用请先看这里）

这个项目可以完成四类任务：

| 目标 | 从哪里开始 | 主要输出 |
| --- | --- | --- |
| 给一张清晰图片添加指定 Zernike 像差 | 编辑并运行 `run_single_image.py` | 像差相位图、PSF、模糊图 |
| 完整运行“生成相位→模拟测量→网络恢复→评价” | 阅读 [`workflows/static_simulation/README.md`](workflows/static_simulation/README.md) | 50 张 SLM 相位、50 张测量图、恢复图像和恢复像差 |
| 导入三维光声 TIFF 后运行 NeuWS | 步骤二后运行 `step3_import_photoacoustic_measurements.py` | 减 2048、置零、第 0 维投影后的 `SLM_rawN.mat` |
| 去除 PA 图像中的电机串扰噪点 | 阅读 [`motor_crosstalk_denoise/README.md`](motor_crosstalk_denoise/README.md) | 去噪 MIP、NCC/匹配位置、前后对照图和指标 |
| 查阅实验条件、结果与失败记录 | 阅读 [`log/README.md`](log/README.md) | 按日期整理的实验记录索引 |
| 使用自己的 MATLAB/相机数据重建 | 查看下方“Static reconstruction”和“Data contract” | `final_I_est.mat`、`final_aberration.mat` |

如果你的目标是第一次完整复现今天验证过的仿真，请不要从旧的
`run_dataset.py` 开始，而应按以下顺序操作：

1. 在 VS Code 中选择解释器
   `/home/xiangwan/miniconda3/envs/neuws/bin/python`；
2. 打开 `workflows/static_simulation/config.py`；
3. 修改输入图片、`data_dir` 和 `scene_name`；
4. 依次运行同目录下 `step1` 到 `step5`；
5. 在 `outputs/SCENE_NAME/evaluation/reconstruction_comparison.png` 查看总览图，
   在 `reconstruction_report.json` 查看定量指标。

重要提醒：当前默认运行名 `test_static_zernike_50` 已经有正式结果，而且配置中
`overwrite=False`。新用户应把 `data_dir` 和 `scene_name` 同时改成一个新的、
相同的名称，例如 `my_first_simulation`，不要直接覆盖现有结果。

项目中统一使用以下术语：

- **清晰物体**：仿真的原始清晰图片；
- **系统像差相位图**：固定、对网络未知的真实像差；
- **SLM 调制相位图**：逐帧变化、对网络已知的相位；
- **调制测量图**：清晰物体经过“固定系统像差 + 对应 SLM 相位”后形成的图像；
- **恢复结果**：网络估计的清晰物体和系统像差相位。

完整的中文入门教程、参数表、输出说明、代码职责表和故障排查见：

**[`workflows/static_simulation/README.md`](workflows/static_simulation/README.md)**

2026-07-30 的实现过程、CUDA 验证和正式指标见：

**[`log/2026-07-30_NeuWS完整静态仿真与CUDA重建.md`](log/2026-07-30_NeuWS完整静态仿真与CUDA重建.md)**

PAWavefrontLab uses and extends code associated with “NeuWS: Neural Wavefront
Shaping for Guidestar-Free Imaging Through Static and Dynamic Scattering Media”.
The NeuWS name below refers to that upstream method, model, and data format.

- [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10306297/)
- [Dryad dataset and format description](https://datadryad.org/dataset/doi%3A10.5061/dryad.6t1g1jx42)
- [Original upstream repository](https://github.com/Intelligent-Sensing/NeuWS)
- [Upstream attribution and project boundary](NOTICE.md)

## Verified local environment

The migration was verified in WSL on an NVIDIA GeForce RTX 5070 Ti (16 GB):

- Python 3.10 in the `neuws` Conda environment
- PyTorch 2.13.0+cu130
- CUDA Toolkit 13.0 (`/usr/local/cuda`)
- cuDNN 9.25 system packages; PyTorch reports cuDNN 9.2
- AOtools 1.0.7

Activate the existing environment with:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neuws
```

For a fresh compatible environment, install the PyTorch build appropriate for the machine first, then install the remaining packages:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
```

The NVIDIA Windows driver is exposed to WSL; it should not be installed again inside WSL. The Linux CUDA Toolkit and cuDNN can coexist with that driver.

## Run directly from VS Code (recommended for this project)

No terminal command is required. First make sure VS Code has selected the interpreter:

```text
/home/xiangwan/miniconda3/envs/neuws/bin/python
```

Then use one of the two root-level entry points:

1. Open `run_single_image.py`, edit the clearly marked settings block at the top, and click **Run Python File**. Edit `INPUT_IMAGE`, `OUTPUT_DIR`, `SIZE`, and the `ZERNIKE_COEFFICIENTS` dictionary.
2. Open `run_dataset.py`, edit its settings block, and click **Run Python File**. Dataset aberrations remain randomly generated and independent from the single-image coefficient dictionary.

Both files contain working defaults for this machine. The command-line interfaces described below remain available and override nothing in the VS Code entry points; the two ways of running are independent.

## Complete five-step static simulation

For the reproducible 50-frame closed-loop simulation, edit the Chinese-commented
settings in `workflows/static_simulation/config.py`, then open the same directory
and run these files in order from VS Code:

1. `step1_prepare_ground_truth.py` creates the normalized clear object, one fixed
   seeded Zernike system aberration, its phase/field, the flat-SLM PSF, and the
   baseline aberrated measurement.
2. `step2_generate_slm_patterns.py` creates 50 known random SLM modulation phases.
3. `step3_simulate_measurements.py` reads the saved object, system aberration and
   SLM phases and creates every camera frame directly from their combined pupil.
   It never applies another blur to the baseline aberrated image.
   For acquired 3-D photoacoustic TIFF stacks, run
   `step3_import_photoacoustic_measurements.py` instead: it computes
   `max(raw - 2048, 0)` and then an axis-0 maximum projection, without requiring
   a fixed 512-layer depth.
4. `step4_reconstruct.py` supplies only the 50 measurements and their known SLM
   phases to the static NeuWS network. The system-aberration ground truth is not
   supplied to the network.
5. `step5_evaluate.py` compares the recovered object and system aberration with
   their simulation ground truths and writes an aggregate report and figures.

The loader normalizes camera measurements by one dataset-wide maximum. The
reconstruction stage therefore saves the raw non-negative network estimate and
uses that recorded maximum to produce the radiometrically restored
`reconstructed_object`; evaluation never compares the normalized network units
directly with the original object.

The default dataset directory is `data/test_static_zernike_50/`. Human-readable
outputs use names such as `clear_object`, `system_aberration_phase`,
`slm_phase_0001`, `modulated_measurement_0001`, and `reconstructed_object`.
The `SLM_simN.mat:proj_sim` and `SLM_rawN.mat:imsdata` files remain alongside them
as the network/hardware interchange contract. A future camera acquisition can
replace only step 3 while keeping steps 2 and 4 unchanged. The exported 16-bit
phase PNGs are generic wrapped-phase previews and still require a device-specific
SLM LUT before hardware use.

The implementation and entry points are intentionally grouped under
`workflows/static_simulation/` so the project root remains focused on the shared
NeuWS model and optical primitives. See
`log/2026-07-30_NeuWS完整静态仿真与CUDA重建.md` for the verified run, metrics,
normalization correction, and the route from simulation to a future camera experiment.

## Program 1: apply a specified aberration to one image

This presentation-oriented program uses the full square field and requires explicit AOtools Noll coefficients. Coefficients are in radians. For example, Noll 4 is Defocus and Noll 7 is one Coma direction:

```bash
python tools/apply_aberration.py \
  --input-image /path/to/object.tif \
  --output-dir outputs/single_aberration \
  --size 256 \
  --coefficient 4=1.5 \
  --coefficient 7=-0.25 \
  --device cuda
```

Unspecified Noll terms are zero. Two other coefficient input formats are supported:

```bash
# Dense Noll 1..N list; it is zero-padded to 28 terms.
python tools/apply_aberration.py ... \
  --coefficients "0,0,0,1.5,0,0,-0.25"

# A .npy, JSON, CSV, or text file.
python tools/apply_aberration.py ... \
  --coefficients-file coefficients.npy
```

The program saves the processed input, 16-bit aberrated PNG, wrapped phase PNG, exact NumPy arrays, and `aberration_result.mat`. The MATLAB file contains `object_image`, `aberrated_image`, `zernike_coefficients`, `aberration_phase`, `aberration_field`, `aberration_amplitude`, and `psf`. `manifest.json` records the complete coefficient list and conventions.

## Program 2: generate a NeuWS dataset

The dataset program deliberately keeps its aberration generation independent from Program 1. It samples one static unknown aberration from the selected distribution, then varies only the SLM modulation across frames. The sampled coefficients are saved in `ground_truth.mat` for inspection, but are not supplied through the single-image coefficient interface.

### Generate SLM patterns only

The project default is a fully illuminated square field. Each pattern is a weighted sum of AOtools Noll modes 1–15, including Piston/Tip/Tilt, with independent Gaussian coefficients of default standard deviation `5 rad`.

```bash
python tools/generate_neuws_data.py patterns \
  --output-dir data/patterns \
  --size 256 --num-frames 100 --seed 0
```

Any positive even square size is supported. By default `aperture_height=size`, so no region is masked. The paper's original `144×256` geometry remains available only when explicitly requested:

```bash
python tools/generate_neuws_data.py patterns \
  --output-dir data/paper_patterns \
  --size 256 --aperture-height 144 --num-frames 100
```

Each run exports:

- `SLM_simN.mat`, variable `proj_sim`, containing active-aperture phase in radians.
- `slm_patterns.npy`, shaped `frames×active_height×size`.
- `slm_coefficients.npy`.
- `slm_png/SLM_simN.png`, a generic 16-bit encoding of wrapped `[0,2π)` phase to `[0,65535]`.
- `manifest.json`, containing dimensions, seed, Noll modes, phase sign and parameters.

The PNG files are not calibrated for a particular SLM. Device-specific LUT, gamma and voltage conversion must be added before hardware use.

### Generate the measurements

The input is converted to grayscale, center-cropped to a square, resized, and normalized to `[0,1]`. The object and unknown aberration remain static; only the SLM pattern changes per frame.

For an easily verified 28-mode Zernike aberration:

```bash
python tools/generate_neuws_data.py simulate \
  --input-image /path/to/object.tif \
  --output-dir data/sim_static \
  --size 256 --num-frames 100 \
  --aberration-mode zernike --aberration-sigma 1 \
  --seed 0 --device auto
```

For paper-style static scattering:

```bash
python tools/generate_neuws_data.py simulate \
  --input-image /path/to/object.tif \
  --output-dir data/sim_scattering \
  --size 256 --num-frames 100 \
  --aberration-mode complex-gaussian --seed 0
```

`complex-gaussian` creates an independent circular complex Gaussian pupil field and normalizes its mean aperture energy to one. `--noise-std` adds optional Gaussian camera noise and defaults to zero. Synthetic runs additionally create `SLM_rawN.mat` (`imsdata`, full square measurements) and `ground_truth.mat` (object, complex aberration, amplitude, phase and coefficients).

The phase convention matches the paper loader:

```text
aperture * exp(-1j * proj_sim)
```

## Static reconstruction

The loader infers the square measurement size and active area from the files. Current project datasets use the full field by default. If `--num_t` is omitted, all continuously numbered samples are used. If `--width` is provided, it must match the inferred size. Frames are loaded from disk batch by batch rather than copied to the GPU all at once.

```bash
python recon_exp_data.py \
  --static_phase \
  --data_dir data/sim_static \
  --scene_name sim_static \
  --num_epochs 1000 --phs_layers 4
```

Absolute data paths are also accepted. The original experimental-data arguments remain available, including `--im_prefix`, `--slm_prefix`, `--num_t`, `--max_intensity`, `--zero_freq`, `--dynamic_scene`, and `--save_per_frame`. The unknown-aberration network still uses the original 28 AOtools Zernike input features; it does not reduce them to 15 or clear low-order modes.

Results are written under `vis/SCENE_NAME/final/`, including `final_I_est.mat`, `final_aberration.mat`, display images, and `training_summary.json`. The `per_frame` directory is created only when `--save_per_frame` is enabled.

## Evaluate a reconstruction

Image metrics include raw PSNR/SSIM and, with `--register`, translation-registered metrics, recovered shift, and valid overlap:

```bash
python tools/evaluate_neuws.py image \
  --ground-truth data/sim_static/ground_truth.mat \
  --ground-truth-var object_image \
  --estimate vis/sim_static/final/final_I_est.mat \
  --estimate-var image \
  --register --output-dir outputs/sim_static/image_metrics
```

Phase metrics use wrapped error inside the aperture. The primary result removes only Piston, Tip and Tilt (Noll 1–3). A separate diagnostic also removes Defocus and Coma (Noll 4, 7 and 8):

```bash
python tools/evaluate_neuws.py phase \
  --ground-truth data/sim_static/ground_truth.mat \
  --ground-truth-var aberration_field \
  --estimate vis/sim_static/final/final_aberration.mat \
  --estimate-var field \
  --output-dir outputs/sim_static/phase_metrics
```

Both commands save `metrics.json` and a comparison figure. Phase evaluation also saves calibrated error arrays in `phase_errors.mat`.

## Data contract and validation

Samples must be continuously numbered from 1:

```text
SLM_sim1.mat  -> proj_sim: active_height × size
SLM_raw1.mat  -> imsdata:  size × size
SLM_sim2.mat
SLM_raw2.mat
...
```

Missing indices, missing variables, empty arrays, non-square or odd-sized measurements, inconsistent dimensions, non-finite values, negative measurements, and invalid normalization ranges raise explicit errors. New data use `manifest.json`; original paper data without a manifest default to phase sign `-1`.

## Tests

Run the complete CPU test suite with pytest. Pytest also collects the existing
`unittest.TestCase` tests, so both test styles run through one entry point:

```bash
python -m pytest -q
```

The large `1000×1000` CUDA acceptance test is opt-in because it consumes substantial GPU memory:

```bash
NEUWS_RUN_LARGE_CUDA=1 python -m pytest -q \
  tests/test_optics_and_evaluation.py::LargeCudaTests::test_1000_forward_backward
```

See [MIGRATION.md](MIGRATION.md) for the MATLAB-to-Python mapping and scope decisions.
