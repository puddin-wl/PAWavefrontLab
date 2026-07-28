# NeuWS: Neural Wavefront Shaping

This repository contains the code for “NeuWS: Neural Wavefront Shaping for Guidestar-Free Imaging Through Static and Dynamic Scattering Media” and a Python/AOtools closed-loop workflow for generating SLM patterns, simulating measurements, reconstructing a static scene, and evaluating the result.

- [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10306297/)
- [Dryad dataset and format description](https://datadryad.org/dataset/doi%3A10.5061/dryad.6t1g1jx42)
- [Original upstream repository](https://github.com/Intelligent-Sensing/NeuWS)

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

Run the CPU test suite with:

```bash
python -m unittest discover -s tests -v
```

The large `1000×1000` CUDA acceptance test is opt-in because it consumes substantial GPU memory:

```bash
NEUWS_RUN_LARGE_CUDA=1 python -m unittest \
  tests.test_optics_and_evaluation.LargeCudaTests -v
```

See [MIGRATION.md](MIGRATION.md) for the MATLAB-to-Python mapping and scope decisions.
