# MATLAB-to-Python migration notes

The old Windows project under `E:\mlp_code` was used only as a behavioral reference. Its MATLAB files and historical `1000×1000` datasets/results are not copied into this repository.

## Functional mapping

| Previous MATLAB responsibility | Python replacement |
| --- | --- |
| `zernfun.m` and `zernidx2nm.m` | AOtools Noll numbering and normalization through `optics.zernike_basis_numpy` / `zernike_basis_torch` |
| Construct the first 15 random SLM modes | `tools/generate_neuws_data.py patterns` |
| Crop the computational pattern to the physical SLM height | `optics.crop_to_aperture` with a centered, configurable even aperture |
| Save per-pattern MATLAB phase files | `SLM_simN.mat` with variable `proj_sim` |
| Forward scattering and camera-frame synthesis | `tools/generate_neuws_data.py simulate`, using the same PSF normalization and convolution convention as NeuWS |
| Static Zernike test aberration | `--aberration-mode zernike` with 28 AOtools modes |
| Static random scattering field | `--aberration-mode complex-gaussian` |
| Apply user-specified aberration to one presentation image | `tools/apply_aberration.py` |
| Image/result comparison scripts | `tools/evaluate_neuws.py image` and `phase` |

## Deliberate scope decisions

- The SLM generator uses Noll modes 1–15 with independent `N(0, 5²)` radian coefficients and does not exclude Piston/Tip/Tilt.
- The reconstruction network remains the upstream 28-feature aberration model. The old local experiment that reduced it to 15 modes and zeroed low-order terms was not migrated.
- The project default is a fully illuminated positive even square field. A smaller centered rectangular aperture remains an explicit compatibility option for paper data.
- Explicit user-provided coefficients belong only to the single-image presentation program. Dataset aberrations remain independently sampled and are saved as ground truth.
- Object and unknown aberration are static by default. The SLM modulation changes frame by frame.
- Phase calibration removes only Piston/Tip/Tilt for the primary metric. Defocus/Coma removal is diagnostic only.
- Sixteen-bit SLM PNG export is a generic phase encoding, not a hardware calibration.
- MATLAB and Octave are not runtime dependencies.

The upstream baseline is retained in local Git as tag `upstream-baseline`; the migration is delivered as subsequent project commits.
