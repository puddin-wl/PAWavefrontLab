# PAWavefrontLab source and attribution notice

## Project identity

PAWavefrontLab is an extended research workflow for photoacoustic wavefront
simulation, measurement preprocessing, and neural reconstruction. The project
name applies to the combined workflow maintained in this repository.

## Upstream NeuWS code

This repository was derived from the NeuWS repository published by the
Intelligent Sensing Lab:

- Repository: <https://github.com/Intelligent-Sensing/NeuWS>
- Paper: *NeuWS: Neural Wavefront Shaping for Guidestar-Free Imaging Through
  Static and Dynamic Scattering Media*
- Local provenance marker: Git tag `upstream-baseline`

The names "NeuWS" and "Neural Wavefront Shaping" continue to identify the
upstream method, model, paper, and compatible measurement format. Renaming this
extended project does not claim ownership of those upstream contributions.

The original `LICENSE.txt` is retained. It contains research-use and
redistribution restrictions and must be reviewed before this repository or any
derived copy is shared with third parties. In particular, a new project name
does not replace or relax the upstream license terms.

## PAWavefrontLab extensions

Development after the `upstream-baseline` tag includes, among other changes:

- Python/AOtools optical primitives and closed-loop simulation;
- configurable Zernike aberration generation;
- reproducible static simulation stages and evaluation tools;
- photoacoustic TIFF and packed-BIN preprocessing;
- real-measurement import and SLM correction utilities;
- tests, Chinese tutorials, experiment logs, and VS Code entry points;
- compatibility updates for the verified local CUDA/PyTorch environment.

Use `git diff upstream-baseline` to inspect the exact code-level boundary.
