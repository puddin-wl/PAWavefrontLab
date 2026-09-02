# Research Log

## Data and acquisition assumptions

- PA volume shape: 600 × 600 × 512
- The 600 × 600 axes form the scanning plane; 512 samples form each depth A-line.
- Sampling rate: 500 MS/s, pending confirmation from acquisition metadata.
- Trigger delay: 12.3 us.
- Raw data is read from its original location and is not modified here.

## Initial observation

- Noise candidates often contain narrow abnormal peaks or peaks outside the stable target-depth region.
- True structures are more spatially continuous and their A-line peaks are concentrated within a narrower depth interval.
- Current comparisons do not show a strong relationship between PD amplitude and noise-point occurrence.

## Next investigation

1. Define noise points from spatial continuity and isolated intensity, without using PD.
2. Extract complete 512-sample A-lines at noise, structure and background locations.
3. Compare peak depth, peak width, amplitude, energy, ringing and neighboring A-line consistency.
4. Determine whether the abnormal points originate from impulsive electronic interference, reconstruction/projection, or acquisition timing.

## Fixed-window denoising v1

- Method: for every 512-sample A-line, replace the full-depth positive maximum with the positive maximum over samples 250:320.
- Absolute gate under the current timing assumption: 12.800–12.940 us.
- PD normalization: not used.
- Spatial smoothing: not used.
- The full-depth peak was outside the gate for 76.37% of pixels with a positive projection value.
- Using the same high local-residual threshold before and after processing, 920 of 1094 isolated bright-point candidates were removed (84.10%).
- The 99.7th-percentile amplitude remained 217 ADC before and after gating, indicating that the strongest in-window structures were retained.
- Interpretation: fixed depth gating is effective against most isolated out-of-window peaks. Faint structures whose response depth shifts outside the fixed gate may be attenuated and must be checked before generalizing the window.

Results: `../results/fixed_window_v1/`

## Random PA/PD A-line comparison v1

- Reproducible random seed: 20260824.
- Selection pool: the six saved representative candidates in each category from `aline_analysis/analysis.json`.
- Random noise point: N4 at (y=567, x=126).
- Random continuous-signal point: S3 at (y=60, x=574).
- PA noise-point peak: sample 502, 13.304 us, +220 ADC relative to baseline; outside the 250:320 PA gate.
- PA continuous-signal peak: sample 291, 12.882 us, +312 ADC relative to baseline; inside the gate and part of a broad bipolar response.
- PD peak at noise point: +1350 ADC at sample 320.
- PD peak at signal point: +1351 ADC at sample 321.
- Immediate observation: the two PD waveforms are nearly identical in amplitude and shape, while the PA waveforms differ strongly. The selected noise A-line shows a late, slowly varying excursion toward the record boundary rather than the compact bipolar response seen at the continuous structure.
- Caveat: this is one random pair and is evidence for forming a hypothesis, not a population-level conclusion.

Results: `../results/random_pa_pd_alines_v1/`

## Multiple noise A-lines v1

- Reviewed all six saved representative noise candidates as separate top-to-bottom lanes; PA and PD are not overlaid.
- PA peak samples: N1=44, N2=61, N3=77, N4=502, N5=94, N6=296.
- Five of six PA peaks are outside the 250:320 response gate; N6 is inside the gate.
- PD peak samples remain tightly grouped from 317 to 322, with peak amplitudes from 1327 to 1350 ADC.
- N1, N2, N3 and N5 show broad low-frequency PA excursions with a narrow local peak at an early record position.
- N4 is a distinct late-record/end-boundary excursion.
- N6 is a distinct in-window event, so fixed gating cannot remove this noise class.
- The stable PD pulse across all six locations argues against laser-energy fluctuation as the common cause of these examples.
- Working hypothesis: at least two mechanisms are present—out-of-window baseline/ringing or timing artifacts, plus an in-window isolated PA event that requires spatial or waveform-shape discrimination.

Results: `../results/multiple_noise_alines_v1/`

## Shifted-template cross-correlation denoising v1

- Template: align N1–N6 by their positive PA peak, normalize their peak amplitudes, and take a robust median over relative samples -180:180.
- Detection: maximum absolute normalized cross-correlation, with threshold 0.70.
- Safety condition: the least-squares fitted template peak must be at least 80 ADC.
- Subtraction amplitude: least-squares coefficient, not the correlation coefficient.
- Synthetic validation recovered both the exact inserted template centers and amplitudes, including a template truncated by the record boundary.
- Noise-point NCC values: 0.899–0.957; all six noise examples were subtracted.
- Continuous-signal NCC values: 0.080–0.205; none of the six saved continuous-signal examples were subtracted.
- Full image: 10,883 of 360,000 pixels matched the template (3.02%).
- Isolated bright-point check: 1,092 of 1,094 original high local-residual candidates were removed (99.82%).
- Projection median remained 50 ADC; projection P99.7 changed only from 217 to 216 ADC.
- Coherent-structure check: only 29 of the top 2,520 coherent pixels matched (1.15%); median and P95 projection change in that mask were both 0 ADC.
- Interpretation: the dominant isolated points contain a highly repeatable PA waveform appearing at variable sample positions. This strongly supports a shifted-template noise model, but does not by itself identify whether the physical origin is sampling jitter, asynchronous interference, or PA-channel timing/baseline behavior.
- Caveat: the first template was fitted from six selected examples. A held-out acquisition is required before treating the threshold and template as general.
- Raw PA and PD files were not modified. This trial saves projection-level outputs only.

Results: `../results/template_xcorr_v1/`
