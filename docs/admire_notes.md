# ADMIRE in zea: hand-off notes

Notes for whoever continues the ADMIRE (Aperture Domain Model Image REconstruction)
work, for example in an environment with a GPU and Hugging Face data access. The
first version was written on a CPU-only machine with JAX and TensorFlow; it was
never run on a GPU, on real data, or on the torch backend.

Branch: `claude/admire-beamformer`.

## Where things are

| What | Where |
| --- | --- |
| Model generation (NumPy), solver and reconstruction (keras ops) | `zea/beamform/admire.py` |
| Pipeline operation, registered as `"admire"` in `ops_registry` and `beamformer_registry` | `zea/ops/pipeline.py`, class `ADMIRE` (right after `MinimumVariance`) |
| Tests | `tests/test_admire.py` |
| Citations (`byram2015model`, `khan2021realtime`) | `docs/source/references.bib` |
| Reference implementation (MATLAB/C/CUDA, Apache-2.0) | <https://github.com/VU-BEAM-Lab/ADMIRE> |

Papers:
- Byram et al., "A model and regularization scheme for ultrasonic beamforming
  clutter reduction", IEEE TUFFC 62(11), 2015, doi:10.1109/TUFFC.2015.007004.
- Khan et al., "A Real-Time, GPU-Based Implementation of Aperture Domain Model
  Image REconstruction", IEEE TUFFC 68(6), 2021, doi:10.1109/TUFFC.2021.3056334.

## The method, briefly

ADMIRE works on **time-delayed (TOF-corrected) channel data**, one image line at a
time, with a sub-aperture of `n_elements` elements centered on the line.

1. Split each line along depth into **non-overlapping STFT windows**, each one
   pulse FWHM long. For a Gaussian pulse with fractional bandwidth `BW`, the FWHM is
   `8 ln2 / (2π · BW · f0)` seconds, about `0.88 / (BW f0)`. Zero-pad each window to
   twice its length and take the DFT along depth.
2. Keep the DFT bins inside `f0 ± 0.5 · 1.2 · BW · f0`; that is usually 2 or 3
   frequencies.
3. For each (window, frequency) there is a **model**: a matrix of complex
   aperture-domain signals (one row per element), one column per point-scatterer
   "predictor". A predictor has a lateral position x, a depth z and a **distance
   offset**, which is an extra transmit path length used to model multipath.
   - The **ROI model** covers predictors inside an ellipse around the window
     center, with half-axes `res_lat` and `res_axl = 2 · res_lat`.
   - The **outer model** covers predictors outside a slightly larger ellipse: a
     lateral range of ±(half the sub-aperture + 1 mm), depths 0 to 1.05 · z_c, and
     offsets from −8 mm to +3.2 mm.
4. **ICA reduction** (`ica=True`, the reference default and the only option on
   GPU): FOBI ICA on each model's columns, then `pinv(W)`. This replaces each
   model with an `n_el × n_el` basis. Without it, deep windows have about 300k
   outer predictors.
5. Concatenate [ROI, outer], normalize the columns to unit norm, and convert to
   real form: `[[Re, -Im], [Im, Re]]`, with observations `y = [Re; Im]`.
6. Fit with **elastic-net regression solved by cyclic coordinate descent**:
   `alpha = 0.9`, `λ = 0.0189 · rms(y)`. y is divided by its standard deviation,
   and λ by the same, before the fit. Tolerance is 0.1 on the largest squared
   coefficient change divided by N.
7. Reconstruct the signal from the **ROI coefficients only**, put it back in the
   selected bins (RF data also gets the conjugates in the negative bins), inverse
   DFT, and keep the first window-length samples. Bins outside the band end up as
   zero, so ADMIRE also acts as a band-pass filter.
8. Sum the elements (as in DAS), then envelope detection and log compression.

**Aperture growth:** each window uses the central
`n = 2·ceil(ceil(z_c / pitch / F) / 2)` elements, clamped to
`[min_num_elements (16), n_el]`. The model is generated with only those rows. In
zea the other rows are zero-padded and masked out; N, the mean, the standard
deviation and the RMS are computed over the masked-in rows only.

**The predictor signal** (`_predictor_signals`, a port of
`generate_modeled_signal_for_predictor.m`) is the product of three terms:
- element directivity, `sinc · cos θ`;
- a window amplitude: the square root of the fraction of a Gaussian pulse centered
  at `z_distance` that falls inside the window (analytic erf integral);
- the phase `exp(j · k · z_distance)`.

Here `z_distance` is twice the depth at which the echo appears on each element
after dynamic receive focusing. The transmit path is modeled as
`offset + 2 z_c − z`, not geometrically.

There are empirical constants: `cal_shift`, `distance_offset_shift`, and a
**wavenumber calibration table indexed by the exact f0 and the selected-frequency
index**. The table only has 4 values, so more than 4 selected frequencies raises an
error. These are kept as defaults in `ADMIREConfig`; `wavenumber_calibration=None`
turns calibration off and made little difference in tests.

## Design decisions in the zea port

- **Split between offline and online work.** `generate_admire_models(depths,
  sound_speed, center_frequency, pitch, n_elements, f_number, analytic, config)`
  returns `ADMIREModels`, holding zero-padded complex64 models
  `(W, F, n_el, P_max)`, `roi_mask`, `aperture_mask`, `window_starts`, bins and
  frequencies. It is wrapped in `zea.internal.cache.cache_output`, so models are
  pickled to `~/.cache/zea/cached_funcs` and keyed on the arguments and the source
  code. `apply_admire(channel_data (n_z, n_lines, n_el, n_ch), models)` is pure
  keras ops.
- **DFT as matrix products** (einsum against cos/sin matrices restricted to the
  selected bins), in real arithmetic only. This avoids complex FFT, scatter and
  backend differences.
- **The solver** `elastic_net_ccd(X (...,N,P), y (...,B,N), row_mask)` is batched:
  each (window, frequency) design matrix is shared by all image lines, which are
  the B right-hand sides. It uses `ops.while_loop` for the sweeps around
  `ops.fori_loop` over the coordinates, with `ops.slice_update`. Convergence is
  global: the loop continues until the *largest* change over the whole batch is
  below tolerance. Zero padding is harmless: zero columns get β = 0, and masked
  rows contribute nothing.
- **IQ data is supported natively.** zea's `tof_correction` phase-rotates IQ data,
  so the aligned IQ data is the analytic signal *with the carrier* sampled along
  depth. This was verified: the spectrum peaks at +f0 (mod `c/(2dz)`). The axial
  sampling rate is `fs_eff = c / (2 dz)` of the **grid**, not the ADC rate. For IQ
  data, bins are mapped to their alias nearest f0, which only requires
  `fs_eff > fitted bandwidth`. RF data requires `f_max < fs_eff / 2`. Both checks
  raise a clear error.
- **The operation (`ops.ADMIRE`):**
  - It sums over transmits first, so ADMIRE runs on the compounded channel data.
    This is a choice; the alternative is per-transmit ADMIRE followed by
    compounding.
  - It reshapes to the grid, then gathers for each column the sub-aperture whose
    center is nearest the column. Elements outside the array are set to zero via a
    padded zero channel.
  - The operation is **not jittable** (`jittable=False`): it needs concrete grid
    and probe values to build the models and the gather indices. Instead it
    caches a `zea.backend.jit`-compiled fit per geometry, in `_fit_cache`.
  - Aperture growth uses the pipeline's `f_number` (zea default 1.0; reference 2).
  - **`Demodulate` sets `center_frequency=0`**, so for IQ data the op falls back to
    `demodulation_frequency`.
- **Serialization.** `config` is stored as a dict, and lists are turned back into
  tuples (JSON round-trips turn tuples into lists, and the cache key needs them
  hashable). `get_dict` drops precomputed `models`.

## Validation done (CPU, JAX; TF for the unit tests)

- The CCD solver matches a line-by-line NumPy port of `ccd_double_precision.c` to
  about 1e-5, including with masked or padded rows.
- The chunked FOBI basis matches a port of `ica.m` up to column order and phase.
- **Synthetic point targets** (33 elements, 5 MHz, BW 0.6, λ/8 grid): the energy
  ADMIRE keeps relative to DAS is about 0.55 for an on-axis target, 0.19 for a
  target 2 mm off-axis and 0.09 at 4 mm. Multipath with a +1 mm offset is not
  separated with a small aperture (≈ 0.54), which is expected since the wavefront
  curvature is nearly identical.
- **zea simulator** (`ops.Simulate` → `Demodulate` → `Beamform(admire)`): an
  anechoic cyst in speckle, 64 elements, 0.3 mm pitch, 0° plane wave,
  `n_elements=33`, F = 2. Contrast goes from −15.4 dB (DAS) to −22.2 dB (ADMIRE);
  CNR goes from 1.41 to 1.08. Visible horizontal banding at window boundaries comes
  from the reference's non-overlapping rectangular windows.
- A stress test with clutter 20 dB brighter, placed at 7–8 mm lateral (*outside*
  the outer model's lateral range for a 33-element aperture), gave poor images.
  Strong clutter must lie within ±(sub-aperture/2 + 1 mm), or the model can't
  attribute it. Use larger `n_elements` (the reference uses 128).
- `tests/test_admire.py`: 16 pass on JAX and on TF. The 2 cross-backend equality
  tests and **torch were not run** (the torch index was blocked in that
  environment).

## Performance numbers (CPU, 4 cores)

- Model generation: about 1 s per window at 1 cm depth and about 6 s per window
  at 3 cm, for 64 elements. A 14–22 mm image with 33 elements (34 windows) took
  about 48 s. Most of the time goes into `_predictor_signals` over the large outer
  grid (300k predictors × n_el). The complex `exp` was already replaced by
  cos/sin, which was 5× faster. Chunks are kept in memory between the two ICA
  passes when they fit within `max_cache_bytes`.
- The fit is fast at these sizes (seconds). On a GPU with 128 elements, expect
  `X` of shape (W, F, 256, 512) in float32 to take about 440 MB for 280 windows;
  use `window_batch_size` to bound memory.

## Suggested next steps (GPU / Hugging Face environment)

1. **Run on real data.** Load a zea dataset from Hugging Face with a linear array
   (plane wave or focused). Use a grid with `dz ≈ λ/8 … λ/4` (IQ data is fine) and
   columns on element positions. Then build the pipeline explicitly, because
   `Pipeline.from_default` doesn't forward beamformer kwargs:
   ```python
   from zea import ops

   pipe = ops.Pipeline(
       [
           ops.Demodulate(),  # only if the data is RF
           ops.Beamform(beamformer="admire", num_patches=1, n_elements=65, window_batch_size=32),
           ops.EnvelopeDetect(),
           ops.Normalize(),
           ops.LogCompress(),
       ],
       jit_options=None,
   )
   ```
   Compare against `delay_and_sum` with contrast and CNR on cysts and a gCNR
   metric if available.
2. **Test the torch backend.** Run `pytest tests/test_admire.py` with all backends.
   Watch `elastic_net_ccd`: on torch, `fori_loop` and `while_loop` are Python
   loops, which work but are slow. Also run the light suite:
   `pytest -m 'not heavy and not notebook'`.
3. **Move model generation to the GPU.** Rewrite `_predictor_signals` and the two
   FOBI accumulations (`Σ x xᴴ`, then `(Σ|x_w|²-like · x_w) x_wᴴ`) in keras ops or
   jax in float32, keeping `eigh`, `svd` and `pinv` in float64 NumPy. This is the
   main bottleneck. Compare against the NumPy path with the existing tests.
4. **Speed up the solver.** CCD is sequential over 4·n_el coordinates. Options:
   stop per problem instead of globally (mask converged problems), or add a FISTA
   or proximal-gradient solver for the same objective
   (step `1/(σ_max(X)²/N + λ(1−α))`, which can be precomputed per model) as a
   `solver=` option. Validate against CCD on the objective value, not on β.
5. **Remove the window-boundary banding** with overlapping windows and
   overlap-add (Hann or Tukey window, 50% overlap). This is a departure from the
   reference release, which forces overlap = 0, so make it opt-in. `apply_admire`
   currently asserts contiguous windows.
6. Possibly add a docs notebook under `docs/source/notebooks/pipeline/`
   comparing DAS, MV and ADMIRE on a Hugging Face dataset.

## Gotchas

- `cache_output` caches generated models on disk. If you change model-generation
  code that isn't reached by a direct call from `generate_admire_models`, clear
  the cache with `zea.internal.cache.clear_cache("generate_admire_models")`.
- Floating-point details matter for aperture growth: `6e-3 / 0.3e-3 / 2` is
  10.000000000000002, so ceil gives 11, then 12 elements (MATLAB behaves the same).
- `Pipeline` sets `with_batch_dim=True` on its operations. When calling
  `ops.ADMIRE` directly, pass data as `(batch, n_tx, n_pix, n_el, n_ch)` or set
  `with_batch_dim=False`.
- The generic `test_beamformers` in `tests/test_operations.py` skips `"admire"`
  on purpose, because it needs a real geometry.
