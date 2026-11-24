# Detector noise characterisation — AtomiumS2 coded-aperture HT and NFP

Measured from the raw arrays only (`/exchange/dark`, `data_white0`, `data_white1`,
`data_dark`, `data_white`). `pdata` and `pref0` were not touched. All CPU, no GPU,
no MPI. Nothing in `/data3` was modified.

Every statement below is tagged **[M]** measured here, or **[I]** inferred.

---

## 1. Summary table

| quantity | HT white0 | HT white1 | NFP white | tag |
|---|---|---|---|---|
| dark offset (ADU) | 1226.2 | — | 1225.5 | M |
| dark fixed pattern, pixel-to-pixel offset spread (ADU) | 52.8 | — | 53.2 | M |
| read noise, temporal (ADU) | 12.14 ± 0.19 | — | 12.08 ± 0.12 | M |
| PTC slope at operating level, `g_eff` (ADU/quantum/px) | 2.033 ± 0.050 | 2.140 ± 0.115 | 1.959 ± 0.017 | M |
| N_eff, pixels per quantum | 2.55 ± 0.20 | 2.65 ± 0.29 | 2.373 ± 0.043 | M |
| ρ(1,0), nearest-neighbour noise correlation | 0.187 ± 0.004 | 0.191 ± 0.013 | 0.1756 ± 0.0008 | M |
| autocorrelation FWHM (px) | 1.23 | 1.24 | 1.22 | M |
| **true gain `g = g_eff × N_eff`** (ADU per noise-equivalent quantum) | **5.19** | **5.67** | **4.65** | M |
| operating signal level (ADU, dark-subtracted) | 5176 | — | 7207 | M |
| noise-equivalent quanta per pixel | ~997 | — | ~1113 | M |
| bad pixels (excluding 32 px border) | 12562 (0.30%) | — | 12697 (0.30%) | M |

**Gain: 5.2 ± 0.5 ADU per noise-equivalent quantum.** [M]

### Does the quoted 5.6 agree?

**Yes, within uncertainty — but only after the correlation correction.** [M] The
naive per-pixel photon-transfer slope is **2.0**, not 5.6. It only becomes ~5.2
once multiplied by N_eff ≈ 2.4. So if any code is using 5.6 ADU/quantum as a
*per-pixel* variance scale (`var = 5.6 × I`), it is overestimating the per-pixel
noise variance by ~2.7×, i.e. the rms by ~1.6×. The number that reproduces the
measured per-pixel variance is **g_eff ≈ 2.0**.

I could not find where 5.6 came from, so this is a guess: [I] using HT white0
frames 0–2, which I show below are contaminated, gives g_eff ≈ 2.37, and
2.37 × 2.4 = 5.7. That would reproduce 5.6 almost exactly.

### Agreement with the earlier October analysis

After finishing the above I found the earlier write-up this figure came from
(`Desktop/coded_paper/noise_model_analysis.md`). It is an independent analysis
and it corroborates this one closely, which is worth more than either alone:

| | October | here |
|---|---|---|
| per-pixel PTC slope | 2.22 | 2.0 ± 0.1 |
| variance amplification / N_eff | 2.55 (Σp² = 0.39) | 2.4 ± 0.2 |
| gain | 5.6 | 5.2 ± 0.5 |
| read noise | 151 ADU² | 147 ADU² (12.1 ADU) |
| ρ(1) | 0.21 | 0.18 |
| PSF FWHM | 1.1–1.3 px | 1.22 px |

Same structure, same conclusion on amplitude fitting, values within ~10%. [I]
The one systematic difference — the October per-pixel slope of 2.22 against 2.0
here — is most likely the contaminated HT white frames, which inflate the slope
and which that pass had no reason to exclude. Two things here are new: the PTC
curvature (§4) and the frame instability (§6).

---

## 2. Method, and the two places textbook EMVA 1288 does not apply directly

Photon transfer from frame differences `(f_i - f_j)/sqrt(2)`, binned by mean
level, robust (interquartile) variance per bin. EMVA 1288 §6. Two deviations
were necessary, both validated on synthetic data with planted values
(`validate.py`):

**(a) The illumination is not flat.** The whites carry the coded aperture and
probe speckle, with |∇I| ≈ 922 ADU/px. A sub-pixel drift δ between two frames
injects δ·∇I into the difference and is read as noise; a drift of only 0.09 px
would have accounted for the *entire* measured noise. Rather than Fourier-shift
the frames to register them — which would low-pass the noise and corrupt both
the PTC amplitude and the NPS — each difference tile is projected out of the span
of {∂I/∂x, ∂I/∂y, I, 1} built from the stack mean. This costs 4 degrees of
freedom per 65536 pixels and leaves the noise untouched. Validated: recovers a
planted 0.02 px drift and restores the gain to 1.8% after a deliberate 12%
inflation. The real drift turns out to be only 0.005–0.008 px, so this correction
changes the answer by <1%; it was needed to *prove* drift was not the story.

**(b) The PTC intercept cannot measure read noise here.** At ~5000 ADU the read
variance is 1.5% of the total, so the intercept is set by sub-percent slope
error. In synthetic case A, with read noise planted at exactly 12.0 ADU, the
intercept returned 5.97 ADU; on the real data it returns a *negative* variance.
Read noise is therefore taken from the dark frames (EMVA 1288 §4), where frame
differencing cancels the fixed pattern. **The dark and PTC values agree**: the
dark gives 12.14 ADU, and the PTC fit floored at the dark value reproduces the
measured residual width to 0.1% (see §5).

N_eff is computed as the sum of the normalised noise autocorrelation over all
lags, with the dark NPS subtracted first to remove read-noise dilution. For
quanta spread over a normalised PSF p, `C(d) = g²λ(p⋆p)(d)`, so
`Σ_d ρ(d) = 1/Σp² = N_eff`. Validated: recovers planted values of 4.07 and 10.17
to within 2%, including with bad pixels present and at 512-px tiles.

---

## 3. Correlation — the noise is NOT independent per pixel [M]

The note was right that the noise is correlated, and this is the single most
consequential result.

- ρ(1,0) = **0.18**, ρ(2,0) = 0.021, ρ(3,0) = 0.004, and consistent with zero
  beyond lag 4 (|ρ| < 5e-4 out to lag 64, checked on 512-px tiles).
- Autocorrelation FWHM **1.22 px**. The correlation is short-range — essentially
  nearest-neighbour only.
- **N_eff = 2.4** (range 2.37–2.65 across the three stacks). *This is the factor
  by which a per-pixel variance understates the quanta*, which is what you asked
  for.
- The NPS falls by 3.4× from low frequency to Nyquist. [I] That is an optical
  MTF rolloff, consistent with a scintillator coupled through optics.

Practical consequence: a per-pixel Poisson model using the true gain 5.2 predicts
3.17% rms at the operating level, but the actual per-pixel rms is **1.98%**. The
noise looks 1.6× *better* than Poisson per pixel. That is not extra information —
it is the same quanta smeared over 2.4 px.

**Binning does not average this down as 1/b.** Measured directly:

| bin | measured rel rms | if noise were independent | ratio |
|---|---|---|---|
| 1 | 1.837% | 1.837% | 1.00 |
| 2 | 1.103% | 0.919% | 1.20 |
| 3 | 0.794% | 0.612% | 1.30 |
| 4 | 0.621% | 0.459% | 1.35 |
| 8 | 0.333% | 0.230% | 1.45 |

The ratio saturates towards √N_eff ≈ 1.6, as it should.

### Detector identity

**Not recorded.** [M] Both HDF5 files have zero attributes; no detector name
appears anywhere in the file or the pipeline, which carries only geometry. So I
cannot confirm the hardware. What the data say: 16-bit (saturation at 65535),
3.04 µm pixel, 2048², offset 1226 ADU, and noise correlated over 2.4 px with an
optical-looking MTF rolloff. [I] That combination is consistent with a
scintillator coupled by magnifying optics to a camera, and is *not* consistent
with direct detection, which would give N_eff ≈ 1.

---

## 4. The PTC is curved [M]

`var/mean` rises monotonically from 1.85 at 1980 ADU to 2.41 at 11600 ADU. This
survives restriction to low-gradient pixels, so it is not a code-edge or drift
artefact. It is well described by a small negative offset:

> **Var[I] ≈ g_eff · (I − I₀) + σ_read²,  g_eff ≈ 2.0 ADU, I₀ ≈ 415 (HT) / 483 (NFP) ADU, σ_read = 12.1 ADU**

That is the variance model I would actually use. At the operating level the
offset reduces the variance ~8% below a purely proportional model.

**Cause not established.** [I] A component of the signal that carries mean but
little per-pixel variance would do this — veiling glare, scintillator afterglow,
or a mild response nonlinearity. I looked for a long-range glare tail in the
autocorrelation and found none out to 64 px, which disfavours (but does not
exclude, since glare could be broader than 64 px) the glare explanation.

---

## 5. Distribution [M]

Difference image normalised by the predicted local σ, using the quadratic PTC
model and a read-noise floor:

- **Gaussian core.** Core (|z|<4) excess kurtosis −0.01, skew ≈ 0. The predicted
  σ is right to 0.1% (IQR-sigma of z = 1.000). The model is accurate.
- Fraction beyond 3σ: 0.00274–0.00277, vs Gaussian 0.00270. No meaningful excess.
- **Heavy tail beyond 5σ**: 1.2–1.7e-5, vs Gaussian 5.7e-7 — a 20–30× excess.
  Beyond 8σ: ~1.1e-5. Max |z| ≈ 44.
- That tail is ~45–70 pixels per 2048² frame.

So: Gaussian to ±4.5σ, then a sparse heavy tail. **Reject it, do not model it.**
A simple |z| > 5 cut on a frame difference removes it and costs 1e-5 of the data.
Note the mean level matters — a straight-line PTC extrapolated into the
code-shadowed pixels passes through zero variance and manufactures spurious
outliers; that is an artefact, not a hot pixel.

---

## 6. Stationarity [M]

**Across the chip:** gain varies only mildly. 512-px tile slopes span 1.93–2.14
(HT, σ = 0.053) and 1.73–2.13 (NFP, σ = 0.095), i.e. ±3–5% about the mean. One
NFP tile sits low at 1.73. Treating the gain as uniform across the chip is good
to about 5%.

**The read noise is not white, and this is a real non-uniformity.** Dark
row-means scatter 5.5× more than white noise allows, column-means 3.6× (NFP:
4.9× and 7.3×), and the dark NPS rises ~9× at the lowest frequency. There is
common-mode / banding structure in the readout. It is small in absolute terms
(12 ADU total read noise against ~98 ADU photon noise) so it does not matter at
the operating level, but it would matter in any near-dark region.

**Over the scan:** white0 g_eff = 2.033, white1 = 2.140, a +5% drift, alongside
the known 4.8% flux drop. [I] I would not read this as a genuine gain change —
white1 also has more residual frame instability, which biases the slope upward.
Call it stable to ~5% over the scan.

**Frame-level instability, and this one needs acting on.** In HT white0, frames
0 and 2 are bad; in HT white1, frame 0. Pairs involving them give gains inflated
up to 2.63 against 2.03 for the rest. These are not noisier in a photon sense:
the excess power is concentrated at *low* spatial frequency (1.3–1.75× below
f = 0.2 cyc/px, only 1.05× at Nyquist), whereas a true gain change would scale the
NPS uniformly. [I] That signature is illumination instability — beam settling at
the start of the white series. **Recommendation: drop the first 2–3 frames of
each HT white stack when building the reference.** The NFP white stack shows no
such effect (per-frame slopes 1.949–1.971, spread ±0.017).

---

## 7. HT vs NFP — do they agree? [M]

Close, but not identical.

- read noise: 12.14 vs 12.08 ADU — **identical**.
- dark offset, FPN: 1226/52.8 vs 1225/53.2 — **identical**. Same detector, same
  settings.
- g_eff: 2.033 vs 1.959 — **3.6% apart**.
- N_eff: 2.55 vs 2.373 — 7% apart, though the HT value is the less reliable one
  (spread ±0.20 vs ±0.04) because of the frame instability.
- gain: 5.19 vs 4.65 — 11% apart.

**Verdict: one noise model for both is defensible, but not at the 5% level.** The
honest common model is `g_eff = 2.0 ± 0.1 ADU`, `N_eff = 2.4 ± 0.2`,
`σ_read = 12.1 ADU`. If you need better than 10% on the gain, use the per-dataset
values, and prefer the NFP number — that stack is clean, the HT one is not.

---

## 8. The two derived numbers

### (a) Noise floor on a flat-field division at the HT signal level

Measured two ways, model-based and model-free, and they agree.

| case | relative rms |
|---|---|
| single frame, photon+read only | 1.90% |
| data frame ÷ one white frame | 2.69% |
| **data frame ÷ mean of 20 whites (the pipeline case)** | **~1.95–2.08%** |

The model-free check (forming an actual ratio of real frames and measuring its
rms) gives 2.692% for a single-frame ratio, implying 1.904% per frame, against
1.90% predicted — exact agreement. Frame ÷ mean-of-19 measures 2.075% against
1.953% predicted; the small excess is because the 19-frame mean still contains
the unstable frames 0 and 2.

**Call the noise floor 2.0% rms per raw pixel.**

**Against your 6.3% residual:** detector noise accounts for
(2.0/6.3)² ≈ **10% of the residual variance**, or about a third of it in
amplitude. **~90% of that residual variance is not detector noise.** This
conclusion is robust to how the residual was computed — if the 6.3% refers to
binned data, the noise floor drops further (1.14% at bin 2, 0.82% at bin 3) and
the gap only widens. Whatever is behind the 6.3% is model error, not photon
statistics. [M] for the floor, [I] for the attribution of the remainder.

### (b) Poisson (amplitude) or Gaussian (intensity) misfit?

**Poisson / amplitude is the maximum-likelihood choice here.** [M]

The deciding measurement is that the variance is signal-proportional, not
constant: photon variance exceeds read variance by a factor of **66** at the
operating level (read noise is 1.5% of the total variance). A Gaussian/intensity
misfit assumes constant variance and is the ML estimator only in the
read-noise-dominated regime. You are nowhere near it — you would need to be below
~250 ADU, which only the deepest code shadows approach. The distribution work in
§5 also shows the noise is Gaussian-cored with variance tracking the PTC to 0.1%,
and a Gaussian with *signal-proportional* variance is equivalent to Poisson at
~1000 quanta. Following Thibault & Guizar-Sicairos (2012), the amplitude metric
Σ(√I_meas − √I_model)² is the variance-stabilised Gauss–Newton form of the
Poisson likelihood, so it is the right one.

Two qualifications, both of which follow from the measurements above:

1. **Use g_eff ≈ 2.0, not 5.2, to convert ADU to counts for the per-pixel
   weight.** Counts `n = I_ADU / 2.0` reproduce the measured per-pixel variance.
   Using the true gain 5.2 would overestimate the noise rms by 1.6×.
2. **The per-pixel likelihood overcounts independent measurements by N_eff ≈ 2.4**,
   because neighbouring residuals are correlated (ρ = 0.18). Consequences: the
   data term is effectively weighted ~2.4× too strongly against any regulariser,
   and χ²/dof and any error bars derived from it are wrong by that factor. The
   cheap fix is to scale the data-term weight by 1/N_eff; the correct fix is to
   pre-whiten with the measured NPS. [I] This is a consequence of the measured
   correlation, not something I tested on your solver.

Caveat on what "quantum" means: the PTC measures ADU per *noise-equivalent*
quantum. For a scintillator with Swank factor < 1, ADU per actually-absorbed
X-ray is ≤ 5.2, and the true absorbed-photon count per pixel is ≥ ~1000. This
does not change (a) or (b), which depend only on the measured variance.

---

## 9. Files, and how to reproduce

```
noise_char/
  detector_noise.py    estimators: masking, drift removal, PTC, NPS, N_eff, distribution
  validate.py          synthetic validation with planted gain/read/PSF  (run this first)
  diagnose.py          PTC curvature, per-pair scatter, tile-size sensitivity
  diagnose2.py         all-pairs gain matrix, NPS of contaminated vs clean pairs, dark structure
  final_analysis.py    the numbers in this report -> out/final_results.json, out/noise_report.png
  check_flatfield.py   model-free check of (a), and the binning table
```

```bash
export CONDA_PREFIX=/home/beams0/CEKMEKCI/miniforge3/envs/holotomocupy
P=$CONDA_PREFIX/bin/python
cd noise_char
$P validate.py          # ~2 min, confirms the estimators recover planted values
$P final_analysis.py    # ~40 min, both datasets
$P check_flatfield.py   # ~3 min
```

`out/noise_report.png` has, clockwise: PTC, NPS, autocorrelation, per-512px-tile
gain, normalised-residual histogram against a Gaussian, per-frame gain (where the
HT frame instability and the NFP stability are both visible).

## References checked against

- EMVA 1288 — PTC (§6), dark characterisation (§4), spatial non-uniformity (§8).
  Followed, with the two documented deviations in §2.
- Janesick, *Photon Transfer: DN to λ*, SPIE 2007 — the `g = g_eff × N_eff`
  correction for charge/light spreading.
- Cunningham, *Applied Linear-Systems Theory*, SPIE Handbook of Medical Imaging
  Vol 1, 2000 — NPS normalisation, and why correlated noise breaks per-pixel
  Poisson (§3, §8b).
- Thibault & Guizar-Sicairos, New J. Phys. **14** 063004 (2012) — amplitude vs
  intensity metric (§8b).

---

## 10. The observation model this implies

Written against what the pipeline actually does: `steps15.py` stage 2 builds
`pdata = (raw − dark)/W/a_k` with `W = ref_norm_scalar = 5683 ADU` (scalar) and
`a_k = bpm4_k/bpm4_0`, and `rec_coded_mpi.py` minimises
`F0_amp = 1/N Σ W_mask (√J − √d)²` with `J = K|x|²`, K the detector PSF.

### 10.1 Raw domain

For frame k, pixel x:

> **y_k(x) = D(x) + g·(p ⊛ n_k)(x) + e_k(x)**
> `n_k(x) ~ Poisson(ν_k(x))` independent, `e_k ~ N(0, σ_r²)`

| symbol | value | tag |
|---|---|---|
| D, dark offset | 1226 ADU (per-pixel, FPN 53 ADU) | M |
| g, gain | 5.2 ± 0.5 ADU per noise-equivalent quantum | M |
| p, detector PSF | Σp² = 1/N_eff = 1/2.4; Gaussian FWHM ≈ 1.05 px | M |
| σ_r, read noise | 12.1 ADU (+ weak row/col common mode) | M |

The Poisson draw happens **before** the blur. That is the whole point: it is why
the per-pixel variance is `g·Σp²·μ = g_eff·μ` with `g_eff = 2.03`, not `g·μ`.

Note `p` here is consistent with the `K` already in the forward model: N_eff =
2.4 implies FWHM 1.02–1.06 px against the ~1.1 px kernel in use. **The same
kernel accounts for the signal blur and the noise correlation.** [M]

### 10.2 In pdata units (what the solver sees)

> **E[d_k] = m_k = K|x_k|² + b**
> **Var[d_k] = (β/a_k)·(m_k − b) + σ_r²/(W a_k)²**
> **Corr[d_k(x), d_k(x')] = ρ(x−x') = (p⋆p)/Σp²,  ρ(1,0) = 0.18**

| constant | value |
|---|---|
| β = g_eff/W | 3.577e-4 |
| b = I₀/W, additive background | **0.0730** |
| σ_r²/W² | 4.563e-6 |
| N_eff | 2.4 |

Measured spread over the HT field (pdata 0.36 to 1.95): **Var[d] spans 6.3×,
Var[√d] spans only 1.16×.**

### 10.3 What to change, by impact

**1. Add the additive background — this is the one substantive model error.** [M
for b, I for its interpretation] The forward model is purely multiplicative,
`J = K|x|²`, but the data contains a component of **7.3% of the level** that
carries mean and essentially no per-pixel variance. That is what the PTC offset
I₀ measures, and a multiplicative model cannot represent it. Its effect is to
*reduce fringe contrast by 7.3%*.

Cleanest fix is one line in stage 2, since `d = (1−b)·I_true + b`:

```python
B_GLARE = 0.0730          # = I0/ref_norm_scalar, from the PTC offset
raw = (raw - B_GLARE) / (1.0 - B_GLARE)
```

applied to `pdata` **and** to `pref0`/`pref1` consistently.

Why fix b rather than fit it: with a free per-pixel probe, b is **degenerate**
against the probe amplitude on flat-field frames — `K|P|² + b` and `K|P'|²` are
indistinguishable when O = 1. It is only weakly identifiable from object frames,
through reduced contrast. The PTC offset measures it from the *noise*, which is
completely independent of image contrast, so it sidesteps the degeneracy
entirely. That is the argument for taking it from this measurement and holding
it fixed.

Caveats, stated plainly. The 7.3% is measured; reading it as diffuse glare is an
inference. I verified it is not a dark-offset error (it scales with flux: 8.0% of
level for HT, 6.7% for NFP) and not short-range glare (no autocorrelation tail
out to 64 px, so the halo must be broad). A response nonlinearity would produce
the same PTC curvature and would need a different fix; I cannot separate the two
from noise data alone. **It is close enough to your 6.3% residual to be worth
testing first** — and the test is cheap, since it is a one-line preprocessing
change.

**2. Keep the unweighted amplitude metric. Do not add per-pixel noise weights.**
[M] `Var[√d]` spans only 1.16× across the field, so unweighted L2 on amplitudes
is within **8%** of optimally weighted, in rms. Per-pixel weighting is not worth
the complexity. `Var[d]` spans 6.3×, so an intensity misfit is *not* defensible.
This confirms the existing `F0_amp` choice on measured grounds.

**3. Divide the data term by N_eff = 2.4.** [M for N_eff] Neighbouring residuals
are correlated (ρ = 0.18), so the per-pixel sum overcounts independent
measurements by 2.4×. This does not move the minimiser of an unregularised fit,
but it does change: the balance against any regulariser or prior, χ²/dof, and any
error bar or stopping criterion derived from them. If a regularisation weight was
tuned empirically it has already silently absorbed this factor — in which case
leave it and just know why.

**4. Optional, for resolution: whiten.** The amplitude transform fixes the
diagonal only. The residual NPS falls 3.4× from low frequency to Nyquist, so
high-frequency residuals are currently underweighted — exactly where the phase
information lives. Proper ML is `Σ_f |r̂(f)|²/ρ̂(f)` on the amplitude residual
`r = √J − √d`, with `ρ̂` the normalised NPS (mean 1). Because the sqrt transform
makes the variance near-stationary, this is legitimate. And `ρ̂ = |P(f)|²/⟨|P|²⟩`
is already available from the same K — no new calibration. The filter is compact
(nearest-neighbour), so a small convolution suffices.

**5. Reference handling.** Drop white0 frames **0 and 2** from `ref0_mean`
(§6, they are illumination-unstable and inflate the apparent gain by 17%).
Inflate the variance by `(1 + 1/N_white)` to account for reference noise — a
+2.8% effect on rms, small but free. Mask the 12562 bad pixels via `W_mask`
rather than patching them; a patched pixel has no noise, so any weighting applied
there is wrong.

### 10.4 What is already correct — do not change

- **The amplitude metric.** Confirmed by measurement (§10.3.2).
- **The detector PSF in the forward model.** Independently corroborated: the
  noise correlation implies the same ~1.05 px kernel.
- **The dezinger** (`remove_outliers`, threshold 0.8). Measured: it replaces
  31–35 px per frame (0.001%), which is precisely the >5σ heavy-tail population
  from §5. Only ~10% of its hits are at code edges, so it is not eating signal.
- **Clipping at zero.** Harmless here — the minimum dark-subtracted signal is
  ~218 ADU, about 10σ above zero, so the clip never fires on real data.
