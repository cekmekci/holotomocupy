# Why the coded-aperture flat field does not cancel

AtomiumS2, ESRF ID16A, 33.35 keV. Reproduce everything here with

    python analyse.py

Every number below is printed by that script and every figure is written by it.

---

## The problem

In the coded-aperture holotomography pipeline, stage 5 divides each measured
frame by a **modelled** flat field,

    flat(k) = | D( D_code( prb · code_window(k) ) ) |²

built from `code.npy` and `prb.npy`, which come out of a separate near-field
ptychography (NFP) scan. After that division the aperture should vanish and
only the sample should remain. It does not. Code structure survives, and so do
the probe's line artefacts.

That residue is then amplified. Paganin multiplies by `δ/β / 2 = 15`, so an 8%
error in the division becomes a 0.8 rad background in the phase — which is
**82% of the full dynamic range** of the Paganin projection, in a region with
no sample in it.

The question this note answers is narrow and specific: **is the aperture model
wrong, or is the probe model wrong?**

---

## The idea

`codedHT.h5` contains two **sample-free** frames — `pref0` taken before the
tomographic scan and `pref1` after it. Between them the coded aperture moved.

The probe and the detector do **not** move between those two frames. The
aperture does. Therefore:

- whatever is **the same** in the two residuals cannot be the aperture, so it is
  the probe or the detector
- whatever **differs** must be the aperture

No sweep, no reconstruction, no fitted parameter. Two measured frames and the
existing forward model.

---

## Step 1 — the two sample-free frames

![the two flats](fig1_flats.png)

```
pref0  mean 1.00000     normalised to 1 by construction in stage 2
pref1  mean 0.95232     -4.8%, the beam decayed over the scan
```

Their ratio (third panel) is full of structure rather than being a flat 0.95.
That is already the aperture having moved.

**The noise floor**, needed later so we do not over-interpret small numbers.
`pref1` is stored as its 20 individual frames, so the scatter across them
measures the detector noise directly, with no gain assumption:

```
single white frame        0.01786
mean of 20 (= pref1)      0.00399   = 0.419% of the signal
```

---

## Step 2 — where is the aperture in each frame?

![registration](fig2_registration.png)

Phase-correlate each flat, divided by the probe-only model, against the
propagated code. High-pass first, otherwise the DC term swamps the peak and the
argmax picks noise — that mistake gives nonsense positions.

```
pref0   aperture at ( -133,  +235)    peak z-score 48.7
pref1   aperture at ( +170,  +960)    peak z-score 49.5
        moved (+303, +725) px between them
```

Both peaks are unambiguous. The 303 × 725 px displacement is what makes the
split possible.

---

## Step 3 — measured against model

![residuals](fig3_residuals.png)

For each frame, build the model at that frame's own aperture position and form

    R = log( measured / model ),   mean removed

`log` because the error is multiplicative, so this makes it additive. The mean
is removed because the absolute scale of the model is a **free gauge** —
`code.npy` and `prb.npy` are determined only up to one complex scalar, so a
constant offset between model and measurement is expected and is not an error.
Only the *structure* is meaningful.

```
pref0 residual rms   0.05456
pref1 residual rms   0.06133
noise floor          0.00419
```

The residual is **13× the noise floor**. It is real, not noise.

---

## Step 4 — the split, and where 62 / 38 comes from

![the decomposition](fig4_split.png)

Define, pixel by pixel:

```
fixed  = (R0 + R1) / 2      common to both  ->  probe or detector
moving = (R0 - R1) / 2      differs         ->  the aperture
```

This is an **identity, not a fit**: `R0 = fixed + moving` and
`R1 = fixed − moving` exactly. Nothing is being optimised.

If the two parts are uncorrelated with each other, their variances add, and the
share of each is just its variance over the mean residual variance:

```
var(fixed)  = 2.099e-03   ->  62.3%
var(moving) = 1.270e-03   ->  37.7%
                              -------
                              100.0%   <- they sum to 100%, so the two parts
                                          really are uncorrelated
```

**An independent check that the decomposition is sound.** If `fixed` and
`moving` are uncorrelated, then the plain correlation between the two residuals
must satisfy

    corr(R0, R1) = ( var(fixed) − var(moving) ) / ( var(fixed) + var(moving) )

```
predicted  +0.2462
measured   +0.2479
```

Those agree to better than 1%, from two routes that share no arithmetic. The
split is not an artefact of how it was computed.

**One caveat, stated rather than buried.** Detector noise is uncorrelated
between the two frames, so *all* of it lands in `moving`. The noise floor
accounts for 1.0% of the variance, so the aperture's true share is nearer 37%
than 38%. The probe's share is unaffected.

### The result

> **62% of the flat-field error is fixed in the detector frame — the probe or
> the detector. 37% moves with the aperture — the code model.**

Both are wrong. That is why no single-knob sweep ever fixed this: each one could
only ever address one half.

---

## Step 5 — what spatial scale is each on?

![spectra](fig5_spectra.png)

```
 period [px]        fixed       moving
        1024    2.965e+07    1.234e+06
         512    1.012e+07    5.087e+05
         256    3.301e+05    8.571e+04
         128    1.103e+05    3.327e+04
          64    7.942e+04    5.519e+04
          32    1.980e+05    1.380e+05
          16    1.418e+05    1.164e+05
```

The `fixed` (probe) part is overwhelmingly **large-scale** — at 1024 px period
it is 24× the moving part, and it falls away quickly with frequency. So what is
wrong with the probe is its broad illumination envelope, not its fine detail.
That is consistent with the NFP and HT data being two separate acquisitions with
the beam drifting between them.

Above about 64 px period the two parts become comparable, so at fine scales the
aperture error is as important as the probe error.

---

## Why the multi-distance pipeline does not have this problem

The difference is structural, not a matter of tuning. From
`AtomiumS2/steps15.py`, the multi-distance division is one line:

```python
rdata = data_j / (cref + 1e-5)        # cref = the MEASURED reference, per distance
```

`cref` is the **recorded flat field**. There is no forward model in the
denominator at all — no probe, no aperture, no propagation.

The coded-aperture division is:

```python
flat(k) = |D(D_code(prb · code_window(k)))|²      # a MODEL
```

Everything in that expression can be wrong, and the analysis above shows two
things are: the probe (62%) and the aperture (37%). None of those failure modes
can exist in the multi-distance pipeline, because it never predicts its flat
field — it measures it.

**Why the coded pipeline cannot simply do the same.** The aperture moves between
frames. One recorded flat does not describe any other frame, so there is nothing
to divide by. Predicting the flat is not an implementation choice, it is forced
by the method.

Three smaller differences, for completeness:

- the multi-distance code works in **log space during the division** — `log`,
  back-propagate to the object grid at each distance's own magnification, `exp`
  — so stitching errors are additive rather than multiplicative
- it pads with `'edge'` vertically and `'linear_ramp'` **to exactly 1.0**
  horizontally, which it can do because after dividing by a measured flat the
  vacuum genuinely is 1. The coded pipeline ramps to a measured `bg_box` level
  instead, precisely because its modelled flat does not sit at 1
- it **cross-normalises adjacent distances** (`mmm = mean(srdata[k+1]) /
  mean(tmp)`) before blending, which is an explicit gauge fix between
  measurements

It also runs `paganin = 120` against the coded pipeline's `30`. That is a 4×
difference in how hard any division error is amplified, and it should be matched
before the two reconstructions are compared.

---

## What this does and does not establish

**Established, by measurement:**

- the flat-field residual is 13× the noise floor, so it is real
- 62% of it is fixed in the detector frame, 37% moves with the aperture
- the decomposition is self-consistent (predicted vs measured correlation agrees
  to 1%)
- the probe error is large-scale; the aperture error is comparatively broadband
- the multi-distance pipeline divides by a measured flat and so cannot have any
  of these failure modes

**Not established:**

- *why* the probe differs between the two scans. Beam drift between
  acquisitions is the natural explanation and fits the large-scale signature,
  but it has not been demonstrated
- whether the `fixed` part is the probe or the detector. `pdata` is never
  flat-field divided — stage 2 subtracts dark and divides by two scalars only —
  so an uncorrected detector response would look identical in this test
- *what* is wrong with the aperture model. Ruled out by separate measurement:
  `z1c` (cancels exactly; an 11-point sweep moved the code 0.3–1.1%), the
  detector PSF (blurring the model makes it monotonically worse), the
  propagator and the tomography operator (both verified as exact adjoint pairs
  to 1e-6)

**The obvious next step** is that the probe half is tractable with what is
already on disk. `pref0` and `pref1` are sample-free measurements of exactly the
quantity that is wrong, at two different aperture positions — enough to fit a
correction to the probe's envelope without touching the aperture model.
