# BoneNano / DanMAX — near-field ptychography (NFP) on scan 80

Recovers **CA2** (the moving coded aperture) and the **probe**, from a scan
with the masks in the beam and no sample.

This is the **standalone NFP** reconstruction — run it when you want CA2 on
its own. It is *not* what the reconstruction folders consume:
`BoneNano_coded_holotomo` carries its own copy of this stage, with a square
aperture *plane*, and reads that. See
[Relationship to `BoneNano_coded_holotomo`](#relationship-to-bonenano_coded_holotomo)
below.

The DanMAX counterpart of `AtomiumS2_coded`'s step 0. The method is the
advisor's; the data handling is ported from
`phascade/experiments/bone_nano_holotomo/step1_scan-0080_nfp`, which is where
these conventions were worked out on this detector.

---

## Relationship to `BoneNano_coded_holotomo`

This folder is the **standalone NFP** reconstruction. The holotomography
folder carries byte-identical copies of `step0.py`, `step0a_convert.ipynb` and
`step0b_shifts.ipynb` so that it stands alone too, the way `AtomiumS2_coded`
does.

```bash
diff BoneNano_nfp/step0.py BoneNano_coded_holotomo/step0.py      # should be empty
```

A non-empty diff means they have drifted. Only the configs differ: the
holotomo copy uses a **square aperture plane** (`npsiz == npsi == 8192`),
because `CodedReader` asserts `code.npy` bins to a square `(ncode, ncode)`.
The detector crop, 2048 x 4096, is the same in both. Its paths point into its
own folder.

Use this folder when you want CA2 on its own. Use the holotomo folder's copy
when you are running the full chain there.

---

## The thing to understand first: there are TWO apertures

CA1 is static and sits upstream. CA2 is the one that moves per frame. The
engines carry a **single** `code` variable, so the mapping is forced:

| physical | variable | where it ends up |
|---|---|---|
| CA1 | part of the **probe** | `prb.npy` |
| CA2 | the **code** | `code.npy` |

`prb.npy` is therefore *the probe at the CA2 plane with CA1 already imprinted
on it*. It is not a clean illumination function and is not meant to be.

Almost every subtlety below follows from this. CA1 is static, strong, and
sits in the same frames as the thing we are trying to measure.

---

## The scans

| scan | contents | used for |
|---|---|---|
| `scan-0067` | dark | dark frame |
| `scan-0068` | flat — **no CA1, no CA2** | flat field, dead-pixel mask |
| `scan-0069` | CA1 alone | *not used*; see "CA1 moved" below |
| `scan-0080` | CA1 + CA2, **no sample**, 121 positions | the NFP scan |

Frames are at `/entry/instrument/orca/data`, `(npos, 2592, 4608)` uint16.

### CA1 moved between scan 69 and scan 80

The phascade notebook measured this directly and it is **not** negligible — it
is why scan 69 is unused here. A probe built from scan 69 would carry CA1 at
the wrong place, and the solve would then have to undo a static error present
in every frame.

Instead the probe is seeded from **scan 80's own mean frame**. CA2 moves, so
it averages out of the mean; what survives is the probe with CA1 where it
actually was during scan 80.

If you ever need scan 69 — say to check CA1 independently — the phascade
notebook's cell 15 high-pass filters both and cross-correlates them.

---

## Run order

```bash
jupyter lab                 # step0a_convert.ipynb, then step0b_shifts.ipynb
mpirun -np <ngpus> ./set_affinity_gpu.sh python step0.py config_step0.conf
```

| step | does | writes |
|---|---|---|
| `step0a_convert.ipynb` | dark, flat, dead-pixel mask, crop, dezinger, scalar flat-normalise | `data_nfp.h5` — `/exchange/{pdata,pref,...}` |
| `step0b_shifts.ipynb` | scan positions by cross-correlation | `/exchange/shifts` appended |
| `step0.py` | the NFP solve, `RecNFPCoded`, MPI | `code.npy`, `prb.npy`, `pos_nfp.npy` |

**`config_step0.conf` is the only place parameters live.** Paths, geometry and
grids. The notebooks read it too. The one thing hardcoded in `step0a` is which
scan number plays which role, kept visible on purpose.

---

## The four conventions that matter

Three of these were wrong in the first draft of this folder, and each one on
its own is the difference between converging and stalling. If you change the
preprocessing, these are what to preserve.

### 1. The registration target needs TWO divisions

```python
rd = pdata / (pref + 1e-6)        # strip the probe envelope (per pixel)
rd = rd / (rd.mean(0) + 1e-6)     # strip the STATIC CA1 pattern
```

The second division is the one that is easy to miss. CA1 does not move, so
without removing it the cross-correlation peak is dominated by a pattern that
is identical in every frame, and **every measured shift is pulled toward
zero**. The symptom is a solve that drops the DC level on iteration 0 and then
goes flat.

Note this per-pixel division is only for *registration*. The data handed to
the solver is normalised by a **scalar**, see point 4.

### 2. Shifts are chained from a mid-scan anchor, and the sign is inverted

```python
step[k] = phase_corr(rd[k], rd[k+1])          # CONSECUTIVE frames
shifts  = integrate_shifts_from_anchor(step, anchor=NPOS//2)   # pos[anchor]-pos[k]
shifts  = centered_full(shifts)
```

Two separate points:

- **Chained, not direct.** Consecutive frames correlate far better than
  distant ones, so integrating from the middle keeps the longest path short.
- **The sign is inverted** relative to a raw `phase_corr`.
  `integrate_shifts_from_anchor` returns `pos[anchor] - pos[k]`, which is the
  engine's convention. Every phascade experiment negates somewhere for this
  reason; here it is inside the integration.

The notebook cross-checks chained against direct-from-anchor. They should
agree to about 1 px.

### 3. The probe is seeded, not cold-started

```python
prb = cl.cl_prop.DT(sqrt(mean_frame))      # back-propagate detector -> CA2 plane
```

`AtomiumS2_coded/step0.py` leaves `prb = 1`, but only because *its* aperture
sits upstream of the focus, so its propagation distance is negative and the
seed diverged. **That does not apply here.** This geometry has
`z1 < focusdetectordistance`, distance is +114.5 mm, and the back-propagation
is well posed. The phascade notebook does the same thing.

`psi` does start cold, `proj = 0`. Both references compute a Paganin seed for
the aperture and then discard it — the phascade notebook says plainly
"Initializing psi with 1 worked better". The dead Paganin code has been
deleted from `step0.py` rather than left commented, because the version it
carried assumed a square grid.

### 4. The flat is divided out as a SCALAR

```python
raw = flat_field_divide(raw, flat_mean)    # only flat.mean() is used
```

Not per pixel. The reference still carries CA1, so a per-pixel divide would
remove structure the forward model is supposed to reconstruct as part of the
probe. Same reasoning as `AtomiumS2_coded/step2`.

---

## Preprocessing chain

Exactly the order the phascade notebook uses:

```
dark_subtract_clip -> fill_dead -> remove_outliers -> flat_field_divide -> bin
```

All four helpers are **inlined** in `step0a` — they are ports of
`phascade.preprocess`, and holotomocupy has no equivalents.

Two details that bite:

- **`remove_outliers` uses a `radius x radius` window, not `2*radius+1`.** So
  `dezinger_radius=3` is a 3x3 filter. Getting this wrong quietly over-smooths
  the mask speckle, which is the entire signal here.
- **The dead-pixel mask comes from the flat** (`flat < 0.05 * median`), and is
  applied to the flat *and* every frame. This detector has dead rows and
  columns; the notebook prints them. Leaving them in means the solver fits
  detector failure as signal.

---

## Grids

**Rectangular.** The orca is 2592 x 4608, so `nz != n` everywhere. `nz`/`npsiz`
are rows, `n`/`npsi` are columns. `parse_args_coded_step0` defaults both to the
square value, so the ESRF configs are unaffected.

**Crop only, no binning** (`bin=0`), a centred 2048 x 4096. Consequences:

- `pdata` is ~4 GB, both in RAM during `step0a` and on disk
- the dezinger runs on full frames
- `nchunk` is down to 2

Raise `bin` for a quick look; the notebooks follow the config automatically.

### `npsiz` / `npsi` must be re-measured whenever the shifts change

The aperture plane must clear `nz`/`n` + 2*max|shift| + 2*`fov_margin`, or
`step0.py`'s FOV filter **silently discards** the outermost positions.
`step0b` prints the minimum it measured. Put that number in the config.

Measured on scan 80: **max|shift| = 492 px in both axes**, so the minimum is
3065 x 5112. The config uses **4096 x 8192**, comfortably above, and all 121
positions survive the FOV filter.

This is not a one-off: the measured shifts depend on the preprocessing, so a
change to the registration changes the required plane. The bin-3 draft of this
folder measured 62 px — the same walk through an 8x coarser grid.

---

## Diagnostics, and what a bad answer looks like

| check | healthy | bad means |
|---|---|---|
| chained vs direct registration | ~1 px | the correlation is not locking onto CA2 — check the `/rd.mean(0)` division |
| registration confidence (`peak/std`) | thousands | a frame with no usable speckle |
| mask speckle | 9.1% rel after /flat, 8.2% after also /mean | approaching the noise = binning has destroyed the signal |
| flat structure | smooth envelope, no sharp periodic texture | a mask was still in during scan 68 |
| `err` in `conv_nfp.csv` | 1.9e-3 -> 4.6e-5 over 512 iters (~40x) | a big drop at iteration 0 then flat = the positions are wrong |

That last row is the important one. A big first step followed by a plateau is
the signature of incoherent positions: the solver fits the mean level and then
cannot stitch the aperture.

---

## Outputs

All into `/data3/cekmekci/BoneNano_nfp/`.

| file | shape | size | consumed by |
|---|---|---|---|
| `code.npy` | `[4096, 8192]` complex64 | 257 MB | the holo / holotomo stages |
| `prb.npy` | `[2048, 4096]` complex64 | 65 MB | same |
| `pos_nfp.npy` | `[121, 2]` float32 | 1 KB | diagnostics only |
| `data_nfp.h5` | — | 3.9 GB | written by `step0a` / `step0b` |
| `conv_nfp.csv` | — | — | convergence history |
| `checkpoints_tiff/` | — | — | per-iteration previews |

`code.npy` is the **transmission**, `exp(1j*proj)` — `RecNFP`'s variable is the
complex log, and `step0.py` exponentiates on the way out. Nothing wraps.

---

## Status — RUN, converged

Scan 80, crop-only (`bin=0`), 2048 x 4096, 512 iterations on 4 GPUs, about
22 minutes.

```
iter     err
  -1     1.875e-03
   0     3.390e-04
  32     5.842e-05
 256     4.713e-05
 480     4.622e-05      still falling slowly
```

**40x reduction.** For contrast, the earlier draft with the three convention
errors managed 12x and was flat after iteration 0 — that is what a broken
position set looks like.

Output sanity: `|code|` in [0.770, 1.209]; `|prb|` in [0.438, 1.346] with its
phase wrapping the full 2*pi, as expected for a probe carrying CA1. Recovered
position span 986 x 983 px, which is 8x the 123 px the bin-3 draft measured —
the same physical walk.

### One thing to check before trusting it downstream

The recovered code phase excursion is **0.55 rad**, which is on the low side
for a few microns of Au at 19.9 keV. It may simply be what this mask is, but
it is the first number to check against the fabrication spec: contrast goes
as `2|sin(phi/2)|`, so a weak mask means weak illumination diversity for
every stage that follows.
