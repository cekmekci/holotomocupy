# BoneNano / DanMAX — coded-aperture holotomography, scan 86

Jointly refines the 3-D object, the probe, CA2 and both position sets with the
Bilinear-Hessian solver, from one propagation distance through a moving mask.

The DanMAX counterpart of `AtomiumS2_coded`. The data handling is ported from
`phascade/experiments/bone_nano_holotomo/step2_scan-0086_catomo`.

---

## Self-contained: stage 0 lives here too

Like `AtomiumS2_coded`, this folder carries the whole chain, the NFP stage
included. You do not need `BoneNano_nfp` to run it.

```
step0a_convert -> step0b_shifts -> step0.py      (scan 80, NO sample: measures CA2)
step1_convert  -> step2_preprocess -> step3_find_shifts -> step6.py   (scan 86)
```

Stage 0 writes `code.npy` and `prb.npy`; step 3 picks them up as the warm
start. Starting from a **measured** aperture rather than solving for one cold
is the whole reason stage 0 exists.

### Relationship to `BoneNano_nfp`

`BoneNano_nfp` is the standalone NFP pipeline, for when you
want CA2 on its own. The files here are **byte-identical copies** of its
`step0.py`, `step0a_convert.ipynb` and `step0b_shifts.ipynb`:

```bash
diff BoneNano_nfp/step0.py BoneNano_coded_holotomo/step0.py      # should be empty
```

If that diff is not empty the two have drifted and one is stale. Fix the
cause, do not patch one side.

Only `config_step0.conf` differs, in two ways:

- **square throughout**: a centred 2048 x 2048 crop and a 4096 x 4096
  aperture plane, where `BoneNano_nfp` takes 2048 x 4096. See the section
  below — square is a simplification, not a requirement.
- paths point into this folder.

Read `BoneNano_nfp/README.md` for the stage-0 conventions in full — in
particular the **two-aperture mapping** (CA1 folded into the probe, CA2 as the
code) and the four preprocessing rules. This folder inherits all of them.

**Both stages must preprocess identically.** Stage 0 fits `code.npy` on data
that was dark-subtracted, dead-fixed, dezingered and scalar flat-normalised in
a specific order; if step 1 normalised differently, the warm start would be
inconsistent with the data it is supposed to explain. The helpers in step 1
are therefore the same ports, in the same order.

---

## Output layout

One root, split the way `AtomiumS2_coded` splits it:

```
/data3/cekmekci/BoneNano_coded_holotomo/
  nfp/                      <- stage 0, the aperture
      data_nfp.h5               scan-80 frames + /exchange/shifts
      code.npy  prb.npy         what step 3 picks up
      pos_nfp.npy  conv_nfp.csv
      checkpoints_tiff/
  ht/                       <- stages 1-6, the sample
      codedHT.h5                everything: geometry, theta, dark, flat_norm,
                                dead_mask, pdata, pref, prb, shifts_*
      checkpoints/  checkpoints_tiff/  conv.csv
```

Keeping them apart matters for more than tidiness: both drivers write a
`checkpoints_tiff/`, and stage 0's probe previews would otherwise sit in the
same directory as stage 6's object slices.

Neither directory exists yet. `step0a` and `step1` create their own.

Rough sizes: `nfp/data_nfp.h5` about 2 GB (121 frames at 2048 square),
`ht/codedHT.h5` about **63 GB** (the full 4001 angles at 2048 square, stored
UNBINNED — see below and "Preprocessing never subsamples the angles"). The
95 GB raw scan is only ever read.

---

## The scans

| scan | contents | used by |
|---|---|---|
| `scan-0067` | dark | step1 |
| `scan-0068` | flat, no CA1 and no CA2 | step1 |
| `scan-0086` | CA1 + CA2 + **rotating sample**, 4001 frames, **95 GB** | step2 |
| `scan-0086.h5` | companion, rotation angle at `/entry/instrument/tom_ry/value` [deg] | step1 |

---

## Run order

```bash
# stage 0 -- the aperture, from scan 80
jupyter lab      # step0a_convert, then step0b_shifts
mpirun -np <ngpus> ./set_affinity_gpu.sh python step0.py config_step0.conf

# stage 6 -- the sample, from scan 86
jupyter lab      # step1_convert, then step2_preprocess, then step3_find_shifts,
                 # then (optional) step5_rec_paganin
mpirun -np <ngpus> ./set_affinity_gpu.sh python step6.py config_step6.conf
```

| step | does | writes |
|---|---|---|
| `step0a_convert.ipynb` | scan 80: dark/flat/dead-mask, crop, dezinger | `data_nfp.h5` |
| `step0b_shifts.ipynb` | scan-80 positions by cross-correlation | `/exchange/shifts` |
| `step0.py` | the NFP solve, `RecNFPCoded` | `code.npy`, `prb.npy` |
| `step1_convert.ipynb` | geometry, angles, dark/flat/dead-mask | `energy`, `z1`, `focusdetectordistance`, `detector_pixelsize`, `voxelsize`, `theta`, `dark`, `flat_norm`, `dead_mask`, `flat_mean` |
| `step2_preprocess.ipynb` | the 95 GB pass | `pdata`, `pref` |
| `step3_find_shifts.ipynb` | warm start, CA2 positions | `prb`, `shifts_code`, `shifts_cor` |
| `step5_rec_paganin.ipynb` | per-angle Paganin + FBP, **optional** | `obj_init.npy` |
| `step6.py` | the joint solve, `RecCoded`, MPI | `checkpoints/`, `conv.csv` |

`step5_rec_paganin.ipynb` reads its own `config_steps15.conf`, not
`config_step6.conf` — see that file's header for why. It is ported directly
from phascade's own notebook for this dataset
(`bone_nano_holotomo/step2_scan-0086_catomo`, cell 30), not from
`AtomiumS2_coded`'s ablated Paganin settings, which are for a different
sample. It always runs **unbinned**, independent of whatever `bin`
`config_step6.conf` is currently set to — `CodedReader.read_obj` bins
`obj_init.npy` down itself, the same rule it applies to every other dataset
(see "What `CodedReader` bins" below), so the seed has to be built once at
full resolution and is then reusable at any bin.

Two differences from `AtomiumS2_coded`, both because the raw scan is 95 GB:

- **step 1 does no format conversion.** The DanMAX orca file is already HDF5,
  so step 1 is light and step 2 carries the cost.
- **the raw dark and white STACKS are not copied in.** `AtomiumS2_coded/step1`
  writes `/exchange/{data, dark, data_white0, data_white1}` — a full copy of
  the scan. Here only the processed mean `dark`, `flat_norm` and `dead_mask`
  go into `codedHT.h5`, which is all step 2 needs.

`step3_find_shifts.ipynb` also carries two diagnostics ported directly from
phascade's own bone_nano notebook for this dataset (cells 14 and 20 of
`1_catomo_recon.ipynb`), both measured-and-inspected rather than
auto-applied:

- **did the probe move between stage 0 (scan 80) and this scan (86)?**
  `BoneNano_nfp`'s own README documents this check turning up a
  non-negligible shift between an earlier pair of scans (69 -> 80), and scan
  69 was abandoned as a result. This is how the same question gets answered
  for 80 -> 86.
- **does an independent chain registration agree** with the absolute one
  computed against stage 0's reference? Healthy is ~1 px, matching
  `BoneNano_nfp`'s own diagnostics table.

Neither diagnostic applies a correction by itself — they are how a bad
registration would be caught before it is baked into `shifts_code`/`shifts_cor`.

---

## Conventions that differ from stage 0, and why

### Positions are measured DIRECTLY, not chained

Stage 0 chains `phase_corr` between consecutive frames from a mid-scan anchor.
This stage does **not**. It correlates every projection against a reference
built from stage 0's own output:

```python
ref_rd   = I_code / I_prb          # stage-0 code+probe through the same chain
pos_code = -phase_corr(ref_rd, rd[k])
```

The reason: stage 0 anchored its aperture plane to scan 80's own random-walk
centre, so its origin says nothing about where CA2 sat during scan 86.
Correlating against the stage-0 reference recovers **the walk and that
constant offset together**, in one shot, with no cumulative drift.

The **negation** is the same convention as everywhere else in this engine
family.

### Square crop — a simplification, not a limitation

This pipeline crops **2048 x 2048**, throwing away the wider half of the
4608-pixel detector axis. That is a deliberate choice for now, not something
the engine requires.

**`RecCoded` does support a rectangular detector.** Verified by construction:
`nz=64, n=128` builds, generates data and takes a gradient. In

```
obj_shape  = [nzobj, nobj, nobj]          # slices x (square slice plane)
proj_shape = [ntheta, nzobj, ndobj]
cl_shift   = ShiftFFT(n, ndobj, nz, nzobj, ...)
```

`nzobj` counts object **slices** and tracks detector **rows**; `nobj`/`ndobj`
is the in-slice width and tracks **columns**. They are independent.

Two things genuinely must be square:

- **the object slice**, `nobj x nobj` — fundamental to tomography, automatic.
- **the code plane** — `RecCoded` carries a single `ncode` scalar and
  hardcodes `code_shape=(ncode, ncode)`, and `read_code` asserts it. This one
  is an artifact of our port, not physics: `ShiftFFTCode` already accepts a
  `(rows, cols)` tuple, so it could be generalised with an `ncodez` in about
  ten lines. Doing so would touch `rec_coded_mpi.py`, which `AtomiumS2_coded`
  also uses.

**To go rectangular later:** widen `n` in `config_step6.conf` and
`config_step0.conf` *together*, keep `npsiz == npsi`, and re-check `ncode`.

### Steps 1-5 never bin — only step 6/`CodedReader` does

This is the second-easiest thing to get wrong here (the first is angle
subsampling, above), and an earlier draft of this folder got it wrong in
exactly the same way: it bin'd `pdata`/`prb`/`code` down in step 3's own
registration math, and scaled `shifts_cor`/`shifts_code` up by `2**bin`
before writing them — i.e. it did part of `CodedReader`'s job a second time,
at the wrong place.

**The rule, matching `AtomiumS2_coded` exactly:** steps 1 through 5 always
run at ONE fixed **native** (unbinned) resolution —
`config_step6.conf`'s `crop_nz`/`crop_n` (2048 x 2048), which stage 0's own
`config_step0.conf` also uses. Binning happens in exactly one place:
`CodedReader`, at step 6 read time, scaling `pdata`/`prb`/`code.npy`/
`detector_pixelsize` down by `config_step6.conf`'s own `bin` — never
anywhere upstream.

Two separate grid pairs in `config_step6.conf`, do not conflate them:

```
crop_nz / crop_n   NATIVE, steps 1-5           2048 x 2048
nz / n             BINNED, step 6/CodedReader  256  x 256   (= crop >> bin)
```

Consequences that follow directly:

- Step 3's registration (`pos_code`, `pos_obj`) is computed in **native,
  unbinned pixels**, and written to `codedHT.h5` with **no scale factor** —
  exactly the units `CodedReader` expects, and exactly what
  `AtomiumS2_coded/step3_find_shifts.ipynb` does.
- Step 3's `SIZING` check (does the measured walk fit?) is done in native
  px too, converting `config_step6.conf`'s `nobj`/`ncode` *up* to native
  units for the comparison (`nobj << bin`), rather than converting the
  measured positions down.
- Reusing stage 0's `code.npy`/`prb.npy` across the two planes is **exact**,
  not approximate, for a second, independent reason: the code→sample
  distance carries a `(z1/z1_code)**2` factor, which is the Fresnel scaling
  theorem's compensation for reinterpreting an array on a different pixel
  grid. The native-resolution rule above is what makes that factor the
  *only* correction needed — no bin-driven resampling on top of it.

### `rho` is reordered

The phascade notebook uses `(prb, code, pos_code, obj, pos_obj)`. holotomocupy
uses `[obj, prb, pos, code, pos_code]`. The config carries the reordered
version of the phascade values:

```
phascade    (0.03, 0.03, 0.01, 1.0,  0.01)
config      [1.0,  0.03, 0.01, 0.03, 0.01]
```

Do not copy the tuple across without reordering it.

---

## Geometry

| | |
|---|---|
| energy | 19.90 keV |
| focus → CA2 (`z1_code`) | 0.14125 m |
| focus → sample (`z1`) | 0.14925 m |
| focus → detector | 0.74625 m |
| detector pixel, unbinned | 230 nm |

**CA2 is before the sample** (`z1_code < z1`), which is the geometry `RecCoded`
implements. `step1` asserts it.

All TODO-confirm against the DanMAX logbook before beamtime.

---

## What `CodedReader` bins, and what that forces

This is the easiest thing to get wrong, so it is worth stating plainly.
**`CodedReader` does the binning itself**, on the way in. So the file must
hold UNBINNED arrays:

| dataset | stored | reader does |
|---|---|---|
| `pdata` | unbinned, 2048 x 2048 | bins by `2**bin` |
| `prb` | unbinned | bins by `2**bin` |
| `code.npy` | unbinned | bins by `2**bin`, then **asserts** the shape is `(ncode, ncode)` |
| `detector_pixelsize` | unbinned | multiplies by `2**bin` |
| `shifts_cor`, `shifts_code` | unbinned px | divides by `2**bin` |

Two consequences:

- **`pdata` is big.** Pre-binning it would bin it twice. Storing it unbinned
  is also what makes a bin ladder possible — `bin2 -> bin1 -> bin0` all read
  the same file and only change `bin` in the config.
- **`ncode` is not a free parameter.** It is fixed by stage 0:
  `ncode == npsi >> bin`, here `4096 >> 3 == 512`. If the measured code walk
  needs more than that, raise `npsi` in `config_step0.conf` and re-run stage
  0 — do not just raise `ncode`, the assert will fire.

### The reconstruction already runs on binned data — nothing extra needed

`config_step6.conf` already has `bin=3`: step 6 solves at 256 x 256 even
though 2048 x 2048 unbinned sits in `codedHT.h5`. `CodedReader` does that
binning on read, for `pdata`, `prb`, `code.npy` and `detector_pixelsize`
alike. For a quick look during the beamtime — not a full run — the knobs are
already in that file: lower `ntheta`, raise `stride`, or raise `bin` further.
No new code needed.

This is a different question from whether **step 2** could write already-
binned data instead. It currently doesn't: it writes the unbinned 2048 x 2048
crop and leaves all binning to `CodedReader`, on purpose, so one `codedHT.h5`
serves bin 3, 2, 1 and 0 just by changing the config. Nothing in the engine
forbids binning earlier instead — `CodedReader` only needs to be told, via
`bin`, how much *further* binning to do on top of what is already on disk. If
step 2 baked in `bin=3` at write time, step 6 would run with `bin=0` and
`CodedReader`'s own binning would be a no-op. That path is not implemented —
step 2 does not average-bin its output — but it is a small addition if the
95 GB write/read ever becomes the bottleneck, not a redesign.

## Before running step 6

Three things step 3 prints that must be acted on:

- **`ncode` consistency.** Step 3 reads `code.npy`'s actual size and tells you
  whether the config matches `npsi >> bin`.
- **Whether the walk fits.** If `max|pos_code|` needs a bigger plane than
  stage 0 produced, step 3 says so and names the fix.
- **Registration confidence.** A projection whose correlation peak is not far
  above the background has a guessed position.

And one choice:

- **`obj_init_file` is commented out**, so step 6 cold-starts the object.
  `CodedReader` allows that. Run `step5_rec_paganin.ipynb` first and
  uncomment `obj_init_file` in `config_step6.conf` for a Paganin+FBP warm
  start instead — ported from the phascade notebook's cell 30, as described
  above. Cold-starting remains a legitimate choice for a quick in-situ look
  (see "The reconstruction already runs on binned data" above); the Paganin
  seed matters more for a converged, polished reconstruction later.

---

## Preprocessing never subsamples the angles

Steps 1, 2 and 3 always convert and register the **full 4001-frame scan** —
there is no "first pass, fewer angles" shortcut at preprocessing time. Step 2
still only reads the 2048 x 2048 cropped region of each frame (the HDF5 slice
is indexed with the crop built from `config_step6.conf`'s `nz`/`n`/`bin`, not
the full 2592 x 4608 orca frame), but it reads and registers all 4001 of them.

**This was not the original design, and the original design was broken.**
The first version of this folder had step1/2/3 select `ntheta=500,
start_theta=0, stride=8` frames themselves — the SAME three config keys
`step6.py`'s `CodedReader` *also* applies, at read time, as an offset/stride
into whatever is on disk (`AtomiumS2_coded`'s own, correct, convention: write
everything once, subsample only when solving). Applying the same stride
**twice** — once at write time, baking a 500-frame file, and again at read
time, striding by 8 a second time into that already-short array — reached
index 3992 into a 500-length array and crashed `step6.py` on startup
(`CodedReader.__init__`'s own bounds check:
`self.ids[-1] >= theta_deg.size`). Worse, a different ntheta/stride
combination could have silently read the wrong frames instead of crashing.

The fix is structural, not a patched value: preprocessing writes the full
scan, period, so there is exactly **one** subsampling point in the whole
pipeline — `config_step6.conf`'s `ntheta`/`start_theta`/`stride`, read by
`CodedReader` directly against the complete stored array. A quick-look run
(fewer angles) is then just a `config_step6.conf` edit, nothing upstream:

```
config_step6.conf:  ntheta=500  start_theta=0  stride=8   <- the ONLY place
                                                              to change this
```

If you ever want MORE angles, or a different stride, or less binning (which
generally wants denser angular sampling to avoid aliasing) — **this one file
is the only thing to edit.** Nothing needs to be rerun upstream, because
nothing upstream was subsampled.

`step5_rec_paganin.ipynb` follows the same rule from the other side: it reads
**everything stored** (`ntheta, nz, n = pdata.shape`), never a
`config_steps15.conf`-sliced subset — a Paganin+FBP seed benefits from more
angles, not fewer, and `config_steps15.conf` has no `start_theta`/`stride`
fields to replicate `config_step6.conf`'s own subsampling, so slicing by a
number there would silently diverge from whatever `config_step6.conf`
actually selects the moment that stops being a trivial 1-of-1 case.

`nchunk=8` is the first knob to lower if step 6 runs out of memory.

---

## Status

**Written, not run.** None of these five steps has been executed. The config
geometry is taken from the phascade notebook and is marked TODO-confirm. The
`ncode`/`nobj` values are provisional until step 3 measures the walk.
