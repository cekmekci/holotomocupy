# MicroCT / DanMAX — 2-D coded-aperture holography, scans 90–94

Recovers a **2-D** sample plane jointly with the probe, the coded aperture and
both position sets, from 121 frames at one propagation distance through a
moving mask.

**This is holography, not holotomography.** There is no rotation, no `theta`,
no Radon. The scan axis is **frames**; the object is one shared 2-D plane. The
engine is `RecHoloCoded` (`src/holotomocupy/rec_holo_coded_mpi.py`), the 2-D
sibling of the `RecCoded` that `BoneNano_coded_holotomo` uses.

Sample: a 2-D microCT test target ("the horizontal vertical"), DanMAX, orca
detector, 19.90 keV.

---

## The thing to understand first: there are TWO apertures

The logbook's "weak CA" and "strong CA" map onto the engine's single `code`
variable the same way CA1/CA2 do in the BoneNano folders:

| physical | moves? | engine variable | ends up in |
|---|---|---|---|
| weak CA | no, static from scan 92 on | part of the **probe** | `prb.npy` |
| strong CA | **yes**, per frame | the **code** | `code.npy` |

`prb.npy` is therefore *the beam with the weak CA already imprinted on it*. It
is not a clean illumination function and is not meant to be. Almost every
subtlety below follows from this.

---

## The scans

| scan | frames | contents | used by |
|---|---|---|---|
| `scan-0090` | 100 | dark | step0a, step1a |
| `scan-0091` | 100 | flat — **no CA at all** | step0a, step1a (flat + dead-pixel mask) |
| `scan-0092` | 100 | **weak CA alone**, no sample | step0a, step1b → `pref` |
| `scan-0093` | **121** | weak + strong CA, no sample, 11×11 raster | **stage 0, the NFP scan** |
| `scan-0094` | **121** | weak + strong CA + **sample**, strong CA moves, sample fixed | **stage 1, the reconstruction scan** |
| `scan-0095` | 121 | weak CA removed, strong CA + sample | not used |
| `scan-0096` | 100 | sample alone, no CA | not used |

Frames: `/entry/instrument/orca/data`, `(nframes, 2592, 4608)` uint16. Motor
values: the companion `scan-00NN.h5` (no `_orca`), at
`/entry/instrument/nf_x/value` and `nf_y/value`.

**Scan 92 is better than what BoneNano had.** It is a 100-frame average of
beam × weak CA with the weak CA already in its final position, so `pref` is a
*direct measurement* rather than something inferred. Note where this does and
does not matter: `pref` is used for the **registration** divisions in step0b
and step1c. It is **not** the probe seed — `step0.py` seeds the probe by
back-propagating `sqrt(mean of scan 93)`, since the strong CA averages out of
that mean. Both are legitimate; do not confuse them.

Scans 95 and 96 are left unused but are worth knowing about: 95 is the same
measurement with the weak CA out (so the probe is the bare beam), and 96 is an
*uncoded* measurement of the same sample, which would make a good ground-truth
comparison for the coded result. Neither is wired up.

---

## Run order

```bash
# stage 0 -- the aperture, from scan 93
jupyter lab      # step0a_convert, then step0b_shifts
mpirun -np <ngpus> ./set_affinity_gpu.sh python step0.py config_step0.conf

# stage 1 -- the sample, from scan 94
jupyter lab      # step1a_convert, step1b_preprocess, step1c_find_shifts
export HDF5_USE_FILE_LOCKING=FALSE          # see "NFS" below -- not optional
mpirun -np <ngpus> -mca opal_common_ucx_opal_mem_hooks 1 \
    ./set_affinity_gpu.sh python step6.py config_step6.conf
```

| step | does | writes |
|---|---|---|
| `step0a_convert.ipynb` | scans 90/91/92/93: dark, flat, dead-mask, crop, dezinger | `data_nfp.h5` |
| `step0b_shifts.ipynb` | strong-CA positions by cross-correlation | `/exchange/shifts` |
| `step0.py` | the NFP solve, `RecNFPCoded`, MPI | `code.npy`, `prb.npy`, `pos_nfp.npy` |
| `step1a_convert.ipynb` | geometry, dark, flat, dead-mask | `energy`, `z1`, `focusdetectordistance`, `detector_pixelsize`, `voxelsize`, `dark`, `flat_norm`, `dead_mask`, `flat_mean` |
| `step1b_preprocess.ipynb` | scan-94 frames + `pref` | `pdata`, `pref` |
| `step1c_find_shifts.ipynb` | warm start, strong-CA positions | `prb`, `shifts_code`, `shifts_cor` |
| `step6.py` | the joint solve, `RecHoloCoded`, MPI | `checkpoints/`, `conv.csv` |

`step6.py` is named for consistency with `AtomiumS2_coded` and
`BoneNano_coded_holotomo`, not because there are six steps.

`step0.py`, `step6.py` and `set_affinity_gpu.sh` are **byte-identical copies**
of the ones in `BoneNano_nfp` / `BoneNano_coded_holo`:

```bash
diff ../BoneNano_nfp/step0.py step0.py            # should be empty
diff ../BoneNano_coded_holo/step6.py step6.py     # should be empty
```

A non-empty diff means they have drifted. Fix the cause, not one side.

### Output layout

```
/data3/cekmekci/MicroCT_coded_holo/
  nfp/     data_nfp.h5  code.npy  prb.npy  pos_nfp.npy  conv_nfp.csv  checkpoints_tiff/
  holo/    codedHolo.h5  checkpoints/  checkpoints_tiff/  conv.csv
           pos_errors/  pos_code_errors/
```

Split the way `AtomiumS2_coded` splits `nfp/` and `ht/`: both drivers write a
`checkpoints_tiff/`, and stage 0's probe previews must not land on top of the
solve's object slices.

---

## Adapting this to a different detector, crop or binning

**This is the section to read before changing any size.** Six numbers are
coupled by arithmetic the engines *assert*; get one wrong and you either crash
at startup or silently throw data away.

### The one rule behind all of it

Steps 0–1 run at **one fixed native (unbinned) resolution**. Binning happens
in exactly one place: `HoloCodedReader`, at solve time, driven by
`config_step6.conf`'s `bin`. Nothing upstream ever bins.

```
crop_nz / crop_n   NATIVE, steps 0-1c        e.g. 2048 x 2048
nz / n             BINNED, step6 only        = crop >> bin
```

### The arithmetic that must hold

| relation | enforced by | failure mode if wrong |
|---|---|---|
| `crop_nz, crop_n <= detector rows, cols` | your own crop | IndexError in step0a/step1b |
| stage-0 `nz`,`n` == `crop_nz`,`crop_n` | you, by hand | stage 0 and stage 1 disagree; the warm start is misregistered |
| stage-0 `bin` == 0 | `config_step0.conf` | `code.npy`/`prb.npy` come out at the wrong scale |
| `npsiz == npsi` | `RecHoloCoded` (single `ncode` scalar) | ValueError at step6 startup |
| `npsi >= crop + 2*max|shift| + 2*fov_margin` | **nothing** — silent | `step0.py`'s FOV filter **silently drops** positions |
| step6 `nz == crop_nz >> bin`, `n == crop_n >> bin` | you, by hand | shape mismatch, or a quietly wrong pixel size |
| **`ncode == npsi >> bin`** | `HoloCodedReader.read_code`, **hard assert** | `ValueError: code.npy binned to (X,X), expected (Y,Y)` |
| `nzobj >= nz + 2*max|pos_obj| + 2*EX + 32` | `step1c` prints it | the shift window reads off the edge of the object plane |
| `nobj >= n + 2*max|pos_obj| + 2*EX + 32` | `step1c` prints it | same |

Two of these are measured, not chosen: **`npsi`** comes from `step0b`'s
measured walk, and **`nobj`/`nzobj`** from `step1c`'s. Both notebooks print
the minimum and say which config to edit. Do not skip them.

### Worked example — the current configuration

```
detector            2592 x 4608         orca, fixed
crop                2048 x 2048         centred; config_step6.conf crop_nz/crop_n
                                        and config_step0.conf nz/n
stage-0 npsi/npsiz  4096 x 4096         measured by step0b, square (required)
step6 bin           2
step6 nz/n          512 x 512           = 2048 >> 2
step6 ncode         1024                = 4096 >> 2   <- HARD, asserted
step6 nzobj/nobj    640 x 640           >= 512 + margin, checked by step1c
```

### Recipe: changing the crop

Say you want 1024 × 1024 instead of 2048 × 2048.

1. `config_step0.conf`: `nz=1024`, `n=1024`.
2. Re-run `step0a_convert`, `step0b_shifts`. **Read step0b's `npsi` print** —
   a smaller crop does not shrink the physical walk, so `npsi` may not change
   at all. Set `npsiz`/`npsi` to what it says (keep them equal).
3. Re-run `step0.py`.
4. `config_step6.conf`: `crop_nz=1024`, `crop_n=1024`, and `nz`/`n` =
   `1024 >> bin`.
5. Recompute `ncode = <new npsi> >> bin`.
6. Re-run `step1a`, `step1b`, `step1c`. **Read step1c's sizing print** and set
   `nzobj`/`nobj` from it.

### Recipe: changing the bin (the bin ladder)

This is the cheap one — **no notebook needs re-running**, because everything
on disk is native-resolution. Edit `config_step6.conf` only:

```
bin=2  ->  nz=512  n=512   ncode=1024    (= 4096 >> 2)
bin=1  ->  nz=1024 n=1024  ncode=2048    (= 4096 >> 1)
bin=0  ->  nz=2048 n=2048  ncode=4096    (= 4096 >> 0)
```

`ncode` **must** change with `bin` — this is the single most common mistake
here and it is exactly the `ValueError` quoted in the table above. `nzobj`/
`nobj` should scale roughly with `nz`/`n` too; re-read step1c's print rather
than guessing. Carry `start_iter` forward to resume from the coarser rung.

### Recipe: a rectangular crop

Allowed here, **unlike the holotomo folder**. There is no Radon, so the object
plane may be rectangular — `nzobj != nobj` is fine. Only the **code plane**
must stay square (`npsiz == npsi`), because `RecHoloCoded` carries a single
`ncode` scalar. So a 2048 × 4096 crop with a square 8192 × 8192 code plane is
legitimate. The current square crop is a simplification, not a requirement.

### Recipe: a different experiment at the same beamline

The geometry block (§ below) is the same optical configuration as
`BoneNano_coded_holotomo`, verified by motor comparison. If you take new data
with the optics unchanged, only `z1` (focus→sample) and the scan numbers
change. Confirm by comparing `nf_z`, `lg_z`, `sg_z`, `hdcm_energy` in the new
`scan-NNNN.h5` against the values quoted below.

---

## Geometry

| | value | status |
|---|---|---|
| energy | 19.90 keV | **verified** |
| focus → CA plane (`z1_code`) | 0.14125 m | **verified** |
| focus → detector (`focusdetectordistance`) | 0.74625 m | **verified** |
| detector pixel, unbinned | 230 nm | **verified** |
| focus → sample (`z1`) | 0.14925 m | **TODO CONFIRM** |

"Verified" means the optics motors in `scan-0093.h5` agree with
`bone_nano_holotomo/scan-0086.h5` to five decimals — `nf_z` 47.40916 vs
47.40914, `lg_z` 1414.79719 vs 1414.79721, `sg_z` 260.97575 vs 260.97589,
`hdcm_energy` 19.90037 vs 19.89996. Same configuration, so BoneNano's numbers
carry over.

**`z1` does not.** That sample sat on the tomography stage; this one does not,
and no motor in these files has been shown to encode focus→sample distance.
`tom_y` is transverse — it moves exactly 20.000 mm between scan 93 (sample
out, −62.215) and scan 94 (sample in, −42.215), which is a swing-out, not a
propagation move. The 0.14925 in `config_step6.conf` is **inherited from
BoneNano and unconfirmed for this sample**. Check it against the logbook.

For the NFP, `config_step0.conf`'s `z1` means **focus → CA plane** (the
reconstructed plane *is* the aperture), so it is 0.14125, not 0.14925. Same
convention as `BoneNano_nfp/config_step0.conf`.

---

## The conventions that matter

Each of these was wrong at least once in this codebase. If you change the
preprocessing, these are what to preserve.

### 1. The registration target needs TWO divisions

```python
rd = pdata / (pref + 1e-6)        # strip the beam envelope (per pixel)
rd = rd / (rd.mean(0) + 1e-6)     # strip the STATIC weak CA
```

The second is the one that is easy to miss. The weak CA does not move, so
without removing it the correlation peak is dominated by a pattern identical
in every frame and **every measured shift is pulled toward zero**. The symptom
is a solve that drops the error on iteration 0 and then goes flat. This exact
omission cost BoneNano a 12× convergence instead of 40×.

This per-pixel division is for *registration only*. The data handed to the
solver is normalised by a **scalar** — see 3.

### 2. Steps 0–1 never bin; shifts are written native

No `bin2` in any registration or writing path. `step1c` writes
`shifts_code`/`shifts_cor` in **native unbinned px with no scale factor** —
`HoloCodedReader` divides by `2**bin` itself. The only legitimate use of
`bin2` in these notebooks is shrinking a display thumbnail, and that variable
is named `PREVIEW_BIN` so it cannot be confused.

### 3. The flat is divided out as a SCALAR

```python
raw = flat_field_divide(raw, flat_mean)    # only flat.mean() is used
```

Not per pixel. The reference still carries the weak CA, so a per-pixel divide
would remove structure the forward model is supposed to reconstruct as part of
the probe.

### 4. Preprocessing chain, exact order

```
dark_subtract_clip -> fill_dead -> remove_outliers -> flat_field_divide
```

- `remove_outliers` uses a **`radius × radius`** window, **not** `2*radius+1`.
  `dezinger_radius=3` is a 3×3 filter. Getting it wrong quietly over-smooths
  the mask speckle, which is the entire signal.
- The dead-pixel mask comes from the **flat** (`flat < 0.05 × median`) and is
  applied to the flat *and* every frame.

Stage 0 and stage 1 **must** preprocess identically, or stage 0's `code.npy`
is fitted on data normalised differently from what step 6 reads.

### 5. Registration is chained in stage 0, direct in stage 1

Stage 0 (`step0b`) chains `phase_corr` between consecutive frames from a
mid-scan anchor. Stage 1 (`step1c`) does **not** — it correlates every frame
directly against a reference built from stage 0's own output, and negates:

```python
pos_code = -phase_corr(ref_rd, rd[k])
```

The reason: stage 0 anchored its aperture plane to scan 93's own grid centre,
which says nothing about where the CA sat in scan 94. Correlating against the
stage-0 reference recovers the walk **and** that constant offset in one shot,
with no cumulative drift. The difference between the two stages is deliberate.

### 6. `rho` ordering

holotomocupy uses `[obj, prb, pos, code, pos_code]`. phascade used
`(prb, code, pos_code, obj, pos_obj)`. **Never copy a phascade tuple across
without reordering it.**

### 7. `obj` is the exponent, not a transmission function

`RecHoloCoded.F4` computes `exp(1j * obj)`, so

```
exp(1j·obj) = exp(1j·obj.real) · exp(−obj.imag)
              └─ phase ─┘        └─ attenuation ─┘
```

The phase to look at is **`obj.real`** (stored as `obj_re` in the
checkpoints). `np.angle(obj)` is meaningless here — it mixes phase and
absorption. The TIFF previews wrote exactly that until 2026-11-24 and looked
like pure noise while the H5 looked clean; see "Known gotchas".

---

## Diagnostics, and what a bad answer looks like

The notebooks print these. They are not decoration — act on them.

| check | where | healthy | bad means |
|---|---|---|---|
| flat has no sharp periodic texture | step0a | smooth envelope | a CA was still in during scan 91 |
| mask speckle survives binning | step0a | clear speckle in `frame/ref` | the bin is erasing the signal |
| row-end jumps in the raster | step0b | row-end steps comparable to within-row | the chain is integrating a jump it cannot resolve |
| chained vs direct registration | step0b | ~1 px | the correlation is not locking onto the strong CA — check the `/rd.mean(0)` division |
| measured shifts vs `nf_x`/`nf_y` motors | step0b | row and col slopes agree to a few %, flat residual | a sign error, a bad division, or a mis-integrated row end |
| scan 92 vs scan 93 mean | step0b | < 1 px | the weak CA moved; `pref` is stale |
| required `npsi` | step0b | ≤ configured | `step0.py` will silently drop positions |
| probe drift, stage 0 → scan 94 | step1c | < 1 px | the stage-0 probe is not where scan 94 has it |
| chain cross-check | step1c | ~1 px | as above |
| `ncode` consistency | step1c | OK | hard `ValueError` at step6 startup |
| sample fixed across frames | step1b | `first − last` shows CA walk, not sample motion | the scan is not what the logbook says |
| `err` in `conv.csv` | step6 | falls monotonically, ~100× | a big drop at iteration 0 then flat = the positions are wrong |

The 11×11 raster is worth a special note. `BoneNano_nfp`'s scan 80 was a
random walk, so consecutive frames were always close. This is a raster:
within a row the stage moves ~4 µm, but at each of the 10 row ends it flies
back across the whole row. Those 10 transitions are the weak point of a
chained registration, which is why `step0b` flags `k % 11 == 10` separately
and why the motor regression is worth running.

---

## Known gotchas

### NFS file locking — `export HDF5_USE_FILE_LOCKING=FALSE`

`/data3` is NFS (`tomodata3-ib:/data3`). HDF5 ≥ 1.10 takes an `fcntl` lock
when opening a file, and over NFS that intermittently fails:

```
BlockingIOError: [Errno 11] Unable to synchronously create file
(unable to lock file, errno = 11, 'Resource temporarily unavailable')
```

Observed on 2026-11-24 at iteration 128 of a 256-iteration run, after
checkpoints 0, 32, 64 and 96 had written fine. It is **not** specific to any
iteration — checkpoint writes are simply the only locking operation in the
loop. The filesystem was already degraded: iteration times had spiked to 17 s,
42 s and 16 s against a normal ~6 s.

Set `HDF5_USE_FILE_LOCKING=FALSE` before `mpirun`. This is safe here: only
rank 0 writes the checkpoint, serially, with `h5py.File(path, 'w')` — there is
no concurrent writer the lock protects against.

A failed write leaves a **0-byte** `checkpoint_NNNN.h5`. Delete it before
re-running; it is not a valid checkpoint and will confuse a later resume.

### Stale checkpoints from an earlier run

`checkpoints/` is not cleared between runs, so a short run leaves higher-
numbered checkpoints from a longer one lying around. With `start_iter=0` this
is harmless — `find_latest_checkpoint` returns `None` — but the directory
listing is misleading. Clear it when starting a genuinely new run.

### TIFF previews were wrong until 2026-11-24

`HoloCodedWriter._write_tiff` wrote `np.angle(obj)` instead of `obj.real`.
Because `obj` is the exponent (convention 7 above), that is a meaningless
quantity: measured on checkpoint 0256, `obj.real` had std 0.239 over
[−1.14, 1.01] while `np.angle(obj)` had std 1.74 spanning the full [−π, π] —
96 % of what a uniformly random angle would give, i.e. essentially pure
wrapping noise. The symptom was "beautiful H5, unreadable TIFF". Fixed in
`src/holotomocupy/reader_holo_coded.py`; **checkpoints written before the fix
are fine**, only their previews were wrong, and they can be re-rendered from
`obj_re` in the H5.

---

## Status — RUN, converging

**Stage 0 (NFP, scan 93):** converged. 512 iterations, 4 GPUs, ~40 s per
32-iteration block.

```
iter      err
  -1      1.992e-03
 448      3.940e-05
 480      3.938e-05      still falling slowly
```

**50.6× reduction.** Outputs: `code.npy` `[4096, 4096]` complex64 (128 MB),
`prb.npy` `[2048, 2048]` complex64 (32 MB), `pos_nfp.npy` `[121, 2]`.

**Stage 1 (the solve, scan 94):** runs and converges. A 256-iteration run
completed; a later run reached iteration 128 before the NFS lock failure
above.

```
initial   2.204e-02
iter 0    1.411e-03
iter 32   3.186e-04
iter 64   2.226e-04
iter 128  1.565e-04      still falling
```

**~140× and still falling** at the point of interruption. Position drift stays
small and well behaved — at iteration 128, `pos` drift was ~0.1–0.3 px and
`pos_code` ~0.004 px, i.e. the warm start was already close.

**Still open:** `z1` is unconfirmed (see Geometry); the bin ladder below
`bin=2` has not been run; scans 95 and 96 are unused, and 96 in particular
would give an uncoded ground truth to compare against.
