# AtomiumS2, coded aperture

Single-distance holotomography with a coded aperture. The holotomography
reconstruction (step 6) is solved with `holotomocupy.rec_coded_mpi.RecCoded`;
the aperture itself is measured beforehand (stage 0) with
`holotomocupy.rec_nfp_coded_mpi.RecNFPCoded`.

[`AtomiumS2`](../AtomiumS2) is the multi-distance run on the same sample. Four
distances there, one distance plus a moving coded aperture here.

Pre-processing (stages 0-5) runs on tomo5 the first time, since stages 1-3
need the raw scan. Stage 5 alone can also rerun directly on Polaris/eagle
afterward -- it only needs `mpi4py` + `h5py driver='mpio'`, same as step 6/7
-- which is what both the REBUILD step inside `polaris_run.sh` and the
standalone `polaris_run_steps15.sh` do. Steps 6 and 7 -- the BH
reconstruction ladder and the drift refinement in between its two passes --
go to Polaris.

---

## Directory layout

One tree, grouped by SCAN rather than flat by filename -- the NFP
calibration scan and the HT scan are different scans with different raw data
and produce different files, so each gets its own subfolder:

```
/data3/cekmekci/AtomiumS2_coded/
    nfp/                  data_nfp.h5, code.npy, prb.npy   (stage 0)   SHARED
    ht/                   codedHT.h5, obj_init.npy         (stages 1-5) SHARED
    rec/<run>/            everything ONE run produces:
        obj_init_corrected.npy                             (stage 5 rebuild)
        pass1/            pass-1 checkpoints, bin2 and bin1 (step 6)
        refined/          pass-2 checkpoints, bin2/bin1/bin0 (step 6)
```

Mirrored on eagle for Polaris at `/eagle/APS_IRI/cekmekci/AtomiumS2_coded/`.

`nfp/` and `ht/` are **shared inputs**: every run reads them and no run writes
to them. Everything a run produces lands under `rec/<run>/`, and the filenames
inside are identical for every run -- only `<run>` differs. The run is named
by the `run=` key at the top of each config in its folder under `runs/`; see
[runs/README.md](runs/README.md) for the convention and for how to make a new
run (`cp -r` plus one `sed`).

Every config's `path` / `path_out` is one of these folders; `code_file` in the
step-6 and steps15 configs is the one path that crosses folders (it lives in
`nfp/` while the rest of that config's files live in `ht/`), so it is always
spelled out as an absolute path rather than a filename.

---

## Pipeline

| stage | file | reads | writes |
|---|---|---|---|
| 0a | `step0a_convert_nfp.ipynb` | raw NFP scan | `data_nfp.h5` |
| 0b | `step0b_preprocess_nfp.ipynb` | `data_nfp.h5` | + `pdata`, `pref`, `shifts` |
| 0 | `step0.py` `config_step0.conf` | `data_nfp.h5` | **`code.npy`**, **`prb.npy`** |
| 1 | `step1_convert.ipynb` | raw HT scan | `codedHT.h5` |
| 2 | `step2_preprocessing.ipynb` | `codedHT.h5` | + `pdata`, `pref0`, `pref1` |
| 3 | `step3_find_shifts.ipynb` | `codedHT.h5`, `code.npy`, `prb.npy` | + `prb`, `shifts_cor`, `shifts_code` |
| 5 | `step5_rec_paganin.ipynb` | all of the above | **`obj_init.npy`** |
| 6 (pass 1) | `step6.py` `config_step6_bin{2,1}_pass1.conf` | `codedHT.h5`, `code.npy`, `obj_init.npy` | checkpoints, TIFFs, `conv.csv` |
| 7 | `step7.py` | the pass-1 bin-1 checkpoint | `correct_correct3D_extra.txt` |
| 5 (rebuild) | `steps15.py` `config_steps15_corrected.conf`, `start_step=5` | all of stage 5's inputs + `correct_correct3D_extra.txt` | **`obj_init_corrected.npy`** |
| 6 (pass 2) | `step6.py` `config_step6_bin{2,1,0}.conf` | `codedHT.h5`, `code.npy`, `obj_init_corrected.npy` | checkpoints, TIFFs, `conv.csv` |

No step 4. The multi-distance pipeline pre-bins there. `CodedReader` bins on
the way in, so one `codedHT.h5` serves every rung.

Steps 1, 2, 3, 5 are also a single batch script, `steps15.py`, gated by
`start_step` in `config_steps15.conf` (`start_step > N` skips stage N; see
"Run" below for the command). `config_steps15.conf`'s own `start_step` is
`5` as committed -- stages 1-3 already ran once and wrote
`shifts_cor`/`shifts_code`/`prb` into `codedHT.h5`; set it back to `1` only
if the raw data or the shift-finding itself needs to be redone.

**Step 6 runs twice, with a seed rebuild in between -- not just a direct
handoff.** Pass 1 (`config_step6_bin{2,1}_pass1.conf`, positions frozen)
gives step 7 an unbiased volume to measure the per-angle drift against.
Patching step 7's correction directly into `pos` on a copy of the pass-1
checkpoint does NOT work (tried, reverted) -- that checkpoint's object was
reconstructed consistent with the OLD, uncorrected positions, so bolting a
corrected `pos` onto it creates a geometric mismatch rather than fixing one.
The actual fix reruns stage 5 (`config_steps15_corrected.conf`,
`start_step=5`) with the correction folded into `pos_obj` BEFORE Paganin,
producing a new seed, `obj_init_corrected.npy`, that is geometrically
consistent with the corrected positions from iteration 0. Pass 2
(`config_step6_bin{2,1,0}.conf`) reads that seed and refines on top of it.
See "Two-pass drift refinement" below.

Stage 0 measures the mask. It is a near-field ptychography scan of the
aperture alone, the coded counterpart of `AtomiumS2/step0.py`. Step 6 starts
from that measured aperture rather than solving for one cold.

**Stage 0 is required here, unlike the multi-distance pipeline's top-level
README, which calls its own step0 optional.** There `RecNFP`'s cold start
works without a measured probe. `RecCoded` has no cold-start path for the
aperture -- `code.npy` only exists if stage 0 runs, and without it there is
nothing to pass as `code_file` to stage 3, stage 5, or step 6.

### What step 6 reads

```
codedHT.h5   /exchange/pdata        [ntheta, nz, n] f32, INTENSITY, unbinned
             /exchange/prb          [nz, n] c64
             /exchange/theta        [ntheta] degrees
             /exchange/shifts_cor   [ntheta, 2] object positions, unbinned px
             /exchange/shifts_code  [ntheta, 2] code positions, unbinned px
             /exchange/{energy, z1, detector_pixelsize, focusdetectordistance}
code.npy     [ncode, ncode] c64, unbinned
obj_init.npy [nzobj, nobj, nobj] complex, unbinned. Optional, cold start without it
```

`obj_init.npy`/`obj_init_corrected.npy` are saved by stage 5 ALREADY at
`(nzobj, nobj, nobj)`, not detector size. Stage 5 pads the raw, flat-divided
intensity data (background `== 1.0` by construction) out to `nobj` BEFORE
Paganin, with `cp.pad(..., 'linear_ramp', end_values=1.0)` ramping across the
WHOLE margin -- `AtomiumS1_HT_RD300/steps15.py`'s own background pad
(lines 916-919 there) -- plus a small `reflect` pad/crop around
`multiPaganin` itself for FFT-ringing suppression, same source. Before
2026-11-11 this stage ran at detector size and `CodedReader.read_obj`
zero-padded the result up to `nobj` afterward, which left a sharp vacuum
discontinuity at the seed's boundary. `read_obj` needed no changes for the
fix -- once the seed already arrives at `nobj` size, its existing
crop/downsample logic (`_place`) just takes over.

**Ramp the whole margin, in both axes.** He ramps only x and pads y with
plain `'edge'`, because his vertical detector edges sit in background, so
replicating them replicates vacuum. Ours carry real structure. A first
version of `pad_taper` here mixed up two different things in his code -- it
borrowed the quintic smoothstep he uses to blend ADJACENT DISTANCES
(`steps15.py:950-961`, multi-distance stitching, width `n//16`) and used it
as the background taper, leaving the rest of the margin as raw
edge-replication. Those replicated rows are identical at every angle, so
they do not rotate, and FBP backprojects them coherently onto the rotation
axis: a bright blob in the middle of the z-padding, measured at `12.3`
against a normal range of `+-2-3`, peaking 5 px from the rotation axis
(`nobj/2 + 24`, per `rot_axis_offset`). Ramping the full margin in both axes
leaves no replicated band to produce it.

`pdata` is intensity, not its square root. The coded fork stored
`sqrt(intensity)` because its misfit was on amplitudes. `RecCoded.F0` compares
`K|psi|^2` against `d` and takes its own sqrt.

---

## Run

Two machines, five phases. `nfp/` and `ht/` are SHARED inputs built once on
tomo5 (Phases 0-2); everything after that is a run under `runs/`, isolated by
its `run=` key (see [runs/README.md](runs/README.md)).

### Phase 0 -- check the raw inputs exist

```bash
ls -d /data3/vnikitin/ca/SiemensLH/code2um_nfp18x18_01                       # NFP raw scan
ls -d /data3/vnikitin/ca/raw/AtomiumS2_codedht_random_007nm                  # HT raw scan
ls -l  /data3/vnikitin/ca/raw/AtomiumS2_HT_007nm_/correct_correct3D.txt
h5ls /data3/cekmekci/AtomiumS2_coded/nfp/data_nfp.h5/exchange                # pdata, pref, shifts, z1, voxelsize
```

**The NFP raw scan (`code2um_nfp18x18_01`) no longer exists on disk.** If
`data_nfp.h5` is missing, it cannot be rebuilt from stage 0a -- restore it
from backup instead of regenerating it. It already carries a geometry patch
(`patch_nfp_geometry.py`); a restored copy is correct, a freshly-converted one
would not be. If `data_nfp.h5` is present with `pdata`/`pref`/`shifts` already
in it, stage 0b is done and Phase 1 skips straight to the `step0.py` call.

### Phase 1 -- tomo5, build the shared inputs

```bash
cd experimental/AtomiumS2_coded

jupyter lab                       # step0a, step0b, ONLY if data_nfp.h5 needs rebuilding -> data_nfp.h5

# --mca opal_common_ucx_opal_mem_hooks 1 is on every mpirun below: UCX warns
# "unable to handle VM_UNMAP event ... may cause performance degradation or
# data corruption" on tomo5 otherwise. If mpirun fails with
# "prterun-exec-failed", tomo5's .bashrc-loaded nvhpc module is leaking
# OPAL_PREFIX into the conda env's own Open MPI -- load hpcx explicitly first:
#   export NVHPCSDK_HOME=/local/nvidia/hpc_sdk/Linux_x86_64/26.1
#   . ${NVHPCSDK_HOME}/comm_libs/13.1/hpcx/latest/hpcx-init.sh
#   hpcx_load

mpirun --mca opal_common_ucx_opal_mem_hooks 1 -np 4 ./set_affinity_gpu.sh python step0.py config_step0.conf
                                  #                         -> nfp/code.npy, nfp/prb.npy

# FROM SCRATCH: config_steps15.conf is committed with start_step=5 (stages
# 1-3 already done). Run all of it once:
sed -i 's/^start_step=.*/start_step=1          # FROM SCRATCH/' config_steps15.conf
mpirun --mca opal_common_ucx_opal_mem_hooks 1 -np 4 ./set_affinity_gpu.sh python steps15.py config_steps15.conf
                                  #                         -> ht/codedHT.h5, ht/obj_init.npy
# set_affinity_gpu.sh matters here as much as GPU pinning: it also sets
# OMP_NUM_THREADS=4. Without it every rank's NumPy/SciPy calls default to
# all 32 cores, so 4 MPI ranks 4x-oversubscribe the machine and every
# thread-using op (astype, where, the dezinger loop in stage 2) fights the
# other 3 ranks for the same cores -- this is why an MPI run can come out
# SLOWER than the single-process notebook doing the same math.
sed -i 's/^start_step=.*/start_step=5/' config_steps15.conf   # restore, so a rerun doesn't redo 1-3
```

Check before moving on, not just at the end:

| after | check | good looks like |
|---|---|---|
| stage 0 | `step0_sanity.ipynb` | predicted vs measured `pref0`, difference AND ratio |
| stage 0 | `nfp_sweep/to_tiff.py` on `code.npy` | `max\|code\|` <= 1 (was 1.79 before `clip_passive`) |
| stage 3 | `h5ls codedHT.h5/exchange` | `shifts_cor`, `shifts_code`, 7200 rows each |
| stage 5 | `python visualize_paganin.py /data3/.../ht/obj_init.npy` | mmap quick-look, no rim blob on the rotation axis |

```bash
python preflight.py runs/baseline/config_step6_bin2_pass1.conf   # check inputs, seconds
NP=4 ./run.sh runs/baseline/config_step6_bin2_pass1              # optional, the coarse pass-1 rung locally
```

That last line is the ENTIRE step 6 + step 7 ladder you can also run right
here instead of Polaris -- repeat per config, same order as Phase 4
(`run.sh` is `step6.py` under `mpirun`; `step7.py` runs with plain `python`).
Reasonable for bin 2 on a small/cropped dataset, not for the full ladder at
bin 0.

### Phase 2 -- transfer, tomo5 -> Polaris

Code goes up with `../../sync_polaris.sh` (rsync mirror, `-n` first). Data
goes up with Globus. The Polaris job needs four files, from TWO folders:

| file | from | size | needed by |
|---|---|---|---|
| `codedHT.h5` | `ht/` | 170 GB | step 6, step 7, stage-5 rebuild |
| `obj_init.npy` | `ht/` | 134 GB | step 6 pass 1 |
| `code.npy` | `nfp/` | 513 MB | step 6, stage-5 rebuild |
| `prb.npy` | `nfp/` | ~4 MB | **stage-5 rebuild only** |

**`prb.npy` is easy to forget and the failure is late.** Step 6 does not read
it (no `prb_file` key; `prb` comes from `codedHT.h5`), but `polaris_run.sh`'s
REBUILD THE SEED stage runs `steps15.py` stage 5, which loads `prb_file`
unconditionally (`steps15.py:435`). Leave it behind and the job dies after
pass 1 and step 7 have already been paid for.

`data_nfp.h5` stays on tomo5 -- stage 5 only opens it for a nominal `z1_code`
(`steps15.py:397`), and `config_steps15_corrected.conf` sets `z1_code`
explicitly, so that branch never runs.

Stripe BOTH output directories before the first run on a run folder -- eagle
defaults to one OST and a bin-0 checkpoint is large:

```bash
R=/eagle/APS_IRI/cekmekci/AtomiumS2_coded/rec
mkdir -p $R/<run>/pass1 $R/<run>/refined
lfs setstripe -c 8 -S 16M $R/<run>/pass1
lfs setstripe -c 8 -S 16M $R/<run>/refined
```

### Phase 3 -- prove the ladder on the debug queue

`runs/toy/` is a normal run at coarser binning (bin 4/3, 400 of 7200 angles),
exercising all five stages in ~1 h on 2 debug-queue nodes. The failure modes
that have actually bitten -- missing `prb.npy`, step 7 diverging -- both
surface after pass 1, where discovering them at prod scale is expensive.

```bash
R=/eagle/APS_IRI/cekmekci/AtomiumS2_coded/rec
mkdir -p $R/toy/pass1 $R/toy/refined
lfs setstripe -c 8 -S 16M $R/toy/pass1
lfs setstripe -c 8 -S 16M $R/toy/refined
cd runs/toy
qsub polaris_run.sh
```

See [runs/README.md](runs/README.md) for the bin-to-grid-size table and the
two things that bite specifically at toy scale (`step7 --ntheta` must match
the toy's own angle count, not 7200; the seed-rebuild stage costs full price
regardless of binning).

### Phase 4 -- Polaris, the real ladder

1. **Pick or make the run.** Each run is a folder under `runs/`, holding its
   six configs and its `polaris_run.sh`. `runs/baseline` is the reference
   ladder; to make another, `cp -r` it and change the `run=` key -- see
   [runs/README.md](runs/README.md).
2. **ssh in and source the environment.**
   ```bash
   ssh polaris
   cd ~/holotomocupy_gpu_reduced/experimental/AtomiumS2_coded
   source ../polaris_env.sh
   ```
3. **If `obj_init.npy` predates a stage-5 code change**, regenerate it in its
   own job first -- `polaris_run.sh` only rebuilds `obj_init_corrected.npy`:
   ```bash
   qsub polaris_run_steps15.sh
   ```
4. **Submit.** From inside the run's own folder -- configs and
   `correct_correct3D_extra.txt` resolve relative to where `polaris_run.sh`
   lives:
   ```bash
   cd runs/baseline
   qsub polaris_run.sh
   qstat -u $USER
   ```
   `polaris_run.sh` does everything from here, as one job: `preflight.py`
   before each rung, `gpu_healthcheck.sh`, refuses to start if
   `correct_correct3D_extra.txt` already exists (see "Two-pass drift
   refinement"), then **pass 1** (`bin2_pass1` -> `bin1_pass1`, positions
   frozen, iter 0 -> 256 -> 512), **step 7** (-> `correct_correct3D_extra.txt`),
   **rebuild the seed** (`steps15.py config_steps15_corrected.conf`,
   `start_step=5` -> `obj_init_corrected.npy`), **pass 2**
   (`bin2` -> `bin1` -> `bin0`, iter 0 -> 256 -> 512 -> 768, refined positions).
5. **Watch it.** `tail -f pbs-<jobid>.out`; each stage prints
   `=== <name> START <date> ===`.
6. **If a stage fails or is preempted**, every `mpiexec` ends `|| exit $?`.
   `ls <path_out>/checkpoints/`, set `start_iter` to the highest
   `checkpoint_NNNN`, comment out the finished `mpiexec` lines, resubmit.

To skip part of the ladder (e.g. `correct_correct3D_extra.txt` is already
good), comment out the corresponding `mpiexec` lines in `polaris_run.sh` --
there is no command-line flag.

`esrf.slurm` is the same job for ESRF. One rung per submission, picked by
`CONFIG=` at the top. `set_affinity_gpu.sh` and `set_affinity_gpu_polaris.sh`
bind one GPU per rank, copied unchanged from `AtomiumS2`.

---

## Configs

Three scan-level files plus five step-6 rungs, split the same way as the
multi-distance folders.

| file | covers | read by |
|---|---|---|
The first two sit at the top level because they build the **shared** inputs
every run reads. The rest belong to one run and live in that run's folder
under `runs/`.

| file | covers | read by |
|---|---|---|
| `config_step0.conf` | the NFP scan, which images the mask alone -- builds `nfp/`, SHARED | `step0a`, `step0b`, `step0.py` |
| `config_steps15.conf` | the holotomography scan, stages 1-5 (`start_step=5` as committed -- see "Pipeline" above) -- builds `ht/`, SHARED | `step1`, `step2`, `step3`, `step5`, `steps15.py` |
| `runs/<run>/config_steps15_corrected.conf` | stage 5 only, rebuilding the seed with step 7's correction in `pos_obj` -- see "Two-pass drift refinement" | `steps15.py`, `start_step=5` |
| `runs/<run>/config_step6_bin2_pass1.conf`, `..._bin1_pass1.conf` | the reconstruction, pass 1 (positions frozen) | `step6.py`, `step7.py` |
| `runs/<run>/config_step6_bin2.conf`, `..._bin1.conf`, `..._bin0.conf` | the reconstruction, pass 2 (refined) | `step6.py` |

The notebooks pull their paths through `parse_args_coded_step0` and
`parse_args_coded_steps15`, so no path is written twice.

`config_steps15.conf`/`config_steps15_corrected.conf` also carry the few
numbers that change per dataset. `ntheta`, `n`, `nobj` (the object grid,
unbinned -- stage 5 pads/tapers up to this before FBP, see "What step 6
reads"; MUST match `config_step6_bin0.conf`'s `nobj`), `rot_axis_offset`,
`paganin`, `fbp_filter`, an optional `z1_code`, and `start_step` for
`steps15.py` (`start_step > N` skips stage N; no stage 4). Leave `z1_code`
unset and stages 3 and 5 take the nominal value out of `data_nfp.h5`. Set it
to the tuned `-17.75e-3` that step 6 uses if you want the seed built on the
same geometry the solver assumes. Structural constants like dark and flat
counts, GPU counts and chunk sizes stay in the notebooks.

All six configs in a run folder carry the same `run=` / `rec=` pair at the
top, and derive every run-specific path from it: the `_pass1` configs'
`path_out` is `rec/<run>/pass1`, the pass-2 configs' is `rec/<run>/refined`
-- see "Directory layout" above and [runs/README.md](runs/README.md).

```
n = nz = 2048 >> bin     nobj = nzobj = n + n//4     ncode = 8192 >> bin
```

| bin | n | nobj | ncode | start_iter | niter | nchunk |
|---|---|---|---|---|---|---|
| 2 | 512 | 640 | 2048 | 0 | 512 | 16 |
| 1 | 1024 | 1280 | 4096 | 512 | 768 | 8 |
| 0 | 2048 | 2560 | 8192 | 768 | 896 | 4 |

---

## Things that will bite

**Each PASS shares one `path_out`, not the whole ladder.** Pass 1's bin 2
starts fresh and reads `obj_init.npy`; pass 2's bin 2 starts fresh and reads
`obj_init_corrected.npy` instead, into a SEPARATE `path_out` (`rec/<run>/pass1` vs
`rec/<run>/refined`). Within each pass, the finer rungs resume from the checkpoint
above, which `CodedReader.read_checkpoint` upsamples x2 -- object, probe,
aperture and both position sets.

**`ntheta` and `stride` are identical on every rung**, all 7200 acquired
angles (`stride=1`). They have to be. `pos` and `pos_code` are per-angle and
ride through the checkpoint, so a rung with a different angle count cannot
resume. 7200 is well above Nyquist at every bin -- about 9x at bin 2, 4.5x at
bin 1, 2.2x at bin 0 -- so this is not about sampling adequacy, it is the full
dataset by choice. That means bin 2 and bin 1 pay for every angle even though
their n is small; set `stride` (and `ntheta` to match) on every config
together if that cost needs trimming for a quick local check.

**Resume is manual.** `find_latest_checkpoint` globs
`checkpoint_*{start_iter:04}.h5` and returns `None` when `start_iter == 0`. It
finds the checkpoint you name, not the newest on disk. After a preemption, run
`ls {path_out}/checkpoints/`, set `start_iter` to the highest, comment out the
finished lines, resubmit. The same keying makes the bin2 to bin1 to bin0
handoff work, so it is not a bug.

**`z1_code = -17.75e-3` is negative and is tuned.** The aperture sits upstream
of the focal spot, so the code-to-sample Fresnel distance is negative too. It
is not the nominal value in `data_nfp.h5`. The fork hardcoded the same number.
Stages 3 and 5 use the nominal value unless you set `z1_code` in
`config_steps15.conf`, so by default the seed and the solver assume slightly
different geometries. That is what the fork did too.

**`rho` has five entries**, `[obj, prb, pos, code, pos_code]`, where the
multi-distance config has four. `rho[obj]` must stay at 1 whenever
`lam_laplacian > 0`. `RecCoded` refuses otherwise, because the Laplacian
gradient is added without `rho_sq['obj']` while the data-misfit half carries it.

**Nothing auto-detects the machine.** All five step-6 config files carry
tomo5 paths as committed.

**`steps15.py` stages 1-3 still only run on tomo5** -- they need `/data3`
raw-scan access, which Polaris does not have. Stage 5 alone has no such
requirement (same `mpi4py` + `h5py driver='mpio'` environment as step 6/7),
so it runs directly on Polaris/eagle both inside `polaris_run.sh` (the
REBUILD step) and standalone (`polaris_run_steps15.sh`) -- avoiding a round
trip of a 170 GB file back to tomo5 and a 134 GB object back to eagle every
time the seed needs rebuilding.

---

## Two-pass drift refinement

Step 3's `shifts_cor` comes from the ESRF `correct_correct3D.txt` table, which
does not fully capture the per-angle drift. Step 7 measures what is left over
by entropy autofocus on a reconstructed volume, same idea as
`AtomiumS1_HT_RD300/step7.py`, adapted to read a `RecCoded` checkpoint.

The catch: re-projecting a volume that was reconstructed with a drift
correction already in `pos` bakes that correction into the object, so an
autofocus search on it is partly measuring the metric's own bias, not the
sample's drift. The fix is the same two-pass split the multi-distance
pipeline uses.

```
PASS 1 (positions frozen)     step6 bin2_pass1  ->  step6 bin1_pass1
STEP 7                        entropy autofocus on the pass-1 checkpoint
                               -> correct_correct3D_extra.txt
REBUILD THE SEED               steps15.py stage 5, start_step=5, with the
                               correction folded into pos_obj
                               -> obj_init_corrected.npy
PASS 2 (refined)               step6 bin2  ->  step6 bin1  ->  step6 bin0
```

**Why there's a rebuild step, not a direct handoff.** The first version of
this pipeline had step 7 write `correct_correct3D_extra.txt` and pass 2 read
it straight into `pos` via `correct3d_extra_file`, same as the multi-distance
pipeline does. That's still how `pos` itself gets corrected. But an earlier
variant also tried applying the correction by patching `pos` directly onto a
COPY of the pass-1 checkpoint and continuing pass 2 from there, skipping a
seed rebuild -- that does not work: the pass-1 object was reconstructed
consistent with the OLD, uncorrected positions, so bolting a corrected `pos`
onto it creates a geometric mismatch rather than fixing one. The actual fix
reruns stage 5 (`config_steps15_corrected.conf`, `start_step=5`, skipping
stages 1-3 since `codedHT.h5` already has `shifts_cor`/`shifts_code`/`prb`)
with `pos_obj = shifts_cor + rot_axis_offset + extra` instead of without
`extra`, producing a new seed that is geometrically consistent with the
corrected positions from iteration 0. Pass 2 then reads BOTH the new seed
(`obj_init_corrected.npy`, bin 2 only) and the correction itself
(`correct3d_extra_file`, applied to `pos` independently, same mechanism the
seed rebuild used) -- not double counting, since `CodedReader.read_pos`
starts from `shifts_cor` without either addition.

`config_step6_bin{2,1}_pass1.conf` differ from their pass-2 counterparts in:
`rho`'s `pos` entry is 0 (so the position gradient is identically zero and
nothing drifts further), `obj_init_file` points at the original
`obj_init.npy` rather than the rebuilt `obj_init_corrected.npy`, there is no
`correct3d_extra_file` key, and `path_out` is a separate tree (`rec/<run>/pass1`
vs `rec/<run>/refined`) so pass 2 cannot overwrite the volume step 7 read.
Pass 1 stops at bin 1 -- no `_pass1` bin-0 config -- since step 7's search
grid only needs to be fine enough to resolve the drift curve, not the final
reconstruction.

`step7.py` writes `correct_correct3D_extra.txt` next to the configs, in the
same two-column (horizontal, vertical) format
`holotomocupy.autofocus.write_correct3d_extra` uses for the multi-distance
pipeline -- both `CodedReader.read_correct3d_extra` (for `pos`) and stage 5
(for `pos_obj`) read it the same way, though stage 5 applies NO
`1/2**bin` scale (it always runs unbinned) where `CodedReader` does.
**`--ntheta` on the `step7.py` call must match pass 1's own angle count**,
not an arbitrary subsample -- `scan_theta` picks its search angles on an
even stride over the FULL raw scan, independent of which angles the
checkpoint was actually built from; a mismatch there (confirmed on the toy
ladder, which runs pass 1 on only 400 of 7200 angles) forward-projects the
checkpoint at angles it has no real information about and makes the search
diverge far more easily. On the real ladder pass 1 already uses all 7200
angles, so `--ntheta 7200` is always consistent there.

Run `step7.py` exactly as `AtomiumS1_HT_RD300/step7.py`, pointed at the
pass-1 bin-1 config so it finds the right checkpoint:

```bash
python step7.py config_step6_bin1_pass1.conf --iter 512 --bin 0 --ntheta 7200
```

`polaris_run.sh` runs the whole ladder -- pass 1, step 7, rebuild the seed,
pass 2 -- in one job and refuses to start if `correct_correct3D_extra.txt`
already exists (see the script's own re-submission-hazard comment).
`obj_init.npy` itself (pass 1's seed, as opposed to pass 2's
`obj_init_corrected.npy`) is NOT rebuilt by `polaris_run.sh` -- that's
`polaris_run_steps15.sh`, run separately beforehand if stage 5's own code
changed (see "Polaris, step by step" above).

---

## Stage 0 runs on `RecNFPCoded`

The coded fork had its own NFP engine. It is not needed -- `RecNFPCoded`
(`holotomocupy.rec_nfp_coded_mpi`) is the advisor's `RecNFP` with one change,
a windowed object-plane shift, and gets everything else (the misfit, the BH
loop, the position-refinement bookkeeping) from `RecNFP` unmodified.

```
RecNFP(Coded)  D( exp(1j * S_pos(proj)) * prb )
coded fork     D( S_r(psi) * q )
```

Same physics. The fork's variable is the aperture transmission, `RecNFP`'s is
its complex log. So `proj` goes in as `-1j * log(psi_init)` and comes out as
`exp(1j * proj)`. Nothing wraps, since a few microns of Au at 33.35 keV is
~1.5 rad.

**Windowing.** The fork gathered an `n + 2*ex` patch per position. Plain
`RecNFP` instead shifts the whole `npsi**2` plane per position -- GPU cost
`nchunk * npsi**2`, ~40 GB across 4 GPUs at `npsi = 8192` -- because its own
use case (calibrating a probe, where the reconstructed plane is barely bigger
than the detector) never needed windowing. `RecNFPCoded` reuses
`shift_fft_code.ShiftFFTCode` (built for the coded engine's aperture shift,
same shape operator) to window this level too, controlled by
`pos_pad_frac` in `config_step0.conf`. `pos_pad_frac=0` degrades it back to
plain `RecNFP`'s exact whole-plane behavior.

One thing the fork did that `RecNFPCoded` still does not: a probe-fit term.
The fork drove the probe gradient with `ref`; here `ref` only seeds the
probe.

---

## Status

Ported from `holotomo-coded-atomiums2`. `RecCoded` is tested across
`test_shift_code.py`, `test_cascade_coded.py`, `test_rec_coded.py` and
`test_io_coded.py` (121 checks, plus MPI parity), and the synthetic demo runs
(`demo/03_coded_holotomography`). `RecNFPCoded` is tested separately in
`test_rec_nfp_coded.py` (15 checks, plus MPI parity): degenerate equivalence
to plain `RecNFP`, cascade Taylor order, and MPI parity. `bash
tests/coded/run.sh` runs all of it, 136 checks serial plus the MPI pair.

`step0.py`, `steps15.py`, `step7.py`, and `rec_nfp_coded_mpi.RecNFPCoded` are
new code, not ported from the fork, so they are the least exercised.
`preflight.py` reads the real `codedHT.h5`, `code.npy` and `obj_init.npy` and
passes.

**The full real-data ladder has not been run end to end yet.** A toy run
(the `bin4`/`bin3` ladder, run through actual Polaris job mechanics) has
validated pass 1 + step 7's own mechanics -- env, configs, `estimate_rho`,
`lam_laplacian`, preflight, invocation order all work. Step 7's divergence
guard correctly fired at that toy scale (400 of 7200 angles informing the
checkpoint, searched against all 7200) and refused to write a bad
correction, so REBUILD THE SEED and pass 2 have never actually executed --
only the real ladder, with pass 1 built from all 7200 angles, can exercise
those stages meaningfully.
