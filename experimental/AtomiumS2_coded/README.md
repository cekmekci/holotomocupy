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
    nfp/             data_nfp.h5, code.npy, prb.npy       (stage 0)
    ht/              codedHT.h5, obj_init.npy,
                     obj_init_corrected.npy               (stages 1-5)
    rec/pass1/       pass-1 checkpoints, bin2 and bin1     (step 6)
    rec/refined/     pass-2 checkpoints, bin2/bin1/bin0    (step 6)
```

Mirrored on eagle for Polaris at `/eagle/APS_IRI/cekmekci/AtomiumS2_coded/`
(same four subfolders), which is what every step-6 config's commented-out
Polaris path pair points at.

Every config's `path` / `path_out` is one of these four folders; `code_file`
in the step-6 and steps15 configs is the one path that crosses folders
(it lives in `nfp/` while the rest of that config's files live in `ht/`),
so it is always spelled out as an absolute path rather than a filename.

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

Two machines. Stages 0-3 (raw conversion, flat correction, shift finding)
need `/data3` and only run on tomo5. Stage 5 (Paganin+FBP) can run either
there, right after stages 1-3, or rerun later directly on Polaris/eagle --
see "Polaris, step by step" below for the standalone rebuild script. Step 6
on can run either locally on tomo5 (small/coarse checks) or, for the real
job, on Polaris.

### tomo5, once per dataset

```bash
cd experimental/AtomiumS2_coded
jupyter lab                       # step0a, step0b          -> data_nfp.h5

# --mca opal_common_ucx_opal_mem_hooks 1 is on every mpirun below: UCX warns
# "unable to handle VM_UNMAP event ... may cause performance degradation or
# data corruption" on tomo5 otherwise, on every run. run.sh already bakes
# this in; the two bare mpirun calls below need it spelled out.
#
# If mpirun still fails with "prterun-exec-failed", tomo5's .bashrc-loaded
# nvhpc module is leaking OPAL_PREFIX into the conda env's own Open MPI --
# load hpcx explicitly first:
#   export NVHPCSDK_HOME=/local/nvidia/hpc_sdk/Linux_x86_64/26.1
#   . ${NVHPCSDK_HOME}/comm_libs/13.1/hpcx/latest/hpcx-init.sh
#   hpcx_load
mpirun --mca opal_common_ucx_opal_mem_hooks 1 -np 4 ./set_affinity_gpu.sh python step0.py config_step0.conf
                                  #                         -> code.npy, prb.npy

jupyter lab                       # step1, step2            -> codedHT.h5
                                  # step3                   -> shifts_cor, shifts_code
                                  # step5                   -> obj_init.npy
# -- or, instead of the four notebooks above (first time only -- set
# start_step=1 in config_steps15.conf; it is committed as 5, see "Configs") --
mpirun --mca opal_common_ucx_opal_mem_hooks 1 -np 4 python steps15.py config_steps15.conf
                                  #                         -> codedHT.h5, obj_init.npy

python preflight.py config_step6_bin2_pass1.conf   # check inputs, seconds
NP=4 ./run.sh config_step6_bin2_pass1              # optional, the coarse pass-1 rung locally
```

That last line is the ENTIRE step 6 + step 7 ladder you can also run right
here instead of Polaris -- just repeat it for each config in turn, same
order as the Polaris section below (`run.sh` is `step6.py` under `mpirun`,
one config per call; `step7.py` runs directly with `python`). Reasonable to
do for bin 2 on a small/cropped dataset; not reasonable for the full ladder
at bin 0, which is what Polaris is for.

**An even faster rehearsal: the toy `bin4`/`bin3` ladder, `tomo5_test/`.**
Even bin 2 takes real time against the full dataset. `tomo5_test/`'s
`config_step6_bin{4,3}{,_pass1}.conf` run the same two-pass mechanism
(checkpoint handoff, position freezing, step 7, pass 2 picking up the
correction) at a much coarser grid (`n=128` then `256`) and on only 400 of
the 7200 angles (`stride=18`), in low single-digit minutes. `tomo5_run.sh`
runs the whole thing. Output goes to its own `rec/test_pass1` /
`rec/test_refined` tree, separate from the real `rec/pass1` / `rec/refined`,
so the two can never collide. Not scientifically meaningful (16x/8x binning,
5% of the angles) -- it's a mechanism check, not a reconstruction.
`polaris_test/`'s `polaris_test_run.sh` is the same toy ladder, run through
the actual Polaris job mechanics (PBS, mpiexec, the GPU healthcheck) instead
of locally -- see its own header for the full stage list, which now
includes a toy REBUILD THE SEED step mirroring the real one.

### Transfer, tomo5 -> Polaris

Code goes up with `../../sync_polaris.sh` (rsync mirror, `-n` first). Data
goes up with Globus. Step 6 needs three files, from TWO different folders
(see "Directory layout" above).

| file | from | size |
|---|---|---|
| `codedHT.h5` | `ht/` | 170 GB |
| `obj_init.npy` | `ht/` | 134 GB |
| `code.npy` | `nfp/` | 513 MB |

`data_nfp.h5` and `prb.npy` (both in `nfp/`) stay on tomo5. Step 6 never opens
them.

Stripe BOTH output directories before the first run. Eagle defaults to one
OST and a bin-0 checkpoint is large.

```bash
lfs setstripe -c 8 -S 16M rec/refined
lfs setstripe -c 8 -S 16M rec/pass1
```

### Polaris, step by step

1. **Edit the five step-6 configs** (`config_step6_bin{2,1}_pass1.conf`,
   `config_step6_bin{2,1,0}.conf`) -- uncomment the Polaris `path` /
   `path_out` pair at the top of each, comment out the tomo5 pair. Nothing
   auto-detects the machine.
2. **ssh in and source the environment.**
   ```bash
   ssh polaris
   cd ~/holotomocupy_gpu_reduced/experimental/AtomiumS2_coded
   source ../polaris_env.sh
   ```
3. **If `obj_init.npy` predates a stage-5 code change** (e.g. the
   pad-to-`nobj`-before-FBP taper added 2026-11-11), regenerate it FIRST, in
   its own job -- `polaris_run.sh` does not rebuild this file, only
   `obj_init_corrected.npy` (step 4 below):
   ```bash
   qsub polaris_run_steps15.sh
   ```
   Its own header explains why this is a separate script: tuning stage 5
   (Paganin params, the taper) means rerunning it repeatedly, and bundling it
   into the 24-node/2-hour `polaris_run.sh` would mean paying for pass 1,
   step 7 and pass 2 on every tuning iteration. Lower its `#PBS -l select=`
   and switch to the debug queue for cheap iteration -- this stage has no
   toy-scale subsetting, so fewer nodes trades cost for wall-clock, not less
   total work.
4. **Submit the real ladder.**
   ```bash
   qsub polaris_run.sh
   qstat -u $USER
   ```
   `polaris_run.sh` does everything from here on, as one job:
   - `preflight.py` on EACH config right before ITS OWN rung runs, not all
     five upfront -- a later rung's checkpoint dependency does not exist
     until the rung before it has actually run, so checking it before the
     ladder even starts would abort a from-scratch submission immediately.
     Still seconds, still before anything expensive runs for that rung.
   - `gpu_healthcheck.sh`, dropping any node whose GPUs will not take a CUDA
     context
   - refuses to start if `correct_correct3D_extra.txt` already exists next
     to the configs (the re-submission guard -- see "Two-pass drift
     refinement" below; if you are intentionally resuming pass 2 only,
     comment out the pass-1, step-7 and rebuild `mpiexec` lines in the
     script first)
   - **pass 1**: `step6.py config_step6_bin2_pass1.conf`, then
     `config_step6_bin1_pass1.conf` (positions frozen, iter 0 -> 512 -> 768)
   - **step 7**: `step7.py` on the pass-1 bin-1 checkpoint -> writes
     `correct_correct3D_extra.txt`
   - **rebuild the seed**: `steps15.py config_steps15_corrected.conf`
     (`start_step=5`) folds that correction into `pos_obj` and reruns
     Paganin+FBP -> `obj_init_corrected.npy`, geometrically consistent with
     the corrected positions from iteration 0 (see "Two-pass drift
     refinement" for why patching `pos` onto the pass-1 checkpoint directly
     does not work)
   - **pass 2**: `step6.py config_step6_bin2.conf` -> `bin1.conf` ->
     `bin0.conf` (iter 0 -> 512 -> 768 -> 896, refined positions)
5. **Watch it.** `tail -f <SCRIPT_DIR>/pbs-<jobid>.out`, or just wait for
   `qstat` to drop the job. Each stage prints `=== <name> START <date> ===`
   so you can tell which rung it is on.
6. **If a stage fails or is preempted**, every `mpiexec` line ends
   `|| exit $?`, so the job stops rather than seeding the next rung from a
   checkpoint that was never written. `ls <path_out>/checkpoints/`, set
   `start_iter` in the config that failed to the highest `checkpoint_NNNN`
   present, comment out the mpiexec lines for stages that already finished,
   resubmit.

To run only part of the ladder (skip pass 1 and step 7 because
`correct_correct3D_extra.txt` is already good, say), comment out the
corresponding `mpiexec` lines in `polaris_run.sh` directly -- that is the
control mechanism, there is no command-line flag.

`esrf.slurm` is the same job for ESRF. One rung per submission, picked by
`CONFIG=` at the top. `set_affinity_gpu.sh` and `set_affinity_gpu_polaris.sh`
bind one GPU per rank, copied unchanged from `AtomiumS2`.

---

## Configs

Three scan-level files plus five step-6 rungs, split the same way as the
multi-distance folders.

| file | covers | read by |
|---|---|---|
| `config_step0.conf` | the NFP scan, which images the mask alone | `step0a`, `step0b`, `step0.py` |
| `config_steps15.conf` | the holotomography scan, stages 1-5 (`start_step=5` as committed -- see "Pipeline" above) | `step1`, `step2`, `step3`, `step5`, `steps15.py` |
| `config_steps15_corrected.conf` | stage 5 only, rebuilding the seed with step 7's correction in `pos_obj` -- see "Two-pass drift refinement" | `steps15.py`, `start_step=5` |
| `config_step6_bin2_pass1.conf`, `config_step6_bin1_pass1.conf` | the reconstruction, pass 1 (positions frozen) | `step6.py`, `step7.py` |
| `config_step6_bin2.conf`, `config_step6_bin1.conf`, `config_step6_bin0.conf` | the reconstruction, pass 2 (refined) | `step6.py` |

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

All five step-6 configs each carry both machines' `path` / `path_out` pairs
in a block at the top (the `_pass1` configs' `path_out` is `rec/pass1`, the
pass-2 configs' is `rec/refined` -- see "Directory layout" above).

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
`obj_init_corrected.npy` instead, into a SEPARATE `path_out` (`rec/pass1` vs
`rec/refined`). Within each pass, the finer rungs resume from the checkpoint
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
`correct3d_extra_file` key, and `path_out` is a separate tree (`rec/pass1`
vs `rec/refined`) so pass 2 cannot overwrite the volume step 7 read.
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
python step7.py config_step6_bin1_pass1.conf --iter 768 --bin 0 --ntheta 7200
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

**The full real-data ladder has not been run end to end yet.** `polaris_test/`
(the toy `bin4`/`bin3` ladder, run through actual Polaris job mechanics) has
validated pass 1 + step 7's own mechanics -- env, configs, `estimate_rho`,
`lam_laplacian`, preflight, invocation order all work. Step 7's divergence
guard correctly fired at that toy scale (400 of 7200 angles informing the
checkpoint, searched against all 7200) and refused to write a bad
correction, so REBUILD THE SEED and pass 2 have never actually executed --
only the real ladder, with pass 1 built from all 7200 angles, can exercise
those stages meaningfully.
