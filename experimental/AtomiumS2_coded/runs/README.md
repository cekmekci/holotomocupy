# Runs

**One folder here = one run.** The folder holds that run's six configs and its
`polaris_run.sh`, and nothing else in the tree knows about it.

## Layout

```
AtomiumS2_coded/                  shared code, shared-input configs
  config_step0.conf               builds nfp/   -- NOT a run, read by step0.py,
                                  step0a/0b/3/5 notebooks, nfp_sweep, paganin_sweep
  config_steps15.conf             builds ht/    -- NOT a run, read by steps15.py,
                                  step1/2/3/5 notebooks, paganin_sweep
  step0.py  steps15.py  step6.py  step7.py  preflight.py  ...   shared, never copied
  runs/
    baseline/                     <- copy this to make a new run
      config_step6_bin2_pass1.conf
      config_step6_bin1_pass1.conf
      config_steps15_corrected.conf
      config_step6_bin2.conf
      config_step6_bin1.conf
      config_step6_bin0.conf
      polaris_run.sh
      correct_correct3D_extra.txt     written here by step7, automatically
      pbs-<jobid>.out
```

On eagle (and the same shape on tomo5's `/data3`):

```
AtomiumS2_coded/
  ht/    codedHT.h5   obj_init.npy            SHARED, read-only, never run-specific
  nfp/   data_nfp.h5  code.npy  prb.npy       SHARED, read-only
  rec/<run>/
         obj_init_corrected.npy               per-run: depends on this run's step7
         pass1/
         refined/
```

Filenames are **identical in every run**. Only `<run>` changes. `ht/` and
`nfp/` are inputs and are never written to by a run.

## How a run names itself

Every config in a run folder starts with the same two keys:

```
run=baseline
rec=/eagle/APS_IRI/cekmekci/AtomiumS2_coded/rec
```

Everything run-specific is derived from them by configparser's
`BasicInterpolation` (enabled in `holotomocupy/config.py` `_load`):

| key | pass 1 | pass 2 |
|---|---|---|
| `path` | `.../ht` shared | `.../ht` shared |
| `path_out` | `%(rec)s/%(run)s/pass1` | `%(rec)s/%(run)s/refined` |
| `obj_init_file` | `obj_init.npy` — shared, from `ht/` | `%(rec)s/%(run)s/obj_init_corrected.npy` |
| `correct3d_extra_file` | *(absent)* | `correct_correct3D_extra.txt` |

Two details that make this work and are worth knowing:

- `obj_init_file` takes an **absolute** path if you give it one
  (`_data_rel`, `config.py:538`), which is how the corrected seed escapes the
  shared `ht/` folder while `path` still points there for `codedHT.h5`.
- `correct3d_extra_file` resolves against the **config's own directory**
  (`cfg.rel`, `config.py:71`), so each run's correction file is isolated with
  no configuration at all. This is the mechanism the whole convention rests on.

## Making a new run

```
cp -r runs/baseline runs/pag1
sed -i 's/^run=.*/run=pag1/'                   runs/pag1/*.conf
sed -i 's/^#PBS -N .*/#PBS -N AtomS2c_pag1/'   runs/pag1/polaris_run.sh
```

Then edit **only the parameter the run is about** — e.g. `paganin=1.0` in
`config_steps15_corrected.conf`, or `rho=1,0.03,0,0,0.01` to freeze the code.

Once, on eagle:

```
R=/eagle/APS_IRI/cekmekci/AtomiumS2_coded/rec
mkdir -p $R/pag1/pass1 $R/pag1/refined
lfs setstripe -c 8 -S 16M $R/pag1/pass1
lfs setstripe -c 8 -S 16M $R/pag1/refined
```

Then `cd runs/pag1 && qsub polaris_run.sh`.

To check a new run before submitting it:

```
python -c "
import sys,glob; sys.path.insert(0,'../../../../src')
from holotomocupy import config as C
for f in sorted(glob.glob('*.conf')):
    p = C.parse_args_coded_steps15(f) if 'steps15' in f else C.parse_args_coded(f)
    print(f, p.path_out if hasattr(p,'path_out') else '', p.obj_init_file)"
```

Every path printed must contain `/rec/<your run>/` and no literal `%(`.

## Shared inputs are built once, not per run

`obj_init.npy` is the δ/β=30 Paganin+FBP seed in `ht/`, shared by every run.
It is built on tomo5 by `steps15.py ../../config_steps15.conf` (stage 5) and
transferred to eagle. Only change it if a run genuinely needs a different
*starting* object (e.g. zeros) — in that case point pass 1's `obj_init_file`
at a different file in `ht/` rather than renaming the shared one.

## Making a cheap run

A smoke test is not a special mechanism, it is a run at coarse binning.
**`toy/` already is one** — run it before committing a 24-node/3-hour prod
allocation:

```
R=/eagle/APS_IRI/cekmekci/AtomiumS2_coded/rec
mkdir -p $R/toy/pass1 $R/toy/refined
lfs setstripe -c 8 -S 16M $R/toy/pass1
lfs setstripe -c 8 -S 16M $R/toy/refined
cd runs/toy && qsub polaris_run.sh
```

bin 4 then 3, 400 of 7200 angles, all five stages, 2 nodes on the debug
queue, about an hour. It reads the same shared `ht/` and `nfp/` inputs as
every other run and writes to `rec/toy/`.

To build another one: copy `baseline`, delete `config_step6_bin0.conf` and
its `mpiexec` line, set `ntheta=400`, `stride=18`, small `niter`/`start_iter`,
`#PBS -q debug`, `#PBS -l select=2`, `#PBS -l walltime=01:00:00`, and take the
size scalars from this table — it is exact, verified against all five rungs
that have existed:

```
bin   n = nz   nobj = nzobj   ncode   nchunk
 0      2048        2560       8192      4
 1      1024        1280       4096      8
 2       512         640       2048     16
 3       256         320       1024     32
 4       128         160        512     64

n = nz = 2048 / 2^bin        nobj = nzobj = 2560 / 2^bin
ncode  = 8192 / 2^bin        nchunk = 4 * 2^bin
```

The old toy ladder used bin 4 then bin 3, `niter` 32 then 48, with
`checkpoint_step=16` and `error_step=8`.

Two things the toy scale does **not** prove, both learned the hard way:

- `step7 --ntheta` must match the toy's own `ntheta` (400), not the raw scan's
  7200. Step 7 picks its search angles by an even stride over the full scan
  regardless of what the checkpoint was built from, so passing 7200 forward-
  projects at thousands of angles the object was never built from, and the
  autofocus diverges hard.
- the seed-rebuild stage is **not** cheap at toy scale. `steps15.py` stage 5
  reads `ntheta`/`nz`/`n` off the raw scan and ignores `cfg.ntheta`/`cfg.n`,
  so that one stage costs full price however coarse the rest of the ladder is.
