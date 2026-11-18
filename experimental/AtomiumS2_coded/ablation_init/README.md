# bin-2 ablations — initialisation, and the tomography mask

Six independent bin-2 step6 runs, each differing from the control in exactly
one thing. No ladder, no step 7, no seed rebuild.

| folder | object starts as | aperture starts as | probe starts as | `mask` |
|---|---|---|---|---|
| `test1_obj0/` | zero | the real `code.npy` from step 0 | the NFP-measured probe | 1.1 |
| `test2_obj0_code1/` | zero | uniform transparent plane (all `1+0j`) | the NFP-measured probe | 1.1 |
| `test3_control/` | the step-5 Paganin+FBP seed | the real `code.npy` from step 0 | the NFP-measured probe | 1.1 |
| `test4_mask075/` | the step-5 Paganin+FBP seed | the real `code.npy` from step 0 | the NFP-measured probe | **0.75** |
| `test5_prb1/` | the step-5 Paganin+FBP seed | the real `code.npy` from step 0 | **1** | 1.1 |
| `test6_obj0_prb1/` | zero | the real `code.npy` from step 0 | **1** | 1.1 |

Tests 1-3 vary the starting point of `obj`/`code`. Test 4 is test 3 with a
tighter tomography mask. Test 5 is test 3 with the probe forced to 1. Test 6
is test 1 with the probe also forced to 1 (so object AND probe both start
cold, aperture stays real).

Nothing in `..` is modified or copied. Each test's `polaris_run.sh` references
`../../step6.py`, `../../preflight.py`, `../../set_affinity_gpu_polaris.sh` and
`../../../polaris_env.sh` in place, and each writes to its own `path_out`, so
the six can run in any order and never collide.

## Where the configs came from

Each `config_step6_bin2.conf` is `../config_step6_bin2_pass1.conf` verbatim
except for `path_out`, `obj_init_file`, `code_file` and (test 4 only) `mask`.
Confirm that at any time with

    diff <(grep -v '^#' ../config_step6_bin2_pass1.conf | grep -v '^\s*$') \
         <(grep -v '^#' test3_control/config_step6_bin2.conf | grep -v '^\s*$')

which should show only those lines. `test4_mask075` is likewise `test3_control`
plus two lines:

    diff <(grep -v '^#' test3_control/config_step6_bin2.conf | grep -v '^\s*$') \
         <(grep -v '^#' test4_mask075/config_step6_bin2.conf | grep -v '^\s*$')

Three details worth knowing, because all are load-bearing.

**The zero-object tests omit `obj_init_file` entirely rather than pointing it at
a missing path.** `preflight.py` treats an absent key as a warning ("the object
starts from ZERO") but a set-but-missing path as a hard error, and
`CodedReader.read_obj` only takes its zero branch when the key is unset.

**The "zero" aperture is all ones, not all zeros.** `code = 0` is a fully opaque
mask — the forward model would predict no signal anywhere and the gradient would
be degenerate. `code = 1` is the transparent, no-prior-structure start, the
analogue of `vars['proj'][:] = 0` in step 0, which stores the phase and saves
`exp(1j*proj) = 1`. `rho[code] = 0.03` with `freeze_code=False`, so the aperture
really is refined during the run.

**`mask` is a SQUARED radius, not a radius.** `config.mask` is passed straight
into `Tomo(...)` as `mask_r`, and `tomo.py:62` builds the circle as
`circ = (x**2 + y**2 < mask_r)` on a grid whose half-width is 1. So `mask=1.1`
is `r = 1.0488` — past the inscribed circle, barely clipping anything — and
test 4's `mask=0.75` is `r = 0.8660`. If a radius of 0.75 is what you want, the
value is 0.5625. Same gotcha as `MASK_R` in the step-5 ablation notebook.

**Forcing the probe to 1 needs a different `in_file`, not a config key.**
There is no `prb_file` option for step6 — `CodedReader.read_prb` always reads
`/exchange/prb` out of `in_file` (the same file `pdata` comes from), with no
override argument. It already has a documented cold-start branch
(`"Ones when absent"`), so `make_prb1_codedht.py` builds a tiny sidecar,
`codedHT_noprb.h5`, that is `codedHT.h5` minus `/exchange/prb` -- via HDF5
EXTERNAL LINKS, so the 35 GB `pdata` is never duplicated, only a few KB of
pointers. Generate it ONCE before running test5 or test6:

    python make_prb1_codedht.py /eagle/APS_IRI/cekmekci/AtomiumS2_coded/ht/codedHT.h5

**`preflight.py` will hard-fail on that shim unless told not to check.** Its
`REQUIRED` tuple includes `/exchange/prb` unconditionally, so it treats the
cold-start probe as an error even though `CodedReader` supports it on purpose.
`preflight.py` is shared, outside this folder, so rather than edit it, skip it
for just these two tests:

    cd test5_prb1        && qsub -v PREFLIGHT=0 polaris_run.sh
    cd test6_obj0_prb1   && qsub -v PREFLIGHT=0 polaris_run.sh

## Running them

Once, before the first submission — `test2` needs the flat aperture, `test5`
and `test6` need the no-prb shim, and `lfs setstripe` does not create missing
parents:

    python make_flat_code.py /eagle/APS_IRI/cekmekci/AtomiumS2_coded/nfp/flat_code.npy
    python make_prb1_codedht.py /eagle/APS_IRI/cekmekci/AtomiumS2_coded/ht/codedHT.h5

    for t in test1_obj0 test2_obj0_code1 test3_control test4_mask075 test5_prb1 test6_obj0_prb1; do
        d=/eagle/APS_IRI/cekmekci/AtomiumS2_coded/rec/ablation_init/$t
        mkdir -p "$d" && lfs setstripe -c 8 -S 16M "$d"
    done

Then, per test (test5 and test6 need `PREFLIGHT=0`, see above):

    cd test1_obj0 && qsub polaris_run.sh

## Sizing

1 node on `debug`, 1 hour, `ntheta=1200, stride=6` (1200 of the 7200 angles,
ids = start_theta + stride*arange(ntheta)).

Running the full 7200 angles would need ~5 hours on 1 node (24x more angles
per rank than the measured 24-node run). Subsampling cuts that down, but not
by picking an arbitrary stride: angular undersampling itself produces streak
artifacts in CT, which would confound the very thing these tests are trying
to observe. `ntheta=1200, stride=6` stays above the `(pi/2)*nobj ~= 1005`
Nyquist angular-sampling estimate at bin2 (`nobj=640`), so it should not add
undersampling streaks of its own. From the 24-node baseline (766s for 75
angles/rank), 1 node at 300 angles/rank (4x that load) is an estimated ~51
minutes -- inside the 1-hour cap, but without much margin.

If a test runs long, **drop `niter` in the config rather than raising the
walltime**. The comparison only needs all six to use the *same* `niter` and
the *same* `stride`, not a converged result.

## Reading the result

The six runs are comparable through the `err=` line in each `pbs-*.out`, and
through the checkpoints in each `path_out` (`checkpoint_step=32`). What the test
answers is whether the Paganin seed actually buys anything over a cold start by
iteration 512 — and if `test1` catches up with `test3`, the seed is not doing
the work it is assumed to be doing.

`test2` is the harder question: whether the aperture is recoverable from the
holotomography data alone, without step 0 having measured it. A failure there is
informative rather than surprising.

`test4` is a different axis entirely — same start as `test3`, so comparing the
two isolates what the tomography mask costs or buys. A tighter mask discards
reconstruction outside r = 0.866, which should suppress whatever lives in the
corners at the price of throwing away any real sample that reaches past it.

`test5` isolates the probe's own contribution: identical to `test3` except the
probe starts at 1 instead of the NFP-measured field, so any gap between the
two is specifically what the measured probe was buying. `test6` does the same
probe ablation but starting from test1's cold object, so object AND probe are
both cold while the aperture stays real -- the hardest cold-start case here.
