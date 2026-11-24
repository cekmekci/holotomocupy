# bin-2 ablations — initialisation, the tomography mask, and freezing the aperture

Eight independent bin-2 step6 runs, each differing from its reference in
exactly one thing. No ladder, no step 7, no seed rebuild.

| folder | object starts as | aperture starts as | aperture refined? | probe starts as | `mask` | misfit |
|---|---|---|---|---|---|---|
| `test1_obj0/` | zero | the real `code.npy` from step 0 | yes | the NFP-measured probe | 1.1 | intensity |
| `test2_obj0_code1/` | zero | uniform transparent plane (all `1+0j`) | yes | the NFP-measured probe | 1.1 | intensity |
| `test3_control/` | the step-5 Paganin+FBP seed | the real `code.npy` from step 0 | yes | the NFP-measured probe | 1.1 | intensity |
| `test4_mask075/` | the step-5 Paganin+FBP seed | the real `code.npy` from step 0 | yes | the NFP-measured probe | **0.75** | intensity |
| `test5_prb1/` | the step-5 Paganin+FBP seed | the real `code.npy` from step 0 | yes | **1** | 1.1 | intensity |
| `test6_obj0_prb1/` | zero | the real `code.npy` from step 0 | yes | **1** | 1.1 | intensity |
| `test7_codefix_prb1/` | the step-5 Paganin+FBP seed | the real `code.npy` from step 0 | **no, frozen** | **1** | 1.1 | intensity |
| `test8_codefix_prb1_amp/` | the step-5 Paganin+FBP seed | the real `code.npy` from step 0 | **no, frozen** | **1** | 1.1 | **amplitude** |
| `test9_codefix_prb1_nolap/` | the step-5 Paganin+FBP seed | the real `code.npy` from step 0 | **no, frozen** | **1** | 1.1 | intensity, **`lam_laplacian=0`** |
| `test10_cold_codefix_nolap/` | **zero** | the real `code.npy` from step 0 | **no, frozen** | **1** | 1.1 | intensity, **`lam_laplacian=0`** |
| `test11_cold_mask064/` | **zero** | the real `code.npy` from step 0 | **no, frozen** | **1** | **0.64** | intensity, `lam_laplacian=0` |
| `test12_cold_amplitude/` | **zero** | the real `code.npy` from step 0 | **no, frozen** | **1** | 1.1 | **amplitude**, `lam_laplacian=0` |

Tests 1-3 vary the starting point of `obj`/`code`. Test 4 is test 3 with a
tighter tomography mask. Test 5 is test 3 with the probe forced to 1. Test 6
is test 1 with the probe also forced to 1 (so object AND probe both start
cold, aperture stays real).

Test 7 is test 5 with the aperture FROZEN. Test 8 is test 7 with the amplitude
misfit, and test 9 is test 7 with the Laplacian off. Tests 8 and 9 are each one
line off test 7. `diff` against those, not against the control.

**Tests 7-9 also unfreeze the object positions and tune rho (2026-11-19), so
they are NOT one line off test 5 and are not directly comparable to tests 1-6.**
Tests 1-6 all inherit `rho=1,0.03,0,0.03,0.01` from
`../config_step6_bin2_pass1.conf`, which has `rho[pos]=0` and therefore never
moves the object positions. These three runs exist to hunt the ring and axis
artefacts, and a frozen misregistration is a prime suspect for those, so
leaving positions pinned would have made the runs unable to answer their own
question. They use `rho=1,0.03,0.01,0,0.01` with `estimate_rho=True`.

Two consequences of that, both load-bearing.

`rho[code]=0` rather than 0.03. `freeze_code=True` already zeroes the code
gradient, but `estimate_rho_coord` skips only entries `<= 0`, so a nonzero
value would make the search spend trials probing a variable that cannot move.
For the same reason, turning `estimate_rho` on while leaving `rho[pos]=0` would
be a no-op for positions -- the start value has to be nonzero or the coordinate
is treated as frozen.

Wall time stays at 1:00:00. Every test in this folder runs in `-q debug`, which
caps at one hour, so 2:00:00 will simply be rejected at submission. Be aware
that `estimate_rho` adds roughly 40% on top of `niter=512` -- three to five
trials of `rho_estimate_niter=16` for each of prb, pos and pos_code -- so tests
7 to 9 may not finish inside the hour. If they are cut short, drop `niter` for
those three rather than raising the wall time, and keep it identical across
whichever tests you intend to compare.

Nothing in `..` is modified or copied. Each test's `polaris_run.sh` references
`../../step6.py`, `../../preflight.py`, `../../set_affinity_gpu_polaris.sh` and
`../../../polaris_env.sh` in place, and each writes to its own `path_out`, so
the nine can run in any order and never collide.

## Why tests 7 and 8 exist (2026-11-19)

Measured on the step-0 reconstruction, against the frames it was itself fitted
to and through the solver's own windowed shifter: the residual is 3.06% rms,
where pure Poisson at the detector's measured 5.6 ADU/quantum and 7665 ADU
above dark predicts 2.70%. So about 78% of that residual is photon noise. It
also decorrelates completely between code positions (mean correlation +0.004
over 9 frames) and carries no probe signature (−0.06 against `|D(prb)|²`). The
aperture is measured about as well as that data allows and there is no large
structured error left in it.

If that holds, refining `code` again inside step 6 — where it is entangled with
an unknown object — can only add variance. Test 7 asks whether freezing it
helps. `freeze_code=True` zeroes the gradient (`rec_coded_mpi.py:1024`), which
makes `rho[code]` inert; `rho` is deliberately left untouched so the diff
against test 5 stays a single line.

Two dead ends, recorded so they are not re-tried. `z1c` cancels exactly out of
the code's apparent scale — the NFP grid is `det_px·|z1c|/fdd` and the
code-to-sample demagnification is `|z1/z1c|`, so the product is the HT voxel
size for any `z1c` — and an 11-point NFP sweep over −18.00 to −17.50 mm moved
the reconstructed code by only 0.3–1.1% rms phase. Separately, the `CODE_SCALE`
factor from the step-5 ablation notebook should not be applied.

**On test 8 and the amplitude misfit.** It was dropped on 2026-11-10 because it
let `pos` jump to wrong positions at this bin. When test 8 was first written
the positions were frozen and that failure mode was out of reach, but since
they were unfrozen on 2026-11-19 it is back in play and this run is
deliberately exposed to it. If `pos` diverges here that reproduces the
2026-11-10 observation rather than contradicting it, so read the position drift
plot before anything else in this run. The reason to try it at all is that if
the residual really is photon-limited then the Poisson-weighted amplitude model
is the better-matched misfit.

`lam_laplacian` MUST drop from 8e-5 to 2e-5 alongside it. The misfit shrinks
about 4x between intensity and amplitude while `||grad^2 u||^2` does not, so the
relative weight is divided by the same 4 (`extra_terms.py:184-193`). `rec_mpi`
warns about this asymmetry at startup. `rec_coded_mpi` does **not**, so nothing
will catch it if it is forgotten.

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

## Why test 9 exists -- the waves at the top and bottom of each projection

Observed in the 2D projections, at the top and bottom borders only, still
present when the iterations start from `obj = 0`, and surviving binning.

**The Fresnel propagator is ruled out, measured rather than argued.**
`Propagation` mirror-pads to 2n, and its `DT` zero-fills the centre and then
uses `_adj_pad`, so the pair is a true adjoint and not a conjugate-kernel
approximation. A dot-product test gives `<Dx,y>` against `<x,DTy>` to 7.2e-7 at
512x512 and 4.0e-7 at a non-square 256x320, and `DT(D x)` puts no excess energy
at any edge. Binning does not discriminate either way -- the Fresnel length is
a physical 0.331 um and the Paganin length 3.212 um, so both shrink in pixels
with binning while staying the same physical size, and binning averages down
noise, not a deterministic artefact.

**The candidate that fits is the biharmonic regularizer's boundary
condition.** `extra_terms.py:80-81` zeroes the ghost slices wherever a rank has
no neighbour, which is the global first and last object slices. The object is
split along z and z is a projection's vertical axis, so those are exactly the
top and bottom borders. The penalty is `(grad^2)^2 u`, so a zero ghost asserts
that the object falls abruptly to nothing just outside the volume, and the
gradient pushes on the end slices every iteration regardless of where the
object started. The left and right edges sit inside a rank and see none of it.

That mechanism *would* produce the artefact; it has not been shown to *be*
producing it. Zero ghosts are the correct condition if the end slices really
are vacuum, and `nzobj = 1.25*nz` does leave a z margin that ought to be empty.
Test 9 settles it. Waves gone means it is this boundary condition, and the fix
is a zero-flux condition rather than a zero one. Waves still there clears the
Laplacian as well, and `mask_oob` is the next thing to look at.

`lam_laplacian=0` is safe to set: `rec_coded_mpi.py:110` gates the
`rho[obj] == 1` constraint on `lam_laplacian > 0`, and the term is simply never
constructed.

## Why test 10 exists -- import nothing but the code (2026-11-20)

One line off test 9, with `obj_init_file` removed so the object starts at zero.
The only thing carried in from another measurement is the aperture.

The reason is a measurement. Running the `probe-fixed` cells of
`step5_paganin_ablation` on the HT data gave

```
vacuum pixels                   61% of the cropped frame
fixed in the detector frame     69% of the vacuum variance
removing it changes hp-RMS      0.01540 -> 0.01543  (+0.1%)
per-angle level                 mean -0.055 (model ~5.7% too bright), std 0.0052
```

69% of the residual in the sample-free region is **fixed in the detector
frame**, and the chance floor for a median over 900 angles is about 0.2%, so
that is real. Because the probe is the only thing that does not move -- the
code travels 3322 px and the sample rotates -- that component is probe-side.
It is also smooth, which is why dividing it out leaves the high-pass metric
untouched: the fine code-shaped texture is in the other 31%. In log units the
fixed map spans about +-0.037, and Paganin at `db=30` multiplies that by 15,
so roughly +-0.55 rad of phase error present at every angle.

The catch is that this error lives on the step-5 path, which builds its flat
field from the **NFP probe**. So `obj_init.npy` has it baked in, and tests 5
and 7-9 inherit it through the seed even though their own probe starts at 1.
Test 10 is the only configuration in the folder that imports neither the probe
nor the seed, so it is the one that can say whether the artefacts survive
without them.

Read it against test 9, which differs only by having the seed. Artefacts gone
means the seed was carrying them. Artefacts still there means it is neither the
probe nor the seed, and the cause is structural. Measured on test8's
`obj_re_vert` for comparison: background swing about 30% of the slice range, a
13% dip 120 px wide centred on the rotation axis, and 6% high-frequency noise.

**Result, 2026-11-20.** The vertical lines on the axis are GONE in test 10 and
were present in test 8. That points at the step-5 Paganin seed carrying them
in rather than the step6 iterations generating them. Noise and the edge
artefacts both remain.

Two things had to be fixed before it would run. `estimate_rho` is off here, and
`rho[pos]` is 0. With `obj = 0` the object is transparent, contributes nothing
to the model, and `d(model)/d(pos)` carries no information -- you would be
refining the position of an object that does not exist yet. The first attempt
left positions free with the rho search on, and because a larger `rho[pos]`
lets positions absorb the missing probe envelope over a 16-iteration trial, the
search extended into a divergent step size and `pos` passed 100 px within a few
iterations. Tests 7-9 start from the seed, so positions are meaningful there
and stay free.

## Why test 11 exists -- the mask was never clipping anything

One line off test 10, `mask` from 1.1 to 0.64.

`mask` is a SQUARED radius in units where the object half-width is 1
(`tomo.py:62`), so

```
mask = 1.1    -> radius 1.0488 = 335.6 voxels   (PAST the 320 grid half-width)
mask = 0.75   -> radius 0.8660 = 277.1 voxels   (test4)
fully measured radius = detector half-width     = 256 voxels
value that clips to it = (n/nobj)^2 = (512/640)^2 = 0.64
```

At 1.1 the circle lies outside the grid entirely and clips nothing. A voxel at
radius `r` projects into `[-r, +r]` as the sample turns, and the detector only
spans `+-256`, so every voxel beyond 256 falls off the detector for part of the
rotation. Partially measured means badly conditioned, and that annulus from 256
to 320 -- out to 452 in the corners -- is where the left and right edge junk
lives. Even test4's 0.75 leaves part of it in.

Note the top and bottom are a different case. z is the rotation axis, the
detector spans 512 in z against the object's 640, so voxels outside that are
never measured at any angle rather than partially. The circular mask does not
address those at all.

0.64 is not a tuned value, it is the circle every angle actually sees, and it
is the right value for any dataset with this grid ratio. It is only valid while
the sample fits inside radius 256, which the step-5 vacuum mask (61% vacuum)
indicates it does.

**On the noise in the vacuum.** Expected, and not a bug. Nothing in this
reconstruction makes vacuum zero -- no positivity, no support constraint, and
with `lam_laplacian=0` no regularizer either. It is a least-squares fit to noisy
data, and projection noise back-projects across the whole volume including
regions that are physically empty. The useful part is that its level is a direct
measurement of the reconstruction noise floor.

## Why test 12 exists -- does the misfit choice account for the noise?

One line off test 10, `model` from intensity to amplitude.

The misfit does not create noise, it weights it. Intensity assumes uniform
variance, so it over-weights bright pixels where the variance is in fact
larger, which is a less efficient estimator and leaves more noise in the result
for the same data. The square root is the variance-stabilising transform for
Poisson, which is why amplitude is the maximum-likelihood choice when photon
noise dominates and intensity is right when read noise does. See Thibault and
Guizar-Sicairos, New J. Phys. 14 063004 (2012).

**Which regime this data is in has not been measured.** The 5.6 ADU/quantum
figure quoted earlier in this folder came from an older note and was never
verified for this detector or this scan, and if the noise is PSF-correlated
then neither model is strictly right, because both assume independent pixels.
A separate detector characterisation (photon transfer curve, noise power
spectrum) is underway to settle it. Until then test 12 is an empirical A/B
against test 10, not a principled choice.

Two traps that do not apply here, both of which have bitten this folder before.
`rho[pos]=0`, so the 2026-11-10 failure where amplitude let `pos` jump to wrong
positions is genuinely out of reach -- unlike test 8, where that same claim
stopped being true once positions were unfrozen. And `lam_laplacian` normally
has to be divided by about 4 when switching to amplitude
(`extra_terms.py:184-193`, and `rec_coded_mpi` does not warn about it), but it
is already 0 here.
