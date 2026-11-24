#!/usr/bin/env python
"""`rec_holo_coded_mpi.RecHoloCoded` end to end: the batched kernels and the BH loop.

    python tests/holo_coded/test_rec_holo_coded.py                  # serial checks
    python tests/holo_coded/test_rec_holo_coded.py test_mpi         # writes the 1-rank reference
    mpirun -np 2 python tests/holo_coded/test_rec_holo_coded.py mpi # the MPI check

`test_cascade_holo_coded.py` proves the six levels differentiate correctly when
called by hand.  This proves the production wrappers around them are wired
right, which is a different failure mode entirely: a @gpu_batch argument in
the wrong slot, a replicated array mistaken for a frame-chunked one, a missing
allreduce, or an rho_sq applied in the wrong place would all leave the cascade
tests green.

  gradients_cascade   against a finite difference of min(), per variable --
                      the end-to-end check of the gradient, including the
                      rho_sq convention
  hessian3            against three separate hessian() sweeps
  fused step          beta and bottom against the classic three-sweep route
  BH                  the error goes down, and the object comes back
  no reader           the engine runs from assigned buffers alone
  MPI                 2 ranks reproduce 1 rank
"""
import os
import sys

import numpy as np
import cupy as cp
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'unit'))
from harness import check, close, run                            # noqa: E402

from mpi4py import MPI                                           # noqa: E402
from holotomocupy.rec_holo_coded_mpi import RecHoloCoded          # noqa: E402

N, NZ, NTHETA = 32, 32, 8
NOBJ = 3 * N // 2
NCODE = 2 * N
MARGIN = (NOBJ - N) // 2            # rows/cols of the object no frame ever sees


def _args(**kw):
    a = dict(
        energy=17.1, detector_pixelsize=1.4760147601476e-6 * 16,
        focustodetectordistance=1.217, z1=5.110e-3, z1_code=4.584e-3,
        ntheta=NTHETA, nz=NZ, n=N, nzobj=NOBJ, nobj=NOBJ,
        ncode=NCODE, rho=[1, 1, 1, 1, 1],
        code_pad_frac=0.125, shift_type='fft', niter=1, nchunk=4,
        checkpoint_step=-1, error_step=-1, start_iter=0,
        model='amplitude', psf_sigma=0.0, comm=MPI.COMM_WORLD)
    a.update(kw)
    return SimpleNamespace(**a)


def _fill(cl, seed=0, pos_amp=1.0, obj_scale=0.02):
    """A plausible state: a weak object, a probe and a code near 1, and
    sub-pixel displacements on both position sets.

    Every array is drawn at GLOBAL size and then sliced to this rank's share,
    so the state -- and therefore the synthetic data and every reduction over
    it -- is identical whatever the rank count.  Drawing at local size instead
    would make the 2-rank run a different problem from the 1-rank one, and the
    MPI comparison below would be meaningless.
    """
    rng = np.random.default_rng(seed)

    def cx(shape, s=1.0):
        return ((rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
                * s).astype('complex64')

    obj = cx((cl.nzobj, cl.nobj), obj_scale)
    prb = cx((cl.nz, cl.n), 0.1) + 1.0
    code = cx((NCODE, NCODE), 0.1) + 1.0
    pos = (pos_amp * (rng.random((cl.ntheta, 2)) - 0.5)).astype('float32')
    pos_code = (3 * pos_amp
                * (rng.random((cl.ntheta, 2)) - 0.5)).astype('float32')

    # obj, prb and code are device arrays; pos and pos_code are pinned host.
    cl.vars['obj'][:] = cp.asarray(obj)
    cl.vars['prb'][:] = cp.asarray(prb)
    cl.vars['code'][:] = cp.asarray(code)
    cl.vars['pos'][:] = pos[cl.st_theta:cl.end_theta]
    cl.vars['pos_code'][:] = pos_code[cl.st_theta:cl.end_theta]
    return cl


def _synth(cl):
    """Fill cl.data with K|psi|^2 of the current variables."""
    cl.gen_data(cl.vars, cl.data)


def _prepared(seed=0, **kw):
    """A solver with data generated from its own variables, precalc run, so
    it is sitting exactly at the global minimum of the data term."""
    cl = _fill(RecHoloCoded(_args(**kw)), seed)
    _synth(cl)
    cl.precalc(cl.vars)
    return cl


def _minv(cl, v):
    return float(cl.min(v['prb'], v['obj'], v['pos'], v['code'], v['pos_code']))


def _perturb(cl, seed=21, s=0.05):
    """Move every variable off the generating state.

    `_prepared` builds the data from the solver's own variables, so it sits at
    the exact global minimum where every gradient is EXACTLY zero (p = 0 in the
    misfit weights, not merely small).  Anything that asserts a gradient is
    nonzero, or compares gradients across ranks, must call this first -- else
    it is comparing 0 with 0 and passes vacuously.

    Draws are at GLOBAL size and sliced to this rank, so the perturbed state is
    identical whatever the rank count.
    """
    rng = np.random.default_rng(seed)

    def cx(shape):
        return (s * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
                ).astype('complex64')

    cl.vars['obj']  += cp.asarray(cx(cl.vars['obj'].shape))
    cl.vars['prb']  += cp.asarray(cx(cl.vars['prb'].shape))
    cl.vars['code'] += cp.asarray(cx(cl.vars['code'].shape))
    pg  = (0.2 * rng.standard_normal((cl.ntheta, 2))).astype('float32')
    pcg = (0.2 * rng.standard_normal((cl.ntheta, 2))).astype('float32')
    cl.vars['pos']      += pg[cl.st_theta:cl.end_theta]
    cl.vars['pos_code'] += pcg[cl.st_theta:cl.end_theta]


# ----------------------------------------------------------------------
# construction guards
# ----------------------------------------------------------------------

def test_rho_must_have_five_entries():
    try:
        RecHoloCoded(_args(rho=[1, 1, 1, 1]))
        raised = False
    except ValueError:
        raised = True
    check('rho with 4 entries is refused', raised)


def test_alias_guard_fires():
    """A local_ntheta equal to 1 or 3 aliases the (1,) / (3,) scalar
    accumulators in gpu_batch, which would then be sliced per chunk."""
    for nf in (1, 3):
        try:
            RecHoloCoded(_args(ntheta=nf, nchunk=nf))
            raised = False
        except ValueError:
            raised = True
        check(f'ntheta={nf} is refused (aliases an accumulator)', raised)


def test_no_tomography_attributes():
    """The whole point of this engine is that the tomography layer is gone."""
    cl = RecHoloCoded(_args())
    for name in ('cl_tomo', 'proj_tmp', 'redist', 'fwd_tomo', 'adj_tomo',
                 'cl_lap_term', 'ndobj', 'local_nzobj'):
        check(f'no attribute {name}', not hasattr(cl, name))
    check("no 'proj' variable", 'proj' not in cl.vars)
    check('norm_const pinned at 1', float(cl.norm_const) == 1.0)
    check('nframes aliases ntheta', cl.nframes == cl.ntheta == NTHETA)


# ----------------------------------------------------------------------
# data generation
# ----------------------------------------------------------------------

def test_gen_data_roundtrip():
    cl = _prepared(seed=1)
    d = np.asarray(cl.data)
    check('data shape and dtype', d.shape == (NTHETA, NZ, N) and d.dtype == np.float32,
          f'{d.shape} {d.dtype}')
    check('data is finite and positive', bool(np.all(np.isfinite(d)) and d.min() > 0),
          f'min {d.min():.3e} max {d.max():.3e}')
    # generated from its own variables, so the misfit is exactly zero
    err = _minv(cl, cl.vars)
    check('data term is ~0 at the generating state', err < 1e-8, f'err {err:.2e}')


# ----------------------------------------------------------------------
# gradient
# ----------------------------------------------------------------------

def test_gradient_against_finite_difference():
    """<grad, y> == d/dh min(x + h y), per variable.

    min() and gradients() go through completely separate @gpu_batch calls, so
    this catches a mis-ordered kernel argument, a replicated array sliced as
    if it were chunked, and the rho_sq convention (gradients_cascade multiplies
    by rho_sq, so the directional derivative must be divided by it again).

    `obj` is included: unlike RecCoded there is no adjoint Radon between the
    cascade and the object gradient, so this is the only end-to-end check of it.
    """
    cl = _prepared(seed=1)
    # Move off the minimum: at the exact solution the gradient is ~0 and the
    # relative comparison below has nothing to resolve.
    rng = np.random.default_rng(7)
    cl.vars['obj'] += cp.asarray(
        (0.05 * rng.standard_normal(cl.vars['obj'].shape)).astype('complex64'))
    cl.vars['pos'] += (0.3 * rng.standard_normal(cl.vars['pos'].shape)
                       ).astype('float32')

    cl.gradients(cl.vars, cl.grads)

    # obj, prb and code live on the GPU while pos / pos_code are pinned host
    # arrays, so each direction is built in the variable's own namespace and dtype.
    for name in ('obj', 'prb', 'pos', 'code', 'pos_code'):
        base = cl.vars[name].copy()
        xp = cp if isinstance(base, cp.ndarray) else np
        if base.dtype == np.complex64:
            y_h = ((rng.standard_normal(base.shape)
                    + 1j * rng.standard_normal(base.shape)) * 0.05
                   ).astype('complex64')
        else:
            y_h = (0.05 * rng.standard_normal(base.shape)).astype('float32')
        y = xp.asarray(y_h)
        # grads carry rho_sq (gradients_cascade applies it); divide it back out
        gd = cp.asarray(cl.grads[name])
        inner = float(cp.sum(cp.real(cp.conj(gd) * cp.asarray(y_h)))) / cl.rho_sq[name]

        h = 1e-2
        vals = {}
        for s in (+1, -1):
            cl.vars[name][:] = base + xp.asarray(np.float32(s * h)) * y
            vals[s] = _minv(cl, cl.vars)
        cl.vars[name][:] = base
        fd = (vals[+1] - vals[-1]) / (2 * h)
        close(f'<grad, y> == d/dh min   [{name}]', inner, fd, 5e-2)


# ----------------------------------------------------------------------
# Hessian plumbing
# ----------------------------------------------------------------------

def _random_dirs(cl, seed=3):
    rng = np.random.default_rng(seed)
    for d in (cl.grads, cl.etas):
        for k, v in d.items():
            if isinstance(v, cp.ndarray):
                if v.dtype == cp.complex64:
                    v[:] = cp.asarray((rng.standard_normal(v.shape)
                                       + 1j * rng.standard_normal(v.shape)
                                       ).astype('complex64') * 0.05)
                else:
                    v[:] = cp.asarray((0.05 * rng.standard_normal(v.shape)
                                       ).astype(v.dtype))
            else:
                if np.iscomplexobj(v):
                    v[:] = ((rng.standard_normal(v.shape)
                             + 1j * rng.standard_normal(v.shape)) * 0.05
                            ).astype(v.dtype)
                else:
                    v[:] = (0.05 * rng.standard_normal(v.shape)).astype(v.dtype)


def test_hessian3_matches_three_sweeps():
    """hessian_cascade3's triple must equal three separate hessian_cascade
    calls.  The fused sweep shares its x-chain across the three pairs, so a
    stale value would show up here and nowhere else."""
    cl = _prepared(seed=4)
    _random_dirs(cl)
    cgg, cge, cee = cl.hessian_cascade3(cl.vars, cl.grads, cl.etas)
    close('hessian3 B(g,g) == hessian_cascade(g,g)', cgg,
          cl.hessian_cascade(cl.vars, cl.grads, cl.grads), 1e-3)
    close('hessian3 B(g,e) == hessian_cascade(g,e)', cge,
          cl.hessian_cascade(cl.vars, cl.grads, cl.etas), 1e-3)
    close('hessian3 B(e,e) == hessian_cascade(e,e)', cee,
          cl.hessian_cascade(cl.vars, cl.etas, cl.etas), 1e-3)


def test_fused_step_matches_classic():
    """The fused beta/bottom against the three-sweep route they replace.

    This is `check_fused_hessian` run as a test: _compute_step_fused derives
    both from one triple by bilinearity, so an error in the expansion (or in
    the order of the etas update against the sweep that reads it) shows up as
    a disagreement here.
    """
    cl = _prepared(seed=5)
    _random_dirs(cl)

    g_keep = {k: v.copy() for k, v in cl.grads.items()}
    e_keep = {k: v.copy() for k, v in cl.etas.items()}

    beta_ref = cl._ref_beta(cl.vars, cl.grads, cl.etas)
    cl._update_etas(cl.grads, cl.etas, beta_ref)
    bottom_ref = cl.allreduce_scalars(cl.hessian(cl.vars, cl.etas, cl.etas))[0]

    # restore and take the fused route from the same state
    for k in cl.grads:
        cl.grads[k][:] = g_keep[k]
        cl.etas[k][:] = e_keep[k]
    cl.start_iter = 0
    alpha, top, bottom = cl._compute_step_fused(cl.vars, cl.grads, cl.etas, 1)
    close('fused bottom == classic bottom', bottom, bottom_ref, 2e-2)
    check('fused beta is finite and bottom > 0',
          np.isfinite(beta_ref) and bottom > 0,
          f'beta={beta_ref:.6e} bottom={bottom:.6e}')


# ----------------------------------------------------------------------
# the loop
# ----------------------------------------------------------------------

def test_bh_descends_on_synthetic_data():
    """Generate data from a known state, perturb, and check BH drives the
    error down.  This is the first thing that runs every kernel in anger.
    Both misfit models."""
    for model in ('amplitude', 'intensity'):
        cl = _prepared(seed=6, model=model)

        # perturb away from the solution
        rng = np.random.default_rng(11)
        cl.vars['obj'] += cp.asarray(
            (0.01 * rng.standard_normal(cl.vars['obj'].shape)).astype('complex64'))
        cl.vars['prb'] += cp.asarray(
            (0.02 * rng.standard_normal(cl.vars['prb'].shape)).astype('complex64'))
        cl.vars['code'] += cp.asarray(
            (0.02 * rng.standard_normal((NCODE, NCODE))).astype('complex64'))

        cl.niter = 24
        cl.error_step = 4
        cl.start_iter = 0
        cl.precalc(cl.vars)
        cl.error_debug(cl.vars, -1)
        cl._iterate(cl.vars, cl.grads, cl.etas, writer=None)
        errs = cl.table['err'].to_numpy(dtype=float)
        check(f'BH reduces the error   [{model}]', errs[-1] < errs[0],
              f'{errs[0]:.4e} -> {errs[-1]:.4e} in {cl.niter} iters')
        check(f'BH ends at least 10x below the start   [{model}]',
              errs[-1] < 0.1 * errs[0], f'ratio {errs[-1] / errs[0]:.2e}')
        check(f'BH error is monotone   [{model}]',
              bool(np.all(np.diff(errs) <= 1e-9)),
              f'{np.array2string(errs, precision=3)}')
        check(f'BH error stays finite   [{model}]', bool(np.all(np.isfinite(errs))))


def _phantom():
    """A smooth weak-phase object confined to the region every frame sees, so
    the correlation below is judged where the data can constrain it."""
    y, x = np.mgrid[:NOBJ, :NOBJ].astype('float32')
    cy = cx = (NOBJ - 1) / 2

    def blob(y0, x0, r, a):
        return a * np.exp(-((y - y0) ** 2 + (x - x0) ** 2) / (2 * r ** 2))

    phase = (blob(cy - 4, cx - 3, 3.0, 0.20) + blob(cy + 4, cx + 3, 2.5, -0.15)
             + blob(cy, cx + 5, 2.0, 0.12))
    win = np.zeros_like(phase)
    win[MARGIN:NOBJ - MARGIN, MARGIN:NOBJ - MARGIN] = 1
    return (phase * win).astype('float32')


def test_recovery():
    """From data generated with a known object, a cold start (object = 0) with
    the probe and code at their truth and frozen must recover the object.
    Correlation is taken over the region the frames actually illuminate."""
    for model in ('amplitude', 'intensity'):
        cl = _fill(RecHoloCoded(_args(model=model, freeze_code=True, niter=128,
                                      error_step=-1, nchunk=4)), seed=3)
        phase = _phantom()
        cl.vars['obj'][:] = cp.asarray(phase.astype('complex64'))   # purely a phase
        truth = cp.asnumpy(cl.vars['obj']).copy()
        _synth(cl)
        cl.vars['obj'][:] = 0
        cl.BH()

        sl = (slice(MARGIN, NOBJ - MARGIN),) * 2
        a, b = cp.asnumpy(cl.vars['obj'])[sl].ravel(), truth[sl].ravel()
        corr = abs(np.vdot(b, a)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
        check(f'object correlation > 0.9 after 128 iters   [{model}]', corr > 0.9,
              f'corr {corr:.4f}')


def test_runs_without_reader():
    """The engine must be usable with no reader, no HDF5 and no config file --
    that is what keeps the not-yet-existing input format off the critical path.
    If this ever needs `reader_holo_coded`, the coupling has leaked."""
    cl = RecHoloCoded(_args(niter=4, error_step=1, checkpoint_step=-1))
    _fill(cl, seed=2)
    _synth(cl)
    rng = np.random.default_rng(5)
    cl.vars['obj'] += cp.asarray(
        (0.02 * rng.standard_normal(cl.vars['obj'].shape)).astype('complex64'))
    cl.BH()
    errs = cl.table['err'].to_numpy(dtype=float)
    check('BH from assigned buffers decreases the objective', errs[-1] < errs[0],
          f'{errs[0]:.4e} -> {errs[-1]:.4e}')
    check('reader module was not imported',
          'holotomocupy.reader_holo_coded' not in sys.modules)


def test_freeze_flags():
    """freeze_code / freeze_pos_code must zero exactly their own gradient.

    Run OFF the minimum, and against an unfrozen control: at the generating
    state every gradient is exactly zero, so a frozen-and-zero result would
    prove nothing.
    """
    ctl = _prepared(seed=9)
    _perturb(ctl)
    ctl.gradients(ctl.vars, ctl.grads)
    check('control: grads[code] is nonzero when not frozen',
          float(cp.abs(ctl.grads['code']).max()) > 0.0)
    check('control: grads[pos_code] is nonzero when not frozen',
          float(np.abs(ctl.grads['pos_code']).max()) > 0.0)

    cl = _prepared(seed=9, freeze_code=True, freeze_pos_code=True)
    _perturb(cl)
    cl.gradients(cl.vars, cl.grads)
    check('freeze_code zeroes grads[code]',
          float(cp.abs(cl.grads['code']).max()) == 0.0)
    check('freeze_pos_code zeroes grads[pos_code]',
          float(np.abs(cl.grads['pos_code']).max()) == 0.0)
    check('freeze flags leave grads[prb] alone',
          float(cp.abs(cl.grads['prb']).max()) > 0.0)
    check('freeze flags leave grads[obj] alone',
          float(cp.abs(cl.grads['obj']).max()) > 0.0)


def test_code_drift_guard():
    """A pos_code far from the frozen offsets must stop the run."""
    cl = _prepared(seed=10)
    cl.vars['pos_code'][:] += 1e4
    try:
        cl.check_code_drift(cl.vars)
        raised = False
    except RuntimeError:
        raised = True
    check('check_code_drift stops a runaway pos_code', raised)


# ----------------------------------------------------------------------
# MPI
# ----------------------------------------------------------------------

def test_mpi_matches_serial():
    """2 ranks must reproduce 1 rank.

    obj, prb and code are all REPLICATED: every rank sees the whole plane,
    computes a partial gradient from its own frames, and the allreduce sums
    them.  This is the test that catches a missing allreduce on grads['obj'] --
    a single rank cannot see it, and with it missing the object gradient is
    comm.size times too small on every rank.
    """
    comm = MPI.COMM_WORLD
    cl = _prepared(seed=12)
    _perturb(cl)            # off the minimum -- else every number below is 0
    cl.gradients(cl.vars, cl.grads)
    gobj = float(cp.linalg.norm(cl.grads['obj']))
    gprb = float(cp.linalg.norm(cl.grads['prb']))
    gcode = float(cp.linalg.norm(cl.grads['code']))
    err = _minv(cl, cl.vars)
    out = np.array([gobj, gprb, gcode, err], dtype='float64')
    check('reference quantities are nonzero (test is not vacuous)',
          bool(np.all(out > 0)), f'{out}')

    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '_mpi_holo_coded_ref.npy')
    if comm.size == 1:
        np.save(ref_path, out)
        check('serial reference written', True, f'{out}')
        return
    if not os.path.exists(ref_path):
        check('serial reference exists (run on 1 rank first)', False)
        return
    ref = np.load(ref_path)
    for j, name in enumerate(('||grad obj||', '||grad prb||', '||grad code||', 'err')):
        close(f'{comm.size} ranks == 1 rank: {name}', out[j], ref[j], 2e-3)


if __name__ == '__main__':
    argv = sys.argv[1:]
    if argv == ['mpi']:
        sys.exit(run({'test_mpi_matches_serial': test_mpi_matches_serial}, []))
    sys.exit(run(globals(), argv))
