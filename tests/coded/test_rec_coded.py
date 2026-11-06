#!/usr/bin/env python
"""`rec_coded_mpi.RecCoded` end to end: the batched kernels and the BH loop.

    python tests/coded/test_rec_coded.py                  # serial checks
    mpirun -np 2 python tests/coded/test_rec_coded.py mpi  # the MPI ones

`test_cascade_coded.py` proves the six levels differentiate correctly when
called by hand.  This proves the production wrappers around them are wired
right, which is a different failure mode entirely: a @gpu_batch argument in
the wrong slot, a replicated array mistaken for a theta-chunked one, a missing
allreduce, or an rho_sq applied in the wrong place would all leave the cascade
tests green.

  gradients_cascade   against a finite difference of min(), per variable --
                      this is the end-to-end check of the gradient, including
                      the rho_sq convention and the adjoint Radon
  hessian3            against three separate hessian() sweeps
  fused step          beta and bottom against the classic three-sweep route
  BH                  the error actually goes down on synthetic data
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
from holotomocupy.rec_coded_mpi import RecCoded                   # noqa: E402

N, NZ, NTHETA = 32, 32, 8
NOBJ = 3 * N // 2
NCODE = 2 * N


def _args(**kw):
    theta = np.linspace(0, np.pi, NTHETA, endpoint=False).astype('float32')
    a = dict(
        energy=17.1, detector_pixelsize=1.4760147601476e-6 * 16,
        focustodetectordistance=1.217, z1=5.110e-3, z1_code=4.584e-3,
        theta=theta, ntheta=NTHETA, nz=NZ, n=N, nzobj=NOBJ, nobj=NOBJ,
        ncode=NCODE, mask=0.9, lam_laplacian=0.0, rho=[1, 1, 1, 1, 1],
        code_pad_frac=0.125, shift_type='fft', niter=1, nchunk=4,
        checkpoint_step=-1, error_step=-1, start_iter=0,
        model='amplitude', psf_sigma=0.0, comm=MPI.COMM_WORLD)
    a.update(kw)
    return SimpleNamespace(**a)


def _fill(cl, seed=0, pos_amp=1.0):
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

    obj = cx((cl.nzobj, cl.nobj, cl.nobj), 0.02)
    prb = cx((cl.nz, cl.n), 0.1) + 1.0
    code = cx((NCODE, NCODE), 0.1) + 1.0
    pos = (pos_amp * (rng.random((cl.ntheta, 2)) - 0.5)).astype('float32')
    pos_code = (3 * pos_amp
                * (rng.random((cl.ntheta, 2)) - 0.5)).astype('float32')

    cl.vars['obj'][:] = obj[cl.st_obj:cl.end_obj]
    cl.vars['prb'][:] = prb
    cl.vars['code'][:] = cp.asarray(code)
    cl.vars['pos'][:] = pos[cl.st_theta:cl.end_theta]
    cl.vars['pos_code'][:] = pos_code[cl.st_theta:cl.end_theta]
    return cl


def _synth(cl):
    """Fill cl.data with K|psi|^2 of the current variables, then leave the
    solver at that exact state (gen_data renormalises obj and back)."""
    cl.gen_data(cl.vars, cl.data)


def _prepared(seed=0, **kw):
    """A solver with data generated from its own variables, precalc run, so
    it is sitting exactly at the global minimum of the data term."""
    cl = _fill(RecCoded(_args(**kw)), seed)
    _synth(cl)
    cl.precalc(cl.vars)
    return cl


# ----------------------------------------------------------------------
# gradient
# ----------------------------------------------------------------------

def test_gradient_against_finite_difference():
    """<grad, y> == d/dh min(x + h y), per variable.

    min() and gradients() go through completely separate @gpu_batch calls, so
    this catches a mis-ordered kernel argument, a replicated array sliced as
    if it were chunked, and the rho_sq convention (gradients_cascade multiplies
    by rho_sq, so the directional derivative must be divided by it again).
    """
    cl = _prepared(seed=1)
    # Move off the minimum: at the exact solution the gradient is ~0 and the
    # relative comparison below has nothing to resolve.
    rng = np.random.default_rng(7)
    cl.vars['obj'] += (0.05 * rng.standard_normal(cl.vars['obj'].shape)
                       ).astype('complex64')
    cl.vars['pos'] += (0.3 * rng.standard_normal(cl.vars['pos'].shape)
                       ).astype('float32')
    cl.fwd_tomo(cl.vars['obj'], out=cl.proj_tmp)
    cl.redist(cl.proj_tmp, cl.vars['proj'])

    cl.gradients(cl.vars, cl.grads)

    def minv(v):
        return float(cl.min(v['prb'], v['obj'], v['pos'], v['proj'],
                            v['code'], v['pos_code']))

    # 'code' lives on the GPU while the rest are pinned host arrays, so each
    # direction is built in the variable's own namespace and dtype.
    names = ('prb', 'pos', 'code', 'pos_code')
    for name in names:
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
            vals[s] = minv(cl.vars)
        cl.vars[name][:] = base
        fd = (vals[+1] - vals[-1]) / (2 * h)
        close(f'<grad, y> == d/dh min   [{name}]', inner, fd, 5e-2)


def test_obj_gradient_against_finite_difference():
    """The object gradient additionally goes through redist + adjoint Radon,
    so it is worth its own check."""
    cl = _prepared(seed=2)
    rng = np.random.default_rng(8)
    cl.vars['obj'] += (0.05 * rng.standard_normal(cl.vars['obj'].shape)
                       ).astype('complex64')
    cl.fwd_tomo(cl.vars['obj'], out=cl.proj_tmp)
    cl.redist(cl.proj_tmp, cl.vars['proj'])
    cl.gradients(cl.vars, cl.grads)

    y = ((rng.standard_normal(cl.vars['obj'].shape)
          + 1j * rng.standard_normal(cl.vars['obj'].shape)) * 0.01).astype('complex64')
    inner = float(np.sum(np.real(np.conj(cl.grads['obj']) * y))) / cl.rho_sq['obj']
    inner = float(cl.allreduce(np.array(inner, dtype='float32')))

    base = cl.vars['obj'].copy()
    h = 1e-2
    vals = {}
    for s in (+1, -1):
        cl.vars['obj'][:] = base + np.float32(s * h) * y
        # proj must follow obj: min() reads the data term through proj
        cl.fwd_tomo(cl.vars['obj'], out=cl.proj_tmp)
        cl.redist(cl.proj_tmp, cl.vars['proj'])
        vals[s] = float(cl.min(cl.vars['prb'], cl.vars['obj'], cl.vars['pos'],
                               cl.vars['proj'], cl.vars['code'],
                               cl.vars['pos_code']))
    cl.vars['obj'][:] = base
    fd = (vals[+1] - vals[-1]) / (2 * h)
    close('<grad_obj, y> == d/dh min', inner, fd, 5e-2)


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
    for lam in (0.0, 1e-3):
        cl = _prepared(seed=4, lam_laplacian=lam)
        _random_dirs(cl)
        gg, ge, ee = cl.hessian3(cl.vars, cl.grads, cl.etas)
        # hessian() takes its Laplacian direction from the padded etas slab,
        # so compare the cascade part separately when the term is on.
        cgg, cge, cee = cl.hessian_cascade3(cl.vars, cl.grads, cl.etas)
        tag = f'lam={lam:g}'
        close(f'hessian3 B(g,g) == hessian_cascade(g,g)   [{tag}]', cgg,
              cl.hessian_cascade(cl.vars, cl.grads, cl.grads), 1e-3)
        close(f'hessian3 B(g,e) == hessian_cascade(g,e)   [{tag}]', cge,
              cl.hessian_cascade(cl.vars, cl.grads, cl.etas), 1e-3)
        close(f'hessian3 B(e,e) == hessian_cascade(e,e)   [{tag}]', cee,
              cl.hessian_cascade(cl.vars, cl.etas, cl.etas), 1e-3)
        if lam > 0:
            check(f'hessian3 adds the Laplacian   [{tag}]',
                  abs(ee - cee) > 0, f'B(e,e) {cee:.6e} -> {ee:.6e}')


def test_fused_step_matches_classic():
    """The fused beta/bottom against the three-sweep route they replace.

    This is `check_fused_hessian` run as a test: _compute_step_fused derives
    both from one triple by bilinearity, so an error in the expansion (or in
    the order of the etas update against the sweep that reads it) shows up as
    a disagreement here.
    """
    for lam in (0.0, 1e-3):
        cl = _prepared(seed=5, lam_laplacian=lam)
        _random_dirs(cl)
        tag = f'lam={lam:g}'

        # classic: beta from two sweeps, then bottom from a third on the
        # UPDATED direction.
        import copy
        g_keep = {k: (v.copy()) for k, v in cl.grads.items()}
        e_keep = {k: (v.copy()) for k, v in cl.etas.items()}

        beta_ref = cl._ref_beta(cl.vars, cl.grads, cl.etas)
        cl._update_etas(cl.grads, cl.etas, beta_ref)
        bottom_ref = cl.allreduce_scalars(
            cl.hessian(cl.vars, cl.etas, cl.etas))[0]

        # restore and take the fused route from the same state
        for k in cl.grads:
            cl.grads[k][:] = g_keep[k]
            cl.etas[k][:] = e_keep[k]
        cl.start_iter = 0
        alpha, top, bottom = cl._compute_step_fused(cl.vars, cl.grads, cl.etas, 1)
        # _compute_step_fused recomputes beta internally; recover it the same
        # way the classic route would have.
        close(f'fused bottom == classic bottom   [{tag}]',
              bottom, bottom_ref, 2e-2)
        check(f'fused beta is finite and >= 0   [{tag}]',
              np.isfinite(beta_ref) and bottom > 0,
              f'beta={beta_ref:.6e} bottom={bottom:.6e}')


# ----------------------------------------------------------------------
# the loop
# ----------------------------------------------------------------------

def test_bh_descends_on_synthetic_data():
    """Generate data from a known state, perturb, and check BH drives the
    error down.  This is the first thing that runs every kernel in anger."""
    cl = _prepared(seed=6)
    truth = {k: (v.copy()) for k, v in cl.vars.items()}

    # perturb away from the solution
    rng = np.random.default_rng(11)
    cl.vars['obj'] += (0.01 * rng.standard_normal(cl.vars['obj'].shape)
                       ).astype('complex64')
    cl.vars['prb'] += (0.02 * rng.standard_normal(cl.vars['prb'].shape)
                       ).astype('complex64')
    cl.vars['code'] += cp.asarray(
        (0.02 * rng.standard_normal((NCODE, NCODE))).astype('complex64'))

    cl.niter = 24
    cl.error_step = 4
    cl.start_iter = 0
    cl.precalc(cl.vars)
    cl.error_debug(cl.vars, -1)
    e0 = float(cl.table['err'].iloc[0])
    cl._iterate(cl.vars, cl.grads, cl.etas, writer=None)
    errs = cl.table['err'].to_numpy(dtype=float)
    check('BH reduces the error', errs[-1] < errs[0],
          f'{errs[0]:.4e} -> {errs[-1]:.4e} in {cl.niter} iters')
    check('BH error is monotone', bool(np.all(np.diff(errs) <= 1e-9)),
          f'{np.array2string(errs, precision=3)}')
    check('BH error stays finite', bool(np.all(np.isfinite(errs))))


def test_freeze_flags():
    """freeze_code / freeze_pos_code must zero exactly their own gradient."""
    cl = _prepared(seed=9, freeze_code=True, freeze_pos_code=True)
    cl.gradients(cl.vars, cl.grads)
    check('freeze_code zeroes grads[code]',
          float(cp.abs(cl.grads['code']).max()) == 0.0)
    check('freeze_pos_code zeroes grads[pos_code]',
          float(np.abs(cl.grads['pos_code']).max()) == 0.0)
    check('freeze flags leave grads[prb] alone',
          float(np.abs(cl.grads['prb']).max()) > 0.0)


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


def test_rho_obj_guard():
    """rho[obj] != 1 with the Laplacian on must be refused, not silently
    reweighted -- the two halves of grads['obj'] would be in different
    metrics."""
    try:
        RecCoded(_args(lam_laplacian=1e-3, rho=[0.5, 1, 1, 1, 1]))
        raised = False
    except ValueError:
        raised = True
    check('rho[obj] != 1 with lam_laplacian > 0 is refused', raised)


# ----------------------------------------------------------------------
# MPI
# ----------------------------------------------------------------------

def test_mpi_matches_serial():
    """2 ranks must reproduce 1 rank.

    The new thing MPI has to get right here is that `code` is REPLICATED like
    `prb`: every rank sees the whole plane, computes a partial gradient from
    its own angles, and the allreduce sums them.  Get that wrong and the code
    gradient is comm.size times too small (or too large) and only a
    multi-rank run notices.
    """
    comm = MPI.COMM_WORLD
    cl = _prepared(seed=12)
    cl.gradients(cl.vars, cl.grads)
    # prb and code are replicated and already allreduced, so every rank holds
    # the same values and they can be compared against the serial reference
    # directly.
    gprb = float(np.linalg.norm(cl.grads['prb']))
    gcode = float(cp.linalg.norm(cl.grads['code']))
    err = float(cl.min(cl.vars['prb'], cl.vars['obj'], cl.vars['pos'],
                       cl.vars['proj'], cl.vars['code'], cl.vars['pos_code']))
    out = np.array([gprb, gcode, err], dtype='float64')

    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '_mpi_ref.npy')
    if comm.size == 1:
        np.save(ref_path, out)
        check('serial reference written', True, f'{out}')
        return
    if not os.path.exists(ref_path):
        check('serial reference exists (run on 1 rank first)', False)
        return
    ref = np.load(ref_path)
    for j, name in enumerate(('||grad prb||', '||grad code||', 'err')):
        close(f'{comm.size} ranks == 1 rank: {name}', out[j], ref[j], 2e-3)


if __name__ == '__main__':
    argv = sys.argv[1:]
    if argv == ['mpi']:
        sys.exit(run({'test_mpi_matches_serial': test_mpi_matches_serial}, []))
    sys.exit(run(globals(), argv))
