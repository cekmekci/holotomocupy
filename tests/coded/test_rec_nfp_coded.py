#!/usr/bin/env python
"""`rec_nfp_coded_mpi.RecNFPCoded`: the windowed object-plane shift for step0.

    python tests/coded/test_rec_nfp_coded.py
    mpirun -np 2 python tests/coded/test_rec_nfp_coded.py mpi

`RecNFPCoded` changes exactly one thing relative to `RecNFP`: level F3 reads a
window of the (prb-sized) global plane per position instead of the whole
thing, via `ShiftFFTCode` (already proven in `test_shift_code.py`). So what
this file checks is specific to the wiring, not the shift math itself:

  degenerate    pos_pad_frac=0 reproduces RecNFP's gradient and value exactly
                -- the F3-family overrides fall through to super() and the
                _tiled_coeff path is untouched
  cascade       Taylor order 2/3 and the gF/dF adjoint identity, windowed
                (pos_pad_frac > 0), same shape as test_cascade_coded.py
  MPI           2 ranks reproduce 1 rank -- pos_off is per-rank (theta-
                distributed, unlike code_off's replicated analogue in
                RecCoded), so this is the one thing specific to NFP's MPI
                layout worth checking separately
"""
import os
import sys

import numpy as np
import cupy as cp
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'unit'))
from harness import check, close, order, run                      # noqa: E402

from mpi4py import MPI                                            # noqa: E402
from holotomocupy.rec_nfp_mpi import RecNFP                        # noqa: E402
from holotomocupy.rec_nfp_coded_mpi import RecNFPCoded             # noqa: E402
from holotomocupy.utils import redot                               # noqa: E402

N, NZ, NTHETA = 32, 32, 6
NOBJ = 3 * N // 2       # the "aperture" plane, deliberately bigger than N
FLOOR = 2e-6
FLOOR_D2 = 3e-5
HS = np.array([8e-2, 4e-2, 2e-2, 1e-2])
rng = np.random.default_rng(7)


def _args(cls=RecNFPCoded, **kw):
    a = dict(
        energy=17.1, detector_pixelsize=1.4760147601476e-6 * 16,
        focustodetectordistance=1.217, z1=5.110e-3,
        ntheta=NTHETA, nz=NZ, n=N, nzobj=NOBJ, nobj=NOBJ,
        rho=[1, 1, 1], shift_type='fft', pos_pad_frac=0.125,
        niter=1, nchunk=NTHETA, checkpoint_step=-1, error_step=-1,
        start_iter=0, model='amplitude', psf_sigma=0.0, comm=MPI.COMM_WORLD)
    a.update(kw)
    return cls(SimpleNamespace(**a))


def _cx(shape, s=1.0):
    return ((rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
            * s).astype('complex64')


def _fill(cl, seed=0, pos_amp=1.0):
    """`proj` is the single GLOBAL aperture plane (like RecCoded's `code`, not
    theta-distributed) -- every rank draws the identical array from the same
    seed, so no slicing is needed there, unlike `pos`."""
    r = np.random.default_rng(seed)
    proj = ((r.standard_normal((cl.nzobj, cl.nobj))
            + 1j * r.standard_normal((cl.nzobj, cl.nobj))) * 0.02
           ).astype('complex64')
    prb = ((r.standard_normal((cl.nz, cl.n))
           + 1j * r.standard_normal((cl.nz, cl.n))) * 0.1 + 1.0
          ).astype('complex64')
    pos = (pos_amp * (r.random((cl.ntheta, 2)) - 0.5)).astype('float32')
    cl.vars['prb'][:]  = cp.asarray(prb)
    cl.vars['proj'][:] = cp.asarray(proj)
    cl.vars['pos'][:]  = cp.asarray(pos[cl.st_theta:cl.end_theta])
    return cl


def _prepared(cls=RecNFPCoded, seed=0, perturb=0.0, **kw):
    """Data generated from the solver's own variables, then (optionally) `pos`
    perturbed away from the point that generated the data, precalc run after.

    At `perturb=0` the gradient is EXACTLY zero (not just small): single
    chunk, so there is no reduction-order noise, and the same deterministic
    forward pass produces the data and the F0 residual. That makes a
    zero-vs-zero MPI comparison trivially true regardless of correctness, so
    `test_mpi_matches_serial` perturbs `pos` (the one thing threaded
    differently per rank, via the theta-distributed `pos_off`) to get a
    genuinely nonzero gradient worth comparing.
    """
    cl = _fill(_args(cls, **kw), seed)
    cl.gen_data(cl.vars, cl.data)
    if perturb:
        r = np.random.default_rng(seed + 1)
        d = (perturb * (r.random((cl.ntheta, 2)) - 0.5)).astype('float32')
        cl.vars['pos'][:] += cp.asarray(d[cl.st_theta:cl.end_theta])
    cl.precalc(cl.vars)
    return cl


# ----------------------------------------------------------------------
# Degenerate case: pos_pad_frac=0 must reproduce RecNFP exactly
# ----------------------------------------------------------------------

def test_degenerate_matches_recnfp():
    """At pos_pad_frac=0 every F3-family override falls through to super(),
    so RecNFPCoded's gradient and value must equal RecNFP's bit-for-bit
    (modulo the extra, unused pos_off plumbing)."""
    cl_c = _prepared(RecNFPCoded, seed=3, pos_pad_frac=0, perturb=0.3)
    cl_p = _prepared(RecNFP,      seed=3, pos_pad_frac=0, perturb=0.3)

    cl_c.gradients(cl_c.vars, cl_c.grads)
    cl_p.gradients(cl_p.vars, cl_p.grads)

    close('degenerate: ||grad prb||',
          float(cp.linalg.norm(cl_c.grads['prb'])),
          float(cp.linalg.norm(cl_p.grads['prb'])), 1e-6)
    close('degenerate: ||grad proj||',
          float(cp.linalg.norm(cl_c.grads['proj'])),
          float(cp.linalg.norm(cl_p.grads['proj'])), 1e-6)
    close('degenerate: ||grad pos||',
          float(np.linalg.norm(cl_c.grads['pos'])),
          float(np.linalg.norm(cl_p.grads['pos'])), 1e-6)

    err_c = cl_c.min(cl_c.vars['prb'], cl_c.vars['proj'], cl_c.vars['pos'])
    err_p = cl_p.min(cl_p.vars['prb'], cl_p.vars['proj'], cl_p.vars['pos'])
    close('degenerate: err', err_c, err_p, 1e-6)


# ----------------------------------------------------------------------
# Cascade: Taylor order and the gF/dF adjoint identity, windowed
# ----------------------------------------------------------------------

class Cascade:
    """One chunk of the windowed F3 cascade, the production loops by hand --
    same shape as test_cascade_coded.py's Cascade, with the two non-F3
    levels (F2, F1) and F0 inherited unchanged from RecNFP."""

    def __init__(self, cl):
        self.cl = cl
        # proj is the single GLOBAL aperture plane here (RecCoded's "code"
        # role), not theta-distributed like RecCoded's own "proj".
        self.proj = cp.asarray(_cx((NOBJ, NOBJ)) * 0.1)
        self.prb  = cp.asarray(_cx((NZ, N)) + 1.0)
        self.pos  = cp.asarray((2 * (rng.random((NTHETA, 2)) - 0.5)).astype('float32'))
        self.d = cp.asarray(
            np.abs(rng.standard_normal((NTHETA, NZ, N))).astype('float32') + 0.5)
        # Freeze the integer offsets at the base point, as precalc does --
        # they must not move with the Taylor step.
        cl.freeze_pos_off(cp.asnumpy(self.pos))
        self.off = cp.asarray(cl.pos_off)

    def x(self):
        return [self.prb, self.proj, self.pos]

    def direction(self, scale=1.0):
        return [cp.asarray(_cx((NZ, N))) * scale,
                cp.asarray(_cx((NOBJ, NOBJ))) * 0.1 * scale,
                cp.asarray((rng.random((NTHETA, 2)) - 0.5).astype('float32')) * scale]

    def _arm(self):
        cl = self.cl
        cl.cl_shift.coeff_cache_reset()
        cl._off_chunk = self.off

    def value(self, x):
        self._arm()
        return float(self.cl.F0(self.cl.apply_F_from(x, 1), self.d).get())

    def dvalue(self, x, y):
        cl = self.cl
        self._arm()
        for i in range(1, len(cl.F))[::-1]:
            fx, y = cl.dF[i](x, y)
            x = fx
        return float(cl.dF0(x, y, self.d).get())

    def d2value(self, x, y, z=None):
        cl = self.cl
        self._arm()
        diag = z is None
        if diag:
            z = y
        w = [None] * 3
        for i in range(1, len(cl.F))[::-1]:
            w = cl.d2F_dF[i](x, y, z, w)
            fx, yn = cl.dF[i](x, y)
            z = yn if diag else cl.dF[i](x, z, return_x=False)
            x, y = fx, yn
        return float(cl.d2F_dF[0](x, y, z, w, self.d).get())

    def gradient(self, x):
        cl = self.cl
        self._arm()
        y = self.d
        for i in range(len(cl.gF)):
            y = cl.gF[i](x, y)
        return [y[0], y[1], y[2]]

    @staticmethod
    def step(x, h, y):
        return [(a + np.float32(h) * b).astype(a.dtype) for a, b in zip(x, y)]

    @staticmethod
    def dot(g, y):
        return (float(redot(g[0], y[0])) + float(redot(g[1], y[1]))
                + float(cp.sum(g[2] * y[2])))


SLOTS = ('prb', 'proj', 'pos')


def _axis(c, k):
    y = [cp.zeros_like(a) for a in c.x()]
    y[k] = c.direction()[k]
    return y


def test_gradient_is_the_adjoint_of_the_cascade():
    cl = _args()
    c = Cascade(cl)
    g = c.gradient(c.x())
    for k, name in enumerate(SLOTS):
        y = _axis(c, k)
        close(f'<grad, y> == dF[y] for {name}',
              c.dot(g, y), c.dvalue(c.x(), y), 2e-3)
    y = c.direction()
    close('<grad, y> == dF[y], all variables', c.dot(g, y), c.dvalue(c.x(), y), 2e-3)


def test_first_derivative_order():
    cl = _args()
    c = Cascade(cl)
    x = c.x()
    f0 = c.value(x)
    for k, name in enumerate(SLOTS):
        y = _axis(c, k)
        g = c.dvalue(x, y)
        e = [abs(c.value(c.step(x, h, y)) - f0 - h * g) for h in HS]
        p = order(HS, e, FLOOR * max(abs(f0), 1.0))
        check(f'cascade dF order 2, {name}', p > 1.8,
              f'slope {p:.2f}, {e[0]:.2e} -> {e[-1]:.2e}'
              + ('  (at float32 rounding)' if p == float('inf') else ''))


def test_second_derivative_order():
    cl = _args()
    c = Cascade(cl)
    x = c.x()
    f0 = c.value(x)
    for k, name in enumerate(SLOTS):
        y = _axis(c, k)
        g, H = c.dvalue(x, y), c.d2value(x, y)
        e = [abs(c.value(c.step(x, h, y)) - f0 - h * g - 0.5 * h * h * H)
             for h in HS]
        p = order(HS, e, FLOOR_D2 * max(abs(f0), 1.0))
        check(f'cascade d2F order 3, {name}', p > 2.8,
              f'slope {p:.2f}, {e[0]:.2e} -> {e[-1]:.2e}'
              + ('  (at float32 rounding)' if p == float('inf') else ''))


# ----------------------------------------------------------------------
# MPI parity
# ----------------------------------------------------------------------

def test_mpi_matches_serial():
    """2 ranks must reproduce 1 rank. pos_off is theta-distributed (each rank
    freezes only its own slice), unlike code_off's replicated analogue in
    RecCoded, so this is the thing specific to this engine worth checking."""
    comm = MPI.COMM_WORLD
    cl = _prepared(RecNFPCoded, seed=11, perturb=0.3)
    cl.gradients(cl.vars, cl.grads)
    gprb = float(cp.linalg.norm(cl.grads['prb']))
    gproj = float(cp.linalg.norm(cl.grads['proj']))
    err = float(cl.min(cl.vars['prb'], cl.vars['proj'], cl.vars['pos']))
    out = np.array([gprb, gproj, err], dtype='float64')

    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '_mpi_nfp_ref.npy')
    if comm.size == 1:
        np.save(ref_path, out)
        check('serial reference written', True, f'{out}')
        return
    if not os.path.exists(ref_path):
        check('serial reference exists (run on 1 rank first)', False)
        return
    ref = np.load(ref_path)
    for j, name in enumerate(('||grad prb||', '||grad proj||', 'err')):
        close(f'{comm.size} ranks == 1 rank: {name}', out[j], ref[j], 2e-3)


if __name__ == '__main__':
    argv = sys.argv[1:]
    if argv == ['mpi']:
        sys.exit(run({'test_mpi_matches_serial': test_mpi_matches_serial}, []))
    sys.exit(run(globals(), argv))
