#!/usr/bin/env python
"""The coded-aperture cascade F0(F1(F2(F3(F4(F5(x)))))) and its differentials.

    python tests/coded/test_cascade_coded.py

The counterpart of `tests/unit/test_cascade.py` for `rec_coded_mpi.RecCoded`.
Six levels instead of five:

    F5  the shift of the projection onto the object plane (m = 1)
    F4  exp(i psi), the sample transmission
    F3  the WINDOWED shift of the coded aperture
    F2  multiply by the probe, then Fresnel propagate code -> sample
    F1  multiply by the sample, then Fresnel propagate sample -> detector
    F0  the data misfit

The three chains below are copied from `gradients_cascade` and
`hessian_cascade3`, so what is tested is the loop that runs in production:

    value     F0(apply_F_from(x, 1), d)
    gradient  y = d;  y = gF[id](x, y)  for id = 0..5
    forward   fx, y = dF[id](x, y)      for id = 5..1,  then dF0
    hessian   w = d2F_dF[id](x, y, z, w) alongside it, then d2F_dF0

checked against each other and against finite differences:

  * <grad, y> == dF[chain][y]        -- the adjoint of every gF level at once
  * value(x+hy) - value - h dF       = O(h^2)
  * ... - h^2/2 d2F                  = O(h^3)
  * B(y+z,y+z) == B(y,y) + 2B(y,z) + B(z,z)  -- the off-diagonal form, which
    is the only check that sees the d2F_dF3 / d2F_dF5 slot pairing

`code` and `pos_code` are the two new slots; a missing conjugate or a dropped
term in the code levels breaks one of the four.
"""
import os
import sys

import numpy as np
import cupy as cp
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'unit'))
from harness import check, close, order, run                      # noqa: E402

FLOOR = 2e-7          # float32 rounding of an objective of order 1

# ... but the SECOND-derivative residual needs a much higher floor.  F0 is a
# reduction over ntheta*nz*n float32 terms, so its value carries ~1e-5 of
# absolute noise here, while the h^3 residual of a direction with a large h^2
# term is 1e-6 or below.  Measured on this fixture: the residual bounces
# between 3e-6 and 1.3e-5 with no trend, so a fitted slope below this floor is
# reading rounding, not calculus -- and will "pass" or "fail" at random.
# Points under the floor are dropped by `order`, which then reports inf and
# the check records an honest "at float32 rounding".  The directions whose
# second derivative genuinely matters (prb, code) are covered exactly, without
# finite differences, by test_prb_code_expansion_is_exact.
FLOOR_D2 = 3e-5

from mpi4py import MPI                                            # noqa: E402
from holotomocupy.rec_coded_mpi import RecCoded                    # noqa: E402
from holotomocupy.utils import redot                               # noqa: E402

N, NZ, NTHETA = 32, 32, 6
NOBJ = 3 * N // 2
NCODE = 2 * N
HS = np.array([8e-2, 4e-2, 2e-2, 1e-2])
rng = np.random.default_rng(5)


def _rec(model='intensity', psf_sigma=0.0):
    theta = np.linspace(0, np.pi, NTHETA, endpoint=False).astype('float32')
    return RecCoded(SimpleNamespace(
        energy=17.1, detector_pixelsize=1.4760147601476e-6 * 16,
        focustodetectordistance=1.217,
        z1=5.110e-3, z1_code=4.584e-3, theta=theta,
        ntheta=NTHETA, nz=NZ, n=N, nzobj=NOBJ, nobj=NOBJ, ncode=NCODE,
        mask=0.9, lam_laplacian=0, rho=[1, 1, 1, 1, 1],
        code_pad_frac=0.125, shift_type='fft',
        niter=1, nchunk=NTHETA, checkpoint_step=-1, error_step=-1,
        start_iter=0, model=model, psf_sigma=psf_sigma, comm=MPI.COMM_WORLD))


def _cx(shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype('complex64')


class Cascade:
    """One chunk: the production loops, callable directly."""

    def __init__(self, cl):
        self.cl = cl
        # A state the model can actually be at: a random object, a probe near
        # 1, a code near 1 (a weakly absorbing aperture), and small
        # displacements with a genuine sub-pixel part.
        self.proj = cp.asarray(_cx((NTHETA, NOBJ, NOBJ)) * 0.1)
        self.prb  = cp.asarray(_cx((NZ, N)) + 1.0)
        self.code = cp.asarray(_cx((NCODE, NCODE)) * 0.1 + 1.0)
        self.pos  = cp.asarray((2 * (rng.random((NTHETA, 2)) - 0.5)).astype('float32'))
        self.pos_code = cp.asarray(
            (5 * (rng.random((NTHETA, 2)) - 0.5)).astype('float32'))
        self.d = cp.asarray(
            np.abs(rng.standard_normal((NTHETA, NZ, N))).astype('float32') + 0.5)
        # The integer offsets are FROZEN at the base point, exactly as
        # precalc freezes them -- they must not follow the Taylor step, or
        # the objective being differentiated is not the one the solver sees.
        cl.freeze_code_off(cp.asnumpy(self.pos_code))
        self.off = cp.asarray(cl.code_off)

    def x(self):
        return [self.prb, self.code, self.pos_code, self.proj, self.pos]

    def direction(self, scale=1.0):
        return [cp.asarray(_cx((NZ, N))) * scale,
                cp.asarray(_cx((NCODE, NCODE))) * 0.1 * scale,
                cp.asarray((rng.random((NTHETA, 2)) - 0.5).astype('float32')) * scale,
                cp.asarray(_cx((NTHETA, NOBJ, NOBJ))) * 0.1 * scale,
                cp.asarray((rng.random((NTHETA, 2)) - 0.5).astype('float32')) * scale]

    def _arm(self):
        cl = self.cl
        cl.cl_shift.coeff_cache_reset()
        cl.cl_shift_code.coeff_cache_reset()
        cl.apply_F_cache_reset()
        cl._off_chunk = self.off
        # mask_1d is pinned HOST memory; @gpu_batch uploads it in production
        cl._set_mask_chunk(cp.asarray(cl.mask_1d))

    def value(self, x):
        self._arm()
        return float(self.cl.F0(self.cl.apply_F_from(x, 1), self.d).get())

    def dvalue(self, x, y):
        """The chained first directional derivative, as gradients_cascade runs it."""
        cl = self._arm() or self.cl
        for i in range(1, len(cl.F))[::-1]:
            fx, y = cl.dF[i](x, y)
            x = fx
        return float(cl.dF0(x, y, self.d).get())

    def d2value(self, x, y, z=None):
        """The chained second directional derivative B(y, z), as
        hessian_cascade runs it.  z defaults to y (the diagonal form)."""
        cl = self._arm() or self.cl
        diag = z is None
        if diag:
            z = y
        w = [None] * 5
        for i in range(1, len(cl.F))[::-1]:
            w = cl.d2F_dF[i](x, y, z, w)
            fx, yn = cl.dF[i](x, y)
            z = yn if diag else cl.dF[i](x, z, return_x=False)
            x, y = fx, yn
        return float(cl.d2F_dF[0](x, y, z, w, self.d).get())

    def x0(self, x):
        """The detector field, i.e. the cascade stopped just before F0."""
        self._arm()
        return self.cl.apply_F_from(x, 1)

    def dx0(self, x, y):
        """d(x0)/dh along y: the dF chain, stopped before dF0."""
        cl = self._arm() or self.cl
        for i in range(1, len(cl.F))[::-1]:
            fx, y = cl.dF[i](x, y)
            x = fx
        return y

    def d2x0(self, x, y):
        """d2(x0)/dh2 along y: the d2F chain's accumulated correction, which is
        exactly what d2F_dF0 receives in its `w` slot."""
        cl = self._arm() or self.cl
        z, w = y, [None] * 5
        for i in range(1, len(cl.F))[::-1]:
            w = cl.d2F_dF[i](x, y, z, w)
            fx, y = cl.dF[i](x, y)
            z = y
            x = fx
        return w

    def gradient(self, x):
        """gF0..gF5, as gradients_cascade runs it.
        Returns [prb, code, pos_code, proj, pos]."""
        cl = self._arm() or self.cl
        y = self.d
        for i in range(len(cl.gF)):
            y = cl.gF[i](x, y)
        # gF5 returns the un-prefiltered Deltapsi; gradients_cascade applies
        # the B-spline coeff once (the identity for the fft backend).
        return [y[0], y[1], y[2], cl.cl_shift.coeff(y[3]), y[4]]

    @staticmethod
    def step(x, h, y):
        """x + h*y, keeping every dtype: a python float promotes complex64."""
        return [(a + np.float32(h) * b).astype(a.dtype) for a, b in zip(x, y)]

    @staticmethod
    def dot(g, y):
        """<g, y>, complex inner product on the field slots and a plain real
        one on the two position slots."""
        return (float(redot(g[0], y[0])) + float(redot(g[1], y[1]))
                + float(cp.sum(g[2] * y[2])) + float(redot(g[3], y[3]))
                + float(cp.sum(g[4] * y[4])))


def _axis(c, k):
    """A direction that moves only variable k, so a broken level is named."""
    y = [cp.zeros_like(a) for a in c.x()]
    y[k] = c.direction()[k]
    return y


SLOTS = ('prb', 'code', 'pos_code', 'proj (obj)', 'pos')


def test_gradient_is_the_adjoint_of_the_cascade():
    """<gF chain, y> == dF chain [y], per variable and for all at once."""
    for model in ('intensity', 'amplitude'):
        cl = _rec(model)
        c = Cascade(cl)
        g = c.gradient(c.x())
        for k, name in enumerate(SLOTS):
            y = _axis(c, k)
            close(f'<grad, y> == dF[y] for {name}   [{model}]',
                  c.dot(g, y), c.dvalue(c.x(), y), 2e-3)
        y = c.direction()
        close(f'<grad, y> == dF[y], all variables   [{model}]',
              c.dot(g, y), c.dvalue(c.x(), y), 2e-3)


def test_first_derivative_order():
    """value(x+hy) - value - h dF = O(h^2), one variable at a time."""
    for model in ('intensity', 'amplitude'):
        cl = _rec(model)
        c = Cascade(cl)
        x = c.x()
        f0 = c.value(x)
        for k, name in enumerate(SLOTS):
            y = _axis(c, k)
            g = c.dvalue(x, y)
            e = [abs(c.value(c.step(x, h, y)) - f0 - h * g) for h in HS]
            p = order(HS, e, FLOOR * max(abs(f0), 1.0))
            check(f'cascade dF order 2, {name}   [{model}]', p > 1.8,
                  f'slope {p:.2f}, {e[0]:.2e} -> {e[-1]:.2e}'
                  + ('  (at float32 rounding)' if p == float('inf') else ''))


def test_second_derivative_order():
    """... - h^2/2 d2F = O(h^3): the Hessian chain, one variable at a time."""
    for model in ('intensity', 'amplitude'):
        cl = _rec(model)
        c = Cascade(cl)
        x = c.x()
        f0 = c.value(x)
        for k, name in enumerate(SLOTS):
            y = _axis(c, k)
            g, H = c.dvalue(x, y), c.d2value(x, y)
            e = [abs(c.value(c.step(x, h, y)) - f0 - h * g - 0.5 * h * h * H)
                 for h in HS]
            p = order(HS, e, FLOOR_D2 * max(abs(f0), 1.0))
            check(f'cascade d2F order 3, {name}   [{model}]', p > 2.6,
                  f'slope {p:.2f}, {e[0]:.2e} -> {e[-1]:.2e}'
                  + ('  (at float32 rounding)' if p == float('inf') else ''))


def test_prb_code_expansion_is_exact():
    """prb and code enter the cascade ONLY through the bilinear map
    u(prb, code) = D_code(prb . S(code)), and everything between u and the
    detector is linear in u.  So along any direction in (prb, code) the
    detector field is EXACTLY quadratic in h:

        x0(x + h y) == x0(x) + h dx0 + (h^2/2) d2x0

    with no remainder.  That makes this an algebraic identity rather than a
    convergence rate, which matters: a finite-difference order test on these
    two variables is swamped by float32 (the h^3 residual sits two orders
    BELOW the reduction noise of F0, so the fitted slope is meaningless).
    Here the cross term -- the only piece the single-variable tests never
    exercise, since each kills the other's direction -- is checked directly
    to float32 precision.

    prb alone and code alone are LINEAR, so their d2x0 must be exactly zero;
    a stray term in d2F_dF2 would show up there first.
    """
    cl = _rec('amplitude')
    c = Cascade(cl)
    x = c.x()
    d = c.direction()

    def rel(a, b):
        return float(cp.abs(a - b).max() / cp.maximum(cp.abs(b).max(), 1e-30))

    cases = {'prb': [0], 'code': [1], 'prb+code': [0, 1]}
    for name, idxs in cases.items():
        y = [cp.zeros_like(a) for a in x]
        for k in idxs:
            y[k] = d[k]
        f0 = c.x0(x)
        d1 = c.dx0(x, y)
        d2 = c.d2x0(x, y)

        # Single variables are linear, so the second-order term vanishes.
        if len(idxs) == 1:
            mag = float(cp.abs(d2).max() / cp.maximum(cp.abs(f0).max(), 1e-30))
            check(f'{name} alone: d2(x0) == 0 (the map is linear)', mag < 1e-5,
                  f'max |d2x0| / max |x0| = {mag:.2e}')

        for h in (2e-1, 5e-2):
            lhs = c.x0(c.step(x, h, y))
            rhs = f0 + np.float32(h) * d1 + np.float32(0.5 * h * h) * d2
            e = rel(lhs, rhs)
            check(f'{name}: x0 expansion exact at h={h:g}', e < 2e-5,
                  f'max rel {e:.2e}')


def test_d2x0_order():
    """Taylor order of the d2F chain measured on the detector FIELD rather
    than on the scalar objective.

    The scalar test above cannot resolve this: F0 reduces ntheta*nz*n float32
    terms, so its value carries ~1e-5 of noise while the h^3 residual is 1e-6
    or less.  x0 is an array of O(1) values with ~1e-7 of relative error per
    element and no reduction, so the same residual is two orders clear of the
    noise and the order is actually measurable.

    This covers d2F_dF1 through d2F_dF5 -- every level this engine adds or
    changes.  The one piece it does not reach is d2F_dF0, which is rec_mpi's
    own and is covered by tests/unit/test_derivatives.py.
    """
    cl = _rec('amplitude')
    c = Cascade(cl)
    x = c.x()
    hs = np.array([8e-2, 4e-2, 2e-2, 1e-2])
    for k, name in enumerate(SLOTS):
        y = _axis(c, k)
        f0 = c.x0(x)
        d1 = c.dx0(x, y)
        d2 = c.d2x0(x, y)
        scale = float(cp.abs(f0).max())
        e1, e2 = [], []
        for h in hs:
            v = c.x0(c.step(x, h, y))
            e1.append(float(cp.abs(v - f0 - np.float32(h) * d1).max()))
            e2.append(float(cp.abs(v - f0 - np.float32(h) * d1
                                   - np.float32(0.5 * h * h) * d2).max()))
        p1 = order(hs, e1, 1e-6 * scale)
        p2 = order(hs, e2, 1e-6 * scale)
        check(f'x0 dF order 2, {name}', p1 > 1.8,
              f'slope {p1:.2f}, {e1[0]:.2e} -> {e1[-1]:.2e}'
              + ('  (at float32 rounding)' if p1 == float('inf') else ''))
        check(f'x0 d2F order 3, {name}', p2 > 2.6,
              f'slope {p2:.2f}, {e2[0]:.2e} -> {e2[-1]:.2e}'
              + ('  (at float32 rounding)' if p2 == float('inf') else ''))


def test_hessian_polarization():
    """B(y+z, y+z) == B(y,y) + 2 B(y,z) + B(z,z).

    The off-diagonal form is what hessian_cascade3 computes for `Bge`, and the
    only check that can see the coefficient crossing in d2F_dF3 and d2F_dF5 --
    get it backwards and every diagonal test above still passes.
    """
    for model in ('intensity', 'amplitude'):
        cl = _rec(model)
        c = Cascade(cl)
        x = c.x()
        y = c.direction()
        z = c.direction()
        s = [a + b for a, b in zip(y, z)]
        Byy = c.d2value(x, y)
        Bzz = c.d2value(x, z)
        Byz = c.d2value(x, y, z)
        Bss = c.d2value(x, s)
        close(f'Hessian polarization identity   [{model}]',
              Bss, Byy + 2 * Byz + Bzz, 2e-3)


def test_hessian_symmetry():
    """B(y,z) == B(z,y). hessian_cascade3 reuses one sweep for both, so an
    asymmetry would make the fused beta silently wrong."""
    cl = _rec('amplitude')
    c = Cascade(cl)
    x = c.x()
    y, z = c.direction(), c.direction()
    close('Hessian symmetry B(y,z) == B(z,y)',
          c.d2value(x, y, z), c.d2value(x, z, y), 2e-3)


if __name__ == '__main__':
    sys.exit(run(globals()))
