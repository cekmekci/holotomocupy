#!/usr/bin/env python
"""The windowed code-plane shift, `shift_fft_code.ShiftFFTCode`.

    python tests/coded/test_shift_code.py

`ShiftFFTCode` gathers a per-projection window out of one global `[ncode,
ncode]` plane instead of transforming the whole plane, so it has to agree with
the whole-plane operator it replaces, and its own derivatives and adjoints
have to be exact.  Four groups:

  pass-through    with `pad=None` every method must BE `ShiftFFT`, bit for bit
  forward         the window against a whole-plane `ShiftFFT` reference
  adjoints        `scatter_windows` as the exact adjoint of the gather, and
                  `dcurlySadjc` as the adjoint of `dcurlySc`
  derivatives     Taylor order of `dcurlySc` (2) and `d2curlySc` (3), plus the
                  polarization identity, which is the only check that can see
                  the d2 slot pairing

The input is band-limited: the FFT shift is the sinc interpolant, and a sharp
edge would measure Gibbs ringing instead of calculus.
"""
import os
import sys

import numpy as np
import cupy as cp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'unit'))
from harness import check, close, order, run                      # noqa: E402

from holotomocupy.shift_fft import ShiftFFT                       # noqa: E402
from holotomocupy.shift_fft_code import (ShiftFFTCode, window_pad,  # noqa: E402
                                         smooth_size)
from holotomocupy.utils import redot                              # noqa: E402

# Small enough to run in seconds, big enough that the pad is not the whole
# window.  ncode = 4n is the production ratio.
N, NZ, NTHETA = 64, 64, 5
NCODE = 4 * N
FRAC = 0.125


def _band_limited(shape, seed, cutoff=0.25):
    """Random field with its spectrum zeroed above `cutoff` of Nyquist."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    A = np.fft.fft2(a)
    fy = np.fft.fftfreq(shape[-2])[:, None]
    fx = np.fft.fftfreq(shape[-1])[None, :]
    A[..., (fy ** 2 + fx ** 2) > cutoff ** 2] = 0
    return cp.asarray(np.fft.ifft2(A), dtype='complex64')


def _setup():
    """A windowed operator, a whole-plane reference, and a consistent
    (code, r, off) triple with a sub-pixel residual."""
    pad = window_pad(N, NZ, FRAC)
    pad_y, pad_x = pad
    win = ShiftFFTCode(N, N + 2 * pad_x, NZ, NZ + 2 * pad_y, nchunk=None,
                       pad=pad, code_shape=(NCODE, NCODE))
    full = ShiftFFT(N, NCODE, NZ, NCODE, nchunk=None)
    code = _band_limited((NCODE, NCODE), 0)[None][0]              # [NCODE, NCODE]
    rng = np.random.default_rng(1)
    # Shifts well inside the plane, each with a non-trivial fractional part.
    r = cp.asarray(rng.uniform(-20, 20, (NTHETA, 2)), dtype='float32')
    off = cp.round(r).astype('int32')
    m = win.unit_mag(NTHETA)
    return win, full, code, r, off, m


# ----------------------------------------------------------------------
# pass-through
# ----------------------------------------------------------------------

def test_pad_none_is_shift_fft():
    """pad=None must reproduce ShiftFFT exactly, so the subclass is a safe
    drop-in for every non-coded caller."""
    a = ShiftFFTCode(N, 2 * N, NZ, 2 * NZ, nchunk=None)      # no pad
    b = ShiftFFT(N, 2 * N, NZ, 2 * NZ, nchunk=None)
    c = _band_limited((NTHETA, 2 * NZ, 2 * N), 2)
    c1 = _band_limited((NTHETA, 2 * NZ, 2 * N), 3)
    rng = np.random.default_rng(4)
    r = cp.asarray(rng.uniform(-5, 5, (NTHETA, 2)), dtype='float32')
    dr = cp.asarray(rng.uniform(-1, 1, (NTHETA, 2)), dtype='float32')
    m = b.unit_mag(NTHETA)
    g = _band_limited((NTHETA, NZ, N), 5)

    for nm, u, v in (
            ('curlySc',   a.curlySc(c, r, m),            b.curlySc(c, r, m)),
            ('dcurlySc',  a.dcurlySc(c, r, m, c1, dr),   b.dcurlySc(c, r, m, c1, dr)),
            ('d2curlySc', a.d2curlySc(c, r, m, c1, dr, c1, dr),
                          b.d2curlySc(c, r, m, c1, dr, c1, dr)),
            ('Sadj',      a.Sadj(g, r, m),               b.Sadj(g, r, m))):
        check(f'pad=None: {nm} identical to ShiftFFT',
              bool(cp.array_equal(u, v)),
              f'max |diff| {float(cp.abs(u - v).max()):.3e}')

    ua, _ = a.dcurlySadjc(c, r, m, g)
    ub, _ = b.dcurlySadjc(c, r, m, g)
    check('pad=None: dcurlySadjc identical to ShiftFFT',
          bool(cp.array_equal(ua, ub)))


# ----------------------------------------------------------------------
# forward
# ----------------------------------------------------------------------

def test_window_matches_whole_plane():
    """The window reproduces a whole-plane shift of the same global code.

    Not exact: truncating the periodic-sinc tail at the window edge is the
    approximation the whole scheme trades for the speed.  Expected ~1e-3 rms
    relative at this size, and it tightens as pad grows (see the next test).
    """
    win, full, code, r, off, m = _setup()
    got = win.curlySc(code, r, m, off)
    # The whole-plane reference needs the code broadcast over projections.
    ref = full.curlySc(cp.broadcast_to(code, (NTHETA,) + code.shape).copy(), r, m)
    err = float(cp.linalg.norm(got - ref) / cp.linalg.norm(ref))
    check('window vs whole plane: forward', err < 5e-3, f'rms rel {err:.2e}')


def test_integer_shift_is_near_exact():
    """On an integer shift the gather carries the whole displacement, so the
    FFT only interpolates `ctr_frac` (zero when ncode - nz is even).  This is
    the case the window is MORE accurate than the whole-plane transform, which
    drags the data through an ncode^2 float32 transform."""
    win, full, code, _, _, m = _setup()
    rng = np.random.default_rng(7)
    r = cp.asarray(rng.integers(-20, 20, (NTHETA, 2)), dtype='float32')
    off = cp.round(r).astype('int32')
    got = win.curlySc(code, r, m, off)
    ref = full.curlySc(cp.broadcast_to(code, (NTHETA,) + code.shape).copy(), r, m)
    err = float(cp.linalg.norm(got - ref) / cp.linalg.norm(ref))
    check('window vs whole plane: integer shift', err < 1e-5,
          f'rms rel {err:.2e}')


def test_accuracy_improves_with_pad():
    """Error falls as the pad grows -- the signature of a truncation error
    rather than a wrong index convention, which would be pad-independent."""
    win, full, code, r, off, m = _setup()
    ref = full.curlySc(cp.broadcast_to(code, (NTHETA,) + code.shape).copy(), r, m)
    errs = []
    for frac in (0.0625, 0.125, 0.25):
        pad_y, pad_x = window_pad(N, NZ, frac)
        w = ShiftFFTCode(N, N + 2 * pad_x, NZ, NZ + 2 * pad_y, nchunk=None,
                         pad=(pad_y, pad_x), code_shape=(NCODE, NCODE))
        got = w.curlySc(code, r, w.unit_mag(NTHETA), off)
        errs.append(float(cp.linalg.norm(got - ref) / cp.linalg.norm(ref)))
    check('accuracy improves with pad', errs[0] > errs[1] > errs[2],
          f'rms rel {errs[0]:.2e} -> {errs[1]:.2e} -> {errs[2]:.2e}')


def test_window_pad_is_cufft_friendly():
    """window_pad must land the window size on a 2-3-5-smooth number, or every
    transform silently takes cuFFT's Bluestein path."""
    for n, nz in ((64, 64), (256, 256), (2048, 1000)):
        pad_y, pad_x = window_pad(n, nz, FRAC)
        for nm, size in (('y', nz + 2 * pad_y), ('x', n + 2 * pad_x)):
            check(f'window_pad({n},{nz}) {nm}-size {size} is smooth',
                  smooth_size(size) == size)


# ----------------------------------------------------------------------
# adjoints
# ----------------------------------------------------------------------

def test_scatter_is_adjoint_of_gather():
    """<gather(c), w> == <c, scatter(w)>.  The windows overlap, so a scatter
    that dropped contributions instead of accumulating would fail here and
    nowhere else."""
    win, _, code, _, off, _ = _setup()
    w = _band_limited((NTHETA, win.nzpsi, win.npsi), 11)
    lhs = float(redot(win._prep(code, off), w))
    rhs = float(redot(code, win.scatter_windows(w, off)))
    close('scatter_windows is the adjoint of the gather', lhs, rhs, rtol=2e-5)


def test_dcurlySadjc_is_adjoint():
    """<dcurlySc(c1, dr), g> == <c1, out1_folded> + <dr, out2>.

    out1 comes back on the WINDOW grid, so folding it to the global plane with
    scatter_windows is part of the adjoint -- this is the identity the cascade
    gradient actually relies on.
    """
    win, _, code, r, off, m = _setup()
    c1 = _band_limited((NCODE, NCODE), 12)
    rng = np.random.default_rng(13)
    dr = cp.asarray(rng.standard_normal((NTHETA, 2)), dtype='float32')
    g = _band_limited((NTHETA, NZ, N), 14)

    lhs = float(redot(win.dcurlySc(code, r, m, c1, dr, off), g))
    out1, out2 = win.dcurlySadjc(code, r, m, g, off)
    rhs = (float(redot(c1, win.scatter_windows(out1, off)))
           + float(redot(dr, out2)))
    close('dcurlySadjc is the adjoint of dcurlySc', lhs, rhs, rtol=2e-4)


# ----------------------------------------------------------------------
# derivatives
# ----------------------------------------------------------------------

def test_dcurlySc_taylor_order():
    """S(c+h*c1, r+h*dr) - S(c,r) - h*dS[c1,dr] = O(h^2)."""
    win, _, code, r, off, m = _setup()
    c1 = _band_limited((NCODE, NCODE), 21)
    rng = np.random.default_rng(22)
    dr = cp.asarray(rng.standard_normal((NTHETA, 2)), dtype='float32')

    f0 = win.curlySc(code, r, m, off)
    d1 = win.dcurlySc(code, r, m, c1, dr, off)
    hs, es = [], []
    for h in (1e-1, 5e-2, 2.5e-2, 1.25e-2):
        fh = win.curlySc(code + h * c1, r + h * dr, m, off)
        hs.append(h)
        es.append(float(cp.abs(fh - f0 - h * d1).max()))
    o = order(hs, es, floor=1e-6)
    check('dcurlySc Taylor order ~2', o > 1.8, f'order {o:.2f}  errs {es}')


def test_d2curlySc_taylor_order():
    """S(c+h*c1, r+h*dr) - S - h*dS - h^2/2 * d2S[(c1,dr),(c1,dr)] = O(h^3).

    On the diagonal the slot pairing is invisible; the polarization test below
    is what sees it.
    """
    win, _, code, r, off, m = _setup()
    c1 = _band_limited((NCODE, NCODE), 31)
    rng = np.random.default_rng(32)
    dr = cp.asarray(rng.standard_normal((NTHETA, 2)), dtype='float32')

    f0 = win.curlySc(code, r, m, off)
    d1 = win.dcurlySc(code, r, m, c1, dr, off)
    d2 = win.d2curlySc(code, r, m, c1, dr, c1, dr, off)
    hs, es = [], []
    for h in (1e-1, 5e-2, 2.5e-2, 1.25e-2):
        fh = win.curlySc(code + h * c1, r + h * dr, m, off)
        hs.append(h)
        es.append(float(cp.abs(fh - f0 - h * d1 - 0.5 * h * h * d2).max()))
    o = order(hs, es, floor=1e-6)
    check('d2curlySc Taylor order ~3', o > 2.6, f'order {o:.2f}  errs {es}')


def test_d2curlySc_polarization():
    """B(y+z, y+z) == B(y,y) + 2 B(y,z) + B(z,z), with the caller CROSSING the
    coefficients for the mixed term exactly as RecCoded.d2F_dF3 does.

    This is the only check that can see the slot pairing: get the crossing
    backwards and the diagonal Taylor test above still passes.
    """
    win, _, code, r, off, m = _setup()
    cy = _band_limited((NCODE, NCODE), 41)
    cz = _band_limited((NCODE, NCODE), 42)
    rng = np.random.default_rng(43)
    dy = cp.asarray(rng.standard_normal((NTHETA, 2)), dtype='float32')
    dz = cp.asarray(rng.standard_normal((NTHETA, 2)), dtype='float32')

    Byy = win.d2curlySc(code, r, m, cy, dy, cy, dy, off)
    Bzz = win.d2curlySc(code, r, m, cz, dz, cz, dz, off)
    # own-slot kernel + caller crossing == the mixed second differential
    Byz = win.d2curlySc(code, r, m, cz, dy, cy, dz, off)
    Bsum = win.d2curlySc(code, r, m, cy + cz, dy + dz, cy + cz, dy + dz, off)

    lhs = Bsum
    rhs = Byy + 2 * Byz + Bzz
    err = float(cp.abs(lhs - rhs).max() / cp.abs(lhs).max())
    check('d2curlySc polarization identity (slot pairing)', err < 1e-4,
          f'max rel {err:.2e}')


# ----------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------

def test_check_drift_fires():
    """A residual past what the pad can absorb must raise, not degrade."""
    win, _, _, _, _, _ = _setup()
    good = cp.zeros((NTHETA, 2), dtype='float32')
    off = cp.zeros((NTHETA, 2), dtype='int32')
    try:
        win.check_drift(good, off)
        ok_quiet = True
    except RuntimeError:
        ok_quiet = False
    check('check_drift silent at zero drift', ok_quiet)

    far = cp.full((NTHETA, 2), 1e3, dtype='float32')
    try:
        win.check_drift(far, off)
        ok_raise = False
    except RuntimeError:
        ok_raise = True
    check('check_drift raises on a large drift', ok_raise)


def test_mismatched_off_raises():
    """An `off` table out of step with the chunk is the likeliest plumbing
    failure; it must raise rather than broadcast into silent garbage."""
    win, _, code, r, _, m = _setup()
    bad = cp.zeros((NTHETA + 1, 2), dtype='int32')
    try:
        win.curlySc(code, r, m, bad)
        raised = False
    except ValueError:
        raised = True
    check('mismatched off raises', raised)


if __name__ == '__main__':
    sys.exit(run(globals()))
