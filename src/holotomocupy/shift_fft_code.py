"""Windowed FFT shift for the coded-aperture plane.

`ShiftFFTCode` is `ShiftFFT` plus a *windowed* mode, used by
`rec_coded_mpi.RecCoded` for level F3 — the shift of the coded aperture.

WHY A WINDOW.  The code plane is one global `[ncode, ncode]` array, typically
`ncode = 4n`, but each projection only ever reads an `[nz, n]` region of it.
Transforming the whole plane per projection, as the object-plane shift must,
does 16x the necessary work and dominated the run.  Instead each projection
gathers its own `[nz + 2*pad_y, n + 2*pad_x]` window out of the global plane
and only that is transformed: 5-6.6x measured end to end on the level-3
operators.

THE PRICE.  Truncating the periodic-sinc tail at the window edge leaves a
residual error of order `1/sqrt(pad)`; the Gibbs ringing off that
discontinuity dominates it, which is what `_taper` smooths.  Measured against
a float64 whole-plane reference: ~1.5e-4 rms at bin 0, ~1.1e-3 at bin 2.  In
exchange the integer part of the shift is carried exactly by the gather, so on
an integer shift the window is 15-20x MORE accurate than the whole-plane
transform, which drags the data through an `ncode**2` float32 transform.

FROZEN OFFSETS.  The caller splits each shift into a frozen integer offset
`off` (carried by the gather, exact and free) and a subpixel residual (carried
by the FFT phase).  `off` must be frozen for the whole solve, not re-derived
from the live `pos_code`: re-rounding would step the window by a pixel each
time a shift crosses a half-integer, making the objective discontinuous and
breaking the line search and the Hessian.  `check_drift` guards the residual
against growing past what the pad can absorb.

UNIT MAGNIFICATION ONLY.  The code plane sits at `z1_code` and is not
magnified relative to itself, so the windowed path takes `m = 1` and raises on
anything else.  With `pad=None` this class is exactly `ShiftFFT` — every
method falls through to `super()` — so it is a safe drop-in anywhere.

Nothing in `shift_fft.py` is modified.
"""

import cupy as cp

from .shift_fft import (ShiftFFT, ascontig, apply_sep_phase, imag_conj_prod,
                        combine_dcurlySc, combine_d2curlySc)


# Gather the [My, Mx] window of each projection out of the global [ncy, ncx]
# plane and apply the separable taper, in one pass.
#
# Index convention — window sample (y, x) of projection t reads global
#   ((y - off[t,0] - pady + ctry) mod ncy, (x - off[t,1] - padx + ctrx) mod ncx)
#
# `ctr` is what makes the window agree with a whole-plane instance, and it is
# easy to miss.  A whole-plane instance has eff_d = (ncode - nz)/2, so `S`
# returns c[i - r + (ncode-nz)/2]: the output region sits *centred* in the code
# plane, not at its corner.  The windowed instance has eff_d = pad, so it
# returns W[i - resid + pad]; substituting the gather above gives
# c[i - r + ctr + (resid - r + off)], which matches iff ctr = (ncode-nz)/2 and
# resid = r - off.  Drop ctr and every window is taken (ncode-nz)/2 pixels off
# — 3072 px at bin 0.
#
# The wrap is required (some windows straddle the plane edge) and reproduces
# what a whole-plane FFT shift already does there.
gather_taper = cp.ElementwiseKernel(
    'raw complex64 c, raw int32 off, raw float32 ty, raw float32 tx, '
    'int32 ncy, int32 ncx, int32 My, int32 Mx, int32 pady, int32 padx, '
    'int32 ctry, int32 ctrx',
    'complex64 out',
    '''
    int x = i % Mx;
    int t = i / Mx;
    int y = t % My;
    t = t / My;
    int gy = (y - off[2 * t]     - pady + ctry) % ncy;  if (gy < 0) gy += ncy;
    int gx = (x - off[2 * t + 1] - padx + ctrx) % ncx;  if (gx < 0) gx += ncx;
    out = c[gy * (long long)ncx + gx] * (ty[y] * tx[x]);
    ''',
    'shift_fft_code_gather_taper',
)

# Adjoint of gather_taper: taper, then scatter-add each window back into the
# global plane.  atomicAdd because windows overlap — a plain indexed store
# would silently drop all but one contribution per pixel.
scatter_taper = cp.ElementwiseKernel(
    'complex64 w, raw int32 off, raw float32 ty, raw float32 tx, '
    'int32 ncy, int32 ncx, int32 My, int32 Mx, int32 pady, int32 padx, '
    'int32 ctry, int32 ctrx',
    'raw complex64 out',
    '''
    int x = i % Mx;
    int t = i / Mx;
    int y = t % My;
    t = t / My;
    int gy = (y - off[2 * t]     - pady + ctry) % ncy;  if (gy < 0) gy += ncy;
    int gx = (x - off[2 * t + 1] - padx + ctrx) % ncx;  if (gx < 0) gx += ncx;
    complex<float> v = w * (ty[y] * tx[x]);
    float* p = reinterpret_cast<float*>(&out[gy * (long long)ncx + gx]);
    atomicAdd(p,     v.real());
    atomicAdd(p + 1, v.imag());
    ''',
    'shift_fft_code_scatter_taper',
)


def smooth_size(m):
    """Smallest 2^a*3^b*5^c >= m.  cuFFT is fast on these and slow (Bluestein)
    on sizes with a large prime factor, which constrains the pad more than
    accuracy does."""
    best = None
    p2 = 1
    while p2 < 2 * m + 1:
        p23 = p2
        while p23 < 2 * m + 1:
            p235 = p23
            while p235 < 2 * m + 1:
                if p235 >= m and (best is None or p235 < best):
                    best = p235
                p235 *= 5
            p23 *= 3
        p2 *= 2
    return best


def window_pad(n, nz, frac):
    """Pad per axis, `(pad_y, pad_x)`, each the smallest pad >= frac*n whose
    window size lands on a cuFFT-friendly value.  `frac=0` disables windowing
    (returns None).

    The two axes are sized independently on purpose.  Forcing one pad to make
    both `n + 2p` and `nz + 2p` smooth can search forever — for n=2048, nz=1000
    it wants two 5-smooth numbers exactly 1048 apart, and those are sparse
    enough that there may be none in any reasonable range.

    For the production grids (square, n a power of two, frac=1/8) both axes
    give n/8 exactly, i.e. M = 1.25*n = 2^k*5.
    """
    if not frac:
        return None
    base = int(n * frac)

    def axis_pad(N):
        M = smooth_size(N + 2 * base)
        # Keep the pad symmetric, so nudge M up to the next smooth even offset.
        while (M - N) % 2:
            M = smooth_size(M + 1)
        return (M - N) // 2

    return axis_pad(nz), axis_pad(n)


class ShiftFFTCode(ShiftFFT):
    """`ShiftFFT` with an optional per-projection window over a global plane.

    `pad` set means this instance is sized to a WINDOW (`npsi = n + 2*pad_x`,
    `nzpsi = nz + 2*pad_y`) rather than to the full plane.  The caller then
    passes the 2-D global plane of shape `code_shape` as `c`, together with
    frozen per-projection integer offsets `off`.  `pad` is an int (same both
    axes) or a `(pad_y, pad_x)` pair.

    `pad=None` leaves every method delegating to `ShiftFFT`.
    """

    def __init__(self, n, npsi, nz, nzpsi, nchunk=None,
                 pad=None, code_shape=None):
        super().__init__(n, npsi, nz, nzpsi, nchunk)

        self.pad = pad
        self.code_shape = code_shape
        if pad is None:
            return

        if code_shape is None:
            raise ValueError('windowed mode (pad != None) needs code_shape')
        self.pad_y, self.pad_x = (pad, pad) if isinstance(pad, int) else pad
        if npsi != n + 2 * self.pad_x or nzpsi != nz + 2 * self.pad_y:
            raise ValueError(
                f'windowed mode expects npsi = n + 2*pad_x and '
                f'nzpsi = nz + 2*pad_y; got n={n} nz={nz} pad={pad} '
                f'npsi={npsi} nzpsi={nzpsi}')
        self.tap_y = self._taper(nzpsi, self.pad_y)
        self.tap_x = self._taper(npsi,  self.pad_x)
        # Centring term: reproduces a whole-plane instance's
        # eff_d = (ncode - nz)/2.  An odd (ncode - nz) leaves half a pixel the
        # integer gather cannot carry, so it goes into the residual instead.
        ncy, ncx = code_shape
        ctr_y, ctr_x = (ncy - nz) * 0.5, (ncx - n) * 0.5
        self.ctr_y, self.ctr_x = int(ctr_y // 1), int(ctr_x // 1)
        self.ctr_frac = cp.asarray(
            [[ctr_y - self.ctr_y, ctr_x - self.ctr_x]], dtype='float32')

    # ------------------------------------------------------------------
    # Windowed-mode helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _taper(M, pad):
        """Raised cosine over the outer `pad` samples, interior exactly 1.

        The window's hard edge is what limits accuracy: truncating the
        periodic-sinc tail leaves ~1/sqrt(pad) rms error, and the Gibbs ringing
        off that discontinuity dominates it.  Smoothing the ring is one
        elementwise multiply and is worth more than 4x the pad at bin 0
        (measured 9.2e-4 -> 2.0e-4 at pad=16, n=2048).
        """
        t = cp.ones(M, dtype='float32')
        if pad > 0:
            i = cp.arange(pad, dtype='float32')
            ramp = (0.5 * (1 - cp.cos(cp.pi * (i + 0.5) / pad))).astype('float32')
            t[:pad] = ramp
            t[M - pad:] = ramp[::-1]
        return t

    def _require_unit(self, m):
        """Windowed mode is unit-magnification only.  The chirp-z path resamples
        the input grid, which a gathered window is not."""
        if not self._is_unit_mag(m):
            raise NotImplementedError(
                'windowed mode does not support magnification != 1')

    @staticmethod
    def _off(off, ntheta):
        off = cp.ascontiguousarray(cp.asarray(off, dtype='int32'))
        if off.shape != (ntheta, 2):
            raise ValueError(f'off must be [{ntheta}, 2] int32, got {off.shape}')
        return off

    def _resid(self, r, off):
        """The subpixel residual the FFT actually interpolates.

        The integer part is carried by the gather offset, where it is exact and
        free; `ctr_frac` is the half pixel left over when (ncode - nz) is odd
        and the integer `ctr` cannot express it.
        """
        r = cp.asarray(r)
        off = cp.asarray(off, dtype='float32')
        # Catches an `off` table out of step with the chunk, which is the
        # likeliest plumbing failure and would otherwise broadcast into silent
        # garbage.
        if off.shape != r.shape:
            raise ValueError(f'off {off.shape} does not match r {r.shape} — '
                             f'the offset table is out of step with the chunk')
        return r - off - self.ctr_frac

    def _prep(self, c, off):
        """Gather + taper each projection's window out of the global plane."""
        if off is None:
            raise ValueError('windowed mode needs per-projection offsets `off`')
        if c.ndim != 2:
            raise ValueError(
                f'windowed mode expects a 2-D global plane, got {c.shape}')
        ncy, ncx = c.shape
        ntheta = cp.asarray(off).shape[0]
        off = self._off(off, ntheta)
        out = cp.empty([ntheta, self.nzpsi, self.npsi], dtype='complex64')
        gather_taper(self.to_complex(ascontig(c)), off, self.tap_y, self.tap_x,
                     cp.int32(ncy), cp.int32(ncx),
                     cp.int32(self.nzpsi), cp.int32(self.npsi),
                     cp.int32(self.pad_y), cp.int32(self.pad_x),
                     cp.int32(self.ctr_y), cp.int32(self.ctr_x), out)
        return out

    def scatter_windows(self, w, off, out=None):
        """Adjoint of the gather: taper each window and scatter-add it into the
        global plane.  Overlapping windows accumulate."""
        if self.pad is None:
            raise RuntimeError('scatter_windows is windowed mode only')
        if out is None:
            out = cp.zeros(self.code_shape, dtype='complex64')
        ncy, ncx = out.shape
        off = self._off(off, w.shape[0])
        scatter_taper(w, off, self.tap_y, self.tap_x,
                      cp.int32(ncy), cp.int32(ncx),
                      cp.int32(self.nzpsi), cp.int32(self.npsi),
                      cp.int32(self.pad_y), cp.int32(self.pad_x),
                      cp.int32(self.ctr_y), cp.int32(self.ctr_x), out)
        return out

    def check_drift(self, r, off):
        """The offsets are frozen for the whole solve while `pos_code` keeps
        being optimized, so the residual drifts away from 0.5 px.  The pad
        absorbs it, but silently losing accuracy is worse than stopping."""
        if self.pad is None:
            return
        d = float(cp.abs(cp.asarray(r) - cp.asarray(off, dtype='float32')).max())
        # Margin the taper ramp needs to stay clear of the data region.
        lim = min(self.pad_y, self.pad_x) - self.n // 16
        if d > lim:
            raise RuntimeError(
                f'pos_code drifted {d:.2f} px from the frozen offset, past the '
                f'{lim} px the pad can absorb — refreeze the offsets or raise '
                f'code_pad_frac')

    # ------------------------------------------------------------------
    # Overridden operators
    #
    # Each is the ShiftFFT body with exactly two substitutions:
    #     phase_separable(r)        ->  phase_separable(_resid(r, off))
    #     to_complex(ascontig(c))   ->  _prep(c, off)
    # applied to every coefficient argument.  Sadj needs neither — it already
    # works on the window grid — and is inherited unchanged.
    # ------------------------------------------------------------------

    def S(self, c, r, m, off=None):
        if self.pad is None:
            return super().S(c, r, m)
        self._require_unit(m)
        py, px = self.phase_separable(self._resid(r, off))
        C = self.fft2(self._prep(c, off))
        apply_sep_phase(C, py[:, :, cp.newaxis], px[:, cp.newaxis, :], C)
        s = self.ifft2(C)
        del C
        return self.from_complex(s[:, :self.nz, :self.n])

    def curlyS(self, psi, r, m, off=None):
        return self.S(psi, r, m, off)

    def curlySc(self, c, r, m, off=None):
        return self.S(c, r, m, off)

    def dcurlySc(self, c, r, m, c1, Deltar, off=None):
        """dS(c,r)/dc . c1 + dS(c,r)/dr . Delta_r."""
        if self.pad is None:
            return super().dcurlySc(c, r, m, c1, Deltar)
        self._require_unit(m)
        py, px = self.phase_separable(self._resid(r, off))
        C  = self.fft2(self._prep(c,  off))
        C1 = self.fft2(self._prep(c1, off))
        D  = self.deriv_factor(Deltar)
        # Fused: out = (C*D + C1)*py*px, in-place into C.
        combine_dcurlySc(C, D, C1,
                         py[:, :, cp.newaxis], px[:, cp.newaxis, :], C)
        del D, C1
        s = self.ifft2(C)
        del C
        return self.from_complex(s[:, :self.nz, :self.n])

    def dcurlySadjc(self, c, r, m, Deltaphi, off=None):
        """Adjoint of (c1, Delta_r) -> dcurlySc(c, r, m, c1, Delta_r), applied
        to Delta_phi.  Returns [out1, out2] with out1 = Sadj(Delta_phi) and
        out2[t, 0/1] = redot(Delta_phi, dS/d(ry/rx) . c).

        In windowed mode out1 is on the WINDOW grid [ntheta, nzpsi, npsi]; the
        caller folds it back into the global plane with `scatter_windows`,
        which applies the taper (the adjoint of the gather's taper).
        """
        if self.pad is None:
            return super().dcurlySadjc(c, r, m, Deltaphi)
        self._require_unit(m)
        ntheta = cp.asarray(r).shape[0]
        py, px = self.phase_separable(self._resid(r, off))
        py_b = py[:, :, cp.newaxis]
        px_b = px[:, cp.newaxis, :]

        # fft(ZeroPad(Delta_phi)) on the window grid — reused for out1 (Sadj)
        # and out2 (the Parseval redots), so it outlives out1.
        Phat = self.fft2(self.pad_output(Deltaphi))

        Sbuf = Phat.copy()
        apply_sep_phase(Sbuf, cp.conj(py_b), cp.conj(px_b), Sbuf)
        out1 = self.from_complex(self.ifft2(Sbuf))
        del Sbuf

        Cshift = self.fft2(self._prep(c, off))
        apply_sep_phase(Cshift, py_b, px_b, Cshift)

        # Im(conj(Phat) * Cshift) without the big complex intermediate.
        PhatC_im = imag_conj_prod(Phat, Cshift)
        del Phat, Cshift

        # Separable sum-then-dot, avoiding the fy[None,:,None]*PhatC_im temp.
        sum_x = PhatC_im.sum(axis=2)   # [ntheta, nzpsi]
        sum_y = PhatC_im.sum(axis=1)   # [ntheta, npsi]
        del PhatC_im

        out2 = cp.empty([ntheta, 2], dtype='float32')
        out2[:, 0] = self.inv_N * (sum_x @ self.fy)
        out2[:, 1] = self.inv_N * (sum_y @ self.fx)
        return [out1, out2]

    def d2curlySc(self, c, r, m, c1, Deltar1, c2, Deltar2, off=None):
        """Second directional derivative on (c, r); d2S/dc2 = 0, so

            Crop(ifft((C*D1*D2 + C1*D1 + C2*D2) * P))

        SLOT PAIRING.  This follows `ShiftFFT.d2curlySc` exactly: each
        coefficient is contracted with the shift in its OWN slot, and the
        CALLER crosses them.  A caller wanting the mixed form B(y, z) along
        y = (c_y, dr_y) and z = (c_z, dr_z) passes

            d2curlySc(c, r, m, c_z, dr_y, c_y, dr_z)

        which is what `RecCoded.d2F_dF3` does, matching `Rec.d2F_dF3`.  On the
        diagonal (y is z) the two conventions agree, so only an off-diagonal
        check — the polarization identity
        B(y+z, y+z) == B(y,y) + 2 B(y,z) + B(z,z) — can see the difference.
        """
        if self.pad is None:
            return super().d2curlySc(c, r, m, c1, Deltar1, c2, Deltar2)
        self._require_unit(m)
        py, px = self.phase_separable(self._resid(r, off))
        D1 = self.deriv_factor(Deltar1)
        D2 = self.deriv_factor(Deltar2)
        C  = self.fft2(self._prep(c,  off))
        C1 = self.fft2(self._prep(c1, off))
        C2 = self.fft2(self._prep(c2, off))
        combine_d2curlySc(C, C1, C2, D1, D2,
                          py[:, :, cp.newaxis], px[:, cp.newaxis, :], C)
        del D1, D2, C1, C2
        s = self.ifft2(C)
        del C
        return self.from_complex(s[:, :self.nz, :self.n])
