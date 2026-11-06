"""Windowed near-field ptychography: `RecNFP` with a windowed object shift.

`RecNFPCoded` is `rec_nfp_mpi.RecNFP` with ONE change: the object-plane shift
(level F3, `psi = S_pos(proj)`) reads a per-position WINDOW out of the global
`proj` plane instead of transforming the whole plane.

WHY.  `RecNFP` was built to calibrate a PROBE: the reconstructed plane
(`nobj`) is detector-size plus a modest scan range, so a whole-plane FFT shift
is cheap -- see `AtomiumS2/step0.py`, `nobj = ceil((n + 2*pos_range)/32)*32`.
Step 0 here calibrates the CODED APERTURE instead -- the whole physical mask,
imaged by tiling many scan positions across it -- so `nobj = npsi = 8192`,
4x the detector's linear size. At that ratio, `RecNFP`'s whole-plane shift
(and its `_tiled_coeff`, which broadcasts a COPY of the whole plane per
position in the chunk before even reaching the shift) costs ~40 GB across 4
GPUs. A subclass is a smaller, safer change than editing `rec_nfp_mpi.py`
itself, which many other experiments (`AtomiumS2`, `Y350a_*`, `AtomiumL1_*`,
`ctxl_*`, `WT_cell1`, ...) depend on unchanged.

THE FIX ALREADY EXISTS.  `RecNFP`'s F3 -- `(prb, proj, pos) -> (prb,
S_pos(proj))` -- has the exact same shape as `rec_coded_mpi.RecCoded`'s F3 --
`(prb, code, pos_code, ...) -> (prb, S_pos_code(code), ...)`: a global plane
in, a detector-sized window out, a per-position shift. `ShiftFFTCode`
(shift_fft_code.py), built for that level, already does windowed gather with
a frozen integer offset plus an FFT-interpolated subpixel residual, with the
accuracy and adjoint properties already proven in `tests/coded/`. This file
threads that same machinery through one more call site; every override below
is a direct copy of the corresponding `RecCoded` method, renamed to this
engine's 3-variable cascade.

`rec_nfp_mpi.py` IS NOT MODIFIED.  Every method not listed here (BH,
_compute_step_fused, estimate_rho_coord, error_debug, vis_debug, apply_step,
...) is inherited unchanged.  `pos_pad_frac = 0` (the default `window_pad`
takes) degrades every override below to calling `super()`, so `RecNFPCoded`
at that setting is `RecNFP` exactly -- see `tests/coded/test_rec_nfp_coded.py`.
"""

import time

import numpy as np
import cupy as cp
import nvtx

from .rec_nfp_mpi import RecNFP
from .shift_fft_code import ShiftFFTCode, window_pad
from .utils import make_pinned, reprod, timer
from .logger_config import logger


class RecNFPCoded(RecNFP):

    def __init__(self, args):
        super().__init__(args)

        # Windowed object-plane shift. frac=0 (window_pad's default-off value)
        # keeps self.cl_shift behaving exactly as RecNFP's whole-plane ShiftFFT
        # -- ShiftFFTCode(pad=None) delegates every method to ShiftFFT -- but
        # the TILING below (F3 etc.) also has to agree with that, hence the
        # `self.pos_pad is None` guard in every override, not just here.
        frac = float(getattr(args, 'pos_pad_frac', 0.125))
        self.pos_pad_frac = frac
        self.pos_pad = window_pad(self.n, self.nz, frac)
        if self.pos_pad is None:
            self.cl_shift = ShiftFFTCode(self.n, self.nobj, self.nz,
                                         self.nzobj, self.nchunk)
        else:
            pad_y, pad_x = self.pos_pad
            self.cl_shift = ShiftFFTCode(
                self.n, self.n + 2 * pad_x, self.nz, self.nz + 2 * pad_y,
                self.nchunk, pad=self.pos_pad,
                code_shape=(self.nzobj, self.nobj))
            if self.rank == 0:
                logger.info(
                    f"pos window: [{self.nz + 2*pad_y}, {self.n + 2*pad_x}] "
                    f"(pad {pad_y}, {pad_x}) out of a [{self.nzobj}, "
                    f"{self.nobj}] plane, pos_pad_frac={frac:g}")

        # Frozen integer part of pos, set once in precalc and threaded through
        # the cascade kernels as an ordinary theta-chunked input, exactly as
        # RecCoded does for code_off.
        self._off_chunk = None
        self.pos_off = make_pinned([self.local_ntheta, 2], dtype='int32')
        self.pos_off[:] = 0

    # ------------------------------------------------------------------
    # Frozen offsets
    # ------------------------------------------------------------------

    def freeze_pos_off(self, pos):
        """Pin the integer part of pos for the whole solve.

        NOT re-derived per evaluation: re-rounding it each time would step the
        gather window by a pixel as a shift crosses a half-integer, making the
        objective discontinuous -- the same reasoning as
        RecCoded.freeze_code_off.

        `cp.asnumpy`, not `np.asarray`: unlike RecCoded's pos_code (pinned
        host memory), RecNFP keeps `pos` on the GPU, and this cupy/numpy
        combination raises on an implicit cupy -> numpy conversion.
        """
        self.pos_off[:] = np.round(cp.asnumpy(pos)).astype('int32')

    def check_pos_drift(self, vars):
        """pos keeps moving while the offset stays frozen; the pad absorbs the
        growing residual until it can't.  One small reduction per iteration,
        a no-op when pos_pad_frac=0 (ShiftFFTCode.check_drift returns early)."""
        self.cl_shift.check_drift(vars['pos'], self.pos_off)

    def precalc(self, vars):
        super().precalc(vars)
        self.freeze_pos_off(vars['pos'])

    def _iterate(self, vars, grads, etas, writer=None):
        """RecNFP's loop with one addition: the drift check that catches pos
        moving past what the frozen window can absorb."""
        self.time_start = time.time()
        for i in range(self.start_iter, self.niter):
            with nvtx.annotate(f"::BH:nfp:{i}"):
                self.check_pos_drift(vars)
                self.compute_gradient(vars, grads)
                if getattr(self, 'fused_hessian', True):
                    alpha = self._compute_step_fused(vars, grads, etas, i)
                else:
                    self.compute_beta(vars, grads, etas, i)
                    alpha = self.compute_alpha(vars, grads, etas)
                self.apply_step(vars, etas, alpha)
                self.log_iter(vars, i, writer)

    # ------------------------------------------------------------------
    # F3: (prb, proj, pos) -> (prb, S_pos(proj)), windowed
    #
    # Each override is `super()`'s body with `_tiled_coeff(x32, n)` (which
    # broadcasts the WHOLE plane once per position) replaced by
    # `coeff_cached(x32)` (no tiling -- ShiftFFTCode's windowed gather reads
    # straight from the single global plane) and `off=self._off_chunk` added
    # to every shift call.  `pos_pad is None` falls through to `super()`
    # exactly, tiling included, so this degrades to RecNFP's own computation.
    # ------------------------------------------------------------------

    def F3(self, x):
        if self.pos_pad is None:
            return super().F3(x)
        x31, x32, x33 = x
        off = self._off_chunk
        c = self.cl_shift.coeff_cached(x32)
        m = self.cl_shift.unit_mag(len(x33))
        return x31, self.cl_shift.curlySc(c, x33, m, off)

    def dF3(self, x, y, return_x=True):
        if self.pos_pad is None:
            return super().dF3(x, y, return_x=return_x)
        x31, x32, x33 = x
        y31, y32, y33 = y
        off = self._off_chunk
        c  = self.cl_shift.coeff_cached(x32)
        c1 = self.cl_shift.coeff_cached(y32)
        m = self.cl_shift.unit_mag(len(x33))
        y22 = self.cl_shift.dcurlySc(c, x33, m, c1, y33, off)
        if return_x:
            x22 = self.cl_shift.curlySc(c, x33, m, off)
            return [x31, x22], [y31, y22]
        return [y31, y22]

    def d2F_dF3(self, x, y, z, w):
        if self.pos_pad is None:
            return super().d2F_dF3(x, y, z, w)
        x31, x32, x33 = x
        y31, y32, y33 = y
        z31, z32, z33 = z
        w31, w32, w33 = w
        off = self._off_chunk
        c  = self.cl_shift.coeff_cached(x32)
        cy = self.cl_shift.coeff_cached(y32)
        cz = self.cl_shift.coeff_cached(z32)
        m = self.cl_shift.unit_mag(len(x33))
        # Coefficients CROSSED: the kernel contracts each coefficient with the
        # shift in its own slot, so the mixed second differential needs c_z
        # against dr_y and c_y against dr_z.  Same convention as
        # RecCoded.d2F_dF3 and this method's own whole-plane version above.
        y22 = self.cl_shift.d2curlySc(c, x33, m, cz, y33, cy, z33, off)
        if w32 is not None:
            cw = self.cl_shift.coeff_cached(w32)
            y22 = y22 + self.cl_shift.dcurlySc(c, x33, m, cw, w33, off)
        return [w31, y22]

    def gF3(self, x, y):
        if self.pos_pad is None:
            return super().gF3(x, y)
        y21, y22 = y
        x = self.apply_F_from(x, 4)
        x31, x32, x33 = x
        off = self._off_chunk
        c = self.cl_shift.coeff_cached(x32)
        m = self.cl_shift.unit_mag(len(x33))
        Deltapsi, y33 = self.cl_shift.dcurlySadjc(c, x33, m, y22, off)
        # Deltapsi is on the WINDOW grid; scatter_windows is the adjoint of
        # the gather -- it tapers and folds each window back into the global
        # plane at its own offset, accumulating where windows overlap. The
        # whole-plane version's `cp.sum(Deltapsi, axis=0)` has no equivalent
        # here since Deltapsi is no longer one array per global pixel.
        y32 = self.cl_shift.scatter_windows(Deltapsi, off)
        return [y21, y32, y33]

    # ------------------------------------------------------------------
    # The five @gpu_batch kernels that call into the F-cascade: each is
    # RecNFP's body with `pos_off` added as an extra PROPER (theta-chunked)
    # input -- grouped with the other proper arrays, as rec_coded_mpi.py
    # does for code_off -- and `self._off_chunk = off` set before the
    # cascade runs.
    # ------------------------------------------------------------------

    @timer
    def gradients_cascade(self, vars, grads):
        grads['prb'][:]  = 0
        grads['proj'][:] = 0

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=3)
        def _gradients_cascade(self, gradpos, gradprb, gradproj,
                               d, pos, off, prb, proj):
            self.cl_shift.coeff_cache_reset()
            self._off_chunk = off
            x = [prb, proj, pos]
            y = d
            for id in range(len(self.gF)):
                y = self.gF[id](x, y)
            gradprb[:]  += y[0]
            gradproj[:] += y[1]
            gradpos[:]   = y[2]

        _gradients_cascade(
            self, grads['pos'], grads['prb'], grads['proj'],
            self.data, vars['pos'], self.pos_off, vars['prb'], vars['proj'],
        )

    @timer
    def hessian_cascade(self, vars, grads, etas):
        out = cp.zeros(1, dtype="float32")

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _hessian_cascade(
            self, out, d,
            x2, y2, z2,   # pos  -- proper (theta-distributed)
            off,          # frozen integer offsets -- proper, same axis as pos
            x0, y0, z0,   # prb  -- non-proper gpu
            x1, y1, z1,   # proj -- non-proper gpu
        ):
            self.cl_shift.coeff_cache_reset()
            self._off_chunk = off
            x = [x0, x1, x2]
            y = [y0, y1, y2]
            z = [z0, z1, z2]
            w = [None, None, None]
            y_is_z = y[0] is z[0]

            for id in range(1, len(self.F))[::-1]:
                w = self.d2F_dF[id](x, y, z, w)
                fx, y = self.dF[id](x, y)
                if y_is_z:
                    z = y
                else:
                    z = self.dF[id](x, z, return_x=False)
                x = fx

            out[:] += self.d2F_dF[0](x, y, z, w, d)

        _hessian_cascade(
            self, out, self.data,
            vars["pos"],  grads["pos"],  etas["pos"],
            self.pos_off,
            vars["prb"],  grads["prb"],  etas["prb"],
            vars["proj"], grads["proj"], etas["proj"],
        )
        return out[0].get()

    @timer
    def hessian_cascade3(self, vars, grads, etas):
        out = cp.zeros(3, dtype="float32")
        assert self.local_ntheta != 3, "local_ntheta==3 aliases the accumulator"

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _hessian_cascade3(
            self, out, d,
            x2, g2, e2,   # pos  -- proper (theta-distributed)
            off,          # frozen integer offsets -- proper, same axis as pos
            x0, g0, e0,   # prb  -- non-proper gpu
            x1, g1, e1,   # proj -- non-proper gpu
        ):
            self.cl_shift.coeff_cache_reset()
            self._off_chunk = off
            x = [x0, x1, x2]
            g = [g0, g1, g2]
            e = [e0, e1, e2]
            wgg = [None, None, None]
            wge = [None, None, None]
            wee = [None, None, None]
            for id in range(1, len(self.F))[::-1]:
                wgg = self.d2F_dF[id](x, g, g, wgg)
                wge = self.d2F_dF[id](x, g, e, wge)
                wee = self.d2F_dF[id](x, e, e, wee)
                fx, gn = self.dF[id](x, g)
                en = self.dF[id](x, e, return_x=False)
                x, g, e = fx, gn, en
            out[0:1] += self.d2F_dF[0](x, g, g, wgg, d)
            out[1:2] += self.d2F_dF[0](x, g, e, wge, d)
            out[2:3] += self.d2F_dF[0](x, e, e, wee, d)

        _hessian_cascade3(
            self, out, self.data,
            vars["pos"],  grads["pos"],  etas["pos"],
            self.pos_off,
            vars["prb"],  grads["prb"],  etas["prb"],
            vars["proj"], grads["proj"], etas["proj"],
        )
        h = out.get()
        return float(h[0]), float(h[1]), float(h[2])

    @timer
    def min(self, prb, proj, pos):
        out = cp.zeros(1, dtype="float32")

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _min(self, out, pos, off, data, prb, proj):
            self.cl_shift.coeff_cache_reset()
            self._off_chunk = off
            x = [prb, proj, pos]
            y = self.apply_F_from(x, 1)
            out[:] += self.F0(y, data)

        _min(self, out, pos, self.pos_off, self.data, prb, proj)
        return float(self.allreduce(np.array([out[0].get()], dtype='float32'))[0])

    def gen_data(self, vars, out):
        """Generate synthetic data. See RecNFP.gen_data for the convention
        (writes INTENSITY, PSF-blurred)."""
        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _gen_data(self, out, pos, off, prb, proj):
            self.cl_shift.coeff_cache_reset()
            self._off_chunk = off
            x = [prb, proj, pos]
            y = self.apply_F_from(x, 1)
            out[:] = self._blur(reprod(y, y))
        _gen_data(self, out, vars['pos'], self.pos_off, vars['prb'], vars['proj'])
