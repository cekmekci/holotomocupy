"""Coded-aperture holotomography: a single-distance BH solver.

`RecCoded` is `rec_mpi.Rec` with the multi-distance machinery removed and a
coded aperture added.  Same bilinear-Hessian method (Carlsson, 2025), same
chunking, same F0 misfit; the cascade is two levels deeper.

THE CASCADE.  Levels are applied highest-index first, as in `rec_mpi`:

    x = [prb, code, pos_code, proj, pos]

    F5   psi    = S_{pos}(proj)              object-plane shift, m = 1
    F4   exp(1j * psi)
    F3   code_s = S_{pos_code}(code)         code-plane shift, WINDOWED
    F2   prb_s  = D_code(prb * code_s)       Fresnel, code plane -> sample
    F1   x0     = D(prb_s * exp(1j psi))     Fresnel, sample -> detector
    F0   misfit                              mask W, PSF K, intensity|amplitude

`rec_mpi.Rec` is the same chain with F3 and F2 deleted and a shrinkage level
added on top.  F0, F1 and the exponential are shared verbatim; F5 is his F3 at
unit magnification.

WHAT IS GONE RELATIVE TO rec_mpi, AND WHY.

  * Multi-distance.  One propagation distance, so there is no `ndist` axis on
    `prb` or `pos`, no `_dist_groups` / `ndistchunk` hoisting, and no
    `_dist_idx`.
  * Magnification.  Follows from the above: with one plane the object shift
    runs at m = 1, so the `*c` shift family is used and never the
    magnification-aware `*mc` one.
  * Shrinkage (`tp`).  A multi-distance drying model; not this geometry.
  * `PrbfitTerm`.  The CA flat field is measured with the code in the beam, so
    K|D prb|^2 is not a model for it.  There is no `self.ref` here.

WHAT IS NEW.

  * `code` -- the aperture transmission, a global `[ncode, ncode]` plane.
    Global like `prb`, so its gradient is allreduced.
  * `pos_code` -- the per-projection code shift, theta-distributed like `pos`.
  * `cl_prop_code` -- the second Fresnel propagator, code plane to sample.
  * `cl_shift_code` -- `ShiftFFTCode`, which shifts each projection's window of
    the code plane rather than the whole plane.  See that module; in particular
    the integer part of `pos_code` is FROZEN for the whole solve
    (`freeze_code_off`) and only the subpixel residual goes through the FFT.

DATA CONVENTION.  `self.data` is the measured INTENSITY, as everywhere else in
this package -- `F0_amp` takes its own sqrt.  (The older coded fork stored
sqrt(intensity); configs carried over from it must drop that sqrt.)
"""

import os
import time
import warnings

import numpy as np
import cupy as cp
import cupy.fft
import nvtx
import pandas as pd
from mpi4py import MPI

from .tomo import Tomo
from .propagation import Propagation
from .shift import Shift
from .shift_fft import ShiftFFT
from .shift_fft_code import ShiftFFTCode, window_pad
from .chunking import Chunking, _axpby
from .extra_terms import LaplacianTerm
from .utils import make_pinned, mshow_approx, redot, reprod, timer
from .psf import psf_taps as _psf_taps, psf_blur as _psf_blur
from .mpi_functions import MPIClass
from .logger_config import logger
from .conv2d_cufftdx import precompile as cufftdx_precompile

np.set_printoptions(legacy="1.25")
warnings.filterwarnings("ignore", message=".*peer.*")
cupy.fft.config.get_plan_cache().set_size(0)   # dont waste GPU memory


class RecCoded:
    # Derive beta and alpha from ONE Hessian sweep instead of three; see the
    # same flag on rec_mpi.Rec. B(y,z) is symmetric, so
    #     beta   = B(g,e) / B(e,e)
    #     bottom = b^2*B(e,e) - 2*b*B(g,e) + B(g,g)
    # all follow from {B(g,g), B(g,e), B(e,e)} of a single pass.
    fused_hessian = True

    # Declared at class level for the same reason rec_mpi declares them: a
    # hand-built args (tests, notebooks) must not be required to carry them,
    # and _pool_bytes_for reads lam_laplacian before __init__ finishes.
    debug_step = -1
    check_approx = False
    lam_laplacian = 0.0

    def __init__(self, args):

        # copy args to elements of the class
        for key, value in vars(args).items():
            setattr(self, key, value)

        if len(args.rho) != 5:
            raise ValueError(
                f"rho must have 5 entries [obj, prb, pos, code, pos_code]; "
                f"got {list(args.rho)}. Use 0 to freeze a variable.")

        # rec_mpi.gradients adds the Laplacian gradient WITHOUT rho_sq['obj']
        # while the misfit half of grads['obj'] carries it, so the two halves
        # share a metric only at rho[obj] = 1 -- which is what every config in
        # this package uses. Refuse rather than silently reweight the
        # regularizer. (Harmless when the term is off.)
        if self.lam_laplacian > 0 and abs(float(args.rho[0]) - 1.0) > 1e-12:
            raise ValueError(
                f"rho[obj] must be 1 when lam_laplacian > 0 (got {args.rho[0]}): "
                f"the Laplacian gradient is added without rho_sq['obj'] while "
                f"the data-misfit half carries it.")

        # list of functionals, gradients, differentials, and second-order differentials
        self.F  = [self.F0, self.F1, self.F2, self.F3, self.F4, self.F5]
        self.gF = [self.gF0, self.gF1, self.gF2, self.gF3, self.gF4, self.gF5]
        self.dF = [self.dF0, self.dF1, self.dF2, self.dF3, self.dF4, self.dF5]
        self.d2F_dF = [self.d2F_dF0, self.d2F_dF1, self.d2F_dF2,
                       self.d2F_dF3, self.d2F_dF4, self.d2F_dF5]

        # Detector oversampling of the Radon transform; see rec_mpi. `nobj` is
        # the OBJECT width, `ndobj` the projection/psi width that R writes onto.
        self.tomo_upsample = int(getattr(args, 'tomo_upsample', 1))
        if self.tomo_upsample not in (1, 2):
            raise ValueError(
                f"tomo_upsample must be 1 or 2, got {self.tomo_upsample}")
        self.ndobj = self.nobj * self.tomo_upsample

        ### multinode processing
        self.cl_mpi = MPIClass(args.comm, self.nzobj, self.ntheta, self.ndobj,
                               'complex64')
        self.local_nzobj = self.cl_mpi.local_nzobj
        self.local_ntheta = self.cl_mpi.local_ntheta
        self.rank      = self.cl_mpi.rank
        self.st_obj    = self.cl_mpi.st_obj
        self.end_obj   = self.cl_mpi.end_obj
        self.st_theta  = self.cl_mpi.st_theta
        self.end_theta = self.cl_mpi.end_theta

        # Chunking tells a proper (theta-chunked) array from a replicated one
        # purely by shape[0] == local_ntheta. `prb` is [nz, n] and `code` is
        # [ncode, ncode], both replicated; if either leading dimension happened
        # to equal this rank's angle count they would be silently sliced per
        # chunk. The scalar accumulators are shape (1,) and (3,).
        for name, lead in (('nz (prb)', self.nz), ('ncode (code)', self.ncode),
                           ('the (1,) accumulator', 1),
                           ('the (3,) accumulator', 3)):
            if lead == self.local_ntheta:
                raise ValueError(
                    f"local_ntheta={self.local_ntheta} aliases {name}; "
                    f"change the rank count or the grid")

        # Code-window size. Needed before _pool_bytes_for, which charges the
        # pool for the per-projection windows; the operator itself is built
        # further down, once the chunking exists.
        frac = float(getattr(args, 'code_pad_frac', 0.125))
        if frac <= 0:
            raise ValueError(f'code_pad_frac must be > 0, got {frac}')
        self.code_pad_frac = frac
        self.code_pad = window_pad(self.n, self.nz, frac)

        nbytes = self._pool_bytes_for()

        # ---- geometry ------------------------------------------------------
        # sample -> detector
        wavelength = 1.24e-09 / self.energy
        z1 = float(np.atleast_1d(self.z1)[0])
        z2 = self.focustodetectordistance - z1
        magnification = self.focustodetectordistance / z1
        distance  = z1 * z2 / self.focustodetectordistance
        voxelsize = self.detector_pixelsize / magnification

        # code -> sample. The code sits upstream of the sample at z1_code; the
        # propagation distance is the one that survives the change of variables
        # from the point-source wave to a plane wave, scaled twice: once by the
        # normalized magnification of the code plane, once by the ratio of the
        # two source distances.
        z1_code = float(self.z1_code)
        z2c = z1 - z1_code
        magnification_code = z1 / z1_code
        distance_code = (z1_code * z2c) / (z1_code + z2c)
        norm_magnification_code = magnification_code / (z1_code / z1)
        distance_code = distance_code * norm_magnification_code ** 2
        distance_code = distance_code * (z1_code / z1) ** 2
        self.voxelsize = voxelsize
        self.distance, self.distance_code = distance, distance_code

        # ---- variable scalings ---------------------------------------------
        self.rho = list(args.rho)
        self.rho_sq = {'obj':      self.rho[0] ** 2,
                       'prb':      self.rho[1] ** 2,
                       'pos':      self.rho[2] ** 2,
                       'code':     self.rho[3] ** 2,
                       'pos_code': self.rho[4] ** 2}
        if not hasattr(self, 'estimate_rho'):
            self.estimate_rho = False
        if not hasattr(self, 'rho_estimate_niter'):
            self.rho_estimate_niter = 16

        # cuFFTDx JIT compile: rank 0 builds the .so, then all ranks proceed
        if self.rank == 0:
            cufftdx_precompile(2 * self.nz, 2 * self.n)
        self.cl_mpi.comm.Barrier()

        # sizes for normalization
        self.data_size = self.ntheta * self.nz * self.n
        self.prb_size  = self.nz * self.n
        self.obj_size  = self.nzobj * self.nobj ** 2
        self.code_size = self.ncode ** 2
        # normalization constant to address work with normal operators
        self.norm_const = np.float32(np.sqrt(self.nobj / self.ntheta))

        # ---- operators -------------------------------------------------
        self.cl_chunking = Chunking(nbytes, self.nchunk)
        self.cl_tomo = Tomo(self.nobj, self.nchunk, self.theta, self.mask,
                            nd=self.ndobj)
        self.cl_prop = Propagation(self.n, self.nz, self.nchunk, 1,
                                   wavelength, voxelsize, np.array([distance]))
        self.cl_prop_code = Propagation(self.n, self.nz, self.nchunk, 1,
                                        wavelength, voxelsize,
                                        np.array([distance_code]))

        # Object-plane shift. One distance means m == 1 everywhere, so 'fft'
        # never takes the chirp-z and is both faster and exact for a
        # band-limited input -- the B-spline stencil error otherwise biases the
        # position gradient. Same reasoning as step 0 in the multi-distance
        # pipeline, which defaults to 'fft' for the same reason.
        shift_type = getattr(args, 'shift_type', 'fft')
        self.shift_type = shift_type
        if shift_type == 'fft':
            self.cl_shift = ShiftFFT(self.n, self.ndobj, self.nz, self.nzobj,
                                     self.nchunk)
        elif shift_type == 'cubic':
            self.cl_shift = Shift(self.n, self.ndobj, self.nz, self.nzobj,
                                  self.nchunk)
        else:
            raise ValueError(f"shift_type must be 'cubic' or 'fft', got {shift_type!r}")

        # Code-plane shift. Always windowed and always FFT: the code is `ncode`
        # wide (typically 4n) but each projection reads only an [nz, n] region,
        # so transforming the whole plane wasted ~16x the work and dominated
        # the run. See shift_fft_code for the accuracy this trades for it.
        pad_y, pad_x = self.code_pad
        self.cl_shift_code = ShiftFFTCode(
            self.n, self.n + 2 * pad_x, self.nz, self.nz + 2 * pad_y,
            self.nchunk, pad=self.code_pad, code_shape=(self.ncode, self.ncode))
        if self.rank == 0:
            logger.info(
                f"code window: [{self.nz + 2*pad_y}, {self.n + 2*pad_x}] "
                f"(pad {pad_y}, {pad_x}) out of a [{self.ncode}, {self.ncode}] "
                f"plane, code_pad_frac={self.code_pad_frac:g}")

        # Frozen integer part of pos_code, set once in precalc and passed to
        # the cascade kernels as an ordinary theta-chunked input.
        self.code_off = None

        if self.lam_laplacian > 0:
            self.cl_lap_term = LaplacianTerm(
                self.lam_laplacian, self.obj_size, self.local_nzobj, self.nobj,
                self.cl_mpi, self.cl_chunking.gpu_batch,
                grad_pad=getattr(self, 'alloc_mode', 'full') != 'gen')

        # Detector PSF: ONE Gaussian on the detector intensity. Identical to
        # rec_mpi -- the whole F0 family below is his, unchanged.
        self.psf_sigma, self.psf_w = _psf_taps(getattr(args, 'psf_sigma', 0.0))
        if self.psf_w is not None and self.rank == 0:
            logger.info(
                f"PSF: one Gaussian on the detector intensity, "
                f"sigma={self.psf_sigma:g} binned det.px "
                f"({self.psf_w.size} taps)")

        # Data-misfit model, 'amplitude' (default) or 'intensity'.
        self.model = getattr(args, 'model', 'amplitude')
        if self.model not in ('intensity', 'amplitude'):
            raise ValueError(f"model must be 'intensity' or 'amplitude', "
                             f"got {self.model!r}")

        # Freeze flags: zero the corresponding gradient every iteration.
        self.freeze_code     = bool(getattr(args, 'freeze_code', False))
        self.freeze_pos_code = bool(getattr(args, 'freeze_pos_code', False))

        self.alloc_arrays()

        # fast refs
        self.gpu_batch = self.cl_chunking.gpu_batch
        self.redot_batch = self.cl_chunking.redot_batch
        self.linear_batch = self.cl_chunking.linear_batch
        self.linear_redot_batch = self.cl_chunking.linear_redot_batch
        self.mulc_batch = self.cl_chunking.mulc_batch
        self.redist = self.cl_mpi.redist
        self.allreduce  = self.cl_mpi.allreduce
        self.allreduce2 = self.cl_mpi.allreduce2
        self.allreduce_scalars = self.cl_mpi.allreduce_scalars

        # save convergence results
        self.table = pd.DataFrame(columns=["iter", "err", "time", "dobj2", "dobj_rel"])

        # apply_F_from memoization (per-chunk scope; reset in every cascade kernel)
        self._apply_F_cache  = {}
        self._apply_F_hits   = 0
        self._apply_F_misses = 0
        # read-only zero slabs for the d2F_dF fallback branches, keyed by shape
        self._zero_buf_cache = {}

    # ------------------------------------------------------------------
    # Memory sizing
    # ------------------------------------------------------------------

    def _pool_bytes_for(self):
        """Worst-case chunking-pool footprint across all gpu_batch callers.

        Candidates (proper-input/output buffers per single call, doubled for
        double-buffering at the end):

          cascade kernels (hessian_cascade3 is the widest):
            3 proj + 1 data + 3 pos + 3 pos_code + 1 code_off + 1 mask
          linear_batch on vars['obj']:     3 obj-shape
          gradient_laplacian (inp_pad=4):  3 obj-shape + 4 obj-slabs
          laplacian hessian3 (inp_pad=4):  2 obj-shape + 8 obj-slabs
          fwd_tomo / adj_tomo:             1 sinogram chunk + 1 obj-shape, where
                                           the sinogram chunk carries EVERY
                                           angle and so grows linearly in ndobj
                                           while the rest grow quadratically in
                                           nobj -- it dominates whenever
                                           ntheta > nobj.

        On top of the cascade candidate sit two groups of kernel-body
        temporaries, which are not pool allocations but come out of the same
        GPU and would otherwise be unaccounted:
          * F0's detector-plane float32 scratch (|x|^2, J, r, p and the
            Hessian's second nested pair) -- 7 of them, as rec_mpi measures.
          * the code windows. Each of c/c1/c2 is gathered into a fresh
            [nchunk, nzpsi, npsi] complex64 buffer and FFT'd, and the FFT
            allocates its own output, so charge 6 window-shaped slabs.
            At code_pad_frac = 1/8 a window is ~1.56x the detector plane, so
            this is the larger of the two groups.
        """
        obj_item   = np.dtype('complex64').itemsize
        obj_slab   = self.nobj * self.nobj * obj_item
        proj_bytes = self.nchunk * self.nzobj * self.ndobj * obj_item
        obj_bytes  = self.nchunk * obj_slab
        tomo_bytes = self.nchunk * self.ntheta * self.ndobj * obj_item + obj_bytes

        data_b = self.nchunk * self.nz * self.n * 4           # [nchunk,nz,n] f32
        pos_b  = self.nchunk * 2 * 4                          # [nchunk,2] f32/i32
        mask_b = self.nchunk * (self.nz + self.n) * 4
        cascade = 3 * proj_bytes + data_b + 7 * pos_b + mask_b

        # kernel-body temporaries
        pad_y, pad_x = self.code_pad
        win_bytes = self.nchunk * (self.nz + 2 * pad_y) * (self.n + 2 * pad_x) * obj_item
        cascade += 7 * self.nchunk * self.nz * self.n * 4     # F0 scratch
        cascade += 6 * win_bytes                              # code windows

        candidates = [cascade, 3 * obj_bytes, tomo_bytes]
        if self.lam_laplacian > 0:
            candidates.append(3 * obj_bytes + 4 * obj_slab)
            candidates.append(2 * obj_bytes + 8 * obj_slab)
        return int(2.1 * max(candidates))   # x2 double-buffering + 10% slack

    def alloc_arrays(self):
        """Allocate all pinned CPU and CuPy GPU buffers.

        `args.alloc_mode = 'gen'` skips the gradient and conjugate-direction
        buffers: generation only touches vars / data / proj_tmp.
        """
        gen = getattr(self, 'alloc_mode', 'full') == 'gen'
        obj_shape  = [self.local_nzobj, self.nobj, self.nobj]
        prb_shape  = [self.nz, self.n]
        code_shape = [self.ncode, self.ncode]
        pos_shape  = [self.local_ntheta, 2]
        proj_shape = [self.local_ntheta, self.nzobj, self.ndobj]

        # vars/etas/grads['obj'] alias the padded scratch owned by cl_lap_term
        # when the Laplacian is active, so a single chunked pass can
        # differentiate any of them; otherwise plain obj-shape buffers.
        if hasattr(self, 'cl_lap_term'):
            obj_buf   = self.cl_lap_term.obj_view
            etas_obj  = self.cl_lap_term.etas_view
            grads_obj = self.cl_lap_term.grads_view
        else:
            obj_buf = make_pinned(obj_shape, dtype='complex64'); obj_buf[:] = 0
            etas_obj  = None if gen else make_pinned(obj_shape, dtype='complex64')
            grads_obj = None if gen else make_pinned(obj_shape, dtype='complex64')
            for b in (etas_obj, grads_obj):
                if b is not None:
                    b[:] = 0

        # pos / pos_code / prb are pinned CPU, like rec_mpi. `code` is a
        # replicated input handed whole to every chunk and is far too big to
        # re-upload per chunk, so it lives on the GPU.
        self.vars = {
            'obj':      obj_buf,
            'pos':      make_pinned(pos_shape,  dtype='float32'),
            'prb':      make_pinned(prb_shape,  dtype='complex64'),
            'proj':     make_pinned(proj_shape, dtype='complex64'),
            'code':     cp.zeros(code_shape,    dtype='complex64'),
            'pos_code': make_pinned(pos_shape,  dtype='float32'),
        }
        self.vars['pos'][:] = 0
        self.vars['pos_code'][:] = 0
        self.vars['prb'][:] = 0
        self.vars['proj'][:] = 0

        # measurement data -- INTENSITY, not its square root.
        self.data = make_pinned([self.local_ntheta, self.nz, self.n], dtype='float32')

        # gradient and conjugate-direction buffers (reconstruction only)
        self.grads, self.etas = {}, {}
        if not gen:
            for ge in self.grads, self.etas:
                ge['pos']      = make_pinned(pos_shape,  dtype='float32')
                ge['pos_code'] = make_pinned(pos_shape,  dtype='float32')
                ge['proj']     = make_pinned(proj_shape, dtype='complex64')
                ge['prb']      = make_pinned(prb_shape,  dtype='complex64')
                # code mirrors prb: replicated, so its gradient is allreduced.
                ge['code']     = cp.zeros(code_shape, dtype='complex64')
            self.etas['obj']  = etas_obj
            self.grads['obj'] = grads_obj
            # etas must start at exactly zero, not at whatever the pinned pool
            # handed back: the first CG step is eta <- 0*eta - grad, and
            # 0*NaN is NaN.
            for k, v in self.etas.items():
                if k != 'obj':      # obj is zeroed at allocation above
                    v[:] = 0
            for k, v in self.grads.items():
                if k != 'obj':
                    v[:] = 0

        self.proj_tmp = make_pinned([self.ntheta, self.local_nzobj, self.ndobj],
                                    dtype='complex64')

        # Frozen integer code offsets, a theta-chunked cascade input like
        # self.data. Pinned for the same reason.
        self.code_off = make_pinned(pos_shape, dtype='int32')
        self.code_off[:] = 0

        # Out-of-grid detector mask, filled by _build_data_mask in precalc.
        # Axis-separable per angle: [:nz] rows, [nz:] columns, pixel weight =
        # their outer product (a dense mask would cost tens of GB). All-ones
        # here so paths that skip precalc keep the unmasked objective.
        self.mask_1d = make_pinned([self.local_ntheta, self.nz + self.n],
                                   dtype='float32')
        self.mask_1d[:] = 1
        # The same information in its compact authoritative form: the half-open
        # keep box (y0, y1, x0, x1) per angle. Diagnostics only.
        self.mask_box = np.empty((self.local_ntheta, 4), dtype='int32')
        self.mask_box[:, 0], self.mask_box[:, 1] = 0, self.nz
        self.mask_box[:, 2], self.mask_box[:, 3] = 0, self.n
        # Neutral until a cascade kernel points them at its own chunk, so a
        # path calling F0 outside the batched call sites sees an unmasked
        # objective rather than an AttributeError.
        self._mask_y = self._mask_x = cp.ones(1, dtype='float32')
        # Likewise for the per-chunk frozen offsets.
        self._off_chunk = None

    def _set_mask_chunk(self, mk):
        """Point the F0 family at one theta-chunk of self.mask_1d. mk is
        [chunk, nz + n]; the two factors broadcast against [chunk, nz, n]."""
        self._mask_y = mk[:, :self.nz, None]
        self._mask_x = mk[:, None, self.nz:]

    # ------------------------------------------------------------------
    # Out-of-grid detector mask
    # ------------------------------------------------------------------

    def _build_data_mask(self, vars):
        """Weight out the detector pixels whose object-grid footprint leaves
        the grid.

        F5 samples the projection plane at

            x = tx - (n-1)/2 - r_x + (ndobj-1)/2

        (no magnification here, unlike rec_mpi where eff_demag scales tx) and
        interpolates with a 4-tap kernel, so a pixel is fully supported only
        for 1 <= x < ndobj-2, and likewise in y against nzobj.  Outside that,
        what the model predicts is decided by the shift kernel's boundary
        condition rather than by the sample, so fitting those pixels only
        pushes the residual into the probe, the code and the in-grid object.
        Partial support counts as no support.

        At m = 1 a pixel can only leave the grid because the sample has moved,
        so with ndobj >= n + 2*max|pos| nothing is ever masked and this is a
        no-op -- the normal case for this geometry.  It earns its keep on a
        large-displacement scan.

        Stored as the two 1-D factors the kernels multiply (`mask_1d`) plus the
        four integers they came from (`mask_box`), never as a dense array.

        Frozen for the whole solve: rebuilding it mid-run would change F
        between iterations, so the errors in conv.csv would stop being
        comparable, and F0 would stop being differentiable in pos (every
        derivative ignores the mask's own dependence on it).  `pos` is taken at
        its initial value and `mask_oob_margin` is the slack that keeps the
        frozen box valid as it drifts.
        """
        margin = float(getattr(self, 'mask_oob_margin', 2.0))
        enabled = bool(getattr(self, 'mask_oob', True))
        ndobj = int(getattr(self, 'ndobj', self.nobj))

        if not enabled:
            self.mask_1d[:] = 1
            self.mask_box[:, 0], self.mask_box[:, 1] = 0, self.nz
            self.mask_box[:, 2], self.mask_box[:, 3] = 0, self.n
            if self.rank == 0:
                logger.warning("data mask: disabled (mask_oob=False)")
            return

        pos = np.asarray(vars['pos'], dtype='float64')        # [ntheta, (y, x)]
        ax = np.arange(self.n,  dtype='float64') - (self.n  - 1) * 0.5
        ay = np.arange(self.nz, dtype='float64') - (self.nz - 1) * 0.5

        # A tap set {floor(v)-1 .. floor(v)+2} fits in [0, N-1] iff
        # 1 <= v < N-2. Both ends carry the margin, so the box stays valid as
        # pos drifts either way.
        def _axis(idx, pos_a, N):
            v = idx[None, :] - pos_a[:, None] + (N - 1) * 0.5
            return (v >= 1.0 + margin) & (v < (N - 2) - margin)

        my = _axis(ay, pos[:, 0], self.nzobj)                 # [ntheta, nz]
        mx = _axis(ax, pos[:, 1], ndobj)                      # [ntheta, n]
        self.mask_1d[:, :self.nz] = my
        self.mask_1d[:, self.nz:] = mx
        for m, o in ((my, 0), (mx, 2)):
            cnt = m.sum(axis=1)
            self.mask_box[:, o]     = np.where(cnt > 0, m.argmax(axis=1), 0)
            self.mask_box[:, o + 1] = self.mask_box[:, o] + cnt

        # Kept fraction per angle, reduced over ranks for the log line.
        if self.local_ntheta > 0:
            loc = (self.mask_1d[:, :self.nz].mean(axis=1)
                   * self.mask_1d[:, self.nz:].mean(axis=1))   # [ntheta]
            stat = [float(loc.sum()), float(loc.min()), float(loc.max()),
                    float(self.local_ntheta), float(np.abs(pos).max())]
        else:
            # neutral elements, so a rank holding no angles does not drag the
            # min to 0 or the max to 1
            stat = [0.0, np.inf, -np.inf, 0.0, 0.0]
        comm = self.cl_mpi.comm
        tot = np.array(stat[0]); comm.Allreduce(MPI.IN_PLACE, tot, op=MPI.SUM)
        lo  = np.array(stat[1]); comm.Allreduce(MPI.IN_PLACE, lo,  op=MPI.MIN)
        hi  = np.array(stat[2]); comm.Allreduce(MPI.IN_PLACE, hi,  op=MPI.MAX)
        cnt = np.array(stat[3]); comm.Allreduce(MPI.IN_PLACE, cnt, op=MPI.SUM)
        rmx = np.array(stat[4]); comm.Allreduce(MPI.IN_PLACE, rmx, op=MPI.MAX)

        empty = int((self.mask_box[:, 1] <= self.mask_box[:, 0]).sum()
                    + (self.mask_box[:, 3] <= self.mask_box[:, 2]).sum())
        empty = comm.allreduce(empty, op=MPI.SUM)

        if self.rank == 0:
            mean = float(tot) / max(float(cnt), 1)
            logger.warning(
                f"data mask: |pos|<={float(rmx):.2f}px margin={margin:g}px -> "
                f"keeps {100 * mean:.1f}% of the pixels per angle "
                f"(min {100 * float(lo):.1f}%, max {100 * float(hi):.1f}%). "
                f"F0 still divides by the FULL ntheta*nz*n, so reported errors "
                f"scale with that fraction and are not comparable with an "
                f"unmasked run.")
            if mean < 0.999:
                logger.warning(
                    f"data mask: ndobj={ndobj} is too small for the sample's "
                    f"travel across it; the discarded pixels are real "
                    f"measurements that only a larger object grid can use.")
            if empty:
                logger.warning(
                    f"data mask: {empty} (angle, axis) boxes are EMPTY -- "
                    f"those projections contribute nothing to the data fit.")

    # ------------------------------------------------------------------
    # Code-plane offsets
    # ------------------------------------------------------------------

    def freeze_code_off(self, pos_code):
        """Pin the integer part of pos_code for the whole solve.

        NOT re-derived per evaluation: `pos_code` is an optimized variable, and
        re-rounding it each time would step the gather window by a pixel as a
        shift crosses a half-integer.  That makes the objective discontinuous,
        which breaks both the quadratic model the BH step is built on and the
        Hessian that scales it.
        """
        self.code_off[:] = np.round(np.asarray(pos_code)).astype('int32')

    def check_code_drift(self, vars):
        """`pos_code` keeps moving while the offsets stay frozen, so the
        residual the FFT interpolates grows past 0.5 px.  The pad absorbs it,
        but silently losing accuracy is worse than stopping.  One small
        reduction per outer iteration."""
        self.cl_shift_code.check_drift(vars['pos_code'], self.code_off)

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def BH(self, writer=None):
        vars, grads, etas = self.vars, self.grads, self.etas
        self.precalc(vars)
        self.error_debug(vars, -1)
        if self.estimate_rho:
            self.estimate_rho_coord(vars, grads, etas,
                                    niter_trial=self.rho_estimate_niter)
        self._iterate(vars, grads, etas, writer)
        self.postcalc(vars)
        return vars

    def _iterate(self, vars, grads, etas, writer=None):
        """Main BH iteration loop. Assumes precalc() has already run."""
        self.time_start = time.time()
        for i in range(self.start_iter, self.niter):
            with nvtx.annotate(f"::BH:{i}"):
                # The window offsets are frozen; this is what catches pos_code
                # drifting past what the pad can absorb.
                self.check_code_drift(vars)
                self.compute_gradient(vars, grads)
                alpha, top, bottom = self.compute_step(vars, grads, etas, i)
                if self.check_approx:
                    self.check_approximation(vars, etas, top, bottom, alpha,
                                             i, writer)
                want = self._is_error_iter(i)
                res = self.apply_step(vars, etas, alpha, obj_norms=want)
                if want:
                    self._record_dobj(res)
                self.log_iter(vars, i, writer)

        # Always checkpoint AT `niter`, independent of checkpoint_step. The
        # loop above is `range(start_iter, niter)`, so i never reaches niter
        # itself -- no choice of checkpoint_step can make vis_debug's
        # periodic `i % checkpoint_step == 0` write ever land exactly there
        # (e.g. niter=512, checkpoint_step=32 writes ...416,448,480, never
        # 512). The next rung's config sets its start_iter to this rung's
        # niter, and find_latest_checkpoint needs an EXACT
        # checkpoint_{start_iter:04}.h5 to resume from. Without this, that
        # lookup silently fails and step6.py falls through to a cold start
        # from the Paganin seed instead of resuming -- no error raised
        # anywhere, the wrong behaviour just runs quietly.
        if writer is not None and self.niter > self.start_iter:
            writer.write_checkpoint(vars, self.niter, self.norm_const,
                                    pos_init=self.pos_init,
                                    pos_code_init=self.pos_code_init)

    def precalc(self, vars):
        """One-time setup at the start of BH: the data mask, the frozen code
        offsets, obj normalization, pos snapshots, and the initial proj."""
        self._build_data_mask(vars)
        self.freeze_code_off(vars['pos_code'])

        # normalize obj to work with normal operators (restored at BH exit)
        vars["obj"] /= self.norm_const
        if self.start_iter == 0:
            vars["obj"] *= self.cl_tomo.mask

        self.pos_init      = np.array(vars['pos'],      dtype='float32')
        self.pos_code_init = np.array(vars['pos_code'], dtype='float32')

        self.fwd_tomo(vars["obj"], out=self.proj_tmp)
        self.redist(self.proj_tmp, vars['proj'])

    def postcalc(self, vars):
        """Restore obj normalization after the BH loop (inverse of precalc)."""
        vars["obj"] *= self.norm_const

    def compute_gradient(self, vars, grads):
        """gradients_cascade + propagate the obj-side gradient through
        fwd_tomo + redist so grads['proj'] is ready for the hessian calls."""
        with nvtx.annotate("gradients"):
            self.gradients(vars, grads)
        with nvtx.annotate(":::BH:fwd_tomo"):
            self.fwd_tomo(grads["obj"], out=self.proj_tmp)
        with nvtx.annotate(":::BH:redist", color='red'):
            self.redist(self.proj_tmp, grads['proj'])

    def compute_step(self, vars, grads, etas, i):
        """One CG step: update the search direction with the new beta and
        return (alpha, top, bottom)."""
        if self.fused_hessian:
            return self._compute_step_fused(vars, grads, etas, i)
        beta = self.compute_beta(vars, grads, etas, i)
        return self.compute_alpha(vars, grads, etas, beta)

    def _compute_step_fused(self, vars, grads, etas, i):
        """beta and alpha from ONE cascade sweep instead of three."""
        with nvtx.annotate(":::BH:calc_step"):
            check = getattr(self, 'check_fused_hessian', False)

            # Reduce BEFORE expanding `bottom`: the expansion folds B(g,e) and
            # B(e,g) into one 2*beta*Bge term, and for the regularization those
            # are equal only globally -- per rank they differ by a biharmonic
            # flux across the slab boundary.
            Qgg, Bge, Qee = self.allreduce_scalars(
                *self.hessian3(vars, grads, etas))

            # Steepest descent on the first iteration: eta_new = -g, so
            # bottom = B(-g,-g) = Qgg, which the expansion already gives at
            # beta = 0. etas is zero there, so the sweep measured
            # Bge = Qee = 0 and only the ratio needs the special case (0/0).
            #
            # beta is floored at 0 (the Powell safeguard): Bge < 0 means the
            # new gradient and the old direction have negative curvature
            # coupling, so beta*e points away from the descent direction, and
            # dropping it falls back to steepest descent for that iteration.
            beta = 0.0 if i == self.start_iter else max(Bge / Qee, 0.0)

            if check and i > self.start_iter:
                self._log_fused_check(i, "beta", beta,
                                      self._ref_beta(vars, grads, etas))

            # etas <- beta*etas - grads. Must run AFTER the sweep: it
            # overwrites the direction the sweep just read.
            top, = self.allreduce_scalars(self._update_etas(grads, etas, beta))

            bottom = beta * beta * Qee - 2.0 * beta * Bge + Qgg
            # A Schur complement (Qgg - Bge^2/Qee): positive quantities that
            # nearly cancel when beta is large. Warn on a non-positive result
            # (nonconvex) or on heavy cancellation.
            scale = abs(beta * beta * Qee) + abs(2.0 * beta * Bge) + abs(Qgg)
            nonconvex = not bottom > 1e-6 * scale

            if check:
                ref_bottom = self.allreduce_scalars(
                    self.hessian(vars, etas, etas))[0]
                self._log_fused_check(i, "bottom", bottom, ref_bottom)

            if nonconvex:
                # The quadratic model along eta is concave or flat, so its
                # stationary point is a MAXIMUM and top/bottom would walk
                # uphill. Under the intensity misfit the indefinite part of
                # d2F0 is p*Re(conj(z) y) with p = K[W(K|x|^2 - d)], which is
                # proportional to the residual and vanishes at the solution;
                # the other term is a square. top = <-grad, eta> > 0 for any
                # descent direction, so |bottom| keeps the step downhill and
                # of the right magnitude.
                logger.warning(
                    f"iter={i}: non-convex alpha denominator bottom={bottom:.6e} "
                    f"from Qgg={Qgg:.6e} Bge={Bge:.6e} Qee={Qee:.6e} "
                    f"beta={beta:.6e} -- stepping along |bottom| instead of "
                    f"taking alpha<0")
                bottom = abs(bottom) if bottom != 0.0 else 1e-6 * scale

            alpha = top / bottom
        return alpha, top, bottom

    def _ref_beta(self, vars, grads, etas):
        """beta the classic way -- two extra sweeps. Verification only."""
        t, b = self.allreduce_scalars(self.hessian(vars, grads, etas),
                                      self.hessian(vars, etas, etas))
        return t / b

    def _log_fused_check(self, i, name, got, ref):
        rel = abs(got - ref) / abs(ref) if ref != 0 else abs(got)
        logger.info(f"iter={i}: fused-check {name:>6}  fused={got:+.9e}  "
                    f"measured={ref:+.9e}  rel={rel:.3e}")

    def _update_etas(self, grads, etas, beta):
        """etas <- beta*etas - grads for every variable, returning the local
        alpha numerator -<eta_new, grad>/rho^2 over the theta-distributed
        variables (plus the replicated ones on rank 0)."""
        top = 0
        for v in ("obj", "pos", "pos_code"):
            if self.rho_sq[v] > 0:
                top -= self.linear_redot_batch(etas[v], grads[v], beta, -1) / self.rho_sq[v]
        # prb and code are replicated across ranks: the direction update must
        # run everywhere, but only rank 0 may add them to the sum that is
        # allreduced, or they would be counted comm.size times.
        dots = {v: self.linear_redot_batch(etas[v], grads[v], beta, -1)
                for v in ("prb", "code")}
        if self.rank == 0:
            for v, d in dots.items():
                if self.rho_sq[v] > 0:
                    top -= d / self.rho_sq[v]
        self.linear_batch(etas['proj'], grads['proj'], beta, -1)
        return top

    def compute_beta(self, vars, grads, etas, i):
        """CG coefficient beta = <grad, H.eta> / <eta, H.eta>.
        Returns 0 on the first iter (pure steepest descent)."""
        if i == self.start_iter:
            return 0
        with nvtx.annotate(":::BH:calc beta"):
            top    = self.hessian(vars, grads, etas)
            bottom = self.hessian(vars, etas, etas)
            top, bottom = self.allreduce2(top, bottom)
            return top / bottom

    def compute_alpha(self, vars, grads, etas, beta):
        """Update the search direction with the new beta, then the step size.
        Returns (alpha, top, bottom)."""
        with nvtx.annotate(":::BH:calc_alpha"):
            top = self._update_etas(grads, etas, beta)
            bottom = self.hessian(vars, etas, etas)
            top, bottom = self.allreduce2(top, bottom)
            if not bottom > 0.0:
                logger.warning(
                    f"non-convex alpha denominator bottom={bottom:.6e} "
                    f"-- stepping along |bottom| instead of taking alpha<0")
                bottom = abs(bottom) if bottom != 0.0 else 1.0
            alpha = top / bottom
        return alpha, top, bottom

    def apply_step(self, vars, etas, alpha, obj_norms=False):
        """var <- var + alpha*eta for every variable.

        With obj_norms, also returns (||obj^{n+1} - obj^n||^2, ||obj^{n+1}||^2)
        as LOCAL sums for the caller to allreduce.  The step is exactly
        alpha*etas['obj'], which is already resident, so no copy of the
        previous object is needed and the two reductions ride along inside the
        object's own update pass.
        """
        out = None
        for v in ("obj", "prb", "pos", "proj", "code", "pos_code"):
            if v == "obj" and obj_norms:
                out = self._apply_step_obj(vars, etas, alpha)
            else:
                self.linear_batch(vars[v], etas[v], 1, alpha)
        return out

    def _apply_step_obj(self, vars, etas, alpha):
        """obj <- obj + alpha*eta_obj, accumulating ||eta_obj||^2 and
        ||obj_new||^2 in the same streaming pass.

        Returns the LOCAL (alpha^2*||eta_obj||^2, ||obj_new||^2). float64
        accumulators: this is a sum of same-sign chunk norms over up to 3e10
        voxels and float32 would eat ~1e-4 of it. 0-d, not (1,), so gpu_batch
        can never mistake them for chunked outputs."""
        x, y = vars['obj'], etas['obj']
        a2 = float(alpha) ** 2
        if isinstance(x, cp.ndarray):
            _axpby(x, x, y, 1, alpha)
            return a2 * float(redot(y, y).get()), float(redot(x, x).get())

        d2 = cp.zeros((), dtype="float64")
        o2 = cp.zeros((), dtype="float64")

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=3)
        def _step_obj(self, out, d2, o2, x, y, alpha):
            _axpby(out, x, y, 1, alpha)
            d2[...] += redot(y, y)
            o2[...] += redot(out, out)

        _step_obj(self, x, d2, o2, x, y, alpha)
        return a2 * float(d2.get()[()]), float(o2.get()[()])

    def log_iter(self, vars, i, writer):
        """Error logging + visualization debug for this iter."""
        with nvtx.annotate(":::BH:calc error", color='gray'):
            self.error_debug(vars, i)
        with nvtx.annotate(":::BH:vis_debug", color='gray'):
            self.vis_debug(vars, i, writer)
        # Drained every iteration so the counts stay per-iteration rather than
        # cumulative; only logged on debug_step.
        ch, cm = self.cl_shift.coeff_cache_stats(reset=True)
        kh, km = self.cl_shift_code.coeff_cache_stats(reset=True)
        ah, am = self.apply_F_cache_stats(reset=True)
        if self.rank == 0 and self.debug_step != -1 and i % self.debug_step == 0:
            logger.info(f"iter={i}: coeff_cache      hits={ch} misses={cm}")
            logger.info(f"iter={i}: coeff_cache_code hits={kh} misses={km}")
            logger.info(f"iter={i}: apply_F_cache    hits={ah} misses={am}")

    # ------------------------------------------------------------------
    # Hessian and gradient sweeps
    # ------------------------------------------------------------------

    def hessian(self, vars, grads, etas):
        """Hessian of the full functional: the data-fit cascade plus the
        regularization.  Both addends are LOCAL; the caller allreduces once."""
        with nvtx.annotate("hessian"):
            w = self.hessian_cascade(vars, grads, etas)
            if hasattr(self, 'cl_lap_term'):
                w += self.cl_lap_term.hessian(grads["obj"])
        return w

    @timer
    def hessian_cascade(self, vars, grads, etas):
        """Cascade computation of the Hessian for the data-fit term, following
        the composition rule (Carlsson, 2025).  For f = F1 o F2,

            d2f = dF1 o d2F2 + d2F1 o dF2

        applied here to f = F0 o F1 o ... o F5.  `out` is a scalar cupy
        accumulator across every chunk."""

        out = cp.zeros(1, dtype="float32")
        # Identity check on the un-sliced arrays: a per-chunk slice would never
        # be `is`-equal even when the two directions are the same buffer.
        y_is_z = grads['prb'] is etas['prb']

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _hessian_cascade(self, out, d,
                             x_pos, y_pos, z_pos,
                             x_pc, y_pc, z_pc,
                             off, mask,
                             x_proj, y_proj, z_proj,
                             x_prb, y_prb, z_prb,
                             x_code, y_code, z_code):
            self.cl_shift.coeff_cache_reset()
            self.cl_shift_code.coeff_cache_reset()
            self.apply_F_cache_reset()
            self._off_chunk = off
            self._set_mask_chunk(mask)

            x = [x_prb, x_code, x_pc, x_proj, x_pos]
            y = [y_prb, y_code, y_pc, y_proj, y_pos]
            z = y if y_is_z else [z_prb, z_code, z_pc, z_proj, z_pos]
            w = [None, None, None, None, None]

            for id in range(1, len(self.F))[::-1]:
                # compute d2F(dFy, dFz) + dF(d2F(y, z))
                w = self.d2F_dF[id](x, y, z, w)
                # propagate the differentials one level down
                fx, y = self.dF[id](x, y)
                z = y if y_is_z else self.dF[id](x, z, return_x=False)
                x = fx
            out[:] += self.d2F_dF[0](x, y, z, w, d)

        _hessian_cascade(self, out, self.data,
                         vars['pos'],      grads['pos'],      etas['pos'],
                         vars['pos_code'], grads['pos_code'], etas['pos_code'],
                         self.code_off, self.mask_1d,
                         vars['proj'],     grads['proj'],     etas['proj'],
                         vars['prb'],      grads['prb'],      etas['prb'],
                         vars['code'],     grads['code'],     etas['code'])
        return out[0].get()

    def hessian3(self, vars, grads, etas):
        """{B(g,g), B(g,e), B(e,e)} -- the same terms hessian() sums, but as
        the three bilinear forms of (g, e), each contributed by a single pass
        over its own data.  Local, like hessian().

        The regularization term ignores its arguments: g and e are the
        interiors of the padded slabs it owns (LaplacianTerm.hessian3).  That
        is what lets the Laplacian ride along in the fused triple instead of
        needing its own sweeps."""
        with nvtx.annotate("hessian3"):
            hgg, hge, hee = self.hessian_cascade3(vars, grads, etas)
            if hasattr(self, 'cl_lap_term'):
                lgg, lge, lee = self.cl_lap_term.hessian3()
                hgg, hge, hee = hgg + lgg, hge + lge, hee + lee
        return hgg, hge, hee

    @timer
    def hessian_cascade3(self, vars, grads, etas):
        """The three bilinear forms {B(g,g), B(g,e), B(e,e)} of the data-fit
        term from ONE cascade sweep.

        Same composition rule as hessian_cascade, but the x-chain -- the part
        that streams proj and data over PCIe, i.e. essentially all of the cost
        -- is advanced once and shared by all three pairs instead of being
        recomputed per call."""

        out = cp.zeros(3, dtype="float32")

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _hessian_cascade3(self, out, d,
                              x_pos, g_pos, e_pos,
                              x_pc, g_pc, e_pc,
                              off, mask,
                              x_proj, g_proj, e_proj,
                              x_prb, g_prb, e_prb,
                              x_code, g_code, e_code):
            self.cl_shift.coeff_cache_reset()
            self.cl_shift_code.coeff_cache_reset()
            self.apply_F_cache_reset()
            self._off_chunk = off
            self._set_mask_chunk(mask)

            x = [x_prb, x_code, x_pc, x_proj, x_pos]
            g = [g_prb, g_code, g_pc, g_proj, g_pos]
            e = [e_prb, e_code, e_pc, e_proj, e_pos]
            # Passing the SAME list object for y and z on the diagonal forms
            # keeps the `y is z` fast paths in d2F_dF1 / d2F_dF2 alive, so the
            # flag becomes structural rather than a runtime comparison.
            wgg = [None, None, None, None, None]
            wge = [None, None, None, None, None]
            wee = [None, None, None, None, None]

            for id in range(1, len(self.F))[::-1]:
                # d2F_dF[id] must see the pre-update x, g, e -- hence all three
                # contractions before dF[id] advances the chains.
                wgg = self.d2F_dF[id](x, g, g, wgg)
                wge = self.d2F_dF[id](x, g, e, wge)
                wee = self.d2F_dF[id](x, e, e, wee)
                fx, gn = self.dF[id](x, g)
                en = self.dF[id](x, e, return_x=False)
                x, g, e = fx, gn, en

            out[0:1] += self.d2F_dF[0](x, g, g, wgg, d)
            out[1:2] += self.d2F_dF[0](x, g, e, wge, d)
            out[2:3] += self.d2F_dF[0](x, e, e, wee, d)

        _hessian_cascade3(self, out, self.data,
                          vars['pos'],      grads['pos'],      etas['pos'],
                          vars['pos_code'], grads['pos_code'], etas['pos_code'],
                          self.code_off, self.mask_1d,
                          vars['proj'],     grads['proj'],     etas['proj'],
                          vars['prb'],      grads['prb'],      etas['prb'],
                          vars['code'],     grads['code'],     etas['code'])
        h = out.get()
        return float(h[0]), float(h[1]), float(h[2])

    def gradients(self, vars, grads):
        """Full gradient: the data-fit cascade, the adjoint Radon that turns
        the proj-side gradient into an object-side one, and the regularizer."""
        self.gradients_cascade(vars, grads)

        with nvtx.annotate(":::BH:redist back", color='red'):
            self.redist(grads['proj'], self.proj_tmp, direction='backward')

        # parallelization over object slices: adjoint Radon
        self.adj_tomo(grads['obj'], self.proj_tmp)
        if hasattr(self, 'cl_lap_term'):
            # Added WITHOUT rho_sq['obj'], while the misfit part of
            # grads['obj'] carries it (see gradients_cascade). The two halves
            # are therefore in the same metric only at rho[obj] = 1, which
            # __init__ asserts whenever this term is active.
            self.cl_lap_term.gradient(grads['obj'])

        # prb and code are replicated: every rank computed the contribution of
        # its own angles, so both need summing across ranks.
        grads['prb'][:] = self.allreduce(grads['prb'])
        grads['code'][:] = cp.asarray(self.allreduce(cp.asnumpy(grads['code'])))

        if self.freeze_code:
            grads['code'][:] = 0
        if self.freeze_pos_code:
            grads['pos_code'][:] = 0

    @timer
    def gradients_cascade(self, vars, grads):
        """Cascade gradient for the data-fit term (Carlsson, 2025): apply the
        adjoint differentials in forward order, gF0 through gF5.

        rho_sq is applied HERE rather than in a separate pass over every
        variable, which is what rec_mpi does and saves a full streaming pass
        over obj and proj.  `_update_etas` divides it back out when it forms
        the alpha numerator.
        """
        # prb and code are non-proper outputs accumulated across theta chunks.
        grads['prb'][:]  = 0
        grads['code'][:] = 0

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=5)
        def _gradients_cascade(self, gradproj, gradpos, gradpos_code,
                               gradprb, gradcode,
                               d, pos, pos_code, off, mask, proj, prb, code):
            self.cl_shift.coeff_cache_reset()
            self.cl_shift_code.coeff_cache_reset()
            self.apply_F_cache_reset()
            self._off_chunk = off
            self._set_mask_chunk(mask)

            x = [prb, code, pos_code, proj, pos]
            y = d
            for id in range(len(self.gF)):
                y = self.gF[id](x, y)

            gradprb      += y[0] * self.rho_sq['prb']
            gradcode     += y[1] * self.rho_sq['code']
            gradpos_code[:] = y[2] * self.rho_sq['pos_code']
            # gF5 returns an un-prefiltered Deltapsi, so the B-spline coeff is
            # applied once here. It is the identity for the fft backend.
            gradproj[:]  = self.cl_shift.coeff(y[3]) * self.rho_sq['obj']
            gradpos[:]   = y[4] * self.rho_sq['pos']

        _gradients_cascade(self,
                           grads['proj'], grads['pos'], grads['pos_code'],
                           grads['prb'], grads['code'],
                           self.data, vars['pos'], vars['pos_code'],
                           self.code_off, self.mask_1d, vars['proj'],
                           vars['prb'], vars['code'])

    @timer
    def fwd_tomo(self, obj, out):
        """Forward tomography operator"""
        @self.gpu_batch(axis_out=1, axis_inp=0, nout=1)
        def _fwd_tomo(self, out, obj):
            out[:] = self.cl_tomo.R(obj)

        _fwd_tomo(self, out, obj)
        return out

    @timer
    def adj_tomo(self, gradu, gradproj):
        @self.gpu_batch(axis_out=0, axis_inp=1, nout=1)
        def _adj_tomo(self, gradu, gradproj):
            gradu[:] = self.cl_tomo.RT(gradproj)

        _adj_tomo(self, gradu, gradproj)

    ####################### Functions for the cascade
    # F*     - functional
    # dF*    - differential
    # d2F*   - second order term for the hessian
    # gF*    - gradient
    #######################################################################

    def apply_F_from(self, x, from_level):
        """Apply F[from_level], ..., F[len(F)-1] inside-out, returning the
        partial cascade value at level (from_level - 1).  Used by the gF*
        methods to fast-forward x through the levels above their own.

        Memoized by (id(x), from_level): within one pass gF0..gF5 share the
        same outer x, so without caching F5 would be re-evaluated six times.
        MUST be paired with apply_F_cache_reset() per chunk -- an id can be
        reused once an earlier object is collected.
        """
        if from_level >= len(self.F):
            return x
        key = (id(x), from_level)
        cached = self._apply_F_cache.get(key)
        if cached is not None:
            self._apply_F_hits += 1
            return cached
        self._apply_F_misses += 1
        upstream = self.apply_F_from(x, from_level + 1)
        result = self.F[from_level](upstream)
        self._apply_F_cache[key] = result
        return result

    def apply_F_cache_reset(self):
        self._apply_F_cache = {}

    def _zero_like_cached(self, a):
        """Read-only all-zeros buffer shaped like `a`.  Cached: the d2F_dF
        fallback branches would otherwise allocate a full plane on every call,
        three times per chunk in hessian_cascade3."""
        key = (a.shape, a.dtype)
        z = self._zero_buf_cache.get(key)
        if z is None:
            z = cp.zeros(a.shape, dtype=a.dtype)
            self._zero_buf_cache[key] = z
        return z

    def apply_F_cache_stats(self, reset=False):
        stats = (self._apply_F_hits, self._apply_F_misses)
        if reset:
            self._apply_F_hits = 0
            self._apply_F_misses = 0
        return stats

    ####### F0(x0) = 1/N \| m . g(K|x0|^2, d) \|^2
    #
    # THIS GROUP IS rec_mpi.Rec's, UNCHANGED.  It touches only the detector
    # plane x0, so it does not care how deep the cascade below it is; the one
    # cascade-dependent line is the apply_F_from(x, 1) in gF0_*, and that is
    # generic over len(self.F).  See the long block above F0 in rec_mpi.py for
    # the derivation of all eight functions, the two models, where the PSF and
    # the mask sit, and why the amplitude path needs a floor.  In brief, with
    # I = |x|^2, J = K I the blurred model intensity, a = sqrt(J), s = sqrt(d)
    # and W = my*mx the detector mask:
    #
    #   intensity   F0 = 1/N sum W (J - d)^2
    #   amplitude   F0 = 1/N sum W (a - s)^2
    #
    # and the two differ only in the pointwise weights g'/2 and g''/2:
    #
    #                g               g'/2                g''/2
    #   intensity    (J - d)^2       J - d               1
    #   amplitude    (sqrt J - s)^2  (a - s)/(2a)        s/(4 a^3)
    #
    # d is the measured INTENSITY in both; the amplitude path takes its own
    # sqrt.  W sits INSIDE the outer K, and the Hessian needs TWO nested
    # convolutions -- both are easy to get wrong and neither is a substitution
    # into a pointwise kernel.

    # Floor on the model amplitude a in the amplitude model's two
    # denominators.  Data are flat-field normalised (a ~ 1), so this is ~1e-3
    # of anything physical; it caps the curvature weight s/(4a^3) instead of
    # letting it diverge.
    _AMP_FLOOR = np.float32(1e-3)

    def _blur(self, x):
        """K x, the one Gaussian on the detector intensity, identity if unset.
        K^T = K (see psf.psf_blur), so this serves both directions."""
        return x if self.psf_w is None else _psf_blur(x, self.psf_w)

    def F0(self, x, d):
        """In: (x0), Out: const.  Dispatch on self.model."""
        return (self.F0_amp if self.model == 'amplitude' else self.F0_int)(x, d)

    def dF0(self, x, y, d, return_x=False):
        """In: (x0,y0), Out: const.  Dispatch on self.model."""
        f = self.dF0_amp if self.model == 'amplitude' else self.dF0_int
        return f(x, y, d)

    def d2F_dF0(self, x, y, z, w, d):
        """In: (x0,y0,z0,w0), Out: const.  Dispatch on self.model."""
        f = self.d2F_dF0_amp if self.model == 'amplitude' else self.d2F_dF0_int
        return f(x, y, z, w, d)

    def gF0(self, x, y):
        """In: x, y = data.  Out: y0.  Dispatch on self.model."""
        return (self.gF0_amp if self.model == 'amplitude' else self.gF0_int)(x, y)

    # ---- intensity model:  F0 = 1/N sum W (J - d)^2,  J = K|x|^2 ----------

    @nvtx.annotate("F0_int", color="green")
    def F0_int(self, x, d):
        """F0 = 1/N sum W (J - d)^2,   J = K|x|^2,  W = my*mx."""
        W = self._mask_y * self._mask_x
        J = self._blur(reprod(x, x))
        t = J - d
        return 1 / self.data_size * cp.sum(W * t * t)

    @nvtx.annotate("dF0_int", color="green")
    def dF0_int(self, x, y, d):
        """dF0[y] = 4/N sum p Re(conj(x) y),   p = K[W (J - d)]."""
        W = self._mask_y * self._mask_x
        J = self._blur(reprod(x, x))
        p = self._blur(W * (J - d))
        return 4 / self.data_size * redot(p * x, y)

    @nvtx.annotate("d2F0_dF0_int", color="purple")
    def d2F_dF0_int(self, x, y, z, w, d):
        """d2F0[y,z] = 4/N sum { 2 t Re(conj(x) y) + p Re(conj(z) y) }
                       + 4/N sum p Re(conj(x) w)        (the w slot, = dF0[w])

            p = K[W (J - d)]
            t = K[W K Re(conj(x) z)]     <- TWO nested blurs, the mask between
                                            them; the curvature weight is 1
        """
        W = self._mask_y * self._mask_x
        J = self._blur(reprod(x, x))
        p = self._blur(W * (J - d))
        t = self._blur(W * self._blur(reprod(x, z)))
        v = 2 * t * reprod(x, y) + p * reprod(z, y)
        if w is not None:
            v += p * reprod(x, w)
        return 4 / self.data_size * cp.sum(v)

    @nvtx.annotate("gF0_int", color="green")
    def gF0_int(self, x, y):
        """grad F0 = (4/N) p x,   p = K[W (J - d)].  y is the data d."""
        d = y
        x = self.apply_F_from(x, 1)
        W = self._mask_y * self._mask_x
        J = self._blur(reprod(x, x))
        p = self._blur(W * (J - d))
        return (np.float32(4 / self.data_size) * p) * x

    # ---- amplitude model:  F0 = 1/N sum W (a - sqrt(d))^2, a = sqrt(K|x|^2)
    # a is floored at _AMP_FLOOR wherever it appears in a DENOMINATOR (the two
    # derivatives); F0 itself is left exact.

    @nvtx.annotate("F0_amp", color="green")
    def F0_amp(self, x, d):
        """F0 = 1/N sum W (sqrt(J) - sqrt(d))^2,   J = K|x|^2,  W = my*mx."""
        W = self._mask_y * self._mask_x
        J = self._blur(reprod(x, x))
        t = cp.sqrt(J) - cp.sqrt(d)
        return 1 / self.data_size * cp.sum(W * t * t)

    @nvtx.annotate("dF0_amp", color="green")
    def dF0_amp(self, x, y, d):
        """dF0[y] = 4/N sum p Re(conj(x) y),   p = K[W (a - sqrt(d))/(2a)],
        a = max(sqrt(J), floor)."""
        W = self._mask_y * self._mask_x
        J = self._blur(reprod(x, x))
        a = cp.maximum(cp.sqrt(J), self._AMP_FLOOR)
        p = self._blur(W * 0.5 * (1.0 - cp.sqrt(d) / a))
        return 4 / self.data_size * redot(p * x, y)

    @nvtx.annotate("d2F0_dF0_amp", color="purple")
    def d2F_dF0_amp(self, x, y, z, w, d):
        """d2F0[y,z] = 4/N sum { 2 t Re(conj(x) y) + p Re(conj(z) y) }
                       + 4/N sum p Re(conj(x) w)        (the w slot, = dF0[w])

            p = K[W (a - sqrt(d))/(2a)]
            t = K[W (sqrt(d)/(4 a^3)) K Re(conj(x) z)]   <- TWO nested blurs;
                                                            the curvature weight
                                                            sits BETWEEN them
        """
        W = self._mask_y * self._mask_x
        J = self._blur(reprod(x, x))
        u = self._blur(reprod(x, z))
        a = cp.maximum(cp.sqrt(J), self._AMP_FLOOR)
        p = self._blur(W * 0.5 * (1.0 - cp.sqrt(d) / a))
        t = self._blur(W * (0.25 * cp.sqrt(d) / (a * a * a)) * u)
        v = 2 * t * reprod(x, y) + p * reprod(z, y)
        if w is not None:
            v += p * reprod(x, w)
        return 4 / self.data_size * cp.sum(v)

    @nvtx.annotate("gF0_amp", color="green")
    def gF0_amp(self, x, y):
        """grad F0 = (4/N) p x,   p = K[W (a - sqrt(d))/(2a)].  y is the data."""
        d = y
        x = self.apply_F_from(x, 1)
        W = self._mask_y * self._mask_x
        J = self._blur(reprod(x, x))
        a = cp.maximum(cp.sqrt(J), self._AMP_FLOOR)
        p = self._blur(W * 0.5 * (1.0 - cp.sqrt(d) / a))
        return (np.float32(4 / self.data_size) * p) * x

    ####### x0 = F1(x11, x12) = D(x11 . x12)
    #
    # x11 is the probe AT THE SAMPLE, the output of the code levels below, so
    # unlike rec_mpi it is per-projection rather than one shared [nz, n] frame.
    # That is the only difference in this group, and it shows up in gF1, where
    # the probe-side gradient is NOT summed over theta.

    @nvtx.annotate("F1", color="green")
    def F1(self, x):
        """In: (x11,x12), Out: x0."""
        x11, x12 = x
        return self.cl_prop.D(x11 * x12, 0)

    @nvtx.annotate("dF1", color="green")
    def dF1(self, x, y, return_x=True):
        """In: (x11,x12),(y11,y12) Out: y0."""
        x11, x12 = x
        y11, y12 = y
        y0 = self.cl_prop.D(y11 * x12 + x11 * y12, 0)
        if return_x:
            return self.cl_prop.D(x11 * x12, 0), y0
        return y0

    @nvtx.annotate("d2F_dF1", color="purple")
    def d2F_dF1(self, x, y, z, w):
        """In: (x11,x12),(y11,y12),(z11,z12),(w11,w12) Out: y0."""
        x11, x12 = x
        y11, y12 = y
        z11, z12 = z
        w11, w12 = w

        if y12 is z12:
            y0 = 2 * y11 * y12
        else:
            y0 = y11 * z12 + z11 * y12

        if w11 is not None:
            y0 += w11 * x12
        if w12 is not None:
            y0 += x11 * w12

        return self.cl_prop.D(y0, 0)

    @nvtx.annotate("gF1", color="green")
    def gF1(self, x, y):
        """In: x, (y0) Out: (y11, y12)."""
        y0 = y
        x11, x12 = self.apply_F_from(x, 2)
        y12 = self.cl_prop.DT(y0, 0)
        # No sum over theta here: x11 is the probe at the sample, which is
        # per-projection because the code shift below it is.
        y11 = y12 * cp.conj(x12)
        y12 = y12 * cp.conj(x11)
        return y11, y12

    ####### (x11, x12) = F2(x21, x22, x23) = (D_code(x21 . x22), x23)
    #
    # The probe at the code plane (x21, shared) times the shifted code (x22,
    # per-projection), propagated from the code plane to the sample.  x23 is
    # the sample transmission, carried through untouched.

    @nvtx.annotate("F2", color="green")
    def F2(self, x):
        """In: (x21,x22,x23), Out: (x11,x12)."""
        x21, x22, x23 = x
        return [self.cl_prop_code.D(x21 * x22, 0), x23]

    @nvtx.annotate("dF2", color="green")
    def dF2(self, x, y, return_x=True):
        """In: (x21,x22,x23),(y21,y22,y23) Out: (y11,y12)."""
        x21, x22, x23 = x
        y21, y22, y23 = y
        y11 = self.cl_prop_code.D(y21 * x22 + x21 * y22, 0)
        y12 = y23
        if return_x:
            return ([self.cl_prop_code.D(x21 * x22, 0), x23], [y11, y12])
        return [y11, y12]

    @nvtx.annotate("d2F_dF2", color="purple")
    def d2F_dF2(self, x, y, z, w):
        """In: (x21,x22,x23),(y...),(z...),(w...) Out: (y11,y12)."""
        x21, x22, x23 = x
        y21, y22, y23 = y
        z21, z22, z23 = z
        w21, w22, w23 = w

        if y22 is z22:
            y1 = 2 * y21 * y22
        else:
            y1 = y21 * z22 + z21 * y22

        if w21 is not None:
            y1 += w21 * x22
        if w22 is not None:
            y1 += x21 * w22

        return [self.cl_prop_code.D(y1, 0), w23]

    @nvtx.annotate("gF2", color="green")
    def gF2(self, x, y):
        """In: x, (y11,y12) Out: (y21,y22,y23)."""
        y11, y12 = y
        x21, x22, x23 = self.apply_F_from(x, 3)
        y22 = self.cl_prop_code.DT(y11, 0)
        # x21 (the probe at the code plane) IS shared across projections, so
        # its gradient sums over this chunk's angles; the caller then sums over
        # chunks and ranks.
        y21 = cp.sum(y22 * cp.conj(x22), axis=0)
        y22 = y22 * cp.conj(x21)
        return [y21, y22, y12]

    ####### (x21,x22,x23) = F3(x31,x32,x33,x34) = (x31, S_{x33}(x32), x34)
    #
    # The coded aperture.  x32 is the GLOBAL [ncode, ncode] plane and x33 the
    # per-projection shift; cl_shift_code gathers each projection's window out
    # of the plane rather than transforming the whole of it, with the integer
    # part of the shift frozen in self._off_chunk.  See shift_fft_code.

    def _m_code(self, r):
        """Unit magnification for this chunk's angles.  The code plane is not
        magnified relative to itself; the cached buffer short-circuits the
        operator's device-side is-unit test."""
        return self.cl_shift_code.unit_mag(r.shape[0])

    def _m_obj(self, r):
        """Unit magnification for the object-plane shift: one distance means
        no magnification, which is the whole reason the *mc shift family of
        rec_mpi is not needed here."""
        return self.cl_shift.unit_mag(r.shape[0])

    @nvtx.annotate("F3", color="green")
    def F3(self, x):
        """In: (x31,x32,x33,x34)  Out: (x21,x22,x23)."""
        x31, x32, x33, x34 = x
        off = self._off_chunk
        c = self.cl_shift_code.coeff_cached(x32)
        x22 = self.cl_shift_code.curlySc(c, x33, self._m_code(x33), off)
        return [x31, x22, x34]

    @nvtx.annotate("dF3", color="green")
    def dF3(self, x, y, return_x=True):
        """In: (x31..x34),(y31..y34)  Out: (y21,y22,y23)."""
        x31, x32, x33, x34 = x
        y31, y32, y33, y34 = y
        off = self._off_chunk
        m = self._m_code(x33)
        c  = self.cl_shift_code.coeff_cached(x32)
        c1 = self.cl_shift_code.coeff_cached(y32)
        y22 = self.cl_shift_code.dcurlySc(c, x33, m, c1, y33, off)
        if return_x:
            x22 = self.cl_shift_code.curlySc(c, x33, m, off)
            return [x31, x22, x34], [y31, y22, y34]
        return [y31, y22, y34]

    @nvtx.annotate("d2F_dF3", color="purple")
    def d2F_dF3(self, x, y, z, w):
        """In: (x31..x34),(y31..y34),(z31..z34),(w31..w34) Out: (w31,y22,w34).

        S is linear in the coefficient field, so the mixed second differential
        needs each coefficient against the OTHER direction's shift.  The
        kernel contracts each coefficient with the shift in its own slot, so
        the crossing happens HERE -- cz against y, cy against z.  Same
        convention as Rec.d2F_dF3; getting it backwards is invisible to a
        diagonal Taylor test and only shows up off the diagonal.
        """
        x31, x32, x33, x34 = x
        y31, y32, y33, y34 = y
        z31, z32, z33, z34 = z
        w31, w32, w33, w34 = w

        off = self._off_chunk
        m = self._m_code(x33)
        c  = self.cl_shift_code.coeff_cached(x32)
        cy = self.cl_shift_code.coeff_cached(y32)
        cz = self.cl_shift_code.coeff_cached(z32)
        y22 = self.cl_shift_code.d2curlySc(c, x33, m, cz, y33, cy, z33, off)

        # The level above (F4, the exponential) produces no correction in the
        # code or pos_code slots, so in this cascade both are always None and
        # this branch never fires. Kept for the same reason rec_mpi keeps it:
        # a level inserted above would silently lose the term otherwise.
        if w32 is not None or w33 is not None:
            cw = (self.cl_shift_code.coeff_cached(w32) if w32 is not None
                  else self._zero_like_cached(c))
            w33u = w33 if w33 is not None else self._zero_like_cached(x33)
            y22 += self.cl_shift_code.dcurlySc(c, x33, m, cw, w33u, off)

        return [w31, y22, w34]

    @nvtx.annotate("gF3", color="green")
    def gF3(self, x, y):
        """In: x, (y21,y22,y23)  Out: (y31,y32,y33,y34)."""
        y21, y22, y23 = y
        x31, x32, x33, x34 = self.apply_F_from(x, 4)
        off = self._off_chunk
        c = self.cl_shift_code.coeff_cached(x32)
        Deltapsi, y33 = self.cl_shift_code.dcurlySadjc(
            c, x33, self._m_code(x33), y22, off)
        # Deltapsi is on the WINDOW grid; fold each window back into the global
        # plane at its own offset. Windows overlap, so this accumulates -- it
        # is the adjoint of the gather. coeff is the identity for the fft
        # backend the code plane always uses, so there is no prefilter here.
        y32 = self.cl_shift_code.scatter_windows(Deltapsi, off)
        return [y21, y32, y33, y23]

    ####### (x31,x32,x33,x34) = F4(x41,x42,x43,x44) = (x41,x42,x43, e^{i x44})

    @staticmethod
    @cp.fuse()
    def _F4_fused(x44):
        return cp.exp(1j * x44)

    @nvtx.annotate("F4", color="green")
    def F4(self, x):
        """In: (x41..x44) Out: (x31..x34)."""
        x41, x42, x43, x44 = x
        return [x41, x42, x43, self._F4_fused(x44)]

    @staticmethod
    @cp.fuse()
    def _dF4_fused(x44, y44):
        x34 = cp.exp(1j * x44)
        return x34, x34 * 1j * y44

    @nvtx.annotate("dF4", color="green")
    def dF4(self, x, y, return_x=True):
        """In: (x41..x44),(y41..y44) Out: (x31..x34),(y31..y34)."""
        x41, x42, x43, x44 = x
        y41, y42, y43, y44 = y
        x34, y34 = self._dF4_fused(x44, y44)
        return ([x41, x42, x43, x34], [y41, y42, y43, y34]) if return_x \
            else [y41, y42, y43, y34]

    @staticmethod
    @cp.fuse()
    def _d2F_dF4_fused(x44, y44, z44, w44):
        y34 = cp.exp(1j * x44) * (-y44 * z44)
        if w44 is not None:
            y34 = y34 + cp.exp(1j * x44) * 1j * w44
        return y34

    @nvtx.annotate("d2F_dF4", color="purple")
    def d2F_dF4(self, x, y, z, w):
        """In: (x41..x44),(y...),(z...),(w...) Out: (w41,w42,w43,y34)."""
        x41, x42, x43, x44 = x
        y41, y42, y43, y44 = y
        z41, z42, z43, z44 = z
        w41, w42, w43, w44 = w
        y34 = self._d2F_dF4_fused(x44, y44, z44, w44)
        return [w41, w42, w43, y34]

    @staticmethod
    @cp.fuse()
    def _gF4_fused(x44, y34):
        return (-1j) * y34 * cp.conj(cp.exp(1j * x44))

    @nvtx.annotate("gF4", color="green")
    def gF4(self, x, y):
        """In: x, (y31..y34) Out: (y31,y32,y33,y44)."""
        y31, y32, y33, y34 = y
        x41, x42, x43, x44 = self.apply_F_from(x, 5)
        return [y31, y32, y33, self._gF4_fused(x44, y34)]

    ####### (x41..x44) = F5(x51..x55) = (x51,x52,x53, S_{x55}(x54))
    #
    # The object-plane shift: rec_mpi's F3 with the magnification dropped.
    # x54 is proj (this rank's projections) and x55 is pos.

    @nvtx.annotate("F5", color="green")
    def F5(self, x):
        """In: (x51..x55)  Out: (x41..x44)."""
        x51, x52, x53, x54, x55 = x
        c = self.cl_shift.coeff_cached(x54)
        return [x51, x52, x53,
                self.cl_shift.curlySc(c, x55, self._m_obj(x55))]

    @nvtx.annotate("dF5", color="green")
    def dF5(self, x, y, return_x=True):
        """In: (x51..x55),(y51..y55)  Out: (y41..y44)."""
        x51, x52, x53, x54, x55 = x
        y51, y52, y53, y54, y55 = y
        m = self._m_obj(x55)
        c  = self.cl_shift.coeff_cached(x54)
        c1 = self.cl_shift.coeff_cached(y54)
        y44 = self.cl_shift.dcurlySc(c, x55, m, c1, y55)
        if return_x:
            x44 = self.cl_shift.curlySc(c, x55, m)
            return [x51, x52, x53, x44], [y51, y52, y53, y44]
        return [y51, y52, y53, y44]

    @nvtx.annotate("d2F_dF5", color="purple")
    def d2F_dF5(self, x, y, z, w):
        """In: (x51..x55),(y...),(z...),(w...)  Out: (w51,w52,w53,y44).

        Coefficients CROSSED at the call site, as in d2F_dF3 -- see the note
        there and the SLOT PAIRING note on ShiftFFT.d2curlySc.
        """
        x51, x52, x53, x54, x55 = x
        y51, y52, y53, y54, y55 = y
        z51, z52, z53, z54, z55 = z
        w51, w52, w53, w54, w55 = w

        m = self._m_obj(x55)
        c  = self.cl_shift.coeff_cached(x54)
        cy = self.cl_shift.coeff_cached(y54)
        cz = self.cl_shift.coeff_cached(z54)
        y44 = self.cl_shift.d2curlySc(c, x55, m, cz, y55, cy, z55)

        # F5 is the outermost level, so w is always all-None here; kept for
        # the same reason as the matching branch in d2F_dF3.
        if w54 is not None or w55 is not None:
            cw = (self.cl_shift.coeff_cached(w54) if w54 is not None
                  else self._zero_like_cached(c))
            w55u = w55 if w55 is not None else self._zero_like_cached(x55)
            y44 += self.cl_shift.dcurlySc(c, x55, m, cw, w55u)

        return [w51, w52, w53, y44]

    @nvtx.annotate("gF5", color="green")
    def gF5(self, x, y):
        """In: x, (y41..y44)  Out: (y51..y55).

        Returns an un-prefiltered Deltapsi in y54; gradients_cascade applies
        cl_shift.coeff once (the identity for the fft backend).
        """
        y41, y42, y43, y44 = y
        # F5 is outermost: apply_F_from(x, 6) is a no-op, so x is already the
        # raw variable tuple.
        x51, x52, x53, x54, x55 = x
        c = self.cl_shift.coeff_cached(x54)
        y54, y55 = self.cl_shift.dcurlySadjc(c, x55, self._m_obj(x55), y44)
        return [y41, y42, y43, y54, y55]

    # ------------------------------------------------------------------
    # Objective, diagnostics, data generation
    # ------------------------------------------------------------------

    @timer
    def min(self, prb, obj, pos, proj, code, pos_code):
        """Value of the functional.

        `obj` is accepted to keep the signature parallel to rec_mpi.Rec.min
        and to make the call sites read as a full variable tuple; the data-fit
        term reaches the object through `proj`.  NOTE that the regularization
        term reads vars['obj'] directly (LaplacianTerm.energy_local takes no
        argument), so at lam_laplacian > 0 a call on TRIAL variables gets the
        trial data-fit but the current regularization energy.  Only
        check_approximation does that, and only for its plot.
        """
        out = cp.zeros(1, dtype="float32")

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _min(self, out, c_proj, c_pos, c_data, c_pc, c_off, c_mask,
                 c_prb, c_code):
            self.cl_shift.coeff_cache_reset()
            self.cl_shift_code.coeff_cache_reset()
            self.apply_F_cache_reset()
            self._off_chunk = c_off
            self._set_mask_chunk(c_mask)
            x = [c_prb, c_code, c_pc, c_proj, c_pos]
            out[:] += self.F0(self.apply_F_from(x, 1), c_data)

        _min(self, out, proj, pos, self.data, pos_code, self.code_off,
             self.mask_1d, prb, code)

        out = out[0]
        if hasattr(self, 'cl_lap_term'):
            out += self.cl_lap_term.energy_local()
        return self.allreduce(np.array(out.get(), dtype='float32'))

    def vis_debug(self, vars, i, writer=None):
        """Per-iter checkpoint write (position-error plots bundled in)."""
        if writer is None or not (i % self.checkpoint_step == 0
                                  and self.checkpoint_step != -1):
            return
        # On a resumed run, i == start_iter would overwrite the checkpoint we
        # just read from; on a fresh run (start_iter == 0) iter 0 is a genuine
        # new state and is saved as checkpoint_0000.
        if i < self.start_iter or (i == self.start_iter and self.start_iter > 0):
            return
        writer.write_checkpoint(vars, i, self.norm_const,
                                pos_init=self.pos_init,
                                pos_code_init=self.pos_code_init)

    def check_approximation(self, vars, etas, top, bottom, alpha, i, writer=None):
        """Compare the real functional along the descent direction with the
        quadratic model that picked alpha:
            f_real(t)   = min(vars + t*etas)
            f_approx(t) = min(vars) - top*t + 0.5*bottom*t**2
        Sampled at npp points in [0, 2*alpha].  With a writer, saves a PNG per
        triggered iteration under writer.path_out/check_approximation/.
        """
        if not (i % self.checkpoint_step == 0 and self.checkpoint_step != -1):
            return

        # lazy-allocate scratch (only on the first triggered iter)
        if not hasattr(self, '_chk_objt'):
            self._chk_objt  = make_pinned(vars['obj'].shape,  'complex64')
            self._chk_projt = make_pinned(vars['proj'].shape, 'complex64')
            self._chk_prbt  = make_pinned(vars['prb'].shape,  'complex64')
            self._chk_post  = make_pinned(vars['pos'].shape,  'float32')
            self._chk_pct   = make_pinned(vars['pos_code'].shape, 'float32')
            self._chk_codet = cp.empty_like(vars['code'])

        objt, prbt, post = self._chk_objt, self._chk_prbt, self._chk_post
        projt, pct, codet = self._chk_projt, self._chk_pct, self._chk_codet

        npp = 5
        t = np.linspace(0, 2 * alpha, npp).astype('float32')
        err_real = np.zeros(npp, dtype='float32')

        for k in range(npp):
            self.linear_batch(vars['obj'],      etas['obj'],      1, t[k], out=objt)
            self.linear_batch(vars['prb'],      etas['prb'],      1, t[k], out=prbt)
            self.linear_batch(vars['pos'],      etas['pos'],      1, t[k], out=post)
            self.linear_batch(vars['proj'],     etas['proj'],     1, t[k], out=projt)
            self.linear_batch(vars['code'],     etas['code'],     1, t[k], out=codet)
            self.linear_batch(vars['pos_code'], etas['pos_code'], 1, t[k], out=pct)
            err_real[k] = self.min(prbt, objt, post, projt, codet, pct)

        f0 = self.min(vars['prb'], vars['obj'], vars['pos'], vars['proj'],
                      vars['code'], vars['pos_code'])
        err_approx = f0 - top * t + 0.5 * bottom * t * t

        if self.rank != 0:
            return

        if writer is None:
            mshow_approx(t, err_real, err_approx, True)
            return

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(t, err_real,   "o-", label="real")
        ax.plot(t, err_approx, "x-", label="approx")
        ax.set_xlabel("t")
        ax.set_ylabel("functional")
        ax.set_title(f"iter {i}: alpha={alpha:.3e}")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        out_dir = os.path.join(writer.path_out, "check_approximation")
        os.makedirs(out_dir, exist_ok=True)
        png_path = os.path.join(out_dir, f"check_approx_{i:04}.png")
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        logger.info(f"check_approximation plot -> {png_path}")

    def _is_error_iter(self, i):
        """True on the iterations error_debug logs.  _iterate asks this before
        apply_step so the convergence norms are only computed when they will
        actually be written."""
        return self.error_step != -1 and i % self.error_step == 0

    def _record_dobj(self, res):
        """Turn apply_step's local (norm2 of the step, norm2 of obj) into the
        numbers logged.  obj is z-distributed, so both sums are partial and
        need one allreduce.  Inside BH obj carries a 1/norm_const, so the
        absolute number is scaled back to physical object units; the relative
        one is scale-free either way."""
        d2, o2 = self.allreduce_scalars(*res)
        self.dobj2 = float(self.norm_const) ** 2 * d2
        self.dobj_rel = float(np.sqrt(d2 / o2)) if o2 > 0 else float('nan')

    def error_debug(self, vars, i):
        """Error logging and CSV export.  i = -1 is the initial-state call
        from BH and is always logged regardless of error_step."""
        if i != -1 and not self._is_error_iter(i):
            return

        t_min = time.time()
        err = self.min(vars["prb"], vars["obj"], vars["pos"], vars["proj"],
                       vars["code"], vars["pos_code"])
        min_time = time.time() - t_min
        if self.rank == 0:
            if i == -1:
                logger.warning(f"Initial {err=:1.5e} ")
                # No step taken yet, so there is no ||obj^{n+1}-obj^n||.
                self.table.loc[len(self.table)] = [i, err, 0, np.nan, np.nan]
            else:
                ittime = time.time() - self.time_start
                dobj2 = getattr(self, 'dobj2', np.nan)
                dobj_rel = getattr(self, 'dobj_rel', np.nan)
                logger.warning(
                    f"iter={i}: {ittime:.4f}sec "
                    f"(-min={ittime - min_time:.4f}sec) {err=:1.5e} "
                    f"|dobj|^2={dobj2:1.5e} rel={dobj_rel:1.3e}")
                self.table.loc[len(self.table)] = [i, err, ittime, dobj2, dobj_rel]
            self.dobj2 = self.dobj_rel = np.nan
            self.time_start = time.time()
            if hasattr(self, 'path_out'):
                name = f"{self.path_out}/conv.csv"
                os.makedirs(os.path.dirname(name), exist_ok=True)
                self.table.to_csv(name, index=False)
                # ... and a per-level copy. The levels of a hierarchical run
                # share path_out, so conv.csv only ever holds the last level
                # that ran; conv_bin{bin}.csv keeps every level's history.
                if hasattr(self, 'bin'):
                    self.table.to_csv(f"{self.path_out}/conv_bin{self.bin}.csv",
                                      index=False)

    def gen_data(self, vars, out):
        """Generate synthetic data.

        Writes INTENSITY, K|psi|^2 -- the same convention the reader uses and
        the same quantity F0 compares against, blur included.
        """
        self.freeze_code_off(vars['pos_code'])
        vars["obj"] /= self.norm_const
        self.fwd_tomo(vars["obj"], out=self.proj_tmp)
        self.redist(self.proj_tmp, vars['proj'])

        @self.gpu_batch(axis_out=0, axis_inp=0, nout=1)
        def _gen_data(self, o, c_proj, c_pos, c_pc, c_off, c_prb, c_code):
            self.cl_shift.coeff_cache_reset()
            self.cl_shift_code.coeff_cache_reset()
            self.apply_F_cache_reset()
            self._off_chunk = c_off
            x = [c_prb, c_code, c_pc, c_proj, c_pos]
            y = self.apply_F_from(x, 1)
            # K|y|^2, not |y|^2: synthetic data has to carry the same blur the
            # model applies, or every test against it is vacuous.
            o[:] = self._blur(reprod(y, y))

        _gen_data(self, out, vars['proj'], vars['pos'], vars['pos_code'],
                  self.code_off, vars['prb'], vars['code'])

        vars["obj"] *= self.norm_const

    # ------------------------------------------------------------------
    # rho tuning
    # ------------------------------------------------------------------

    def estimate_rho_coord(self, vars, grads, etas, niter_trial=16, max_extend=8):
        """Coordinate search on rho[prb, pos, code, pos_code] over a geometric
        grid {..., init/2, init, 2*init, ...} centred on the current rho_sq.

        rec_mpi searches [prb, pos, tp]; this searches the two code variables
        in tp's place.  rho[obj] is the reference scale and is never moved --
        here it is also pinned to 1 by __init__ whenever the Laplacian is on.

        For each variable in order: run three short BH trials at
        {init/2, init, 2*init}; if the middle wins keep it, else extend up or
        down until improvement stops, capped by max_extend rungs.  Each trial
        restores vars/grads/etas/table/start_iter to the snapshot taken here,
        runs `_iterate` silently, and scores with self.min().  A trial that
        blows up scores inf, so the search steps past divergent rho.

        Trial errors are memoised on the full rho vector, so the centre probe
        of each coordinate after the first -- the vector that just won -- is
        free.
        """
        snap_vars       = {k: v.copy() for k, v in vars.items()}
        snap_table      = self.table.copy()
        snap_start_iter = self.start_iter
        snap_niter      = self.niter
        snap_error_step = self.error_step
        snap_ckpt_step  = self.checkpoint_step

        self.niter           = niter_trial
        self.start_iter      = 0
        # Trials are silent by default; rho_trial_error_step=N logs the error
        # every N iterations inside each trial, which is the only way to tell
        # a slow descent from a first-step blow-up behind an inf score.
        self.error_step      = int(getattr(self, 'rho_trial_error_step', -1))
        self.checkpoint_step = -1

        order = ('obj', 'prb', 'pos', 'code', 'pos_code')

        def _reset_trial():
            for k, v in vars.items():
                v[:] = snap_vars[k]
            for buf in grads.values(): buf[:] = 0
            for buf in etas.values():  buf[:] = 0
            self.table = pd.DataFrame(
                columns=["iter", "err", "time", "dobj2", "dobj_rel"])
            self.start_iter = 0

        def _run_trial(rho_vec):
            _reset_trial()
            self.rho_sq = {k: rho_vec[j] ** 2 for j, k in enumerate(order)}
            try:
                self._iterate(vars, grads, etas, writer=None)
                err = float(self.min(vars['prb'], vars['obj'], vars['pos'],
                                     vars['proj'], vars['code'],
                                     vars['pos_code']))
                if not np.isfinite(err):
                    err = float('inf')
            except Exception as e:
                if self.rank == 0:
                    logger.warning(f'rho trial {rho_vec} crashed '
                                   f'({type(e).__name__}: {e}) -> err=inf')
                err = float('inf')
            return err

        # Errors keyed by the full rho vector, shared across coordinates: the
        # winner of one coordinate is exactly the centre probe of the next.
        trial_cache = {}

        def _coord(base, idx, name, init):
            seen = {}

            def probe(val):
                rv = list(base); rv[idx] = val
                key = tuple(rv)
                if key in trial_cache:
                    e = trial_cache[key]
                    seen.setdefault(val, e)
                    if self.rank == 0:
                        logger.warning(f'  {name}={val:g}  err={e:.4e} (cached)')
                    return e
                e = _run_trial(rv)
                trial_cache[key] = e
                seen[val] = e
                if self.rank == 0:
                    logger.warning(f'  {name}={val:g}  err={e:.4e}')
                return e

            e_c  = probe(init)
            e_up = probe(init * 2)
            e_dn = probe(init / 2)
            if e_c <= e_up and e_c <= e_dn:
                best = init
            elif e_up < e_dn:
                cur_v, cur_e = init * 2, e_up
                for _ in range(max_extend):
                    nxt = cur_v * 2
                    e_nxt = probe(nxt)
                    if e_nxt >= cur_e: break
                    cur_v, cur_e = nxt, e_nxt
                best = cur_v
            else:
                cur_v, cur_e = init / 2, e_dn
                for _ in range(max_extend):
                    nxt = cur_v / 2
                    e_nxt = probe(nxt)
                    if e_nxt >= cur_e: break
                    cur_v, cur_e = nxt, e_nxt
                best = cur_v
            if self.rank == 0:
                logger.warning(f'  -> best {name}={best:g}')
            return best, sorted(seen.items())

        base = [float(np.sqrt(self.rho_sq[k])) for k in order]
        if self.rank == 0:
            logger.warning(f'estimate_rho_coord: start from {base}, '
                           f'niter_trial={niter_trial}')

        history = {}
        for idx in range(1, len(order)):          # obj stays put
            name = order[idx]
            if base[idx] <= 0:
                history[name] = []                # frozen: nothing to scale
                continue
            base[idx], history[name] = _coord(base, idx, name, base[idx])

        # Restore state so the outer BH loop starts clean.
        _reset_trial()
        self.table           = snap_table
        self.start_iter      = snap_start_iter
        self.niter           = snap_niter
        self.error_step      = snap_error_step
        self.checkpoint_step = snap_ckpt_step

        self.rho_sq = {k: base[j] ** 2 for j, k in enumerate(order)}
        self.rho    = list(base)
        if self.rank == 0:
            logger.warning(f'estimate_rho_coord: final rho = {base}')
        return history
