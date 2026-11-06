"""I/O for the coded-aperture pipeline.

`CodedReader` is a standalone reader rather than a subclass of `Reader`: the
coded preprocessing writes a different file set from the multi-distance ESRF
pipeline, so almost nothing would be inherited.  The differences:

  * one distance, so the datasets carry no `{k}` suffix
  * the data is stored UNBINNED and is binned on the way in, where the
    multi-distance pipeline pre-bins at step 4 into `pdata{k}_{bin}`
  * the code plane and the initial object are separate `.npy` files, both big
    enough to want streaming rather than a single read
  * there is no flat field to read: the CA flat has the code in the beam, so
    `PrbfitTerm` is not used and `RecCoded` has no `self.ref`

Expected inputs (names as produced by the coded preprocessing):

    <h5_file>                       HDF5
      /exchange/energy                [1]     keV
      /exchange/detector_pixelsize    [1]     m, unbinned
      /exchange/focusdetectordistance [1]     m
      /exchange/z1                    [1]     m, source to sample
      /exchange/theta                 [ntheta]  DEGREES
      /exchange/shifts_cor            [ntheta, 2]  object positions, unbinned px, (y, x)
      /exchange/shifts_code           [ntheta, 2]  code positions,   unbinned px, (y, x)
      /exchange/pdata                 [ntheta, nz, n]  INTENSITY, unbinned
      /exchange/prb                   [nz, n]  complex64, unbinned
    <code_file>                       .npy, [ncode, ncode], unbinned
    <obj_init_file>                   .npy, [nzobj, nobj, nobj] complex, unbinned (optional)

DATA IS INTENSITY.  `RecCoded.F0` compares `K|psi|^2` against `d`, and the
amplitude model takes its own sqrt, so nothing is square-rooted on load.  The
older coded fork stored `sqrt(intensity)` in `cl.data` because its misfit was
written on amplitudes; a config carried over from it must drop that sqrt or
every error it reports is the square of what it should be.

`CodedWriter` DOES subclass `Writer`: the checkpoint format is this package's
own, so the only differences are the two extra variables and the absence of a
distance axis, and his z-batched object write, TIFF previews and position
plots are all worth inheriting.
"""

import os

import h5py
import numpy as np
import cupy as cp

from .reader import find_latest_checkpoint          # noqa: F401  (re-exported)
from .writer import Writer
from .logger_config import logger


def _bin2(a, nbin, axes):
    """Average-bin `a` by 2**nbin along each axis in `axes`."""
    for _ in range(nbin):
        for ax in axes:
            sl0 = [slice(None)] * a.ndim
            sl1 = [slice(None)] * a.ndim
            sl0[ax] = slice(0, None, 2)
            sl1[ax] = slice(1, None, 2)
            a = 0.5 * (a[tuple(sl0)] + a[tuple(sl1)])
    return a


class CodedReader:
    """MPI-aware reader for the coded-aperture inputs.

    Captures every fixed parameter at construction, like `Reader`, so each
    read_* needs nothing beyond its output buffer.  Acquisition parameters are
    read once and exposed as attributes (`energy`, `detector_pixelsize`,
    `focustodetectordistance`, `z1`, `theta`).

    `z1_code` is NOT read from the file: the code-plane distance that actually
    focuses the aperture differs from the nominal one recorded at the beamline,
    and the fork pinned it by hand.  It is a config value, passed to
    `RecCoded` directly.
    """

    def __init__(self, h5_file, code_file, comm,
                 st_obj, end_obj, nzobj, nobj,
                 st_theta, end_theta, ntheta,
                 nz, n, ncode, bin,
                 obj_init_file=None, start_theta=0, stride=1):
        self.h5_file   = h5_file
        self.code_file = code_file
        self.obj_init_file = obj_init_file
        self.comm      = comm
        self.rank      = comm.Get_rank()
        self.st_obj    = st_obj
        self.end_obj   = end_obj
        self.nzobj     = nzobj
        self.nobj      = nobj
        self.st_theta  = st_theta
        self.end_theta = end_theta
        self.ntheta    = ntheta
        self.nz        = nz
        self.n         = n
        self.ncode     = ncode
        self.bin       = int(bin)
        self.start_theta = int(start_theta)
        self.stride      = int(stride)

        with h5py.File(h5_file, 'r', driver='mpio', comm=self.comm) as f:
            self.detector_pixelsize      = float(f['/exchange/detector_pixelsize'][0])
            self.focustodetectordistance = float(f['/exchange/focusdetectordistance'][0])
            self.z1                      = float(f['/exchange/z1'][0])
            self.energy                  = float(f['/exchange/energy'][0])
            theta_deg = np.asarray(f['/exchange/theta'][:]).reshape(-1)
            self.ids = (self.start_theta
                        + self.stride * np.arange(ntheta)).astype('int')
            if self.ids[-1] >= theta_deg.size:
                raise ValueError(
                    f"ntheta={ntheta} with start_theta={start_theta} and "
                    f"stride={stride} reaches index {self.ids[-1]}, past the "
                    f"{theta_deg.size} angles in {h5_file}")
            self.theta = np.deg2rad(theta_deg[self.ids]).astype('float32')
            self.shifts_cor  = np.asarray(
                f['/exchange/shifts_cor'][:], dtype='float32')[self.ids]
            self.shifts_code = np.asarray(
                f['/exchange/shifts_code'][:], dtype='float32')[self.ids]
        # detector_pixelsize is per BINNED pixel from here on, as in Reader
        self.detector_pixelsize *= 2 ** self.bin

    # ------------------------------------------------------------------

    def _theta_slice(self, i0, i1):
        """Indices into the on-disk angle axis for this rank's [i0, i1)."""
        return slice(self.start_theta + (self.st_theta + i0) * self.stride,
                     self.start_theta + (self.st_theta + i1) * self.stride,
                     self.stride)

    def read_data(self, out=None):
        """This rank's projections, binned, as INTENSITY -- no sqrt.

        Streamed in theta-batches and binned in place: the unbinned frames are
        16x the binned ones at bin 2, and reading them all at once to bin
        afterwards is the peak-memory spike this avoids.
        """
        if out is None:
            out = np.empty([self.end_theta - self.st_theta, self.nz, self.n],
                           dtype='float32')
        local = self.end_theta - self.st_theta
        with h5py.File(self.h5_file, 'r') as f:
            ds = f['/exchange/pdata']
            nz_un, n_un = ds.shape[1], ds.shape[2]
            batch = max(1, (1 << 28) // max(nz_un * n_un * 4, 1))   # ~256 MB
            for i0 in range(0, local, batch):
                i1 = min(i0 + batch, local)
                slc = np.asarray(ds[self._theta_slice(i0, i1)], dtype='float32')
                slc = _bin2(slc, self.bin, axes=(1, 2))
                # A negative count is a flat-field artefact, not a measurement;
                # the amplitude model would take sqrt of it.
                np.maximum(slc, 0, out=slc)
                out[i0:i1] = slc
                del slc
        if self.rank == 0:
            logger.info(f'data: [{self.ntheta}, {self.nz}, {self.n}] intensity '
                        f'at bin {self.bin} from {self.h5_file}')
        return out

    def read_prb(self, out=None):
        """The probe at the code plane, binned.  Ones when absent."""
        if out is None:
            out = np.empty([self.nz, self.n], dtype='complex64')
        with h5py.File(self.h5_file, 'r') as f:
            if '/exchange/prb' in f:
                prb = np.asarray(f['/exchange/prb'][:], dtype='complex64')
                prb = _bin2(prb, self.bin, axes=(0, 1))
                out[:] = prb[:self.nz, :self.n]
                if self.rank == 0:
                    logger.info(f'probe read from {self.h5_file}')
            else:
                out[:] = 1
                if self.rank == 0:
                    logger.warning('no /exchange/prb in the file; probe set to 1')
        return out

    def read_code(self, out=None):
        """The coded aperture, binned, into a [ncode, ncode] GPU array."""
        code = np.load(self.code_file).astype('complex64')
        code = _bin2(code, self.bin, axes=(0, 1))
        if code.shape != (self.ncode, self.ncode):
            raise ValueError(
                f'{self.code_file} binned to {code.shape}, expected '
                f'({self.ncode}, {self.ncode}) -- check ncode against bin')
        if out is None:
            return cp.asarray(code)
        out[:] = cp.asarray(code)
        if self.rank == 0:
            logger.info(f'code: [{self.ncode}, {self.ncode}] at bin {self.bin} '
                        f'from {self.code_file}')
        return out

    def read_pos(self, out=None):
        """Object positions for this rank's angles, scaled to the binned grid."""
        pos = self.shifts_cor[self.st_theta:self.end_theta] / 2 ** self.bin
        if out is None:
            return pos.astype('float32')
        out[:] = pos
        return out

    def read_pos_code(self, out=None):
        """Code positions for this rank's angles, scaled to the binned grid."""
        pos = self.shifts_code[self.st_theta:self.end_theta] / 2 ** self.bin
        if out is None:
            return pos.astype('float32')
        out[:] = pos
        return out

    def read_obj(self, out):
        """Initial object for this rank's z slab, binned isotropically.

        Streamed one unbinned z-row at a time into a binned accumulator, so the
        peak is one unbinned slice rather than the whole volume.  Zeroed when
        no init file is configured or present -- a cold start is legitimate
        here, unlike the multi-distance pipeline where step 5 always supplies
        a Paganin guess.
        """
        out[:] = 0
        path = self.obj_init_file
        have = os.path.exists(path) if (path and self.rank == 0) else False
        have = self.comm.bcast(have if self.rank == 0 else None, root=0)
        if not have:
            if self.rank == 0:
                logger.warning('no initial object; starting from zero')
            return out

        mmap = np.load(path, mmap_mode='r')
        factor = 2 ** self.bin
        nzobj_full = mmap.shape[0]
        nobj_bin = mmap.shape[1] // factor
        sxy = min(nobj_bin, self.nobj)

        if factor == 1:
            for k in range(self.end_obj - self.st_obj):
                z = self.st_obj + k
                if z < nzobj_full:
                    out[k, :sxy, :sxy] = np.asarray(
                        mmap[z], dtype='complex64')[:sxy, :sxy]
        else:
            N = nobj_bin * factor                      # truncate to a bin multiple
            acc = np.empty((nobj_bin, nobj_bin), dtype='complex64')
            for k in range(self.end_obj - self.st_obj):
                acc[:] = 0
                count = 0
                for dz in range(factor):
                    z = (self.st_obj + k) * factor + dz
                    if z >= nzobj_full:
                        break
                    row = np.asarray(mmap[z], dtype='complex64')[:N, :N]
                    acc += row.reshape(nobj_bin, factor,
                                       nobj_bin, factor).mean(axis=(1, 3))
                    count += 1
                if count > 0:
                    out[k, :sxy, :sxy] = (acc / count)[:sxy, :sxy]
        del mmap
        if self.rank == 0:
            logger.info(f'initial object from {path} at bin {self.bin}')
        return out

    def read_checkpoint(self, path, out_obj=None, out_prb=None, out_pos=None,
                        out_code=None, out_pos_code=None):
        """Read a CodedWriter checkpoint, upsampling a coarser one to this grid.

        This is what makes the bin2 -> bin1 -> bin0 ladder work: the rungs
        share one `path_out` and hand off through `start_iter`, so every level
        but the first reads a checkpoint written at half its width.  `Reader`
        does the same for the multi-distance pipeline; the extra work here is
        that the aperture and its positions scale too.

        Scale is inferred, never configured: `scale = self.n / checkpoint n`
        for the detector-plane arrays, and the object's own z and x/y scales
        are taken from the object dataset so they stay right under
        tomo_upsample.  Repetition, not interpolation -- the solver smooths it
        out in the first few iterations and a nearest-neighbour upsample
        cannot invent detail that the coarse rung did not have.

        Positions are PIXEL coordinates, so both sets are multiplied by the
        scale; the aperture is a plane like the object and is repeated.
        """
        with h5py.File(path, 'r', driver='mpio', comm=self.comm) as f:
            scale = self.n // f['prb_abs'].shape[-1]

            if out_prb is not None:
                prb = (np.asarray(f['prb_abs'][0])
                       * np.exp(1j * np.asarray(f['prb_phase'][0]))
                       ).astype('complex64')
                for axis in (1, 0):
                    prb = np.repeat(prb, scale, axis=axis)
                out_prb[:] = prb
                del prb

            if out_pos is not None:
                # Writer stores [ntheta, ndist, 2] with ndist = 1.
                out_pos[:] = np.asarray(
                    f['pos'][self.st_theta:self.end_theta, 0],
                    dtype='float32') * scale
            if out_pos_code is not None:
                out_pos_code[:] = np.asarray(
                    f['pos_code'][self.st_theta:self.end_theta],
                    dtype='float32') * scale

            if out_obj is not None:
                ds_re, ds_im = f['obj_re'], f['obj_im']
                nzobj0, nobj0 = ds_re.shape[0], ds_re.shape[1]
                scale_z  = max(1, self.nzobj // nzobj0)
                scale_xy = max(1, self.nobj // nobj0)
                st_src  = self.st_obj // scale_z
                end_src = self.end_obj // scale_z
                n0      = self.end_obj - self.st_obj
                nz_src  = max(1, end_src - st_src)
                if self.rank == 0 and scale_xy > 1:
                    logger.warning(
                        f'checkpoint {os.path.basename(path)}: obj '
                        f'[{nzobj0}, {nobj0}, {nobj0}] -> [{self.nzobj}, '
                        f'{self.nobj}, {self.nobj}] (z x{scale_z}, '
                        f'x/y x{scale_xy}); prb/pos x{scale}')

                z_batch = max(1, (1 << 28) // max(nobj0 * nobj0 * 8, 1))
                for i0 in range(0, n0, z_batch):
                    i1 = min(i0 + z_batch, n0)
                    src_i0 = int(i0 * nz_src / n0)
                    src_i1 = min(int((i1 - 1) * nz_src / n0) + 1, nz_src)
                    blk = np.zeros((src_i1 - src_i0, nobj0, nobj0),
                                   dtype='complex64')
                    blk.real[:] = ds_re[st_src + src_i0:st_src + src_i1]
                    blk.imag[:] = ds_im[st_src + src_i0:st_src + src_i1]
                    if scale_xy > 1:
                        for axis in (2, 1):
                            blk = np.repeat(blk, scale_xy, axis=axis)
                    idx = np.clip((np.arange(i0, i1) * nz_src / n0).astype(np.intp),
                                  0, nz_src - 1) - src_i0
                    out_obj[i0:i1] = blk[idx]
                    del blk

            if out_code is not None:
                code = (np.asarray(f['code_re'])
                        + 1j * np.asarray(f['code_im'])).astype('complex64')
                scale_c = max(1, self.ncode // code.shape[-1])
                if scale_c > 1:
                    for axis in (1, 0):
                        code = np.repeat(code, scale_c, axis=axis)
                out_code[:] = cp.asarray(code)
                del code

        if self.rank == 0:
            logger.info(f'resumed from {path} (x{scale})')


class CodedWriter(Writer):
    """`Writer` plus the two coded variables.

    Constructed with `ndist = 1` so the parent's dist-shaped datasets (`pos`
    as [ntheta, 1, 2], `prb_*` as [1, nz, n]) still apply; `write_checkpoint`
    hands the parent a dist-shaped VIEW of the coded variables and then appends

        /code_re, /code_im   (ncode, ncode)   float32, rank 0
        /pos_code            (ntheta, 2)      float32, assembled from all ranks

    Everything else -- the z-batched object write, the TIFF previews, the
    position-drift plot -- is the parent's.
    """

    def __init__(self, path_out, comm,
                 st_obj, end_obj, nzobj, nobj,
                 st_theta, end_theta, ntheta,
                 nz, n, ncode):
        super().__init__(path_out, comm, st_obj, end_obj, nzobj, nobj,
                         st_theta, end_theta, ntheta, 1, nz, n)
        self.ncode = ncode

    def write_checkpoint(self, vars, i, norm_const, pos_init=None,
                         pos_code_init=None, **kw):
        # The parent indexes pos as [ndist, local_ntheta, 2] and prb as
        # [ndist, nz, n]; a reshape is a view, so this costs nothing.
        shim = dict(vars)
        shim['pos'] = np.asarray(vars['pos'])[None]
        shim['prb'] = np.asarray(vars['prb'])[None]
        shim.pop('tp', None)
        super().write_checkpoint(
            shim, i, norm_const,
            pos_init=None if pos_init is None else np.asarray(pos_init)[None],
            **kw)

        path = os.path.join(self.h5_dir, f"checkpoint_{i:04}.h5")
        pos_code = self._cpu(vars['pos_code'])
        with h5py.File(path, 'a', driver='mpio', comm=self.comm) as f:
            ds = f.create_dataset('pos_code', shape=(self.ntheta, 2),
                                  dtype='float32')
            ds[self.st_theta:self.end_theta] = pos_code
        self.comm.Barrier()
        if self.rank == 0:
            code = self._cpu(vars['code'])
            with h5py.File(path, 'a') as f:
                f.create_dataset('code_re', data=code.real.astype('float32'))
                f.create_dataset('code_im', data=code.imag.astype('float32'))
            logger.info(f'CodedWriter: code + pos_code appended to {path}')
        self.comm.Barrier()

        if pos_code_init is not None:
            self._save_code_pos_plot(
                np.asarray(vars['pos_code']) - np.asarray(pos_code_init), i)

    def _save_code_pos_plot(self, delta_local, i):
        """Code-position drift per angle.  The parent's plot covers `pos`
        only, and pos_code drifting is a distinct failure mode here: the
        window offsets are frozen for the whole solve, so a large drift means
        the subpixel residual has left the pad (which `check_code_drift`
        turns into an error) long before the reconstruction looks wrong."""
        if isinstance(delta_local, cp.ndarray):
            delta_local = delta_local.get()
        parts = self.comm.gather(np.asarray(delta_local), root=0)
        if self.rank != 0:
            return
        delta = np.concatenate(parts, axis=0)            # [ntheta, 2]
        a = np.abs(delta)
        logger.warning(
            f"iter={i}: pos_code abs drift [px]  "
            f"y=({a[:, 0].mean():.4f}+-{a[:, 0].std():.4f} max={a[:, 0].max():.4f})  "
            f"x=({a[:, 1].mean():.4f}+-{a[:, 1].std():.4f} max={a[:, 1].max():.4f})")

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        for d, label in enumerate(('y', 'x')):
            axes[d].plot(np.arange(delta.shape[0]), delta[:, d])
            axes[d].set_ylabel(f"pos_code {label} drift [px]")
            axes[d].grid(True)
        axes[-1].set_xlabel("theta index")
        fig.tight_layout()
        out_dir = os.path.join(self.path_out, "pos_code_errors")
        os.makedirs(out_dir, exist_ok=True)
        png = os.path.join(out_dir, f"pos_code_error_{i:04}.png")
        fig.savefig(png, dpi=150)
        plt.close(fig)
        logger.info(f"pos_code drift plot -> {png}")
