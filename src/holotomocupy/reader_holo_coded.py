"""I/O for the 2-D coded-aperture holography pipeline.

`HoloCodedReader` is `reader_coded.CodedReader` with the tomography removed.
The contract is deliberately the same file the coded preprocessing already
writes, so a 2-D experiment reuses those notebooks with the angle loop taken
out.  Exactly two things differ.

  * there is no `/exchange/theta`: the scan axis is FRAMES, not angles
  * `obj_init` is a 2-D plane, `[nzobj, nobj]`, not a volume

Expected inputs (names as produced by the coded preprocessing):

    <h5_file>                       HDF5
      /exchange/energy                [1]     keV
      /exchange/detector_pixelsize    [1]     m, unbinned
      /exchange/focusdetectordistance [1]     m        (NOTE: no "to")
      /exchange/z1                    [1]     m, source to sample
      /exchange/shifts_cor            [nframes, 2]  object positions, unbinned px, (y, x)
      /exchange/shifts_code           [nframes, 2]  code positions,   unbinned px, (y, x)
      /exchange/pdata                 [nframes, nz, n]  INTENSITY, unbinned
      /exchange/prb                   [nz, n]  complex64, unbinned (optional)
    <code_file>                       .npy, [ncode, ncode], unbinned
    <obj_init_file>                   .npy, [nzobj, nobj] complex, unbinned (optional)

The dataset paths live in `_LAYOUT` below, and every read goes through it.
The 2-D data this was written for does not exist yet, so the layout is the
house convention rather than something validated against a real file; adapting
to a different one should be an edit of that dict, not of the methods.

DATA IS INTENSITY.  `RecHoloCoded.F0` compares `K|psi|^2` against `d`, and the
amplitude model takes its own sqrt, so nothing is square-rooted on load.

NO FLAT FIELD, for the same reason `CodedReader` reads none: the coded flat has
the mask in the beam and the mask moves frame to frame, so it is not a model
for `K|D prb|^2` and `PrbfitTerm` does not apply.  (The multi-distance
`Reader.read_ref` does read one, because there the flat IS the propagated
probe.)  Dark is subtracted in preprocessing in every pipeline here.

`z1_code` is NOT read from the file.  The code-plane distance that actually
focuses the aperture differs from the nominal one recorded at the beamline, so
it is a tuned config value passed to `RecHoloCoded` directly.

`HoloCodedWriter` does NOT subclass `Writer`, unlike `CodedWriter`.  The parent
writes the object as a z-batched VOLUME and carries an `ndist` axis; with a 2-D
object there is nothing to batch over and no distance axis, so inheriting would
mean faking both.  The checkpoint dataset names are kept identical so the
format stays readable by the same tooling.
"""

import os

import h5py
import numpy as np
import cupy as cp

from .reader import find_latest_checkpoint          # noqa: F401  (re-exported)
from .reader_coded import _bin2
from .logger_config import logger

# Dataset paths inside the input HDF5. PROVISIONAL -- mirrors reader_coded
# minus `theta`, pending the first real 2-D file. Adapt HERE, not in the
# methods below.
_LAYOUT = {
    'energy':      '/exchange/energy',
    'det_px':      '/exchange/detector_pixelsize',
    'fdd':         '/exchange/focusdetectordistance',
    'z1':          '/exchange/z1',
    'shifts':      '/exchange/shifts_cor',
    'shifts_code': '/exchange/shifts_code',
    'data':        '/exchange/pdata',
    'prb':         '/exchange/prb',          # optional; probe = 1 if missing
}

# Everything but `prb`, which is allowed to be absent.
_REQUIRED = tuple(k for k in _LAYOUT if k != 'prb')


class HoloCodedReader:
    """MPI-aware reader for the 2-D coded-aperture inputs.

    Captures every fixed parameter at construction, like `CodedReader`, so each
    read_* needs nothing beyond its output buffer.  Acquisition parameters are
    read once and exposed as attributes (`energy`, `detector_pixelsize`,
    `focustodetectordistance`, `z1`).

    Frames are distributed across ranks (`st_theta`/`end_theta`, named for the
    engine's own attributes); the object, probe and code are replicated, so
    unlike `CodedReader` there is no object-slab split.
    """

    def __init__(self, h5_file, code_file, comm,
                 nzobj, nobj, st_theta, end_theta, ntheta,
                 nz, n, ncode, bin,
                 obj_init_file=None, start_theta=0, stride=1):
        self.h5_file   = h5_file
        self.code_file = code_file
        self.obj_init_file = obj_init_file
        self.comm      = comm
        self.rank      = comm.Get_rank()
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

        with h5py.File(h5_file, 'r') as f:
            # Report every missing key at once: a layout mismatch should cost
            # one run, not one run per key. See the module docstring.
            missing = [_LAYOUT[k] for k in _REQUIRED if _LAYOUT[k] not in f]
            if missing:
                raise KeyError(
                    f'{h5_file} is missing {missing}. Expected the layout in '
                    f'reader_holo_coded._LAYOUT = {_LAYOUT}')

            self.detector_pixelsize      = float(f[_LAYOUT['det_px']][0])
            self.focustodetectordistance = float(f[_LAYOUT['fdd']][0])
            self.z1                      = float(f[_LAYOUT['z1']][0])
            self.energy                  = float(f[_LAYOUT['energy']][0])

            nframes_file = f[_LAYOUT['data']].shape[0]
            self.ids = (self.start_theta
                        + self.stride * np.arange(ntheta)).astype('int')
            if self.ids[-1] >= nframes_file:
                raise ValueError(
                    f"nframes={ntheta} with start_theta={start_theta} and "
                    f"stride={stride} reaches index {self.ids[-1]}, past the "
                    f"{nframes_file} frames in {h5_file}")
            self.shifts_cor  = np.asarray(
                f[_LAYOUT['shifts']][:], dtype='float32')[self.ids]
            self.shifts_code = np.asarray(
                f[_LAYOUT['shifts_code']][:], dtype='float32')[self.ids]
        # detector_pixelsize is per BINNED pixel from here on, as in Reader
        self.detector_pixelsize *= 2 ** self.bin

    # ------------------------------------------------------------------

    def _frame_slice(self, i0, i1):
        """Indices into the on-disk frame axis for this rank's [i0, i1)."""
        return slice(self.start_theta + (self.st_theta + i0) * self.stride,
                     self.start_theta + (self.st_theta + i1) * self.stride,
                     self.stride)

    def read_data(self, out=None):
        """This rank's frames, binned, as INTENSITY -- no sqrt.

        Streamed in batches and binned in place: the unbinned frames are 16x
        the binned ones at bin 2, and reading them all at once to bin
        afterwards is the peak-memory spike this avoids.
        """
        if out is None:
            out = np.empty([self.end_theta - self.st_theta, self.nz, self.n],
                           dtype='float32')
        local = self.end_theta - self.st_theta
        with h5py.File(self.h5_file, 'r') as f:
            ds = f[_LAYOUT['data']]
            nz_un, n_un = ds.shape[1], ds.shape[2]
            batch = max(1, (1 << 28) // max(nz_un * n_un * 4, 1))   # ~256 MB
            for i0 in range(0, local, batch):
                i1 = min(i0 + batch, local)
                slc = np.asarray(ds[self._frame_slice(i0, i1)], dtype='float32')
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
            out = cp.empty([self.nz, self.n], dtype='complex64')
        with h5py.File(self.h5_file, 'r') as f:
            if _LAYOUT['prb'] in f:
                prb = np.asarray(f[_LAYOUT['prb']][:], dtype='complex64')
                prb = _bin2(prb, self.bin, axes=(0, 1))
                out[:] = cp.asarray(prb[:self.nz, :self.n])
                if self.rank == 0:
                    logger.info(f'probe read from {self.h5_file}')
            else:
                out[:] = 1
                if self.rank == 0:
                    logger.warning(f'no {_LAYOUT["prb"]} in the file; '
                                   f'probe set to 1')
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
        """Object positions for this rank's frames, scaled to the binned grid.

        No `rot_axis_offset` and no `correct3d_extra` here, unlike
        `CodedReader`: both exist to reconcile a rotation axis, and there is no
        rotation.
        """
        pos = (np.array(self.shifts_cor[self.st_theta:self.end_theta],
                        dtype='float32') / 2 ** self.bin)
        if out is None:
            return pos.astype('float32')
        out[:] = pos
        return out

    def read_pos_code(self, out=None):
        """Code positions for this rank's frames, scaled to the binned grid."""
        pos = self.shifts_code[self.st_theta:self.end_theta] / 2 ** self.bin
        if out is None:
            return pos.astype('float32')
        out[:] = pos
        return out

    def read_obj(self, out):
        """Initial object, binned, CENTRED in the object grid.

        Zeroed when no init file is configured or present -- a cold start is
        legitimate here, as it is for the coded engine and unlike the
        multi-distance pipeline where step 5 always supplies a Paganin guess.

        Centring matters for the same reason it does in `CodedReader`.
        `ShiftFFT` takes the detector window centred in the object grid, so a
        seed smaller than the grid has to sit in the MIDDLE of it and a larger
        one has to be cropped symmetrically.  Anchoring at index 0 misregisters
        the seed against the forward model, and the data term then rebuilds the
        object where it does expect it -- two offset copies, summed.
        """
        out[:] = 0
        path = self.obj_init_file
        have = bool(path) and os.path.exists(path)
        have = self.comm.bcast(have if self.rank == 0 else None, root=0)
        if not have:
            if self.rank == 0:
                logger.warning('no initial object; starting from zero')
            return out

        seed = np.load(path).astype('complex64')
        seed = _bin2(seed, self.bin, axes=(0, 1))

        def _place(seed_n, grid_n):
            """(src0, dst0, count) mapping a centred seed onto a centred grid."""
            if seed_n >= grid_n:
                return (seed_n - grid_n) // 2, 0, grid_n     # crop, centred
            return 0, (grid_n - seed_n) // 2, seed_n         # place, centred

        ys0, yd0, nyc = _place(seed.shape[0], self.nzobj)
        xs0, xd0, nxc = _place(seed.shape[1], self.nobj)
        out[yd0:yd0 + nyc, xd0:xd0 + nxc] = cp.asarray(
            seed[ys0:ys0 + nyc, xs0:xs0 + nxc])
        if self.rank == 0:
            logger.info(f'initial object from {path} at bin {self.bin} '
                        f'({seed.shape} -> [{self.nzobj}, {self.nobj}])')
        return out

    def read_checkpoint(self, path, out_obj=None, out_prb=None, out_pos=None,
                        out_code=None, out_pos_code=None):
        """Read a `HoloCodedWriter` checkpoint, upsampling a coarser one.

        This is what makes the bin2 -> bin1 -> bin0 ladder work: the rungs
        share one `path_out` and hand off through `start_iter`, so every level
        but the first reads a checkpoint written at half its width.

        Scale is inferred, never configured: `scale = self.n / checkpoint n`
        for the detector-plane arrays, with the object and code taking their
        own from their datasets.  Repetition, not interpolation -- the solver
        smooths it out in the first few iterations, and a nearest-neighbour
        upsample cannot invent detail the coarse rung did not have.

        Positions are PIXEL coordinates, so both sets are multiplied by scale.
        """
        with h5py.File(path, 'r') as f:
            scale = self.n // f['prb_abs'].shape[-1]

            if out_prb is not None:
                prb = (np.asarray(f['prb_abs'])
                       * np.exp(1j * np.asarray(f['prb_phase']))
                       ).astype('complex64')
                for axis in (1, 0):
                    prb = np.repeat(prb, scale, axis=axis)
                out_prb[:] = cp.asarray(prb)
                del prb

            if out_pos is not None:
                out_pos[:] = np.asarray(
                    f['pos'][self.st_theta:self.end_theta],
                    dtype='float32') * scale
            if out_pos_code is not None:
                out_pos_code[:] = np.asarray(
                    f['pos_code'][self.st_theta:self.end_theta],
                    dtype='float32') * scale

            if out_obj is not None:
                obj = (np.asarray(f['obj_re'])
                       + 1j * np.asarray(f['obj_im'])).astype('complex64')
                s_y = max(1, self.nzobj // obj.shape[0])
                s_x = max(1, self.nobj // obj.shape[1])
                if (s_y, s_x) != (1, 1):
                    obj = np.repeat(np.repeat(obj, s_y, axis=0), s_x, axis=1)
                    if self.rank == 0:
                        logger.warning(
                            f'checkpoint {os.path.basename(path)}: obj '
                            f'upsampled y x{s_y}, x x{s_x}; prb/pos x{scale}')
                out_obj[:] = cp.asarray(obj[:self.nzobj, :self.nobj])
                del obj

            if out_code is not None:
                code = (np.asarray(f['code_re'])
                        + 1j * np.asarray(f['code_im'])).astype('complex64')
                s_c = max(1, self.ncode // code.shape[-1])
                if s_c > 1:
                    for axis in (1, 0):
                        code = np.repeat(code, s_c, axis=axis)
                out_code[:] = cp.asarray(code)
                del code

        if self.rank == 0:
            logger.info(f'resumed from {path} (x{scale})')


class HoloCodedWriter:
    """Checkpoint writer for the 2-D coded engine.

    Standalone rather than a `Writer` subclass -- see the module docstring.
    The dataset names match `CodedWriter`'s so the format stays uniform:

        /obj_re, /obj_im       (nzobj, nobj)    float32, rank 0
        /prb_abs, /prb_phase   (nz, n)          float32, rank 0
        /code_re, /code_im     (ncode, ncode)   float32, rank 0
        /pos                   (nframes, 2)     float32, assembled from all ranks
        /pos_code              (nframes, 2)     float32, assembled from all ranks

    `pos` is (nframes, 2) here, NOT the parent's (ntheta, ndist, 2): there is
    one distance, and faking the axis would only mean indexing it away again.

    File layout matches `Writer` so `find_latest_checkpoint` works unchanged:
    {path_out}/checkpoints/checkpoint_{iter:04}.h5, with TIFF previews in
    {path_out}/checkpoints_tiff/.
    """

    def __init__(self, path_out, comm, nzobj, nobj,
                 st_theta, end_theta, ntheta, nz, n, ncode):
        self.path_out = path_out
        self.comm     = comm
        self.rank     = comm.Get_rank()
        self.nzobj    = nzobj
        self.nobj     = nobj
        self.st_theta = st_theta
        self.end_theta = end_theta
        self.ntheta   = ntheta
        self.nz       = nz
        self.n        = n
        self.ncode    = ncode
        self.h5_dir   = os.path.join(path_out, 'checkpoints')
        self.tiff_dir = os.path.join(path_out, 'checkpoints_tiff')
        if self.rank == 0:
            os.makedirs(self.h5_dir,   exist_ok=True)
            os.makedirs(self.tiff_dir, exist_ok=True)
        comm.Barrier()

    @staticmethod
    def _cpu(x):
        return x.get() if isinstance(x, cp.ndarray) else np.asarray(x)

    def write_checkpoint(self, vars, i, norm_const=1.0,
                         pos_init=None, pos_code_init=None, **kw):
        """Write checkpoint `i`.

        `norm_const` is accepted and applied to the object for signature
        parity with `Writer`; `RecHoloCoded` pins it at 1, so it is a no-op
        there and exists so a caller written against the coded engine works
        unchanged.
        """
        path = os.path.join(self.h5_dir, f'checkpoint_{i:04}.h5')

        # The two frame-distributed variables, assembled from every rank.
        pos      = self._cpu(vars['pos'])
        pos_code = self._cpu(vars['pos_code'])
        all_pos      = self.comm.gather(pos,      root=0)
        all_pos_code = self.comm.gather(pos_code, root=0)

        if self.rank == 0:
            obj  = self._cpu(vars['obj']) * float(norm_const)
            prb  = self._cpu(vars['prb'])
            code = self._cpu(vars['code'])
            with h5py.File(path, 'w') as f:
                f.create_dataset('obj_re',    data=obj.real.astype('float32'))
                f.create_dataset('obj_im',    data=obj.imag.astype('float32'))
                f.create_dataset('prb_abs',   data=np.abs(prb).astype('float32'))
                f.create_dataset('prb_phase', data=np.angle(prb).astype('float32'))
                f.create_dataset('code_re',   data=code.real.astype('float32'))
                f.create_dataset('code_im',   data=code.imag.astype('float32'))
                f.create_dataset('pos',       data=np.concatenate(all_pos, 0
                                                                  ).astype('float32'))
                f.create_dataset('pos_code',  data=np.concatenate(all_pos_code, 0
                                                                  ).astype('float32'))
            self._write_tiff(obj, code, i)
            logger.info(f'HoloCodedWriter: checkpoint -> {path}')
        self.comm.Barrier()

        if pos_init is not None:
            self._save_pos_plot(np.asarray(self._cpu(vars['pos']))
                                - np.asarray(pos_init), i, 'pos')
        if pos_code_init is not None:
            self._save_pos_plot(np.asarray(self._cpu(vars['pos_code']))
                                - np.asarray(pos_code_init), i, 'pos_code')

    def _write_tiff(self, obj, code, i):
        """Object phase and aperture phase, for a glance without opening h5."""
        try:
            import tifffile
        except ImportError:
            return
        tifffile.imwrite(os.path.join(self.tiff_dir, f'obj_phase_{i:04}.tiff'),
                         np.angle(obj).astype('float32'))
        tifffile.imwrite(os.path.join(self.tiff_dir, f'code_phase_{i:04}.tiff'),
                         np.angle(code).astype('float32'))

    def _save_pos_plot(self, delta_local, i, label):
        """Per-frame position drift.

        `pos_code` drifting is a distinct failure mode: the window offsets are
        frozen for the whole solve, so a large drift means the subpixel
        residual has left the pad -- which `check_code_drift` turns into an
        error -- long before the reconstruction looks wrong.
        """
        parts = self.comm.gather(np.asarray(delta_local), root=0)
        if self.rank != 0:
            return
        delta = np.concatenate(parts, axis=0)
        a = np.abs(delta)
        logger.warning(
            f"iter={i}: {label} abs drift [px]  "
            f"y=({a[:, 0].mean():.4f}+-{a[:, 0].std():.4f} max={a[:, 0].max():.4f})  "
            f"x=({a[:, 1].mean():.4f}+-{a[:, 1].std():.4f} max={a[:, 1].max():.4f})")

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        for d, lab in enumerate(('y', 'x')):
            axes[d].plot(np.arange(delta.shape[0]), delta[:, d])
            axes[d].set_ylabel(f'{label} {lab} drift [px]')
            axes[d].grid(True)
        axes[-1].set_xlabel('frame index')
        fig.tight_layout()
        out_dir = os.path.join(self.path_out, f'{label}_errors')
        os.makedirs(out_dir, exist_ok=True)
        png = os.path.join(out_dir, f'{label}_error_{i:04}.png')
        fig.savefig(png, dpi=150)
        plt.close(fig)
        logger.info(f'{label} drift plot -> {png}')
