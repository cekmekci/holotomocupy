#!/usr/bin/env python
"""`reader_holo_coded` against a synthetic file.

    python tests/holo_coded/test_io_holo_coded.py
    mpirun -np 2 python tests/holo_coded/test_io_holo_coded.py

The 2-D data this reader was written for does not exist yet, so what is tested
is the reader against its OWN contract -- the layout in
`reader_holo_coded._LAYOUT`.  That catches a binning slip, a transposed shift,
a lost frame slice or a broken checkpoint round trip; it cannot catch the
layout itself being wrong, which only a real file can.  See the module
docstring there.

  read_*            shapes, dtypes and the per-rank frame split
  data is INTENSITY the one convention that silently squares every error
  binning           halves the grid, conserves the mean, scales the positions
  checkpoint        round trips every variable
  upsampling        a coarse checkpoint read onto a finer grid (the bin ladder)
  missing keys      reported all at once, with the layout
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import cupy as cp
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'unit'))
from harness import check, close, run                            # noqa: E402

from mpi4py import MPI                                           # noqa: E402
from holotomocupy.reader_holo_coded import (                     # noqa: E402
    HoloCodedReader, HoloCodedWriter, _LAYOUT)

NFRAMES, NZ, N = 8, 32, 32          # UNBINNED
NOBJ = 3 * N // 2
NCODE = 2 * N
ENERGY, DET_PX, FDD, Z1 = 17.1, 2.3616e-05, 1.217, 5.110e-3

comm = MPI.COMM_WORLD
RANK, SIZE = comm.Get_rank(), comm.Get_size()


def _tmpdir():
    """One temp dir shared by every rank.

    `tempfile.mkdtemp()` returns a DIFFERENT path per rank, so rank 0 would
    write the fixture into its own directory while every other rank looked
    into an empty one, failed to open the file, and desynchronised at the next
    collective -- a HANG, not an error. Rank 0 makes it and broadcasts it.
    """
    return comm.bcast(tempfile.mkdtemp() if RANK == 0 else None, root=0)


def _rmtree(d):
    comm.Barrier()
    if RANK == 0:
        shutil.rmtree(d, ignore_errors=True)


def _split(ntheta):
    """The same frame split MPIClass uses, so the test agrees with the engine."""
    q, r = divmod(ntheta, SIZE)
    st = RANK * q + min(RANK, r)
    return st, st + q + (1 if RANK < r else 0)


def _make_file(d, seed=0):
    """Write a synthetic input set matching _LAYOUT exactly."""
    rng = np.random.default_rng(seed)
    h5 = os.path.join(d, 'holo.h5')
    code_npy = os.path.join(d, 'code.npy')
    obj_npy = os.path.join(d, 'obj_init.npy')

    data = (rng.random((NFRAMES, NZ, N)).astype('float32') + 0.5)
    shifts = (10 * (rng.random((NFRAMES, 2)) - 0.5)).astype('float32')
    shifts_code = (20 * (rng.random((NFRAMES, 2)) - 0.5)).astype('float32')
    prb = (rng.random((NZ, N)) + 1j * rng.random((NZ, N))).astype('complex64')

    if RANK == 0:
        with h5py.File(h5, 'w') as f:
            f.create_dataset(_LAYOUT['energy'], data=np.array([ENERGY]))
            f.create_dataset(_LAYOUT['det_px'], data=np.array([DET_PX]))
            f.create_dataset(_LAYOUT['fdd'],    data=np.array([FDD]))
            f.create_dataset(_LAYOUT['z1'],     data=np.array([Z1]))
            f.create_dataset(_LAYOUT['shifts'],      data=shifts)
            f.create_dataset(_LAYOUT['shifts_code'], data=shifts_code)
            f.create_dataset(_LAYOUT['data'],        data=data)
            f.create_dataset(_LAYOUT['prb'],         data=prb)
        np.save(code_npy, (rng.random((NCODE, NCODE))
                           + 1j * rng.random((NCODE, NCODE))).astype('complex64'))
        np.save(obj_npy, (0.1 * rng.random((NOBJ, NOBJ))).astype('complex64'))
    comm.Barrier()
    return h5, code_npy, obj_npy, data, shifts, shifts_code


def _reader(d, bin=0, obj_init=None):
    h5 = os.path.join(d, 'holo.h5')
    st, end = _split(NFRAMES)
    return HoloCodedReader(
        h5, os.path.join(d, 'code.npy'), comm,
        nzobj=NOBJ >> bin, nobj=NOBJ >> bin,
        st_theta=st, end_theta=end, ntheta=NFRAMES,
        nz=NZ >> bin, n=N >> bin, ncode=NCODE >> bin, bin=bin,
        obj_init_file=obj_init)


# ----------------------------------------------------------------------

def test_geometry_and_shapes():
    d = _tmpdir()
    try:
        _make_file(d)
        r = _reader(d)
        close('energy', r.energy, ENERGY, 1e-12)
        close('focustodetectordistance', r.focustodetectordistance, FDD, 1e-12)
        close('z1', r.z1, Z1, 1e-12)
        close('detector_pixelsize at bin 0', r.detector_pixelsize, DET_PX, 1e-12)

        data = r.read_data()
        st, end = _split(NFRAMES)
        check('data shape is this rank s frames',
              data.shape == (end - st, NZ, N), f'{data.shape}')
        check('data dtype float32', data.dtype == np.float32, str(data.dtype))
        prb = r.read_prb()
        check('prb shape/dtype', prb.shape == (NZ, N)
              and prb.dtype == cp.complex64, f'{prb.shape} {prb.dtype}')
        code = r.read_code()
        check('code shape/dtype', code.shape == (NCODE, NCODE)
              and code.dtype == cp.complex64, f'{code.shape} {code.dtype}')
        pos, pc = r.read_pos(), r.read_pos_code()
        check('pos shapes', pos.shape == (end - st, 2) and pc.shape == (end - st, 2),
              f'{pos.shape} {pc.shape}')
    finally:
        _rmtree(d)


def test_data_is_intensity_not_amplitude():
    """The one convention that silently squares every reported error.

    RecHoloCoded.F0 compares K|psi|^2 against d and the amplitude model takes
    its own sqrt, so the reader must hand back exactly what is on disk.
    """
    d = _tmpdir()
    try:
        _, _, _, data, _, _ = _make_file(d, seed=3)
        got = _reader(d).read_data()
        st, end = _split(NFRAMES)
        err = float(np.abs(got - data[st:end]).max())
        check('read_data returns the stored INTENSITY, un-sqrt-ed', err == 0.0,
              f'max |got - stored| = {err:.3e}')
        # and it is NOT the square root of it
        sq = float(np.abs(got - np.sqrt(data[st:end])).max())
        check('read_data is not sqrt(intensity)', sq > 1e-3, f'{sq:.3e}')
    finally:
        _rmtree(d)


def test_frame_split_covers_every_frame_once():
    d = _tmpdir()
    try:
        _, _, _, data, _, _ = _make_file(d, seed=4)
        got = _reader(d).read_data()
        total = comm.allreduce(got.shape[0], MPI.SUM)
        check(f'{SIZE} rank(s) cover all {NFRAMES} frames exactly once',
              total == NFRAMES, f'{total}')
        s = comm.allreduce(float(got.sum()), MPI.SUM)
        close('summed data matches the file', s, float(data.sum()), 1e-5)
    finally:
        _rmtree(d)


def test_binning():
    d = _tmpdir()
    try:
        _, _, _, data, shifts, shifts_code = _make_file(d, seed=5)
        r = _reader(d, bin=1)
        got = r.read_data()
        st, end = _split(NFRAMES)
        check('binned data shape', got.shape == (end - st, NZ // 2, N // 2),
              f'{got.shape}')
        close('binning conserves the mean',
              float(got.mean()), float(data[st:end].mean()), 2e-6)
        close('detector_pixelsize doubles at bin 1',
              r.detector_pixelsize, DET_PX * 2, 1e-12)
        close('positions halve at bin 1',
              float(np.abs(r.read_pos()).sum()),
              float(np.abs(shifts[st:end]).sum()) / 2, 1e-5)
        close('code positions halve at bin 1',
              float(np.abs(r.read_pos_code()).sum()),
              float(np.abs(shifts_code[st:end]).sum()) / 2, 1e-5)
        check('code binned', r.read_code().shape == (NCODE // 2, NCODE // 2))
    finally:
        _rmtree(d)


def test_obj_init_is_centred():
    """A seed smaller than the grid must land in the MIDDLE of it, or the data
    term rebuilds the object where it does expect it -- two offset copies."""
    d = _tmpdir()
    try:
        _make_file(d, seed=6)
        small = NOBJ // 2
        path = os.path.join(d, 'small.npy')
        if RANK == 0:
            seed = np.ones((small, small), dtype='complex64')
            np.save(path, seed)
        comm.Barrier()
        out = cp.zeros((NOBJ, NOBJ), dtype='complex64')
        _reader(d, obj_init=path).read_obj(out)
        a = cp.asnumpy(out).real
        lo, hi = (NOBJ - small) // 2, (NOBJ - small) // 2 + small
        check('seed placed in the centre', bool(np.all(a[lo:hi, lo:hi] == 1)))
        check('margin left as vacuum',
              a[:lo, :].max() == 0 and a[hi:, :].max() == 0,
              f'edges max {a[:lo, :].max():.3g}')
        # cold start when the file is absent
        out2 = cp.ones((NOBJ, NOBJ), dtype='complex64')
        _reader(d, obj_init=os.path.join(d, 'nope.npy')).read_obj(out2)
        check('absent obj_init gives a cold start',
              float(cp.abs(out2).max()) == 0.0)
    finally:
        _rmtree(d)


def test_checkpoint_roundtrip():
    d = _tmpdir()
    try:
        _make_file(d, seed=7)
        st, end = _split(NFRAMES)
        rng = np.random.default_rng(11)
        vars = {
            'obj':  cp.asarray((rng.random((NOBJ, NOBJ))
                                + 1j * rng.random((NOBJ, NOBJ))).astype('complex64')),
            'prb':  cp.asarray((rng.random((NZ, N))
                                + 1j * rng.random((NZ, N))).astype('complex64')),
            'code': cp.asarray((rng.random((NCODE, NCODE))
                                + 1j * rng.random((NCODE, NCODE))).astype('complex64')),
            'pos':      (10 * rng.random((NFRAMES, 2))).astype('float32')[st:end],
            'pos_code': (10 * rng.random((NFRAMES, 2))).astype('float32')[st:end],
        }
        w = HoloCodedWriter(d, comm, NOBJ, NOBJ, st, end, NFRAMES, NZ, N, NCODE)
        w.write_checkpoint(vars, 0)
        comm.Barrier()
        path = os.path.join(d, 'checkpoints', 'checkpoint_0000.h5')
        check('checkpoint written', os.path.exists(path))

        r = _reader(d)
        o = cp.zeros_like(vars['obj'])
        p = cp.zeros_like(vars['prb'])
        c = cp.zeros_like(vars['code'])
        ps = np.zeros((end - st, 2), 'float32')
        pc = np.zeros((end - st, 2), 'float32')
        r.read_checkpoint(path, out_obj=o, out_prb=p, out_pos=ps,
                          out_code=c, out_pos_code=pc)
        for name, got, want in (('obj', o, vars['obj']),
                                ('code', c, vars['code'])):
            e = float(cp.abs(got - want).max())
            check(f'{name} round trips', e < 1e-5, f'max |d| {e:.3e}')
        # prb goes through abs/angle, so compare that way
        e = float(cp.abs(cp.abs(p) - cp.abs(vars['prb'])).max())
        check('prb magnitude round trips', e < 1e-5, f'{e:.3e}')
        for name, got, want in (('pos', ps, vars['pos']),
                                ('pos_code', pc, vars['pos_code'])):
            e = float(np.abs(got - want).max())
            check(f'{name} round trips', e < 1e-5, f'max |d| {e:.3e}')
    finally:
        _rmtree(d)


def test_checkpoint_upsamples_for_the_bin_ladder():
    """A bin-1 checkpoint read onto the bin-0 grid: every plane doubles and
    both position sets scale by 2.  Without this the shared-path_out ladder
    silently falls back to a cold start."""
    d = _tmpdir()
    try:
        _make_file(d, seed=8)
        st, end = _split(NFRAMES)
        rng = np.random.default_rng(12)
        half = dict(
            obj=cp.asarray((rng.random((NOBJ // 2, NOBJ // 2))).astype('complex64')),
            prb=cp.asarray((rng.random((NZ // 2, N // 2))).astype('complex64')),
            code=cp.asarray((rng.random((NCODE // 2, NCODE // 2))).astype('complex64')),
            pos=(3 * rng.random((NFRAMES, 2))).astype('float32')[st:end],
            pos_code=(3 * rng.random((NFRAMES, 2))).astype('float32')[st:end])
        w = HoloCodedWriter(d, comm, NOBJ // 2, NOBJ // 2, st, end, NFRAMES,
                            NZ // 2, N // 2, NCODE // 2)
        w.write_checkpoint(half, 0)
        comm.Barrier()
        path = os.path.join(d, 'checkpoints', 'checkpoint_0000.h5')

        r = _reader(d, bin=0)        # the FINE grid
        o = cp.zeros((NOBJ, NOBJ), 'complex64')
        p = cp.zeros((NZ, N), 'complex64')
        c = cp.zeros((NCODE, NCODE), 'complex64')
        ps = np.zeros((end - st, 2), 'float32')
        pc = np.zeros((end - st, 2), 'float32')
        r.read_checkpoint(path, out_obj=o, out_prb=p, out_pos=ps,
                          out_code=c, out_pos_code=pc)
        check('obj upsampled to the fine grid', o.shape == (NOBJ, NOBJ))
        check('code upsampled', c.shape == (NCODE, NCODE))
        check('prb upsampled', p.shape == (NZ, N))
        # nearest-neighbour repeat: every 2x2 block equals the coarse pixel
        e = float(cp.abs(o[0::2, 0::2] - half['obj']).max())
        check('obj upsample is a 2x repeat', e < 1e-6, f'{e:.3e}')
        e = float(np.abs(ps - 2 * half['pos']).max())
        check('pos scaled by 2', e < 1e-5, f'{e:.3e}')
        e = float(np.abs(pc - 2 * half['pos_code']).max())
        check('pos_code scaled by 2', e < 1e-5, f'{e:.3e}')
    finally:
        _rmtree(d)


def test_missing_keys_are_all_reported_at_once():
    """A layout mismatch should cost one run, not one run per key."""
    d = _tmpdir()
    try:
        if RANK == 0:
            with h5py.File(os.path.join(d, 'holo.h5'), 'w') as f:
                f.create_dataset(_LAYOUT['energy'], data=np.array([ENERGY]))
            np.save(os.path.join(d, 'code.npy'),
                    np.ones((NCODE, NCODE), 'complex64'))
        comm.Barrier()
        try:
            _reader(d)
            msg, raised = '', False
        except KeyError as e:
            msg, raised = str(e), True
        check('a short file is refused', raised)
        n = sum(1 for k in _LAYOUT if k not in ('prb', 'energy')
                and _LAYOUT[k] in msg)
        check('every missing key is named, not just the first', n >= 5,
              f'{n} of 6 named')
        check('the expected layout is printed', '_LAYOUT' in msg)
    finally:
        _rmtree(d)


if __name__ == '__main__':
    sys.exit(run(globals(), sys.argv[1:]))
