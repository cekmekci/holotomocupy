#!/usr/bin/env python
"""`reader_coded` and the step6 driver path, against a synthetic input set.

    python tests/coded/test_io_coded.py
    mpirun -np 2 python tests/coded/test_io_coded.py

Builds a `codedHT.h5` + `code.npy` + `obj_init.npy` in a temp directory
matching the layout documented in `reader_coded`, then runs the whole chain a
driver runs: config -> reader -> RecCoded -> BH -> checkpoint -> resume.

This is the only coverage the I/O gets: the real inputs live on the beamline
filesystem and are not touched from here.  So what it proves is that the code
paths are consistent with the DOCUMENTED layout -- binning, the intensity
convention, the position scaling, the checkpoint round trip -- not that the
layout matches what the preprocessing actually writes.  One run against a real
file is still needed for that.
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
from holotomocupy.rec_coded_mpi import RecCoded                   # noqa: E402
from holotomocupy.config import parse_args_coded                  # noqa: E402
from holotomocupy.mpi_functions import MPIClass                   # noqa: E402
from holotomocupy.reader_coded import (CodedReader, CodedWriter,   # noqa: E402
                                        find_latest_checkpoint)

COMM = MPI.COMM_WORLD

# Unbinned sizes on "disk"; BIN=1 halves every length.
NZ_UN, N_UN, NCODE_UN, NOBJ_UN = 32, 32, 64, 48
NTHETA = 8
BIN = 1
NZ, N, NCODE, NOBJ = NZ_UN // 2, N_UN // 2, NCODE_UN // 2, NOBJ_UN // 2


def _make_inputs(d):
    """Write a synthetic input set, rank 0 only, and return the true arrays."""
    rng = np.random.default_rng(0)
    pdata = (rng.random((NTHETA, NZ_UN, N_UN)).astype('float32') + 0.5)
    prb = (rng.standard_normal((NZ_UN, N_UN))
           + 1j * rng.standard_normal((NZ_UN, N_UN))).astype('complex64')
    code = (rng.standard_normal((NCODE_UN, NCODE_UN))
            + 1j * rng.standard_normal((NCODE_UN, NCODE_UN))).astype('complex64')
    obj = (rng.standard_normal((NOBJ_UN, NOBJ_UN, NOBJ_UN)) * 0.01
           + 1j * rng.standard_normal((NOBJ_UN, NOBJ_UN, NOBJ_UN)) * 0.01
           ).astype('complex64')
    theta_deg = np.linspace(0, 180, NTHETA, endpoint=False).astype('float64')
    shifts_cor = (4 * (rng.random((NTHETA, 2)) - 0.5)).astype('float32')
    shifts_code = (8 * (rng.random((NTHETA, 2)) - 0.5)).astype('float32')

    if COMM.rank == 0:
        with h5py.File(os.path.join(d, 'codedHT.h5'), 'w') as f:
            g = f.create_group('exchange')
            g['energy'] = np.array([17.1])
            g['detector_pixelsize'] = np.array([1.4760147601476e-6 * 16])
            g['focusdetectordistance'] = np.array([1.217])
            g['z1'] = np.array([5.110e-3])
            g['theta'] = theta_deg
            g['shifts_cor'] = shifts_cor
            g['shifts_code'] = shifts_code
            g['pdata'] = pdata
            g['prb'] = prb
        np.save(os.path.join(d, 'code.npy'), code)
        np.save(os.path.join(d, 'obj_init.npy'), obj)
    COMM.Barrier()
    return dict(pdata=pdata, prb=prb, code=code, obj=obj,
                theta_deg=theta_deg, shifts_cor=shifts_cor,
                shifts_code=shifts_code)


def _bin_xy(a, nbin, axes):
    for _ in range(nbin):
        for ax in axes:
            s0 = [slice(None)] * a.ndim; s0[ax] = slice(0, None, 2)
            s1 = [slice(None)] * a.ndim; s1[ax] = slice(1, None, 2)
            a = 0.5 * (a[tuple(s0)] + a[tuple(s1)])
    return a


def _reader(d, mpi):
    return CodedReader(
        os.path.join(d, 'codedHT.h5'), os.path.join(d, 'code.npy'), COMM,
        mpi.st_obj, mpi.end_obj, NOBJ, NOBJ,
        mpi.st_theta, mpi.end_theta, NTHETA,
        NZ, N, NCODE, BIN,
        obj_init_file=os.path.join(d, 'obj_init.npy'))


def _config(d, path_out, **over):
    """Write a config file and parse it, so parse_args_coded is on the path
    this test covers rather than a hand-built namespace."""
    body = dict(path=d, in_file='codedHT.h5', code_file='code.npy',
                obj_init_file='obj_init.npy', path_out=path_out,
                bin=BIN, nchunk=4, ntheta=NTHETA, start_theta=0, stride=1,
                niter=3, start_iter=0, checkpoint_step=2, error_step=1,
                nz=NZ, n=N, nzobj=NOBJ, nobj=NOBJ, ncode=NCODE, mask=0.9,
                z1_code=4.584e-3, code_pad_frac=0.125, model='amplitude',
                psf_sigma=0, lam_laplacian=0, rho='1,0.03,0.01,0.03,0.01',
                shift_type='fft', log_level='ERROR')
    body.update(over)
    path = os.path.join(d, 'cfg.conf')
    if COMM.rank == 0:
        with open(path, 'w') as f:
            for k, v in body.items():
                f.write(f'{k}={v}\n')
    COMM.Barrier()
    return parse_args_coded(path)


# ----------------------------------------------------------------------

def test_reader_against_the_documented_layout():
    d = tempfile.mkdtemp(prefix='htc_coded_') if COMM.rank == 0 else None
    d = COMM.bcast(d, root=0)
    try:
        truth = _make_inputs(d)
        mpi = MPIClass(COMM, NOBJ, NTHETA, NOBJ, 'complex64')
        r = _reader(d, mpi)

        close('energy', r.energy, 17.1, 1e-12)
        close('detector_pixelsize scaled by 2**bin',
              r.detector_pixelsize, 1.4760147601476e-6 * 16 * 2 ** BIN, 1e-12)
        check('theta converted to radians',
              np.allclose(r.theta, np.deg2rad(truth['theta_deg']), atol=1e-5),
              f'{r.theta[:3]} vs {np.deg2rad(truth["theta_deg"])[:3]}')

        # data: binned, INTENSITY (no sqrt)
        out = np.empty([mpi.end_theta - mpi.st_theta, NZ, N], dtype='float32')
        r.read_data(out=out)
        want = _bin_xy(truth['pdata'].astype('float64'), BIN, (1, 2))
        want = want[mpi.st_theta:mpi.end_theta]
        e = float(np.abs(out - want).max() / np.abs(want).max())
        check('data is binned intensity, not its sqrt', e < 1e-6,
              f'max rel {e:.2e}')

        # probe and code: binned
        prb = np.empty([NZ, N], dtype='complex64')
        r.read_prb(out=prb)
        wprb = _bin_xy(truth['prb'].astype('complex128'), BIN, (0, 1))
        check('probe binned', float(np.abs(prb - wprb).max()) < 1e-5)

        code = cp.zeros([NCODE, NCODE], dtype='complex64')
        r.read_code(out=code)
        wcode = _bin_xy(truth['code'].astype('complex128'), BIN, (0, 1))
        check('code binned to [ncode, ncode]',
              float(cp.abs(code - cp.asarray(wcode)).max()) < 1e-5,
              f'shape {code.shape}')

        # positions: this rank's angles, divided by 2**bin
        pos = r.read_pos()
        wpos = truth['shifts_cor'][mpi.st_theta:mpi.end_theta] / 2 ** BIN
        check('pos sliced per rank and scaled by 1/2**bin',
              np.allclose(pos, wpos, atol=1e-6))
        posc = r.read_pos_code()
        wposc = truth['shifts_code'][mpi.st_theta:mpi.end_theta] / 2 ** BIN
        check('pos_code sliced per rank and scaled by 1/2**bin',
              np.allclose(posc, wposc, atol=1e-6))

        # object: binned isotropically in all three axes
        obj = np.zeros([mpi.end_obj - mpi.st_obj, NOBJ, NOBJ], dtype='complex64')
        r.read_obj(out=obj)
        wobj = _bin_xy(truth['obj'].astype('complex128'), BIN, (0, 1, 2))
        wobj = wobj[mpi.st_obj:mpi.end_obj]
        e = float(np.abs(obj - wobj).max() / max(np.abs(wobj).max(), 1e-30))
        check('initial object binned in all 3 axes', e < 1e-4,
              f'max rel {e:.2e}')
    finally:
        COMM.Barrier()
        if COMM.rank == 0:
            shutil.rmtree(d, ignore_errors=True)


def test_missing_obj_init_is_a_cold_start():
    """A cold start is legitimate here; it must not raise."""
    d = tempfile.mkdtemp(prefix='htc_coded_') if COMM.rank == 0 else None
    d = COMM.bcast(d, root=0)
    try:
        _make_inputs(d)
        mpi = MPIClass(COMM, NOBJ, NTHETA, NOBJ, 'complex64')
        r = CodedReader(
            os.path.join(d, 'codedHT.h5'), os.path.join(d, 'code.npy'), COMM,
            mpi.st_obj, mpi.end_obj, NOBJ, NOBJ,
            mpi.st_theta, mpi.end_theta, NTHETA,
            NZ, N, NCODE, BIN, obj_init_file=None)
        obj = np.ones([mpi.end_obj - mpi.st_obj, NOBJ, NOBJ], dtype='complex64')
        r.read_obj(out=obj)
        check('no obj_init -> zeroed object, no error',
              float(np.abs(obj).max()) == 0.0)
    finally:
        COMM.Barrier()
        if COMM.rank == 0:
            shutil.rmtree(d, ignore_errors=True)


def test_driver_round_trip():
    """The whole step6 path: config -> reader -> RecCoded -> BH -> checkpoint
    -> resume.  The resumed state must match what was written."""
    d = tempfile.mkdtemp(prefix='htc_coded_') if COMM.rank == 0 else None
    d = COMM.bcast(d, root=0)
    try:
        _make_inputs(d)
        out_dir = os.path.join(d, 'rec')
        args = _config(d, out_dir)
        args.comm = COMM

        mpi = MPIClass(COMM, args.nzobj, args.ntheta, args.nobj, 'complex64')
        reader = CodedReader(
            args.in_file, args.code_file, COMM,
            mpi.st_obj, mpi.end_obj, args.nzobj, args.nobj,
            mpi.st_theta, mpi.end_theta, args.ntheta,
            args.nz, args.n, args.ncode, args.bin,
            obj_init_file=args.obj_init_file,
            start_theta=args.start_theta, stride=args.stride)
        writer = CodedWriter(
            args.path_out, COMM, mpi.st_obj, mpi.end_obj, args.nzobj, args.nobj,
            mpi.st_theta, mpi.end_theta, args.ntheta,
            args.nz, args.n, args.ncode)

        args.energy = reader.energy
        args.focustodetectordistance = reader.focustodetectordistance
        args.z1 = reader.z1
        args.detector_pixelsize = reader.detector_pixelsize
        args.theta = reader.theta

        cl = RecCoded(args)
        reader.read_data(out=cl.data)
        reader.read_obj(out=cl.vars['obj'])
        reader.read_prb(out=cl.vars['prb'])
        reader.read_code(out=cl.vars['code'])
        reader.read_pos(out=cl.vars['pos'])
        reader.read_pos_code(out=cl.vars['pos_code'])
        check('driver: inputs loaded', True)

        cl.BH(writer)
        # niter=3, checkpoint_step=2 -> periodic checkpoints at i = 0 and
        # i = 2 (range(0,3) never reaches i=3, so no choice of
        # checkpoint_step can land one there), PLUS the unconditional
        # end-of-run checkpoint _iterate always writes at i = niter = 3 --
        # that one is what the next rung's start_iter=niter actually resumes
        # from; without it that handoff silently falls back to a cold start.
        # find_latest_checkpoint only looks when start_iter > 0 (his
        # convention: resuming is explicit), so the resume iteration is
        # named here rather than discovered.
        written = sorted(os.listdir(os.path.join(args.path_out, 'checkpoints')))
        check('driver: BH ran and wrote checkpoints',
              written == ['checkpoint_0000.h5', 'checkpoint_0002.h5',
                          'checkpoint_0003.h5'],
              f'{written}')

        ckpt_handoff = find_latest_checkpoint(args.path_out, args.niter)
        check('find_latest_checkpoint locates the end-of-run handoff point',
              ckpt_handoff is not None and ckpt_handoff.endswith('checkpoint_0003.h5'),
              f'{ckpt_handoff}')

        # postcalc has already undone the 1/norm_const, so vars are in
        # driver units -- the same units the checkpoint stores.
        want = {k: (v.get() if isinstance(v, cp.ndarray) else np.array(v))
                for k, v in cl.vars.items() if k != 'proj'}

        ckpt = find_latest_checkpoint(args.path_out, 2)
        check('find_latest_checkpoint locates the resume point',
              ckpt is not None and ckpt.endswith('checkpoint_0002.h5'), f'{ckpt}')
        got = {
            'obj': np.zeros_like(want['obj']),
            'prb': np.zeros_like(want['prb']),
            'pos': np.zeros_like(want['pos']),
            'pos_code': np.zeros_like(want['pos_code']),
        }
        code_buf = cp.zeros([args.ncode, args.ncode], dtype='complex64')
        reader.read_checkpoint(ckpt, out_obj=got['obj'], out_prb=got['prb'],
                               out_pos=got['pos'], out_code=code_buf,
                               out_pos_code=got['pos_code'])
        got['code'] = code_buf.get()

        for k in ('obj', 'prb', 'pos', 'code', 'pos_code'):
            w = want[k]
            scale = max(float(np.abs(w).max()), 1e-30)
            e = float(np.abs(got[k] - w).max() / scale)
            # prb/code go through an abs/angle round trip in the checkpoint,
            # so they carry a float32 polar conversion; the rest are exact.
            tol = 1e-5 if k in ('prb',) else 1e-6
            check(f'checkpoint round trip: {k}', e < tol, f'max rel {e:.2e}')
    finally:
        COMM.Barrier()
        if COMM.rank == 0:
            shutil.rmtree(d, ignore_errors=True)


def test_correct3d_extra_two_column():
    """`read_correct3d_extra` takes step7.py's layout, not the fork's.

    Peter's format -- columns (horizontal, vertical), one row per index --
    same as `Reader.read_correct3d_extra` for the multi-distance pipeline.
    Checks the column swap (file is x,y; CodedReader returns y,x) and that
    `read_pos` adds it AFTER scaling `shifts_cor` by `1/2**bin`, matching
    `Reader.read_pos`'s order.
    """
    d = tempfile.mkdtemp(prefix='htc_coded_') if COMM.rank == 0 else None
    d = COMM.bcast(d, root=0)
    try:
        truth = _make_inputs(d)
        mpi = MPIClass(COMM, NOBJ, NTHETA, NOBJ, 'complex64')

        rng = np.random.default_rng(1)
        # one extra row past ntheta, as write_correct3d_extra emits (ntheta+1
        # rows for ESRF's 0..180 inclusive grid); only the first ntheta are used
        extra_xy = rng.standard_normal((NTHETA + 1, 2)).astype('float32')
        extra_path = os.path.join(d, 'correct_correct3D_extra.txt')
        if COMM.rank == 0:
            np.savetxt(extra_path, extra_xy, fmt='%.8e')
        COMM.Barrier()

        r = CodedReader(
            os.path.join(d, 'codedHT.h5'), os.path.join(d, 'code.npy'), COMM,
            mpi.st_obj, mpi.end_obj, NOBJ, NOBJ,
            mpi.st_theta, mpi.end_theta, NTHETA,
            NZ, N, NCODE, BIN, obj_init_file=os.path.join(d, 'obj_init.npy'),
            correct3d_extra_file=extra_path)

        got = r.read_correct3d_extra()
        want_yx = extra_xy[mpi.st_theta:mpi.end_theta, ::-1] / 2 ** BIN
        check('read_correct3d_extra: column order (y,x) and bin scale',
              np.allclose(got, want_yx, atol=1e-6),
              f'max abs {np.abs(got - want_yx).max():.2e}')

        pos = r.read_pos()
        want_pos = (truth['shifts_cor'][mpi.st_theta:mpi.end_theta]
                   / 2 ** BIN) + want_yx
        check('read_pos: shifts_cor scaled by bin, extra added after',
              np.allclose(pos, want_pos, atol=1e-6),
              f'max abs {np.abs(pos - want_pos).max():.2e}')

        r_noextra = CodedReader(
            os.path.join(d, 'codedHT.h5'), os.path.join(d, 'code.npy'), COMM,
            mpi.st_obj, mpi.end_obj, NOBJ, NOBJ,
            mpi.st_theta, mpi.end_theta, NTHETA,
            NZ, N, NCODE, BIN, obj_init_file=os.path.join(d, 'obj_init.npy'))
        check('read_correct3d_extra: absent file -> None',
              r_noextra.read_correct3d_extra() is None)
    finally:
        COMM.Barrier()
        if COMM.rank == 0:
            shutil.rmtree(d, ignore_errors=True)


def test_read_obj_centers_a_smaller_seed():
    """read_obj must CENTER a seed smaller than nobj, not corner-anchor it.

    `test_reader_against_the_documented_layout`'s fixture always has
    nobj_bin == nobj exactly (no margin), so corner-anchoring and centering
    are indistinguishable there -- that is how this bug shipped. This test
    uses a real margin (nobj > the binned seed width), the AtomiumS2 case
    where obj_init.npy (2048 px) is smaller than the padded object grid.
    """
    d = tempfile.mkdtemp(prefix='htc_coded_') if COMM.rank == 0 else None
    d = COMM.bcast(d, root=0)
    try:
        _make_inputs(d)   # codedHT.h5 + code.npy; obj_init.npy overwritten below

        bin_ = 1
        nzobj_un, nobj_un = 8, 16          # unbinned seed -> binned (4, 8, 8)
        nzobj, nobj = 8, 24                # grid: margins on ALL THREE axes
        rng = np.random.default_rng(3)
        obj = (rng.standard_normal((nzobj_un, nobj_un, nobj_un))
              + 1j * rng.standard_normal((nzobj_un, nobj_un, nobj_un))
              ).astype('complex64')
        obj_path = os.path.join(d, 'obj_init.npy')
        if COMM.rank == 0:
            np.save(obj_path, obj)
        COMM.Barrier()

        mpi = MPIClass(COMM, nzobj, NTHETA, nobj, 'complex64')
        r = CodedReader(
            os.path.join(d, 'codedHT.h5'), os.path.join(d, 'code.npy'), COMM,
            mpi.st_obj, mpi.end_obj, nzobj, nobj,
            mpi.st_theta, mpi.end_theta, NTHETA,
            NZ, N, NCODE, bin_, obj_init_file=obj_path)

        got = np.zeros([mpi.end_obj - mpi.st_obj, nobj, nobj], dtype='complex64')
        r.read_obj(out=got)

        sz  = nzobj_un // 2 ** bin_         # 4  -- seed z extent, binned
        sxy = nobj_un // 2 ** bin_          # 8  -- seed x/y extent, binned
        zlo = (nzobj - sz) // 2             # 2  -- z margin per side
        lo  = (nobj - sxy) // 2             # 8  -- x/y margin per side
        wobj = _bin_xy(obj.astype('complex128'), bin_, (0, 1, 2))

        # Gather the full grid so the z check does not depend on how the
        # slab happens to be split across ranks.
        full = np.concatenate(COMM.allgather(got), axis=0)

        centre = full[zlo:zlo + sz, lo:lo + sxy, lo:lo + sxy]
        e = float(np.abs(centre - wobj).max() / max(np.abs(wobj).max(), 1e-30))
        check('read_obj: seed lands CENTERED in x, y AND z',
              e < 1e-4, f'max rel {e:.2e}')

        check('read_obj: the x/y margin is vacuum',
              float(np.abs(full[:, :lo, :lo]).max()) == 0.0,
              f'max abs {float(np.abs(full[:, :lo, :lo]).max()):.2e}')

        # The z margin is what the second bug got wrong: the seed was
        # anchored at z=0, leaving the vacuum all at the TOP of the grid
        # instead of split evenly, so every slab held the wrong cross-section.
        check('read_obj: the z margin is vacuum on BOTH sides, not just the top',
              float(np.abs(full[:zlo]).max()) == 0.0
              and float(np.abs(full[zlo + sz:]).max()) == 0.0,
              f'below {float(np.abs(full[:zlo]).max()):.2e}  '
              f'above {float(np.abs(full[zlo + sz:]).max()):.2e}')
    finally:
        COMM.Barrier()
        if COMM.rank == 0:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(run(globals(), sys.argv[1:]))
