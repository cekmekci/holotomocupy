#!/usr/bin/env python
"""
Step 7 — per-angle drift refinement by entropy autofocus, coded aperture.

    python step7.py config_step6_bin1_pass1.conf [options]           one GPU
    mpiexec -n 8 ... set_affinity_gpu_polaris.sh python step7.py ...  eight

Same driver as `AtomiumS1_HT_RD300/step7.py`, over a `RecCoded` checkpoint
instead of `Rec`'s. Everything below `read_slab` is generic over the engine
-- `holotomocupy.autofocus` only ever sees a reconstructed object slab and a
sinogram, so nothing here is coded-aperture specific except the config
parser and the search-grid-to-file-units scale factor.

Under MPI the z slices are split over the ranks and the 256-bin histogram is
allreduced, so every rank runs the same Nelder-Mead on the same numbers and
only rank 0 writes. The answer does not depend on the rank count.

Takes the step-6 checkpoint, re-projects a slab of it, and searches for the
per-angle shift that minimises the entropy of the FBP. Writes
`correct_correct3D_extra.txt` NEXT TO THE CONFIG (--out overrides), which is
exactly where step 6 looks for it; step 6 adds it to the positions it reads,
so the next run is a step-6 rerun and nothing else.

Run this against the PASS-1 checkpoint (`config_step6_bin1_pass1.conf`,
iteration 768), then rerun the full ladder in PASS 2 with
`correct3d_extra_file` pointed at the file this writes.

THETA SIGN -- the one thing fixed here, and it matches the advisor's own
coded repo rather than departing from it.  `autofocus.scan_theta` negates
theta because the MULTI-DISTANCE step 5 reconstructs at -theta_raw
(reader.py:193).  This pipeline does not negate anywhere: steps15.py stage 5
uses `np.deg2rad(theta_deg)` and CodedReader uses
`np.deg2rad(theta_deg[self.ids])`, both +theta.  So `--theta-sign` defaults
to +1 here and the negation is undone -- exactly as
holotomo-coded-atomiums2/poly_shifts/find_shifts.py defaults its own
`--theta-sign` to 1.0 for the same reason.  R_{-t} u = R_{+t}(M u) for a
reflection M that entropy cannot see, so the wrong sign fits the MIRRORED
object's drift and fails silently rather than loudly.

WHAT THIS MEASURES.  `d = project_chunked(cl, obj, nzc)` forward-projects the
checkpoint object, so the sinogram it searches came from one consistent
volume.  Per the advisor's own write-up of the same algorithm
(holotomo-coded-atomiums2/poly_shifts/README.md, "what it does and does not
do"), the truth there is zero drift by construction, and what comes back is
the drift the method invents on this sample rather than the scan's motion.
That README also notes what a real residual would need -- measured
projections in place of that one line -- which nothing here does.
"""
import argparse
import glob
import os
import sys
import time

import h5py
import numpy as np
import cupy as cp

from holotomocupy.autofocus import (EXTRA_NAME, Focus, ShiftFourier,
                                    fit_drift, grey_range,
                                    project_chunked, rigid_part, scan_theta,
                                    summary_png, write_correct3d_extra,
                                    z_split)
from holotomocupy.config import parse_args_coded
from holotomocupy.tomo import Tomo


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('config', help='the step-6 config of the run to refine')
    p.add_argument('--iter', type=int, default=0,
                   help='checkpoint to read (0 = the highest one present)')
    p.add_argument('--ntheta', type=int, default=None,
                   help='angles to search at, on an even stride '
                        '(default: a quarter of the scan)')
    p.add_argument('--nslice', type=int, default=64,
                   help='centred z slices; every evaluation reconstructs all '
                        'of them, so this is the cost knob')
    p.add_argument('--bin', type=int, default=0,
                   help='extra x/y binning LEVEL on top of the checkpoint, '
                        'factor 2**bin as everywhere else: 0 = none (default), '
                        '1 = 2x2, 2 = 4x4')
    p.add_argument('--deg', type=int, default=9, help='Legendre degree')
    p.add_argument('--maxfev', type=int, default=400,
                   help='Nelder-Mead budget PER degree')
    p.add_argument('--zchunk', type=int, default=32,
                   help='slices reconstructed at once; sets peak memory, not '
                        'the answer')
    p.add_argument('--out', default=None,
                   help='where the files land (default: next to the config, which is where step 6 looks)')
    p.add_argument('--theta-sign', type=int, choices=(1, -1), default=1,
                   help='+1 (default) matches THIS pipeline: steps15.py '
                        'stage 5 and CodedReader both reconstruct at '
                        '+deg2rad(theta), and find_shifts.py in '
                        'holotomo-coded-atomiums2 defaults to +1 likewise. '
                        '-1 is the multi-distance convention (reader.py '
                        'negates); searching at the wrong sign fits the '
                        'drift of the MIRRORED object and fails silently.')
    p.add_argument('--score-radius', type=float, default=1.0,
                   help='fraction of the inscribed-cylinder radius that is '
                        'SCORED. 1.0 (default) is autofocus.cylinder_mask '
                        'exactly, i.e. the advisor\'s behaviour. Lower it to '
                        'exclude rotationally-symmetric artefacts near the '
                        'rim, which no per-angle drift can sharpen but a '
                        'runaway one can smear flat: with nobj=2560 the '
                        'measured data only reaches r = n/nobj = 0.80 and '
                        'everything beyond is pad_taper, and stage 5\'s '
                        'mask_r=0.9 apodises at r = sqrt(0.9) = 0.949 '
                        '(tomo.py:62 tests a SQUARED radius). 0.75 scores '
                        'real data only.')
    p.add_argument('--dry-run', action='store_true',
                   help='print the geometry and the units, then stop')
    return p.parse_args()


def mpi_comm():
    """COMM_WORLD when launched under mpiexec with >1 rank, else None.

    Keyed on the launcher's env so a plain `python step7.py` never initialises
    MPI and behaves exactly as it did before.
    """
    if not any(k in os.environ for k in
               ('PMI_RANK', 'PMIX_RANK', 'OMPI_COMM_WORLD_RANK')):
        return None
    from mpi4py import MPI
    return MPI.COMM_WORLD if MPI.COMM_WORLD.size > 1 else None


def latest_checkpoint(path_out, want):
    d = os.path.join(path_out, 'checkpoints')
    if want:
        p = os.path.join(d, f'checkpoint_{want:04d}.h5')
        if not os.path.exists(p):
            raise SystemExit(f'{p}: no such checkpoint')
        return p, want
    found = sorted(glob.glob(os.path.join(d, 'checkpoint_[0-9]*.h5')))
    if not found:
        raise SystemExit(f'{d}: no checkpoints')
    return found[-1], int(os.path.basename(found[-1])[11:15])


def checkpoint_shape(ckpt):
    with h5py.File(ckpt, 'r') as f:
        return f['obj_re'].shape


def scored_cylinder(n, nz, frac):
    """[nz, n, n] bool: the inscribed cylinder, shrunk to `frac` of its radius.

    `frac=1.0` is `autofocus.cylinder_mask` exactly.  Shrinking it matters
    because entropy is a property of the WHOLE scored histogram, and this
    seed carries two strong rotationally-symmetric features near the rim: a
    bowl bottoming out at r = n/nobj = 0.80 (the measured-data edge, beyond
    which stage 5's pad_taper invents the signal) and a bright ring at
    r = sqrt(mask_r) = 0.949 from stage 5's FBP apodisation.  Neither can be
    sharpened by any per-angle drift -- they are the same at every angle --
    but a runaway drift CAN smear them flat, which is worth ~1.6 nats
    against ~0.004 nats for genuinely focusing the sample.  Scoring only
    r < 0.75 leaves the optimiser nothing to vandalise.
    """
    yy, xx = np.mgrid[0:n, 0:n] - (n - 1) / 2.0
    return cp.broadcast_to(
        cp.asarray(np.sqrt(yy**2 + xx**2) < frac * n / 2), (nz, n, n))


def read_slab(ckpt, nb, nbz, nslice):
    """A centred z slab of obj_re, binned nbz in z and nb in x/y.

    nbz = nb (tomo_upsample is always 1 for the coded engine, m=1) makes the
    search grid isotropic -- the object voxel is one projection-plane pixel
    in every axis here, unlike the magnified multi-distance case.
    """
    with h5py.File(ckpt, 'r') as f:
        ds = f['obj_re']
        nzobj, nobj = ds.shape[0], ds.shape[1]
        ny = nobj // nb
        nz = min(nslice, nzobj // nbz)
        z0 = (nzobj // nbz - nz) // 2
        out = np.empty((nz, ny, ny), dtype='float32')
        for i in range(nz):
            raw = np.asarray(ds[(z0 + i) * nbz:(z0 + i + 1) * nbz,
                                :ny * nb, :ny * nb], dtype='float32')
            out[i] = raw.mean(axis=0).reshape(ny, nb, ny, nb).mean(axis=(1, 3))
    return out


def main():
    a = parse()
    comm = mpi_comm()
    rank, size = (0, 1) if comm is None else (comm.rank, comm.size)
    log = print if rank == 0 else (lambda *x, **k: None)
    # set_affinity_gpu_polaris.sh pins one GPU per rank; without it the ranks
    # would all pile onto device 0
    if comm is not None and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        cp.cuda.Device(rank % cp.cuda.runtime.getDeviceCount()).use()

    # default --out is the config's own directory, which is where step 6
    # looks for correct_correct3D_extra.txt
    if a.out is None:
        a.out = os.path.dirname(os.path.abspath(a.config)) or '.'
    cfg = parse_args_coded(a.config)
    if rank == 0:
        os.makedirs(a.out, exist_ok=True)

    ckpt, it = latest_checkpoint(cfg.path_out, a.iter)
    up = cfg.tomo_upsample      # always 1 for the coded engine (m=1)
    # The ladder shares path_out, so the newest checkpoint is not necessarily
    # at this config's bin. Take the grid from the checkpoint itself.
    nzck, nck = checkpoint_shape(ckpt)[:2]
    nb = 2 ** max(0, a.bin)
    n = nck // nb
    nbz = nb * up
    nz = min(a.nslice, nzck // nbz)
    if size > nz:
        raise SystemExit(f'{size} ranks for {nz} z slices; run with fewer')
    zlo, zhi = z_split(nz, rank, size)
    nzc = min(a.zchunk or (zhi - zlo), zhi - zlo)

    # One search pixel in UNBINNED object px (= unbinned detector px, m=1, so
    # there is no separate raw-detector-vs-object-plane scale to track, unlike
    # the magnified multi-distance case). nobj0 is nobj at bin 0.
    nobj0 = cfg.nobj * 2 ** cfg.bin
    scale = nobj0 * up / n

    with h5py.File(cfg.in_file, 'r') as f:
        ntheta_scan = len(f['/exchange/theta'])
    if a.ntheta is None:
        a.ntheta = max(1, ntheta_scan // 4)
    theta_deg, theta_neg = scan_theta(cfg.in_file, a.ntheta)
    # scan_theta always negates, for the multi-distance convention. Undo that
    # unless -1 was asked for: R_{-t} u = R_{+t}(M u) for a reflection M that
    # entropy cannot see, so the wrong sign fits the MIRRORED object's drift
    # and the curve does not transfer back. This pipeline is +theta.
    theta = theta_neg if a.theta_sign == -1 else -theta_neg

    log(f'config   {a.config}')
    log(f'  object {cfg.nzobj} x {cfg.nobj}^2 at bin {cfg.bin}  '
        f'({nobj0} x/y at bin 0)')
    log(f'  ckpt   {ckpt}  (iteration {it}, {nzck} x {nck}^2)')
    log(f'search   {a.ntheta} of {ntheta_scan} angles, grid {n} '
        f'(bin {nb} in x/y, {nbz} in z), {nz} slices, '
        f'{nzc} per chunk, degree {a.deg}')
    log(f'ranks    {size} over {nz} z slices ({zhi - zlo} on rank 0); the '
        f'sinogram and the shift are replicated, the FBP is not')
    log(f'angles   {theta_deg[0]:.4f}..{theta_deg[-1]:.4f} deg from '
        f'{cfg.in_file}::/exchange/theta, sign {a.theta_sign:+d} '
        f'({"NOT negated, matching stage 5 and CodedReader" if a.theta_sign == 1 else "negated, multi-distance convention"})')
    log(f'units    1 search px = {scale:g} unbinned object px')
    if a.dry_run:
        return

    t0 = time.time()
    obj = read_slab(ckpt, nb, nbz, a.nslice)
    log(f'read     {obj.shape} in {time.time()-t0:.1f} s')

    cl = Tomo(n, nzc, theta, -1.0)
    sh = ShiftFourier(nz, n)
    sel = scored_cylinder(n, nzc, a.score_radius)
    log(f'scored   r < {a.score_radius:g} of the inscribed cylinder '
        f'({100 * a.score_radius**2:.0f}% of its area)'
        + ('  [= autofocus.cylinder_mask, the advisor\'s region]'
           if a.score_radius == 1.0 else
           '  [trimmed: see --score-radius]'))
    t0 = time.time()
    d = project_chunked(cl, obj, nzc)
    cp.cuda.Stream.null.synchronize()
    del obj
    log(f'project  {tuple(d.shape)} in {time.time()-t0:.1f} s')

    t0 = time.time()
    lo, hi = grey_range(cl, d, sel, nzc)
    if comm is not None:
        lo, hi = comm.bcast((lo, hi), root=0)
    log(f'score    entropy, 256 bins in [{lo:.4g}, {hi:.4g}] from the '
        f'uncorrected FBP ({time.time()-t0:.1f} s)')

    fc = Focus(cl, sh, d, theta, sel, lo, hi, nzc, comm=comm)
    q0, mid0 = fc.sweep(np.zeros((a.ntheta, 2), 'float32'), want_mid=True)
    log(f'         entropy at no correction {q0:.5f}')

    t0 = time.time()
    log(f'Nelder-Mead from zero, degree 1 to {a.deg}, '
        f'<= {a.maxfev} evaluations per degree:')
    fc.trace.clear()
    coef, rungs = fit_drift(fc, a.deg, maxfev=a.maxfev, log=log)
    s = fc.s_of(np.linalg.lstsq(fc.V, coef.ravel(), rcond=None)[0])
    q1, mid1 = fc.sweep(s, want_mid=True)
    log(f'  {len(fc.trace)} evaluations, {fc.nit} iterations, '
        f'{time.time()-t0:.1f} s;  entropy {q1:.5f}, '
        f'{q1 - q0:+.5f} against no correction (negative is better)')
    log(f'  found    y ptp {np.ptp(s[:, 0]):.3f}  x ptp {np.ptp(s[:, 1]):.3f} '
        f'search px')
    log('  of which a rigid object shift of ' + '/'.join(
        f'{v:+.2f}' for v in rigid_part(s, fc.G)[2])
        + ' px in z/y/x, which no tomogram can see')

    if rank != 0:
        return
    tag = os.path.join(a.out, f'step7_it{it}_n{n}_th{a.ntheta}_nz{nz}')
    np.savetxt(f'{tag}_shifts.csv', np.column_stack([theta_deg, s]),
               delimiter=',', header='theta_deg,sy,sx', comments='')
    np.savetxt(f'{tag}_trace.csv', np.array(fc.trace), delimiter=',',
               header='entropy,identifiable_px,' + ','.join(
                   f'c{k}{ax}' for k in range(a.deg + 1) for ax in 'yx'),
               comments='')
    summary_png(f'{tag}.png', fc, theta_deg, s, (mid0, mid1, None),
                title=f'step 7, {os.path.basename(cfg.path_out)} '
                      f'iteration {it}: {len(fc.trace)} evaluations')

    # ShiftFourier applies s as a periodic FFT phase ramp (exp(-2pi*i*f*s)),
    # so s and s +/- n*search-grid-width score IDENTICALLY once the optimizer
    # already has what little real signal there is -- a flat direction
    # Nelder-Mead is free to wander in without the convergence tests ever
    # firing. Fine downstream of fit_drift, where only the (wrapped) score is
    # used; not fine here: write_correct3d_extra writes the RAW, un-wrapped
    # value, and CodedReader.read_pos adds it to object positions with no
    # periodicity at all, so a search that wandered 400 periods away corrupts
    # every position by that same 400-period offset. Caught here rather than
    # downstream, where it would otherwise surface as a inexplicable
    # ZeroDivisionError deep inside the next step-6 rung's first iteration.
    found_ptp = max(np.ptp(s[:, 0]), np.ptp(s[:, 1]))
    if found_ptp > 2 * n:
        raise SystemExit(
            f'step7: found shift spans {found_ptp:.0f} search px on a {n} px '
            f'grid -- the optimizer wandered off (see {tag}_trace.csv: '
            f'degree-by-degree entropy should DECREASE each rung; "maximum '
            f'number of function evaluations exceeded" with no improvement '
            f'means the higher degrees have no real signal to fit). Lower '
            f'--deg, or use more angles/finer --bin, rather than writing '
            f'this out -- CodedReader.read_pos adds it to positions with no '
            f'wraparound, so this would corrupt the next rung silently.')

    extra = os.path.join(a.out, EXTRA_NAME)
    s_out = write_correct3d_extra(extra, coef, a.ntheta, ntheta_scan, scale)
    print(f'wrote    {os.path.abspath(extra)}  ({ntheta_scan + 1} rows, '
          f'columns horizontal,vertical)')
    print(f'         file px: y ptp {np.ptp(s_out[:, 0]):.3f}  '
          f'x ptp {np.ptp(s_out[:, 1]):.3f}')
    print(f'         {tag}_shifts.csv  {tag}_trace.csv  {tag}.png')
    print('next     rerun step 6 (pass 2); it reads the file from this directory')


if __name__ == '__main__':
    sys.exit(main())
