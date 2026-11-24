#!/usr/bin/env python
"""Demo 4 (coded aperture) -- single-distance 2-D holography through a mask.

    ./run.sh 04_coded_holography.py                  # 1 GPU
    ./run_mpi.sh 4 04_coded_holography.py            # 4 GPUs

Forward-simulates ONE propagation distance through a 2-D phase object with a
gold coded aperture in the beam, then recovers the object, the aperture and
BOTH position sets jointly with the Bilinear-Hessian solver
`RecHoloCoded`.  The 2-D sibling of `03_coded_holotomography.py`: same beam,
same geometry, same aperture, same solver family -- the difference is that the
sample is ONE plane rather than a volume, so there is no tomography, no angles
and no Paganin+FBP start.  The scan axis is FRAMES, each with its own object
shift and its own code shift.

Illumination diversity does the work that extra distances do in multi-distance
holography: the mask moves from frame to frame, so every frame sees a different
speckle pattern through the same object.

GEOMETRY.  The ESRF single-coded-aperture settings, as in demo 03:

    33.35 keV, detector at 1.28 m, sample at 4.214 mm, code at **-18 mm**

The code distance is NEGATIVE: the aperture sits upstream of the FOCAL SPOT, so
the code -> sample Fresnel distance comes out negative too.

THE OBJECT is `phantoms.star_projection`, a Siemens star as a weak phase
object (phase range [0, pi/8]).  THE PROBE is flat, so the aperture is the ONLY
source of illumination diversity and `prb` carries no error.

THE STARTING POINT.  Object = 0 (a cold start).  The code is the truth plus
some error -- a real run calibrates the aperture first, by near-field
ptychography on the mask alone -- and both position sets are off by up to
~0.5 px.  `--code-err 0` starts from the exact code; `--freeze-code`
additionally stops it being refined.

This script needs no reader: it forward-simulates straight into the solver's
buffers.  Results are written as one `.npz` by rank 0.
"""
import argparse
import os
import sys

import numpy as np
import cupy as cp
from mpi4py import MPI
from types import SimpleNamespace

from holotomocupy.rec_holo_coded_mpi import RecHoloCoded
from holotomocupy.logger_config import logger, set_log_level

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phantoms import coded_aperture, star_projection, thickness_for_phase  # noqa: E402

# ---- ESRF single-coded-aperture geometry (as in demo 03) -------------------
ENERGY = 33.35                        # keV
DET_PIXEL = 3 * 1.01250442970688e-6   # NATIVE detector pixel [m]
CROP = 2048                           # full detector width [native px]
Z_CODE = -18e-3                       # focus -> code; NEGATIVE = upstream of
                                      # the focal spot
Z_SAMPLE = 4.214e-3                   # focus -> sample
Z_DET = 1.28                          # focus -> detector

CA_FEATURE_UM = 1.0                   # fabricated cell size, at BIN=3
CA_THICKNESS_UM = 3.0                 # Au thickness; a realistic mask, not the
                                      # best one (t_pi = 6.49 um)
CA_ANGLE_DEG = 45.0                   # lattice rotation
CA_SUPERSAMPLE = 2                    # sub-samples per pixel, antialiases edges
OBJ_MARGIN = 64                       # nobj = n + this; must clear the object
                                      # shift plus the shift kernel's taps, or
                                      # _build_data_mask starts discarding real
                                      # measurements at the frame edge
OBJ_JITTER_PX = 30.0                  # object shift range, +- half of this
CODE_JITTER_FRAC = 1.0 / 8.0          # per-frame code shift, +- this * n

SHIFT = 'fft'          # one distance means m == 1 for the object shift, so
                       # 'fft' never takes the chirp-z: faster and exact here


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n', type=int, default=128,
                   help=f'detector size in pixels; {CROP} / n must be a power '
                        'of two. 128 (default) keeps the demo to a few minutes')
    p.add_argument('--nframes', type=int, default=64, help='scan frames')
    p.add_argument('--niter', type=int, default=257, help='BH iterations')
    p.add_argument('--model', default='amplitude', choices=('amplitude', 'intensity'),
                   help='data-misfit model; both take the measured INTENSITY')
    p.add_argument('--feature-um', type=float, default=None,
                   help='aperture cell size [um]; default keeps it at '
                        f'{CA_FEATURE_UM} um of code-plane width whatever n, '
                        'so it stays equally well sampled')
    p.add_argument('--thickness-um', type=float, default=CA_THICKNESS_UM,
                   help='Au thickness [um]')
    p.add_argument('--code-err', type=float, default=0.02,
                   help='error on the starting code, as a fraction of its '
                        'contrast (0 = start from the exact aperture)')
    p.add_argument('--freeze-code', action='store_true',
                   help='hold the aperture fixed; only its positions refine')
    p.add_argument('--nchunk', type=int, default=16, help='frames per GPU pass')
    p.add_argument('--out', default='./demo_out/rec_holo_coded',
                   help='output directory')
    return p.parse_args()


def geometry(n):
    """Every derived length, in phascade's convention: one shared index grid,
    a different physical pixel size at each plane."""
    ratio = CROP // n
    if n < 1 or CROP % n or (ratio & (ratio - 1)):
        raise SystemExit(f'--n {n}: {CROP} / n must be a power of two')
    bin_ = int(np.log2(ratio))
    det_px = DET_PIXEL * ratio
    plane_px = lambda z: det_px * z / Z_DET
    g = SimpleNamespace(
        n=n, bin=bin_, det_px=det_px,
        voxelsize=plane_px(Z_SAMPLE),
        code_px=abs(plane_px(Z_CODE)),
        # parallel-beam-equivalent Fresnel distances, z_ref = z_sample
        d1=Z_SAMPLE ** 2 * (1 / Z_CODE - 1 / Z_SAMPLE),     # code -> sample
        d2=Z_SAMPLE ** 2 * (1 / Z_SAMPLE - 1 / Z_DET),      # sample -> det
        wavelength=1.24e-09 / ENERGY)
    g.nobj = n + OBJ_MARGIN
    g.scan_px = int(np.ceil(CODE_JITTER_FRAC * n))
    q = max(64, n // 8)
    g.ncode = int(np.ceil((n + 2 * g.scan_px + 32) / q) * q)
    # Smallest alias-free grid per leg; Propagation pads n -> 2n, so the guard
    # is n/2 a side and Nyquist content deflects by lambda*D/(2p) px.
    g.min_grid = {k: g.wavelength * abs(d) / g.voxelsize ** 2
                  for k, d in (('code->sample', g.d1), ('sample->det', g.d2))}
    g.alias_margin = min(n / v for v in g.min_grid.values())
    return g


def _corr(a, b):
    """Normalised cross-correlation, accumulated in float64.

    The cast is not cosmetic. `np.vdot` on float32/complex64 accumulates in
    the input dtype, and over the ~4e6 elements of an n=2048 field the
    relative error reaches ~2e-4 -- enough to report a correlation ABOVE 1,
    which is how this was found. float64 costs nothing here (one reduction
    per run) and keeps the number meaningful at every grid size.
    """
    dt = 'complex128' if np.iscomplexobj(a) or np.iscomplexobj(b) else 'float64'
    a = a.ravel().astype(dt)
    b = b.ravel().astype(dt)
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.abs(np.vdot(a, b)) / d) if d else float('nan')


def main():
    a = parse()
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    set_log_level('INFO')

    g = geometry(a.n)
    n, nobj, ncode = g.n, g.nobj, g.ncode
    # Default the feature to a fixed number of CODE-PLANE PIXELS rather than a
    # fixed physical size, so coarsening the detector does not quietly push the
    # mask under Nyquist. At bin 3 this is the CA_FEATURE_UM of params.py.
    feature_m = (a.feature_um * 1e-6 if a.feature_um is not None
                 else CA_FEATURE_UM * 1e-6 * 2 ** (g.bin - 3))
    px_per_cell = feature_m / g.code_px

    if rank == 0:
        logger.info(f'n={n} (bin {g.bin}) nobj={nobj} ncode={ncode} '
                    f'nframes={a.nframes}  code scan +-{g.scan_px}px')
        logger.info(f'pixel size: code {g.code_px * 1e9:.1f} nm   '
                    f'sample {g.voxelsize * 1e9:.1f} nm   '
                    f'det {g.det_px * 1e9:.1f} nm')
        logger.info(f'D_eff: code->sample {g.d1 * 1e3:.4f} mm   '
                    f'sample->det {g.d2 * 1e3:.4f} mm   (D1 < 0: the code is '
                    f'upstream of the focal spot)')
        logger.info(f'aperture: {feature_m * 1e6:.2f} um cells = '
                    f'{px_per_cell:.2f} code px '
                    f'({"good" if px_per_cell >= 4 else "MARGINAL"}), '
                    f'{a.thickness_um:.2f} um Au')
        logger.info(f'alias margin {g.alias_margin:.2f}x '
                    f'({"clean" if g.alias_margin >= 2 else "MARGINAL"})')
        if px_per_cell < 2:
            raise SystemExit('aperture cells are sub-Nyquist on the code plane')

    obj = star_projection(nobj)
    prb = np.ones((n, n), dtype='complex64')   # flat: see the module docstring
    code = coded_aperture(ncode, feature_m, g.code_px, a.thickness_um * 1e-6,
                          ENERGY, angle_deg=CA_ANGLE_DEG,
                          supersample=CA_SUPERSAMPLE)
    if rank == 0:
        logger.info(f'code phase excursion {np.ptp(np.angle(code)):.3f} rad '
                    f'(t_pi = {thickness_for_phase(np.pi, ENERGY) * 1e6:.2f} um '
                    f'of Au at this energy)')

    rng = np.random.default_rng(10)
    nf = a.nframes
    pos          = (OBJ_JITTER_PX * (rng.random((nf, 2)) - 0.5)).astype('float32')
    pos_err      = (rng.random((nf, 2)) - 0.5).astype('float32')
    pos_code     = rng.uniform(-g.scan_px, g.scan_px, (nf, 2)).astype('float32')
    pos_code_err = (rng.random((nf, 2)) - 0.5).astype('float32')

    cl = RecHoloCoded(SimpleNamespace(
        energy=ENERGY, detector_pixelsize=g.det_px,
        focustodetectordistance=Z_DET, z1=Z_SAMPLE, z1_code=Z_CODE,
        ntheta=nf, nz=n, n=n, nzobj=nobj, nobj=nobj, ncode=ncode,
        rho=[1, 0.05, 0.02, 0.05, 0.02], code_pad_frac=0.125,
        shift_type=SHIFT, model=a.model, psf_sigma=0.0,
        freeze_code=a.freeze_code, freeze_pos_code=False,
        niter=a.niter, nchunk=a.nchunk,
        checkpoint_step=-1, error_step=max(a.niter // 8, 1),
        start_iter=0, comm=comm))

    # forward: obj, prb and code are replicated; each rank owns a slice of frames
    cl.vars['obj'][:]      = cp.asarray(obj)
    cl.vars['prb'][:]      = cp.asarray(prb)
    cl.vars['code'][:]     = cp.asarray(code)
    cl.vars['pos'][:]      = pos[cl.st_theta:cl.end_theta]
    cl.vars['pos_code'][:] = pos_code[cl.st_theta:cl.end_theta]
    cl.gen_data(cl.vars, cl.data)      # INTENSITY, what F0 compares against

    # inverse: a cold object, the exact flat probe (nothing to recover there),
    # a measured-but-imperfect aperture, and both position sets off by ~0.5 px
    cl.vars['obj'][:] = 0
    cl.vars['prb'][:] = 1
    code0 = code
    if a.code_err > 0:
        # Scaled by the code's own contrast, so --code-err reads as a fraction
        # whatever the thickness is set to.
        c = float(np.abs(code - code.mean()).std())
        code0 = (code + a.code_err * c
                 * (rng.standard_normal(code.shape)
                    + 1j * rng.standard_normal(code.shape))).astype('complex64')
    cl.vars['code'][:]     = cp.asarray(code0)
    cl.vars['pos'][:]      = (pos + pos_err)[cl.st_theta:cl.end_theta]
    cl.vars['pos_code'][:] = (pos_code + pos_code_err)[cl.st_theta:cl.end_theta]

    cl.BH()

    # --- the honest numbers: distance to the truth, not to the start --------
    def _mean(x):
        return comm.allreduce(float(x), MPI.SUM) / size

    p1 = _mean(np.abs(np.asarray(cl.vars['pos'])
                      - pos[cl.st_theta:cl.end_theta]).mean())
    p0 = _mean(np.abs(pos_err[cl.st_theta:cl.end_theta]).mean())
    c1 = _mean(np.abs(np.asarray(cl.vars['pos_code'])
                      - pos_code[cl.st_theta:cl.end_theta]).mean())
    c0 = _mean(np.abs(pos_code_err[cl.st_theta:cl.end_theta]).mean())

    if rank == 0:
        got = cp.asnumpy(cl.vars['obj'])
        code_got = cp.asnumpy(cl.vars['code'])
        # Score where the frames actually look: the central n x n, minus the
        # shift range -- outside it the reconstruction is zero by construction
        # while the phantom is not. Real part: the imaginary part is
        # beta/delta of the signal and scoring it only adds noise.
        m = int(np.ceil(OBJ_JITTER_PX / 2)) + 2
        sl = (slice(nobj // 2 - n // 2 + m, nobj // 2 + n // 2 - m),) * 2
        logger.info(f'object corr      {_corr(got.real[sl], obj.real[sl]):.4f}')
        logger.info(f'code   corr      {_corr(code_got, code):.4f}  '
                    f'(started at {_corr(code0, code):.4f})')
        logger.info(f'pos      error   {p1:.4f} px (started at {p0:.4f})')
        logger.info(f'pos_code error   {c1:.4f} px (started at {c0:.4f})')
        os.makedirs(a.out, exist_ok=True)
        np.savez(os.path.join(a.out, 'result.npz'), obj=got, code=code_got,
                 obj_true=obj, code_true=code)
        logger.info(f'wrote {os.path.abspath(a.out)}/result.npz')


if __name__ == '__main__':
    main()
