#!/usr/bin/env python
"""Demo step 6 (coded aperture) — single-distance holotomography through a mask.

    ./run.sh 03_coded_holotomography.py              # 1 GPU
    ./run_mpi.sh 4 03_coded_holotomography.py        # 4 GPUs

Forward-simulates ONE propagation distance through a 3-D phantom with a gold
coded aperture in the beam, then recovers object, aperture and BOTH position
sets jointly with the Bilinear-Hessian solver. Same data, same reconstruction
as [`03_coded_holotomography.ipynb`](03_coded_holotomography.ipynb); this is
the parameterised, multi-GPU entry point next to it.

The difference from `02_holotomography.py` is the beam, not the sample. There,
four distances give the four different contrast transfer functions that make
the phase recoverable. Here there is one distance, but the mask shifts from
angle to angle, so every projection sees a different speckle pattern and that
diversity does the same job. The cascade is two levels deeper
(`rec_coded_mpi` has the diagram) and there are two more variables, `code` and
`pos_code`.

GEOMETRY.  The ESRF single-coded-aperture settings, taken from
`phascade/simulations/esrf_simulation/params.py`, which is the validated
simulation of this experiment:

    33.35 keV, detector at 1.28 m, sample at 4.214 mm, code at **-18 mm**

The code distance is NEGATIVE: the aperture sits upstream of the FOCAL SPOT,
not between the focus and the sample, so the code -> sample Fresnel distance
comes out negative too. `RecCoded`'s geometry block reproduces `phascade`'s
`Geometry` for it to machine precision.

THE APERTURE IS BUILT ON ITS OWN PLANE.  Each plane has a different physical
pixel size on the shared index grid -- 342 nm at the code against 80 nm at the
sample here. Building the mask on the sample voxel would misstate the
fabricated feature size by 4.3x, so `coded_aperture` takes the code-plane
pixel size explicitly.

THE PROBE IS FLAT.  As in `phascade`'s own baseline (`PROBE_NPY = None`): a
unit plane wave, so the aperture is the ONLY source of illumination
diversity. Truth and starting guess are both exactly 1, so `prb` carries no
error and nothing is asked of it here; demo 2 is where a structured probe is
recovered.

WHERE THE STARTING POINT COMES FROM.  A real coded run calibrates the aperture
first, by near-field ptychography on the mask alone -- that is what
`01_nfp_probe.py` does for a probe. So the honest start is a MEASURED code,
the truth plus some error, rather than a flat one. `--code-err 0` starts from
the exact code; `--freeze-code` additionally stops it being refined.

The object DOES get a Paganin-style start. `multi_paganin` needs several
distances only to invert several different contrast-transfer functions at
once; with one it reduces to the textbook single-distance formula. What it
needs divided out first is the usual flat field -- what the detector would see
with no sample -- which is not one fixed frame here, since the aperture moves
every angle. `paganin_volume_coded` below gets it from the forward model
itself, run with `obj` forced to zero at the calibration code/pos_code, which
is what a real calibration scan (no sample in the beam) would measure.
"""
import argparse
import os
import sys

import numpy as np
import cupy as cp
from mpi4py import MPI
from types import SimpleNamespace

from holotomocupy.rec_coded_mpi import RecCoded
from holotomocupy.reader_coded import CodedWriter
from holotomocupy.paganin import multi_paganin
from holotomocupy.logger_config import logger, set_log_level

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phantoms import coded_aperture, phantom3d, thickness_for_phase  # noqa: E402

# ---- ESRF single-coded-aperture geometry (phascade esrf_simulation/params.py)
ENERGY = 33.35                        # keV
DET_PIXEL = 3 * 1.01250442970688e-6   # NATIVE detector pixel [m]
CROP = 2048                           # full detector width [native px]
Z_CODE = -18e-3                       # focus -> code; NEGATIVE = upstream of
                                      # the focal spot
Z_SAMPLE = 4.214e-3                   # focus -> sample
Z_DET = 1.28                          # focus -> detector

CA_FEATURE_UM = 1.0                   # fabricated cell size, at BIN=3
CA_THICKNESS_UM = 3.0                 # Au thickness. Contrast is 2|sin(phi/2)|
                                      # and peaks at t_pi = 6.49 um here; this
                                      # is a realistic mask, not the best one.
CA_ANGLE_DEG = 45.0                   # lattice rotation
CA_SUPERSAMPLE = 2                    # sub-samples per pixel, antialiases edges
OBJ_MARGIN = 64                       # nobj = n + this, absorbs the object shift.
                                      # params.py uses 32 because its sample
                                      # jitter is zero; the sample is displaced
                                      # +-15 px here, as in demo 2, so the
                                      # margin has to clear that plus the
                                      # shift kernel's taps -- otherwise
                                      # _build_data_mask starts discarding
                                      # real measurements at the frame edge.
CODE_JITTER_FRAC = 1.0 / 8.0          # per-angle code shift, +- this * n
PAGANIN_DELTA_BETA = 100.0            # matches phantom3d's default delta=1, beta=1e-2

SHIFT = 'fft'          # one distance means m == 1 for the object shift, so
                       # 'fft' never takes the chirp-z: faster and exact here,
                       # unlike in 02 where every plane but the first magnifies.


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bin', type=int, default=4,
                   help='detector binning; n = %d / 2**bin. 3 is the '
                        'params.py setting (n=256); 4 (default) halves it so '
                        'the demo stays a few minutes' % CROP)
    p.add_argument('--ntheta', type=int, default=360, help='projection angles')
    p.add_argument('--niter', type=int, default=129, help='BH iterations')
    p.add_argument('--feature-um', type=float, default=None,
                   help='aperture cell size [um]; default keeps it at '
                        f'{CA_FEATURE_UM} um of code-plane width whatever the '
                        'bin, so it stays equally well sampled')
    p.add_argument('--thickness-um', type=float, default=CA_THICKNESS_UM,
                   help='Au thickness [um]')
    p.add_argument('--code-err', type=float, default=0.02,
                   help='error on the starting code, as a fraction of its '
                        'contrast (0 = start from the exact aperture)')
    p.add_argument('--freeze-code', action='store_true',
                   help='hold the aperture fixed; only its positions refine')
    p.add_argument('--nchunk', type=int, default=16, help='angles per GPU pass')
    p.add_argument('--out', default='./demo_out/rec_coded', help='output directory')
    return p.parse_args()


def geometry(bin):
    """Every derived length, in phascade's convention: one shared index grid,
    a different physical pixel size at each plane."""
    det_px = DET_PIXEL * 2 ** bin
    n = CROP // 2 ** bin
    plane_px = lambda z: det_px * z / Z_DET
    g = SimpleNamespace(
        n=n, det_px=det_px,
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


def paganin_volume_coded(cl, nobj, delta_beta):
    """Single-distance Paganin + FBP, adapted to the coded-aperture engine.

    `multi_paganin` is the same function `02_holotomography.py` uses; with one
    distance it IS the textbook single-distance Paganin formula, nothing here
    is coded-aperture specific. What is specific is the flat field it needs
    divided out first: in plain holotomography that is one fixed frame,
    because the probe does not move, but here the aperture shifts every angle,
    so the flat field is a different speckle pattern at every angle. It is
    built from the cascade itself: `obj` forced to zero (identity
    transmission) at the TRUE code and pos_code, exactly what a calibration
    scan with no sample in the beam would measure.

    MPI: each rank processes its own angles, then `redist` reshapes the
    (this rank's angles, full object plane) stack into (all angles, this
    rank's z-slab) before FBP, mirroring `02_holotomography.py`'s
    `paganin_volume`.
    """
    obj_truth = cl.vars['obj'].copy()
    cl.vars['obj'][:] = 0
    flat = np.empty_like(cl.data)
    cl.gen_data(cl.vars, flat)
    cl.vars['obj'][:] = obj_truth

    mags = cl.cl_shift.unit_mag(1)      # m = 1: one distance, no magnification
    data_c = cp.asarray(cl.data)
    flat_c = cp.asarray(flat)
    psi = np.empty((cl.local_ntheta, nobj, nobj), dtype='complex64')
    for i in range(cl.local_ntheta):
        r = cp.asarray(cl.vars['pos'][i:i + 1])
        rdata = data_c[i:i + 1] / (flat_c[i:i + 1] + 1e-5)
        big = cp.exp(cl.cl_shift.curlySback(
            cp.log(rdata.astype('complex64')), r, mags)[0].real)
        ph = multi_paganin(big[None], cl.cl_prop.distance, cl.cl_prop.wavelength,
                           cl.cl_prop.voxelsize, delta_beta)
        psi[i] = cp.asnumpy(ph + 1j * ph / delta_beta)

    allpsi = np.empty((cl.ntheta, cl.local_nzobj, nobj), dtype='complex64')
    cl.redist(psi, allpsi, 'backward')
    gpsi = cp.asarray(allpsi)
    out = cp.empty((cl.local_nzobj, nobj, nobj), dtype='complex64')
    for z in range(0, cl.local_nzobj, cl.nchunk):   # cl_tomo is sized for nchunk slices
        z1 = min(z + cl.nchunk, cl.local_nzobj)
        out[z:z1] = cl.cl_tomo.fbp(cp.ascontiguousarray(gpsi[:, z:z1]), 'ramp')
    return out


def _corr(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.abs(np.vdot(a, b)) / d) if d else float('nan')


def _obj_mask(nobj, mask):
    """The solver's own circular support. Scoring outside it only measures the
    mask: the reconstruction is zero there by construction, the phantom is not."""
    yy, xx, _ = np.mgrid[0:nobj, 0:nobj, 0:nobj] - (nobj - 1) / 2
    return (yy ** 2 + xx ** 2) < (mask * nobj / 2) ** 2


def main():
    a = parse()
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    set_log_level('INFO')

    g = geometry(a.bin)
    n, nobj, ncode = g.n, g.nobj, g.ncode
    # Default the feature to a fixed number of CODE-PLANE PIXELS rather than a
    # fixed physical size, so coarsening the bin does not quietly push the mask
    # under Nyquist. At bin 3 this is the CA_FEATURE_UM of params.py.
    feature_m = (a.feature_um * 1e-6 if a.feature_um is not None
                 else CA_FEATURE_UM * 1e-6 * 2 ** (a.bin - 3))
    px_per_cell = feature_m / g.code_px

    if rank == 0:
        logger.info(f'bin {a.bin}: n={n} nobj={nobj} ncode={ncode} '
                    f'ntheta={a.ntheta}  code scan +-{g.scan_px}px')
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

    obj = phantom3d(nobj)
    prb = np.ones((n, n), dtype='complex64')   # flat: see the module docstring
    code = coded_aperture(ncode, feature_m, g.code_px, a.thickness_um * 1e-6,
                          ENERGY, angle_deg=CA_ANGLE_DEG,
                          supersample=CA_SUPERSAMPLE)
    if rank == 0:
        logger.info(f'code phase excursion {np.ptp(np.angle(code)):.3f} rad '
                    f'(t_pi = {thickness_for_phase(np.pi, ENERGY) * 1e6:.2f} um '
                    f'of Au at this energy)')

    rng = np.random.default_rng(10)
    pos          = (30 * (rng.random((a.ntheta, 2)) - 0.5)).astype('float32')
    pos_err      = (rng.random((a.ntheta, 2)) - 0.5).astype('float32')
    pos_code     = rng.uniform(-g.scan_px, g.scan_px,
                               (a.ntheta, 2)).astype('float32')
    pos_code_err = (rng.random((a.ntheta, 2)) - 0.5).astype('float32')
    theta = np.linspace(0, np.pi, a.ntheta, dtype='float32')

    cl = RecCoded(SimpleNamespace(
        energy=ENERGY, detector_pixelsize=g.det_px,
        focustodetectordistance=Z_DET, z1=Z_SAMPLE, z1_code=Z_CODE,
        theta=theta, ntheta=a.ntheta, nz=n, n=n, nzobj=nobj, nobj=nobj,
        ncode=ncode, mask=0.9, lam_laplacian=0,
        rho=[1, 0.05, 0.02, 0.05, 0.02], code_pad_frac=0.125,
        shift_type=SHIFT, freeze_code=a.freeze_code, freeze_pos_code=False,
        niter=a.niter, nchunk=a.nchunk,
        checkpoint_step=max(a.niter // 4, 1), error_step=max(a.niter // 8, 1),
        start_iter=0, comm=comm))

    writer = CodedWriter(path_out=a.out, comm=comm,
                         st_obj=cl.st_obj, end_obj=cl.end_obj,
                         nzobj=nobj, nobj=nobj,
                         st_theta=cl.st_theta, end_theta=cl.end_theta,
                         ntheta=a.ntheta, nz=n, n=n, ncode=ncode)

    # forward: each rank owns a slab of obj and a slice of the angles
    cl.vars['obj'][:]      = obj[cl.st_obj:cl.end_obj]
    cl.vars['prb'][:]      = prb
    cl.vars['code'][:]     = cp.asarray(code)
    cl.vars['pos'][:]      = pos[cl.st_theta:cl.end_theta]
    cl.vars['pos_code'][:] = pos_code[cl.st_theta:cl.end_theta]
    cl.gen_data(cl.vars, cl.data)      # INTENSITY, what F0 compares against

    # inverse: a Paganin+FBP start for the object, the exact flat probe
    # (nothing to recover there), a measured-but-imperfect aperture, and both
    # position sets off by ~0.5 px
    cl.vars['obj'][:] = cp.asnumpy(paganin_volume_coded(cl, nobj, PAGANIN_DELTA_BETA))
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

    cl.BH(writer=writer)

    # --- the honest numbers: distance to the truth, not to the start --------
    def _mean(x):
        return comm.allreduce(float(x), MPI.SUM) / size

    p1 = _mean(np.abs(np.asarray(cl.vars['pos'])
                      - pos[cl.st_theta:cl.end_theta]).mean())
    p0 = _mean(np.abs(pos_err[cl.st_theta:cl.end_theta]).mean())
    c1 = _mean(np.abs(np.asarray(cl.vars['pos_code'])
                      - pos_code[cl.st_theta:cl.end_theta]).mean())
    c0 = _mean(np.abs(pos_code_err[cl.st_theta:cl.end_theta]).mean())

    slabs = comm.gather(np.asarray(cl.vars['obj']), root=0)
    code_got = cp.asnumpy(cl.vars['code'])

    if rank == 0:
        full = np.concatenate(slabs, axis=0)
        # Real part, inside the solver's support: the imaginary part is
        # beta/delta = 1e-2 of the signal and scoring it only adds noise.
        m = _obj_mask(nobj, 0.9)
        logger.info(f'object corr      {_corr(full.real[m], obj.real[m]):.4f}')
        logger.info(f'code   corr      {_corr(code_got, code):.4f}  '
                    f'(started at {_corr(code0, code):.4f})')
        logger.info(f'pos      error   {p1:.4f} px (started at {p0:.4f})')
        logger.info(f'pos_code error   {c1:.4f} px (started at {c0:.4f})')
        logger.info(f'wrote {os.path.abspath(a.out)}  '
                    f'(checkpoints/ and checkpoints_tiff/)')


if __name__ == '__main__':
    main()
