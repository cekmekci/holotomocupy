#!/usr/bin/env python
"""
Step 0 (coded aperture) — recover the CODED APERTURE and the probe by
near-field ptychography on the mask alone.

Launch with mpirun / mpiexec, one rank per GPU:

    mpirun -np <ngpus> ./set_affinity_gpu.sh python step0.py config_step0.conf

The coded counterpart of `AtomiumS2/step0.py`, which calibrates a probe the
same way for the multi-distance pipeline. Its output, `code.npy` and
`prb.npy`, is what step 3 and step 6 take as the known aperture: step 6 starts
from a MEASURED mask rather than solving for one cold, which is how the
experiment is actually done.

INPUT IS ALREADY CONVERTED.  Unlike the multi-distance step 0, this reads the
preprocessed `data_nfp.h5` written by `step0a_convert_nfp.ipynb` and
`step0b_preprocess_nfp.ipynb` -- dark/flat correction and the cross-correlation
scan positions are interactive enough to be worth doing in a notebook.
Required datasets:

    /exchange/pdata    [npos, n, n]  float32, dark/flat-corrected INTENSITY
    /exchange/pref     [n, n]        float32, the reference (no mask)
    /exchange/shifts   [npos, 2]     float32, scan positions [row, col] px
    /exchange/{energy, z1, detector_pixelsize, focusdetectordistance, voxelsize}

THE APERTURE COMES BACK AS A LOG.  `RecNFP`'s forward model is

    D( exp(1j * S_pos(proj)) * prb )

while the coded fork's engine wrote the same physics as `D( S_r(psi) * q )`.
The fork's variable is the aperture TRANSMISSION, RecNFP's is its complex LOG,
so the Paganin seed goes in as `-1j*log(psi_init)` and the result is saved as
`exp(1j*proj)`. Nothing wraps: a few microns of Au at this energy is ~1.5 rad,
well inside the branch cut.

MEMORY.  This uses `RecNFPCoded`, not plain `RecNFP`: `RecNFP` shifts the
whole `npsi**2` plane for every position (the fork gathered an `n + 2*ex`
patch instead, and so does `RecNFPCoded`, via `shift_fft_code.ShiftFFTCode`
-- see `rec_nfp_coded_mpi.py`). Window size is set by `pos_pad_frac` in the
config; `pos_pad_frac=0` falls back to RecNFP's exact whole-plane behavior
(the ~40 GB / 4 GPU cost that made this windowing necessary at npsi=8192).
`nchunk` is still the knob to drop first if memory is still tight.
"""

import os
import sys

import numpy as np
import cupy as cp
import h5py
from mpi4py import MPI
from types import SimpleNamespace

from holotomocupy.rec_nfp_coded_mpi import RecNFPCoded
from holotomocupy.config import parse_args_coded_step0
from holotomocupy.logger_config import logger, set_log_level

cp.cuda.set_pinned_memory_allocator(None)

# --- Parse configuration file -------------------------------------------
args = parse_args_coded_step0(sys.argv[1])
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
set_log_level(args.log_level)

# --- Geometry and the scan, from the preprocessed HDF5 ------------------
with h5py.File(args.in_file, 'r') as fid:
    shifts = np.asarray(fid['/exchange/shifts'], dtype='float32')
    z1                    = float(fid['/exchange/z1'][0])
    detector_pixelsize    = float(fid['/exchange/detector_pixelsize'][0])
    focusdetectordistance = float(fid['/exchange/focusdetectordistance'][0])
    energy                = float(fid['/exchange/energy'][0])
    voxelsize             = float(fid['/exchange/voxelsize'][0])
    npos_all, n_file      = fid['/exchange/pdata'].shape[:2]

if n_file != args.n:
    raise SystemExit(f'{args.in_file} holds {n_file} px frames, config says n={args.n}')

wavelength = 1.24e-9 / energy
distance = z1 * (focusdetectordistance - z1) / focusdetectordistance

# --- Centre the scan and drop positions that leave the aperture plane ---
# The patch a position reads is n wide; the plane is npsi. A position further
# than this from the centre would read past the edge, where the model has
# nothing, so it contributes only wrap-around.
shifts[:, 0] -= (shifts[:, 0].max() + shifts[:, 0].min()) / 2
shifts[:, 1] -= (shifts[:, 1].max() + shifts[:, 1].min()) / 2
fov_limit = args.npsi // 2 - args.n // 2 - args.fov_margin
keep = np.where((np.abs(shifts[:, 0]) < fov_limit)
                & (np.abs(shifts[:, 1]) < fov_limit))[0]
shifts = shifts[keep]
npos = len(keep)
if rank == 0:
    logger.info(f'FOV limit {fov_limit} px -> kept {npos}/{npos_all} positions')
    logger.info(f'energy {energy:g} keV  voxel {voxelsize * 1e9:.2f} nm  '
                f'D {distance * 1e3:.4f} mm')

# --- Initialise the solver ----------------------------------------------
# RecNFPCoded calls the scan axis `ntheta`, and `nobj`/`nzobj` is the plane
# it reconstructs -- here the whole coded aperture.
cl = RecNFPCoded(SimpleNamespace(
    energy                  = energy,
    detector_pixelsize      = detector_pixelsize,
    focustodetectordistance = focusdetectordistance,
    z1                      = z1,
    ntheta                  = npos,
    nz                      = args.n,
    n                       = args.n,
    nzobj                   = args.npsi,
    nobj                    = args.npsi,
    rho                     = args.rho,
    niter                   = args.niter,
    nchunk                  = args.nchunk,
    checkpoint_step         = args.checkpoint_step,
    error_step              = args.error_step,
    start_iter              = args.start_iter,
    psf_sigma               = args.psf_sigma,
    model                   = args.model,
    estimate_rho            = args.estimate_rho,
    rho_estimate_niter      = args.rho_estimate_niter,
    rho_trial_error_step    = args.rho_trial_error_step,
    pos_pad_frac            = args.pos_pad_frac,
    path_out                = args.path_out,
    shift_type              = args.shift_type,
    comm                    = comm,
))

# --- Load this rank's frames and the reference --------------------------
# cl.data is the INTENSITY, not its square root: RecNFP.F0 compares K|psi|^2
# against d. step0b already divided out the flat, so the frames sit at ~1.
with h5py.File(args.in_file, 'r') as fid:
    ref = np.asarray(fid['/exchange/pref'], dtype='float32')
    local = keep[cl.st_theta:cl.end_theta]
    for i, j in enumerate(local):
        cl.data[i] = fid['/exchange/pdata'][j]

# --- Starting point ------------------------------------------------------
# Probe: back-propagate sqrt(ref) from the detector to the aperture plane.
# Aperture: Paganin per position, tiled into the plane, then its complex log.
cl.vars['prb'][:] = cl.cl_prop.DT(cp.asarray(np.sqrt(ref))[None], 0)[0]


def _paganin(frame, delta_beta, alpha=1e-1):
    """Single-distance Paganin phase of one flat-normalised intensity frame."""
    fx = cp.fft.fftfreq(frame.shape[-1], d=voxelsize).astype('float32')
    fx, fy = cp.meshgrid(fx, fx)
    taylor = 1 + wavelength * distance * cp.pi * delta_beta * (fx ** 2 + fy ** 2)
    num = taylor * cp.fft.fft2(frame)
    return delta_beta * 0.5 * cp.log(cp.real(cp.fft.ifft2(num / (taylor ** 2 + alpha))))


# Every rank tiles its own positions, then one allreduce stitches the plane:
# the aperture is global, so each rank only sees part of it.
acc = cp.zeros([args.npsi, args.npsi], dtype='float32')
cnt = cp.zeros([args.npsi, args.npsi], dtype='float32')
ishifts = np.round(shifts).astype('int32')
ref_g = cp.asarray(ref)
for i in range(cl.local_ntheta):
    j = cl.st_theta + i
    ph = _paganin(cp.asarray(cl.data[i]) / (ref_g + 1e-5), args.paganin)
    sty = args.npsi // 2 - ishifts[j, 0] - args.n // 2
    stx = args.npsi // 2 - ishifts[j, 1] - args.n // 2
    acc[sty:sty + args.n, stx:stx + args.n] += ph
    cnt[sty:sty + args.n, stx:stx + args.n] += 1
acc = cp.asarray(comm.allreduce(cp.asnumpy(acc), op=MPI.SUM))
cnt = cp.asarray(comm.allreduce(cp.asnumpy(cnt), op=MPI.SUM))
acc /= cp.maximum(cnt, 1)

# Smooth the seams the tiling leaves, then hand RecNFP the complex LOG.
v = cp.arange(-args.npsi // 2, args.npsi // 2) / args.npsi
vx, vy = cp.meshgrid(v, v)
flt = cp.exp(-1000 * (vx ** 2 + vy ** 2)).astype('float32')
psi_init = cp.exp(1j * acc).astype('complex64')
psi_init = cp.fft.fftshift(cp.fft.fftn(cp.fft.fftshift(psi_init)))
psi_init = cp.fft.fftshift(cp.fft.ifftn(cp.fft.fftshift(psi_init * flt)))
cl.vars['proj'][:] = (-1j * cp.log(psi_init.astype('complex64'))).astype('complex64')
cl.vars['pos'][:] = cp.asarray(shifts[cl.st_theta:cl.end_theta])
del acc, cnt, psi_init, ref_g

# --- Reconstruct ---------------------------------------------------------
logger.info('Run NFP reconstruction')
cl.BH()

# --- Save ----------------------------------------------------------------
# proj is the complex LOG of the aperture; the holotomography stage wants the
# transmission, so exponentiate on the way out.
if rank == 0:
    code = cp.asnumpy(cp.exp(1j * cl.vars['proj'])).astype('complex64')
    prb  = cp.asnumpy(cl.vars['prb']).astype('complex64')
    os.makedirs(os.path.dirname(args.code_out) or '.', exist_ok=True)
    np.save(args.code_out, code)
    np.save(args.prb_out,  prb)
    logger.info(f'code {code.shape} -> {args.code_out}')
    logger.info(f'prb  {prb.shape} -> {args.prb_out}')

    pos_err = np.abs(cp.asnumpy(cl.vars['pos']) - shifts[cl.st_theta:cl.end_theta])
    logger.info(f'position refinement: mean |dy| {pos_err[:, 0].mean():.3f} px, '
                f'mean |dx| {pos_err[:, 1].mean():.3f} px')
