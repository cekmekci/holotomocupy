#!/usr/bin/env python
"""
Step 0 (DanMAX, coded aperture) — recover CA2 and the probe by near-field
ptychography on the masks alone, with no sample in the beam.

Adapted from `AtomiumS2_coded/step0.py`. Two differences, both from the
beamline rather than the method:

  * the DanMAX orca is 2592 x 4608, so the grids are RECTANGULAR: `nz`/`npsiz`
    are carried separately from `n`/`npsi`. (`parse_args_coded_step0` defaults
    both to the square value, so the ESRF configs are unaffected.)
  * there are TWO apertures. CA1 is static and is folded into the PROBE; CA2
    is the one that moves and is the `code`. So `prb.npy` out of here is the
    probe at the CA2 plane with CA1 already imprinted on it.

Launch with mpirun / mpiexec, one rank per GPU:

    mpirun -np <ngpus> -mca opal_common_ucx_opal_mem_hooks 1 ./set_affinity_gpu.sh python step0.py config_step0.conf

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

    /exchange/pdata    [npos, nz, n] float32, dark/flat-corrected INTENSITY
    /exchange/pref     [nz, n]       float32, the reference (no mask)
    /exchange/shifts   [npos, 2]     float32, scan positions [row, col] px
    /exchange/{energy, z1, detector_pixelsize, focusdetectordistance, voxelsize}

THE APERTURE COMES BACK AS A LOG.  `RecNFP`'s forward model is

    D( exp(1j * S_pos(proj)) * prb )

while the coded fork's engine wrote the same physics as `D( S_r(psi) * q )`.
The fork's variable is the aperture TRANSMISSION, RecNFP's is its complex LOG,
so `proj` starts at 0 (psi = 1) and the result is saved as `exp(1j*proj)`.
Nothing wraps: a few microns of Au at this energy is ~1.5 rad, well inside
the branch cut.

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
    npos_all, nz_file, n_file = fid['/exchange/pdata'].shape[:3]

if (nz_file, n_file) != (args.nz, args.n):
    raise SystemExit(f'{args.in_file} holds [{nz_file}, {n_file}] px frames, '
                     f'config says nz={args.nz} n={args.n}')

wavelength = 1.24e-9 / energy
distance = z1 * (focusdetectordistance - z1) / focusdetectordistance

# --- Centre the scan and drop positions that leave the aperture plane ---
# The patch a position reads is n wide; the plane is npsi. A position further
# than this from the centre would read past the edge, where the model has
# nothing, so it contributes only wrap-around.
shifts[:, 0] -= (shifts[:, 0].max() + shifts[:, 0].min()) / 2
shifts[:, 1] -= (shifts[:, 1].max() + shifts[:, 1].min()) / 2
fov_limit_y = args.npsiz // 2 - args.nz // 2 - args.fov_margin
fov_limit_x = args.npsi  // 2 - args.n  // 2 - args.fov_margin
keep = np.where((np.abs(shifts[:, 0]) < fov_limit_y)
                & (np.abs(shifts[:, 1]) < fov_limit_x))[0]
shifts = shifts[keep]
npos = len(keep)
if rank == 0:
    logger.info(f'FOV limit y {fov_limit_y} / x {fov_limit_x} px -> '
                f'kept {npos}/{npos_all} positions')
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
    nz                      = args.nz,
    n                       = args.n,
    nzobj                   = args.npsiz,
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
    clip_passive            = args.clip_passive,
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
# Aperture: psi = 1, i.e. proj = 0. The bone_nano NFP notebook computes a
# Paganin seed and then throws it away -- "Initializing psi with 1 worked
# better" -- and AtomiumS2_coded/step0.py cold-starts for its own reason, so
# both references agree here.
cl.vars['proj'][:] = 0

# Probe: back-propagate sqrt(MEAN FRAME) from the detector to the CA2 plane,
# which is what the bone_nano notebook does (its cell 17, rebuilding q_init
# from scan 80's own mean rather than from the CA1-only scan 69, because CA1
# moved between the two).
#
# CA2 moves from frame to frame so it averages out of the mean; what survives
# is the probe with the STATIC CA1 imprinted on it, which is exactly the
# `prb` this stage is defined to recover.
#
# AtomiumS2_coded/step0.py leaves prb = 1 instead, but for a reason that does
# not apply here: its aperture sits UPSTREAM of the focus so its propagation
# distance is negative, and the seed diverged. This geometry has z1 < fdd and
# distance > 0, so the back-propagation is well posed.
#
# The mean is over ALL frames, so each rank contributes its own slice.
_local_sum = np.zeros((args.nz, args.n), dtype='float64')
for i in range(cl.end_theta - cl.st_theta):
    _local_sum += cl.data[i]
_mean_frame = comm.allreduce(_local_sum, op=MPI.SUM) / npos
cl.vars['prb'][:] = cl.cl_prop.DT(
    cp.asarray(np.sqrt(np.maximum(_mean_frame, 0)), dtype='complex64')[None], 0)[0]
if rank == 0:
    _a = cp.abs(cl.vars['prb'])
    logger.info(f'prb seed: back-propagated sqrt(mean frame), '
                f'|prb| in [{float(_a.min()):.4f}, {float(_a.max()):.4f}]')
del _local_sum, _mean_frame


# The Paganin seed AtomiumS2_coded/step0.py keeps commented out is deleted
# here rather than carried: both references discard it (that script because
# its distance is negative, the bone_nano notebook because psi = 1 simply
# worked better), and the version it carried was written for a SQUARE grid.

cl.vars['pos'][:] = cp.asarray(shifts[cl.st_theta:cl.end_theta])

# --- Reconstruct ---------------------------------------------------------
logger.info('Run NFP reconstruction')
cl.BH()

# --- Save ----------------------------------------------------------------
# proj is the complex LOG of the aperture; the holotomography stage wants the
# transmission, so exponentiate on the way out.
# `pos` is THETA-DISTRIBUTED: each rank holds only its own [st_theta, end_theta)
# slice, so it has to be gathered before saving -- writing cl.vars['pos'] from
# rank 0 alone stores just npos/size of them (81 of 324 on 4 ranks).  gather is
# collective, so it must run on EVERY rank, outside the `if rank == 0` below.
# get_local_chunk hands out contiguous blocks in rank order, so concatenating
# in that order rebuilds the global ordering.
_pos_parts = comm.gather(cp.asnumpy(cl.vars['pos']), root=0)

if rank == 0:
    code = cp.asnumpy(cp.exp(1j * cl.vars['proj'])).astype('complex64')
    prb  = cp.asnumpy(cl.vars['prb']).astype('complex64')
    os.makedirs(os.path.dirname(args.code_out) or '.', exist_ok=True)
    np.save(args.code_out, code)
    np.save(args.prb_out,  prb)
    logger.info(f'code {code.shape} -> {args.code_out}')
    logger.info(f'prb  {prb.shape} -> {args.prb_out}')

    # These are the KEPT positions, i.e. indexed like `shifts` after the
    # fov_limit filter above, not like the raw /exchange/shifts.
    pos_all = np.concatenate(_pos_parts, axis=0)
    assert pos_all.shape == shifts.shape, (
        f'gathered pos is {pos_all.shape}, expected {shifts.shape} -- '
        'the gather or the chunking is wrong, do not trust this file')
    pos_out = os.path.join(os.path.dirname(args.code_out), 'pos_nfp.npy')
    np.save(pos_out, pos_all)
    logger.info(f'pos  {pos_all.shape} -> {pos_out}')

    pos_err = np.abs(pos_all - shifts)
    logger.info(f'position refinement: mean |dy| {pos_err[:, 0].mean():.3f} px, '
                f'mean |dx| {pos_err[:, 1].mean():.3f} px, '
                f'max {np.abs(pos_err).max():.3f} px')
