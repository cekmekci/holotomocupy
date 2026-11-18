#!/usr/bin/env python
"""Stages 1-5 (coded aperture) -- batch port of the HT notebooks, MPI.

    mpirun -np <nranks> python steps15.py config_steps15.conf

Same job as `step1_convert.ipynb` + `step2_preprocessing.ipynb` +
`step3_find_shifts.ipynb` + `step5_rec_paganin.ipynb`, as one script, for
Polaris and for any run too big to sit in a notebook kernel. The notebooks
stay as they are for interactive use on tomo5; this is their batch twin, the
same split `AtomiumS1_HT_RD300` keeps between `steps15.py` and its own
notebooks.

No stage 4. The multi-distance pipeline pre-bins there; `CodedReader` bins on
the way in, so one `codedHT.h5` at bin 0 serves every step-6 rung.

`start_step` in `config_steps15.conf` gates which stages (re)run, same
`start_step > N` convention as the multi-distance `steps15.py`:

    start_step=1   run everything (default)
    start_step=2   skip stage 1 (codedHT.h5 exists with /exchange/data)
    start_step=3   skip stages 1-2 (pdata/pref0/pref1 already written)
    start_step=5   skip stages 1-3 (shifts_cor/shifts_code/prb already written)

Frames are split across MPI ranks by `ntheta` for stages 1-3 (I/O and
per-frame work); stage 5's FBP is split by z instead, one rank per GPU, since
reconstructing a z-slab needs every angle.
"""
import os
import sys
import threading

import h5py
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndimage
from mpi4py import MPI

from holotomocupy.config import parse_args_coded_steps15
from holotomocupy.mpi_functions import MPIClass, get_local_chunk
from holotomocupy.propagation import Propagation
from holotomocupy.tomo import Tomo
from holotomocupy.logger_config import logger, set_log_level

cfg = parse_args_coded_steps15(sys.argv[1])
set_log_level(cfg.log_level)

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
ngpus = cp.cuda.runtime.getDeviceCount()
cp.cuda.Device(rank % ngpus).use()

start_step = cfg.start_step
WITH_CA  = cfg.path
fname    = cfg.in_file                     # codedHT.h5

# Structural constants, not per-dataset: same values the notebooks hardcode.
ndark, nref = 20, 20
ny, nx = cfg.n, cfg.n
chunk  = 200                                # step 2 batch size
ROT_AXIS_OFFSET_PX = np.array(cfg.rot_axis_offset, dtype='float32')

def log(msg):
    if rank == 0:
        logger.info(msg)

# ===========================================================================
# STAGE 1 -- convert raw EDF scan to codedHT.h5
# ===========================================================================
if start_step <= 1:
    import dxchange

    log(f'Stage 1: convert {cfg.raw_ht} -> {fname}')
    path_cht  = cfg.raw_ht + '/'
    ntheta    = cfg.ntheta

    energy                  = 33.35
    wavelength              = 1.2398419840550367e-09 / energy
    focusToDetectorDistance = 1.28
    sx0                     = 1.286e-3
    detector_pixelsize      = 3.03751e-6
    sx_mm         = 4.236
    z1            = sx_mm * 1e-3 - sx0
    z2            = focusToDetectorDistance - z1
    distance      = (z1 * z2) / focusToDetectorDistance
    magnification = focusToDetectorDistance / z1
    voxelsize     = detector_pixelsize / magnification

    def read_edf_motors(filename):
        mne_vals = pos_vals = None
        with open(filename, 'r', encoding='latin-1') as fh:
            for line in fh:
                line = line.strip()
                if line.startswith('motor_mne'):
                    mne_vals = line.split('=')[1].replace(';', '').split()
                elif line.startswith('motor_pos'):
                    pos_vals = line.split('=')[1].replace(';', '').split()
                if '}' in line:
                    break
        if mne_vals is None or pos_vals is None:
            return None
        return dict(zip(mne_vals, [float(v) for v in pos_vals]))

    if rank == 0:
        os.makedirs(WITH_CA, exist_ok=True)
        attrs = np.loadtxt(f'{path_cht}attributes.txt', comments='#').astype('float32')
        srcur = attrs[:, 0]
        correct = np.loadtxt(f'{path_cht}correct.txt').astype('float32')
        shifts  = correct[:ntheta, ::-1]

        dark_stack = np.zeros((ndark, ny, nx), dtype='uint16')
        ref0_stack = np.zeros((nref,  ny, nx), dtype='uint16')
        ref1_stack = np.zeros((nref,  ny, nx), dtype='uint16')
        for i in range(ndark):
            dark_stack[i] = dxchange.read_edf(f'{path_cht}darkend{i:04d}.edf')[0]
        for i in range(nref):
            ref0_stack[i] = dxchange.read_edf(f'{path_cht}ref{i:04d}_0000.edf')[0]
            ref1_stack[i] = dxchange.read_edf(f'{path_cht}ref{i:04d}_7200.edf')[0]
    comm.Barrier()

    # Each rank reads its own slice of projection frames and motor headers;
    # rank 0 gathers and writes, same pattern as the multi-distance steps15.py.
    st, end = get_local_chunk(ntheta, rank, size)
    local_theta = np.zeros(end - st, dtype='float32')
    local_spy   = np.zeros(end - st, dtype='float32')
    local_spz   = np.zeros(end - st, dtype='float32')
    local_sx    = np.zeros(end - st, dtype='float32')
    local_frames = np.zeros((end - st, ny, nx), dtype='uint16')
    for j, i_raw in enumerate(range(st, end)):
        fn = f'{path_cht}{cfg.pfile}{i_raw:04d}.edf'
        local_frames[j] = dxchange.read_edf(fn)[0]
        motor = read_edf_motors(fn)
        if motor is not None:
            local_theta[j] = motor.get('somega', np.nan)
            local_spy[j]   = motor.get('spy', np.nan)
            local_spz[j]   = motor.get('spz', np.nan)
            local_sx[j]    = motor.get('sx', np.nan)
    log(f'stage1: rank 0 read frames {st}:{end} of {ntheta}')

    with h5py.File(fname, 'w', driver='mpio', comm=comm) as f:
        dset = f.create_dataset('/exchange/data', shape=(ntheta, ny, nx),
                                dtype='uint16', chunks=(1, ny, nx))
        theta_ds = f.create_dataset('/exchange/theta', shape=(ntheta,), dtype='float32')
        spy_ds   = f.create_dataset('/exchange/spy',   shape=(ntheta,), dtype='float32')
        spz_ds   = f.create_dataset('/exchange/spz',   shape=(ntheta,), dtype='float32')
        sx_ds    = f.create_dataset('/exchange/sx',    shape=(ntheta,), dtype='float32')
        comm.Barrier()
        dset[st:end]     = local_frames
        theta_ds[st:end] = local_theta
        spy_ds[st:end]   = local_spy
        spz_ds[st:end]   = local_spz
        sx_ds[st:end]    = local_sx
    del local_frames
    comm.Barrier()

    if rank == 0:
        with h5py.File(fname, 'a') as f:
            spy_arr = f['/exchange/spy'][:]
            spz_arr = f['/exchange/spz'][:]
            f.create_dataset('/exchange/dark',        data=dark_stack)
            f.create_dataset('/exchange/data_white0', data=ref0_stack)
            f.create_dataset('/exchange/data_white1', data=ref1_stack)
            f.create_dataset('/exchange/pos_shifts',
                             data=np.stack([spy_arr, spz_arr], axis=1))
            f.create_dataset('/exchange/srcur',  data=srcur)
            f.create_dataset('/exchange/attrs',  data=attrs)
            f.create_dataset('/exchange/shifts', data=shifts)
            f.create_dataset('/exchange/energy',               data=np.array([energy]))
            f.create_dataset('/exchange/wavelength',           data=np.array([wavelength]))
            f.create_dataset('/exchange/detector_pixelsize',   data=np.array([detector_pixelsize]))
            f.create_dataset('/exchange/focusdetectordistance', data=np.array([focusToDetectorDistance]))
            f.create_dataset('/exchange/z1',             data=np.array([z1]))
            f.create_dataset('/exchange/z2',             data=np.array([z2]))
            f.create_dataset('/exchange/distance',       data=np.array([distance]))
            f.create_dataset('/exchange/magnification',  data=np.array([magnification]))
            f.create_dataset('/exchange/voxelsize',      data=np.array([voxelsize]))
        log(f'stage1: done, {ntheta} frames -> {fname}')
    comm.Barrier()

# ===========================================================================
# STAGE 2 -- dark/flat correction, chunked -> pdata, pref0, pref1
# ===========================================================================
if start_step <= 2:
    log('Stage 2: preprocess')

    def remove_outliers(data, dezinger, dezinger_threshold):
        res = data.copy()
        w = [dezinger, dezinger]
        for k in range(data.shape[0]):
            d0 = cp.array(data[k])
            fdata = ndimage.median_filter(d0, w)
            res[k] = cp.where(cp.abs(d0 - fdata) > fdata * dezinger_threshold,
                              fdata, d0).get()
        return res

    def process_ref_stack(stack, dark_mean):
        r = stack.copy()
        r -= dark_mean[np.newaxis]
        r[r < 0] = 0
        r[:, 1320 // 3:1320 // 3 + 25 // 3, 890 // 3:890 // 3 + 25 // 3] = \
            r[:, 1280 // 3:1280 // 3 + 25 // 3, 890 // 3:890 // 3 + 25 // 3]
        return remove_outliers(r, 3, 0.8)

    with h5py.File(fname, 'r', driver='mpio', comm=comm) as f:
        dark_stack = f['/exchange/dark'][:].astype('float32')
        ref0_stack = f['/exchange/data_white0'][:].astype('float32')
        ref1_stack = f['/exchange/data_white1'][:].astype('float32')
        bpm4   = f['/exchange/attrs'][:, 1]
        ntheta, nz, n = f['/exchange/data'].shape

    dark_mean = dark_stack.mean(axis=0)
    ref0_proc = process_ref_stack(ref0_stack, dark_mean)
    ref1_proc = process_ref_stack(ref1_stack, dark_mean)
    ref0_mean = ref0_proc.mean(axis=0)
    ref_norm_scalar = float(np.mean(ref0_mean))
    pref0 = ref0_mean / ref_norm_scalar
    pref1 = ref1_proc / ref_norm_scalar
    bpm4_ref = float(bpm4[0])

    st, end = get_local_chunk(ntheta, rank, size)
    with h5py.File(fname, 'a', driver='mpio', comm=comm) as f:
        for key in ['/exchange/pdata', '/exchange/pref0', '/exchange/pref1']:
            if key in f:
                del f[key]
        comm.Barrier()
        dset_out = f.create_dataset('/exchange/pdata', shape=(ntheta, nz, n),
                                    dtype='float32', chunks=(1, nz, n))
        comm.Barrier()
        dset_in = f['/exchange/data']
        for ic0 in range(st, end, chunk):
            ic1 = min(ic0 + chunk, end)
            raw = dset_in[ic0:ic1].astype('float32')
            raw -= dark_mean[np.newaxis]
            raw[raw < 0] = 0
            raw[:, 1320 // 3:1320 // 3 + 25 // 3, 890 // 3:890 // 3 + 25 // 3] = \
                raw[:, 1280 // 3:1280 // 3 + 25 // 3, 890 // 3:890 // 3 + 25 // 3]
            raw = remove_outliers(raw, 3, 0.8)
            raw /= ref_norm_scalar
            c_bpm4 = (bpm4[ic0:ic1] / bpm4_ref).astype('float32')
            raw /= c_bpm4[:, np.newaxis, np.newaxis]
            raw = np.clip(raw, 0, None)
            dset_out[ic0:ic1] = raw
        comm.Barrier()
        if rank == 0:
            f.create_dataset('/exchange/pref0', data=pref0)
            f.create_dataset('/exchange/pref1', data=pref1)
    log(f'stage2: done, pdata/pref0/pref1 -> {fname}')
    comm.Barrier()

# ===========================================================================
# STAGE 3 -- CA shifts (xcorr) and object shifts -> shifts_cor, shifts_code, prb
# ===========================================================================
if start_step <= 3:
    log('Stage 3: find shifts')

    def phase_corr_gpu(a_gpu, b_gpu):
        ip  = cp.fft.fft2(a_gpu) * cp.conj(cp.fft.fft2(b_gpu))
        cc  = cp.fft.fftshift(cp.fft.ifft2(ip).real)
        idx = cp.unravel_index(cp.argmax(cc), cc.shape)
        dy  = int(idx[0].get()) - cc.shape[0] // 2
        dx  = int(idx[1].get()) - cc.shape[1] // 2
        return np.array([dy, dx], dtype='float32')

    with h5py.File(fname, 'r', driver='mpio', comm=comm) as f:
        energy = float(f['/exchange/energy'][0])
        det_px = float(f['/exchange/detector_pixelsize'][0])
        fdd    = float(f['/exchange/focusdetectordistance'][0])
        z1     = float(f['/exchange/z1'][0])
        ntheta, nz, n = f['/exchange/pdata'].shape
        shifts = f['/exchange/shifts'][:].astype('float32')
        theta_deg = f['/exchange/theta'][:]

    z1_code = cfg.z1_code
    if z1_code is None:
        with h5py.File(cfg.nfp_file, 'r') as f:
            z1_code = float(f['/exchange/z1'][0])

    code = np.load(cfg.code_file).astype('complex64')
    prb  = np.load(cfg.prb_file).astype('complex64')
    ncode = code.shape[0]

    wavelength = 1.24e-9 / energy
    z2 = fdd - z1
    magnification = fdd / z1
    distance  = z1 * z2 / fdd
    voxelsize = det_px / magnification
    z2c = z1 - z1_code
    magnification_code = z1 / z1_code
    distance_code = (z1_code * z2c) / (z1_code + z2c)
    norm_magnification_code = magnification_code / (z1_code / z1)
    distance_code = distance_code * norm_magnification_code**2 * (z1_code / z1)**2

    cl_prop      = Propagation(n, nz, 1, 1, wavelength, voxelsize, [distance])
    cl_prop_code = Propagation(n, nz, 1, 1, wavelength, voxelsize, [distance_code])
    cl_prop_8192      = Propagation(ncode, ncode, 1, 1, wavelength, voxelsize, [distance])
    cl_prop_code_8192 = Propagation(ncode, ncode, 1, 1, wavelength, voxelsize, [distance_code])

    prb_gpu  = cp.array(prb)
    prb_samp = cl_prop_code.D(prb_gpu, 0)
    ref_gpu  = cl_prop.D(prb_samp, 0)
    ref = cp.abs(ref_gpu).real**2
    ref = ref.get()
    del prb_gpu, prb_samp, ref_gpu

    code_gpu       = cp.array(code)
    code_samp_8192 = cl_prop_code_8192.D(code_gpu, 0)
    cdata_gpu      = cl_prop_8192.D(code_samp_8192, 0)
    cdata = cp.abs(cdata_gpu).real**2
    del code_gpu, code_samp_8192, cdata_gpu, code
    cdata_gpu_f32 = cp.array(cdata, dtype='float32')
    cdata = cdata.get()

    st, end = get_local_chunk(ntheta, rank, size)
    pad = (ncode - n) // 2
    local_shifts_code = np.zeros((end - st, 2), dtype='float32')
    with h5py.File(fname, 'r', driver='mpio', comm=comm) as f:
        pdata_dset = f['/exchange/pdata']
        for j, k in enumerate(range(st, end)):
            frame = pdata_dset[k].astype('float32')
            rdata_k = frame / (ref + 1e-6)
            rdata_pad = cp.pad(cp.array(rdata_k), ((pad, pad), (pad, pad)),
                               constant_values=1)
            local_shifts_code[j] = -phase_corr_gpu(cdata_gpu_f32, rdata_pad)
    shifts_code = np.zeros((ntheta, 2), dtype='float32')
    comm.Allgatherv(local_shifts_code,
                    [shifts_code, comm.allgather(local_shifts_code.size),
                     comm.allgather(st * 2), MPI.FLOAT])
    log(f'stage3: shifts_code row [{shifts_code[:,0].min():.1f},{shifts_code[:,0].max():.1f}]')

    nsrc = 1800   # the ESRF correction table's own grid, same as the notebook
    cc = np.loadtxt(cfg.correction_file)[:nsrc][:, ::-1].astype(np.float32)
    x_old = np.linspace(theta_deg.min(), theta_deg.max(), nsrc)
    cc = np.stack([np.interp(theta_deg, x_old, cc[:, i]) for i in range(2)],
                 axis=1).astype(np.float32)
    shifts_cor = shifts + cc

    if rank == 0:
        with h5py.File(fname, 'a') as f:
            for key in ['/exchange/prb', '/exchange/shifts_cor', '/exchange/shifts_code']:
                if key in f:
                    del f[key]
            f.create_dataset('/exchange/prb',         data=prb.astype('complex64'))
            f.create_dataset('/exchange/shifts_cor',  data=shifts_cor.astype('float32'))
            f.create_dataset('/exchange/shifts_code', data=shifts_code.astype('float32'))
        log(f'stage3: done, prb/shifts_cor/shifts_code -> {fname}')
    comm.Barrier()

# ===========================================================================
# STAGE 5 -- per-angle Paganin + FBP -> obj_init.npy
# ===========================================================================
if start_step <= 5:
    log('Stage 5: Paganin + FBP')

    def multiPaganin(data, dist, wavelength, voxelsize, delta_beta, alpha):
        fy = cp.fft.fftfreq(data.shape[-2], d=voxelsize).astype('float32')
        fx = cp.fft.fftfreq(data.shape[-1], d=voxelsize).astype('float32')
        fx, fy = cp.meshgrid(fx, fy)
        rad_freq = cp.fft.fft2(data)
        taylorExp = 1 + wavelength * dist * cp.pi * delta_beta * (fx**2 + fy**2)
        numerator = taylorExp * rad_freq
        denominator = taylorExp**2 + alpha
        phase = cp.log(cp.real(cp.fft.ifft2(numerator / denominator)))
        return delta_beta * 0.5 * phase

    def fourier_shift_2d(arr, dc, dr):
        """Sub-pixel shift. Real in -> real out; complex in -> complex out."""
        nr, nc = arr.shape
        fr = cp.fft.fftfreq(nr).astype('float32')
        fc = cp.fft.fftfreq(nc).astype('float32')
        Fc, Fr = cp.meshgrid(fc, fr)
        ramp = cp.exp(-2j * cp.pi * (Fr * dr + Fc * dc)).astype('complex64')
        out = cp.fft.ifft2(cp.fft.fft2(arr.astype('complex64')) * ramp)
        return out if arr.dtype.kind == 'c' else out.real.astype('float32')

    def pad_taper(proj, nzobj, nobj, bg):
        """Linear ramp from the edge to `bg` across the whole margin, in both axes.
        His steps15.py:916-919 ramps x only and replicates the edge in y; our vertical
        edges carry structure and replicated rows are the same at every angle, which
        FBP back-projects onto the rotation axis."""
        nz_, n_ = proj.shape
        py0, px0 = (nzobj - nz_) // 2, (nobj - n_) // 2
        return cp.pad(proj, ((py0, nzobj - nz_ - py0), (px0, nobj - n_ - px0)),
                      'linear_ramp', end_values=bg)

    with h5py.File(fname, 'r', driver='mpio', comm=comm) as f:
        energy = float(f['/exchange/energy'][0])
        det_px = float(f['/exchange/detector_pixelsize'][0])
        fdd    = float(f['/exchange/focusdetectordistance'][0])
        z1     = float(f['/exchange/z1'][0])
        ntheta, nz, n = f['/exchange/pdata'].shape
        theta_deg   = f['/exchange/theta'][:].astype('float32')
        shifts_code = f['/exchange/shifts_code'][:].astype('float32')
        shifts_cor  = f['/exchange/shifts_cor'][:].astype('float32')
    nobj = cfg.nobj
    nzobj = nobj * nz // n   # preserves aspect ratio; nz==n==nzobj==nobj today

    z1_code = cfg.z1_code
    if z1_code is None:
        with h5py.File(cfg.nfp_file, 'r') as f:
            z1_code = float(f['/exchange/z1'][0])

    pos_obj = shifts_cor + ROT_AXIS_OFFSET_PX

    # step7.py's per-angle drift correction, optional -- rebuilds the Paganin
    # seed geometrically consistent with pass 2's corrected `pos`. Same 2-column
    # (horizontal, vertical) convention CodedReader.read_correct3d_extra reads,
    # but NO 1/2**bin scale: the file is in unbinned object px and stage 5 runs
    # unbinned.
    if cfg.correct3d_extra_file:
        raw = np.loadtxt(cfg.correct3d_extra_file)
        if raw.shape[0] < ntheta:
            raise SystemExit(f'{cfg.correct3d_extra_file}: {raw.shape[0]} rows, '
                             f'need at least {ntheta}')
        extra = raw[:ntheta, ::-1].astype('float32')   # (horizontal, vertical) -> (row, col)
        pos_obj = pos_obj + extra
        if rank == 0:
            logger.info(f'stage5: correct3d_extra {cfg.correct3d_extra_file} '
                       f'applied -- y ptp {np.ptp(extra[:, 0]):.4f}  '
                       f'x ptp {np.ptp(extra[:, 1]):.4f}  unbinned px')
    elif rank == 0:
        logger.info('stage5: no correct3d_extra_file -- pos_obj is '
                   'shifts_cor + rot_axis_offset only')

    wavelength = 1.24e-9 / energy
    z2 = fdd - z1
    magnification = fdd / z1
    distance  = z1 * z2 / fdd
    voxelsize = det_px / magnification
    z2c = z1 - z1_code
    magnification_code = z1 / z1_code
    distance_code = (z1_code * z2c) / (z1_code + z2c)
    norm_magnification_code = magnification_code / (z1_code / z1)
    distance_code = distance_code * norm_magnification_code**2 * (z1_code / z1)**2

    code_gpu = cp.asarray(np.load(cfg.code_file).astype('complex64'))
    prb_gpu  = cp.asarray(np.load(cfg.prb_file).astype('complex64'))
    ncode_full = code_gpu.shape[0]

    cl_prop      = Propagation(n, nz, 1, 1, wavelength, voxelsize, [distance])
    cl_prop_code = Propagation(n, nz, 1, 1, wavelength, voxelsize, [distance_code])

    def code_window(k, G=8):
        """Code over the detector at angle k: integer crop from shifts_code, sub-pixel
        remainder by Fourier shift (G px guard band so nothing wraps in)."""
        dy, dx = int(round(shifts_code[k, 0])), int(round(shifts_code[k, 1]))
        cr, cc = ncode_full // 2 - dy, ncode_full // 2 - dx
        w = code_gpu[cr - nz // 2 - G: cr + nz // 2 + G, cc - n // 2 - G: cc + n // 2 + G]
        return fourier_shift_2d(w, dc=float(shifts_code[k, 1] - dx),
                                dr=float(shifts_code[k, 0] - dy))[G:-G, G:-G]

    def flat_field(k):
        """What the detector records with no sample, as RecCoded.F2 models it: the code
        window times the probe, propagated TOGETHER code plane -> sample -> detector."""
        return cp.abs(cl_prop.D(cl_prop_code.D(prb_gpu * code_window(k), 0), 0)) ** 2

    # --- stage 5 knobs (config_steps15.conf; notebook step5_paganin_ablation.ipynb) ---
    DELTA_BETA   = cfg.paganin
    ALPHA        = cfg.paganin_alpha
    SIGMA_SMOOTH = (cfg.smooth_fwhm / (2 * np.sqrt(2 * np.log(2)))
                    if cfg.smooth_fwhm > 0 else 0.0)
    CROP         = cfg.edge_crop
    r0, r1, c0, c1 = cfg.bg_box
    bg_box = (int(r0 * nz), int(r1 * nz), int(c0 * n), int(c1 * n))
    assert bg_box[0] >= CROP and bg_box[2] >= CROP and bg_box[1] <= nz - CROP and bg_box[3] <= n - CROP, \
        'bg_box overlaps the stripped rim (edge_crop); move it inside the kept region'
    if rank == 0:
        logger.info(f'stage5: delta_beta={DELTA_BETA} alpha={ALPHA} smooth_fwhm={cfg.smooth_fwhm} '
                    f'(sigma {SIGMA_SMOOTH:.3f} px) edge_crop={CROP} mask_r={cfg.mask_r} '
                    f'bg_box={cfg.bg_box} paganin_bg_box={cfg.paganin_bg_box}')

    def _ratio(frame, k):
        """Blurred measurement over blurred flat field, his epsilon 1e-5 (his :894-895)."""
        num = cp.asarray(frame)
        den = flat_field(k)
        if SIGMA_SMOOTH > 0:
            num = ndimage.gaussian_filter(num, SIGMA_SMOOTH)
            den = ndimage.gaussian_filter(den, SIGMA_SMOOTH)
        return num / (den + 1e-5)

    # Each rank Paganin-corrects its own angle slice, same split as stage 3.
    # get_local_chunk hands out contiguous blocks in rank order, so angle 0 is
    # always rank 0's first frame.
    pad8 = nobj // 8
    st, end = get_local_chunk(ntheta, rank, size)
    local_paganin = np.zeros((end - st, nzobj, nobj), dtype='float32')

    # Pad level: mean of bg_box on angle 0, measured BEFORE the loop (every rank needs
    # it on its first frame) and broadcast so all ranks pad to exactly the same level.
    with h5py.File(fname, 'r', driver='mpio', comm=comm) as f:
        ratio0 = _ratio(f['/exchange/pdata'][0].astype('float32'), 0)
        bg_fixed = float(cp.mean(ratio0[bg_box[0]:bg_box[1], bg_box[2]:bg_box[3]]))
        del ratio0
    bg_fixed = comm.bcast(bg_fixed, root=0)

    with h5py.File(fname, 'r', driver='mpio', comm=comm) as f:
        pdata_dset = f['/exchange/pdata']
        for j, k in enumerate(range(st, end)):
            frame = pdata_dset[k].astype('float32')
            kept = _ratio(frame, k)[CROP:nz - CROP, CROP:n - CROP]   # strip the unreliable rim
            proj = pad_taper(kept, nzobj, nobj, bg=bg_fixed)
            proj = cp.pad(proj, pad8, 'reflect')                      # his: against FFT wrap-around
            phase = multiPaganin(proj, distance, wavelength, voxelsize, DELTA_BETA, ALPHA)
            phase = phase[pad8:-pad8, pad8:-pad8]
            phase = fourier_shift_2d(phase, dc=-pos_obj[k, 1], dr=-pos_obj[k, 0])
            local_paganin[j] = phase.get()
    del code_gpu, prb_gpu
    cp.get_default_memory_pool().free_all_blocks()
    log(f'stage5: paganin done for local angles {st}:{end}; pad level {bg_fixed:.5f}')

    # --- one global background off the phase (his :1231, :1263) -------------
    # Mean of paganin_bg_box (fractions of the PADDED frame) on angle 0's phase,
    # which is rank 0's local_paganin[0]; broadcast. Commutes with the
    # angles->z transpose below.
    pb = cfg.paganin_bg_box
    pbox = (int(pb[0] * nzobj), int(pb[1] * nzobj), int(pb[2] * nobj), int(pb[3] * nobj))
    assert pbox[1] > pbox[0] and pbox[3] > pbox[2], 'paganin_bg_box is empty'
    global_bg = (float(np.mean(local_paganin[0, pbox[0]:pbox[1], pbox[2]:pbox[3]]))
                 if rank == 0 else None)
    global_bg = comm.bcast(global_bg, root=0)
    local_paganin -= global_bg
    log(f'stage5: subtracted global_bg {global_bg:.6f} (mean of paganin_bg_box {pb} on angle 0)')

    # Paganin was split over ANGLES; FBP needs every angle for each z-slab, so
    # the data has to be redistributed to a z-split. MPIClass.backward is
    # exactly that transpose -- (local_ntheta, nzobj, nobj) -> (ntheta,
    # local_nzobj, nobj) -- via one Alltoallw over cached subarray datatypes.
    #
    # It replaces an Allgatherv that replicated the WHOLE sinogram onto EVERY
    # rank (2026-11-11). That allocated ntheta*nzobj*nobj floats per rank --
    # 176 GiB each at 7200 x 2560^2, x4 ranks/node, on top of each rank's own
    # 47 GiB local_paganin still alive until the del below -- and OOM-killed
    # the job the moment the last rank finished Paganin. The old comment
    # calling it "small, ~1 frame-stack per rank" was describing the per-rank
    # SLICE while the code allocated the full GLOBAL array. Each rank only
    # ever reads its own z-slab out of it, so the replication was pure waste.
    mpi_redist = MPIClass(comm, nzobj, ntheta, nobj, 'float32')
    sino_local = np.zeros((ntheta, mpi_redist.local_nzobj, nobj), dtype='float32')
    mpi_redist.backward(local_paganin, sino_local)
    del local_paganin
    mpi_redist.close()
    log(f'stage5: redistributed angles->z, local slab {sino_local.shape}')

    # Real-first FBP: reconstruct the real phase once (float32), then imag = real /
    # delta_beta (his :1304). A complex sinogram makes Tomo.RT leak the real
    # channel's spurious imaginary part (~0.4% of Re) into Im, which is
    # delta_beta times weaker -- the fine vertical lines.
    FBP_FILTER = cfg.fbp_filter
    theta_rad  = np.deg2rad(theta_deg)
    NCHUNK     = 8

    zst, zend = get_local_chunk(nzobj, rank, size)
    cl = Tomo(nobj, NCHUNK, theta_rad, mask_r=cfg.mask_r)
    local_obj = np.zeros((zend - zst, nobj, nobj), dtype='complex64')
    for z0 in range(0, zend - zst, NCHUNK):
        z1_ = min(z0 + NCHUNK, zend - zst)
        # sino_local is ALREADY this rank's slab, so index it from 0, not zst
        L = cl.fbp(cp.asarray(sino_local[:, z0:z1_, :]), FBP_FILTER).get()
        local_obj[z0:z1_].real = L
        local_obj[z0:z1_].imag = L / DELTA_BETA
    log(f'stage5: FBP done for local z {zst}:{zend} ')
    del sino_local

    # Counts/displs in PLANE units, not elements: local_obj.size (zend-zst *
    # nobj * nobj) overflows MPI's 32-bit int count at size=4 (4.19e9 for a
    # 640-slice rank, nzobj=2560) -- it only happened to fit at size=8 (2 nodes,
    # 2.10e9, ~2.4% under INT_MAX) by accident, which is why this surfaced the
    # first time a 1-node/4-rank run actually reached this line. A contiguous
    # nobj*nobj plane datatype keeps every count a small integer (<=nzobj)
    # regardless of how many ranks the job uses.
    plane_t = MPI.COMPLEX8.Create_contiguous(nobj * nobj).Commit()
    obj_counts = np.array(comm.allgather(zend - zst))
    obj_displs = np.array([sum(obj_counts[:r]) for r in range(size)])
    obj_fbp = np.zeros((nzobj, nobj, nobj), dtype='complex64') if rank == 0 else None
    recvbuf = [obj_fbp, obj_counts, obj_displs, plane_t] if rank == 0 else None
    comm.Gatherv([local_obj, zend - zst, plane_t], recvbuf, root=0)
    plane_t.Free()
    if rank == 0:
        np.save(cfg.obj_init_file, obj_fbp)
        logger.info(f'stage5: done, obj_init -> {cfg.obj_init_file}')
    comm.Barrier()

log('steps15: done')
