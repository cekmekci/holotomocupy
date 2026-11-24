#!/usr/bin/env python
"""Run the production Paganin on each swept NFP result and show every stage.

One row per angle, one column per stage of what stage 5 actually does:

    measured -> flat (model) -> division -> padded -> Paganin phase

This is `steps15.py` stage 5, binned for speed the way `step5_paganin_ablation`
bins it, and NOTHING ELSE.  No per-angle low-order surface, no CODE_SCALE, no
sub-pixel code offsets.  Those were ablation experiments and all of them were
either retracted or left unapplied; the point here is to see what the pipeline
as it stands produces from each trial's `code.npy`/`prb.npy`.

Faithful to production:
  - flat = |D(D_code(prb * code_window(k)))|^2, code_window from the integer
    shifts_code in codedHT.h5 with the sub-pixel remainder by Fourier shift
  - both sides blurred by smooth_fwhm=17 px BEFORE dividing, his epsilon 1e-5
  - edge_crop=36 stripped, then pad_taper's linear ramp to the angle-0 bg_box
    level, then pad8 reflect against FFT wrap-around
  - multiPaganin at delta_beta=30, alpha=1e-2
  - shifted by -(shifts_cor + rot_axis_offset)
  - one global_bg (angle-0 paganin_bg_box) subtracted from every angle

ONE DEPARTURE, deliberate.  The flat is blurred by the trial's own `psf_sigma`.
Production has psf_sigma=0 and does not blur it, but a trial whose code was
reconstructed WITH a PSF must be scored against a model that has one, or the
comparison is rigged against it.  At psf_sigma=0 this is a no-op and the path
is exactly production.

    python paganin_sweep.py
    python paganin_sweep.py --root /data3/cekmekci/AtomiumS2_coded/nfp_sweep
    python paganin_sweep.py --bin 3 --angles 0 1800 3600 5400
"""
import argparse
import glob
import os

import numpy as np

ROOT     = '/data3/cekmekci/AtomiumS2_coded/nfp_psf'
BASE_NFP = '/data3/cekmekci/AtomiumS2_coded/nfp'    # the psf_sigma = 0 reference
PATH_HT  = '/data3/cekmekci/AtomiumS2_coded/ht/codedHT.h5'
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))

# config_steps15.conf, verbatim
DELTA_BETA     = 30.0
ALPHA          = 1e-2
SMOOTH_FWHM    = 17.0          # unbinned detector px
EDGE_CROP      = 36            # unbinned detector px
BG_BOX         = (0.05, 0.15, 0.05, 0.15)
PAGANIN_BG_BOX = (0.15, 0.25, 0.15, 0.25)
NOBJ_FACTOR    = 2560 / 2048   # nobj=2560 against a 2048 detector
ROT_AXIS_OFF   = (0.0, -24.0)  # (row, col) unbinned px
Z1_CODE        = -17.75e-3


def trial_psf(d):
    cfg = os.path.join(d, 'config_step0.conf')
    if os.path.exists(cfg):
        for ln in open(cfg):
            if ln.split('=', 1)[0].strip() == 'psf_sigma':
                return float(ln.split('=', 1)[1].split('#')[0])
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--bin', type=int, default=2, help='2**bin downsample of the detector')
    ap.add_argument('--angles', type=int, nargs='+', default=[0, 1800, 3600, 5400])
    ap.add_argument('--baseline', action='store_true', default=True,
                    help='include the psf_sigma=0 nfp/ result as the reference')
    ap.add_argument('--no-baseline', dest='baseline', action='store_false')
    ap.add_argument('--db', type=float, nargs='*', default=None,
                    help='sweep delta_beta on the first trial, e.g. --db 5 15 30 60 120')
    a = ap.parse_args()

    import cupy as cp
    import cupyx.scipy.ndimage as ndimage
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'image.interpolation': 'none', 'image.resample': False})
    from holotomocupy.propagation import Propagation
    from holotomocupy.psf import psf_taps, psf_blur

    # ---- trials -------------------------------------------------------------
    ts = [(d, trial_psf(d)) for d in sorted(glob.glob(os.path.join(a.root, '*')))
          if os.path.exists(os.path.join(d, 'code.npy'))]
    if a.baseline and os.path.exists(os.path.join(BASE_NFP, 'code.npy')):
        ts.insert(0, (BASE_NFP, 0.0))
    if not ts:
        raise SystemExit(f'no finished trials under {a.root}')

    # ---- geometry, binning --------------------------------------------------
    nb = 2 ** a.bin
    with h5py.File(PATH_HT, 'r') as f:
        energy = float(f['/exchange/energy'][0])
        det_px = float(f['/exchange/detector_pixelsize'][0])
        fdd    = float(f['/exchange/focusdetectordistance'][0])
        z1     = float(f['/exchange/z1'][0])
        ntheta_full, nz_full, n_full = f['/exchange/pdata'].shape
        theta_deg   = f['/exchange/theta'][:].astype('float32')
        shifts_code = f['/exchange/shifts_code'][:].astype('float32')
        shifts_cor  = f['/exchange/shifts_cor'][:].astype('float32')

    nz, n   = nz_full // nb, n_full // nb
    nobj    = round(n * NOBJ_FACTOR)
    nzobj   = round(nz * NOBJ_FACTOR)
    pad8    = nobj // 8
    crop    = round(EDGE_CROP / nb)
    sigma   = SMOOTH_FWHM / (2 * np.sqrt(2 * np.log(2)))      # unbinned px
    pos_obj = (shifts_cor + np.array(ROT_AXIS_OFF, dtype='float32')) / nb

    wavelength     = 1.24e-9 / energy
    distance       = z1 * (fdd - z1) / fdd
    voxelsize_full = det_px / (fdd / z1)
    voxelsize      = voxelsize_full * nb
    z2c  = z1 - Z1_CODE
    distance_code  = ((Z1_CODE * z2c) / (Z1_CODE + z2c)
                      * ((z1 / Z1_CODE) / (Z1_CODE / z1)) ** 2 * (Z1_CODE / z1) ** 2)

    r0, r1, c0, c1 = BG_BOX
    bg_box = (int(r0 * nz), int(r1 * nz), int(c0 * n), int(c1 * n))
    pb = PAGANIN_BG_BOX
    pbox = (int(pb[0] * nzobj), int(pb[1] * nzobj), int(pb[2] * nobj), int(pb[3] * nobj))

    print(f'{len(ts)} trial(s)   bin {nb}   detector {nz}x{n}   object {nzobj}x{nobj}')
    print(f'distance {distance*1e3:.4f} mm   distance_code {distance_code*1e3:.4f} mm   '
          f'voxel {voxelsize*1e9:.3f} nm')
    print(f'delta_beta {DELTA_BETA}  alpha {ALPHA}  smooth sigma {sigma:.3f} unbinned px  '
          f'crop {crop} binned px\n')

    cl_prop      = Propagation(n_full, nz_full, 1, 1, wavelength, voxelsize_full, [distance])
    cl_prop_code = Propagation(n_full, nz_full, 1, 1, wavelength, voxelsize_full, [distance_code])

    # ---- the production helpers, verbatim ------------------------------------
    def multiPaganin(data, dist, wl, vox, delta_beta, alpha):
        fy = cp.fft.fftfreq(data.shape[-2], d=vox).astype('float32')
        fx = cp.fft.fftfreq(data.shape[-1], d=vox).astype('float32')
        fx, fy = cp.meshgrid(fx, fy)
        taylorExp = 1 + wl * dist * cp.pi * delta_beta * (fx ** 2 + fy ** 2)
        num = taylorExp * cp.fft.fft2(data)
        return delta_beta * 0.5 * cp.log(cp.real(cp.fft.ifft2(num / (taylorExp ** 2 + alpha))))

    def fourier_shift_2d(arr, dc, dr):
        nr, nc = arr.shape
        Fc, Fr = cp.meshgrid(cp.fft.fftfreq(nc).astype('float32'),
                             cp.fft.fftfreq(nr).astype('float32'))
        out = cp.fft.ifft2(cp.fft.fft2(arr.astype('complex64'))
                           * cp.exp(-2j * cp.pi * (Fr * dr + Fc * dc)).astype('complex64'))
        return out if arr.dtype.kind == 'c' else out.real.astype('float32')

    def bin2(x):
        if nb == 1:
            return x
        h, w = x.shape
        return x.reshape(h // nb, nb, w // nb, nb).mean(axis=(1, 3))

    def pad_taper(proj, bg):
        nz_, n_ = proj.shape
        py0, px0 = (nzobj - nz_) // 2, (nobj - n_) // 2
        return cp.pad(proj, ((py0, nzobj - nz_ - py0), (px0, nobj - n_ - px0)),
                      'linear_ramp', end_values=bg)

    # ---- per trial ------------------------------------------------------------
    keep = []
    first_padded = first_name = None
    for d, psf in ts:
        name = os.path.basename(d)
        code_gpu = cp.asarray(np.load(os.path.join(d, 'code.npy')).astype('complex64'))
        prb_gpu  = cp.asarray(np.load(os.path.join(d, 'prb.npy')).astype('complex64'))
        ncode = code_gpu.shape[0]
        _, psf_w = psf_taps(psf)

        def code_window(k, G=8):
            dy, dx = int(round(shifts_code[k, 0])), int(round(shifts_code[k, 1]))
            cr, cc = ncode // 2 - dy, ncode // 2 - dx
            w = code_gpu[cr - nz_full // 2 - G: cr + nz_full // 2 + G,
                         cc - n_full // 2 - G: cc + n_full // 2 + G]
            return fourier_shift_2d(w, dc=float(shifts_code[k, 1] - dx),
                                    dr=float(shifts_code[k, 0] - dy))[G:-G, G:-G]

        def flat_field(k):
            I = cp.abs(cl_prop.D(cl_prop_code.D(prb_gpu * code_window(k), 0), 0)) ** 2
            return I if psf_w is None else psf_blur(I, psf_w)

        stages, bg = {}, None
        with h5py.File(PATH_HT, 'r') as f:
            for k in a.angles:
                frame = cp.asarray(f['/exchange/pdata'][k].astype('float32'))
                flat  = flat_field(k)
                num   = ndimage.gaussian_filter(frame, sigma)
                den   = ndimage.gaussian_filter(flat, sigma)
                proj  = bin2(num / (den + 1e-5))
                if bg is None:                      # measured once, on the first angle
                    bg = float(cp.mean(proj[bg_box[0]:bg_box[1], bg_box[2]:bg_box[3]]))
                kept   = proj[crop:nz - crop, crop:n - crop]
                padded = pad_taper(kept, bg)
                p      = cp.pad(padded, pad8, 'reflect')
                phase  = multiPaganin(p, distance, wavelength, voxelsize,
                                      DELTA_BETA, ALPHA)[pad8:-pad8, pad8:-pad8]
                phase  = fourier_shift_2d(phase, dc=-pos_obj[k, 1], dr=-pos_obj[k, 0])
                stages[k] = dict(meas=cp.asnumpy(bin2(frame)), flat=cp.asnumpy(bin2(flat)),
                                 div=cp.asnumpy(proj), pad=cp.asnumpy(padded),
                                 phase=cp.asnumpy(phase), padgpu=padded.copy())
                del frame, flat, num, den, proj, kept, padded, p, phase

        gbg = float(np.mean(stages[a.angles[0]]['phase'][pbox[0]:pbox[1], pbox[2]:pbox[3]]))
        for k in a.angles:
            stages[k]['phase'] -= gbg
        del code_gpu, prb_gpu
        cp.get_default_memory_pool().free_all_blocks()

        dv = stages[a.angles[0]]['div']
        ph = stages[a.angles[0]]['phase']
        # How flat is the background of the PAGANIN PHASE where there is no sample?
        # The sample never reaches the object-grid margin, so the outer frame of the
        # padded object is vacuum by construction and the phase there should be flat.
        # Anything else is already in the seed, before step6 sees it.
        # Background flatness measured INSIDE the real data, outside the sample.
        # NOT in the pad margin -- that is pad_taper's synthetic linear ramp, so its
        # spread measures the ramp rather than anything about the reconstruction.
        mgn = (nobj - (n - 2 * crop)) // 2            # where the measured data starts
        b0, b1 = mgn + 4, mgn + 36                    # a band just inside it, clear of the sample
        band = np.concatenate([ph[b0:b1, b0:-b0].ravel(), ph[-b1:-b0, b0:-b0].ravel(),
                               ph[b0:-b0, b0:b1].ravel(), ph[b0:-b0, -b1:-b0].ravel()])
        core = ph[mgn:-mgn, mgn:-mgn]                 # all of the measured region
        print(f'{name:12s} psf {psf:4.2f}  pad {bg:.5f}  global_bg {gbg:+.5f}  '
              f'division [{dv.min():.4f}, {dv.max():.4f}]  phase ptp(data region) {np.ptp(core):.4f}')
        print(f'{"":12s} background band (in data, off sample): ptp {np.ptp(band):.4f} = '
              f'{100*np.ptp(band)/np.ptp(core):.0f}% of the data-region range, std {band.std():.4f}')

        # WHERE DOES THAT BACKGROUND COME FROM?  multiPaganin is
        #     phase = (db/2) * log( IFFT[ T/(T^2+alpha) * FFT(padded) ] ),  T = 1 + lam*d*pi*db*f^2
        # At low frequency T -> 1, so the filter tends to 1/(1+alpha) = 0.990 -- it does NOT
        # amplify low frequencies.  The background therefore has to be ALREADY in the division,
        # just multiplied by db/2 = 15 on the way out.  This checks that arithmetic: if the
        # prediction matches, the Paganin stage is faithfully reporting an error it was handed,
        # and the fix belongs upstream in the division rather than in the filter.
        dvb = dv[crop + 4:crop + 36, crop + 4:-(crop + 4)]        # same band, in the DIVISION
        pred = DELTA_BETA * 0.5 * np.ptp(np.log(dvb)) / (1 + ALPHA)
        print(f'{"":12s} same band in the DIVISION: ptp {np.ptp(dvb):.5f}  '
              f'-> (db/2)*ptp(log) / (1+alpha) = {pred:.4f}   vs measured {np.ptp(band):.4f}'
              f'   ({100*pred/np.ptp(band):.0f}% explained by plain amplification)')

        # ---- the per-trial sheet: angles down, stages across -------------------
        cols = [('meas', 'measured', 'gray'), ('flat', 'flat (model)', 'gray'),
                ('div', 'division', 'gray'), ('pad', 'padded', 'gray'),
                ('phase', 'Paganin phase', 'gray')]
        fig, ax = plt.subplots(len(a.angles), 5, figsize=(21, 4.3 * len(a.angles)),
                               squeeze=False)
        for i, k in enumerate(a.angles):
            for j, (key, lab, cm) in enumerate(cols):
                img = stages[k][key]
                lo, hi = np.percentile(img, [1, 99])
                im = ax[i][j].imshow(img, cmap=cm, vmin=lo, vmax=hi)
                ax[i][j].set_title(f'angle {k} ({theta_deg[k]:.1f} deg)\n{lab}'
                                   if j == 0 else lab, fontsize=9)
                ax[i][j].axis('off')
                fig.colorbar(im, ax=ax[i][j], fraction=0.046)
        fig.suptitle(f'{name}   psf_sigma {psf:g} px   bin {nb}   '
                     f'delta_beta {DELTA_BETA:g}', fontsize=12)
        plt.tight_layout(rect=(0, 0, 1, 0.97))
        od = os.path.join(OUT_DIR, name)
        os.makedirs(od, exist_ok=True)
        fig.savefig(os.path.join(od, 'stages.png'), dpi=100)
        plt.close(fig)
        print(f'             -> {os.path.join(od, "stages.png")}')
        keep.append((name, psf, stages[a.angles[0]]['div'], stages[a.angles[0]]['phase']))
        if first_padded is None:
            first_padded, first_name = stages[a.angles[0]]['padgpu'], name

    # ---- delta_beta sweep on the first trial ----------------------------------
    # The division is fixed; only multiPaganin reruns. If the background scales linearly
    # with delta_beta then it is pure amplification of something already in the division.
    if a.db:
        k0 = a.angles[0]
        padded = first_padded
        mgn = (nobj - (n - 2 * crop)) // 2
        b0, b1 = mgn + 4, mgn + 36
        print(f'\ndelta_beta sweep, {first_name}, angle {k0}')
        print(f'{"delta_beta":>11} {"phase ptp":>10} {"background ptp":>15} {"bg/db":>9}')
        figs = []
        for db in a.db:
            pp = cp.pad(padded, pad8, 'reflect')
            q = multiPaganin(pp, distance, wavelength, voxelsize, db,
                             ALPHA)[pad8:-pad8, pad8:-pad8]
            q = cp.asnumpy(fourier_shift_2d(q, dc=-pos_obj[k0, 1], dr=-pos_obj[k0, 0]))
            q -= np.mean(q[pbox[0]:pbox[1], pbox[2]:pbox[3]])
            bnd = np.concatenate([q[b0:b1, b0:-b0].ravel(), q[-b1:-b0, b0:-b0].ravel(),
                                  q[b0:-b0, b0:b1].ravel(), q[b0:-b0, -b1:-b0].ravel()])
            core_ = q[mgn:-mgn, mgn:-mgn]
            print(f'{db:11g} {np.ptp(core_):10.4f} {np.ptp(bnd):15.4f} {np.ptp(bnd)/db:9.5f}')
            figs.append((db, q))
        fig, ax = plt.subplots(1, len(figs), figsize=(4.3 * len(figs), 4.8), squeeze=False)
        for j, (db, q) in enumerate(figs):
            lo, hi = np.percentile(q, [1, 99])
            im = ax[0][j].imshow(q, cmap='gray', vmin=lo, vmax=hi)
            ax[0][j].set_title(f'delta_beta {db:g}', fontsize=10); ax[0][j].axis('off')
            fig.colorbar(im, ax=ax[0][j], fraction=0.046)
        fig.suptitle(f'{first_name}, angle {k0}: Paganin phase vs delta_beta '
                     f'(same division throughout)', fontsize=12)
        plt.tight_layout(rect=(0, 0, 1, 0.94))
        out = os.path.join(OUT_DIR, 'db_sweep.png')
        fig.savefig(out, dpi=110); plt.close(fig)
        print(f'-> {out}')
        print('a background that scales linearly with delta_beta (constant bg/db) is pure')
        print('amplification; one that does not is being made by the filter itself')

    # ---- across trials, one colour scale per row ------------------------------
    k0 = a.angles[0]
    dlo, dhi = np.percentile(keep[0][2], [1, 99])
    plo, phi = np.percentile(keep[0][3], [1, 99])
    fig, ax = plt.subplots(2, len(keep), figsize=(4.3 * len(keep), 9), squeeze=False)
    for j, (name, psf, div, ph) in enumerate(keep):
        for i, (img, lo, hi, lab) in enumerate(((div, dlo, dhi, 'division'),
                                                (ph, plo, phi, 'Paganin phase'))):
            im = ax[i][j].imshow(img, cmap='gray', vmin=lo, vmax=hi)
            ax[i][j].set_title(f'{name}\npsf {psf:g}\n{lab}' if i == 0 else lab, fontsize=9)
            ax[i][j].axis('off')
            fig.colorbar(im, ax=ax[i][j], fraction=0.046)
    fig.suptitle(f'angle {k0}, every trial, one colour scale per row '
                 f'(set by {keep[0][0]})', fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(OUT_DIR, 'summary.png')
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f'\n-> {out}')


if __name__ == '__main__':
    main()
