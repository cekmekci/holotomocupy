#!/usr/bin/env python
"""Where does the NFP residual live, and is it the probe?

Three measurements on `measured - model` over several NFP frames, using the
solver's own windowed shifter so the model is the one step0 actually fitted.

1. rms(residual) vs DISTANCE FROM THE FRAME EDGE.  `Propagation` pads 2x with a
   symmetric (mirrored) continuation, so the outer margin of every modelled
   frame rests on a guess rather than on the real field, over roughly
   sqrt(lambda*|d|)/voxelsize ~ 19 px here.  If the residual dies within a few
   tens of px it is that, and CROP=128 already excludes it.  If it runs
   hundreds of px deep the edge is a symptom and not the cause.

2. CORRELATION BETWEEN RESIDUALS AT DIFFERENT POSITIONS.  The probe is fixed in
   the detector frame; the code moves.  So a residual that is the same map at
   every position is probe-side, and one that decorrelates moves with the code.
   This needs no model of what the stripes are.

3. CORRELATION WITH THE PROBE-ONLY INTENSITY |D(prb)|^2.  Direct test of
   "are those stripes the probe".

Nothing is normalized; residuals are plain `measured - model`.

    python residual_anatomy.py
    python residual_anatomy.py --nk 9 --trial /data3/cekmekci/AtomiumS2_coded/nfp
"""
import argparse
import os

import numpy as np

CROP_REF = 128        # the crop step0_sanity uses for its statistics, drawn for reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trial', default='/data3/cekmekci/AtomiumS2_coded/nfp',
                    help='folder holding code.npy, prb.npy, pos_nfp.npy, data_nfp.h5')
    ap.add_argument('--nk', type=int, default=9, help='frames, spread by |pos|')
    ap.add_argument('--out', default=None, help='png path (default: inside --trial)')
    a = ap.parse_args()

    import cupy as cp
    import cupyx.scipy.ndimage as ndimage
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from holotomocupy.propagation import Propagation
    from holotomocupy.shift_fft_code import ShiftFFTCode, window_pad

    d = a.trial
    h5 = os.path.join(d, 'data_nfp.h5')
    with h5py.File(h5, 'r') as f:
        energy = float(f['/exchange/energy'][0])
        fdd    = float(f['/exchange/focusdetectordistance'][0])
        z1     = float(f['/exchange/z1'][0])
        vox    = float(f['/exchange/voxelsize'][0])
        shifts = f['/exchange/shifts'][:].astype('float32')
        npos, nz, n = f['/exchange/pdata'].shape

    wavelength = 1.24e-9 / energy
    distance = z1 * (fdd - z1) / fdd
    fres_px = np.sqrt(wavelength * abs(distance)) / vox
    print(f'{d}\n  distance {distance*1e3:.4f} mm  voxel {vox*1e9:.3f} nm  '
          f'Fresnel length sqrt(lam*|d|)/v = {fres_px:.1f} px')

    start = shifts.copy()
    start[:, 0] -= (start[:, 0].max() + start[:, 0].min()) / 2
    start[:, 1] -= (start[:, 1].max() + start[:, 1].min()) / 2
    off_all = np.round(start).astype('int32')
    pos = start
    p_ref = os.path.join(d, 'pos_nfp.npy')
    if os.path.exists(p_ref):
        r = np.load(p_ref).astype('float32')
        if r.shape == pos.shape:
            pos = r
    rad = np.hypot(pos[:, 0], pos[:, 1])
    sel = sorted({int(np.argmin(np.abs(rad - t)))
                  for t in np.linspace(rad.min(), rad.max(), a.nk)})
    print(f'  {len(sel)} frames: {sel}')

    code_gpu = cp.asarray(np.load(os.path.join(d, 'code.npy')).astype('complex64'))
    prb_gpu  = cp.asarray(np.load(os.path.join(d, 'prb.npy')).astype('complex64'))
    ncode = code_gpu.shape[0]
    Dnfp = Propagation(n, nz, 1, 1, wavelength, vox, [distance])
    pad_y, pad_x = window_pad(n, nz, 0.125)
    sh = ShiftFFTCode(n, n + 2 * pad_x, nz, nz + 2 * pad_y, nchunk=1,
                      pad=(pad_y, pad_x), code_shape=(ncode, ncode))
    m1 = sh.unit_mag(1)

    prb_only = cp.asnumpy(cp.abs(Dnfp.D(prb_gpu, 0)) ** 2)      # probe signature at the detector

    R, M = [], []
    with h5py.File(h5, 'r') as f:
        for k in sel:
            rk = cp.asarray(pos[k:k + 1], dtype='float32')
            ok = cp.asarray(off_all[k:k + 1], dtype='int32')
            sh.check_drift(rk, ok)
            w = sh.S(code_gpu, rk, m1, ok)[0]
            mod = cp.abs(Dnfp.D(prb_gpu * w, 0)) ** 2
            meas = cp.asarray(f['/exchange/pdata'][k].astype('float32'))
            R.append(cp.asnumpy(meas - mod)); M.append(cp.asnumpy(meas))
            del w, mod, meas
    del code_gpu, prb_gpu, sh
    cp.get_default_memory_pool().free_all_blocks()
    R = np.stack(R); M = np.stack(M)

    # --- 1. residual vs distance from the frame edge -------------------------
    yy, xx = np.mgrid[:nz, :n]
    dist = np.minimum(np.minimum(yy, nz - 1 - yy), np.minimum(xx, n - 1 - xx))
    edges = np.unique(np.round(np.geomspace(1, min(nz, n) // 2, 40)).astype(int))
    prof = np.array([[np.sqrt((r[(dist >= lo) & (dist < hi)] ** 2).mean())
                      for lo, hi in zip(edges[:-1], edges[1:])] for r in R])
    ctr = 0.5 * (edges[1:] + edges[:-1])
    deep = prof[:, ctr > 400].mean()
    print(f'\n1. residual rms vs depth from the edge (plateau beyond 400 px = {deep:.5f})')
    for c, v in zip(ctr, prof.mean(0)):
        if c < 40 or c in ctr[::6]:
            print(f'   {c:7.1f} px   {v:.5f}   x{v/deep:5.2f} of the plateau')

    # --- 2. do residuals at different code positions agree? ------------------
    c_ = (slice(CROP_REF, nz - CROP_REF), slice(CROP_REF, n - CROP_REF))
    V = np.stack([r[c_].ravel() for r in R])
    V = V - V.mean(1, keepdims=True)
    C = (V @ V.T) / np.outer(np.linalg.norm(V, axis=1), np.linalg.norm(V, axis=1))
    off_diag = C[~np.eye(len(sel), dtype=bool)]
    fixed = R[:, c_[0], c_[1]].mean(0)
    frac = fixed.var() / R[:, c_[0], c_[1]].var(axis=0).mean()
    print(f'\n2. residual correlation between code positions: '
          f'mean {off_diag.mean():+.4f}  min {off_diag.min():+.4f}  max {off_diag.max():+.4f}')
    print(f'   variance of the frame-averaged (fixed) residual / mean per-frame variance: {frac:.4f}')
    print('   high = fixed in the detector frame (probe-side); ~0 = moves with the code')

    # --- 3. is it the probe? -------------------------------------------------
    p = prb_only[c_] - prb_only[c_].mean()
    cp_fix = float((fixed - fixed.mean()).ravel() @ p.ravel()
                   / (np.linalg.norm(fixed - fixed.mean()) * np.linalg.norm(p)))
    print(f'\n3. corr(fixed residual, |D(prb)|^2) = {cp_fix:+.4f}')
    per = [float((r[c_] - r[c_].mean()).ravel() @ p.ravel()
                 / (np.linalg.norm(r[c_] - r[c_].mean()) * np.linalg.norm(p))) for r in R]
    print(f'   per frame: ' + ' '.join(f'{v:+.3f}' for v in per))

    # --- 4. is what is left simply photon noise? -----------------------------
    # Two independent readings of the same question.
    #
    # (a) Counting. If the residual were pure Poisson, rms/signal = 1/sqrt(q),
    #     so it implies q quanta per pixel. The raw frame says how many ADU sit
    #     above dark. Their ratio is the detector gain the residual would need.
    #     If that lands near the measured ~5.6 ADU/quantum it is noise.
    # (b) Spectrum. Poisson noise is white before the detector PSF and only
    #     gently shaped after it. Model error would instead put its power where
    #     the code has features, i.e. follow the model's own spectrum.
    with h5py.File(h5, 'r') as f:
        dark = f['/exchange/data_dark'][:].astype('float32').mean(0) if '/exchange/data_dark' in f else None
        raw = np.stack([f['/exchange/data'][k].astype('float32') for k in sel]) \
            if '/exchange/data' in f else None
    print('\n4a. noise accounting')
    if raw is None or dark is None:
        print('    no /exchange/data or /exchange/data_dark -- skipped')
    else:
        adu = (raw - dark)[:, c_[0], c_[1]]
        rel = np.array([np.sqrt((r[c_] ** 2).mean()) / M[i][c_].mean()
                        for i, r in enumerate(R)])
        q_needed = 1.0 / rel ** 2
        gain = adu.reshape(len(sel), -1).mean(1) / q_needed
        print(f'    mean ADU above dark      {adu.reshape(len(sel),-1).mean(1).mean():10.1f}')
        print(f'    residual rms / signal    {rel.mean():10.4f}')
        print(f'    quanta it implies        {q_needed.mean():10.1f} per px')
        print(f'    => gain the residual needs {gain.mean():8.3f} ADU/quantum '
              f'(per frame {" ".join(f"{g:.2f}" for g in gain)})')
        print('    measured detector gain is ~5.6 ADU/quantum; close = the residual IS noise')

    def radial_psd(img):
        F = np.abs(np.fft.fftshift(np.fft.fft2(img - img.mean()))) ** 2
        s = img.shape[0]
        ry, rx = np.mgrid[:s, :s] - s // 2
        rr = np.hypot(ry, rx).astype(int)
        return np.bincount(rr.ravel(), F.ravel()) / np.maximum(np.bincount(rr.ravel()), 1)

    S = min(1024, nz - 2 * CROP_REF)
    sl = (slice(nz // 2 - S // 2, nz // 2 + S // 2), slice(n // 2 - S // 2, n // 2 + S // 2))
    psd_r = np.mean([radial_psd(r[sl]) for r in R], axis=0)[:S // 2]
    psd_m = np.mean([radial_psd(m[sl]) for m in M], axis=0)[:S // 2]
    fq = np.arange(len(psd_r)) / S                              # cycles per px
    band = (fq > 0.02) & (fq < 0.45)
    slope = np.polyfit(np.log(fq[band]), np.log(psd_r[band]), 1)[0]
    print(f'\n4b. radial power spectrum over {S}^2, band 0.02-0.45 cyc/px')
    print(f'    residual log-log slope {slope:+.3f}   (0 = white = noise; '
          f'negative = structured like the signal)')
    print(f'    model    log-log slope '
          f'{np.polyfit(np.log(fq[band]), np.log(psd_m[band]), 1)[0]:+.3f}')

    # --- figure --------------------------------------------------------------
    fig, ax = plt.subplots(1, 5, figsize=(28, 5))
    for i, k in enumerate(sel):
        ax[0].loglog(ctr, prof[i], lw=0.7, alpha=0.5)
    ax[0].loglog(ctr, prof.mean(0), 'k-o', ms=3, label='mean')
    ax[0].axvline(fres_px, color='r', ls='--', lw=1, label=f'Fresnel {fres_px:.0f} px')
    ax[0].axvline(CROP_REF, color='b', ls=':', lw=1, label=f'CROP {CROP_REF}')
    ax[0].set(xlabel='distance from the frame edge [px]', ylabel='rms(measured - model)',
              title='1. does the residual live at the edge?')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which='both')

    im = ax[1].imshow(C, cmap='RdBu_r', vmin=-1, vmax=1)
    ax[1].set(title=f'2. residual correlation across code positions\nmean off-diagonal '
                    f'{off_diag.mean():+.3f}', xticks=range(len(sel)), yticks=range(len(sel)))
    ax[1].set_xticklabels(sel, fontsize=6, rotation=90); ax[1].set_yticklabels(sel, fontsize=6)
    fig.colorbar(im, ax=ax[1], fraction=0.046)

    v = np.percentile(np.abs(fixed), 99)
    im = ax[2].imshow(fixed, cmap='seismic', vmin=-v, vmax=v)
    ax[2].set_title(f'frame-averaged residual (the FIXED part)\n{100*frac:.1f}% of the variance',
                    fontsize=10); ax[2].axis('off'); fig.colorbar(im, ax=ax[2], fraction=0.046)

    lo, hi = np.percentile(prb_only[c_], [1, 99])
    im = ax[3].imshow(prb_only[c_], cmap='gray', vmin=lo, vmax=hi)
    ax[3].set_title(f'3. probe only, |D(prb)|^2\ncorr with the fixed residual {cp_fix:+.3f}',
                    fontsize=10); ax[3].axis('off'); fig.colorbar(im, ax=ax[3], fraction=0.046)

    ax[4].loglog(fq[1:], psd_r[1:], 'r-', label='residual')
    ax[4].loglog(fq[1:], psd_m[1:] * psd_r[band].mean() / psd_m[band].mean(), 'k-',
                 alpha=0.6, label='measured frame (scaled)')
    ax[4].axhline(psd_r[band].mean(), color='b', ls='--', lw=1, label='white')
    ax[4].set(xlabel='cycles / px', ylabel='power',
              title=f'4. residual spectrum, slope {slope:+.2f}\n0 = noise, negative = structure')
    ax[4].legend(fontsize=8); ax[4].grid(alpha=0.3, which='both')

    plt.tight_layout()
    out = a.out or os.path.join(d, 'residual_anatomy.png')
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f'\n-> {out}')


if __name__ == '__main__':
    main()
