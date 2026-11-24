#!/usr/bin/env python
"""Which detector blur best explains pref0, with the code and probe held fixed?

NOTHING IS RECONSTRUCTED HERE.  `code.npy` and `prb.npy` are taken as given and
the PSF is applied to the FORWARD model only:

    model(sigma) = K_sigma( |D( D_code( prb * code_window ) )|^2 )

against the measured `pref0`.  That is the whole point.  Putting the PSF inside
step0 instead turns it into a deconvolution, and step0 constrains neither the
code nor the probe, so it runs away -- measured 2026-11-20, |code| reached 2.29
against the hard bound of 1 for a passive gold mask, worsening monotonically
with sigma.  Here there is no inverse problem, so there is nothing to diverge.

WHAT THIS MEASURES.  `code.npy` was reconstructed with psf_sigma=0, so the
solver already absorbed part of the detector blur into it.  The sigma that wins
here is therefore the RESIDUAL model-vs-measurement mismatch, not the detector's
true PSF.  That is the number the HT division actually needs; the true PSF is a
separate question and needs a constrained step0 or a direct measurement.

pref0 is HELD OUT -- step0 never saw it -- so an improvement cannot come from
the extra blur merely giving a solver more freedom.  Nothing is being fitted.

The code is registered ONCE at sigma=0 and that position is reused for every
trial, because the aperture does not move when you change the model's blur.
Re-registering per sigma is how the earlier collect.py run produced nonsense
positions off blown-up inputs.

    python psf_forward.py
    python psf_forward.py --sigma 0 0.5 1 1.5 2 2.5 3
    python psf_forward.py --trial /data3/cekmekci/AtomiumS2_coded/nfp_psf/psf_1p20
"""
import argparse
import os

import numpy as np

BASE_NFP = '/data3/cekmekci/AtomiumS2_coded/nfp'
PATH_HT  = '/data3/cekmekci/AtomiumS2_coded/ht/codedHT.h5'
CROP     = 128          # px dropped at every edge; the modelled rim is unreliable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sigma', type=float, nargs='+',
                    default=[0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4],
                    help='trial PSF sigmas in UNBINNED detector px')
    ap.add_argument('--trial', default=BASE_NFP,
                    help='folder holding code.npy and prb.npy')
    ap.add_argument('--view', type=int, default=512, help='centre crop shown 1:1')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    import cupy as cp
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['image.interpolation'] = 'none'
    matplotlib.rcParams['image.resample'] = False
    from cupyx.scipy.ndimage import gaussian_filter
    from holotomocupy.propagation import Propagation
    from holotomocupy.psf import psf_taps, psf_blur

    with h5py.File(PATH_HT, 'r') as f:
        energy = float(f['/exchange/energy'][0])
        det_px = float(f['/exchange/detector_pixelsize'][0])
        fdd    = float(f['/exchange/focusdetectordistance'][0])
        z1     = float(f['/exchange/z1'][0])
        pref0  = f['/exchange/pref0'][:].astype('float32')
    with h5py.File(os.path.join(BASE_NFP, 'data_nfp.h5'), 'r') as f:
        z1c = float(f['/exchange/z1'][0])

    nz, n = pref0.shape
    wavelength    = 1.24e-9 / energy
    distance      = z1 * (fdd - z1) / fdd
    voxelsize     = det_px / (fdd / z1)
    distance_code = (z1 - z1c) * z1 / z1c

    code = cp.asarray(np.load(os.path.join(a.trial, 'code.npy')).astype('complex64'))
    prb  = cp.asarray(np.load(os.path.join(a.trial, 'prb.npy')).astype('complex64'))
    ncode = code.shape[0]
    D     = Propagation(n, nz, 1, 1, wavelength, voxelsize, [distance])
    Dcode = Propagation(n, nz, 1, 1, wavelength, voxelsize, [distance_code])

    print(f'code/prb from {a.trial}')
    print(f'z1 {z1 * 1e3:.4f} mm  z1c {z1c * 1e3:.4f} mm  distance {distance * 1e3:.4f} mm  '
          f'distance_code {distance_code * 1e3:.4f} mm')

    # --- register the code once, at sigma = 0 -----------------------------
    ref = cp.abs(D.D(Dcode.D(prb, 0), 0)) ** 2
    P8  = Propagation(ncode, ncode, 1, 1, wavelength, voxelsize, [distance])
    P8c = Propagation(ncode, ncode, 1, 1, wavelength, voxelsize, [distance_code])
    cdata = cp.abs(P8.D(P8c.D(code, 0), 0)) ** 2
    del P8, P8c
    cp.get_default_memory_pool().free_all_blocks()
    pad = (ncode - n) // 2
    rpad = cp.pad(cp.asarray(pref0) / (ref + 1e-6), ((pad, pad), (pad, pad)), constant_values=1)
    ccm = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fft2(cdata) * cp.conj(cp.fft.fft2(rpad))).real)
    iy, ix = (int(v.get()) for v in cp.unravel_index(cp.argmax(ccm), ccm.shape))
    sy, sx = -(iy - ncode // 2), -(ix - ncode // 2)
    del rpad, ccm, cdata, ref
    cp.get_default_memory_pool().free_all_blocks()

    lim = ncode // 2 - max(nz, n) // 2
    if abs(sy) > lim or abs(sx) > lim:
        raise SystemExit(f'registration failed: code at ({sy}, {sx}), outside +-{lim}. '
                         'That means the code/probe in --trial are not usable here.')
    print(f'code registered at ({sy:+d}, {sx:+d}) px, reused for every sigma\n')

    cr, cx = ncode // 2 - sy, ncode // 2 - sx
    w = code[cr - nz // 2: cr + nz // 2, cx - n // 2: cx + n // 2]
    sharp = cp.abs(D.D(Dcode.D(prb * w, 0), 0)) ** 2        # the model, computed ONCE
    del code, prb, w, D, Dcode
    cp.get_default_memory_pool().free_all_blocks()

    # --- sweep: the only thing that changes is the blur on the model ------
    meas = cp.asarray(pref0)
    c_ = (slice(CROP, nz - CROP), slice(CROP, n - CROP))
    hp = lambda x: x - gaussian_filter(x, 16.0)             # isolate the code texture

    print(f'{"sigma":>7} {"rms(diff)":>11} {"rms(hp diff)":>13} {"ratio mean":>11} '
          f'{"ratio std":>10}')
    rows = []
    for s in a.sigma:
        _, taps = psf_taps(s)
        mod = sharp if taps is None else psf_blur(sharp, taps)
        dif = (meas - mod)[c_]
        rat = (meas / (mod + 1e-6))[c_]
        r = (float(cp.sqrt(cp.mean(dif ** 2))), float(cp.sqrt(cp.mean(hp(dif) ** 2))),
             float(cp.mean(rat)), float(cp.std(rat)))
        print(f'{s:7.2f} {r[0]:11.5f} {r[1]:13.5f} {r[2]:11.5f} {r[3]:10.5f}')
        rows.append((s, cp.asnumpy(meas - mod), cp.asnumpy(meas / (mod + 1e-6))))
        del mod, dif, rat

    best = min(range(len(rows)), key=lambda i: float(np.std(rows[i][2][c_])))
    print(f'\nlowest ratio scatter at sigma = {rows[best][0]:g} '
          f'(ratio std {float(np.std(rows[best][2][c_])):.5f} vs '
          f'{float(np.std(rows[0][2][c_])):.5f} at sigma = 0)')
    if best in (0, len(rows) - 1):
        print('  that is at the EDGE of the bracket -- extend --sigma in that direction')

    # --- the picture ------------------------------------------------------
    v = a.view
    cz = (slice(nz // 2 - v // 2, nz // 2 + v // 2), slice(n // 2 - v // 2, n // 2 + v // 2))
    dv = float(np.percentile(np.abs(rows[0][1][c_]), 99))   # limits from sigma = 0
    rr = float(np.percentile(np.abs(rows[0][2][c_] - 1), 99))
    ncol = len(rows)
    fig, ax = plt.subplots(2, ncol, figsize=(3.6 * ncol, 7.8), squeeze=False)
    for j, (s, dif, rat) in enumerate(rows):
        for i, (img, lo, hi, lab) in enumerate(
                ((dif[cz], -dv, dv, 'measured - model'),
                 (rat[cz], 1 - rr, 1 + rr, 'measured / model'))):
            im = ax[i][j].imshow(img, cmap='seismic', vmin=lo, vmax=hi)
            ax[i][j].set_title(f'sigma {s:g}\n{lab}' if i == 0 else lab, fontsize=9)
            ax[i][j].axis('off')
            fig.colorbar(im, ax=ax[i][j], fraction=0.046)
    fig.suptitle(f'pref0 against the forward model, code/prb fixed from '
                 f'{os.path.basename(a.trial)}  --  {v} px centre crop 1:1, '
                 f'one colour scale across sigma', fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    out = a.out or os.path.join(a.trial, 'psf_forward.png')
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f'\n-> {out}')


if __name__ == '__main__':
    main()
