#!/usr/bin/env python
"""Separate the causes of the flat-field mismatch by their frequency signature.

`analyse.py` showed 75% of the residual moves with the aperture. That lumps
together four different faults, because a PSF error and a propagation error
both act on the CODE's structure and so move with it:

    wrong PSF                |H| rolls off monotonically,  arg(H) = 0
    sub-pixel registration   |H| = 1,                      arg(H) linear in f
    wrong distance_code      |H| OSCILLATES in f^2, with poles
    wrong code contrast      |H| flat but != 1

The oscillation is the discriminator. Under the CTF approximation a pure phase
object at distance z gives I(f) ~ 1 + 2 sin(pi lam z f^2) Phi(f), so modelling
the wrong distance gives

    H(f) = sin(pi lam z_true f^2) / sin(pi lam z_model f^2)

which oscillates and blows up wherever the modelled sine vanishes. A blur is
exp(-2 pi^2 sigma^2 f^2) -- smooth and monotonic. Nothing about a blur can
imitate a pole, so these two are distinguishable from one frame.

Measured on pref0, which is SAMPLE-FREE, unbinned and never went through
Paganin -- so none of the stage-5 machinery can contaminate the answer.

    python transfer.py
"""
import argparse
import os

import numpy as np

HT   = '/data3/cekmekci/AtomiumS2_coded/ht/codedHT.h5'
NFP  = '/data3/cekmekci/AtomiumS2_coded/nfp'
OUT  = os.path.dirname(os.path.abspath(__file__))
CROP = 160
SY, SX = -133, 235          # aperture position in pref0, from analyse.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trial', default=NFP, help='folder holding code.npy / prb.npy')
    ap.add_argument('--psf', type=float, default=0.0,
                    help='blur the MODEL by this sigma, detector px. Use the sigma the '
                         'trial was reconstructed with, or the beamline value.')
    ap.add_argument('--clip', action='store_true',
                    help='enforce |code| <= 1 before use. Physical for a passive absorber, '
                         'and unlike a low-pass it removes the deconvolution overshoot '
                         'without touching frequency content -- which matters here, since '
                         'smoothing would reintroduce the very blur we are testing for.')
    ap.add_argument('--tag', default='', help='suffix for the output figure')
    a = ap.parse_args()

    import cupy as cp
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from holotomocupy.propagation import Propagation
    matplotlib.rcParams.update({'image.interpolation': 'none', 'image.resample': False})

    with h5py.File(HT, 'r') as f:
        E   = float(f['/exchange/energy'][0]); dpx = float(f['/exchange/detector_pixelsize'][0])
        fdd = float(f['/exchange/focusdetectordistance'][0]); z1 = float(f['/exchange/z1'][0])
        p0  = f['/exchange/pref0'][:].astype('float32')
    with h5py.File(os.path.join(NFP, 'data_nfp.h5'), 'r') as f:
        z1c = float(f['/exchange/z1'][0])
    nz, n = p0.shape
    lam   = 1.24e-9 / E
    dist  = z1 * (fdd - z1) / fdd
    vox   = dpx * z1 / fdd
    dcode = (z1 - z1c) * z1 / z1c

    from holotomocupy.psf import psf_taps, psf_blur
    code = cp.asarray(np.load(os.path.join(a.trial, 'code.npy')).astype('complex64'))
    prb  = cp.asarray(np.load(os.path.join(a.trial, 'prb.npy')).astype('complex64'))
    nc   = code.shape[0]
    amax = float(cp.abs(code).max())
    print(f'code/prb from {a.trial}')
    print(f'  |code| max {amax:.4f}' + ('   NON-PHYSICAL for a passive absorber' if amax > 1.01 else ''))
    if a.clip:
        mag = cp.abs(code)
        code = cp.where(mag > 1.0, code / cp.maximum(mag, 1e-9), code)
        print(f'  clipped to |code| <= 1: {100*float((mag > 1.0).mean()):.1f}% of pixels were over')
    _, psf_w = psf_taps(a.psf)
    print(f'  model blurred by sigma = {a.psf:g} px' if psf_w is not None
          else '  model NOT blurred (psf 0)')
    D    = Propagation(n, nz, 1, 1, lam, vox, [dist])
    Dc   = Propagation(n, nz, 1, 1, lam, vox, [dcode])
    cr, cx = nc // 2 - SY, nc // 2 - SX
    w = code[cr - nz // 2:cr + nz // 2, cx - n // 2:cx + n // 2]
    _I = cp.abs(D.D(Dc.D(prb * w, 0), 0)) ** 2
    model = cp.asnumpy(_I if psf_w is None else psf_blur(_I, psf_w))
    del _I
    del code, prb, w
    cp.get_default_memory_pool().free_all_blocks()

    c_ = (slice(CROP, nz - CROP), slice(CROP, n - CROP))
    M, P = model[c_], p0[c_]
    S = M.shape[0]

    # ---- complex transfer, radially binned ---------------------------------
    # H(f) = < Pf conj(Mf) > / < |Mf|^2 >, the least-squares filter taking model
    # to measurement in each shell. Mean removed so DC does not dominate.
    Pf = np.fft.fft2(P - P.mean())
    Mf = np.fft.fft2(M - M.mean())
    yy, xx = np.mgrid[:S, :S]
    fy = np.fft.fftfreq(S)[yy]; fx = np.fft.fftfreq(S)[xx]
    fr = np.hypot(fy, fx)
    nbin = 140
    idx = np.clip((fr / 0.5 * nbin).astype(int), 0, nbin - 1)
    # HALF-PLANE ONLY.  For a real image Pf(-f) = conj(Pf(f)), so a full radial
    # shell pairs every point with its conjugate and the imaginary part of the
    # cross-spectrum cancels identically -- arg(H) would come out 0.00 at every
    # frequency no matter what the data did. Keeping fx > 0 preserves it.
    half = (fx > 0)
    cross = (Pf * np.conj(Mf))[half]
    ih, mh = idx[half], (np.abs(Mf) ** 2)[half]
    num = np.bincount(ih.ravel(), cross.real.ravel(), nbin) \
        + 1j * np.bincount(ih.ravel(), cross.imag.ravel(), nbin)
    den = np.bincount(ih.ravel(), mh.ravel(), nbin)
    cnt = np.bincount(ih.ravel(), minlength=nbin)
    ok  = (cnt > 20) & (den > 0)
    H   = np.where(ok, num / np.maximum(den, 1e-30), np.nan)
    f   = (np.arange(nbin) + 0.5) / nbin * 0.5
    pw  = den / np.maximum(cnt, 1)                     # model power per shell

    # weight the fits by model power: shells where the model has no signal
    # carry no information about H
    band = ok & (f > 0.01) & (f < 0.45) & (pw > 0)
    print('=' * 70)
    print('TRANSFER FUNCTION  H(f) = <Pf conj(Mf)> / <|Mf|^2>   on pref0')
    print('=' * 70)
    print(f'{"cyc/px":>8} {"period px":>10} {"|H|":>8} {"arg(H) deg":>11} {"model power":>12}')
    for i in range(0, nbin, 7):
        if ok[i] and f[i] < 0.46:
            print(f'{f[i]:8.4f} {1/f[i]:10.1f} {abs(H[i]):8.4f} '
                  f'{np.degrees(np.angle(H[i])):11.2f} {pw[i]/np.nanmax(pw):12.4f}')

    # ---- hypothesis 1: a blur -----------------------------------------------
    m = band & (np.abs(H) > 0)
    A_sig = np.polyfit(f[m] ** 2, np.log(np.abs(H[m])), 1, w=pw[m])
    sig2 = -A_sig[0] / (2 * np.pi ** 2)
    sig  = np.sqrt(sig2) if sig2 > 0 else float('nan')
    print(f'\nBLUR fit  |H| = A exp(-2 pi^2 sigma^2 f^2)')
    print(f'   A = {np.exp(A_sig[1]):.4f}   sigma = {sig:.3f} px'
          + ('   (negative sigma^2 -- the model is BLURRIER than the data)'
             if sig2 <= 0 else ''))
    resid_blur = np.log(np.abs(H[m])) - np.polyval(A_sig, f[m] ** 2)
    print(f'   residual scatter about that fit: {resid_blur.std():.4f} in log|H|')

    # ---- hypothesis 2: a shift ---------------------------------------------
    ph = np.unwrap(np.angle(H[m]))
    sl = np.polyfit(f[m], ph, 1, w=pw[m])
    print(f'\nSHIFT fit  arg(H) = -2 pi f . d')
    print(f'   implied |d| = {abs(sl[0]) / (2 * np.pi):.4f} px'
          f'   (a radial fit only bounds the magnitude)')
    print(f'   phase rms {np.degrees(ph).std():.2f} deg -- near zero means no net shift')

    # ---- hypothesis 3: a propagation-distance error -------------------------
    # H = sin(pi lam z_true f^2) / sin(pi lam z_model f^2). Scan z_true and see
    # which, if any, explains |H| better than the blur does.
    print(f'\nDISTANCE fit  H = sin(pi lam z1 f^2)/sin(pi lam z0 f^2), z0 = {dcode*1e3:.4f} mm')
    f_m = f[m] / vox                                   # cycles per metre
    best = None
    for frac in np.linspace(0.90, 1.10, 201):
        zt = dcode * frac
        pred = np.sin(np.pi * lam * zt * f_m ** 2) / np.sin(np.pi * lam * dcode * f_m ** 2)
        good = np.isfinite(pred) & (np.abs(pred) > 1e-6)
        if good.sum() < 20:
            continue
        r = np.sum(pw[m][good] * (np.log(np.abs(H[m][good])) - np.log(np.abs(pred[good]))) ** 2)
        r /= pw[m][good].sum()
        if best is None or r < best[0]:
            best = (r, frac, zt)
    r0 = np.sum(pw[m] * resid_blur ** 2) / pw[m].sum()
    print(f'   best z = {best[2]*1e3:.4f} mm  ({100*(best[1]-1):+.2f}% off)'
          f'   weighted residual {best[0]:.4f}')
    print(f'   the BLUR hypothesis scores {r0:.4f} on the same measure')
    print('   -> ' + ('the distance error explains it better'
                      if best[0] < r0 else 'the blur explains it better'))

    # ---- figure --------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    ax[0].semilogy(f[ok], np.abs(H[ok]), 'k.', ms=4, label='measured |H|')
    ax[0].semilogy(f[m], np.exp(np.polyval(A_sig, f[m] ** 2)), 'r-',
                   label=f'blur fit, sigma={sig:.2f} px')
    pred = np.abs(np.sin(np.pi * lam * best[2] * f_m ** 2)
                  / np.sin(np.pi * lam * dcode * f_m ** 2))
    ax[0].semilogy(f[m], pred, 'b-', alpha=.7, label=f'distance fit, {100*(best[1]-1):+.1f}%')
    ax[0].axhline(1, color='0.6', lw=.8)
    ax[0].set(xlabel='cycles / px', ylabel='|H|', ylim=(1e-2, 1e2),
              title='amplitude: blur is monotonic,\na distance error oscillates')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    ax[1].plot(f[ok], np.degrees(np.angle(H[ok])), 'k.', ms=4)
    ax[1].axhline(0, color='r', lw=1)
    ax[1].set(xlabel='cycles / px', ylabel='arg(H) [deg]', ylim=(-180, 180),
              title='phase: a translation would be a\nstraight line through the origin')
    ax[1].grid(alpha=.3)

    ax[2].loglog(f[ok], pw[ok] / np.nanmax(pw), 'k-')
    ax[2].axvspan(0.01, 0.45, color='g', alpha=.08, label='fit band')
    ax[2].set(xlabel='cycles / px', ylabel='model power (normalised)',
              title='where the model has signal\n(H is meaningless elsewhere)')
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3, which='both')
    plt.tight_layout()
    _out = os.path.join(OUT, f'fig6_transfer{a.tag}.png')
    fig.savefig(_out, dpi=110); plt.close(fig)
    print(f'\n-> {_out}')


if __name__ == '__main__':
    main()
