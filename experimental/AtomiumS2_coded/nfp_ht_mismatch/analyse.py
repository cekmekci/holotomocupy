#!/usr/bin/env python
"""Split the HT flat-field residual into a probe part and a code part.

THE IDEA.  codedHT.h5 holds two SAMPLE-FREE frames: pref0 taken before the scan
and pref1 after.  The coded aperture sits in a different place in each.  The
probe and the detector do not move between them; the aperture does.  So:

    whatever is the SAME in the two residuals  -> probe or detector
    whatever DIFFERS                           -> the aperture model

No sweep, no reconstruction, no fitted parameter.  The only inputs are the two
measured flats and the forward model built from code.npy / prb.npy.

Writes every intermediate as a figure next to this script, and prints every
number the README quotes.

    python analyse.py
"""
import os

import numpy as np

HT   = '/data3/cekmekci/AtomiumS2_coded/ht/codedHT.h5'
NFP  = '/data3/cekmekci/AtomiumS2_coded/nfp'
OUT  = os.path.dirname(os.path.abspath(__file__))
CROP = 160          # px dropped at each edge; the modelled rim is unreliable
HPS  = 8            # high-pass sigma used only for REGISTRATION, never for the residual


def main():
    import cupy as cp
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from cupyx.scipy.ndimage import gaussian_filter
    from holotomocupy.propagation import Propagation
    matplotlib.rcParams.update({'image.interpolation': 'none', 'image.resample': False})

    # ---- 1. the two sample-free frames -------------------------------------
    with h5py.File(HT, 'r') as f:
        E    = float(f['/exchange/energy'][0])
        dpx  = float(f['/exchange/detector_pixelsize'][0])
        fdd  = float(f['/exchange/focusdetectordistance'][0])
        z1   = float(f['/exchange/z1'][0])
        p0   = f['/exchange/pref0'][:].astype('float32')
        p1s  = f['/exchange/pref1'][:].astype('float32')     # the 20-frame stack
    p1 = p1s.mean(0)
    with h5py.File(os.path.join(NFP, 'data_nfp.h5'), 'r') as f:
        z1c = float(f['/exchange/z1'][0])

    nz, n = p0.shape
    lam   = 1.24e-9 / E
    dist  = z1 * (fdd - z1) / fdd
    vox   = dpx * z1 / fdd
    dcode = (z1 - z1c) * z1 / z1c

    print('=' * 72)
    print('1. THE TWO SAMPLE-FREE FRAMES')
    print('=' * 72)
    print(f'pref0  mean {p0.mean():.5f}   (normalised to 1 by construction in stage 2)')
    print(f'pref1  mean {p1.mean():.5f}   = {100*p1.mean()/p0.mean()-100:+.1f}% '
          f'-- the beam decayed over the scan')
    print(f'geometry: z1 {z1*1e3:.4f} mm  z1c {z1c*1e3:.4f} mm  '
          f'distance {dist*1e3:.4f} mm  distance_code {dcode*1e3:.4f} mm')

    # noise floor: pref0/pref1 are each a mean of 20 whites, so the frame-to-frame
    # scatter divided by sqrt(20) is the noise left in them.
    lev = p1s.reshape(len(p1s), -1).mean(1)
    p1n = p1s / lev[:, None, None] * lev.mean()        # remove per-frame flux drift
    sigma_single = p1n.std(0, ddof=1).mean()
    sigma_mean   = sigma_single / np.sqrt(len(p1s))
    print(f'\nnoise: 20 white frames scatter by {sigma_single:.5f} each;')
    print(f'       their mean therefore carries {sigma_mean:.5f} '
          f'= {100*sigma_mean/p1.mean():.3f}% of the signal')

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for a, im, t, kw in ((ax[0], p0, 'pref0  (before the scan)', dict(cmap='gray')),
                         (ax[1], p1, 'pref1  (after, mean of 20)', dict(cmap='gray')),
                         (ax[2], p1 / (p0 + 1e-6), 'pref1 / pref0',
                          dict(cmap='seismic', vmin=0.6, vmax=1.4))):
        if 'vmin' not in kw:
            kw.update(dict(vmin=np.percentile(im, 1), vmax=np.percentile(im, 99)))
        h = a.imshow(im, **kw); a.set_title(t, fontsize=11); a.axis('off')
        fig.colorbar(h, ax=a, fraction=0.046)
    fig.suptitle('The two sample-free frames. Their ratio is full of structure, which already '
                 'says the aperture is not in the same place.', fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f'{OUT}/fig1_flats.png', dpi=110); plt.close(fig)

    # ---- 2. where is the aperture in each? ---------------------------------
    code = cp.asarray(np.load(os.path.join(NFP, 'code.npy')).astype('complex64'))
    prb  = cp.asarray(np.load(os.path.join(NFP, 'prb.npy')).astype('complex64'))
    nc   = code.shape[0]
    D    = Propagation(n, nz, 1, 1, lam, vox, [dist])
    Dc   = Propagation(n, nz, 1, 1, lam, vox, [dcode])

    ref = cp.abs(D.D(Dc.D(prb, 0), 0)) ** 2                  # probe only
    P8  = Propagation(nc, nc, 1, 1, lam, vox, [dist])
    P8c = Propagation(nc, nc, 1, 1, lam, vox, [dcode])
    cdata = cp.abs(P8.D(P8c.D(code, 0), 0)) ** 2             # code only, whole plane
    del P8, P8c
    cp.get_default_memory_pool().free_all_blocks()
    hp = lambda x, s=HPS: x - gaussian_filter(x, s)
    Fc = cp.fft.fft2(hp(cdata)); del cdata
    pad = (nc - n) // 2

    print('\n' + '=' * 72)
    print('2. WHERE IS THE APERTURE IN EACH FRAME?')
    print('=' * 72)
    print('phase correlation of (flat / probe-only model) against the propagated code.')
    print('High-passed first, or the DC term swamps the peak and the argmax picks noise.')

    def register(pref, label, axis):
        r  = cp.asarray(pref) / (ref + 1e-6)
        rp = cp.pad(hp(r), ((pad, pad), (pad, pad)), constant_values=0)
        cc = cp.fft.fftshift(cp.fft.ifft2(Fc * cp.conj(cp.fft.fft2(rp))).real)
        iy, ix = (int(v.get()) for v in cp.unravel_index(cp.argmax(cc), cc.shape))
        z  = float((cc.max() - cp.median(cc)) / cc.std())
        sy, sx = -(iy - nc // 2), -(ix - nc // 2)
        B = 400
        sub = cp.asnumpy(cc[iy - B:iy + B, ix - B:ix + B])
        h = axis.imshow(sub, cmap='magma', extent=(sx - B, sx + B, sy + B, sy - B))
        axis.plot(sx, sy, 'c+', ms=16, mew=2)
        axis.set_title(f'{label}: peak at ({sy:+d}, {sx:+d})\nz-score {z:.0f}', fontsize=11)
        axis.set_xlabel('col shift [px]'); axis.set_ylabel('row shift [px]')
        print(f'  {label}  aperture at ({sy:+5d}, {sx:+5d})   peak z-score {z:5.1f}')
        del r, rp, cc
        return sy, sx

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    s0 = register(p0, 'pref0', ax[0])
    s1 = register(p1, 'pref1', ax[1])
    fig.suptitle(f'The aperture moved ({s1[0]-s0[0]:+d}, {s1[1]-s0[1]:+d}) px between the two '
                 f'frames. That is what makes the split below possible.', fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f'{OUT}/fig2_registration.png', dpi=110); plt.close(fig)
    print(f'  -> the aperture moved ({s1[0]-s0[0]:+d}, {s1[1]-s0[1]:+d}) px between them')

    # ---- 3. measured vs model, and the log residual ------------------------
    def model(sy, sx):
        cr, cx = nc // 2 - sy, nc // 2 - sx
        w = code[cr - nz // 2:cr + nz // 2, cx - n // 2:cx + n // 2]
        return cp.abs(D.D(Dc.D(prb * w, 0), 0)) ** 2

    m0, m1 = model(*s0), model(*s1)
    c_ = (slice(CROP, nz - CROP), slice(CROP, n - CROP))
    R0 = cp.log(cp.asarray(p0) / (m0 + 1e-6))[c_]
    R1 = cp.log(cp.asarray(p1) / (m1 + 1e-6))[c_]
    R0 -= R0.mean(); R1 -= R1.mean()      # the overall scale is a free gauge, not an error

    print('\n' + '=' * 72)
    print('3. MEASURED vs MODEL')
    print('=' * 72)
    print('residual = log(measured / model), mean removed.  log so that a multiplicative')
    print('error becomes additive; the mean is dropped because the absolute scale of the')
    print('model is a free gauge (code.npy and prb.npy are only fixed up to one scalar).')
    print(f'  pref0 residual rms {float(R0.std()):.5f}')
    print(f'  pref1 residual rms {float(R1.std()):.5f}')
    print(f'  for scale, the noise floor from above is {sigma_mean/p1.mean():.5f}')

    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    for row, (pref, mod, R, lab) in enumerate(((p0, m0, R0, 'pref0'), (p1, m1, R1, 'pref1'))):
        mn = cp.asnumpy(mod)
        v  = float(np.percentile(np.abs(cp.asnumpy(R)), 99))
        for a, im, t, kw in ((ax[row][0], pref[c_], f'{lab}: measured', dict(cmap='gray')),
                             (ax[row][1], mn[c_], 'model', dict(cmap='gray')),
                             (ax[row][2], cp.asnumpy(R), 'log(measured / model)',
                              dict(cmap='seismic', vmin=-v, vmax=v))):
            if 'vmin' not in kw:
                kw.update(dict(vmin=np.percentile(pref[c_], 1), vmax=np.percentile(pref[c_], 99)))
            h = a.imshow(im, **kw); a.set_title(t, fontsize=11); a.axis('off')
            fig.colorbar(h, ax=a, fraction=0.046)
    fig.suptitle('If the model were right the third column would be flat. It is not, '
                 'in either frame.', fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f'{OUT}/fig3_residuals.png', dpi=110); plt.close(fig)

    # ---- 4. the split ------------------------------------------------------
    vbar = 0.5 * (float(R0.var()) + float(R1.var()))
    corr = float((R0 * R1).mean() / (R0.std() * R1.std()))

    print('\n' + '=' * 72)
    print('4. THE SPLIT')
    print('=' * 72)
    print('Write R0 = F + M0 and R1 = F + M1, with F the part fixed in the detector')
    print('frame and M0, M1 the aperture error at the two different positions.')
    print()
    print('The naive estimator (R0+R1)/2 is BIASED and must not be used here:')
    print('      (R0+R1)/2 = F + (M0+M1)/2,   var = var(F) + var(M)/2')
    print('With only two frames (M0+M1)/2 does not average away -- it keeps half the')
    print('moving variance and charges it to F. Likewise (R0-R1)/2 recovers only')
    print('var(M)/2, so that route reports the moving part at half its true size.')
    print()
    print('The unbiased estimator is the covariance, because M0 and M1 are independent:')
    print('      cov(R0, R1) = cov(F + M0, F + M1) = var(F)')
    print('so corr(R0, R1) IS the fixed fraction directly.')
    print()
    vf  = corr * vbar
    vm  = vbar - vf
    print(f'  corr(R0, R1)                 = {corr:+.4f}')
    print(f'  -> var(fixed)  = {vf:.3e}   {100*corr:4.1f}%  probe or detector')
    print(f'  -> var(moving) = {vm:.3e}   {100*(1-corr):4.1f}%  the aperture')
    print()
    nf = (sigma_mean / p1.mean()) ** 2 / vbar
    print(f'Detector noise is uncorrelated between the frames, so it sits entirely in')
    print(f'the moving part: {100*nf:.1f}% of the variance. The aperture share is')
    print(f'therefore about {100*(1-corr)-100*nf:.0f}%, not {100*(1-corr):.0f}%.')
    print()
    print('FOR CONTRAST, the biased numbers this script used to print:')
    print(f'  var((R0+R1)/2) / var(R) = {float((0.5*(R0+R1)).var())/vbar:4.1%}  '
          f'<- inflated by var(M)/2')
    print(f'  var((R0-R1)/2) / var(R) = {float((0.5*(R0-R1)).var())/vbar:4.1%}  '
          f'<- half the true moving share')

    fixed  = 0.5 * (R0 + R1)      # shown only to SEE the common structure, not to size it
    moving = 0.5 * (R0 - R1)
    fig, ax = plt.subplots(1, 4, figsize=(21, 5.4))
    v = float(np.percentile(np.abs(cp.asnumpy(R0)), 99))
    for a, im, t in ((ax[0], R0, 'R0 = log(pref0 / model0)'),
                     (ax[1], R1, 'R1 = log(pref1 / model1)'),
                     (ax[2], fixed, f'(R0+R1)/2 -- common structure\nprobe / detector is {100*corr:.0f}% by corr'),
                     (ax[3], moving, f'(R0-R1)/2 -- differing structure\naperture is {100*(1-corr):.0f}% by corr')):
        h = a.imshow(cp.asnumpy(im), cmap='seismic', vmin=-v, vmax=v)
        a.set_title(t, fontsize=11); a.axis('off')
        fig.colorbar(h, ax=a, fraction=0.046)
    fig.suptitle('The decomposition. Same colour scale throughout, so the two right-hand '
                 'panels are directly comparable.', fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f'{OUT}/fig4_split.png', dpi=110); plt.close(fig)

    # ---- 5. what scale is each part on? ------------------------------------
    def radial(a):
        A = cp.asnumpy(cp.abs(cp.fft.fftshift(cp.fft.fft2(a - a.mean()))) ** 2)
        s = A.shape[0]
        yy, xx = np.mgrid[:s, :s] - s // 2
        rr = np.hypot(yy, xx).astype(int)
        return np.bincount(rr.ravel(), A.ravel()) / np.maximum(np.bincount(rr.ravel()), 1)

    S = fixed.shape[0]
    pf, pm = radial(fixed)[:S // 2], radial(moving)[:S // 2]
    fq = np.arange(len(pf)) / S
    print('\n' + '=' * 72)
    print('5. WHAT SPATIAL SCALE IS EACH PART ON?')
    print('=' * 72)
    print(f'{"period [px]":>12} {"fixed":>12} {"moving":>12}')
    for per in (1024, 512, 256, 128, 64, 32, 16, 8, 4):
        i = int(round(S / per))
        if 0 < i < len(pf):
            print(f'{per:12d} {pf[i]:12.3e} {pm[i]:12.3e}')

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.loglog(fq[1:], pf[1:], label=f'(R0+R1)/2, probe/detector is {100*corr:.0f}%')
    ax.loglog(fq[1:], pm[1:], label=f'(R0-R1)/2, aperture is {100*(1-corr):.0f}%')
    ax.axhline((sigma_mean / p1.mean()) ** 2 * S * S / 2, color='k', ls=':', lw=1,
               label='approx noise floor')
    ax.set(xlabel='cycles / px', ylabel='power', title='Where each part lives in frequency')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')
    plt.tight_layout(); fig.savefig(f'{OUT}/fig5_spectra.png', dpi=110); plt.close(fig)

    print(f'\nfigures -> {OUT}/fig1..fig5*.png')


if __name__ == '__main__':
    main()
