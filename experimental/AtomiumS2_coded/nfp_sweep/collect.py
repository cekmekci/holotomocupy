#!/usr/bin/env python
"""Look at what each trial z1c predicts, against what was measured.

Two comparisons per trial, both of them measurement vs forward model, shown as
a plain DIFFERENCE and a plain RATIO.

  NFP   in-sample.  pdata[k] against |D_nfp(prb * code_window(pos[k]))|^2, the
        frames step0 was fitted to, at the trial's own geometry and using its
        own refined pos_nfp.npy.  Several positions, spread by distance from
        the scan centre, because the aperture rim is seen by far fewer frames
        than the middle.

        The window comes from `ShiftFFTCode`, the SAME shifter RecNFPCoded
        used, not a hand-rolled crop plus Fourier shift.  That matters more
        than it sounds.  The solver freezes off = round(STARTING shifts) once
        in precalc and never re-rounds, so by the end of the solve the FFT is
        carrying a residual of 2.2 px on average and 10.8 px at worst through
        a 256 px padded, raised-cosine tapered window.  Re-splitting at the
        final position into an 8 px untapered window -- which is what this
        script and step0_sanity used to do -- is a different operator, and the
        difference lands in `measured - model` where it reads as physics.

  HT    held out.  pref0 from codedHT.h5 -- the holotomography reference frame,
        code in, no sample -- against |D(D_code(prb * code_window(sy, sx)))|^2
        at the HT geometry with this trial's z1c.  (sy, sx) is re-found per
        trial by the same phase correlation steps15 stage 3 uses, because a
        different code registers in a slightly different place.  step0 never
        saw this frame, so a wrong z1c cannot be absorbed into it.

NOTHING IS NORMALISED.  Measurement and model are plotted as they are.  The
only thing shared is the colour limits: they are taken from the first trial and
reused for all of them, which is what makes the trials comparable by eye, and
the values used are printed.  Ratios are measurement/model, centred on 1.

Two files per trial, because a 2048^2 frame cannot be shown honestly in a grid,
saved under this script's own directory (OUT_DIR) rather than next to the
trial data on /data3, so results are easy to browse locally:

    <OUT_DIR>/<trial>/cmp_full.png    whole frame, downsampled by the figure
    <OUT_DIR>/<trial>/cmp_zoom.png    a --view px centre crop at one array element per pixel

and two across trials:

    <OUT_DIR>/summary_ht.png          the HT difference and ratio for every z1c
    <OUT_DIR>/summary_nfp.png         the NFP difference and ratio for every z1c

    python collect.py
    python collect.py --view 768 --nfp-k 3
"""
import argparse
import glob
import os
import re

import numpy as np

ROOT     = '/data3/cekmekci/AtomiumS2_coded/nfp_sweep'
BASE_NFP = '/data3/cekmekci/AtomiumS2_coded/nfp'      # the 2026-11-18 run, z1c = -17.75 mm
PATH_HT  = '/data3/cekmekci/AtomiumS2_coded/ht/codedHT.h5'
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))  # plots saved here, not on /data3
CROP     = 128      # px ignored at every edge when summarising; the modelled rim is unreliable
POS_PAD_FRAC = 0.125   # config_step0.conf's; sets the window RecNFPCoded shifted in


def _trial_h5(d):
    """The data_nfp.h5 this trial was reconstructed from.

    The z1c sweep gives each trial its own overlay file with patched geometry.
    The psf sweep does not -- psf_sigma is a config value, so every trial reads
    the real file in place and only `path_out` differs.
    """
    own = os.path.join(d, 'data_nfp.h5')
    return own if os.path.exists(own) else os.path.join(BASE_NFP, 'data_nfp.h5')


def _trial_psf(d):
    """psf_sigma this trial was reconstructed with, from its own config.

    The model here must blur by the SAME amount the solve did, or the
    comparison is rigged against whichever trial modelled the detector.
    """
    cfg = os.path.join(d, 'config_step0.conf')
    if not os.path.exists(cfg):
        return 0.0
    for ln in open(cfg):
        if ln.split('=', 1)[0].strip() == 'psf_sigma':
            return float(ln.split('=', 1)[1].split('#')[0])
    return 0.0


def trials(root=None):
    """Any subfolder holding a finished result. Serves the z1c sweep
    (`z1c_*um`, geometry in each trial's own overlay h5) and the psf sweep
    (`psf_*`, all sharing the real h5 so only psf_sigma differs)."""
    root = root or ROOT
    out = []
    for d in sorted(glob.glob(os.path.join(root, '*'))):
        if not os.path.exists(os.path.join(d, 'code.npy')):
            continue
        m = re.search(r'z1c_(\d+)um$', d)
        if m:
            z1c = -int(m.group(1)) * 1e-6
        else:
            import h5py as _h
            h5 = _trial_h5(d)
            with _h.File(h5, 'r') as f:
                z1c = float(f['/exchange/z1'][0])
        out.append((z1c, d, _trial_psf(d)))
    return out


# --------------------------------------------------------------------------
def predict_nfp(d, z1c, psf_sigma, sel_n, cp, h5py, Propagation):
    """Measured and modelled NFP frames at `sel_n` positions spread by |pos|."""
    from holotomocupy.shift_fft_code import ShiftFFTCode, window_pad
    from holotomocupy.psf import psf_taps, psf_blur
    # Same taps the solve used. psf_blur acts on the detector INTENSITY,
    # which is why it is applied after the modulus and not folded into D.
    _, psf_w = psf_taps(psf_sigma)

    h5 = _trial_h5(d)
    with h5py.File(h5, 'r') as f:
        energy = float(f['/exchange/energy'][0])
        fdd    = float(f['/exchange/focusdetectordistance'][0])
        vox    = float(f['/exchange/voxelsize'][0])
        shifts = f['/exchange/shifts'][:].astype('float32')
        npos, nz, n = f['/exchange/pdata'].shape

    distance = z1c * (fdd - z1c) / fdd
    Dnfp = Propagation(n, nz, 1, 1, 1.24e-9 / energy, vox, [distance])

    # step0.py's own centring. These are also what the solver rounded to get
    # the frozen gather offsets, which it never re-derives afterwards.
    start = shifts.copy()
    start[:, 0] -= (start[:, 0].max() + start[:, 0].min()) / 2
    start[:, 1] -= (start[:, 1].max() + start[:, 1].min()) / 2
    off_all = np.round(start).astype('int32')

    pos = start
    p_ref = os.path.join(d, 'pos_nfp.npy')
    refined = os.path.exists(p_ref)
    if refined:
        r = np.load(p_ref).astype('float32')
        if r.shape == pos.shape:
            pos = r
        else:
            refined = False

    rad = np.hypot(pos[:, 0], pos[:, 1])
    sel = [int(np.argmin(np.abs(rad - t)))
           for t in np.linspace(rad.min(), rad.max(), sel_n)]

    code_gpu = cp.asarray(np.load(os.path.join(d, 'code.npy')).astype('complex64'))
    prb_gpu  = cp.asarray(np.load(os.path.join(d, 'prb.npy')).astype('complex64'))
    ncode = code_gpu.shape[0]

    # The solver's own windowed shifter, built exactly as RecNFPCoded builds it.
    pad_y, pad_x = window_pad(n, nz, POS_PAD_FRAC)
    sh = ShiftFFTCode(n, n + 2 * pad_x, nz, nz + 2 * pad_y, nchunk=1,
                      pad=(pad_y, pad_x), code_shape=(ncode, ncode))
    m1 = sh.unit_mag(1)

    out = []
    with h5py.File(h5, 'r') as f:
        for k in sel:
            r_k = cp.asarray(pos[k:k + 1], dtype='float32')
            o_k = cp.asarray(off_all[k:k + 1], dtype='int32')
            # Raises if the refined position drifted past what the pad can
            # absorb, i.e. if this window is not the one the solver used.
            sh.check_drift(r_k, o_k)
            w = sh.S(code_gpu, r_k, m1, o_k)[0]
            mod = cp.abs(Dnfp.D(prb_gpu * w, 0)) ** 2
            if psf_w is not None:
                mod = psf_blur(mod, psf_w)
            meas = cp.asarray(f['/exchange/pdata'][k].astype('float32'))
            resid = float(np.abs(pos[k] - off_all[k]).max())
            out.append((k, (float(pos[k, 0]), float(pos[k, 1])), resid,
                        cp.asnumpy(meas), cp.asnumpy(mod)))
            del meas, mod, w
    del code_gpu, prb_gpu, Dnfp, sh
    cp.get_default_memory_pool().free_all_blocks()
    return out, refined, (pad_y, pad_x)


def predict_ht(d, z1c, psf_sigma, cp, h5py, Propagation):
    """Measured and modelled HT reference frame pref0 at this trial's z1c."""
    with h5py.File(PATH_HT, 'r') as f:
        energy = float(f['/exchange/energy'][0])
        det_px = float(f['/exchange/detector_pixelsize'][0])
        fdd    = float(f['/exchange/focusdetectordistance'][0])
        z1     = float(f['/exchange/z1'][0])
        pref0  = f['/exchange/pref0'][:].astype('float32')
    nz, n = pref0.shape
    wavelength = 1.24e-9 / energy
    distance   = z1 * (fdd - z1) / fdd
    voxelsize  = det_px / (fdd / z1)
    distance_code = (z1 - z1c) * z1 / z1c         # steps15 stage 3's, reduced

    from holotomocupy.psf import psf_taps, psf_blur
    _, psf_w = psf_taps(psf_sigma)

    code_gpu = cp.asarray(np.load(os.path.join(d, 'code.npy')).astype('complex64'))
    prb_gpu  = cp.asarray(np.load(os.path.join(d, 'prb.npy')).astype('complex64'))
    ncode = code_gpu.shape[0]

    D     = Propagation(n, nz, 1, 1, wavelength, voxelsize, [distance])
    Dcode = Propagation(n, nz, 1, 1, wavelength, voxelsize, [distance_code])

    # --- where does this code sit on the detector? (stage 3's phase correlation)
    ref = cp.abs(D.D(Dcode.D(prb_gpu, 0), 0)) ** 2
    P8  = Propagation(ncode, ncode, 1, 1, wavelength, voxelsize, [distance])
    P8c = Propagation(ncode, ncode, 1, 1, wavelength, voxelsize, [distance_code])
    cdata = cp.abs(P8.D(P8c.D(code_gpu, 0), 0)) ** 2
    del P8, P8c
    cp.get_default_memory_pool().free_all_blocks()

    pad = (ncode - n) // 2
    rpad = cp.pad(cp.asarray(pref0) / (ref + 1e-6), ((pad, pad), (pad, pad)),
                  constant_values=1)
    ccm = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fft2(cdata) * cp.conj(cp.fft.fft2(rpad))).real)
    iy, ix = (int(v.get()) for v in cp.unravel_index(cp.argmax(ccm), ccm.shape))
    sy, sx = -(iy - ncode // 2), -(ix - ncode // 2)
    del rpad, ccm, cdata, ref
    cp.get_default_memory_pool().free_all_blocks()

    cr, cx = ncode // 2 - sy, ncode // 2 - sx
    # ccm comes from an FFT (circular) cross-correlation, so the peak can land
    # anywhere in [0, ncode) including near the edge -- wrap the window instead
    # of slicing, or it silently comes back empty when cr/cx is near the edge.
    rows = (cr - nz // 2 + cp.arange(nz)) % ncode
    cols = (cx - n // 2 + cp.arange(n)) % ncode
    w = code_gpu[rows[:, None], cols[None, :]]
    mod = cp.abs(D.D(Dcode.D(prb_gpu * w, 0), 0)) ** 2
    if psf_w is not None:
        mod = psf_blur(mod, psf_w)
    mod = cp.asnumpy(mod)
    del code_gpu, prb_gpu, w, D, Dcode
    cp.get_default_memory_pool().free_all_blocks()
    return pref0, mod, (sy, sx)


# --------------------------------------------------------------------------
def panel(fig, ax, img, title, **kw):
    im = ax.imshow(img, **kw)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)


def quad(fig, axrow, meas, mod, lim, tag):
    """measured | model | measured - model | measured / model, one row."""
    g, dv, r0, r1 = lim
    panel(fig, axrow[0], meas, f'{tag}: measured', cmap='gray', vmin=g[0], vmax=g[1])
    panel(fig, axrow[1], mod,  'model',            cmap='gray', vmin=g[0], vmax=g[1])
    panel(fig, axrow[2], meas - mod, 'measured - model', cmap='seismic', vmin=-dv, vmax=dv)
    panel(fig, axrow[3], meas / mod, 'measured / model', cmap='seismic', vmin=r0, vmax=r1)


def stats(meas, mod, c_):
    dif = (meas - mod)[c_]
    rat = (meas / mod)[c_]
    return (float(np.sqrt((dif ** 2).mean())), float(meas[c_].mean()), float(mod[c_].mean()),
            float(rat.mean()), float(rat.std()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--view', type=int, default=512, help='centre crop shown 1:1')
    ap.add_argument('--nfp-k', type=int, default=3, help='NFP positions per trial')
    ap.add_argument('--only', nargs='+', help='trial folder names, e.g. z1c_17750um')
    ap.add_argument('--root', default=ROOT,
                    help='sweep root. nfp_sweep (z1c, default) or nfp_psf (detector PSF)')
    ap.add_argument('--baseline', action='store_true',
                    help='also include the existing 2026-11-18 nfp/ result as a reference row. '
                         'On its own it checks this script against a known-good solve before '
                         'any sweep GPU time is spent.')
    a = ap.parse_args()

    ts = trials(a.root)
    if a.only:
        ts = [r for r in ts if os.path.basename(r[1]) in a.only]
    if a.baseline and os.path.exists(os.path.join(BASE_NFP, 'code.npy')):
        # the existing nfp/ run: psf_sigma = 0, i.e. the no-PSF reference point
        with __import__('h5py').File(os.path.join(BASE_NFP, 'data_nfp.h5'), 'r') as f:
            ts.insert(0, (float(f['/exchange/z1'][0]), BASE_NFP, 0.0))
    if not ts:
        raise SystemExit(f'no finished trials under {a.root} -- run run_sweep.sh first')

    import cupy as cp
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['image.interpolation'] = 'none'
    matplotlib.rcParams['image.resample'] = False
    from holotomocupy.propagation import Propagation

    print(f'{len(ts)} finished trial(s)\n')
    lim_nfp = lim_ht = None
    keep_ht, keep_nfp = [], []

    for z, d, psf in ts:
        name = os.path.basename(d)
        nfp, refined, pads = predict_nfp(d, z, psf, a.nfp_k, cp, h5py, Propagation)
        ht_meas, ht_mod, sxy = predict_ht(d, z, psf, cp, h5py, Propagation)

        nz, n = ht_meas.shape
        c_ = (slice(CROP, nz - CROP), slice(CROP, n - CROP))
        v = a.view
        cz = (slice(nz // 2 - v // 2, nz // 2 + v // 2), slice(n // 2 - v // 2, n // 2 + v // 2))

        # Colour limits from the FIRST trial, reused for all of them. Display only.
        if lim_ht is None:
            g = tuple(np.percentile(ht_meas[c_], [1, 99]))
            dv = float(np.percentile(np.abs((ht_meas - ht_mod)[c_]), 99))
            rr = float(np.percentile(np.abs((ht_meas / ht_mod)[c_] - 1), 99))
            lim_ht = (g, dv, 1 - rr, 1 + rr)
            m0, d0 = nfp[0][3], nfp[0][4]
            gn = tuple(np.percentile(m0[c_], [1, 99]))
            dvn = float(np.percentile(np.abs((m0 - d0)[c_]), 99))
            rrn = float(np.percentile(np.abs((m0 / d0)[c_] - 1), 99))
            lim_nfp = (gn, dvn, 1 - rrn, 1 + rrn)
            print(f'NFP window {nz + 2 * pads[0]} x {n + 2 * pads[1]} '
                  f'(pad {pads[0]}, {pads[1]}) via ShiftFFTCode, '
                  f'pos_pad_frac={POS_PAD_FRAC:g} -- the solver\'s own shifter')
            print(f'colour limits from {name}, reused for every trial:')
            print(f'  HT   grey [{g[0]:.4f}, {g[1]:.4f}]  diff +-{dv:.4f}  ratio 1+-{rr:.4f}')
            print(f'  NFP  grey [{gn[0]:.4f}, {gn[1]:.4f}]  diff +-{dvn:.4f}  ratio 1+-{rrn:.4f}\n')

        ks = stats(ht_meas, ht_mod, c_)
        print(f'{z * 1e3:9.3f} mm  psf {psf:.2f} px  {name}')
        print(f'   HT pref0  code at ({sxy[0]:+5d},{sxy[1]:+5d}) px   '
              f'rms(diff) {ks[0]:.5f}  mean meas {ks[1]:.4f} model {ks[2]:.4f}  '
              f'ratio {ks[3]:.5f} +- {ks[4]:.5f}')
        for k, p, resid, me, mo in nfp:
            s = stats(me, mo, c_)
            print(f'   NFP {k:4d}  pos ({p[0]:+8.1f},{p[1]:+8.1f}) res {resid:5.2f} px  '
                  f'rms(diff) {s[0]:.5f}  mean meas {s[1]:.4f} model {s[2]:.4f}  '
                  f'ratio {s[3]:.5f} +- {s[4]:.5f}')
        if not refined:
            print('   (no pos_nfp.npy -- NFP model is at step0b\'s starting positions)')

        for sfx, sl in (('full', (slice(None), slice(None))), ('zoom', cz)):
            nrow = 1 + len(nfp)
            fig, ax = plt.subplots(nrow, 4, figsize=(17, 4.3 * nrow), squeeze=False)
            quad(fig, ax[0], ht_meas[sl], ht_mod[sl], lim_ht, f'HT pref0  z1c {z*1e3:.3f} mm')
            for i, (k, p, resid, me, mo) in enumerate(nfp):
                quad(fig, ax[i + 1], me[sl], mo[sl], lim_nfp,
                     f'NFP frame {k}  pos ({p[0]:+.0f},{p[1]:+.0f})  resid {resid:.2f} px')
            fig.suptitle(f'{name}   z1c = {z*1e3:.3f} mm   psf_sigma = {psf:g} px   '
                         + ('whole frame' if sfx == 'full' else f'{v} px centre crop, 1:1'),
                         fontsize=11)
            plt.tight_layout(rect=(0, 0, 1, 0.98))
            trial_out = os.path.join(OUT_DIR, name)
            os.makedirs(trial_out, exist_ok=True)
            out = os.path.join(trial_out, f'cmp_{sfx}.png')
            fig.savefig(out, dpi=110); plt.close(fig)
            print(f'   -> {out}')

        keep_ht.append((z, psf, ht_meas[cz] - ht_mod[cz], ht_meas[cz] / ht_mod[cz]))
        keep_nfp.append((z, psf, nfp[0][3][cz] - nfp[0][4][cz], nfp[0][3][cz] / nfp[0][4][cz],
                         nfp[0][0]))
        print(flush=True)

    # --- the two cross-trial sheets: same colour limits down each column ---
    for tag, keep, lim in (('ht', keep_ht, lim_ht), ('nfp', keep_nfp, lim_nfp)):
        ncol = len(keep)
        fig, ax = plt.subplots(2, ncol, figsize=(4.2 * ncol, 9), squeeze=False)
        for j, row in enumerate(keep):
            z, psf, dif, rat = row[0], row[1], row[2], row[3]
            label = f'{z*1e3:.3f} mm, psf {psf:g} px'
            panel(fig, ax[0][j], dif, f'{label}: measured - model',
                  cmap='seismic', vmin=-lim[1], vmax=lim[1])
            panel(fig, ax[1][j], rat, 'measured / model',
                  cmap='seismic', vmin=lim[2], vmax=lim[3])
        what = ('HT pref0' if tag == 'ht' else f'NFP frame {keep[0][4]}')
        fig.suptitle(f'{what}, {a.view} px centre crop 1:1, one colour scale across z1c',
                     fontsize=12)
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        out = os.path.join(OUT_DIR, f'summary_{tag}.png')
        fig.savefig(out, dpi=110); plt.close(fig)
        print(f'-> {out}')


if __name__ == '__main__':
    main()
