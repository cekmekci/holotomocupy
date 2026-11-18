#!/usr/bin/env python
"""Sweep rot_axis_offset directly against reconstruction sharpness.

    python sweep_rot_axis.py config_steps15.conf [options]

`rot_axis_offset` (currently -24.0 horizontal, unverified -- check_rot_axis.py
found no ESRF alignment for this coded scan to compare it against) is only
ever CONSUMED in one place: steps15.py stage 5 does

    phase = multiPaganin(proj, ...)                               # no axis
    phase = fourier_shift_2d(phase, dc=-pos_obj[k,1], dr=-pos_obj[k,0])  # axis HERE
    local_paganin[j] = phase.get()                                # -> FBP

multiPaganin is a per-projection Fourier filter with no angle/position
dependence at all -- the axis only enters at the fourier_shift_2d step, right
before the stack goes into FBP. So Paganin retrieval is run ONCE per
projection (the expensive part) and cached; every candidate axis value only
re-does the cheap shift + FBP + sharpness score.

Score is the same minimum-entropy-of-FBP metric used everywhere else in this
pipeline (holotomocupy.autofocus), on a small, CENTRED z slab -- not the full
volume -- so this is a diagnostic sweep, not a reconstruction.

Writes a CSV (offset, entropy) and a PNG (the curve, plus the sharpest and
the currently-configured reconstruction side by side) to --out.
"""
import argparse
import os
import sys
import time

import cupy as cp
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from holotomocupy.autofocus import cylinder_mask, entropy
from holotomocupy.config import parse_args_coded_steps15
from holotomocupy.propagation import Propagation
from holotomocupy.tomo import Tomo


def multi_paganin(data, dist, wavelength, voxelsize, delta_beta, alpha):
    """Verbatim from steps15.py stage 5."""
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
    """Verbatim from steps15.py stage 5."""
    nr, nc = arr.shape
    fr = cp.fft.fftfreq(nr).astype('float32')
    fc = cp.fft.fftfreq(nc).astype('float32')
    Fc, Fr = cp.meshgrid(fc, fr)
    ramp = cp.exp(-2j * cp.pi * (Fr * dr + Fc * dc)).astype('complex64')
    return cp.fft.ifft2(cp.fft.fft2(arr.astype('complex64')) * ramp).real.astype('float32')


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('config', help='config_steps15.conf for this scan')
    p.add_argument('--ntheta', type=int, default=400,
                   help='angles, on an even stride (Paganin cost scales with this)')
    p.add_argument('--nz', type=int, default=64,
                   help='centred z rows kept AFTER Paganin, for the FBP sweep '
                        '(cheap to make large; the sinogram cache is '
                        'ntheta x nz x n, not ntheta x n_full x n)')
    p.add_argument('--lo', type=float, default=-35.0, help='sweep low end, horizontal, unbinned px')
    p.add_argument('--hi', type=float, default=-15.0, help='sweep high end, horizontal, unbinned px')
    p.add_argument('--step', type=float, default=0.5, help='sweep step, unbinned px')
    p.add_argument('--vertical', type=float, default=None,
                   help='vertical (row) component, fixed across the sweep '
                        '(default: the row entry already in rot_axis_offset)')
    p.add_argument('--margin', type=int, default=8,
                   help='extra z rows cached beyond --nz on each side, so the '
                        'per-candidate FFT shift sees real neighbouring rows '
                        'instead of wrapping around the crop edge. Must '
                        'exceed the largest |vertical shift| actually present '
                        '(shifts_cor row component + the vertical candidate); '
                        'that is normally well under a pixel here, so the '
                        'default is generous, not tight.')
    p.add_argument('--delta-beta', type=float, default=20.0)
    p.add_argument('--paganin-alpha', type=float, default=1e-3)
    p.add_argument('--out', default='.')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    cfg = parse_args_coded_steps15(a.config)

    row0 = float(cfg.rot_axis_offset[0]) if a.vertical is None else a.vertical
    candidates = np.round(np.arange(a.lo, a.hi + a.step / 2, a.step), 6)
    print(f'config   {a.config}')
    print(f'  current rot_axis_offset = {tuple(cfg.rot_axis_offset)}  '
          f'(horizontal {cfg.rot_axis_offset[1]:+.3f})')
    print(f'  sweeping horizontal over {candidates[0]:+.3f}..{candidates[-1]:+.3f} '
          f'step {a.step}  ({len(candidates)} points), vertical fixed at {row0:+.3f}')
    print(f'  {a.ntheta} angles, {a.nz} z rows (+/-{a.margin} margin, '
          f'FBP filter {cfg.fbp_filter!r})')

    with h5py.File(cfg.in_file, 'r') as f:
        energy = float(f['/exchange/energy'][0])
        det_px = float(f['/exchange/detector_pixelsize'][0])
        fdd    = float(f['/exchange/focusdetectordistance'][0])
        z1     = float(f['/exchange/z1'][0])
        nz_un, n_un = f['/exchange/pdata'].shape[1:]
        ntheta_scan = len(f['/exchange/theta'])
        shifts_cor  = f['/exchange/shifts_cor'][:].astype('float32')
        shifts_code = f['/exchange/shifts_code'][:].astype('float32')

    idx = np.round(np.linspace(0, ntheta_scan, a.ntheta, endpoint=False)).astype(int)

    z1_code = cfg.z1_code
    if z1_code is None:
        with h5py.File(cfg.nfp_file, 'r') as f:
            z1_code = float(f['/exchange/z1'][0])

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

    code = np.load(cfg.code_file).astype('complex64')
    prb  = np.load(cfg.prb_file).astype('complex64')
    ncode_full = code.shape[0]

    print(f'  distance={distance:.4g} m  distance_code={distance_code:.4g} m  '
          f'voxelsize={voxelsize:.4e} m')
    if a.dry_run:
        return

    cl_prop           = Propagation(n_un, nz_un, 1, 1, wavelength, voxelsize, [distance])
    cl_prop_code      = Propagation(n_un, nz_un, 1, 1, wavelength, voxelsize, [distance_code])
    cl_prop_8192      = Propagation(ncode_full, ncode_full, 1, 1, wavelength, voxelsize, [distance])
    cl_prop_code_8192 = Propagation(ncode_full, ncode_full, 1, 1, wavelength, voxelsize, [distance_code])

    prb_gpu = cp.array(prb)
    ref = cp.abs(cl_prop.D(cl_prop_code.D(prb_gpu, 0), 0)).real**2
    del prb_gpu

    code_gpu = cp.array(code)
    cdata = cp.abs(cl_prop_8192.D(cl_prop_code_8192.D(code_gpu, 0), 0)).real**2
    del code_gpu, cl_prop_8192, cl_prop_code_8192

    # --- stage 1: Paganin ONCE per projection, full width, no axis involved.
    # Cropped to `nz` rows centred on the full frame AFTER retrieval -- cropping
    # beforehand would distort multiPaganin's own FFT boundary.
    nz_pad = a.nz + 2 * a.margin
    z0 = (nz_un - nz_pad) // 2
    cache = cp.empty((len(idx), nz_pad, n_un), dtype='float32')
    t0 = time.time()
    with h5py.File(cfg.in_file, 'r') as f:
        ds = f['/exchange/pdata']
        for j, k in enumerate(idx):
            frame = cp.asarray(ds[int(k)], dtype='float32')
            dy = int(round(shifts_code[k, 0]))
            dx = int(round(shifts_code[k, 1]))
            cr = ncode_full // 2 - dy
            cc = ncode_full // 2 - dx
            cdata_crop = cdata[cr - nz_un // 2: cr + nz_un // 2,
                               cc - n_un // 2: cc + n_un // 2]
            proj = frame / (cdata_crop * ref + 1e-6)
            phase = multi_paganin(proj, distance, wavelength, voxelsize,
                                  a.delta_beta, a.paganin_alpha)
            cache[j] = phase[z0:z0 + nz_pad]
            del frame, proj, phase
    del cdata, ref
    cp.get_default_memory_pool().free_all_blocks()
    print(f'Paganin  {len(idx)} projections in {time.time()-t0:.1f} s (run once, cached)')

    theta_deg = None
    with h5py.File(cfg.in_file, 'r') as f:
        theta_deg = f['/exchange/theta'][:][idx]
    theta_rad = np.deg2rad(theta_deg).astype('float32')   # +theta, matching stage 5

    cl = Tomo(n_un, a.nz, theta_rad, mask_r=0)
    sel = cylinder_mask(n_un, a.nz)
    shifts_cor_sel = shifts_cor[idx]

    def reconstruct(col_offset):
        # Shift the PADDED rows, then crop to the centre a.nz after shifting --
        # not before. fourier_shift_2d is a circular FFT shift, so a nonzero
        # vertical (row) component applied to an already-cropped array would
        # wrap content from the far edge of the crop back in, rather than
        # pulling in the real neighbouring rows that exist in the full frame.
        # The margin on each side only has to exceed the actual |row shift|,
        # which is normally well under a pixel here (see --margin's help).
        pos_col = shifts_cor_sel[:, 1] + col_offset
        pos_row = shifts_cor_sel[:, 0] + row0
        sino = cp.empty((len(idx), a.nz, n_un), dtype='float32')
        for j in range(len(idx)):
            shifted = fourier_shift_2d(cache[j], dc=-float(pos_col[j]),
                                       dr=-float(pos_row[j]))
            sino[j] = shifted[a.margin:a.margin + a.nz]
            del shifted
        u = cl.fbp(sino, cfg.fbp_filter)
        del sino
        return u

    # Fixed ONCE, from the currently-configured offset, and reused for every
    # candidate below -- two entropies are only comparable if the histogram
    # bins are identical (see autofocus.grey_range's own docstring). Scoring
    # each candidate against its own auto-range would let a candidate that
    # happens to clip more/less structure win for a reason that has nothing
    # to do with sharpness.
    cur_c = float(cfg.rot_axis_offset[1])
    cur_u = reconstruct(cur_c)
    lo, hi = float(cp.percentile(cur_u[sel], 0.5)), float(cp.percentile(cur_u[sel], 99.5))
    cur_q = entropy(cur_u, sel, lo, hi)

    print(f'Sweeping {len(candidates)} offsets ...')
    results = []
    mids = {}
    t0 = time.time()
    for c in candidates:
        u = reconstruct(c)
        q = entropy(u, sel, lo, hi)
        results.append((c, q))
        mids[round(float(c), 6)] = cp.asnumpy(u[a.nz // 2])
        del u
    print(f'  done in {time.time()-t0:.1f} s')

    results = np.array(results)
    best_i = np.argmin(results[:, 1])
    best_c, best_q = results[best_i]

    tag = os.path.join(a.out, f'rot_axis_sweep_n{a.ntheta}_nz{a.nz}')
    np.savetxt(f'{tag}.csv', results, delimiter=',',
              header='horizontal_offset_unbinned_px,entropy', comments='')
    print(f'wrote    {tag}.csv')
    print()
    print(f'sharpest at horizontal = {best_c:+.3f} unbinned px  (entropy {best_q:.5f})')
    print(f'currently configured   = {cur_c:+.3f} unbinned px  (entropy {cur_q:.5f})')
    print(f'difference             = {abs(best_c - cur_c):.3f} unbinned px')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
        ax[0].plot(results[:, 0], results[:, 1], '.-')
        ax[0].axvline(cur_c, color='tab:orange', ls='--', label=f'configured {cur_c:+.2f}')
        ax[0].axvline(best_c, color='tab:red', ls='--', label=f'sharpest {best_c:+.2f}')
        ax[0].set(xlabel='horizontal rot_axis_offset (unbinned px)',
                  ylabel='entropy (lower = sharper)', title='sharpness vs axis offset')
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)
        vmin, vmax = np.percentile(mids[round(float(best_c), 6)], [0.5, 99.5])
        ax[1].imshow(cp.asnumpy(cur_u[a.nz // 2]), cmap='gray', vmin=vmin, vmax=vmax)
        ax[1].set_title(f'configured ({cur_c:+.2f})', fontsize=10)
        ax[1].axis('off')
        ax[2].imshow(mids[round(float(best_c), 6)], cmap='gray', vmin=vmin, vmax=vmax)
        ax[2].set_title(f'sharpest ({best_c:+.2f})', fontsize=10)
        ax[2].axis('off')
        fig.tight_layout()
        fig.savefig(f'{tag}.png', dpi=130)
        print(f'wrote    {tag}.png')
    except Exception as e:
        print(f'(plot skipped: {e})')


if __name__ == '__main__':
    sys.exit(main())
