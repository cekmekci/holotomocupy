#!/usr/bin/env python
"""Lay out an NFP reconstruction per trial z1c, so step0 can be run at each one.

z1c = -17.75 mm was never measured.  It is the value the advisor read off the
code sharpness in an NFP reconstruction, and both the NFP scan and the
holotomography stage have used it ever since.  The HT-side sweep in
step5_paganin_ablation can only measure the RATIO z1c(HT)/z1c(NFP); to pin the
absolute value, the NFP solve itself has to be run at several z1c and scored.

z1 enters step0 only through

    distance  = z1 * (fdd - z1) / fdd
    voxelsize = |det_px * z1 / fdd|

and those two reach the solver only as the Fresnel number

    kappa = lambda * distance / voxelsize**2 = lambda * fdd * (fdd - z1) / (det_px**2 * z1)

so kappa is the whole of the dependence.  Over -18.0 .. -17.5 mm it moves by
about 2.8%, which is far larger than the 5.6e-4 the HT scale sweep found --
i.e. the NFP solve is NOT insensitive to z1c, and this sweep is worth running.
`--dry-run` prints the kappa table without touching anything.

NO 12 GB COPIES.  Each trial gets a few-kB `data_nfp.h5` whose pdata/pref/
shifts are HDF5 EXTERNAL LINKS back into the real file, and whose z1 and
voxelsize are its own patched scalars.  step0.py reads them through the links
without noticing.  The real data_nfp.h5 is opened read-only and never written.

    python prep_sweep.py                      # default grid, -18.00 .. -17.50 mm
    python prep_sweep.py --dry-run
    python prep_sweep.py --z1c -17.75 -17.80  # explicit values, mm

Then `bash run_sweep.sh` (which this prints at the end), then `collect.py`.
"""
import argparse
import os
import shutil

import h5py
import numpy as np

HERE     = os.path.dirname(os.path.abspath(__file__))
BASE_CFG = os.path.join(HERE, os.pardir, 'config_step0.conf')
SRC_H5   = '/data3/cekmekci/AtomiumS2_coded/nfp/data_nfp.h5'
ROOT     = '/data3/cekmekci/AtomiumS2_coded/nfp_sweep'

# Linked straight out of SRC_H5. Everything else is written per trial so that
# the only difference between trials is the geometry.
LINKED = ['pdata', 'pref', 'shifts']
SCALARS = ['energy', 'detector_pixelsize', 'focusdetectordistance']


def trial_dir(z1c_m):
    """-17.75e-3 -> '<ROOT>/z1c_17750um'. z1c is always negative, so the sign
    is dropped rather than put in a path."""
    return os.path.join(ROOT, f'z1c_{round(-z1c_m * 1e9) // 1000:05d}um')


def write_overlay(path, z1c, det_px, fdd, src_rel):
    vox = abs(det_px * z1c / fdd)
    with h5py.File(path, 'w') as f:
        g = f.create_group('exchange')
        for k in LINKED:
            g[k] = h5py.ExternalLink(src_rel, f'/exchange/{k}')
        with h5py.File(SRC_H5, 'r') as s:
            for k in SCALARS:
                g.create_dataset(k, data=s[f'/exchange/{k}'][:])
        g.create_dataset('z1',        data=np.array([z1c], dtype='float32'))
        g.create_dataset('voxelsize', data=np.array([vox], dtype='float32'))
    return vox


def write_config(path, out_dir):
    """Copy the base config with only `path`, `path_out` and `checkpoint_step`
    changed, so every other knob is provably identical across trials."""
    with open(BASE_CFG) as f:
        lines = f.readlines()
    seen = set()
    out = []
    for ln in lines:
        key = ln.split('=', 1)[0].strip()
        if key in ('path', 'path_out'):
            out.append(f'{key}={out_dir}\n')
            seen.add(key)
        elif key == 'checkpoint_step':
            # per-trial tiff checkpoints are ~0.5 GB of nothing we compare
            out.append('checkpoint_step=-1\n')
            seen.add(key)
        else:
            out.append(ln)
    missing = {'path', 'path_out', 'checkpoint_step'} - seen
    if missing:
        raise SystemExit(f'{BASE_CFG} has no {sorted(missing)} line to rewrite')
    with open(path, 'w') as f:
        f.writelines(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--z1c', type=float, nargs='+', default=None,
                    help='trial values in mm (negative). Default -18.00..-17.50 by 0.05')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--force', action='store_true',
                    help='re-make a trial folder that already holds a result')
    a = ap.parse_args()

    z1cs = (np.array(a.z1c) if a.z1c is not None
            else np.round(np.arange(-18.00, -17.499, 0.05), 3)) * 1e-3

    with h5py.File(SRC_H5, 'r') as s:
        energy = float(s['/exchange/energy'][0])
        det_px = float(s['/exchange/detector_pixelsize'][0])
        fdd    = float(s['/exchange/focusdetectordistance'][0])
        z1_now = float(s['/exchange/z1'][0])
    lam = 1.24e-9 / energy

    kappa = lambda z: lam * fdd * (fdd - z) / (det_px ** 2 * z)
    k0 = kappa(z1_now)

    print(f'{SRC_H5}\n  energy {energy} keV  det_px {det_px * 1e6:.5f} um  '
          f'fdd {fdd} m  z1 now {z1_now * 1e3:.4f} mm\n')
    print(f'{"z1c [mm]":>10} {"distance [mm]":>14} {"voxel [nm]":>11} '
          f'{"kappa":>12} {"vs -17.75":>10}')
    for z in z1cs:
        d = z * (fdd - z) / fdd
        v = abs(det_px * z / fdd)
        print(f'{z * 1e3:10.3f} {d * 1e3:14.5f} {v * 1e9:11.4f} '
              f'{kappa(z):12.5f} {kappa(z) / k0 - 1:+10.2%}')
    print(f'\nkappa spread over the grid: '
          f'{kappa(z1cs.max()) / kappa(z1cs.min()) - 1:+.2%}')

    if a.dry_run:
        print('\n--dry-run: nothing written')
        return

    os.makedirs(ROOT, exist_ok=True)
    made = []
    for z in z1cs:
        d = trial_dir(z)
        if os.path.exists(os.path.join(d, 'code.npy')) and not a.force:
            print(f'skip {d} (already has code.npy; --force to redo)')
            continue
        if os.path.isdir(d) and a.force:
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
        # relative link so the tree survives being moved as a whole
        rel = os.path.relpath(SRC_H5, d)
        vox = write_overlay(os.path.join(d, 'data_nfp.h5'), z, det_px, fdd, rel)
        write_config(os.path.join(d, 'config_step0.conf'), d)
        made.append(d)
        print(f'made {d}  (z1 {z * 1e3:.3f} mm, voxel {vox * 1e9:.4f} nm)')

    print(f'\n{len(made)} trial(s) laid out under {ROOT}')
    if made:
        print('next:  bash run_sweep.sh          # one step0 per trial, sequential')


if __name__ == '__main__':
    main()
