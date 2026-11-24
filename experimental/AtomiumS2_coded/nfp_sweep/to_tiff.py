#!/usr/bin/env python
"""Write each finished trial's code.npy and prb.npy as 32-bit TIFFs for ImageJ.

Both are COMPLEX, which ImageJ has no type for, so each one goes out twice --
phase and amplitude.  Phase is `angle()`, which for the aperture is the same
quantity step0's own checkpoint tiffs hold (it saves `proj.real`, and
code = exp(1j*proj)).

Per trial, full size:

    <trial>/code_phase.tiff   <trial>/code_amp.tiff     8192^2
    <trial>/prb_phase.tiff    <trial>/prb_amp.tiff      2048^2

and across trials, one stack per quantity so ImageJ can scroll z1c with the
arrow keys.  Slices run in the order printed at the end, most negative z1c
first.  The stacks are a centre crop (--crop, default 2048) because a full
8192^2 stack of 11 trials is 2.9 GB per quantity.

    <ROOT>/stack_code_phase.tiff   <ROOT>/stack_code_amp.tiff
    <ROOT>/stack_prb_phase.tiff    <ROOT>/stack_prb_amp.tiff

Safe to run while run_sweep.sh is still going -- it only reads trials that
already have a code.npy, and re-running picks up the ones that finished since.

    python to_tiff.py
    python to_tiff.py --crop 4096          # bigger stack slices
    python to_tiff.py --no-full            # stacks only, skip the per-trial files
"""
import argparse
import glob
import os
import re

import numpy as np
import tifffile

ROOT     = '/data3/cekmekci/AtomiumS2_coded/nfp_sweep'
BASE_NFP = '/data3/cekmekci/AtomiumS2_coded/nfp'      # the 2026-11-18 run, z1c = -17.75 mm


def centre(a, size):
    if size is None or size >= a.shape[0]:
        return a
    c = a.shape[0] // 2
    return a[c - size // 2: c + size // 2, c - size // 2: c + size // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crop', type=int, default=2048,
                    help='centre crop used for the stacks; 0 = full size')
    ap.add_argument('--no-full', action='store_true',
                    help='skip the per-trial full-size tiffs, write only the stacks')
    ap.add_argument('--baseline', action='store_true',
                    help='include the existing nfp/ result as an extra slice')
    a = ap.parse_args()
    crop = a.crop or None

    ts = []
    for d in sorted(glob.glob(os.path.join(ROOT, 'z1c_*um'))):
        m = re.search(r'z1c_(\d+)um$', d)
        if m and os.path.exists(os.path.join(d, 'code.npy')):
            ts.append((-int(m.group(1)) * 1e-6, d))
    if a.baseline and os.path.exists(os.path.join(BASE_NFP, 'code.npy')):
        ts.append((-17.75e-3, BASE_NFP))
    ts.sort(key=lambda t: t[0])
    if not ts:
        raise SystemExit(f'no finished trials under {ROOT} yet')

    stacks = {k: [] for k in ('code_phase', 'code_amp', 'prb_phase', 'prb_amp')}
    for z, d in ts:
        print(f'{z * 1e3:9.3f} mm  {os.path.basename(d)}')
        for what in ('code', 'prb'):
            arr = np.load(os.path.join(d, f'{what}.npy'))
            for part, fn in (('phase', np.angle), ('amp', np.abs)):
                v = fn(arr).astype('float32')
                stacks[f'{what}_{part}'].append(centre(v, crop))
                if not a.no_full:
                    out = os.path.join(d, f'{what}_{part}.tiff')
                    tifffile.imwrite(out, v)
                    print(f'   {os.path.basename(out):18s} {v.shape[0]}^2  '
                          f'[{v.min():+.4f}, {v.max():+.4f}]')
            del arr

    print()
    for name, sl in stacks.items():
        out = os.path.join(ROOT, f'stack_{name}.tiff')
        tifffile.imwrite(out, np.stack(sl), imagej=True)
        print(f'-> {out}  {len(sl)} slices of {sl[0].shape[0]}^2')
    print('\nslice order (ImageJ slice 1 first):')
    for i, (z, d) in enumerate(ts, 1):
        print(f'  {i:3d}  z1c {z * 1e3:9.3f} mm   {os.path.basename(d)}')


if __name__ == '__main__':
    main()
