#!/usr/bin/env python
"""Cross-check rot_axis_offset against the axis ESRF reconstructed this scan on.

Every multi-distance steps15.py calls esrf_meta.nabu_axis() and warns when the
configured rotation axis disagrees with ESRF's own by more than 0.5 raw px.
The coded steps15.py never did, so rot_axis_offset has never been checked
against anything -- it is the one number in the coded pipeline with no source.

    python check_rot_axis.py [config_steps15.conf] [--voxelsize M]

Reads only text metadata: the config, the processed .h5 HEADER (for voxelsize),
and ESRF's own *.conf / *.par in the raw drop. No frames, no GPU, no writes.
"""

import argparse
import os
import sys

import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from holotomocupy import esrf_meta                          # noqa: E402
from holotomocupy.config import parse_args_coded_steps15    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('config', nargs='?',
                   default=os.path.join(HERE, 'config_steps15.conf'))
    p.add_argument('--voxelsize', type=float, default=None,
                   help='metres; default reads /exchange/voxelsize from the h5')
    a = p.parse_args()

    cfg = parse_args_coded_steps15(a.config)

    # drop_dirs() globs <path>/<pfile>_/ , so `path` is the directory CONTAINING
    # the scan folder, not the scan folder itself.
    raw_root = os.path.dirname(cfg.raw_ht.rstrip('/'))
    pfile = cfg.pfile
    configured = float(cfg.rot_axis_offset[1])   # (row, col); col is horizontal

    print(f'config        {os.path.basename(a.config)}')
    print(f'  raw root    {raw_root}')
    print(f'  pfile       {pfile}')
    print(f'  rot_axis_offset (row, col) = {tuple(cfg.rot_axis_offset)}')
    print(f'  -> horizontal component being checked: {configured:+.5f} raw px')

    voxelsize = a.voxelsize
    if voxelsize is None:
        h5 = cfg.in_file if os.path.isabs(cfg.in_file) \
            else os.path.join(cfg.path, cfg.in_file)
        try:
            with h5py.File(h5, 'r') as f:
                voxelsize = float(f['/exchange/voxelsize'][0])
            print(f'  voxelsize   {voxelsize:.6e} m  (from {h5})')
        except Exception as e:
            print(f'\nERROR: could not read /exchange/voxelsize from {h5}: {e}')
            print('Pass it explicitly:  --voxelsize <metres>')
            return 2
    else:
        print(f'  voxelsize   {voxelsize:.6e} m  (--voxelsize)')

    print(f'\nsearch path:')
    for d in esrf_meta.drop_dirs(raw_root, pfile):
        print(f'  {"EXISTS " if os.path.isdir(d) else "missing"}  {d}')

    nabu = esrf_meta.nabu_axis(raw_root, pfile, voxelsize)
    pyhst = esrf_meta.pyhst_axis(raw_root, pfile, voxelsize)

    print()
    if nabu is not None and nabu.get('rcs') is not None:
        print(f'nabu   {os.path.basename(nabu["source"])} -> '
              f'{nabu["rcs"]:+.5f} raw px   {nabu["note"]}')
        for c in nabu['candidates']:
            print(f'    candidate {c["file"]:28s} axis_pos {c["axis_pos"]:.6f}'
                  f'  -> {c["rcs"]:+.5f}'
                  + (f'  shifts {c["shifts"]}' if c['shifts'] else ''))
    else:
        print('nabu   no naburec/*.conf with a rotation_axis_position')

    # Logged as a cross-check only. A .par written before the alignment pass
    # that produced naburec can be tens of px off; nabu wins when both exist.
    if pyhst is not None and pyhst.get('rcs') is not None:
        print(f'PyHST  {os.path.basename(pyhst["source"])} -> '
              f'{pyhst["rcs"]:+.5f} raw px   (cross-check only)')
    else:
        print('PyHST  no <pfile>_rec_.par with an axis')

    best = nabu['rcs'] if (nabu and nabu.get('rcs') is not None) else \
        (pyhst['rcs'] if (pyhst and pyhst.get('rcs') is not None) else None)
    if best is None:
        print('\nNo ESRF axis found -- nothing to compare against.')
        return 1

    src = 'nabu' if (nabu and nabu.get('rcs') is not None) else 'PyHST (weak)'
    d = abs(best - configured)
    print(f'\nconfigured {configured:+.5f}   {src} {best:+.5f}   '
          f'difference {d:.5f} raw px')
    if d > 0.5:
        print(f'MISMATCH (> 0.5 px). If {src} is right, retype the horizontal '
              f'component as {best:.5f} in config_steps15.conf AND every '
              f'config_step6_*.conf, then REBUILD the seed (stage 5) -- the '
              f'seed has the old axis baked in, so changing step 6 alone makes '
              f'them disagree, which is worse than both being wrong together.')
        return 1
    print('OK (within 0.5 px)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
