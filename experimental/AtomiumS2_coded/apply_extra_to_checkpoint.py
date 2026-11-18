#!/usr/bin/env python
"""Bake step 7's correction into a COPY of a checkpoint's `pos`, once.

    python apply_extra_to_checkpoint.py \\
        --src   /eagle/.../rec/pass1/checkpoints/checkpoint_0768.h5 \\
        --extra /eagle/.../correct_correct3D_extra.txt \\
        --bin 1 \\
        --dst   /eagle/.../rec/refined/checkpoints/checkpoint_0768.h5

Replaces the old design (pass 2 restarts the WHOLE ladder from obj_init.npy,
with pos = nominal + extra): that throws away pass 1's converged obj/prb/code
and rebuilds everything from the Paganin seed again, which is what produced
the bad pass-2 result. This script keeps pass 1's finished object -- the
*only* thing that changes is `pos`, by exactly the correction step 7 found.
The result is an ordinary checkpoint; config_step6_bin0.conf resumes it the
same way it would resume any other rung's handoff, no new machinery in
step6.py/reader_coded.py/RecCoded at all.

SCALE: `--bin` must be the SOURCE checkpoint's own bin (1, for the pass-1
bin-1 handoff), not the bin you intend to resume into. CodedReader.read_checkpoint
multiplies `pos` by `self.n // checkpoint_n` on every resume (that is how the
coarse-to-fine upsample works) -- so a bin-0 config resuming THIS checkpoint
will double the correction you apply here automatically. Scale for bin 1,
same as CodedReader.read_correct3d_extra would if a bin-1 reader called it,
and let the existing resume path do the rest.

RUN ONCE per checkpoint. Running it twice on its own OUTPUT would add the
correction a second time -- this script does not check for that, the same
way nothing stops you from pointing --src at an already-patched file by
mistake. Keep the one-shot nature in mind; there is no "already applied" flag.
"""
import argparse
import os
import shutil
import sys

import h5py
import numpy as np


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', required=True, help='pass-1 checkpoint to read from')
    p.add_argument('--extra', required=True, help='correct_correct3D_extra.txt from step7.py')
    p.add_argument('--bin', type=int, required=True,
                   help="the SOURCE checkpoint's own bin level (not the "
                        'resume target -- see the module docstring)')
    p.add_argument('--dst', required=True,
                   help='where the patched checkpoint is written (the new '
                        'path_out/checkpoints/checkpoint_NNNN.h5 the next '
                        'config will resume)')
    p.add_argument('--zbatch', type=int, default=64,
                   help='z-slab size when copying obj_re/obj_im, to avoid '
                        'loading the whole (multi-GB) volume at once')
    p.add_argument('--dry-run', action='store_true',
                   help='print the shapes and the found correction, write nothing')
    return p.parse_args()


def main():
    a = parse()

    if not os.path.exists(a.src):
        raise SystemExit(f'--src does not exist: {a.src}')
    if not os.path.exists(a.extra):
        raise SystemExit(f'--extra does not exist: {a.extra}')

    with h5py.File(a.src, 'r') as f:
        ntheta = f['pos'].shape[0]
        ndist  = f['pos'].shape[1]
        pos_old = np.asarray(f['pos'][:], dtype='float32')     # (ntheta, ndist, 2)
        it = int(f.attrs['iter'])
        obj_shape = f['obj_re'].shape

    print(f'source   {a.src}')
    print(f'  iteration {it}, pos {pos_old.shape}, obj {obj_shape}')

    raw = np.loadtxt(a.extra)
    if raw.shape[0] < ntheta:
        raise SystemExit(f'{a.extra}: {raw.shape[0]} rows, need at least {ntheta}')
    scale = np.float32(1.0 / 2 ** a.bin)
    # Same convention as CodedReader.read_correct3d_extra: 2 columns
    # (horizontal, vertical) on disk -> reversed to (vertical, horizontal) =
    # (row, col), matching pos's own axis order, then scaled to this bin.
    extra = (raw[:ntheta, ::-1] * scale).astype('float32')     # (ntheta, 2)
    print(f'extra    {a.extra}')
    print(f'  bin {a.bin} scale {float(scale):g}  y ptp {np.ptp(extra[:, 0]):.4f}  '
          f'x ptp {np.ptp(extra[:, 1]):.4f}  (this checkpoint\'s px units)')

    pos_new = pos_old + extra[:, None, :]
    moved = np.sqrt(((pos_new - pos_old) ** 2).sum(axis=-1))
    print(f'pos      unchanged in shape; per-angle shift applied: '
          f'mean {moved.mean():.4f}  max {moved.max():.4f} px')

    if a.dry_run:
        print('\n--dry-run: nothing written.')
        return

    os.makedirs(os.path.dirname(a.dst), exist_ok=True)
    if os.path.exists(a.dst):
        raise SystemExit(f'--dst already exists, refusing to overwrite: {a.dst}\n'
                         f'(delete it first if you mean to redo this)')

    with h5py.File(a.src, 'r') as fin, h5py.File(a.dst, 'w') as fout:
        for k, v in fin.attrs.items():
            fout.attrs[k] = v

        fout.create_dataset('pos', data=pos_new)

        for name in ('prb_abs', 'prb_phase', 'pos_code', 'code_re', 'code_im'):
            if name in fin:
                fout.create_dataset(name, data=fin[name][:])

        nz = obj_shape[0]
        for name in ('obj_re', 'obj_im'):
            ds_in = fin[name]
            ds_out = fout.create_dataset(name, shape=ds_in.shape, dtype=ds_in.dtype)
            for z0 in range(0, nz, a.zbatch):
                z1 = min(z0 + a.zbatch, nz)
                ds_out[z0:z1] = ds_in[z0:z1]
            print(f'  copied {name} ({ds_in.shape})')

    print(f'\nwrote    {a.dst}')
    print(f'next     point the resuming config at this path_out with '
          f'start_iter={it}, and (if it is still set) remove '
          f'correct3d_extra_file -- the correction is already baked into '
          f"pos, read_checkpoint never re-applies it, but leaving the key "
          f'there is dead and confusing, not dangerous.')


if __name__ == '__main__':
    sys.exit(main())
