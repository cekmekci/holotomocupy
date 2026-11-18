"""Write a uniform transparent aperture for the test-2 initialisation ablation.

`CodedReader.read_code` has no cold-start path -- it always np.loads `code_file`
and requires the array to bin down to the config's `ncode` -- so "start the
aperture from scratch" has to be handed in as a real file.

All ones, not zeros.  code = 0 is a fully opaque mask: the forward model would
predict no signal anywhere and the gradient would be degenerate.  code = 1 is
the transparent, no-prior-structure start, which is what step0 means by
`vars['proj'][:] = 0` (it stores the phase and saves exp(1j*proj) = 1).

    python make_flat_code.py /eagle/APS_IRI/cekmekci/AtomiumS2_coded/nfp/flat_code.npy

Writes 8192 x 8192 complex64 = 537 MB, matching code.npy so it bins to
ncode=2048 at bin 2.  Shape and dtype are taken from the real code.npy when it
is reachable, so the two can never disagree.
"""
import os
import sys

import numpy as np

REAL_CODE = '/eagle/APS_IRI/cekmekci/AtomiumS2_coded/nfp/code.npy'
FALLBACK_SHAPE = (8192, 8192)


def main(out_path, reference=REAL_CODE):
    if os.path.exists(out_path):
        raise SystemExit(f'{out_path} already exists -- delete it first if you mean to rewrite')

    if os.path.exists(reference):
        ref = np.load(reference, mmap_mode='r')
        shape, dtype = ref.shape, ref.dtype
        print(f'shape/dtype taken from {reference}: {shape} {dtype}')
    else:
        shape, dtype = FALLBACK_SHAPE, np.dtype('complex64')
        print(f'{reference} not reachable; falling back to {shape} {dtype}')

    if len(shape) != 2 or shape[0] != shape[1]:
        raise SystemExit(f'expected a square 2-D aperture, got {shape}')

    flat = np.ones(shape, dtype=dtype)
    np.save(out_path, flat)
    print(f'wrote {out_path}  {flat.shape} {flat.dtype}  '
          f'({flat.nbytes / 1e6:.0f} MB, all {flat.flat[0]})')


if __name__ == '__main__':
    if len(sys.argv) not in (2, 3):
        raise SystemExit('usage: python make_flat_code.py <out.npy> [reference_code.npy]')
    main(*sys.argv[1:])
