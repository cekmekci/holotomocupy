#!/usr/bin/env python
"""Quick-look TIFFs of a stage-5 Paganin+FBP seed (obj_init.npy / obj_init_corrected.npy).

    python visualize_paganin.py /eagle/.../ht/obj_init.npy [out_dir]

Reads via mmap (never loads the full 134 GB array), and writes the same two
views Writer.write_checkpoint uses for step 6's checkpoints -- see
src/holotomocupy/writer.py's docstring -- so the object's outer boundary
(the thing today's pad-to-nobj-before-FBP fix changed) can be eyeballed the
same way a checkpoint would be, without running step 6 first:

    <stem>_midslice.tiff  (nobj, nobj)   -- z = nzobj//2
    <stem>_vert.tiff      (nzobj, nobj)  -- y = nobj//2, vertical cut through the axis

Real part only (obj_init is complex64; absorption/phase live in re/im the
same way a checkpoint's obj_re/obj_im do).
"""
import os
import sys

# Must be set BEFORE numpy/OpenBLAS ever gets imported. A bare login-node
# python sizes OpenBLAS's thread pool to the full visible core count (64) --
# shared with every other user's processes -- so pthread_create fails
# (RLIMIT_NPROC) and numpy misdiagnoses it as "importing numpy from its
# source directory." Same fix as polaris_env.sh's OPENBLAS_NUM_THREADS=1,
# baked in here since this script is meant to run bare on a login node, not
# under that env file.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
from holotomocupy.utils import write_tiff


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
    stem = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0])

    obj = np.load(path, mmap_mode='r')
    nzobj, nobj = obj.shape[0], obj.shape[1]
    print(f'{path}: shape {obj.shape} {obj.dtype}')

    mid = np.ascontiguousarray(obj[nzobj // 2].real).astype('float32')
    write_tiff(mid, f'{stem}_midslice')
    print(f'wrote {stem}_midslice.tiff  {mid.shape}  (z={nzobj // 2})')

    vert = np.ascontiguousarray(obj[:, nobj // 2, :].real).astype('float32')
    write_tiff(vert, f'{stem}_vert')
    print(f'wrote {stem}_vert.tiff  {vert.shape}  (y={nobj // 2})')


if __name__ == '__main__':
    main()
