"""Write a sidecar HDF5 file that is `codedHT.h5` MINUS `/exchange/prb`, so
`CodedReader.read_prb` takes its documented "ones when absent" branch:

    def read_prb(self, out=None):
        \"\"\"The probe at the code plane, binned.  Ones when absent.\"\"\"
        ...
        if '/exchange/prb' in f:
            ...
        else:
            out[:] = 1
            logger.warning('no /exchange/prb in the file; probe set to 1')

There is no `prb_file` config key for step6 -- `read_prb` always reads from
`in_file` (the same file as `pdata`), with no override argument -- so making
the probe start at 1 means handing step6 a DIFFERENT `in_file`, not a
different key.

Rather than copying the 35 GB file, this writes a tiny shim containing
EXTERNAL LINKS to every dataset `CodedReader` actually touches --
`detector_pixelsize`, `focusdetectordistance`, `z1`, `energy`, `theta`,
`shifts_cor`, `shifts_code`, `pdata` -- and simply omits `prb`. External
links resolve lazily through the real file, so `pdata` is never duplicated;
only a few KB of pointers are written here.

    python make_prb1_codedht.py /eagle/APS_IRI/cekmekci/AtomiumS2_coded/ht/codedHT.h5

Writes `codedHT_noprb.h5` next to the real file. Point a test's `in_file` at
that name (same `path=`) to start the probe from 1.
"""
import os
import sys

import h5py

# Every /exchange/* key CodedReader.__init__ / read_pdata / read_prb touch,
# EXCEPT prb itself.
KEYS = (
    'detector_pixelsize',
    'focusdetectordistance',
    'z1',
    'energy',
    'theta',
    'shifts_cor',
    'shifts_code',
    'pdata',
)


def main(real_path):
    real_path = os.path.abspath(real_path)
    out_path = os.path.join(os.path.dirname(real_path), 'codedHT_noprb.h5')
    if os.path.exists(out_path):
        raise SystemExit(f'{out_path} already exists -- delete it first if you mean to rewrite')

    with h5py.File(real_path, 'r') as f:
        missing = [k for k in KEYS if f'/exchange/{k}' not in f]
        if missing:
            raise SystemExit(f'{real_path} is missing required keys: {missing}')
        if '/exchange/prb' not in f:
            print(f'NOTE: {real_path} has no /exchange/prb already -- '
                  f'it already behaves like this shim would.')

    with h5py.File(out_path, 'w') as f:
        for k in KEYS:
            f[f'/exchange/{k}'] = h5py.ExternalLink(real_path, f'/exchange/{k}')

    with h5py.File(out_path, 'r') as f:
        assert '/exchange/prb' not in f
        for k in KEYS:
            _ = f[f'/exchange/{k}'].shape if f[f'/exchange/{k}'].shape else None
        pdata_shape = f['/exchange/pdata'].shape
        z1 = float(f['/exchange/z1'][0])

    print(f'wrote {out_path}')
    print(f'  linked keys: {", ".join(KEYS)}')
    print(f'  /exchange/prb present: False  (read_prb will warn and set probe = 1)')
    print(f'  spot check: pdata {pdata_shape}, z1 {z1 * 1e3:.4f} mm  (matches {real_path})')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: python make_prb1_codedht.py <real_codedHT.h5>')
    main(sys.argv[1])
