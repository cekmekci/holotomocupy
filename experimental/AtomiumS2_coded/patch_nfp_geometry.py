"""Patch /exchange/z1 and /exchange/voxelsize in a COPY of data_nfp.h5.

step0a_convert_nfp.ipynb used to subtract the focus offset sx0 from the
coded-aperture position z1c, but z1c is already measured from the focus (unlike
the sample's sx_mm in steps15.py stage 1, which is a motor coordinate and does
need the subtraction).  The raw NFP scan at config_step0.conf's `raw_nfp` no
longer exists, so step0a cannot be re-run -- this patches the two derived
scalars in a copy of the surviving file instead.

Everything else in the file is unaffected.  pdata/pref/shifts are produced by
step0b, which reads the geometry only to print it, so they do not depend on
these values.

    python patch_nfp_geometry.py /data3/cekmekci/AtomiumS2_coded/nfp/data_nfp.h5
"""
import sys

import h5py
import numpy as np

SX0    = 1.286e-3       # [m] focus position
Z1C    = -17.75e-3      # [m] coded-aperture position, already relative to the focus
DET_PX = 3.03751e-6     # [m] detector pixel size
FDD    = 1.28           # [m] focus-to-detector distance

UNPATCHED_Z1 = Z1C - SX0


def main(path):
    if 'vnikitin' in path:
        raise SystemExit(f'refusing to write inside vnikitin\'s tree -- {path}')

    z1  = Z1C
    vox = abs(DET_PX / (FDD / z1))

    with h5py.File(path, 'a') as f:
        old_z1  = float(f['/exchange/z1'][0])
        old_vox = float(f['/exchange/voxelsize'][0])

        # if not np.isclose(old_z1, UNPATCHED_Z1, rtol=1e-4):
        #     raise SystemExit(
        #         f'/exchange/z1 is {old_z1:.6e}, expected the un-patched '
        #         f'{UNPATCHED_Z1:.6e} -- refusing to patch {path}')

        f['/exchange/z1'][0]        = np.float32(z1)
        f['/exchange/voxelsize'][0] = np.float32(vox)

    dist_old = old_z1 * (FDD - old_z1) / FDD
    dist_new = z1 * (FDD - z1) / FDD
    print(f'patched {path}')
    print(f'  z1         {old_z1 * 1e3:9.4f} mm  ->  {z1 * 1e3:9.4f} mm')
    print(f'  voxelsize  {old_vox * 1e9:9.3f} nm  ->  {vox * 1e9:9.3f} nm')
    print(f'  (distance  {dist_old * 1e3:9.4f} mm  ->  {dist_new * 1e3:9.4f} mm)')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__.strip().splitlines()[-1].strip())
    main(sys.argv[1])
