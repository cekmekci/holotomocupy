#!/usr/bin/env python
"""
Pre-flight check -- run before step 6, fails the job on a missing or
mis-shaped input.

    python preflight.py config_step6_bin2.conf

The failures this exists for are the quiet ones. `obj_init_file` is optional
by design, so a path typo does not raise -- it starts the reconstruction from
zero and you find out hours later from a bad volume. `ncode` is a config
number, not read from `code.npy`, so a mismatch surfaces deep inside the
windowed shift rather than here. And a checkpoint handoff that cannot find
the rung above it silently restarts from the Paganin seed.

Everything checked is cheap: HDF5 headers and `.npy` headers, no frames and
no volumes read. Exit 0 = go, 1 = stop. Warnings do not fail, but they must
not scroll past unseen either.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h5py                                                    # noqa: E402
from holotomocupy.config import parse_args_coded               # noqa: E402

REQUIRED = ('/exchange/pdata', '/exchange/prb', '/exchange/theta',
            '/exchange/shifts_cor', '/exchange/shifts_code',
            '/exchange/energy', '/exchange/z1',
            '/exchange/detector_pixelsize',
            '/exchange/focusdetectordistance')


def npy_shape(path):
    """Shape and dtype from a .npy header, without reading the array."""
    with open(path, 'rb') as f:
        version = np.lib.format.read_magic(f)
        read = (np.lib.format.read_array_header_1_0 if version[0] == 1
                else np.lib.format.read_array_header_2_0)
        shape, _, dtype = read(f)
    return shape, dtype


def main():
    cfg = parse_args_coded(sys.argv[1])
    errors, warnings = [], []
    f2 = 2 ** cfg.bin

    print(f'config  {os.path.basename(sys.argv[1])}')
    print(f'  bin {cfg.bin}  n={cfg.n} nobj={cfg.nobj} ncode={cfg.ncode} '
          f'ntheta={cfg.ntheta} stride={cfg.stride}')
    print(f'  in     {cfg.in_file}')
    print(f'  code   {cfg.code_file}')
    print(f'  obj0   {cfg.obj_init_file}')
    print(f'  out    {cfg.path_out}')

    # --- the HDF5 -------------------------------------------------------
    if not os.path.exists(cfg.in_file):
        errors.append(f'in_file does not exist: {cfg.in_file}')
    else:
        with h5py.File(cfg.in_file, 'r') as f:
            for k in REQUIRED:
                if k not in f:
                    errors.append(f'{cfg.in_file} has no {k}')
            if '/exchange/pdata' in f:
                ntheta0, nz0, n0 = f['/exchange/pdata'].shape
                print(f'  pdata  [{ntheta0}, {nz0}, {n0}] {f["/exchange/pdata"].dtype}')
                if n0 != cfg.n * f2 or nz0 != cfg.nz * f2:
                    errors.append(
                        f'pdata is {nz0}x{n0} px but bin={cfg.bin} with '
                        f'n={cfg.n} expects {cfg.nz * f2}x{cfg.n * f2}')
                need = cfg.start_theta + cfg.stride * (cfg.ntheta - 1) + 1
                if need > ntheta0:
                    errors.append(
                        f'ntheta={cfg.ntheta} stride={cfg.stride} '
                        f'start_theta={cfg.start_theta} needs {need} angles, '
                        f'pdata has {ntheta0}')
            for k in ('/exchange/shifts_cor', '/exchange/shifts_code'):
                if k in f and f[k].shape[0] < cfg.ntheta * cfg.stride:
                    errors.append(f'{k} has {f[k].shape[0]} rows, too few for '
                                  f'ntheta={cfg.ntheta} stride={cfg.stride}')

    # --- the aperture ---------------------------------------------------
    if not os.path.exists(cfg.code_file):
        errors.append(f'code_file does not exist: {cfg.code_file}  '
                      f'(run step0.py first)')
    else:
        shape, dtype = npy_shape(cfg.code_file)
        print(f'  code   {shape} {dtype}')
        if len(shape) != 2 or shape[0] != shape[1]:
            errors.append(f'code.npy is {shape}, expected a square plane')
        elif shape[0] != cfg.ncode * f2:
            errors.append(f'code.npy is {shape[0]} px; bin={cfg.bin} with '
                          f'ncode={cfg.ncode} expects {cfg.ncode * f2}')
        if not np.issubdtype(dtype, np.complexfloating):
            errors.append(f'code.npy is {dtype}, expected complex')

    # --- the object seed, and the handoff -------------------------------
    # Only the start_iter=0 rung reads obj_init; the others inherit the object
    # through the checkpoint, so a missing seed matters at exactly one level.
    if cfg.start_iter == 0:
        if not cfg.obj_init_file:
            warnings.append('no obj_init_file: the object starts from ZERO. '
                            'Intended only for a cold start.')
        elif not os.path.exists(cfg.obj_init_file):
            errors.append(f'obj_init_file is set but missing: '
                          f'{cfg.obj_init_file}  (run step5, or clear the key '
                          f'to start from zero deliberately)')
        else:
            shape, dtype = npy_shape(cfg.obj_init_file)
            print(f'  obj0   {shape} {dtype}')
            if len(shape) != 3:
                errors.append(f'obj_init.npy is {shape}, expected 3-D')
            elif shape[1] != cfg.nobj * f2:
                want = cfg.nobj * f2
                if shape[1] > want:
                    warnings.append(
                        f'obj_init.npy is {shape[1]} px wide but this level '
                        f'wants {want}: read_obj CROPS it, losing the outer '
                        f'ring of the seed.')
                elif shape[1] < want:
                    # Expected: step5 writes the seed n wide, the solver pads
                    # the object to n + n/4 so the sample's travel stays
                    # inside it. read_obj CENTERS the seed in that larger
                    # canvas -- the rotation axis and shift operators assume
                    # the array's geometric centre is the object's centre --
                    # so the margin is split evenly and starts at vacuum.
                    print(f'  note   seed is {shape[1]} px centered into a '
                          f'{want} px object; the {(want - shape[1]) // 2} px '
                          f'margin per side starts at zero (vacuum), as intended')
    else:
        ckpt = os.path.join(cfg.path_out, 'checkpoints',
                            f'checkpoint_{cfg.start_iter:04}.h5')
        if not os.path.exists(ckpt):
            errors.append(
                f'start_iter={cfg.start_iter} but {ckpt} does not exist. '
                f'find_latest_checkpoint keys on the NAME, not the newest on '
                f'disk, so this rung would silently restart from the Paganin '
                f'seed. Run the level above first, or set start_iter to a '
                f'checkpoint that is there.')
        else:
            with h5py.File(ckpt, 'r') as f:
                src_n = f['prb_abs'].shape[-1]
            print(f'  resume {os.path.basename(ckpt)}  (prb {src_n} px '
                  f'-> {cfg.n}, x{cfg.n // src_n if src_n else 0})')
            if src_n and cfg.n % src_n:
                errors.append(f'checkpoint prb is {src_n} px and this level is '
                              f'{cfg.n}: not an integer upsample')

    # --- the constraint RecCoded asserts at construction -----------------
    if cfg.lam_laplacian > 0 and abs(cfg.rho[0] - 1.0) > 1e-12:
        errors.append(f'lam_laplacian={cfg.lam_laplacian:g} needs rho[obj]=1, '
                      f'got {cfg.rho[0]} -- RecCoded refuses this at startup')
    if len(cfg.rho) != 5:
        errors.append(f'rho has {len(cfg.rho)} entries, expected 5 '
                      f'[obj, prb, pos, code, pos_code]')

    # --- output dir ------------------------------------------------------
    parent = os.path.dirname(cfg.path_out.rstrip('/')) or '.'
    if not os.path.isdir(parent):
        errors.append(f'path_out parent does not exist: {parent}')

    for w in warnings:
        print(f'WARNING: {w}')
    for e in errors:
        print(f'ERROR:   {e}')
    if errors:
        print(f'\npreflight FAILED ({len(errors)} error'
              f'{"s" if len(errors) > 1 else ""})')
        return 1
    print(f'\npreflight OK' + (f' ({len(warnings)} warning'
                               f'{"s" if len(warnings) > 1 else ""})'
                               if warnings else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
