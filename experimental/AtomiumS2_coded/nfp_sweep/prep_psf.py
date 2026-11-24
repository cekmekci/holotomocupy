#!/usr/bin/env python
"""Lay out an NFP reconstruction per trial detector PSF, to calibrate psf_sigma.

WHY.  Every config in this pipeline has `psf_sigma=0.0`, so neither step0 nor
step6 models the detector blur.  The transfer function measured on the HT side
(step5_paganin_ablation's tf-est cell) is not flat -- fitting
|H| = A*exp(-2*pi^2*sigma^2*f^2) over f >= 0.0203 cyc/binned-px gives A = 0.9757
and sigma = 1.391 UNBINNED detector px.  A monotonic roll-off with frequency is
a blur, and an unmodelled one makes the forward model predict a sharper code
than the detector can record.  Dividing a blurred measurement by a sharp model
is what leaves code-shaped residue.

1.391 is NOT the detector PSF.  step0 models no PSF either, so the solver has
already absorbed part of the blur into `code.npy` -- it had to, to fit a sharp
model to blurred data.  Once step0 models the PSF properly the required value
changes, and not in a direction that can be predicted analytically, because
blur and propagation do not commute for intensity and the code reaches the
detector through a different distance in the two scans.

SO MEASURE IT.  Putting the same sigma in both stages makes it self-calibrating:
if the PSF is modelled correctly everywhere, the residual |H(f)| on the HT side
comes out FLAT.  Still rolling off means sigma is too small; rising with
frequency means too large.  That is a fixed point, not a tuned knob, and it
transfers to the next dataset.

    for each trial sigma:
        run step0 with psf_sigma = sigma        -> this trial's code.npy, prb.npy
        point the HT model at it, same sigma    -> step6 wants sigma / 2**bin
        re-measure |H(f)|                       -> flat = done

NO DATA COPIES AND NO OVERLAYS, unlike prep_sweep.py: psf_sigma is a config
value rather than something stored in the HDF5, so each trial just reads the
real `data_nfp.h5` in place (`path`) and writes to its own folder (`path_out`).
The baseline nfp/ result is never touched and remains the psf_sigma = 0 point.

    python prep_psf.py                      # 0.8, 1.0, 1.2, 1.4
    python prep_psf.py --psf 1.6 1.8
    python prep_psf.py --dry-run

Then run them with the shared runner, pointed at this root:

    ROOT=/data3/cekmekci/AtomiumS2_coded/nfp_psf bash run_sweep.sh
"""
import argparse
import os
import shutil

HERE     = os.path.dirname(os.path.abspath(__file__))
BASE_CFG = os.path.join(HERE, os.pardir, 'config_step0.conf')
DATA_DIR = '/data3/cekmekci/AtomiumS2_coded/nfp'       # holds the real data_nfp.h5
ROOT     = '/data3/cekmekci/AtomiumS2_coded/nfp_psf'


def trial_dir(sigma):
    """1.2 -> '<ROOT>/psf_1p20'.  Sorts lexically in numeric order."""
    return os.path.join(ROOT, f'psf_{sigma:.2f}'.replace('.', 'p'))


def write_config(path, out_dir, sigma):
    """Copy the base config changing only psf_sigma, path, path_out and
    checkpoint_step, so every other knob is provably identical across trials."""
    with open(BASE_CFG) as f:
        lines = f.readlines()
    seen, out = set(), []
    for ln in lines:
        key = ln.split('=', 1)[0].strip()
        if key == 'path':
            out.append(f'path={DATA_DIR}\n')          # read the real h5 in place
        elif key == 'path_out':
            out.append(f'path_out={out_dir}\n')
        elif key == 'psf_sigma':
            out.append(f'psf_sigma={sigma:g}'.ljust(22)
                       + '# Gaussian sigma in RAW detector px -- the NFP runs unbinned, so no\n'
                       + ' ' * 22 + '# bin factor enters (rec_nfp_mpi.py:97-99). The HT side must use\n'
                       + ' ' * 22 + f'# THE SAME physical blur, i.e. {sigma:g}/2**bin in step6.\n')
        elif key == 'checkpoint_step':
            # LEFT ON. prep_sweep.py disables these to save disk, which was wrong to
            # carry over here: adding a blur turns step0 into an unconstrained
            # deconvolution and the aperture runs away (|code| reached 2.29 at
            # sigma=1.4, against the hard physical bound of 1 for a passive mask).
            # Seeing WHEN that happens is the whole point, and it is only 2.7 GB a run:
            #   projNNNN.tiff  8192^2 aperture phase, every 32 iters
            #   prbNNNN.tiff   probe phase,  prbNNNN.npy  the complex probe
            out.append(ln)
        else:
            out.append(ln)
            continue
        seen.add(key)
    missing = {'path', 'path_out', 'psf_sigma', 'checkpoint_step'} - seen
    if missing:
        raise SystemExit(f'{BASE_CFG} has no {sorted(missing)} line to rewrite')
    with open(path, 'w') as f:
        f.writelines(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--psf', type=float, nargs='+', default=[0.8, 1.0, 1.2, 1.4],
                    help='trial sigmas in RAW detector px')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--force', action='store_true',
                    help='re-make a trial folder that already holds a result')
    a = ap.parse_args()

    src = os.path.join(DATA_DIR, 'data_nfp.h5')
    if not os.path.exists(src):
        raise SystemExit(f'{src} not found')

    print(f'reading   {src}  (in place, never copied)')
    print(f'baseline  {DATA_DIR}/code.npy is the psf_sigma = 0 point, untouched\n')
    print(f'{"sigma [raw px]":>14} {"bin0":>7} {"bin1":>7} {"bin2":>7} {"bin3":>7}   '
          f'<- what step6 should use at each rung')
    for s in a.psf:
        print(f'{s:14.2f} ' + ' '.join(f'{s / 2 ** b:7.4f}' for b in range(4)))
    print()

    if a.dry_run:
        print('--dry-run: nothing written')
        return

    os.makedirs(ROOT, exist_ok=True)
    made = []
    for s in a.psf:
        d = trial_dir(s)
        if os.path.exists(os.path.join(d, 'code.npy')) and not a.force:
            print(f'skip {d} (already has code.npy; --force to redo)')
            continue
        if os.path.isdir(d) and a.force:
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
        write_config(os.path.join(d, 'config_step0.conf'), d, s)
        made.append(d)
        print(f'made {d}  (psf_sigma = {s:g} raw detector px)')

    print(f'\n{len(made)} trial(s) laid out under {ROOT}')
    if made:
        print(f'next:  ROOT={ROOT} bash run_sweep.sh')
        print('then:  for each trial, point the HT model at its code.npy/prb.npy with the')
        print('       SAME sigma (divided by 2**bin) and re-measure |H(f)|. Flat = done.')


if __name__ == '__main__':
    main()
