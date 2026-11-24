"""Validation of the estimators in detector_noise.py against synthetic data
with a KNOWN gain, read noise and PSF.

If the pipeline cannot recover planted values it cannot be trusted on the real
frames, so this runs first.  Three cases:
  A  uncorrelated Poisson + read noise, flat field      -> N_eff = 1
  B  Poisson spread over a known PSF                    -> N_eff = 1/sum(p^2)
  C  case B on top of the real coded-aperture structure, with a planted
     sub-pixel drift, to test the drift-removal step
"""
import numpy as np
import h5py
from detector_noise import (ptc, nps_tiles, autocorr_from_nps, n_eff_from_autocorr,
                            make_pair, correlation_length, iqr_sigma)

rng = np.random.default_rng(0)
N = 1024
good = np.ones((N, N), bool)
good[:32] = good[-32:] = good[:, :32] = good[:, -32:] = False


def psf_gauss(sigma, half=6):
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    p = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return p / p.sum()


def synth(lam_field, gain, read, p=None):
    """Poisson quanta -> optional PSF spreading -> ADU + gaussian read."""
    q = rng.poisson(lam_field).astype(np.float64)
    if p is not None:
        from scipy.signal import fftconvolve
        q = fftconvolve(q, p, mode="same")
    return gain * q + rng.normal(0, read, q.shape)


def measure(f1, f2, ref, label, gain_true, read_true, neff_true):
    D = (f1 - f2) / np.sqrt(2.0)
    M = 0.5 * (f1 + f2)
    fit = ptc(D, M, good)
    nps2d, nt, _ = nps_tiles(D, good, tile=256)
    rho = autocorr_from_nps(nps2d)
    ne = [n_eff_from_autocorr(rho, r) for r in (4, 8, 12)]
    g_eff_true = gain_true / neff_true
    print(f"{label}")
    print(f"   gain_eff  measured {fit['gain_eff']:7.4f}   expected {g_eff_true:7.4f}"
          f"   ratio {fit['gain_eff']/g_eff_true:6.4f}")
    print(f"   read      measured {np.sqrt(max(fit['intercept'],0)):7.3f}   expected {read_true:7.3f}"
          f"   (intercept {fit['intercept']:.1f})")
    print(f"   N_eff     measured r4={ne[0]:.3f} r8={ne[1]:.3f} r12={ne[2]:.3f}"
          f"   expected {neff_true:.3f}")
    print(f"   gain=g_eff*N_eff   {fit['gain_eff']*ne[1]:7.4f}   expected {gain_true:7.4f}")
    print(f"   rho(1,0) {rho[rho.shape[0]//2, rho.shape[1]//2+1]:.4f}  fwhm {correlation_length(rho)}")
    return fit, ne


print("=" * 72)
print("A  flat field, uncorrelated, gain=2.5 ADU/q, read=12 ADU, lam=2000")
print("=" * 72)
gain, read, lam = 2.5, 12.0, 2000.0
lamf = np.full((N, N), lam)
# give the field a range of levels so the PTC has leverage, as in the real data
yy = np.linspace(0.3, 2.0, N)[:, None] * np.ones((1, N))
lamf = lam * yy
f1 = synth(lamf, gain, read)
f2 = synth(lamf, gain, read)
measure(f1, f2, None, "A", gain, read, 1.0)

print()
print("=" * 72)
print("B  same, quanta spread over a gaussian PSF sigma=0.6 px")
print("=" * 72)
p = psf_gauss(0.6)
neff = 1.0 / np.sum(p ** 2)
f1 = synth(lamf, gain, read, p)
f2 = synth(lamf, gain, read, p)
measure(f1, f2, None, "B", gain, read, neff)

print()
print("=" * 72)
print("C  PSF noise on the REAL coded structure + planted drift 0.02 px")
print("=" * 72)
with h5py.File("/data3/cekmekci/AtomiumS2_coded/ht/codedHT.h5", "r") as f:
    w = f["exchange/data_white0"][0, 512:512 + N, 512:512 + N].astype(np.float64)
    dk = f["exchange/dark"][:].mean(0)[512:512 + N, 512:512 + N]
struct = np.maximum(w - dk, 10.0)
lam_struct = struct / gain          # quanta implied by the real structure
drift = 0.02

from scipy.ndimage import shift as ndshift
base1 = lam_struct
base2 = ndshift(lam_struct, (drift, drift), order=3, mode="nearest")
f1 = synth(base1, gain, read, p)
f2 = synth(base2, gain, read, p)
ref = 0.5 * (ndshift(struct, (0, 0), order=1) + ndshift(struct, (drift, drift), order=3))

print("  --- without drift removal")
D = (f1 - f2) / np.sqrt(2.0); M = 0.5 * (f1 + f2)
fit = ptc(D, M, good)
print(f"   gain_eff {fit['gain_eff']:.4f}  (expected {gain/neff:.4f})  "
      f"-> inflated by {fit['gain_eff']/(gain/neff):.2f}x")

print("  --- with tile-wise gradient projection")
stack = np.stack([f1, f2])
D2, M2, dr = make_pair(stack, 0, 1, ref)
fit2 = ptc(D2, M2, good)
nps2d, nt, _ = nps_tiles(D2, good, tile=256)
rho = autocorr_from_nps(nps2d)
print(f"   gain_eff {fit2['gain_eff']:.4f}  (expected {gain/neff:.4f})  "
      f"ratio {fit2['gain_eff']/(gain/neff):.4f}")
print(f"   N_eff r8 {n_eff_from_autocorr(rho,8):.3f}  (expected {neff:.3f})")
print(f"   recovered drift  {np.abs(dr).mean(axis=(0,1))}  (planted {drift})")
print(f"   gain = g_eff*N_eff = {fit2['gain_eff']*n_eff_from_autocorr(rho,8):.4f} "
      f"(expected {gain})")
