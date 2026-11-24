"""Targeted diagnostics for the four anomalies left by the first pass:

  D1  why is the PTC intercept negative (is the curve genuinely curved?)
  D2  per-pair gain scatter, and sensitivity to the detrend tile size
  D3  long-range autocorrelation: is there a veiling-glare tail that would
      explain sub-linear variance at low signal?
  D4  structure of the dark noise (row/column common mode) and the
      read-noise-corrected photon N_eff
"""
import numpy as np
import h5py
from detector_noise import (load, bad_pixel_mask, make_pair, ptc, nps_tiles,
                            autocorr_from_nps, n_eff_from_autocorr, iqr_sigma,
                            detrend_tilewise, radial_profile)

np.set_printoptions(linewidth=150, suppress=True)
d = load("ht")
good = bad_pixel_mask(d["dark"], np.concatenate([d["white0"], d["white1"]], 0))
dmean = d["dark"].mean(0)
sub = d["white0"] - dmean
ref = sub.mean(0)

print("=" * 72)
print("D1  PTC shape: local slope vs level")
print("=" * 72)
D, M, _ = make_pair(sub, 0, 1, ref)
f = ptc(D, M, good, nbin=24)
m, v = f["mean"], f["var"]
sl = np.gradient(v, m)
for i in range(len(m)):
    print(f"   level {m[i]:8.1f}   var {v[i]:9.1f}   var/level {v[i]/m[i]:6.3f}"
          f"   local slope {sl[i]:6.3f}")
# quadratic fit
c = np.polyfit(m, v, 2)
print(f"   quadratic fit var = {c[0]:.3e} m^2 + {c[1]:.4f} m + {c[2]:.1f}")
print(f"   slope at HT level 5187: {2*c[0]*5187+c[1]:.4f}")
lin = np.polyfit(m, v, 1)
print(f"   linear fit    slope {lin[0]:.4f}  intercept {lin[1]:.1f}")

print()
print("  same, but restricted to pixels far from any code edge (|grad| low)")
gy, gx = np.gradient(ref)
G = np.hypot(gx, gy)
lowg = good & (G < np.percentile(G[good], 25))
flo = ptc(D, M, lowg, nbin=18, min_count=800)
for i in range(0, len(flo["mean"]), 3):
    print(f"   level {flo['mean'][i]:8.1f}  var {flo['var'][i]:9.1f}"
          f"  ratio {flo['var'][i]/flo['mean'][i]:6.3f}")
print(f"   low-grad linear fit slope {np.polyfit(flo['mean'],flo['var'],1)[0]:.4f}")

print()
print("=" * 72)
print("D2  per-pair gain, and detrend tile-size sensitivity")
print("=" * 72)
for k in range(8):
    Dk, Mk, dr = make_pair(sub, k, k + 1, ref)
    fk = ptc(Dk, Mk, good, nbin=24)
    lv = np.median(Mk[good])
    # local slope at the common level 5187 from a quadratic
    cc = np.polyfit(fk["mean"], fk["var"], 2)
    print(f"   pair {k}-{k+1}  level {lv:7.1f}  global slope {fk['gain_eff']:6.3f}"
          f"  slope@5187 {2*cc[0]*5187+cc[1]:6.3f}  drift {np.abs(dr).mean(axis=(0,1))}")

print()
for tile in (64, 128, 256, 512):
    raw = (sub[0] * np.median(sub[1]) / np.median(sub[0]) - sub[1]) / np.sqrt(2)
    Dt, _ = detrend_tilewise(raw, ref / np.sqrt(2), tile=tile)
    Mt = 0.5 * (sub[0] * np.median(sub[1]) / np.median(sub[0]) + sub[1])
    ft = ptc(Dt, Mt, good, nbin=24)
    cc = np.polyfit(ft["mean"], ft["var"], 2)
    print(f"   detrend tile {tile:4d}  global slope {ft['gain_eff']:6.3f}"
          f"  slope@5187 {2*cc[0]*5187+cc[1]:6.3f}")

print()
print("=" * 72)
print("D3  long-range autocorrelation (512 tiles, lags to 40)")
print("=" * 72)
nps2d, nt, _ = nps_tiles(D, good, tile=512, max_tiles=14)
rho = autocorr_from_nps(nps2d)
c0 = rho.shape[0] // 2
print("   rho along x, lags 0..20:")
print("  ", np.round(rho[c0, c0:c0 + 21], 5))
print("   azimuthally averaged rho vs radius:")
yy, xx = np.mgrid[-c0:c0, -c0:c0]
R = np.hypot(yy, xx)
for r in [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]:
    s = (R >= r - 0.5) & (R < r + 0.5)
    print(f"      r={r:3d}  rho={rho[s].mean():+.5f}  cumsum N_eff={n_eff_from_autocorr(rho, max(r,1)):.3f}")

print()
print("=" * 72)
print("D4  dark noise structure, and read-corrected photon N_eff")
print("=" * 72)
dk = d["dark"]
ddiff = (dk[0] - dk[1]) / np.sqrt(2)
print(f"   dark diff sigma (robust) {iqr_sigma(ddiff[good]):.3f} ADU")
r_ = ddiff.copy(); r_[~good] = np.nan
rowm = np.nanmean(r_, 1); colm = np.nanmean(r_, 0)
print(f"   row-mean std   {np.nanstd(rowm):.4f} ADU  (white-noise expectation "
      f"{iqr_sigma(ddiff[good])/np.sqrt(np.sum(good,1).mean()):.4f})")
print(f"   col-mean std   {np.nanstd(colm):.4f} ADU  (expectation "
      f"{iqr_sigma(ddiff[good])/np.sqrt(np.sum(good,0).mean()):.4f})")
ndk, _, _ = nps_tiles(ddiff, good, tile=256)
rdk = autocorr_from_nps(ndk)
print(f"   dark rho(1,0) {rdk[128,129]:+.5f}  rho(0,1) {rdk[129,128]:+.5f}"
      f"  N_eff r8 {n_eff_from_autocorr(rdk,8):.3f}")
fr, pdk = radial_profile(ndk)
print("   dark NPS radial profile (freq, power):")
for i in range(0, 64, 6):
    print(f"      f={fr[i]:.4f}  P={pdk[i]:9.2f}")

# read-corrected photon N_eff: subtract the dark NPS from the white NPS
nw, _, _ = nps_tiles(D, good, tile=256)
nphot = nw - ndk
rho_p = autocorr_from_nps(nphot)
print()
print(f"   white  N_eff r8 = {n_eff_from_autocorr(autocorr_from_nps(nw),8):.3f}")
print(f"   photon N_eff r8 (dark NPS removed) = {n_eff_from_autocorr(rho_p,8):.3f}")
print(f"   photon rho(1,0) {rho_p[128,129]:.4f}  rho(2,0) {rho_p[128,130]:.4f}")
print("   photon rho along x:", np.round(rho_p[128, 128:128 + 10], 4))
