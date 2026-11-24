"""Second diagnostic pass.

  E1  gain for ALL frame pairs, not just consecutive: is the excess confined
      to particular frames (illumination instability) or spread evenly?
  E2  NPS of a high-gain pair vs a low-gain pair: is the excess white
      (photon-like, so really gain) or low-frequency (illumination)?
  E3  long-range autocorrelation out to 64 px (veiling-glare tail?)
  E4  dark structure, and photon N_eff with the dark NPS removed
"""
import numpy as np
from detector_noise import (load, bad_pixel_mask, make_pair, ptc, nps_tiles,
                            autocorr_from_nps, n_eff_from_autocorr, iqr_sigma,
                            radial_profile)

np.set_printoptions(linewidth=170, suppress=True)
d = load("ht")
good = bad_pixel_mask(d["dark"], np.concatenate([d["white0"], d["white1"]], 0))
dmean = d["dark"].mean(0)
sub = d["white0"] - dmean
ref = sub.mean(0)
LVL = 5187.0


def slope_at(D, M, lvl=LVL):
    f = ptc(D, M, good, nbin=24)
    c = np.polyfit(f["mean"], f["var"], 2)
    return 2 * c[0] * lvl + c[1]


print("=" * 72)
print("E1  slope@5187 for all pairs (i,j), i<j, of white0")
print("=" * 72)
n = 12
G = np.full((n, n), np.nan)
for i in range(n):
    for j in range(i + 1, n):
        D, M, _ = make_pair(sub, i, j, ref)
        G[i, j] = slope_at(D, M)
print("   rows i, cols j:")
print(np.round(G, 2))
off = np.array([G[i, i + 1] for i in range(n - 1)])
far = np.array([G[i, j] for i in range(n) for j in range(i + 4, n)])
print(f"\n   consecutive pairs (j=i+1): mean {np.nanmean(off):.3f} spread {np.nanstd(off):.3f}")
print(f"   well-separated (j>=i+4):   mean {np.nanmean(far):.3f} spread {np.nanstd(far):.3f}")
# per-frame excess: if frame k is noisy, every pair containing k is high
Gs = np.where(np.isnan(G), G.T, G)
per_frame = np.nanmean(Gs, 1)
print("   mean slope of all pairs containing frame k:")
for k in range(n):
    print(f"      frame {k:2d}  {per_frame[k]:.3f}")

print()
print("=" * 72)
print("E2  NPS of the highest- vs lowest-gain consecutive pair")
print("=" * 72)
ihi = int(np.nanargmax(off)); ilo = int(np.nanargmin(off))
print(f"   high pair {ihi}-{ihi+1} (slope {off[ihi]:.3f}), "
      f"low pair {ilo}-{ilo+1} (slope {off[ilo]:.3f})")
profs = {}
for tag, k in (("high", ihi), ("low", ilo)):
    D, M, _ = make_pair(sub, k, k + 1, ref)
    nps2d, nt, _ = nps_tiles(D, good, tile=256)
    fr, p = radial_profile(nps2d)
    profs[tag] = p
    rho = autocorr_from_nps(nps2d)
    print(f"   {tag}: ntiles {nt}  var {np.nanmean(p):.1f}  "
          f"N_eff r8 {n_eff_from_autocorr(rho,8):.3f}  rho10 {rho[128,129]:.4f}")
print("   freq    P_high    P_low    ratio")
for i in range(0, 64, 4):
    print(f"   {fr[i]:.4f}  {profs['high'][i]:9.1f} {profs['low'][i]:9.1f}"
          f"   {profs['high'][i]/profs['low'][i]:6.3f}")

print()
print("=" * 72)
print("E3  long-range autocorrelation, 512 px tiles")
print("=" * 72)
D, M, _ = make_pair(sub, ilo, ilo + 1, ref)
nps2d, nt, _ = nps_tiles(D, good, tile=512, max_tiles=14)
rho = autocorr_from_nps(nps2d)
c0 = rho.shape[0] // 2
print(f"   tiles used {nt}")
print("   rho along x, lags 0..12:", np.round(rho[c0, c0:c0 + 13], 5))
yy, xx = np.mgrid[-c0:c0, -c0:c0]
R = np.hypot(yy, xx)
print("   radius   rho        N_eff(sum to r)")
for r in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]:
    s = (R >= r - 0.5) & (R < r + 0.5)
    print(f"      {r:3d}   {rho[s].mean():+.6f}   {n_eff_from_autocorr(rho, r):.3f}")

print()
print("=" * 72)
print("E4  dark structure and photon N_eff")
print("=" * 72)
dk = d["dark"]
ddiff = (dk[0] - dk[1]) / np.sqrt(2)
sg = iqr_sigma(ddiff[good])
print(f"   dark diff sigma {sg:.3f} ADU")
r_ = np.where(good, ddiff, np.nan)
rowm = np.nanmean(r_, 1); colm = np.nanmean(r_, 0)
npx = np.sum(good, 1).mean()
print(f"   row-mean std {np.nanstd(rowm):.4f}  white-noise expectation {sg/np.sqrt(npx):.4f}"
      f"  -> excess {np.nanstd(rowm)/(sg/np.sqrt(npx)):.2f}x")
print(f"   col-mean std {np.nanstd(colm):.4f}  white-noise expectation {sg/np.sqrt(npx):.4f}"
      f"  -> excess {np.nanstd(colm)/(sg/np.sqrt(npx)):.2f}x")
ndk, ntd, _ = nps_tiles(ddiff, good, tile=256)
rdk = autocorr_from_nps(ndk)
print(f"   dark rho(1,0) {rdk[128,129]:+.5f}  N_eff r8 {n_eff_from_autocorr(rdk,8):.3f}")
frd, pd_ = radial_profile(ndk)
print("   dark NPS:", " ".join(f"{frd[i]:.3f}:{pd_[i]:.0f}" for i in range(0, 64, 8)))

nw, _, _ = nps_tiles(D, good, tile=256)
frw, pw = radial_profile(nw)
print("   white NPS:", " ".join(f"{frw[i]:.3f}:{pw[i]:.0f}" for i in range(0, 64, 8)))
nph = nw - ndk
rp = autocorr_from_nps(nph)
print(f"\n   white  N_eff r8 = {n_eff_from_autocorr(autocorr_from_nps(nw),8):.4f}")
print(f"   photon N_eff r8 = {n_eff_from_autocorr(rp,8):.4f}   (dark NPS removed)")
print(f"   photon N_eff r12= {n_eff_from_autocorr(rp,12):.4f}")
print("   photon rho along x:", np.round(rp[128, 128:128 + 8], 4))
