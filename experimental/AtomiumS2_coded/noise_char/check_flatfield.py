"""Model-free check of deliverable (a).

Everything above goes through a fitted PTC.  This forms an ACTUAL flat-field
ratio from real frames and measures its rms directly, so the predicted noise
floor can be confirmed without relying on the fit.  Also checks how the floor
falls under binning, which matters because correlated noise does not average
down as fast as independent noise.
"""
import numpy as np
from detector_noise import (load, bad_pixel_mask, detrend_tilewise, iqr_sigma, EDGE)

d = load("ht")
good = bad_pixel_mask(d["dark"], np.concatenate([d["white0"], d["white1"]], 0))
dm = d["dark"].mean(0)
sub = d["white0"] - dm
ref = sub.mean(0)
lvl = float(np.median((d["data_lvl"] - dm)[good]))
print(f"HT operating level {lvl:.1f} ADU\n")

# clean frames only (0 and 2 were flagged unstable)
a, b = sub[5], sub[6]

print("1. direct ratio of two single white frames  w_a / w_b")
r = a / np.maximum(b, 1.0)
rr, _ = detrend_tilewise(r - 1.0, ref)        # remove structure/drift/scale
s = iqr_sigma(rr[good])
print(f"   measured rms {100*s:.3f}%   -> per-frame {100*s/np.sqrt(2):.3f}%")

print("\n2. single frame / mean of the other 19 whites (the pipeline case)")
rest = np.delete(np.arange(len(sub)), 5)
wbar = sub[rest].mean(0)
r2 = sub[5] / np.maximum(wbar, 1.0)
rr2, _ = detrend_tilewise(r2 - 1.0, ref)
s2 = iqr_sigma(rr2[good])
print(f"   measured rms {100*s2:.3f}%   predicted sqrt(1+1/19)*per-frame "
      f"{100*(s/np.sqrt(2))*np.sqrt(1+1/19):.3f}%")

print("\n3. scaling under binning (difference image, so pure noise)")
D = (a * np.median(b) / np.median(a) - b) / np.sqrt(2)
Dd, _ = detrend_tilewise(D, ref / np.sqrt(2))
M = 0.5 * (a + b)
core = (slice(EDGE, -EDGE), slice(EDGE, -EDGE))
Dc, Mc, gc = Dd[core], M[core], good[core]
for bsz in (1, 2, 3, 4, 6, 8):
    n = (Dc.shape[0] // bsz) * bsz
    Db = Dc[:n, :n].reshape(n // bsz, bsz, n // bsz, bsz).mean((1, 3))
    Mb = Mc[:n, :n].reshape(n // bsz, bsz, n // bsz, bsz).mean((1, 3))
    gb = gc[:n, :n].reshape(n // bsz, bsz, n // bsz, bsz).all((1, 3))
    rel = iqr_sigma(Db[gb]) / np.median(Mb[gb])
    ideal = (iqr_sigma(Dc[gc]) / np.median(Mc[gc])) / bsz
    print(f"   bin {bsz}: rel rms {100*rel:.4f}%   independent-noise ideal "
          f"{100*ideal:.4f}%   ratio {rel/ideal:.3f}")

print("\n4. implied quanta and the correlated-noise correction, at the HT level")
for g_eff, neff, tag in ((2.033, 2.554, "HT white0"), (1.959, 2.373, "NFP")):
    var = g_eff * lvl
    print(f"   {tag}: g_eff {g_eff:.3f}  N_eff {neff:.3f}  gain {g_eff*neff:.2f} ADU/q")
    print(f"      quanta/px {lvl/(g_eff*neff):.0f}   "
          f"naive per-pixel Poisson rms {100/np.sqrt(lvl/(g_eff*neff)):.3f}%   "
          f"actual {100*np.sqrt(var)/lvl:.3f}%")
