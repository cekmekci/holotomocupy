"""Final detector noise characterisation, HT and NFP.

Applies the corrections validated in validate.py and justified by diagnose*.py:

  * drift / flux removal by tile-wise projection on {dI/dx, dI/dy, I, 1}
    (validated: recovers a planted 0.02 px drift; real drift is ~0.005 px)
  * read noise taken from the DARK frames, not the PTC intercept.  At these
    signal levels read^2 is ~1% of the variance, so the intercept is set by
    sub-percent slope error -- validate.py case A returns 5.97 ADU for a
    planted 12.0 ADU.
  * unstable frames excluded.  Pairs involving the first few frames of a white
    stack give inflated variance whose excess is concentrated at LOW spatial
    frequency (1.3-1.75x below f=0.2 c/px, 1.05x at Nyquist), i.e. illumination
    instability, not photon noise, which would scale the NPS uniformly.
  * N_eff from the autocorrelation with the DARK NPS subtracted, removing the
    read-noise dilution (validated: recovers planted 4.07 and 10.17 to 2%).
  * gain and variance quoted as LOCAL values at the operating signal level,
    because the PTC is measurably curved (var/mean rises with level).

Outputs out/final_results.json and out/noise_report.png.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from detector_noise import (EDGE, load, bad_pixel_mask, make_pair, ptc, nps_tiles,
                            autocorr_from_nps, n_eff_from_autocorr, iqr_sigma,
                            radial_profile, correlation_length, distribution,
                            robust_sigma, OUT)

NPAIR_FRAMES = 12      # frames of each stack used for the all-pairs matrix


def curve_at(D, M, good, lvl, nbin=24):
    """Quadratic fit of var vs mean; return (var, dvar/dmean) at `lvl`."""
    f = ptc(D, M, good, nbin=nbin)
    c = np.polyfit(f["mean"], f["var"], 2)
    return float(np.polyval(c, lvl)), float(2 * c[0] * lvl + c[1]), f, c


def stack_gain(sub, ref, good, lvl, nframe=NPAIR_FRAMES):
    """All-pairs gain, automatic rejection of unstable frames."""
    n = min(nframe, len(sub))
    V = np.full((n, n), np.nan)
    S = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            D, M, _ = make_pair(sub, i, j, ref)
            v, s, _, _ = curve_at(D, M, good, lvl)
            V[i, j] = V[j, i] = v
            S[i, j] = S[j, i] = s
    per_frame = np.nanmedian(S, 1)
    med, mad = np.median(per_frame), 1.4826 * np.median(np.abs(per_frame - np.median(per_frame)))
    bad_frames = np.where(per_frame > med + 3 * mad)[0]
    keep = np.array([k for k in range(n) if k not in bad_frames])
    sel = np.ix_(keep, keep)
    vs = V[sel][np.triu_indices(len(keep), 1)]
    ss = S[sel][np.triu_indices(len(keep), 1)]
    return dict(
        n_frames_used=int(len(keep)),
        bad_frames=[int(b) for b in bad_frames],
        per_frame_slope=[float(x) for x in per_frame],
        slope=float(np.median(ss)), slope_spread=float(robust_sigma(ss)),
        var=float(np.median(vs)), var_spread=float(robust_sigma(vs)),
        slope_all_pairs=float(np.nanmedian(S[np.triu_indices(n, 1)])),
        keep=[int(k) for k in keep],
    )


def analyse(dataset, lvl_override=None):
    print(f"\n{'='*72}\n{dataset.upper()}\n{'='*72}", flush=True)
    d = load(dataset)
    whites = [("white0", d["white0"])] + ([("white1", d["white1"])] if "white1" in d else [])
    allw = np.concatenate([w for _, w in whites], 0)
    good = bad_pixel_mask(d["dark"], allw)
    dmean = d["dark"].mean(0)
    lvl = lvl_override or float(np.median((d["data_lvl"] - dmean)[good]))
    print(f"good px {100*good.mean():.3f}%   operating level {lvl:.1f} ADU", flush=True)

    out = dict(good_frac=float(good.mean()),
               n_bad=int((~good).sum() - (2048 ** 2 - 1984 ** 2)),
               operating_level_adu=lvl,
               dark_offset_adu=float(np.median(dmean[good])),
               dark_fpn_adu=float(robust_sigma(dmean[good])))

    # ---- read noise from darks (EMVA 1288 temporal, fixed pattern cancels)
    dk = d["dark"]
    sig = [iqr_sigma(((dk[k] - dk[k + 1]) / np.sqrt(2))[good]) for k in range(len(dk) - 1)]
    out["read_adu"] = float(np.mean(sig))
    out["read_adu_spread"] = float(np.std(sig))
    ddiff = (dk[0] - dk[1]) / np.sqrt(2)
    nps_dark, _, _ = nps_tiles(ddiff, good, tile=256)
    rho_d = autocorr_from_nps(nps_dark)
    out["dark_n_eff"] = n_eff_from_autocorr(rho_d, 8)
    r_ = ddiff[EDGE:-EDGE, EDGE:-EDGE]
    gi = good[EDGE:-EDGE, EDGE:-EDGE]
    r_ = np.where(gi, r_, np.nan)
    npx = np.sum(gi, 1).mean()
    out["dark_row_excess"] = float(np.nanstd(np.nanmean(r_, 1)) / (out["read_adu"] / np.sqrt(npx)))
    out["dark_col_excess"] = float(np.nanstd(np.nanmean(r_, 0)) / (out["read_adu"] / np.sqrt(npx)))
    print(f"  read {out['read_adu']:.2f} +- {out['read_adu_spread']:.2f} ADU, "
          f"offset {out['dark_offset_adu']:.1f}, FPN {out['dark_fpn_adu']:.1f}", flush=True)

    out["stacks"] = {}
    for name, w in whites:
        print(f"  {name} ...", flush=True)
        sub = w - dmean
        ref = sub.mean(0)
        g = stack_gain(sub, ref, good, lvl)
        print(f"    unstable frames {g['bad_frames']}  -> slope {g['slope']:.3f}"
              f" +- {g['slope_spread']:.3f}  (all-pairs median {g['slope_all_pairs']:.3f})",
              flush=True)

        # representative clean pair for NPS / distribution / tiles
        k = g["keep"]
        D, M, drift = make_pair(sub, k[0], k[len(k) // 2], ref)
        v_at, s_at, fit, qc = curve_at(D, M, good, lvl)

        nps_w, nt, _ = nps_tiles(D, good, tile=256)
        rho_tot = autocorr_from_nps(nps_w)
        rho_ph = autocorr_from_nps(nps_w - nps_dark)
        fr, prof = radial_profile(nps_w)

        # N_eff averaged over several clean pairs: the single-pair value is
        # the noisiest number in the whole analysis, because the lag sum
        # accumulates low-frequency residual.  rho(1,0) is reported alongside
        # as a local, much more stable measure of the same correlation.
        nes, r10s = [], []
        for a in range(0, len(k) - 1, 2):
            Dk, _, _ = make_pair(sub, k[a], k[a + 1], ref)
            nwk, _, _ = nps_tiles(Dk, good, tile=256)
            rk = autocorr_from_nps(nwk - nps_dark)
            nes.append(n_eff_from_autocorr(rk, 8))
            r10s.append(float(rk[128, 129]))
        neff = float(np.mean(nes))

        st = dict(
            slope=g["slope"], slope_spread=g["slope_spread"],
            var_at_level=g["var"], var_spread=g["var_spread"],
            per_frame_slope=g["per_frame_slope"], bad_frames=g["bad_frames"],
            n_frames_used=g["n_frames_used"],
            slope_all_pairs=g["slope_all_pairs"],
            drift_px=float(np.abs(drift).mean()),
            level_median=float(np.median(ref[good])),
            level_p5=float(np.percentile(ref[good], 5)),
            level_p95=float(np.percentile(ref[good], 95)),
            n_eff=neff,
            n_eff_spread=float(np.std(nes)), n_eff_npairs=len(nes),
            n_eff_pairs=[float(x) for x in nes],
            rho10=float(np.mean(r10s)), rho10_spread=float(np.std(r10s)),
            n_eff_total=n_eff_from_autocorr(rho_tot, 8),
            n_eff_r12=n_eff_from_autocorr(rho_ph, 12),
            rho_cut=[float(x) for x in rho_ph[128, 128:138]],
            fwhm=correlation_length(rho_ph),
            nps_freq=fr.tolist(), nps_prof=prof.tolist(),
            nps_lowf_over_nyq=float(np.nanmean(prof[:4]) / np.nanmean(prof[-4:])),
            ptc_mean=fit["mean"].tolist(), ptc_var=fit["var"].tolist(),
            ptc_var_plain=fit["var_plain"].tolist(),
            dist=distribution(D, M, good, qc, out["read_adu"]),
        )
        # true gain
        st["gain_adu_per_quantum"] = st["slope"] * neff
        # chip stationarity on the clean pair
        tiles = []
        for i in range(4):
            for j in range(4):
                sl = (slice(i * 512, (i + 1) * 512), slice(j * 512, (j + 1) * 512))
                try:
                    _, s_t, _, _ = curve_at(D[sl], M[sl], good[sl], lvl, nbin=12)
                    tiles.append(float(s_t))
                except Exception:
                    pass
        st["tile_slopes"] = tiles
        st["tile_slope_mean"] = float(np.mean(tiles))
        st["tile_slope_std"] = float(np.std(tiles))
        st["tile_slope_range"] = [float(np.min(tiles)), float(np.max(tiles))]
        out["stacks"][name] = st
        print(f"    var@{lvl:.0f} {st['var_at_level']:.0f}  N_eff {neff:.3f}  "
              f"gain {st['gain_adu_per_quantum']:.3f} ADU/q  "
              f"tiles {st['tile_slope_range'][0]:.2f}-{st['tile_slope_range'][1]:.2f}", flush=True)
    return out


def derived(res):
    """The two numbers requested, plus the supporting table."""
    ht = res["ht"]
    s = ht["stacks"]["white0"]
    lvl = ht["operating_level_adu"]
    V = s["var_at_level"]            # ADU^2, single frame, at the HT level
    rel1 = np.sqrt(V) / lvl
    nw = 20                          # whites averaged into the reference
    out = dict(level=lvl, var_single=V, rel_single=float(rel1))
    out["rel_ratio_Nwhite20"] = float(np.sqrt(V + V / nw) / lvl)
    out["rel_ratio_single_white"] = float(np.sqrt(2 * V) / lvl)
    # binning: correlated noise does not average down as 1/b^2 in variance
    rho = np.array(s["rho_cut"])
    binvals = {}
    for b in (1, 2, 3, 4):
        # variance of a b x b block mean = (1/b^4) sum_{lags} C(lag) * overlap
        w = 0.0
        for dy in range(-b + 1, b):
            for dx in range(-b + 1, b):
                r = np.hypot(dy, dx)
                k = int(round(r))
                rr = rho[k] if k < len(rho) else 0.0
                w += (b - abs(dy)) * (b - abs(dx)) * rr
        binvals[b] = float(np.sqrt(V * w / b ** 4) / lvl)
    out["rel_single_vs_bin"] = binvals
    out["rel_ratio_vs_bin"] = {b: float(v * np.sqrt(1 + 1.0 / nw)) for b, v in binvals.items()}
    out["read_frac_of_var"] = float(ht["read_adu"] ** 2 / V)
    out["photon_over_read"] = float(V / ht["read_adu"] ** 2)
    return out


def figure(res):
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for ds, col in (("ht", "C0"), ("nfp", "C1")):
        if ds not in res:
            continue
        for nm, ls in (("white0", "-"), ("white1", "--")):
            if nm not in res[ds]["stacks"]:
                continue
            s = res[ds]["stacks"][nm]
            lab = f"{ds}/{nm}"
            ax[0, 0].plot(s["ptc_mean"], s["ptc_var"], ls, color=col, label=lab)
            ax[0, 1].semilogy(s["nps_freq"], s["nps_prof"], ls, color=col, label=lab)
            ax[0, 2].plot(range(len(s["rho_cut"])), s["rho_cut"], ls + "o", color=col, label=lab)
            ax[1, 0].plot(s["per_frame_slope"], ls + "o", color=col, label=lab)
            h = np.array(s["dist"]["hist"], float)
            z = np.linspace(-8, 8, len(h))
            ax[1, 1].semilogy(z, h / h.sum() / (z[1] - z[0]) + 1e-12, ls, color=col, label=lab)
            ax[1, 2].plot(s["tile_slopes"], ls + "o", color=col, label=lab)
    z = np.linspace(-8, 8, 400)
    ax[1, 1].semilogy(z, np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi), "k:", label="gaussian")
    for a, t, xl, yl in [
        (ax[0, 0], "photon transfer curve", "mean level [ADU]", "variance [ADU$^2$]"),
        (ax[0, 1], "noise power spectrum", "frequency [cyc/px]", "NPS [ADU$^2$]"),
        (ax[0, 2], "noise autocorrelation (photon)", "lag [px]", r"$\rho$"),
        (ax[1, 0], "per-frame PTC slope", "frame index", "slope [ADU/quantum]"),
        (ax[1, 1], "normalised residual", r"$z=D/\sigma$", "pdf"),
        (ax[1, 2], "slope per 512px chip tile", "tile", "slope [ADU/quantum]"),
    ]:
        a.set_title(t); a.set_xlabel(xl); a.set_ylabel(yl); a.legend(fontsize=7); a.grid(alpha=.3)
    ax[1, 1].set_ylim(1e-7, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "noise_report.png"), dpi=110)
    print(f"wrote {OUT}/noise_report.png")


if __name__ == "__main__":
    res = {}
    for ds in ("ht", "nfp"):
        res[ds] = analyse(ds)
    res["derived"] = derived(res)
    with open(os.path.join(OUT, "final_results.json"), "w") as f:
        json.dump(res, f, indent=1)
    figure(res)
    print(json.dumps(res["derived"], indent=1))
