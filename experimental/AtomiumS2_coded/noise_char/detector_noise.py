"""Detector noise characterisation for the AtomiumS2 coded-aperture datasets.

Measures, from RAW arrays only (never pdata/pref):
  1. photon transfer curve   -> gain (ADU per noise-equivalent quantum), read noise
  2. NPS / autocorrelation   -> correlation length, N_eff (pixels per quantum)
  3. distribution shape      -> gaussianity, tails
  4. stationarity            -> across chip, and white0 vs white1
  5. HT vs NFP agreement

Method follows EMVA 1288 (PTC, dark characterisation, tile-wise repeat for
non-uniformity) adapted to a structured (non-uniform) illumination field, and
Cunningham (SPIE Handbook of Medical Imaging Vol 1, 2000) for the NPS treatment
of correlated noise.

Key deviation from textbook EMVA 1288: the standard assumes a spatially uniform
flat field, so that a frame difference contains only noise.  Here the "white"
frames contain the coded aperture and probe speckle, with |grad I| ~ 900 ADU/px.
A sub-pixel drift d between two frames therefore injects a term d.grad(I) into
the difference, which would be mis-read as noise.  We remove it by least-squares
projection of each difference tile onto the basis {dI/dx, dI/dy, I, 1} built
from the *stack mean* (noise suppressed by sqrt(N)).  This costs 4 degrees of
freedom per tile and, unlike Fourier resampling, does not filter the noise
itself -- resampling would alter both the PTC amplitude and the measured NPS.

Everything is CPU numpy/scipy.  No GPU, no MPI.

Usage:
    python detector_noise.py --dataset ht
    python detector_noise.py --dataset nfp
    python detector_noise.py --dataset both
"""

import argparse
import json
import os

import h5py
import numpy as np
from scipy import ndimage

HT_PATH = "/data3/cekmekci/AtomiumS2_coded/ht/codedHT.h5"
NFP_PATH = "/data3/cekmekci/AtomiumS2_coded/nfp/data_nfp.h5"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

TILE = 256          # tile size for drift removal and NPS
NPS_TILE = 256
EDGE = 32           # pixels trimmed from the chip border everywhere


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load(dataset):
    """Return dict with raw uint16/float32 stacks, as float64 ADU."""
    if dataset == "ht":
        with h5py.File(HT_PATH, "r") as f:
            d = dict(
                dark=f["exchange/dark"][:].astype(np.float64),
                white0=f["exchange/data_white0"][:].astype(np.float64),
                white1=f["exchange/data_white1"][:].astype(np.float64),
            )
            # a handful of real projections, for the signal level only
            d["data_lvl"] = f["exchange/data"][0].astype(np.float64)
            d["bpm4"] = f["exchange/attrs"][:, 1].astype(np.float64)
    else:
        with h5py.File(NFP_PATH, "r") as f:
            d = dict(
                dark=f["exchange/data_dark"][:].astype(np.float64),
                white0=f["exchange/data_white"][:].astype(np.float64),
            )
            d["data_lvl"] = f["exchange/data"][0].astype(np.float64)
    return d


# --------------------------------------------------------------------------
# bad pixel mask
# --------------------------------------------------------------------------

def robust_sigma(x):
    """MAD-based sigma, resistant to tails."""
    m = np.median(x)
    return 1.4826 * np.median(np.abs(x - m))


def iqr_sigma(x):
    """Interquartile sigma.  Insensitive to <~10% outliers; this is the
    estimator used for every PTC variance, so cosmic rays and hot pixels do
    not inflate the gain."""
    q1, q3 = np.percentile(x, [25.0, 75.0])
    return (q3 - q1) / 1.3489795


def bad_pixel_mask(dark, whites, nsig=10.0):
    """True = good pixel.

    Only genuinely defective pixels are flagged.  In particular the ordinary
    pixel-to-pixel dark offset spread (robust sigma ~55 ADU here) is NOT a
    defect -- it is fixed pattern and cancels exactly in a frame difference --
    so the offset cut is deliberately loose.  Likewise pixels merely shadowed
    by the gold code are valid, just dim, so the response cut is set to catch
    only truly dead pixels.
    """
    dmean = dark.mean(0)
    dstd = dark.std(0)
    resp = whites.mean(0) - dmean

    bad = (dark >= 65535).any(0) | (whites >= 65535).any(0)   # saturated
    bad |= dstd > 3.0 * np.median(dstd)                       # noisy / RTS
    bad |= np.abs(dmean - np.median(dmean)) > nsig * robust_sigma(dmean)
    bad |= resp < 0.05 * np.median(resp)                      # dead
    # grow by one pixel: a hot pixel bleeds into neighbours through the PSF
    bad = ndimage.binary_dilation(bad, np.ones((3, 3), bool))

    good = ~bad
    good[:EDGE, :] = good[-EDGE:, :] = False
    good[:, :EDGE] = good[:, -EDGE:] = False
    return good


# --------------------------------------------------------------------------
# difference images with drift / flux removal
# --------------------------------------------------------------------------

def detrend_tilewise(diff, ref, tile=TILE, basis_grad=True):
    """Remove, tile by tile, the component of `diff` lying in the span of
    {d ref/dx, d ref/dy, ref, 1}.

    ref is the (nearly noise-free) stack mean.  The gradient terms absorb
    sub-pixel drift between the two frames, the `ref` term absorbs a residual
    flux-scale mismatch, the constant absorbs offset drift.

    Returns (residual, coefficient array).  Coefficients for the gradient
    terms are the fitted drift in pixels (times 1/sqrt2 scaling of the caller).
    """
    gy, gx = np.gradient(ref)
    cols = [gx, gy, ref, np.ones_like(ref)] if basis_grad else [ref, np.ones_like(ref)]
    ny, nx = diff.shape
    res = diff.copy()
    coefs = np.zeros((ny // tile, nx // tile, len(cols)))
    for i in range(ny // tile):
        for j in range(nx // tile):
            sl = (slice(i * tile, (i + 1) * tile), slice(j * tile, (j + 1) * tile))
            A = np.stack([c[sl].ravel() for c in cols], 1)
            b = diff[sl].ravel()
            ok = np.isfinite(b)
            c, *_ = np.linalg.lstsq(A[ok], b[ok], rcond=None)
            res[sl] = (b - A @ c).reshape(tile, tile)
            coefs[i, j] = c
    return res, coefs


def make_pair(stack, idx_a, idx_b, ref, detrend=True):
    """Difference image D = (a - b)/sqrt2 and mean level M = (a+b)/2.

    D has the variance of a single frame; M is the signal level that variance
    should be compared against.  Static structure, fixed-pattern noise, the
    dark offset and the dark fixed pattern all cancel exactly in D.
    """
    a, b = stack[idx_a], stack[idx_b]
    # coarse flux match first so the least-squares step only has to fix a
    # residual; median ratio is robust to the code structure.
    s = np.median(b) / np.median(a) if np.median(a) != 0 else 1.0
    raw = (a * s - b) / np.sqrt(2.0)
    M = 0.5 * (a * s + b)
    if detrend:
        D, coefs = detrend_tilewise(raw, ref / np.sqrt(2.0))
        drift = coefs[..., :2] * np.sqrt(2.0)   # undo the 1/sqrt2 on the basis
    else:
        D, drift = raw, None
    return D, M, drift


# --------------------------------------------------------------------------
# photon transfer curve
# --------------------------------------------------------------------------

def ptc(D, M, good, nbin=40, lo=2.0, hi=98.0, min_count=2000):
    """Bin pixels by mean level, robust variance per bin.  EMVA 1288 sec 6.

    Returns dict with bin centres, variance, counts, and the straight-line fit
    var = gain_eff * mean + read^2 over the bins.
    """
    m = M[good]
    d = D[good]
    edges = np.percentile(m, np.linspace(lo, hi, nbin + 1))
    edges = np.unique(edges)
    idx = np.digitize(m, edges) - 1
    cen, var, var_plain, cnt = [], [], [], []
    for k in range(len(edges) - 1):
        sel = idx == k
        n = int(sel.sum())
        if n < min_count:
            continue
        dk = d[sel]
        cen.append(float(np.median(m[sel])))
        var.append(float(iqr_sigma(dk) ** 2))      # robust: used for the fit
        var_plain.append(float(dk.var()))          # non-robust: tail diagnostic
        cnt.append(n)
    cen = np.array(cen); var = np.array(var)
    var_plain = np.array(var_plain); cnt = np.array(cnt)

    # weighted straight line; variance of a variance estimate ~ 2 var^2 / n
    w = cnt / (2.0 * np.maximum(var, 1e-9) ** 2)
    A = np.stack([cen, np.ones_like(cen)], 1)
    W = w[:, None]
    coef, *_ = np.linalg.lstsq(A * W, var * w, rcond=None)
    gain_eff, inter = float(coef[0]), float(coef[1])

    # parameter uncertainty from the weighted normal equations
    cov = np.linalg.inv((A * W).T @ A)
    resid = var - A @ coef
    chi2 = float(np.sum(w * resid ** 2) / max(len(cen) - 2, 1))
    err = np.sqrt(np.diag(cov) * max(chi2, 1.0))

    return dict(mean=cen, var=var, var_plain=var_plain, count=cnt,
                gain_eff=gain_eff, gain_eff_err=float(err[0]),
                intercept=inter, intercept_err=float(err[1]),
                read_adu=float(np.sqrt(inter)) if inter > 0 else float("nan"),
                chi2=chi2)


def ptc_gradient_controlled(D, M, ref, good, nbin=16, ngbin=4):
    """Cross-check: 2-D fit  var = g*M + r^2 + c*|grad M|^2.

    If the drift removal worked, c should be consistent with zero and g should
    match the 1-D fit.  A large positive c means residual motion is still
    leaking into the variance.
    """
    gy, gx = np.gradient(ref)
    G2 = gx ** 2 + gy ** 2
    m, d, g2 = M[good], D[good], G2[good]
    me = np.percentile(m, np.linspace(2, 98, nbin + 1))
    ge = np.percentile(g2, np.linspace(0, 96, ngbin + 1))
    rows, rhs, wts = [], [], []
    for i in range(nbin):
        for j in range(ngbin):
            sel = (m >= me[i]) & (m < me[i + 1]) & (g2 >= ge[j]) & (g2 < ge[j + 1])
            n = int(sel.sum())
            if n < 2000:
                continue
            v = iqr_sigma(d[sel]) ** 2
            rows.append([np.median(m[sel]), 1.0, np.median(g2[sel])])
            rhs.append(v); wts.append(n / (2 * v ** 2))
    A = np.array(rows); b = np.array(rhs); w = np.array(wts)
    coef, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
    return dict(gain_eff=float(coef[0]), intercept=float(coef[1]),
                grad_coef=float(coef[2]), npoint=len(b))


# --------------------------------------------------------------------------
# NPS / autocorrelation
# --------------------------------------------------------------------------

def nps_tiles(D, good, tile=NPS_TILE, max_tiles=160, bad_frac=0.03):
    """Average 2-D noise power spectrum over tiles.

    Bad pixels and outliers are zeroed rather than causing the tile to be
    dropped -- at tile=512 every tile contains at least one defect, so
    requiring pure tiles would leave none.  The zeroed fraction is <1%, which
    costs a little broadband power but does not shape the spectrum.  The
    power is rescaled by 1/f_kept so the total still matches the variance.

    Normalised so that the mean of the NPS over all frequencies equals the
    pixel variance (Parseval).  Returns (nps2d, ntiles, variance).
    """
    ny, nx = D.shape
    acc = np.zeros((tile, tile)); n = 0
    var_acc = 0.0
    for i in range(ny // tile):
        for j in range(nx // tile):
            sl = (slice(i * tile, (i + 1) * tile), slice(j * tile, (j + 1) * tile))
            g = good[sl]
            if 1.0 - g.mean() > bad_frac:
                continue
            t = D[sl].astype(np.float64)
            t = np.where(g, t, 0.0)
            t = t - t.sum() / max(g.sum(), 1)
            t = np.where(g, t, 0.0)
            # zero out cosmic rays / hot pixels rather than rejecting the whole
            # tile: at 512 px a tile almost always contains one, and discarding
            # tiles would leave no data.  The outlier fraction is ~1e-5, so the
            # variance lost is negligible and spectrally flat.
            s = iqr_sigma(t[g])
            out = (np.abs(t) > 8 * s) | ~g
            if out.mean() > bad_frac:
                continue
            t[out] = 0.0
            keep = 1.0 - out.mean()
            t = np.where(out, 0.0, t - t.sum() / (t.size * keep))
            F = np.fft.fft2(t)
            acc += (F.real ** 2 + F.imag ** 2) / (tile * tile * keep)
            var_acc += t.sum() ** 2 / t.size
            n += 1
            if n >= max_tiles:
                break
        if n >= max_tiles:
            break
    if n == 0:
        raise RuntimeError("no clean tiles")
    return acc / n, n, var_acc / n


def autocorr_from_nps(nps2d):
    """Normalised spatial autocorrelation rho(dy,dx), rho(0,0)=1, centred."""
    ac = np.fft.ifft2(nps2d).real
    ac = np.fft.fftshift(ac)
    return ac / ac[ac.shape[0] // 2, ac.shape[1] // 2]


def n_eff_from_autocorr(rho, rmax=12):
    """N_eff = sum over all lags of the normalised autocorrelation.

    For noise generated by quanta each spread over a normalised PSF p,
    C(d) = g^2 lam (p (*) p)(d), so sum_d C(d) = g^2 lam (sum p)^2 = g^2 lam
    while C(0) = g^2 lam sum p^2.  Hence sum_d rho(d) = 1/sum p^2 = N_eff,
    the number of pixels each quantum is effectively spread over, and the
    factor by which a per-pixel variance understates the quanta.
    """
    cy, cx = rho.shape[0] // 2, rho.shape[1] // 2
    w = rho[cy - rmax:cy + rmax + 1, cx - rmax:cx + rmax + 1]
    return float(w.sum())


def radial_profile(img, nbin=64):
    ny, nx = img.shape
    fy = np.fft.fftshift(np.fft.fftfreq(ny))
    fx = np.fft.fftshift(np.fft.fftfreq(nx))
    R = np.hypot(*np.meshgrid(fy, fx, indexing="ij"))
    img = np.fft.fftshift(img)
    edges = np.linspace(0, 0.5, nbin + 1)
    idx = np.digitize(R.ravel(), edges) - 1
    v = img.ravel()
    prof = np.array([v[idx == k].mean() if np.any(idx == k) else np.nan
                     for k in range(nbin)])
    return 0.5 * (edges[1:] + edges[:-1]), prof


def correlation_length(rho):
    """FWHM of the central lobe along x and y, by linear interpolation,
    and the 1/e radius of a radial cut."""
    cy, cx = rho.shape[0] // 2, rho.shape[1] // 2
    out = {}
    for name, cut in (("x", rho[cy, cx:cx + 16]), ("y", rho[cy:cy + 16, cx])):
        half = 0.5
        k = np.argmax(cut < half)
        if k == 0:
            out[name] = 0.0
        else:
            f = (cut[k - 1] - half) / (cut[k - 1] - cut[k])
            out[name] = 2.0 * (k - 1 + f)        # full width = 2 * half width
    return out


# --------------------------------------------------------------------------
# distribution
# --------------------------------------------------------------------------

def distribution(D, M, good, varcoef, read_adu, min_level=1000.0):
    """Normalised residual z = D / sigma(M).

    sigma comes from the QUADRATIC PTC fit, not a linear extrapolation: the
    curve bends, and a straight line extrapolated into the code-shadowed
    pixels passes through zero variance and produces spurious huge z.  Pixels
    below min_level are excluded for the same reason, and the predicted
    variance is floored at the measured read noise.
    """
    sel = good & (M > min_level)
    var = np.polyval(varcoef, M[sel])
    sig = np.sqrt(np.maximum(var, read_adu ** 2))
    z = D[sel] / sig
    z = z[np.isfinite(z)]
    rs = iqr_sigma(z)
    core = z[np.abs(z) < 4]
    return dict(
        n=int(z.size),
        mean=float(z.mean()), std=float(z.std()), iqr_sigma=float(rs),
        skew=float(np.mean(((z - z.mean()) / z.std()) ** 3)),
        kurtosis_excess=float(np.mean(((z - z.mean()) / z.std()) ** 4) - 3.0),
        core_skew=float(np.mean(((core - core.mean()) / core.std()) ** 3)),
        core_kurtosis_excess=float(np.mean(((core - core.mean()) / core.std()) ** 4) - 3.0),
        frac_gt3=float(np.mean(np.abs(z) > 3)), gauss_gt3=2.6998e-3,
        frac_gt5=float(np.mean(np.abs(z) > 5)), gauss_gt5=5.733e-7,
        frac_gt8=float(np.mean(np.abs(z) > 8)), gauss_gt8=1.244e-15,
        max_abs=float(np.abs(z).max()),
        hist=np.histogram(np.clip(z, -8, 8), bins=161, range=(-8, 8))[0].tolist(),
    )


# --------------------------------------------------------------------------
# driver per dataset
# --------------------------------------------------------------------------

def analyse_stack(name, stack, dark, good, results, nps_pairs=3):
    """Full PTC + NPS + distribution on one white stack."""
    dmean = dark.mean(0)
    sub = stack - dmean
    ref = sub.mean(0)

    npair = min(8, len(sub) - 1)
    pair_fits, drifts = [], []
    Dstore = None
    for k in range(npair):
        D, M, drift = make_pair(sub, k, k + 1, ref)
        f = ptc(D, M, good)
        pair_fits.append((f["gain_eff"], f["intercept"]))
        drifts.append(np.abs(drift).mean(axis=(0, 1)).tolist())
        if k == 0:
            Dstore, Mstore = D, M
            main = f
            gc = ptc_gradient_controlled(D, M, ref, good)
            # no-detrend control, to show how much drift the fit removed
            Dn, Mn, _ = make_pair(sub, k, k + 1, ref, detrend=False)
            nodet = ptc(Dn, Mn, good)

    pf = np.array(pair_fits)
    res = dict(
        gain_eff=main["gain_eff"], gain_eff_err=main["gain_eff_err"],
        intercept=main["intercept"], read_adu=main["read_adu"],
        chi2=main["chi2"],
        gain_eff_pairspread=float(pf[:, 0].std()),
        gain_eff_pairmean=float(pf[:, 0].mean()),
        read_adu_pairmean=float(np.sqrt(np.maximum(pf[:, 1], 0)).mean()),
        npairs=npair,
        drift_px_mean=float(np.mean(drifts)),
        drift_px_max=float(np.max(np.abs(drifts))),
        no_detrend=dict(gain_eff=nodet["gain_eff"], read_adu=nodet["read_adu"]),
        grad_control=gc,
        ptc_curve=dict(mean=main["mean"].tolist(), var=main["var"].tolist(),
                       var_plain=main["var_plain"].tolist(),
                       count=main["count"].tolist()),
        level_median=float(np.median(ref[good])),
        level_p10=float(np.percentile(ref[good], 10)),
        level_p90=float(np.percentile(ref[good], 90)),
    )

    # --- NPS, averaged over several pairs for stability
    nps_acc, ntot = None, 0
    for k in range(min(nps_pairs, npair)):
        D, M, _ = make_pair(sub, k, k + 1, ref)
        n2, nt, _ = nps_tiles(D, good)
        nps_acc = n2 if nps_acc is None else nps_acc + n2
        ntot += nt
    nps2d = nps_acc / min(nps_pairs, npair)
    rho = autocorr_from_nps(nps2d)
    fr, prof = radial_profile(nps2d)
    res["nps"] = dict(
        ntiles=ntot,
        freq=fr.tolist(), profile=prof.tolist(),
        white_level=float(np.median(prof)),
        nps_dc_over_mean=float(prof[0] / np.nanmean(prof)),
        rho_10=float(rho[rho.shape[0] // 2, rho.shape[1] // 2 + 1]),
        rho_01=float(rho[rho.shape[0] // 2 + 1, rho.shape[1] // 2]),
        rho_11=float(rho[rho.shape[0] // 2 + 1, rho.shape[1] // 2 + 1]),
        rho_20=float(rho[rho.shape[0] // 2, rho.shape[1] // 2 + 2]),
        rho_30=float(rho[rho.shape[0] // 2, rho.shape[1] // 2 + 3]),
        n_eff_r4=n_eff_from_autocorr(rho, 4),
        n_eff_r8=n_eff_from_autocorr(rho, 8),
        n_eff_r12=n_eff_from_autocorr(rho, 12),
        fwhm=correlation_length(rho),
    )
    res["rho_cut"] = rho[rho.shape[0] // 2, rho.shape[1] // 2:rho.shape[1] // 2 + 12].tolist()

    # --- distribution
    res["dist"] = distribution(Dstore, Mstore, good, main["gain_eff"], main["intercept"])

    # --- stationarity across the chip: 4x4 blocks of 512
    tiles = []
    for i in range(4):
        for j in range(4):
            sl = (slice(i * 512, (i + 1) * 512), slice(j * 512, (j + 1) * 512))
            try:
                ft = ptc(Dstore[sl], Mstore[sl], good[sl], nbin=14, min_count=1500)
                tiles.append(dict(i=i, j=j, gain_eff=ft["gain_eff"],
                                  read_adu=ft["read_adu"],
                                  level=float(np.median(Mstore[sl][good[sl]]))))
            except Exception:
                pass
    res["tiles"] = tiles
    gt = np.array([t["gain_eff"] for t in tiles])
    res["tile_gain_mean"] = float(gt.mean())
    res["tile_gain_std"] = float(gt.std())
    res["tile_gain_min"] = float(gt.min())
    res["tile_gain_max"] = float(gt.max())

    results[name] = res
    return res, Dstore, Mstore, ref


def analyse_dark(dark, good):
    """EMVA 1288 dark characterisation: temporal read noise from frame
    differences (cancels fixed pattern), plus the dark NPS."""
    d = dark
    sig = []
    for k in range(len(d) - 1):
        diff = (d[k] - d[k + 1]) / np.sqrt(2.0)
        sig.append(iqr_sigma(diff[good]))
    sig = np.array(sig)
    diff = (d[0] - d[1]) / np.sqrt(2.0)
    nps2d, nt, _ = nps_tiles(diff, good)
    rho = autocorr_from_nps(nps2d)
    fr, prof = radial_profile(nps2d)
    return dict(
        offset_adu=float(np.median(d.mean(0)[good])),
        read_adu_temporal=float(sig.mean()),
        read_adu_spread=float(sig.std()),
        read_adu_plain=float((d[0] - d[1])[good].std() / np.sqrt(2)),
        fpn_adu=float(iqr_sigma(d.mean(0)[good])),
        rho_10=float(rho[rho.shape[0] // 2, rho.shape[1] // 2 + 1]),
        rho_01=float(rho[rho.shape[0] // 2 + 1, rho.shape[1] // 2]),
        n_eff_r8=n_eff_from_autocorr(rho, 8),
        nps_profile=prof.tolist(), nps_freq=fr.tolist(),
        fwhm=correlation_length(rho),
    )


def run(dataset, results):
    print(f"\n{'='*70}\n{dataset.upper()}\n{'='*70}", flush=True)
    d = load(dataset)
    whites_all = np.concatenate([d["white0"], d.get("white1", d["white0"][:0])], 0)
    good = bad_pixel_mask(d["dark"], whites_all)
    print(f"good pixels: {good.sum()} / {good.size} ({100*good.mean():.3f}%)", flush=True)

    res = dict(good_frac=float(good.mean()),
               n_bad=int((~good).sum()),
               data_level_median=float(np.median((d["data_lvl"] - d["dark"].mean(0))[good])),
               white_shape=list(d["white0"].shape))
    if "bpm4" in d:
        b = d["bpm4"]
        res["bpm4_drift_pct"] = float(100 * (b.max() - b.min()) / b.mean())

    print("  dark ...", flush=True)
    res["dark"] = analyse_dark(d["dark"], good)
    print(f"    offset {res['dark']['offset_adu']:.1f} ADU, "
          f"read {res['dark']['read_adu_temporal']:.2f} ADU", flush=True)

    sub = dict()
    print("  white0 ...", flush=True)
    r0, D0, M0, ref0 = analyse_stack("white0", d["white0"], d["dark"], good, sub)
    print(f"    gain_eff {r0['gain_eff']:.3f} +- {r0['gain_eff_err']:.3f} ADU/q, "
          f"read {r0['read_adu']:.2f} ADU, N_eff(r8) {r0['nps']['n_eff_r8']:.2f}", flush=True)

    if "white1" in d:
        print("  white1 ...", flush=True)
        r1, *_ = analyse_stack("white1", d["white1"], d["dark"], good, sub)
        print(f"    gain_eff {r1['gain_eff']:.3f} +- {r1['gain_eff_err']:.3f} ADU/q, "
              f"read {r1['read_adu']:.2f} ADU, N_eff(r8) {r1['nps']['n_eff_r8']:.2f}", flush=True)

    res["stacks"] = sub
    results[dataset] = res

    np.savez_compressed(os.path.join(OUT, f"arrays_{dataset}.npz"),
                        D=D0[::2, ::2].astype(np.float32),
                        M=M0[::2, ::2].astype(np.float32),
                        good=good[::2, ::2])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="both", choices=["ht", "nfp", "both"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    results = {}
    for ds in (["ht", "nfp"] if a.dataset == "both" else [a.dataset]):
        run(ds, results)
    with open(os.path.join(OUT, "noise_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {OUT}/noise_results.json")


if __name__ == "__main__":
    main()
