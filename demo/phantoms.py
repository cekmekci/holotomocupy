"""Synthetic sample and probe for the demo: a Siemens star and a 3-D phantom.

Shared by the notebooks and by the MPI scripts next to them, so the two always
build the same data.  Pure numpy/scipy except where cupy is faster; no MPI.
"""
import os
import urllib.request

import numpy as np
import scipy.ndimage as ndimage
from scipy.fft import fft2, fftn, fftshift, ifft2, ifftn

PRB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'data', 'prb_id16a')
# The measured probe is 2 x 64 MB, too big for git, so it is fetched on first
# use and cached in PRB_DIR (gitignored).  Same files tests/nfp downloads.
PRB_URL = ('https://g-110014.fd635.8443.data.globus.org/holotomocupy/'
           'examples_synthetic/data/prb_id16a')
PRB_FILES = ('prb_abs_2048.tiff', 'prb_phase_2048.tiff')


def fetch_probe():
    """Download the ID16A probe once; rank 0 only, others wait."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank, barrier = comm.Get_rank(), comm.Barrier
    except Exception:
        rank, barrier = 0, lambda: None
    if rank == 0:
        os.makedirs(PRB_DIR, exist_ok=True)
        for f in PRB_FILES:
            dest = os.path.join(PRB_DIR, f)
            if os.path.exists(dest):
                continue
            print(f'demo: fetching {f} (64 MB) into {PRB_DIR}', flush=True)
            tmp = dest + '.part'          # so an interrupted get is not cached
            urllib.request.urlretrieve(f'{PRB_URL}/{f}', tmp)
            os.replace(tmp, dest)
    barrier()


# ----------------------------------------------------------- 2-D: the star --
def siemens_star(nobj, step_deg=15):
    """(nobj, nobj) float32 Siemens-star mask, spokes every step_deg."""
    def rotate(pts, ang, c):
        s, co = np.sin(ang), np.cos(ang)
        return (pts - c) @ np.array([[co, -s], [s, co]], 'float32').T + c

    def tri(X, Y, t):
        (x1, y1), (x2, y2), (x3, y3) = t
        d = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if d == 0:
            return np.zeros_like(X, dtype=bool)
        a = ((y2 - y3) * (X - x3) + (x3 - x2) * (Y - y3)) / d
        b = ((y3 - y1) * (X - x3) + (x1 - x3) * (Y - y3)) / d
        return (a >= 0) & (b >= 0) & (1 - a - b >= 0)

    t0 = np.array([(1.5 * nobj // 16, nobj // 2 - nobj // 32),
                   (1.5 * nobj // 16, nobj // 2 + nobj // 32),
                   (nobj // 2 - nobj // 128, nobj // 2)], dtype='float32')
    yy, xx = np.mgrid[0:nobj, 0:nobj]
    c = np.array([nobj / 2, nobj / 2], dtype='float32')
    star = np.zeros((nobj, nobj), dtype='float32')
    for deg in range(0, 360, step_deg):
        star += tri(xx, yy, rotate(t0, np.deg2rad(deg), c)).astype('float32')
    return star / (star.max() or 1.0)


def star_projection(nobj, delta_beta=29.0, smooth=8.0):
    """complex64 (nobj, nobj): the star as a transmission, -delta + i*beta."""
    star = siemens_star(nobj)
    v = np.arange(-nobj // 2, nobj // 2, dtype='float32') / nobj
    vx, vy = np.meshgrid(v, v, indexing='ij')
    g = fftshift(np.exp(-smooth * (vx**2 + vy**2)).astype('float32'))
    star = ifft2(fft2(star) * g).real
    star = star / star.max() * (np.pi / 8)          # phase range [0, pi/8]
    return (-star + 1j * star / delta_beta).astype('complex64')


# -------------------------------------------------------- 3-D: the phantom --
def _frame_edges(cube, p1, p2):
    """The 12 edges of the axis-aligned cube [p1, p2]^3, set to 1."""
    for a, b in ((p1, p1), (p1, p2), (p2, p1), (p2, p2)):
        cube[p1:p2, a, b] = 1
        cube[a, p1:p2, b] = 1
        cube[a, b, p1:p2] = 1


def _rotate3d(vol, ang_xy=28, ang_xz=45, order=1):
    a, b = np.deg2rad(ang_xy), np.deg2rad(ang_xz)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    A = np.linalg.inv(Ry @ Rz)
    c = (np.array(vol.shape) - 1) / 2.0
    return ndimage.affine_transform(vol, A, offset=c - A @ c, order=order,
                                    mode='constant', cval=0.0, prefilter=False)


def phantom3d(n, delta=1.0, beta=1e-2):
    """complex64 (n, n, n): nested cube frames, rotated off-axis and smoothed.

    Edges in every direction and a wide range of feature sizes, which is what
    makes it a fair test of resolution and of the position refinement.
    """
    amps = np.array([3, -3, 1, 3, -4, 1, 4], dtype='float32')
    dil = (np.array([33, 28, 25, 21, 16, 10, 3], dtype='float32') / 256.0) * n
    ax = np.arange(-n // 2, n // 2, dtype='float32')
    x, y, z = np.meshgrid(ax, ax, ax, indexing='ij')
    r2 = x * x + y * y + z * z
    del x, y, z

    obj = np.zeros((n, n, n), dtype='float32')
    cube = np.zeros((n, n, n), dtype='float32')
    r = int(n * 0.2)
    for amp, d in zip(amps, dil):
        cube.fill(0.0)
        _frame_edges(cube, n // 2 - r // 2, n // 2 + r // 2)
        fc = fftn(fftshift(cube), workers=-1)
        fs = fftn(fftshift((r2 < d * d).astype('float32')), workers=-1)
        obj += amp * (fftshift(ifftn(fc * fs, workers=-1)).real > 1.0)

    obj = _rotate3d(obj)
    obj = np.roll(obj, (-10 * n // 256, -15 * n // 256), axis=(1, 2))
    np.maximum(obj, 0, out=obj)

    v = np.arange(-n // 2, n // 2, dtype='float32') / n
    vx, vy, vz = np.meshgrid(v, v, v, indexing='ij')
    filt = fftshift(np.exp(-3.0 * (vx**2 + vy**2 + vz**2)).astype('float32'))
    obj = ifftn(fftn(obj) * filt).real
    obj[obj < 0] = 0
    return (obj * (-delta + 1j * beta)).astype('complex64')


# ------------------------------------------------------------- the probe ----
def id16a_probe(n, ndist=1, smooth=4.0):
    """complex64 (ndist, n, n): the measured ID16A probe, cropped and smoothed.

    Normalised to unit mean amplitude so the data are ~1 and the regularisation
    weights mean the same thing at any size.
    """
    from holotomocupy.utils import read_tiff

    fetch_probe()
    a = read_tiff(os.path.join(PRB_DIR, PRB_FILES[0]))[:ndist]
    p = read_tiff(os.path.join(PRB_DIR, PRB_FILES[1]))[:ndist]
    if a.shape[0] < ndist:
        raise ValueError(f'probe has {a.shape[0]} planes, need ndist={ndist}')
    if n > a.shape[1]:
        raise ValueError(f'probe is {a.shape[1]}^2, too small for n={n}')
    prb = (a * np.exp(1j * p)).astype('complex64')
    c0, c1 = prb.shape[1] // 2, prb.shape[2] // 2
    prb = prb[:, c0 - n // 2:c0 + n // 2, c1 - n // 2:c1 + n // 2]

    v = np.arange(-n // 2, n // 2, dtype='float32') / n
    vx, vy = np.meshgrid(v, v, indexing='ij')
    filt = fftshift(np.exp(-smooth * (vx**2 + vy**2)).astype('float32'))
    prb = ifft2(fft2(prb) * filt).astype('complex64')
    return prb / np.mean(np.abs(prb), axis=(1, 2))[:, None, None]


# -------------------------------------------------------- the coded mask ----
# Ported from phascade/simulations/esrf_simulation/physics.py, which is the
# validated single-coded-aperture ESRF simulation.  Three things in here are
# easy to get wrong and all three matter:
#
#   * the pattern is built on the APERTURE's own plane pixel size, not the
#     sample voxel.  Those differ by the code-plane magnification -- 342 nm
#     against 80 nm in the ESRF geometry -- so using the voxel misstates the
#     fabricated feature size by 4.3x.
#   * the transmission is real gold: exp(i*k*(-delta + i*beta)*t*pattern) with
#     delta, beta from xraylib.  Contrast goes as 2|sin(phi/2)|, so it PEAKS at
#     the thickness that gives pi (6.49 um of Au at 33.35 keV) -- a micron is
#     only a quarter of that and is why thin apertures underperform.
#   * the lattice is generated directly in rotated coordinates from a hash, so
#     a rotated mask costs no interpolation, and supersampling antialiases the
#     cell edges instead of an FFT low-pass.

# delta, beta for Au (19.3 g/cm^3), from xraylib.  Pinned here because
# xraylib is NOT a holotomocupy dependency and the demo must run without it;
# au_optical_constants prefers the live value when it is installed.
_AU_PINNED = {33.35: (2.863698e-06, 1.190692e-07),
              17.10: (1.066593e-05, 1.306277e-06)}


def au_optical_constants(energy_kev, material='Au', density=19.3):
    """(delta, beta) at energy_kev.  xraylib when importable, else the table."""
    try:
        import xraylib
        return (1 - xraylib.Refractive_Index_Re(material, energy_kev, density),
                xraylib.Refractive_Index_Im(material, energy_kev, density))
    except ImportError:
        key = min(_AU_PINNED, key=lambda k: abs(k - energy_kev))
        if abs(key - energy_kev) > 1e-6 or material != 'Au':
            raise ImportError(
                f'xraylib is needed for {material} at {energy_kev} keV; '
                f'pinned values exist only for Au at {sorted(_AU_PINNED)} keV')
        return _AU_PINNED[key]


def thickness_for_phase(phase_rad, energy_kev, material='Au', density=19.3):
    """Absorber thickness [m] giving `phase_rad` of phase shift.  t_pi is the
    thickness where the code contrast 2|sin(phi/2)| is largest."""
    delta, _ = au_optical_constants(energy_kev, material, density)
    k = 2 * np.pi / (1.24e-09 / energy_kev)
    return phase_rad / (k * delta)


def _hash_bits(iu, iv, seed):
    """Deterministic 0/1 per integer cell index (splitmix64 finalizer), so the
    pattern can be generated directly in rotated coordinates with no
    interpolation."""
    u64 = np.uint64
    seed_term = u64((seed & 0xFFFFFFFF) * 0x165667B19E3779F9 % (1 << 64))
    h = (iu.astype(u64) * u64(0x9E3779B97F4A7C15)) ^ \
        (iv.astype(u64) * u64(0xC2B2AE3D27D4EB4F)) ^ seed_term
    h ^= h >> u64(30); h *= u64(0xBF58476D1CE4E5B9)
    h ^= h >> u64(27); h *= u64(0x94D049BB133111EB)
    h ^= h >> u64(31)
    return (h & u64(1)).astype(np.float32)


def code_pattern(n, feature_m, pixel_size_m, angle_deg=45.0, seed=0,
                 supersample=2, rows_per_chunk=512, shape='square'):
    """(n, n) float32 fill fraction per pixel of a random binary lattice.

    shape='square': each occupied cell filled completely, feature = side,
    fill ~50%.  shape='circle': inscribed disc per cell, feature = diameter,
    fill = 0.5*pi/4 = 39.3%.  `supersample` averages s x s sub-samples per
    pixel to antialias the edges.
    """
    if shape not in ('square', 'circle'):
        raise ValueError(f"shape must be 'square' or 'circle', got {shape!r}")
    s = max(1, int(supersample))
    w = float(feature_m) / float(pixel_size_m)          # cell side, in pixels
    if w < 1e-6:
        raise ValueError('feature size collapses to zero pixels')
    th = np.deg2rad(angle_deg)
    ct, st = np.cos(th), np.sin(th)

    sub = (np.arange(s, dtype=np.float64) + 0.5) / s - 0.5
    cols = np.arange(n, dtype=np.float64) - n / 2.0 + 0.5
    xs = (cols[:, None] + sub[None, :]).ravel()

    out = np.empty((n, n), dtype=np.float32)
    for r0 in range(0, n, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, n)
        rows = np.arange(r0, r1, dtype=np.float64) - n / 2.0 + 0.5
        ys = (rows[:, None] + sub[None, :]).ravel()
        Y, X = np.meshgrid(ys, xs, indexing='ij')
        u = (X * ct + Y * st) / w
        v = (-X * st + Y * ct) / w
        iu = np.floor(u).astype(np.int64)
        iv = np.floor(v).astype(np.int64)
        bits = _hash_bits(iu, iv, seed)
        if shape == 'circle':
            du, dv = u - iu - 0.5, v - iv - 0.5
            bits = bits * ((du * du + dv * dv) <= 0.25)
        out[r0:r1] = bits.reshape(r1 - r0, s, n, s).mean(axis=(1, 3))
    return out


def coded_aperture(n, feature_m, pixel_size_m, thickness_m, energy_kev,
                   angle_deg=45.0, seed=0, supersample=2, shape='square',
                   material='Au', density=19.3):
    """complex64 (n, n): exp(i*k*(-delta + i*beta)*t*pattern).

    `pixel_size_m` must be the APERTURE plane's pixel size, not the sample
    voxel -- see the note at the top of this section.
    """
    pattern = code_pattern(n, feature_m, pixel_size_m, angle_deg, seed,
                           supersample, shape=shape)
    delta, beta = au_optical_constants(energy_kev, material, density)
    k = 2 * np.pi / (1.24e-09 / energy_kev)
    arg = 1j * k * pattern * (-delta + 1j * beta) * float(thickness_m)
    return np.exp(arg).astype('complex64')
