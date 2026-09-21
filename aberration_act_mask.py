"""
Reconstruction of the CMB aberration dipole with a lensing quadratic estimator.

Defaults: lmax 3000 at 3 arcmin resolution, reconstruction kept out to L = 5.

The released ACT DR6 healpix mask is put onto the CAR grid with
pixell.reproject.healpix2map(..., method="spline"), which reads the healpix map
at each CAR pixel centre by bilinear interpolation.  That is the right choice
for a mask: the result is a convex combination of nearby input values, so it
cannot leave [0, 1], where the harmonic method rings around the sharp edges of
a footprint and drives an all-positive mask negative.  --dr6-method harm and
--dr6-method average (the old area-average binning) are kept so the three can
be compared on the same run; the printed w1, w2, w4 ratios are where to look.

The response is measured with boosts along x, y and z, and reported two ways
from exactly the same paired sims:

  * the 3x3 matrix R, which is the L = 1 part, and
  * the (L, M) response <a_LM> / (-beta) at every L out to --lout, printed as
    the coefficients themselves rather than collapsed into a power spectrum.

The second contains the first: taking the L = 1 rows of the alm response back
through l1_vector returns R exactly, which is why both come out of one array.
Everything above L = 1 is the leakage the boost leaves behind.
"""

import os
import sys


# 0. Process and thread layout
#
# The sims are independent of one another, so the stages below hand them to a
# pool of worker processes inside this one job (--nproc) rather than relying on
# one process per shard and a separate queue slot for each.  Threads have to be
# capped to match: without that, --nproc copies of an SHT each try to take the
# whole node and the run gets slower, not faster.  numpy, healpy and pixell read
# these variables when their backends load, which is why this sits above the
# imports instead of next to argparse.


def _peek(flag, default):
    """Value of `--flag V` or `--flag=V`, read straight off the command line."""
    argv = sys.argv[1:]
    for k, a in enumerate(argv):
        if a == flag and k + 1 < len(argv):
            return argv[k + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return default


# sched_getaffinity, not cpu_count: under a scheduler the job usually owns a
# subset of the node and cpu_count reports the whole machine.
if hasattr(os, "sched_getaffinity"):
    CORES = len(os.sched_getaffinity(0))
else:
    CORES = os.cpu_count() or 1

NPROC = max(1, int(_peek("--nproc", "1")))
# A thread budget already in the environment (SLURM sets one from
# --cpus-per-task) is what gets divided up; --threads overrides the division.
_BUDGET = int(os.environ.get("OMP_NUM_THREADS") or CORES)
THREADS = max(1, int(_peek("--threads", "0")) or _BUDGET // NPROC)

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "DUCC0_NUM_THREADS"):
    os.environ[_var] = str(THREADS)

import argparse
import hashlib
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context

import numpy as np
import healpy as hp
import camb

from pixell import enmap, curvedsky, aberration, reproject, utils
from falafel import qe
import pytempura


# 1. Configuration

DR6_DEFAULT_NAME = "mask_act_dr6_lensing_v1_healpix_nside_4096_baseline.fits"

parser = argparse.ArgumentParser(
    description="CMB aberration reconstruction with a lensing quadratic estimator")
parser.add_argument("--quick", action="store_true")
parser.add_argument("--seed-offset", type=int, default=0)
parser.add_argument("--noise", type=float, default=10.0,
                    help="white noise in T, uK-arcmin")
parser.add_argument("--pol-noise-factor", type=float, default=np.sqrt(2.0))
parser.add_argument("--beam", type=float, default=0.0, help="FWHM in arcmin")
parser.add_argument("--lmax", type=int, default=3000)
parser.add_argument("--res", type=float, default=3.0,
                    help="arcmin; pass 0 for the finest resolution "
                         "that still band-limits mlmax safely")
parser.add_argument("--lout", type=int, default=5,
                    help="highest reconstruction multipole kept.  The "
                         "alm response is reported over all of it")
parser.add_argument("--mask", default="act",
                    choices=["none", "box", "dec_band", "act", "dr6"])
parser.add_argument("--dr6-mask", default=None)
parser.add_argument("--dr6-variant", default="baseline")
parser.add_argument("--dr6-smooth-arcmin", type=float, default=0.0)
parser.add_argument("--dr6-method", default="spline",
                    choices=["spline", "harm", "average"],
                    help="healpix -> CAR interpolation.  spline is "
                         "pixell's recommendation for masks and "
                         "hitcounts; harm rings around sharp edges; "
                         "average is the old area-average binning, "
                         "kept as a cross-check")
parser.add_argument("--dr6-order", type=int, default=1, choices=[0, 1],
                    help="spline order: 1 bilinear, 0 nearest")
parser.add_argument("--dr6-rot", default=None,
                    help="coordinate rotation for the healpix file, "
                         "e.g. gal,cel.  The DR6 masks are already "
                         "equatorial, so the default is none")
parser.add_argument("--fsky", type=float, default=0.2,
                    help="for --mask act: target w1")
parser.add_argument("--act-dec", type=float, nargs=2, default=[-60.0, 22.0])
parser.add_argument("--ra-center", type=float, default=180.0)
parser.add_argument("--apod-deg", type=float, default=3.0)
parser.add_argument("--box-dec", type=float, nargs=2, default=[-20.0, 20.0])
parser.add_argument("--box-ra", type=float, nargs=2, default=[-60.0, 60.0])
parser.add_argument("--dec-band", type=float, default=20.0)
parser.add_argument("--n-meanfield", type=int, default=800)
parser.add_argument("--n-response", type=int, default=60)
parser.add_argument("--n-data", type=int, default=400)
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--nshards", type=int, default=1)
# Read in section 0, before the imports; declared here so they appear in --help
# and so an unknown-argument error still catches a typo.
parser.add_argument("--nproc", type=int, default=1,
                    help="worker processes inside this job.  Each holds its "
                         "own maps, so memory scales with it")
parser.add_argument("--threads", type=int, default=0,
                    help="threads per worker (default: the job's thread "
                         "budget divided by --nproc)")
parser.add_argument("--cache", default="cache_aberration")
parser.add_argument("--response-noise", action="store_true")
parser.add_argument("--response-beta", type=float, default=None)
parser.add_argument("--mask-thumb-deg", type=float, default=0.5)
parser.add_argument("--no-plots", dest="plots", action="store_false")
parser.add_argument("--plot-dir", default=None)
args = parser.parse_args()

# multipoles
LMIN = 2
LMAX = args.lmax
MLMAX = LMAX + 500
if args.res and args.res > 0:
    RES = args.res * utils.arcmin
else:
    RES = min(4.0, 10800.0 / (1.08 * MLMAX)) * utils.arcmin

# noise information
NOISE_UK_ARCMIN = args.noise
POL_NOISE_FACTOR = args.pol_noise_factor
BEAM_FWHM_ARCMIN = args.beam
TCMB_UK = 2.7255e6
C_KMS = 299792.458

# number of simulations
N_MEANFIELD = args.n_meanfield
N_RESPONSE = args.n_response
N_DATA = args.n_data

if args.quick:
    LMIN, LMAX, MLMAX = 2, 1000, 1200
    RES = 8.0 * utils.arcmin
    N_MEANFIELD, N_RESPONSE, N_DATA = 12, 4, 12


# The reconstruction is kept out to L_OUT, not just L = 1.  Storage is packed
# as (l, m) with l outer and m = 0..l inner, so index(l, m) = l(l+1)/2 + m and
# (1,0), (1,1) sit at 1 and 2.
L_OUT = args.lout

_pack_l = []
_pack_m = []
_pack_idx = []
for l in range(L_OUT + 1):
    for m in range(l + 1):
        _pack_l.append(l)
        _pack_m.append(m)
        _pack_idx.append(hp.Alm.getidx(MLMAX, l, m))

_PACK_L = np.array(_pack_l, dtype=int)
_PACK_M = np.array(_pack_m, dtype=int)
_PACK_IDX = np.array(_pack_idx)
_PACK_W = np.where(_PACK_M == 0, 1.0, 2.0) 
_NPACK = len(_PACK_L)
_ELLS = np.arange(L_OUT + 1)


# masking details
BOX_DEC_RANGE = tuple(args.box_dec)
BOX_RA_RANGE = tuple(args.box_ra)
DEC_BAND_DEG = args.dec_band
ACT_DEC = tuple(args.act_dec)
APOD = args.apod_deg


def dr6_path():
    """Where the released DR6 mask is."""
    if args.dr6_mask:
        return os.path.expanduser(args.dr6_mask)
    if os.environ.get("ACT_DR6_MASK"):
        return os.path.expanduser(os.environ["ACT_DR6_MASK"])
    return DR6_DEFAULT_NAME


DR6_PATH = dr6_path() if args.mask == "dr6" else None
# Whatever nside the mask file turns out to be.  Filled in when it is read or
# reloaded from cache; binning goes straight from the native grid.
DR6_NSIDE = 0

# A key naming every parameter the mask actually depends on, and only those,
# so an unused argument cannot invalidate a cache.
if args.mask == "none":
    MASK_KEY = "none"
elif args.mask == "dr6":
    MASK_KEY = f"dr6_{args.dr6_variant}_{args.dr6_method}"
    if args.dr6_method == "spline":
        MASK_KEY += str(args.dr6_order)
    if args.dr6_rot:
        MASK_KEY += "_" + args.dr6_rot.replace(",", "2")
    if args.dr6_smooth_arcmin > 0:
        MASK_KEY += f"_sm{args.dr6_smooth_arcmin:g}"
elif args.mask == "act":
    MASK_KEY = (f"act_dec{ACT_DEC[0]:g}to{ACT_DEC[1]:g}"
                f"_rac{args.ra_center:g}_fsky{args.fsky:g}")
elif args.mask == "box":
    MASK_KEY = (f"box_dec{BOX_DEC_RANGE[0]:g}to{BOX_DEC_RANGE[1]:g}"
                f"_ra{BOX_RA_RANGE[0]:g}to{BOX_RA_RANGE[1]:g}")
elif args.mask == "dec_band":
    MASK_KEY = f"decband{DEC_BAND_DEG:g}"

# --apod-deg does not enter the dr6 mask: the released one is already
# apodised, and --dr6-smooth-arcmin is the knob that changes it.
if args.mask != "dr6":
    MASK_KEY += f"_apod{APOD:g}"

# boost info
BETA = aberration.beta          # 0.001235 default
BDIR = aberration.dir_equ       # (ra, dec) in radians

RESPONSE_NOISE = args.response_noise
RESPONSE_BETA = BETA if args.response_beta is None else args.response_beta
RESPONSE_CENTRAL = RESPONSE_BETA != BETA

# These settings affect only the response cache, so they go in the file tags
# rather than in _config: putting them in the directory hash would discard the
# mean-field and data sims as well, which do not depend on them.
RESP_SUFFIX = "noisy" if RESPONSE_NOISE else "clean"
RESP_SUFFIX += f"_b{RESPONSE_BETA:g}"
if RESPONSE_CENTRAL:
    RESP_SUFFIX += "_ctr"

# Every estimator also gets its own case so the whisker plot can show how much
# the polarisation actually buys.
ESTIMATORS = ["TT", "TE", "EE"]
CASES = {
    "TT": ["TT"],
    "TE": ["TE"],
    "EE": ["EE"],
    "T+P": ["TT", "TE", "EE"],
}

os.makedirs(args.cache, exist_ok=True)


# 2. Small helpers

def l1_vector(packed):
    """Cartesian vector a of the L=1 part of a packed alm array."""
    a10 = packed[1].real
    a11 = packed[2]
    return np.array([
        -np.sqrt(3.0 / (2.0 * np.pi)) * a11.real,
        np.sqrt(3.0 / (2.0 * np.pi)) * a11.imag,
        np.sqrt(3.0 / (4.0 * np.pi)) * a10,
    ])


def l1_to_alm(a):
    """Inverse of l1_vector: a Cartesian L=1 vector as a packed alm array."""
    out = np.zeros(_NPACK, dtype=np.complex128)
    out[1] = a[2] * np.sqrt(4.0 * np.pi / 3.0)
    out[2] = (-a[0] + 1j * a[1]) * np.sqrt(2.0 * np.pi / 3.0)
    return out


def pack(alm):
    """Reconstruction alm -> the L <= L_OUT coefficients, as stored."""
    return np.asarray(alm)[_PACK_IDX].astype(np.complex128)


def cl_from_power(power):
    """Per-mode |a|^2 (packed) -> C_L, with m<0 counted."""
    power = np.asarray(power, dtype=float)
    total = np.zeros(L_OUT + 1)
    for k in range(_NPACK):
        total[_PACK_L[k]] += _PACK_W[k] * power[k]
    return total / (2 * _ELLS + 1.0)


def unit_vector(ra, dec):
    """Unit vector from equatorial angles in radians."""
    return np.array([np.cos(dec) * np.cos(ra),
                     np.cos(dec) * np.sin(ra),
                     np.sin(dec)])


def angle_between(u, v):
    """Angle between two vectors, in degrees."""
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def to_radec(vec):
    """(ra, dec) in degrees for a Cartesian vector."""
    v = vec / np.linalg.norm(vec)
    ra = np.degrees(np.arctan2(v[1], v[0]) % (2 * np.pi))
    dec = np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0)))
    return ra, dec


D_TRUE = unit_vector(BDIR[0], BDIR[1])
A_TRUE = -BETA * D_TRUE          # the input aberration potential as a vector
V_TRUE_KMS = BETA * C_KMS * D_TRUE
C1_INPUT = 4.0 * np.pi * BETA ** 2 / 9.0

# Boost directions used to measure the columns of R, one per Cartesian axis.
RESPONSE_DIRS = [(0.0, 0.0),           # x
                 (np.pi / 2, 0.0),     # y
                 (0.0, np.pi / 2)]     # z


# 3. CAMB, filters, noise, normalisation

print("theory spectra from CAMB ...", flush=True)
pars = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122,
                       ns=0.965, As=2.1e-9, tau=0.06)
# lens_potential_accuracy is only wanted for the C_L^phiphi reference curve on
# the leakage plot; it does not feed the sims.
pars.set_for_lmax(MLMAX + 500, lens_potential_accuracy=1)

# Note: unlensed right now, something to check in future
camb_results = camb.get_results(pars)
camb_cls = camb_results.get_cmb_power_spectra(
    pars, raw_cl=True, spectra=["unlensed_scalar"])["unlensed_scalar"]

cltt = camb_cls[:MLMAX + 1, 0].copy()
clee = camb_cls[:MLMAX + 1, 1].copy()
clbb = camb_cls[:MLMAX + 1, 2].copy()    # BB is not used
clte = camb_cls[:MLMAX + 1, 3].copy()

# Real lensing power at the same low L, purely as a yardstick for the leakage.
_pp = camb_results.get_lens_potential_cls(lmax=max(L_OUT, 10))[:, 0]
_l = _ELLS[1:].astype(float)
CLPP = np.zeros(L_OUT + 1)
CLPP[1:] = 2.0 * np.pi * _pp[1:L_OUT + 1] / (_l * (_l + 1.0)) ** 2

# Noise, converted from uK-arcmin into the same dimensionless units.
noise_rms = NOISE_UK_ARCMIN * utils.arcmin / TCMB_UK
nltt = np.full(MLMAX + 1, noise_rms ** 2)
nlee = POL_NOISE_FACTOR ** 2 * nltt
nlbb = nlee.copy()

if BEAM_FWHM_ARCMIN > 0:
    bl = hp.gauss_beam(BEAM_FWHM_ARCMIN * utils.arcmin, lmax=MLMAX)
    nltt = nltt / bl ** 2
    nlee = nlee / bl ** 2
    nlbb = nlbb / bl ** 2

# A mask couples scales, so hold the noise flat above LMAX instead of letting
# the beam deconvolution run away.  This changes the numbers, so keep it.
nltt[LMAX + 1:] = nltt[LMAX]
nlee[LMAX + 1:] = nlee[LMAX]
nlbb[LMAX + 1:] = nlbb[LMAX]

ucls = {"TT": cltt, "EE": clee, "BB": clbb, "TE": clte}          # weights
tcls = {"TT": cltt + nltt, "EE": clee + nlee,                    # filters
        "BB": clbb + nlbb, "TE": clte}

# Signal and noise covariances in the T, E, B basis, for generating the sims.
signal_ps = np.zeros((3, 3, MLMAX + 1))
signal_ps[0, 0] = cltt
signal_ps[1, 1] = clee
signal_ps[2, 2] = clbb
signal_ps[0, 1] = clte
signal_ps[1, 0] = clte

noise_ps = np.zeros((3, 3, MLMAX + 1))
noise_ps[0, 0] = nltt
noise_ps[1, 1] = nlee
noise_ps[2, 2] = nlbb


def inverse_variance_filter(total_cl):
    """1 / (C_l + N_l) inside [LMIN, LMAX], zero outside."""
    f = np.zeros(MLMAX + 1)
    f[LMIN:LMAX + 1] = 1.0 / total_cl[LMIN:LMAX + 1]
    return f


filt_T = inverse_variance_filter(tcls["TT"])
filt_E = inverse_variance_filter(tcls["EE"])
filt_B = inverse_variance_filter(tcls["BB"])

# Analytic full-sky normalisation A_L, one array per estimator.  get_norms
# returns [gradient, curl]; aberration is a pure gradient, so we take index 0.
# The whole low-L array is kept, not just L = 1: the leakage at L >= 2 has to
# be normalised at its own L to mean anything.
print("normalisations from tempura ...", flush=True)
Als = pytempura.get_norms(ESTIMATORS, ucls, ucls, tcls, LMIN, LMAX,
                          k_ellmax=MLMAX)


def usable_norm(a):
    """A_L with L=0 and any unusable entry pushed to infinity.
    """
    a = np.asarray(a, dtype=float).copy()
    a[~np.isfinite(a) | (a <= 0)] = np.inf
    a[0] = np.inf                      # no L = 0 gradient mode
    return a


AL = {}
A1 = {}
for e in ESTIMATORS:
    AL[e] = usable_norm(np.asarray(Als[e][0])[:L_OUT + 1])
    A1[e] = float(AL[e][1])            # the value at L = 1

# Combined normalisation per case, per L.  The unnormalised estimators add
# directly, so the inverse-variance coadd is N_L * sum_e q_e with
# N_L = 1 / sum_e (1 / A_L^e).
CASE_NORM_L = {}
CASE_NORM = {}
NORM_PACK = {}
for case in CASES:
    inv = np.zeros(L_OUT + 1)
    for e in CASES[case]:
        inv = inv + 1.0 / AL[e]        # 1/inf is 0, so unusable L drop out
    norm = np.zeros(L_OUT + 1)
    good = inv > 0
    norm[good] = 1.0 / inv[good]
    CASE_NORM_L[case] = norm
    CASE_NORM[case] = float(norm[1])
    NORM_PACK[case] = norm[_PACK_L]


# 4. Masking and map geometry

shape, wcs = enmap.fullsky_geometry(res=RES)
px = qe.pixelization(shape=shape, wcs=wcs)

# fullsky_geometry defaults to the fejer1 variant, whose quadrature is exact up
# to lmax = ny - 1.  At the default 3 arcmin that is 3599, comfortably above
# mlmax = 3500, but --res and --lmax move independently, so say so rather than
# letting an under-resolved grid quietly alias power into the reconstruction.
_LMAX_GRID = shape[0] - 1
if MLMAX > _LMAX_GRID:
    print(f"WARNING: mlmax {MLMAX} exceeds what a {RES / utils.arcmin:.2f} "
          f"arcmin grid holds ({_LMAX_GRID}).\n"
          f"         Use --res {10800.0 / MLMAX:.2f} or finer, or lower "
          f"--lmax.", flush=True)

# A CAR geometry is separable, so the 1-D axes are enough to build the mask and
# to integrate it exactly.  That avoids three full-sky float64 maps.
_dec_ax, _ra_ax = enmap.posaxes(shape, wcs)
DEC_AX = np.degrees(_dec_ax)
RA_AX = (np.degrees(_ra_ax) + 180.0) % 360.0 - 180.0

_ddec = abs(wcs.wcs.cdelt[1]) * utils.degree
_dra = abs(wcs.wcs.cdelt[0]) * utils.degree
# Exact solid angle of a CAR row.  The clip is what makes the polar rows come
# out right, so it stays.
ROW_AREA = _dra * (np.sin(np.clip(_dec_ax + _ddec / 2, -np.pi / 2, np.pi / 2))
                   - np.sin(np.clip(_dec_ax - _ddec / 2,
                                    -np.pi / 2, np.pi / 2)))


def _w_factor_of(m, n):
    """<W^n> over the sphere, for any mask on this CAR geometry."""
    row_mean = (np.asarray(m, dtype=np.float64) ** n).mean(axis=1)
    return float(np.dot(ROW_AREA, row_mean) / ROW_AREA.sum())


def edge_taper(x, lo, hi, width):
    """1 well inside [lo, hi], 0 outside, raised-cosine ramp of `width`."""
    if width <= 0:
        return ((x > lo) & (x < hi)).astype(float)
    rise = np.clip((x - lo) / width, 0.0, 1.0)
    fall = np.clip((hi - x) / width, 0.0, 1.0)
    return 0.25 * (1 - np.cos(np.pi * rise)) * (1 - np.cos(np.pi * fall))


def act_parts(halfwidth):
    """The separable (dec, ra) factors of the ACT-like patch."""
    ra_rel = (RA_AX - args.ra_center + 180.0) % 360.0 - 180.0
    taper_dec = edge_taper(DEC_AX, ACT_DEC[0], ACT_DEC[1], APOD)
    taper_ra = edge_taper(ra_rel, -halfwidth, halfwidth, APOD)
    return taper_dec, taper_ra


def act_fsky(halfwidth):
    """w1 of the patch, exactly, using separability."""
    taper_dec, taper_ra = act_parts(halfwidth)
    dec_part = np.dot(ROW_AREA, taper_dec) / ROW_AREA.sum()
    return float(dec_part) * float(taper_ra.mean())


def solve_ra_halfwidth(target):
    """RA half-width that makes w1 equal `target`, apodisation included."""
    lo = 1e-4
    hi = 180.0 - max(APOD, 0.0) - 0.5      # stop the two ramps from meeting
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if act_fsky(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


RA_HALFWIDTH = solve_ra_halfwidth(args.fsky) if args.mask == "act" else float("nan")


# The released ACT DR6 mask.  Binning nside 4096 onto the CAR grid costs a
# ~0.8 GB read and a minute or so, so the result is cached next to the sims.
# DR6_W_NATIVE holds w1, w2, w4 of the healpix mask itself, where the pixels
# are equal area and the plain mean of mask**n is exact.
DR6_W_NATIVE = None


def dr6_cache_paths():
    """Where the binned mask and its native w factors are kept."""
    d = os.path.join(args.cache, "masks")
    os.makedirs(d, exist_ok=True)
    key = (f"{os.path.basename(DR6_PATH)}_{os.path.getsize(DR6_PATH)}"
           f"_res{RES:.12g}_sm{args.dr6_smooth_arcmin:g}"
           f"_{args.dr6_method}_o{args.dr6_order}_rot{args.dr6_rot}")
    name = (f"dr6_{args.dr6_variant}_{args.dr6_method}_"
            + hashlib.md5(key.encode()).hexdigest()[:10])
    base = os.path.join(d, name)
    return base + ".fits", base + "_w.npz"


def project_average(m, nside, chunk=1 << 23):
    """Healpix -> CAR by averaging every healpix centre landing in each CAR
    pixel.  Healpix pixels are equal area, so this is an area average."""
    ny, nx = shape
    num = np.zeros(ny * nx)
    den = np.zeros(ny * nx, dtype=np.int64)
    for lo in range(0, m.size, chunk):
        ipix = np.arange(lo, min(lo + chunk, m.size))
        theta, phi = hp.pix2ang(nside, ipix)
        y, x = enmap.sky2pix(shape, wcs, [np.pi / 2 - theta, phi])
        flat = ((np.round(y).astype(np.int64) % ny) * nx
                + (np.round(x).astype(np.int64) % nx))
        num += np.bincount(flat, weights=m[ipix].astype(float),
                           minlength=ny * nx)
        den += np.bincount(flat, minlength=ny * nx)
    out = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    return enmap.enmap(out.reshape(ny, nx), wcs), den.reshape(ny, nx)


def healpix_to_car(m, nside):
    """Healpix mask -> an enmap on this CAR geometry.

    pixell's reproject.healpix2map does the work.  The default, and what the
    released masks want, is method="spline" with order=1: it reads the input
    map at each CAR pixel centre by bilinear interpolation, so the result is a
    convex combination of nearby healpix values and therefore cannot leave
    [0, 1].  method="harm" goes through spherical harmonics instead, which
    preserves the power spectrum but rings around the sharp edges of a mask and
    can push an all-positive input negative; it is here to be compared against,
    not to be used.  method="average" is the old area-average binning, exact in
    the mean but only defined where a healpix centre happens to land.

    spin=[0] because this is one scalar field, not a T, Q, U triple, and
    extensive=False because a mask is intensive: its value does not scale with
    pixel size the way a hitcount does.
    """
    if args.dr6_method == "average":
        out, den = project_average(m, nside)
        band = np.abs(DEC_AX) < 70.0
        note = (f"{den[band].mean():.1f} healpix centres per CAR pixel within "
                f"|dec| < 70, {100.0 * (den == 0).mean():.2f}% of pixels empty")
        # enmap, not a bare array: the caller writes this out with
        # enmap.write_map, which needs the wcs to still be attached.
        return enmap.enmap(np.asarray(out, dtype=np.float64), wcs), note

    if args.dr6_method == "spline":
        kw = dict(method="spline", order=args.dr6_order)
        note = "bilinear" if args.dr6_order == 1 else "nearest neighbour"
    else:
        # No point asking for more multipoles than the output grid can hold.
        kw = dict(method="harm", lmax=_LMAX_GRID)
        note = f"harmonic, lmax {_LMAX_GRID}"

    out = reproject.healpix2map(m, shape, wcs, spin=[0], rot=args.dr6_rot,
                                extensive=False, **kw)
    out = np.asarray(out, dtype=np.float64)
    # Says whether the interpolation stayed inside the range of the input.  For
    # spline it has to; for harm it will not, and the size of the excursion is
    # the ringing.  Report it before clipping it away.
    lo, hi = float(out.min()), float(out.max())
    frac = float(np.mean((out < -1e-6) | (out > 1.0 + 1e-6)))
    note += f", range [{lo:+.4f}, {hi:+.4f}], {100.0 * frac:.3f}% outside [0,1]"
    return enmap.enmap(np.clip(out, 0.0, 1.0), wcs), note


def load_dr6_mask():
    """The released DR6 lensing mask, reprojected onto this CAR geometry."""
    global DR6_W_NATIVE, DR6_NSIDE
    car_path, w_path = dr6_cache_paths()
    if os.path.exists(car_path):
        print(f"DR6 mask: reusing {car_path}", flush=True)
        cached = np.load(w_path)
        DR6_W_NATIVE = cached["w"]
        DR6_NSIDE = int(cached["nside"])
        return enmap.read_map(car_path)

    print(f"DR6 mask: reading {DR6_PATH}", flush=True)
    t0 = time.time()
    # nest=False makes healpy reorder a NESTED file for us, and everything
    # downstream assumes RING: get_interp_val inside reproject's spline path,
    # map2alm_healpix inside its harmonic one, and pix2ang in project_average.
    m = hp.read_map(DR6_PATH, field=0, dtype=np.float32)
    # The released file has UNSEEN pixels, and the mask gets squared and
    # fourth-powered later, so this clip is part of the definition.
    m = np.clip(np.nan_to_num(m), 0.0, 1.0)
    DR6_NSIDE = hp.npix2nside(m.size)
    print(f"   nside {DR6_NSIDE}, {m.size} pixels, {time.time() - t0:.1f} s",
          flush=True)

    if args.dr6_smooth_arcmin > 0:
        print(f"   smoothing by {args.dr6_smooth_arcmin:g} arcmin FWHM "
              f"(an SHT at nside {DR6_NSIDE}, be patient)", flush=True)
        smoothed = hp.smoothing(m.astype(np.float64),
                                fwhm=args.dr6_smooth_arcmin * utils.arcmin)
        m = np.clip(smoothed, 0.0, 1.0).astype(np.float32)

    # Exact, and the reference the CAR mask is judged against.
    DR6_W_NATIVE = np.array([float(np.mean(m.astype(np.float64) ** n))
                             for n in (1, 2, 4)])

    t0 = time.time()
    out, note = healpix_to_car(m, DR6_NSIDE)
    del m
    print(f"   {args.dr6_method} -> {shape[0]} x {shape[1]} CAR in "
          f"{time.time() - t0:.1f} s", flush=True)
    print(f"   {note}", flush=True)
    # The healpix w factors above are exact; these are what the CAR mask
    # actually delivers.  A per cent or so of disagreement in w4 is the
    # interpolation smoothing the apodised edge, not a bug, but a large gap
    # means the grid is too coarse for the features in the mask.
    car_w = np.array([_w_factor_of(out, n) for n in (1, 2, 4)])
    print("   w1, w2, w4:  healpix "
          + " ".join(f"{x:.4f}" for x in DR6_W_NATIVE)
          + "   CAR " + " ".join(f"{x:.4f}" for x in car_w)
          + "   ratio " + " ".join(f"{x:.4f}" for x in car_w / DR6_W_NATIVE),
          flush=True)

    # Temp-then-rename: parallel shards all build this, and a half-written
    # FITS read by another shard is silent corruption.
    tmp = f"{car_path}.tmp{os.getpid()}.fits"
    enmap.write_map(tmp, out)
    os.replace(tmp, car_path)
    np.savez(w_path, w=DR6_W_NATIVE, nside=DR6_NSIDE)
    print(f"   cached as {car_path}", flush=True)
    return out


def build_mask():
    """1 where the sky is kept, 0 where it is cut."""
    ones = np.ones(shape[1])
    if args.mask == "dr6":
        # Already an enmap on this geometry.  float64 because the mask gets
        # squared and fourth-powered, and read_map hands back whatever dtype
        # the cached FITS was written with.
        return enmap.enmap(np.asarray(load_dr6_mask(), dtype=np.float64), wcs)
    if args.mask == "none":
        keep = np.outer(np.ones(shape[0]), ones)
    elif args.mask == "act":
        taper_dec, taper_ra = act_parts(RA_HALFWIDTH)
        keep = np.outer(taper_dec, taper_ra)
    elif args.mask == "box":
        keep = 1.0 - np.outer(
            edge_taper(DEC_AX, BOX_DEC_RANGE[0], BOX_DEC_RANGE[1], APOD),
            edge_taper(RA_AX, BOX_RA_RANGE[0], BOX_RA_RANGE[1], APOD))
    elif args.mask == "dec_band":
        keep = 1.0 - np.outer(
            edge_taper(DEC_AX, -DEC_BAND_DEG, DEC_BAND_DEG, APOD), ones)
    return enmap.enmap(keep, wcs)


mask = build_mask()


def w_factor(n):
    """<W^n> over the sphere."""
    return _w_factor_of(mask, n)


w1 = w_factor(1)
w2 = w_factor(2)
w4 = w_factor(4)


def mask_thumbnail():
    """A small block-averaged copy of the mask, for the coverage figure.

    Returned as (weights, dec, ra) with dec and ra in degrees and both axes
    ascending, so whatever plots it needs no geometry code and no pixell.
    RA is sampled at block centres rather than averaged: RA_AX wraps at +-180
    somewhere in the middle of the row.
    """
    f = max(1, int(round(args.mask_thumb_deg * utils.degree / RES)))
    m = np.asarray(mask, dtype=np.float64)
    if f == 1:
        thumb, dec, ra = m, DEC_AX, RA_AX
    else:
        ny = (m.shape[0] // f) * f
        nx = (m.shape[1] // f) * f
        # Block average: split each axis into (nblock, f) and mean over f.
        blocks = m[:ny, :nx].reshape(ny // f, f, nx // f, f)
        thumb = blocks.mean(axis=(1, 3))
        dec = DEC_AX[:ny][f // 2::f][:ny // f]
        ra = RA_AX[:nx][f // 2::f][:nx // f]
    iy = np.argsort(dec)
    ix = np.argsort(ra)
    thumb = thumb[iy, :][:, ix]
    return (np.ascontiguousarray(thumb, dtype=np.float32),
            np.asarray(dec, np.float32)[iy],
            np.asarray(ra, np.float32)[ix])


def mask_at(ra, dec):
    """Mask value at one equatorial position, radians."""
    pos = np.array([[dec], [ra]])
    return float(np.asarray(mask.at(pos, order=1)).ravel()[0])


# The cache stores reconstructions, which depend on every setting above.
# Keying the directory on those settings is what stops a run with a different
# mask, lmax or noise level from silently reloading the wrong sims.  "v2" marks
# the switch from a stored 3-vector to stored alm.
_config = ("v2", L_OUT, LMIN, LMAX, MLMAX, round(RES, 12), NOISE_UK_ARCMIN,
           round(float(POL_NOISE_FACTOR), 12), BEAM_FWHM_ARCMIN,
           MASK_KEY, round(float(BETA), 12),
           tuple(round(float(x), 12) for x in np.asarray(BDIR).ravel()))
if args.seed_offset:
    _config = _config + ("seed", args.seed_offset)

# MASK_KEY names the analytic masks completely, but the dr6 mask comes out of a
# file, so hash the mask itself: a re-downloaded or edited file, or a different
# variant left under the same name, then cannot silently reuse these sims.
MASK_DIGEST = hashlib.md5(
    np.ascontiguousarray(np.asarray(mask, dtype=np.float64))).hexdigest()[:8]
if args.mask == "dr6":
    _config = _config + ("dr6", MASK_DIGEST)

_seed_tag = f"_seed{args.seed_offset}" if args.seed_offset else ""
CACHE_DIR = os.path.join(
    args.cache,
    f"{MASK_KEY}_lmax{LMAX}{_seed_tag}_"
    + hashlib.md5(repr(_config).encode()).hexdigest()[:8])
os.makedirs(CACHE_DIR, exist_ok=True)
PLOT_DIR = args.plot_dir or os.path.join(CACHE_DIR, "plots")


# 5. Simulation and reconstruction

def seeds(stage, i):
    """Non-overlapping (signal, noise) seeds."""
    base = {"meanfield": 1_000_000,
            "response": 2_000_000,
            "data": 3_000_000}[stage]
    base += 10_000_000 * args.seed_offset
    return base + 2 * i, base + 2 * i + 1


def cmb_map(seed):
    """Unaberrated T, Q, U realisation of the theory spectra."""
    return curvedsky.rand_map((3,) + shape, wcs, signal_ps, lmax=MLMAX,
                              seed=seed)


def noise_map(seed):
    """Instrument noise.  Added after aberration: noise is not boosted."""
    return curvedsky.rand_map((3,) + shape, wcs, noise_ps, lmax=MLMAX,
                              seed=seed)


def reconstruct(tqu):
    """Sky map -> {estimator: packed alm of the reconstruction, L <= L_OUT}.

    The mask is applied here, so both legs of the quadratic estimator carry it.
    """
    alm = curvedsky.map2alm(mask * tqu, lmax=MLMAX)          # T, E, B
    fT = qe.filter_alms(alm[0], filt_T, lmin=LMIN, lmax=LMAX)
    fE = qe.filter_alms(alm[1], filt_E, lmin=LMIN, lmax=LMAX)
    fB = qe.filter_alms(alm[2], filt_B, lmin=LMIN, lmax=LMAX)
    recon = qe.qe_all(px, ucls, MLMAX, fTalm=fT, fEalm=fE, fBalm=fB,
                      estimators=ESTIMATORS)
    return {e: pack(recon[e][0]) for e in ESTIMATORS}


def coadd_alm(vecs, case):
    """Normalised reconstruction alm for one case, all L <= L_OUT."""
    total = np.zeros(_NPACK, dtype=np.complex128)
    for e in CASES[case]:
        total = total + vecs[e]
    return NORM_PACK[case] * total


def coadd(vecs, case):
    """Normalised L=1 vector of the reconstructed phi, for one case."""
    return l1_vector(coadd_alm(vecs, case))


# Each sim stores a few kB (the alm out to L_OUT), so a finished stage reloads
# instantly, interrupted runs resume, and shards can run in parallel.

def cache_has(tag):
    """True if the sim is already done.  Used to build the to-do lists, where
    opening every npz just to find out costs real time on a shared filesystem.
    """
    return os.path.exists(os.path.join(CACHE_DIR, tag + ".npz"))


def cache_get(tag):
    path = os.path.join(CACHE_DIR, tag + ".npz")
    if not os.path.exists(path):
        return None
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def cache_put(tag, vecs):
    path = os.path.join(CACHE_DIR, tag + ".npz")
    # The temp name must end in .npz, or np.savez appends the suffix itself and
    # the rename looks for a file that does not exist.  Renaming keeps parallel
    # shards from reading each other's half-written files.
    tmp = f"{path}.tmp{os.getpid()}.npz"
    np.savez(tmp, **vecs)
    os.replace(tmp, path)


def mine(i):
    """True if simulation i belongs to this shard."""
    return i % args.nshards == args.shard


def collect(tags):
    """Cached results for a list of tags, skipping any that are missing."""
    out = []
    for tag in tags:
        v = cache_get(tag)
        if v is not None:
            out.append(v)
    return out


class Progress:
    """Per-sim timing with a running estimate of what is left.

    With --nproc > 1 the rate is wall clock per finished sim, that is the
    throughput of the whole pool, not the cost of one reconstruction.
    """

    def __init__(self, total, label):
        self.total = total
        self.label = label
        self.done = 0
        self.t0 = time.time()

    def tick(self, i):
        self.done += 1
        rate = (time.time() - self.t0) / self.done
        left = rate * (self.total - self.done)
        print(f"    {self.label} {i:6d}   {rate:6.1f} s/sim   "
              f"{self.done}/{self.total}   {left / 60:6.1f} min left",
              flush=True)


# 5b. Worker pool
#
# Sharding splits the sims across processes that each redo the whole setup and
# each need their own place in the queue.  This splits them across processes
# inside one allocation instead: ask for one node, get one queue slot, keep
# every core busy.  The two compose, so a job array of N shards x --nproc still
# works, and --nshards 1 --nproc N is the common case.
#
# Fork is what makes the workers cheap.  CAMB, tempura, the mask and the
# geometry are all built once, in the parent, and inherited copy-on-write, so a
# worker starts in milliseconds and adds no setup time.  What is *not* shared is
# the maps a worker allocates while it reconstructs, so peak memory grows
# roughly linearly with --nproc.  That is the trade: cores for RAM.
#
# Each sim is keyed by its own seed and written to its own cache file, so how
# the work is split changes nothing about the result.  --nproc is a wall-clock
# knob only, and a run interrupted halfway still resumes sim by sim.

_EXEC = None


def executor():
    """The pool, created on first use so a fully cached run never forks."""
    global _EXEC
    if _EXEC is None:
        _EXEC = ProcessPoolExecutor(max_workers=NPROC,
                                    mp_context=get_context("fork"))
    return _EXEC


def close_pool():
    """Shut the workers down and hand their memory back before the analysis."""
    global _EXEC
    if _EXEC is not None:
        _EXEC.shutdown(wait=True)
        _EXEC = None
    _ABER.clear()          # the --nproc 1 path keeps one here in the parent


def run_tasks(todo, work, label):
    """Run work(item) for every item in todo, in parallel when --nproc > 1.

    `work` has to be a module-level function, since the items are pickled and
    sent to the workers; it returns the sim index, only so that the progress
    line can say which one came back.  --nproc 1 runs the same calls in this
    process, with no pool involved at all.
    """
    if not todo:
        return
    prog = Progress(len(todo), label)
    if NPROC == 1:
        for item in todo:
            prog.tick(work(item))
        return
    futures = [executor().submit(work, item) for item in todo]
    try:
        for f in as_completed(futures):
            prog.tick(f.result())
    except BrokenProcessPool:
        raise SystemExit(
            f"\nA worker died.  On a cluster that is almost always the job "
            f"hitting its memory limit:\neach of the {NPROC} workers holds its "
            f"own maps.  Lower --nproc or ask for more memory.\nFinished sims "
            f"are cached, so rerunning picks up where this left off.")


# An Aberrator costs about as much memory as a map and is not free to build, so
# each worker keeps the one it is using and drops it before building the next.
# The stage loops go leg by leg, which makes that one build per worker per leg,
# exactly as many as the serial version did per process.
_ABER = {}


def aberrator_for(key, direction, beta):
    """The Aberrator for one boost, built at most once per worker per leg."""
    if _ABER.get("key") != key:
        _ABER.clear()                      # free the old one before allocating
        _ABER["key"] = key
        _ABER["obj"] = aberration.Aberrator(shape, wcs,
                                            dir=np.array(direction), beta=beta)
    return _ABER["obj"]


# 6. Mean field
# With a mask the estimator has a nonzero expectation even with no aberration.
# We measure it from unaberrated sims and subtract.

def _task_meanfield(i):
    s_sig, s_noi = seeds("meanfield", i)
    cache_put(f"mf_{i:05d}", reconstruct(cmb_map(s_sig) + noise_map(s_noi)))
    return i


def run_meanfield():
    print(f"\n[A] mean field: {N_MEANFIELD} unaberrated sims", flush=True)
    todo = [i for i in range(N_MEANFIELD)
            if mine(i) and not cache_has(f"mf_{i:05d}")]

    run_tasks(todo, _task_meanfield, "sim")

    return collect([f"mf_{i:05d}" for i in range(N_MEANFIELD)])


# 7. Response matrix
# Both legs of each pair use the same CMB realisations.

def response_leg(s_sig, s_noi, aberrator=None):
    """One reconstruction for the response stage.

    Noise is optional.  It cancels realisation by realisation between the legs
    of a pair, so including it leaves the response unbiased but adds variance;
    the noise *level* still enters through the filters and the normalisation.
    """
    tqu = cmb_map(s_sig)
    if aberrator is not None:
        tqu = aberrator(tqu)
    if RESPONSE_NOISE:
        tqu = tqu + noise_map(s_noi)
    return reconstruct(tqu)


def _task_response_off(i):
    s_sig, s_noi = seeds("response", i)
    cache_put(f"resp_{RESP_SUFFIX}_off_{i:05d}", response_leg(s_sig, s_noi))
    return i


def _task_response(item):
    """One boosted leg of one response sim.  item is (axis, sign, sim)."""
    j, sgn, i = item
    label = f"{j}p" if sgn > 0 else f"{j}m"
    # dir is kept fixed and the sign carried by beta: a negative beta is
    # exactly the deboost, so this is the -e_j leg.
    aberrator = aberrator_for((j, sgn), RESPONSE_DIRS[j], sgn * RESPONSE_BETA)
    s_sig, s_noi = seeds("response", i)
    cache_put(f"resp_{RESP_SUFFIX}_{label}_{i:05d}",
              response_leg(s_sig, s_noi, aberrator))
    return i


def resp_pair(i, j):
    """Difference for sim i, axis j, scaled so that <D> = -beta_resp R e_j.

    None if either leg is missing, so shards and interrupted runs work.
    """
    plus = cache_get(f"resp_{RESP_SUFFIX}_{j}p_{i:05d}")
    if plus is None:
        return None
    if RESPONSE_CENTRAL:
        other = cache_get(f"resp_{RESP_SUFFIX}_{j}m_{i:05d}")
        denom = 2.0
    else:
        other = cache_get(f"resp_{RESP_SUFFIX}_off_{i:05d}")
        denom = 1.0
    if other is None:
        return None
    return {e: (plus[e] - other[e]) / denom for e in ESTIMATORS}


def run_response():
    kind = "centred" if RESPONSE_CENTRAL else "one-sided"
    noise_word = "with" if RESPONSE_NOISE else "without"
    print(f"\n[B] response: {N_RESPONSE} sims x 3 axes, {kind} difference, "
          f"beta {RESPONSE_BETA:.4e}, {noise_word} noise", flush=True)

    # The unboosted leg is only needed for the one-sided difference; the
    # centred difference cancels the mean field between its own two legs.
    if not RESPONSE_CENTRAL:
        todo = [i for i in range(N_RESPONSE) if mine(i)
                and not cache_has(f"resp_{RESP_SUFFIX}_off_{i:05d}")]
        run_tasks(todo, _task_response_off, "unboosted")

    signs = [+1, -1] if RESPONSE_CENTRAL else [+1]

    # One leg at a time, sims within a leg in parallel: every worker then builds
    # the leg's Aberrator once and reuses it for all the sims it is given.
    for j in range(3):
        for sgn in signs:
            label = f"{j}p" if sgn > 0 else f"{j}m"
            todo = [(j, sgn, i) for i in range(N_RESPONSE) if mine(i)
                    and not cache_has(f"resp_{RESP_SUFFIX}_{label}_{i:05d}")]
            run_tasks(todo, _task_response, f"axis {'xyz'[j]}{label[-1]}")

    # One dict per axis, keyed by sim index, so the three axes can be matched
    # up sim by sim for the leakage even when a shard is only partly done.
    out = []
    for j in range(3):
        d = {}
        for i in range(N_RESPONSE):
            pair = resp_pair(i, j)
            if pair is not None:
                d[i] = pair
        out.append(d)
    return out


def response_alm(diffs, case):
    """Per-sim (L,M) response to a boost along x, y, z.

    Returns S with shape (n_sims, npack, 3).  Column j is the reconstruction
    alm produced by a boost of beta_resp along e_j, divided by -beta_resp, so
    that it is the response to a unit boost and carries no beta scaling of its
    own.  The mean field cancels inside each pair before this is called.

    The three columns are the same three boosts the response matrix is built
    from, so the L=1 rows of S are the 3x3 matrix R written in the alm basis:
    l1_vector(S[i, :, j]) is exactly column j of R for sim i.  Everything at
    L >= 2 is the leakage the same boost leaves behind, reported as the alm
    themselves rather than collapsed into a power spectrum.

    Only sims present for all three axes are used, so that a column is never
    averaged over a different set of realisations from its neighbours.
    """
    common = sorted(set(diffs[0]) & set(diffs[1]) & set(diffs[2]))
    S = np.empty((len(common), _NPACK, 3), dtype=np.complex128)
    for a, i in enumerate(common):
        for j in range(3):
            S[a, :, j] = coadd_alm(diffs[j][i], case) / (-RESPONSE_BETA)
    return S


def alm_response(S):
    """Mean (L,M) response, its per-sim scatter and the error on the mean."""
    n = len(S)
    mean = S.mean(axis=0)
    sd_re = S.real.std(axis=0, ddof=1)
    sd_im = S.imag.std(axis=0, ddof=1)
    return dict(R=mean, sd_re=sd_re, sd_im=sd_im,
                err_re=sd_re / np.sqrt(n), err_im=sd_im / np.sqrt(n), n=n)


def response_matrix(S):
    """The 3x3 response, its per-sim scatter, and the error on its mean.

    Read off the L=1 rows of the same S the alm response uses, so the matrix
    and the (L,M) table can never drift apart.
    """
    n = len(S)
    V = np.empty((n, 3, 3))
    for a in range(n):
        for j in range(3):
            V[a, :, j] = l1_vector(S[a, :, j])
    R = V.mean(axis=0)
    sd = V.std(axis=0, ddof=1)
    return R, sd, sd / np.sqrt(n)


# 7b. Leakage into L >= 2
# Contract the three measured columns with the true input vector and you have
# the coherent field the real boost puts on the sky.  This is a derived view of
# the same S: the alm response above is the primary result, and this is the
# single number per L that follows from it.

def leakage_from_alm(S):
    """Coherent C_L of the boost-induced signal, MC-noise debiased."""
    P = S @ A_TRUE                              # (n, npack)
    n = len(P)
    mean = P.mean(axis=0)
    var = (P.real.var(axis=0, ddof=1) + P.imag.var(axis=0, ddof=1)) / n
    # <|mean|^2> = |truth|^2 + var/n, so subtract it rather than reporting a
    # leakage that is really just Monte Carlo noise.
    cl = cl_from_power(np.abs(mean) ** 2 - var)

    # Jackknife error bar.  Needs n >= 4 to mean anything.
    tot = P.sum(axis=0)
    jk = np.empty((n, L_OUT + 1))
    for k in range(n):
        rest = np.delete(P, k, axis=0)
        v_k = (rest.real.var(axis=0, ddof=1)
               + rest.imag.var(axis=0, ddof=1)) / (n - 1)
        jk[k] = cl_from_power(np.abs((tot - P[k]) / (n - 1)) ** 2 - v_k)
    err = np.sqrt((n - 1) / n * ((jk - jk.mean(axis=0)) ** 2).sum(axis=0))
    return dict(cl=cl, err=err, n=n, alm=mean)


def spectra_from_sims(mf_vecs, data_vecs, case):
    """Noise floor, mean-field power, and the data-sim view of the leakage."""
    M = np.array([coadd_alm(v, case) for v in mf_vecs])
    D = np.array([coadd_alm(v, case) for v in data_vecs])
    n_m = len(M)
    n_d = len(D)
    mbar = M.mean(axis=0)
    var_m = M.real.var(axis=0, ddof=1) + M.imag.var(axis=0, ddof=1)
    var_d = D.real.var(axis=0, ddof=1) + D.imag.var(axis=0, ddof=1)

    noise = cl_from_power(var_m)                        # per-realisation N_L
    mfcl = cl_from_power(np.abs(mbar) ** 2 - var_m / n_m)
    mean = D.mean(axis=0) - mbar
    datleak = cl_from_power(np.abs(mean) ** 2 - var_d / n_d - var_m / n_m)
    return noise, mfcl, datleak


# 8. Aberrated data simulations

def _task_data(i):
    aberrator = aberrator_for("data", BDIR, BETA)
    s_sig, s_noi = seeds("data", i)
    boosted = aberrator(cmb_map(s_sig))
    cache_put(f"dat_{i:05d}", reconstruct(boosted + noise_map(s_noi)))
    del boosted
    return i


def run_data():
    print(f"\n[C] data: {N_DATA} aberrated sims", flush=True)
    todo = [i for i in range(N_DATA)
            if mine(i) and not cache_has(f"dat_{i:05d}")]

    run_tasks(todo, _task_data, "sim")

    return collect([f"dat_{i:05d}" for i in range(N_DATA)])


def fit(v, mf, Rinv):
    """One reconstructed vector -> (amplitude, direction)."""
    a_hat = Rinv @ (v - mf)
    amp = np.dot(a_hat, A_TRUE) / np.dot(A_TRUE, A_TRUE)
    direction = -a_hat / np.linalg.norm(a_hat)
    return amp, direction


# 9. Analysis and reporting

def per_estimator_meanfield(mf_vecs):
    """Mean field of each estimator separately, in units of the input signal."""
    n = len(mf_vecs)
    scale = np.linalg.norm(A_TRUE)
    print(f"\nmean field per estimator, in units of the input dipole "
          f"(from {n} sims)")
    print(f"   {'':4s}      {'x':>9s} {'y':>9s} {'z':>9s}      |mf|")
    for e in ESTIMATORS:
        q = np.array([l1_vector(d[e]) for d in mf_vecs])
        v = A1[e] * q.mean(axis=0) / scale
        err = A1[e] * q.std(axis=0, ddof=1).max() / np.sqrt(n) / scale
        print(f"   {e:4s}  "
              + " ".join(f"{x:+9.3f}" for x in v)
              + f"   {np.linalg.norm(v):8.3f}   +-{err:.3f} on the mean, "
              f"per component")
    for case in CASES:
        if len(CASES[case]) < 2:
            continue                  # already printed above, as an estimator
        v = np.mean([coadd(d, case) for d in mf_vecs], axis=0) / scale
        print(f"   {case:12s}"
              + " ".join(f"{x:+9.3f}" for x in v)
              + f"   {np.linalg.norm(v):8.3f}")


def report_alm_response(ar, noise):
    """The (L,M) response to a unit boost along x, y, z, printed directly.

    One block per L.  Each entry is <a_LM> / (-beta) for a boost along that
    axis, with the Monte Carlo error on the mean beside it.  The L=1 block is
    the response matrix in the alm basis; every block below it is leakage.
    """
    R, er, ei = ar["R"], ar["err_re"], ar["err_im"]
    print(f"\n(L,M) response to a unit boost along x, y, z, from {ar['n']} "
          f"paired sims per axis")
    print(f"   entries are <a_LM>/(-beta); the L=1 block is R in the alm "
          f"basis, L>=2 is leakage")
    # Column blocks are 21 characters wide, built the same way in every line
    # so the axis labels, the Re/Im pairs and the per-L summary all line up.
    axline = " " * 6 + "".join(f"  {a:^19s}" for a in "xyz")
    colline = ("     M" + "".join(f"  {'Re':>9s} {'Im':>9s}" for _ in range(3))
               + "    MC err")
    for l in range(1, L_OUT + 1):
        print(f"\n   L = {l}")
        print(axline)
        print(colline)
        for k in range(_NPACK):
            if _PACK_L[k] != l:
                continue
            cells = "".join(f"  {R[k, j].real:+9.5f} {R[k, j].imag:+9.5f}"
                            for j in range(3))
            e = max(er[k].max(), ei[k].max())
            print(f"     {_PACK_M[k]:d}{cells}   {e:.1e}")
        # sqrt(C_L) of each column: the rms response this L carries per unit
        # boost, and the same quantity the figure draws as bars.  A value at
        # the floor is consistent with this L carrying no response at all.
        rms = [np.sqrt(max(cl_from_power(np.abs(R[:, j]) ** 2)[l], 0.0))
               for j in range(3)]
        flo = [np.sqrt(max(cl_from_power(er[:, j] ** 2 + ei[:, j] ** 2)[l],
                           0.0)) for j in range(3)]
        print("   rms" + "".join(f"  {r:^19.3e}" for r in rms))
        print(" floor" + "".join(f"  {f:^19.3e}" for f in flo))
    print(f"\n   reconstruction noise N_L for reference:  "
          + "  ".join(f"L={l}: {noise[l]:.2e}" for l in range(1, L_OUT + 1)))


def report_leakage(leak, noise, mfcl, c1_recon):
    """The leakage that follows from the response, contracted with the truth."""
    cl = leak["cl"]
    err = leak["err"]
    print(f"\nleakage of the L=1 boost into higher L, from {leak['n']} paired "
          f"response sims")
    print(f"   input dipole C_1 = {C1_INPUT:.3e}, recovered C_1 = "
          f"{c1_recon:.3e}  ({c1_recon / C1_INPUT:.3f} of input)")
    print(f"     L      C_L^leak      /C_1^rec     MC err      N_L^recon"
          f"     leak/N_L")
    for l in range(1, L_OUT + 1):
        print(f"   {l:3d}   {cl[l]:+.4e}    {cl[l] / c1_recon:9.5f}   "
              f"{err[l]:.2e}   {noise[l]:.3e}   {cl[l] / noise[l]:9.2e}")
    tail = np.clip(cl[2:], 0.0, None)
    ratio = (tail * (2 * _ELLS[2:] + 1)).sum() / (c1_recon * 3.0)
    print(f"   total power at L>=2, relative to the recovered L=1 dipole: "
          f"{ratio:.4f}")
    if cl[2] != 0:
        print(f"   mean field power at L=2 is {mfcl[2] / cl[2]:.1f}x the "
              f"leakage there, which is why this is measured from the paired\n"
              f"   differences and not from the aberrated sims")


def analyse(case, mf_vecs, diffs, data_vecs):
    print(f"\n{'=' * 72}\n {case}\n{'=' * 72}")

    S = response_alm(diffs, case)
    ar = alm_response(S)
    R, Rsd, dR = response_matrix(S)
    Rinv = np.linalg.inv(R)
    scale = np.max(np.abs(R))
    print(f"response matrix R, from {ar['n']} paired sims per axis")
    for i in range(3):
        print("     " + "   ".join(f"{R[i, j]:+9.5f} +- {dR[i, j]:7.5f}"
                                   for j in range(3)))
    print(f"   Monte Carlo error   {np.max(dR) / scale * 100:.3f}% of the "
          f"largest element")
    print(f"   per-sim scatter     {np.max(Rsd) / scale * 100:.1f}% of it "
          f"(sd, not the error on the mean)")
    print(f"   |R - I|_max         {np.max(np.abs(R - np.eye(3))):.4f}")
    print(f"   condition number    {np.linalg.cond(R):.2f}")

    # Which sky directions the footprint actually constrains, largest first.
    evals, evecs = np.linalg.eigh(0.5 * (R + R.T))
    for k in range(2, -1, -1):
        ra_e, dec_e = to_radec(evecs[:, k])
        print(f"   eigenvalue {evals[k]:+8.5f}  along ra {ra_e:6.1f}, "
              f"dec {dec_e:+6.1f}")

    mf_all = np.array([coadd(v, case) for v in mf_vecs])
    mf = mf_all.mean(axis=0)
    n_mf = len(mf_all)

    dat = np.array([coadd(v, case) for v in data_vecs])
    amp_list = []
    dir_list = []
    a_hat_list = []
    for v in dat:
        amp, direction = fit(v, mf, Rinv)
        amp_list.append(amp)
        dir_list.append(direction)
        a_hat_list.append(Rinv @ (v - mf))
    amps = np.array(amp_list)
    dirs = np.array(dir_list)
    a_hats = np.array(a_hat_list)

    vel = -a_hats * C_KMS                       # velocity in km/s
    velraw = -(dat - mf) * C_KMS                # the same without R^-1
    n_dat = len(amps)

    sigma = amps.std(ddof=1)
    err_mean = sigma * np.sqrt(1.0 / n_dat + 1.0 / n_mf)   # includes MF noise

    # Forecast from the analytic normalisation, for a full sky with no mean
    # field: sigma(A) = sqrt(3 A_1 / 4pi) / beta.
    sigma_pred = np.sqrt(3 * CASE_NORM[case] / (4 * np.pi)) / BETA

    # Amplitude with no response correction, to show the mask suppression.
    naive = (dat - mf) @ A_TRUE / np.dot(A_TRUE, A_TRUE)

    ang = np.array([angle_between(d, D_TRUE) for d in dirs])
    mean_dir = -a_hats.mean(axis=0)
    mean_dir = mean_dir / np.linalg.norm(mean_dir)
    ra_m, dec_m = to_radec(mean_dir)

    print(f"\nmean field / expected signal at L=1  = "
          f"{np.linalg.norm(mf) / np.linalg.norm(R @ A_TRUE):.2f}")
    print(f"\namplitude, from {n_dat} aberrated sims and a "
          f"{n_mf}-sim mean field")
    print(f"   mean A               = {amps.mean():+.4f} +- {sigma:.4f} "
          f"(sd)    (expect 1)")
    print(f"                          {amps.mean():+.4f} +- {err_mean:.4f} "
          f"(error on the mean, mean field included)")
    print(f"   scatter sigma(A)     = {sigma:.4f}"
          f"       (full-sky forecast {sigma_pred:.4f})")
    print(f"   before R correction  = {naive.mean():+.4f}")
    print(f"   beta_hat             = {BETA * amps.mean():.4e}"
          f"   (input {BETA:.4e})")

    print(f"\nvelocity components, km/s   (input "
          + " ".join(f"{x:+.1f}" for x in V_TRUE_KMS) + ")")
    for k in range(3):
        nm = "xyz"[k]
        m = vel[:, k].mean()
        sem = vel[:, k].std(ddof=1) / np.sqrt(n_dat)
        sd = vel[:, k].std(ddof=1)
        print(f"   v{nm}   = {m:+9.2f} +- {sd:8.2f} (sd)   "
              f"+- {sem:6.2f} (on the mean)   "
              f"residual {m - V_TRUE_KMS[k]:+7.2f} "
              f"({(m - V_TRUE_KMS[k]) / sem:+.1f} sigma)")

    print(f"\ndirection")
    print(f"   mean of {n_dat} sims     = ra {ra_m:6.1f} deg, "
          f"dec {dec_m:6.1f} deg"
          f"   ({angle_between(mean_dir, D_TRUE):.1f} deg from truth)")
    print(f"   scatter sigma(dir)   = {np.sqrt(np.mean(ang ** 2)):5.1f} deg")
    print(f"   per-sim error        = median {np.median(ang):5.1f} deg, "
          f"best {ang.min():.1f}, worst {ang.max():.1f}")

    # Higher L: the alm response itself, then the leakage it implies.
    leak = leakage_from_alm(S)
    noise, mfcl, datleak = spectra_from_sims(mf_vecs, data_vecs, case)
    c1_recon = cl_from_power(np.abs(l1_to_alm(R @ A_TRUE)) ** 2)[1]
    report_alm_response(ar, noise)
    report_leakage(leak, noise, mfcl, c1_recon)
    print(f"   cross-check from the aberrated sims: C_2 = "
          f"{datleak[2]:+.3e} against {leak['cl'][2]:+.3e} from the response")

    # Null test: build a mean field from the first half of the unaberrated sims
    # and apply it to the second half.  They contain no aberration, so A = 0.
    h = n_mf // 2
    null = np.array([fit(v, mf_all[:h].mean(axis=0), Rinv)[0]
                     for v in mf_all[h:]])
    null_err = null.std(ddof=1) * np.sqrt(1.0 / len(null) + 1.0 / h)
    print(f"\nnull test, split-half mean field on the unaberrated sims")
    print(f"   mean A               = {null.mean():+.4f} +- "
          f"{null.std(ddof=1):.4f} (sd)    (expect 0)")
    print(f"                          {null.mean():+.4f} +- {null_err:.4f} "
          f"(error on the mean)")

    return dict(sigma=sigma, amp=amps, naive=naive, dir=dirs, vel=vel,
                velraw=velraw, R=R, Rsd=Rsd, Rerr=dR,
                Ralm=ar["R"], Ralm_sd_re=ar["sd_re"], Ralm_sd_im=ar["sd_im"],
                leak=leak["cl"], leakerr=leak["err"],
                noise=noise, mfcl=mfcl, datleak=datleak, n_dat=n_dat,
                n_mf=n_mf, n_resp=ar["n"])


def sim_budget(results, n_resp):
    """Turn the measured scatter into the sim counts a target precision needs."""
    print(f"\n{'=' * 72}\n SIM BUDGET\n{'=' * 72}")
    print("error on <A> is sigma_A sqrt(1/N_data + 1/N_mf): the mean-field")
    print("residual is shared by every data sim, so it counts the same way.")
    print("Numbers below assume N_mf = 2 N_data.\n")
    targets = (0.05, 0.03, 0.02, 0.01)
    print(f"   {'case':5s} {'sigma_A':>8s} {'now':>8s}   "
          + " ".join(f"{f'{t:.0%}':>15s}" for t in targets))
    print(f"   {'':5s} {'':8s} {'':8s}   "
          + " ".join(f"{'N_data / N_mf':>15s}" for _ in targets))
    for case in results:
        r = results[case]
        s = r["sigma"]
        now = s * np.sqrt(1.0 / r["n_dat"] + 1.0 / r["n_mf"])
        cells = []
        for t in targets:
            n = math.ceil(1.5 * (s / t) ** 2)
            cells.append(f"{n:>6d} /{2 * n:>7d}")
        print(f"   {case:5s} {s:8.4f} {now:8.4f}   " + " ".join(cells))

    last = results[list(results)[-1]]
    cl = last["leak"]
    err = last["leakerr"]
    unresolved = [l for l in range(2, L_OUT + 1) if cl[l] <= 2 * err[l]]
    print(f"\nresponse sims: {n_resp} paired.  The 2-sigma floor on the "
          f"leakage falls as 1/sqrt(N),")
    if unresolved:
        print(f"   so L = {unresolved[0]} upward is currently unresolved; "
              f"{4 * n_resp} sims would halve the floor.")
    else:
        print(f"   and every L out to {L_OUT} is already resolved above it.")


# 10. Main, Claude helped here, double check

def main():
    print(f"{'=' * 72}")
    print(f"lmin, lmax     = {LMIN}, {LMAX}  (mlmax {MLMAX}), L_out {L_OUT}")
    print(f"resolution     = {RES / utils.arcmin:.2f} arcmin, "
          f"map {shape[0]} x {shape[1]}")
    print(f"noise          = {NOISE_UK_ARCMIN:.1f} uK-arcmin in T, "
          f"x{POL_NOISE_FACTOR:.3f} in Q/U, beam {BEAM_FWHM_ARCMIN:.1f} arcmin")
    print(f"beta           = {BETA:.4e} toward ra {np.degrees(BDIR[0]):.1f}, "
          f"dec {np.degrees(BDIR[1]):.1f} deg")
    print(f"               = " + " ".join(f"{x:+.1f}" for x in V_TRUE_KMS)
          + " km/s in x, y, z")
    print(f"mask           = {MASK_KEY}"
          f"  ->  f_sky {w1:.4f}, w2 {w2:.4f}, w4 {w4:.4f}")

    if args.mask == "act":
        print(f"                 dec {ACT_DEC[0]:g} to {ACT_DEC[1]:g}, "
              f"ra {args.ra_center - RA_HALFWIDTH:.1f} to "
              f"{args.ra_center + RA_HALFWIDTH:.1f} "
              f"({w1 * 41253:.0f} deg^2)")
    if args.mask == "dr6":
        print(f"                 {os.path.basename(DR6_PATH)}")
        print(f"                 variant {args.dr6_variant}, nside "
              f"{DR6_NSIDE} -> CAR by {args.dr6_method}"
              + (f" order {args.dr6_order}" if args.dr6_method == "spline"
                 else "")
              + (f", rot {args.dr6_rot}" if args.dr6_rot else "")
              + f", digest {MASK_DIGEST}")
        print(f"                 {w1 * 41253:.0f} deg^2, "
              f"sqrt(w4)/w2 {np.sqrt(w4) / w2:.3f}")
        print(f"                 healpix w1 {DR6_W_NATIVE[0]:.4f}, "
              f"w2 {DR6_W_NATIVE[1]:.4f}, w4 {DR6_W_NATIVE[2]:.4f}")
    print(f"                 mask at the boost direction = "
          f"{mask_at(BDIR[0], BDIR[1]):.3f}")

    n_legs = 6 if RESPONSE_CENTRAL else 4
    n_recon = N_MEANFIELD + N_DATA + N_RESPONSE * n_legs
    print(f"sims           = {N_MEANFIELD} mean field, {N_RESPONSE} response, "
          f"{N_DATA} data ({n_recon} reconstructions)")
    print(f"cache          = {CACHE_DIR}")
    if args.nshards > 1:
        print(f"shard          = {args.shard} of {args.nshards}")
    print(f"parallel       = {NPROC} worker{'' if NPROC == 1 else 's'} x "
          f"{THREADS} thread{'' if THREADS == 1 else 's'}"
          f"   ({CORES} cores visible)")
    if NPROC > 1:
        # A rough floor, not a budget: a worker holds several maps of this size
        # at once inside the quadratic estimator.
        gb = 3 * shape[0] * shape[1] * 8 / 2 ** 30
        print(f"                 one TQU map is {gb:.2f} GB, and a worker "
              f"needs several at once")
    print(f"estimators     = {', '.join(ESTIMATORS)}")
    print(f"{'=' * 72}")

    mf_vecs = run_meanfield()
    diffs = run_response()
    data_vecs = run_data()
    close_pool()          # everything below is serial: take the memory back

    # A partly finished set of shards has nothing to analyse yet.
    have_all_axes = all(len(d) > 0 for d in diffs)
    if len(mf_vecs) < 2 or len(data_vecs) < 2 or not have_all_axes:
        print("\nNot all stages have results yet.  Run the remaining shards, "
              "then rerun\nwith the default --nshards 1 to collect them and "
              "print the report.")
        return

    print(f"\ncollected {len(mf_vecs)} mean-field, {len(diffs[0])} response "
          f"and {len(data_vecs)} data sims")

    per_estimator_meanfield(mf_vecs)

    results = {}
    for case in CASES:
        results[case] = analyse(case, mf_vecs, diffs, data_vecs)

    print(f"\n{'=' * 72}\n COMPARISON\n{'=' * 72}")
    base = results["T+P"]["sigma"]
    for case in results:
        pred = np.sqrt(CASE_NORM["T+P"] / CASE_NORM[case])
        print(f"sigma(A):  {case:5s} / T+P = "
              f"{results[case]['sigma'] / base:6.3f}"
              f"   (analytic forecast {1.0 / pred:6.3f})")

    sim_budget(results, results[list(results)[-1]]["n_resp"])
    write_summary(results)


def write_summary(results):
    """Everything the figures need, in one small file."""
    cases = list(results)
    thumb, thumb_dec, thumb_ra = mask_thumbnail()

    w_native = DR6_W_NATIVE if DR6_W_NATIVE is not None else np.full(3, np.nan)
    mask_file = os.path.basename(DR6_PATH) if DR6_PATH else ""
    mask_variant = args.dr6_variant if args.mask == "dr6" else ""

    out = dict(cases=np.array(cases), v_true=V_TRUE_KMS, d_true=D_TRUE,
               beta=float(BETA), c_kms=float(C_KMS), ells=_ELLS, clpp=CLPP,
               w1=float(w1), w2=float(w2), w4=float(w4),
               n_mf=results[cases[0]]["n_mf"], n_resp=int(N_RESPONSE),
               n_data=results[cases[0]]["n_dat"], lmax=int(LMAX),
               lout=int(L_OUT), mask_key=np.array(MASK_KEY),
               fsky_target=float(args.fsky),
               mask_kind=np.array(args.mask),
               mask_digest=np.array(MASK_DIGEST),
               mask_file=np.array(mask_file),
               mask_variant=np.array(mask_variant),
               dr6_nside=int(DR6_NSIDE),
               w_native=w_native,
               mask_at_dipole=float(mask_at(BDIR[0], BDIR[1])),
               res_arcmin=float(RES / utils.arcmin),
               pack_l=_PACK_L, pack_m=_PACK_M,
               n_resp_used=int(results[cases[0]]["n_resp"]),
               mask_method=np.array(args.dr6_method if args.mask == "dr6"
                                    else ""),
               mask_thumb=thumb, mask_thumb_dec=thumb_dec,
               mask_thumb_ra=thumb_ra,
               dipole_radec=np.degrees(np.asarray(BDIR, dtype=float)))

    per_case_keys = ["vel", "velraw", "amp", "naive", "dir", "R",
                     "Rsd", "Rerr", "Ralm", "Ralm_sd_re", "Ralm_sd_im",
                     "leak", "leakerr", "noise", "mfcl", "datleak"]
    for k in range(len(cases)):
        r = results[cases[k]]
        for key in per_case_keys:
            out[f"{key}_{k}"] = np.asarray(r[key])

    path = os.path.join(CACHE_DIR, "summary.npz")
    np.savez(path, **out)
    print(f"\nwrote {path}")

    if not args.plots:
        return

    # Look for the plotting module next to this file, not just in the working
    # directory, so an absolute-path launch still finds it.
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import aberration_plots
    for f in aberration_plots.make_all(aberration_plots.load(path), PLOT_DIR):
        print(f"wrote {f}")


if __name__ == "__main__":
    main()