"""
Reconstruction of the CMB aberration dipole with a lensing quadratic estimator.


  Stage I  mean field   MF from unaberrated masked and noisy sims.

  Stage II  response    R, a 3x3 matrix measured by boosting along
                        x, y and z.  Each measurement is a
                        difference of two reconstructions of the
                        CMB, with and without the boost.

  Stage III  data sims  Independent aberrated masked sims.  For each,
                            a_found = R^-1 (v - MF)
                            A     = (a_found . a_true) / (a_true . a_true)
                            d_est = -a_found / |a_found|
                        A should come out at 1 and d_est at the input direction.


Footprint
---------
--mask act is an ACT-like patch normalised to a target sky fraction
(--fsky 0.2): a declination band (--act-dec, default -60 to +22 deg) whose RA
half-width is solved for so that w1 comes out at exactly the requested f_sky,
apodisation included.  --mask box / dec_band / none also still work.

--mask dr6 uses the real thing: the released ACT DR6 lensing analysis mask,
the apodised mask that was applied to the CMB maps before the quadratic
estimator, which is where reconstruct() applies it here too.  Get it with

    wget https://phy-act1.princeton.edu/public/data/dr6_lensing_v1/maps/\
baseline/mask_act_dr6_lensing_v1_healpix_nside_4096_baseline.fits

(about 0.8 GB; other variants, e.g. galcut040, sit in the sibling
directories, and the whole release is also on LAMBDA and on the NERSC
portal).  Leave it in the working directory, or point --dr6-mask or
$ACT_DR6_MASK at it.  Cite Madhavacheril et al 2304.05203 and Qu et al
2304.05202.  It is healpix nside 4096, equatorial, 1 = keep, and already
contains the galactic cut and the point-source and cluster holes, so
--apod-deg does not apply to it.

The mask is area-averaged down to --dr6-nside (default: whatever matches the
map resolution) and then interpolated onto the CAR grid, and the projection is
cached under <cache>/masks so the shards do not each repeat it.  The header
compares the w factors of the projected mask against the exact healpix ones;
if they disagree by more than a percent the pixels are too coarse for the
source holes.  Two things the real footprint does that the smooth patch does
not: w4/w2^2 is worse than the area alone suggests, and the holes couple much
more power into high L, so the leakage numbers below want a finer --res than
the L = 1 amplitude does.  Note also that the real analysis weights by the
DR6 inverse-variance maps as well, which the white-noise model here does not,
so this is the DR6 footprint and not a DR6 forecast.


Higher L
--------
Stage II does not only give the 3x3 response at L = 1.  The same paired
difference, read at L >= 2, *is* the leakage: the coherent power a pure L = 1
boost deposits at higher multipoles once the mask couples scales.  Because
both legs share a CMB realisation the mean field cancels exactly and cosmic
variance largely cancels too, so this is far cleaner than differencing the
aberrated sims against the mean field (which is also reported, as a check).

Every stage therefore caches the reconstructed alm up to --lout (default 20)
instead of a single 3-vector.  That is a few kB per sim and changes the cache
format, so old caches are not reused.


How many sims
-------------
  mean field   the dominant cost.  Its residual is a *coherent* offset shared
               by every data sim, so it enters the error on <A> as
               sigma_A / sqrt(N_mf), exactly like the data sims do.
  response     paired, so cosmic variance cancels; ~60 is plenty for R itself.
               It is the high-L leakage that wants more: the smallest leakage
               resolvable at 2 sigma falls only as 1/sqrt(N_response).
  data         error on <A> is sigma_A / sqrt(N_data).

Cutting to f_sky = 0.2 costs roughly sqrt(w4)/w2 ~ 2.3 in sigma_A relative to
the full sky, and R becomes ill-conditioned, which costs more again, so the
counts below are several times the old defaults.  The report ends with a
budget block that turns the *measured* sigma_A into the N needed for a given
target, so the first short run tells you what the long one should be.


Long runs are resumable and shardable.  Per-sim results are cached as tiny
.npz files, so the same command can be launched with --shard 0..N-1 --nshards N
on N cores or nodes; running it once more with the defaults collects everything
and prints the report, writes summary.npz and draws the figures.

AI Statement: AI (Claude) was used to create the caching and parser code. Also used to check for errors in the code.
Helped write some functions as well - further checking still needed to confirm everything is working as intended.
The f_sky solver, the packed low-L storage, the higher-L leakage estimator and
the plotting module were also written with Claude; the leakage normalisation
and the debiasing in particular still want an independent check.
The DR6 mask loader (--mask dr6) was written with Claude as well: the
healpix -> CAR projection settings and the w-factor resolution check are the
parts to look at first, and it has not yet been run against the real file.
"""

import argparse
import hashlib
import math
import os
import time

import numpy as np
import healpy as hp
import camb

from pixell import enmap, curvedsky, aberration, reproject, utils
from falafel import qe
import pytempura


# 1. Configuration

# The released ACT DR6 lensing analysis mask: the apodised mask that was
# applied to the CMB maps before the quadratic estimator, which is exactly
# where reconstruct() applies it here.  Healpix nside 4096, equatorial, 1 =
# keep, and the galactic cut and the point-source and cluster holes are
# already in it.
DR6_DEFAULT_NAME = "mask_act_dr6_lensing_v1_healpix_nside_4096_baseline.fits"
DR6_URL_BASE = ("https://phy-act1.princeton.edu/public/data/dr6_lensing_v1/"
                "maps/{variant}/")

parser = argparse.ArgumentParser(
    description="CMB aberration reconstruction with a lensing quadratic estimator")
parser.add_argument("--quick", action="store_true",
                    help="low resolution, low lmax, few sims: smoke test only")
parser.add_argument("--seed-offset", type=int, default=0,
                    help="shift all realisations to an independent set; 0 "
                         "reproduces the original sims and reuses their cache")
parser.add_argument("--noise", type=float, default=10.0, metavar="UK_ARCMIN",
                    help="white noise level in temperature, uK-arcmin")
parser.add_argument("--pol-noise-factor", type=float, default=np.sqrt(2.0),
                    help="Q/U noise relative to T (sqrt(2) for a "
                         "polarisation-modulated experiment)")
parser.add_argument("--beam", type=float, default=0.0, metavar="FWHM_ARCMIN",
                    help="Gaussian beam FWHM; 0 for no beam")
parser.add_argument("--lmax", type=int, default=2000,
                    help="highest CMB multipole used by the estimator")
parser.add_argument("--res", type=float, default=None, metavar="ARCMIN",
                    help="map resolution; default 4 arcmin, or fine enough "
                         "for lmax if that needs less")
parser.add_argument("--lout", type=int, default=20,
                    help="highest multipole of the reconstruction that is "
                         "kept and analysed; L>=2 is the leakage")
parser.add_argument("--mask", default="act",
                    choices=["none", "box", "dec_band", "act", "dr6"])
parser.add_argument("--dr6-mask", default=None, metavar="PATH",
                    help="for --mask dr6: the released ACT DR6 lensing "
                         "analysis mask, healpix nside 4096, equatorial.  "
                         "Defaults to $ACT_DR6_MASK, then to "
                         "./" + DR6_DEFAULT_NAME)
parser.add_argument("--dr6-variant", default="baseline",
                    help="for --mask dr6: which released variant the file is "
                         "(baseline, galcut040, tonly, ...).  Used for "
                         "labelling and the cache key only; the file itself "
                         "comes from --dr6-mask")
parser.add_argument("--dr6-nside", type=int, default=None,
                    help="for --mask dr6: area-average the healpix mask to "
                         "this nside before projecting to CAR.  Default is "
                         "the power of two closest to the map resolution, "
                         "which keeps the w factors honest; 0 keeps the "
                         "native nside and point-samples it")
parser.add_argument("--dr6-smooth-arcmin", type=float, default=0.0,
                    help="for --mask dr6: extra Gaussian smoothing FWHM "
                         "applied to the released mask, in arcmin.  0 (the "
                         "default) uses the released apodisation as it is; a "
                         "nonzero value tapers the point-source holes and so "
                         "changes the high-L leakage")
parser.add_argument("--fsky", type=float, default=0.2,
                    help="for --mask act: target sky fraction w1; the RA "
                         "half-width of the patch is solved for to hit it")
parser.add_argument("--act-dec", type=float, nargs=2, default=[-60.0, 22.0],
                    metavar=("LO", "HI"),
                    help="for --mask act: declination range of the kept "
                         "patch, degrees")
parser.add_argument("--ra-center", type=float, default=180.0,
                    help="for --mask act: RA centre of the kept patch, "
                         "degrees; the default contains the dipole direction, "
                         "as the real ACT footprint does")
parser.add_argument("--apod-deg", type=float, default=3.0,
                    help="raised-cosine taper width in degrees (0 = binary "
                         "mask, which leaks far more into high L)")
parser.add_argument("--box-dec", type=float, nargs=2, default=[-20.0, 20.0],
                    metavar=("LO", "HI"),
                    help="declination range of the removed rectangle, degrees")
parser.add_argument("--box-ra", type=float, nargs=2, default=[-60.0, 60.0],
                    metavar=("LO", "HI"),
                    help="RA range of the removed rectangle, degrees; the "
                         "range runs from LO eastward to HI and may wrap")
parser.add_argument("--dec-band", type=float, default=20.0,
                    help="for --mask dec_band: |dec| below this is removed")
parser.add_argument("--n-meanfield", type=int, default=800)
parser.add_argument("--n-response", type=int, default=60)
parser.add_argument("--n-data", type=int, default=400)
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--nshards", type=int, default=1)
parser.add_argument("--cache", default="cache_aberration")
parser.add_argument("--response-noise", action="store_true")
parser.add_argument("--response-beta", type=float, default=None)
parser.add_argument("--no-plots", dest="plots", action="store_false",
                    help="skip the figures (summary.npz is still written)")
parser.add_argument("--plot-dir", default=None,
                    help="where the figures go; default <cache dir>/plots")
args = parser.parse_args()

# multipoles
LMIN, LMAX = 2, args.lmax   # CMB multipoles used by the estimator
MLMAX = LMAX + 500          # band limit of the internal harmonic transforms
RES = (args.res if args.res is not None
       else min(4.0, 10800.0 / (1.08 * MLMAX))) * utils.arcmin

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

if RES / utils.arcmin > 10800.0 / MLMAX:
    print(f"WARNING: {RES / utils.arcmin:.2f} arcmin pixels cannot carry "
          f"mlmax {MLMAX}; use --res {10800.0 / (1.08 * MLMAX):.2f} or less")


# The reconstruction is kept out to L_OUT, not just L = 1.  Storage is packed
# as (l, m) with l outer and m = 0..l inner, so index(l, m) = l(l+1)/2 + m and
# (1,0), (1,1) sit at 1 and 2.
L_OUT = args.lout
_PACK_L = np.concatenate([np.full(l + 1, l, dtype=int)
                          for l in range(L_OUT + 1)])
_PACK_M = np.concatenate([np.arange(l + 1) for l in range(L_OUT + 1)])
_PACK_W = np.where(_PACK_M == 0, 1.0, 2.0)   # m < 0 is implied by reality
_PACK_IDX = np.array([hp.Alm.getidx(MLMAX, l, m)
                      for l, m in zip(_PACK_L, _PACK_M)])
_ELLS = np.arange(L_OUT + 1)


# masking details
BOX_DEC_RANGE = tuple(args.box_dec)
BOX_RA_RANGE = tuple(args.box_ra)
DEC_BAND_DEG = args.dec_band
ACT_DEC = tuple(args.act_dec)
APOD = args.apod_deg


def _dr6_path():
    """Where the released mask is, or a message saying how to get it."""
    # An explicit --dr6-mask has to be the file that is used: falling back to
    # a different one found in the working directory would mean analysing a
    # mask nobody asked for.
    if args.dr6_mask:
        p = os.path.expanduser(args.dr6_mask)
        if not os.path.exists(p):
            raise FileNotFoundError(f"--dr6-mask {args.dr6_mask} does not exist")
        return p
    env = os.environ.get("ACT_DR6_MASK")
    if env:
        p = os.path.expanduser(env)
        if os.path.exists(p):
            return p
        print(f"WARNING: $ACT_DR6_MASK is set to {env}, which does not exist")
    if os.path.exists(DR6_DEFAULT_NAME):
        return DR6_DEFAULT_NAME
    raise FileNotFoundError(
        "--mask dr6 needs the released ACT DR6 lensing analysis mask.  "
        f"Download it with\n\n    wget "
        + DR6_URL_BASE.format(variant=args.dr6_variant)
        + f"mask_act_dr6_lensing_v1_healpix_nside_4096_{args.dr6_variant}.fits"
        + "\n\nthen point --dr6-mask or $ACT_DR6_MASK at it, or leave it in "
          "the working directory.\n(Cite Madhavacheril et al 2304.05203 and "
          "Qu et al 2304.05202 if you use it.)")


def _match_nside(res_rad):
    """Power-of-two nside whose pixels are closest to the map resolution.

    A healpix pixel is sqrt(4 pi / 12) / nside radians across.  Averaging the
    mask down to this nside before interpolating onto the CAR grid means the
    CAR pixels see the mean weight over their own footprint rather than a
    point sample of it, which is what keeps w2 and w4 from coming out high.
    """
    want = np.sqrt(4.0 * np.pi / 12.0) / res_rad
    return int(np.clip(2 ** int(round(np.log2(max(want, 1.0)))), 32, 4096))


if args.mask == "dr6":
    DR6_PATH = _dr6_path()
    DR6_NSIDE = (_match_nside(RES) if args.dr6_nside is None
                 else int(args.dr6_nside))      # 0 means keep the native one
else:
    DR6_PATH, DR6_NSIDE = None, 0

# A key naming every parameter the mask actually depends on, and only those:
# putting the box ranges in the cache key unconditionally would invalidate a
# dec_band cache whenever an unused box argument changed.
if args.mask == "none":
    MASK_KEY = "none"
elif args.mask == "dr6":
    MASK_KEY = (f"dr6_{args.dr6_variant}_"
                + (f"n{DR6_NSIDE}" if DR6_NSIDE else "native"))
    if args.dr6_smooth_arcmin > 0:
        MASK_KEY += f"_sm{args.dr6_smooth_arcmin:g}"
elif args.mask == "act":
    MASK_KEY = ("act_dec{:g}to{:g}_rac{:g}_fsky{:g}"
                .format(*ACT_DEC, args.ra_center, args.fsky))
elif args.mask == "box":
    MASK_KEY = ("box_dec{:g}to{:g}_ra{:g}to{:g}"
                .format(*BOX_DEC_RANGE, *BOX_RA_RANGE))
elif args.mask == "dec_band":
    MASK_KEY = f"decband{DEC_BAND_DEG:g}"
else:
    raise ValueError(args.mask)
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

# These settings affect only the response cache, so they go in the tags rather
# than in _config: putting them in the directory hash would discard the
# mean-field and data sims as well, which do not depend on them.
RESP_SUFFIX = (("noisy" if RESPONSE_NOISE else "clean")
               + f"_b{RESPONSE_BETA:g}"
               + ("_ctr" if RESPONSE_CENTRAL else ""))

# estimators and cases to analyze.  Every estimator also gets its own case so
# the whisker plot can show how much the polarisation actually buys.
ESTIMATORS = ["TT", "TE", "EE"]
CASES = {
    "TT": ["TT"],
    "TE": ["TE"],
    "EE": ["EE"],
    "T+P": ["TT", "TE", "EE"],
}

os.makedirs(args.cache, exist_ok=True)

# CACHE_DIR and PLOT_DIR are built at the end of section 4, once the mask
# exists: the dr6 mask comes from a file rather than from the arguments, so
# the only honest way to key the cache on it is to hash the mask itself.


def l1_vector(packed):
    """Cartesian vector a of the L=1 part of a packed alm array."""
    a10 = packed[1].real
    a11 = packed[2]
    return np.array([
        -np.sqrt(3.0 / (2.0 * np.pi)) * a11.real,
        np.sqrt(3.0 / (2.0 * np.pi)) * a11.imag,
        np.sqrt(3.0 / (4.0 * np.pi)) * a10,
    ])


def pack(alm):
    """Reconstruction alm -> the L <= L_OUT coefficients, as stored."""
    return np.asarray(alm)[_PACK_IDX].astype(np.complex128)


def cl_from_power(power):
    """Per-mode |a|^2 (packed) -> C_L, with m<0 counted."""
    num = np.bincount(_PACK_L, weights=_PACK_W * np.asarray(power, float),
                      minlength=L_OUT + 1)
    return num / (2 * _ELLS + 1.0)


def unit_vector(ra, dec):
    """Unit vector from equatorial angles in radians."""
    return np.array([np.cos(dec) * np.cos(ra),
                     np.cos(dec) * np.sin(ra),
                     np.sin(dec)])


def angle_between(u, v):
    """Angle between two vectors, in degrees."""
    c = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def to_radec(vec):
    """(ra, dec) in degrees for a Cartesian vector."""
    v = vec / np.linalg.norm(vec)
    return (np.degrees(np.arctan2(v[1], v[0]) % (2 * np.pi)),
            np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0))))


D_TRUE = unit_vector(BDIR[0], BDIR[1])
A_TRUE = -BETA * D_TRUE          # the input aberration potential as a vector
V_TRUE_KMS = BETA * C_KMS * D_TRUE
# C_1 of that input, for reference: a_10 and a_11 follow from l1_vector.
C1_INPUT = 4.0 * np.pi * BETA ** 2 / 9.0

# Boost directions used to measure the columns of R, one per Cartesian axis.
RESPONSE_DIRS = [(0.0, 0.0),           # x
                 (np.pi / 2, 0.0),     # y
                 (0.0, np.pi / 2)]     # z


# CAMB, Filters, Noise, and Normalization

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
try:
    _pp = camb_results.get_lens_potential_cls(lmax=max(L_OUT, 10))[:, 0]
    CLPP = np.zeros(L_OUT + 1)
    _l = _ELLS[1:].astype(float)
    CLPP[1:] = 2.0 * np.pi * _pp[1:L_OUT + 1] / (_l * (_l + 1.0)) ** 2
    CLPP[~np.isfinite(CLPP)] = 0.0
except Exception as exc:                                  # noqa: BLE001
    print(f"   (no lensing potential for the reference curve: {exc})")
    CLPP = np.zeros(L_OUT + 1)

# Noise, converted from uK-arcmin into the same dimensionless units.
noise_rms = NOISE_UK_ARCMIN * utils.arcmin / TCMB_UK
nltt = np.full(MLMAX + 1, noise_rms ** 2)
nlee = POL_NOISE_FACTOR ** 2 * nltt
nlbb = nlee.copy()

if BEAM_FWHM_ARCMIN > 0:
    bl = hp.gauss_beam(BEAM_FWHM_ARCMIN * utils.arcmin, lmax=MLMAX)
    nltt, nlee, nlbb = (n / bl ** 2 for n in (nltt, nlee, nlbb))

# The filters zero everything outside [LMIN, LMAX] anyway, but a mask couples
# scales, so stop the noise from growing without bound above LMAX to be safe
for n in (nltt, nlee, nlbb):
    n[LMAX + 1:] = n[LMAX]

ucls = {"TT": cltt, "EE": clee, "BB": clbb, "TE": clte}          # weights
tcls = {"TT": cltt + nltt, "EE": clee + nlee,                    # filters
        "BB": clbb + nlbb, "TE": clte}

# Signal and noise covariances in the T, E, B basis, for generating the sims.
signal_ps = np.zeros((3, 3, MLMAX + 1))
signal_ps[0, 0] = cltt
signal_ps[1, 1] = clee
signal_ps[2, 2] = clbb
signal_ps[0, 1] = signal_ps[1, 0] = clte

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
# Unlike before we keep the whole low-L array, not just L = 1: the leakage at
# L >= 2 has to be normalised at its own L to mean anything.
print("normalisations from tempura ...", flush=True)
Als = pytempura.get_norms(ESTIMATORS, ucls, ucls, tcls, LMIN, LMAX, k_ellmax=MLMAX)


def _safe_norm(a):
    """A_L with anything unusable (L=0, zeros, nans) pushed to infinity."""
    a = np.asarray(a, dtype=float).copy()
    a[~np.isfinite(a) | (a <= 0)] = np.inf
    a[0] = np.inf                      # no L = 0 gradient mode
    return a


AL = {e: _safe_norm(np.asarray(Als[e][0])[:L_OUT + 1]) for e in ESTIMATORS}
A1 = {e: float(AL[e][1]) for e in ESTIMATORS}       # the value at L = 1
if not all(np.isfinite(v) for v in A1.values()):
    raise RuntimeError("tempura gave no usable normalisation at L = 1")

# Combined normalisation per case, now per L.  The unnormalised estimators add
# directly, so the inverse-variance coadd is N_L * sum_e q_e with
# N_L = 1 / sum_e (1 / A_L^e); at L = 1 this is the old CASE_NORM.
with np.errstate(divide="ignore", invalid="ignore"):
    _inv = {c: sum(1.0 / AL[e] for e in ests) for c, ests in CASES.items()}
CASE_NORM_L = {c: np.where(v > 0, 1.0 / np.where(v > 0, v, 1.0), 0.0)
               for c, v in _inv.items()}
CASE_NORM = {c: float(CASE_NORM_L[c][1]) for c in CASES}
NORM_PACK = {c: CASE_NORM_L[c][_PACK_L] for c in CASES}


# 4. Masking and Map Geometry

shape, wcs = enmap.fullsky_geometry(res=RES)
px = qe.pixelization(shape=shape, wcs=wcs)

# A CAR geometry is separable, so the 1-D axes are enough to build the mask and
# to integrate it exactly.  That avoids three full-sky float64 maps, which is
# worth having back now that the sim counts are several times larger.
try:
    _dec_ax, _ra_ax = enmap.posaxes(shape, wcs)
except Exception:                                          # noqa: BLE001
    _pm = enmap.posmap(shape, wcs)
    _dec_ax, _ra_ax = _pm[0][:, 0].copy(), _pm[1][0, :].copy()
    del _pm

DEC_AX = np.degrees(_dec_ax)
RA_AX = (np.degrees(_ra_ax) + 180.0) % 360.0 - 180.0
_ddec = abs(wcs.wcs.cdelt[1]) * utils.degree
_dra = abs(wcs.wcs.cdelt[0]) * utils.degree
# Exact solid angle of a CAR row, poles included.
ROW_AREA = _dra * (np.sin(np.clip(_dec_ax + _ddec / 2, -np.pi / 2, np.pi / 2))
                   - np.sin(np.clip(_dec_ax - _ddec / 2,
                                    -np.pi / 2, np.pi / 2)))


def edge_taper(x, lo, hi, width):
    """1 well inside [lo, hi], 0 outside, raised-cosine ramp of `width`."""
    if width <= 0:
        return ((x > lo) & (x < hi)).astype(float)
    rise = np.clip((x - lo) / width, 0.0, 1.0)
    fall = np.clip((hi - x) / width, 0.0, 1.0)
    return 0.25 * (1 - np.cos(np.pi * rise)) * (1 - np.cos(np.pi * fall))


def _act_parts(halfwidth):
    """The separable (dec, ra) factors of the ACT-like patch."""
    ra_rel = (RA_AX - args.ra_center + 180.0) % 360.0 - 180.0
    return (edge_taper(DEC_AX, ACT_DEC[0], ACT_DEC[1], APOD),
            edge_taper(ra_rel, -halfwidth, halfwidth, APOD))


def _act_fsky(halfwidth):
    """w1 of the patch, exactly, using separability."""
    td, tr = _act_parts(halfwidth)
    return float(ROW_AREA @ td / ROW_AREA.sum()) * float(tr.mean())


def solve_ra_halfwidth(target):
    """RA half-width that makes w1 equal `target`, apodisation included."""
    hi = 180.0 - max(APOD, 0.0) - 0.5      # stop the two ramps from meeting
    if not 0.0 < target < 1.0:
        raise ValueError(f"--fsky {target} must be in (0, 1)")
    if _act_fsky(hi) < target:
        raise ValueError(
            f"--fsky {target} is unreachable with --act-dec "
            f"{ACT_DEC[0]:g} {ACT_DEC[1]:g} and --apod-deg {APOD:g}: the full "
            f"RA range only gives {_act_fsky(hi):.3f}.  Widen the "
            f"declination range.")
    lo = 1e-4
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _act_fsky(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


RA_HALFWIDTH = (solve_ra_halfwidth(args.fsky) if args.mask == "act"
                else float("nan"))


# --- the released ACT DR6 mask ----------------------------------------------
# Projecting nside 4096 onto the CAR grid costs a ~0.8 GB read and a minute or
# so, which is wasteful once per run and actively bad once per shard, so the
# projected mask is cached next to the sims.  W_NATIVE holds w1, w2, w4 of the
# healpix mask itself: healpix pixels are equal area, so the plain mean of
# mask**n is exact, and comparing it with the CAR values says whether the
# resolution is fine enough for the source holes.
DR6_W_NATIVE = None


def _dr6_cache_paths():
    """Where the projected mask and its native w factors are kept."""
    d = os.path.join(args.cache, "masks")
    os.makedirs(d, exist_ok=True)
    key = (f"{os.path.basename(DR6_PATH)}_{os.path.getsize(DR6_PATH)}"
           f"_n{DR6_NSIDE}_res{RES:.12g}_sm{args.dr6_smooth_arcmin:g}")
    base = os.path.join(
        d, f"dr6_{args.dr6_variant}_"
           + (f"n{DR6_NSIDE}" if DR6_NSIDE else "native")
           + f"_{hashlib.md5(key.encode()).hexdigest()[:10]}")
    return base + ".fits", base + "_w.npy"


def load_dr6_mask():
    """The released DR6 lensing mask, projected onto this CAR geometry."""
    global DR6_W_NATIVE
    car_path, w_path = _dr6_cache_paths()
    if os.path.exists(car_path) and os.path.exists(w_path):
        cached = enmap.read_map(car_path)
        if tuple(cached.shape[-2:]) == tuple(shape[-2:]):
            print(f"DR6 mask: reusing {car_path}", flush=True)
            DR6_W_NATIVE = np.load(w_path)
            return cached
        # Belt and braces: RES is already in the cache key, so this should
        # not happen unless the geometry convention itself changed.
        print(f"DR6 mask: cached projection is {tuple(cached.shape[-2:])} but "
              f"this run wants {tuple(shape[-2:])}; rebuilding", flush=True)
        del cached

    print(f"DR6 mask: reading {DR6_PATH}", flush=True)
    t0 = time.time()
    # nest=False makes healpy reorder a NESTED file for us; the spline
    # projection below assumes RING.
    m = hp.read_map(DR6_PATH, field=0, dtype=np.float32)
    m = np.clip(np.nan_to_num(m), 0.0, 1.0)
    nside_in = hp.npix2nside(m.size)
    print(f"   nside {nside_in}, {m.size} pixels, "
          f"{time.time() - t0:.1f} s", flush=True)

    if args.dr6_smooth_arcmin > 0:
        print(f"   smoothing by {args.dr6_smooth_arcmin:g} arcmin FWHM "
              f"(this is an SHT at nside {nside_in}, be patient)", flush=True)
        m = np.clip(hp.smoothing(m.astype(np.float64),
                                 fwhm=args.dr6_smooth_arcmin * utils.arcmin),
                    0.0, 1.0).astype(np.float32)

    # Exact, and the reference the CAR mask is judged against.
    DR6_W_NATIVE = np.array([float(np.mean(m.astype(np.float64) ** n))
                             for n in (1, 2, 4)])

    if DR6_NSIDE and DR6_NSIDE != nside_in:
        # ud_grade averages, so w1 is preserved exactly and the sub-pixel
        # structure is turned into partial weights rather than point samples.
        print(f"   averaging down to nside {DR6_NSIDE}", flush=True)
        m = hp.ud_grade(m, DR6_NSIDE).astype(np.float32)

    if not hasattr(reproject, "healpix2map"):
        raise RuntimeError("this pixell has no reproject.healpix2map; "
                           "upgrade pixell")
    t0 = time.time()
    # method="spline" rather than the default "harm": harmonic interpolation
    # rings around the edges and the holes and can put small negative values
    # into a mask that is then squared and fourth-powered.  rot=None because
    # the released mask is already equatorial - do not let it default to
    # "gal,equ" as the older reproject helpers did.
    car = reproject.healpix2map(m, shape=shape, wcs=wcs, method="spline",
                                order=1, spin=[0], rot=None)
    del m
    print(f"   projected onto {shape[0]} x {shape[1]} CAR in "
          f"{time.time() - t0:.1f} s", flush=True)

    out = enmap.enmap(np.clip(np.asarray(car, dtype=np.float64), 0.0, 1.0),
                      wcs)
    del car
    # Temp-then-rename, so two shards starting at once cannot leave a half
    # written mask behind for the third one to read.
    tmp = f"{car_path}.tmp{os.getpid()}.fits"
    enmap.write_map(tmp, out)
    os.replace(tmp, car_path)
    np.save(w_path, DR6_W_NATIVE)
    print(f"   cached as {car_path}", flush=True)
    return out


def build_mask():
    """1 where the sky is kept, 0 where it is cut."""
    w = APOD
    ones = np.ones(shape[1])
    if args.mask == "none":
        keep = np.outer(np.ones(shape[0]), ones)
    elif args.mask == "dr6":
        return load_dr6_mask()             # already an enmap on this geometry
    elif args.mask == "act":
        td, tr = _act_parts(RA_HALFWIDTH)
        keep = np.outer(td, tr)
    elif args.mask == "box":
        lo, hi = BOX_RA_RANGE
        if not (-180.0 <= lo < hi <= 180.0):
            raise ValueError(
                f"--box-ra {lo} {hi}: edge_taper cannot handle a wrapping or "
                f"reversed range; give lo < hi within [-180, 180]")
        keep = 1.0 - np.outer(edge_taper(DEC_AX, *BOX_DEC_RANGE, w),
                              edge_taper(RA_AX, *BOX_RA_RANGE, w))
    elif args.mask == "dec_band":
        keep = 1.0 - np.outer(
            edge_taper(DEC_AX, -DEC_BAND_DEG, DEC_BAND_DEG, w), ones)
    else:
        raise ValueError(args.mask)
    return enmap.enmap(keep, wcs)


mask = build_mask()


def w_factor(n):
    """<W^n> over the sphere."""
    return float(ROW_AREA @ (np.asarray(mask) ** n).mean(axis=1)
                 / ROW_AREA.sum())


w1, w2, w4 = (w_factor(n) for n in (1, 2, 4))

# A helper for the report: is the boost direction actually inside the patch?
def mask_at(ra, dec):
    """Mask value at one equatorial position, radians."""
    pos = np.array([[dec], [ra]])
    try:
        return float(np.asarray(mask.at(pos, order=1)).ravel()[0])
    except Exception:                                      # noqa: BLE001
        y, x = enmap.sky2pix(shape, wcs, pos)
        return float(np.asarray(mask)[int(round(y[0])) % shape[0],
                                      int(round(x[0])) % shape[1]])


# The cache stores reconstructions, which depend on every setting above.  Keying
# the directory on those settings means a run with a different mask, lmax or
# noise level can never silently reload the wrong sims.  Change any of them and
# you get a fresh directory; change nothing and the old results are reused.
# "v2" marks the switch from a stored 3-vector to stored alm: the old files
# cannot be read as the new format, so they must not share a directory.
_config = ("v2", L_OUT, LMIN, LMAX, MLMAX, round(RES, 12), NOISE_UK_ARCMIN,
           round(float(POL_NOISE_FACTOR), 12), BEAM_FWHM_ARCMIN,
           MASK_KEY, round(float(BETA), 12),
           tuple(round(float(x), 12) for x in np.asarray(BDIR).ravel()))
if args.seed_offset:
    _config = _config + ("seed", args.seed_offset)

# MASK_KEY names the analytic masks completely, but the dr6 mask comes out of a
# file, so hash the mask itself: a re-downloaded or edited file, or a different
# variant left under the same name, then cannot silently reuse these sims.
# Appended only for dr6, so every other mask hashes exactly as it did before
# this option existed and the existing caches stay reachable.
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


# 5. Simulation and Reconstruction

def seeds(stage, i):
    """Non-overlapping (signal, noise) seeds."""
    base = {"meanfield": 1_000_000, "response": 2_000_000, "data": 3_000_000}[stage]
    base += 10_000_000 * args.seed_offset
    return base + 2 * i, base + 2 * i + 1


def cmb_map(seed):
    """Unaberrated T, Q, U realisation of the theory spectra."""
    return curvedsky.rand_map((3,) + shape, wcs, signal_ps, lmax=MLMAX, seed=seed)


def noise_map(seed):
    """Instrument noise.  Added after aberration: noise is not boosted."""
    return curvedsky.rand_map((3,) + shape, wcs, noise_ps, lmax=MLMAX, seed=seed)


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
    return NORM_PACK[case] * sum(vecs[e] for e in CASES[case])


def coadd(vecs, case):
    """Normalised L=1 vector of the reconstructed phi, for one case."""
    return l1_vector(coadd_alm(vecs, case))


# --- caching ----------------------------------------------------------------
# Note that Claude has generated caching so need to look at the code again.

# Each sim stores a few kB (the alm out to L_OUT), so a finished stage reloads
# instantly, interrupted runs resume, and shards can run in parallel.

def cache_get(tag):
    path = os.path.join(CACHE_DIR, tag + ".npz")
    if not os.path.exists(path):
        return None
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def cache_put(tag, vecs):
    path = os.path.join(CACHE_DIR, tag + ".npz")
    # The temporary name must end in .npz: np.savez silently appends that
    # suffix otherwise, and the rename would then look for a file that does
    # not exist.  Writing to a temp name and renaming keeps the cache safe
    # against interrupted runs and against two shards writing at once.
    tmp = f"{path}.tmp{os.getpid()}.npz"
    np.savez(tmp, **vecs)
    os.replace(tmp, path)


def mine(i):
    """True if simulation i belongs to this shard."""
    return i % args.nshards == args.shard


def collect(tags):
    """Cached results for a list of tags, skipping any that are missing."""
    return [v for v in (cache_get(t) for t in tags) if v is not None]


class Progress:
    """Per-sim timing with a running estimate of what is left."""

    def __init__(self, total, label):
        self.total, self.label = total, label
        self.done, self.t0 = 0, time.time()

    def tick(self, i):
        self.done += 1
        rate = (time.time() - self.t0) / self.done
        left = rate * (self.total - self.done)
        print(f"    {self.label} {i:6d}   {rate:6.1f} s/sim   "
              f"{self.done}/{self.total}   {left / 60:6.1f} min left",
              flush=True)


# 6.  Mean field
# With a mask the estimator has a nonzero expectation even with no aberration.
# We measure it from unaberrated sims and subtract.

def run_meanfield():
    print(f"\n[A] mean field: {N_MEANFIELD} unaberrated sims", flush=True)
    todo = [i for i in range(N_MEANFIELD)
            if mine(i) and cache_get(f"mf_{i:05d}") is None]
    prog = Progress(len(todo), "sim")
    for i in todo:
        s_sig, s_noi = seeds("meanfield", i)
        cache_put(f"mf_{i:05d}", reconstruct(cmb_map(s_sig) + noise_map(s_noi)))
        prog.tick(i)
    return collect([f"mf_{i:05d}" for i in range(N_MEANFIELD)])


# 7. Response Matrix
# Both legs of each pair use the same CMB realisations

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


def _resp_pair(i, j):
    """Difference for sim i, axis j, scaled so that <D> = -beta_resp R e_j.

    Returns None if either leg is missing, so shards and interrupted runs are
    handled the same way as before.
    """
    plus = cache_get(f"resp_{RESP_SUFFIX}_{j}p_{i:05d}")
    if plus is None:
        return None
    if RESPONSE_CENTRAL:
        other, denom = cache_get(f"resp_{RESP_SUFFIX}_{j}m_{i:05d}"), 2.0
    else:
        other, denom = cache_get(f"resp_{RESP_SUFFIX}_off_{i:05d}"), 1.0
    if other is None:
        return None
    return {e: (plus[e] - other[e]) / denom for e in ESTIMATORS}


def run_response():
    print(f"\n[B] response: {N_RESPONSE} sims x 3 axes, "
          f"{'centred' if RESPONSE_CENTRAL else 'one-sided'} difference, "
          f"beta {RESPONSE_BETA:.4e}, "
          f"{'with' if RESPONSE_NOISE else 'without'} noise", flush=True)

    # The unboosted leg is only needed for the one-sided difference; the
    # centred difference cancels the mean field between its own two legs.
    if not RESPONSE_CENTRAL:
        todo = [i for i in range(N_RESPONSE) if mine(i)
                and cache_get(f"resp_{RESP_SUFFIX}_off_{i:05d}") is None]
        prog = Progress(len(todo), "unboosted")
        for i in todo:
            s_sig, s_noi = seeds("response", i)
            cache_put(f"resp_{RESP_SUFFIX}_off_{i:05d}",
                      response_leg(s_sig, s_noi))
            prog.tick(i)

    signs = (+1, -1) if RESPONSE_CENTRAL else (+1,)
    for j, direction in enumerate(RESPONSE_DIRS):
        for sgn in signs:
            label = f"{j}{'p' if sgn > 0 else 'm'}"
            todo = [i for i in range(N_RESPONSE) if mine(i)
                    and cache_get(f"resp_{RESP_SUFFIX}_{label}_{i:05d}") is None]
            if not todo:
                continue
            # dir is kept fixed and the sign carried by beta: a negative beta
            # is exactly the deboost, so this is the -e_j leg.
            aberrator = aberration.Aberrator(shape, wcs,
                                             dir=np.array(direction),
                                             beta=sgn * RESPONSE_BETA)
            prog = Progress(len(todo), f"axis {'xyz'[j]}{'+' if sgn > 0 else '-'}")
            for i in todo:
                s_sig, s_noi = seeds("response", i)
                cache_put(f"resp_{RESP_SUFFIX}_{label}_{i:05d}",
                          response_leg(s_sig, s_noi, aberrator))
                prog.tick(i)
            del aberrator

    # One dict per axis, keyed by sim index, so the three axes can be matched
    # up sim by sim for the leakage even when a shard is only partly done.
    out = []
    for j in range(3):
        d = {}
        for i in range(N_RESPONSE):
            pair = _resp_pair(i, j)
            if pair is not None:
                d[i] = pair
        out.append(d)
    return out


def response_matrix(diffs, case):
    """The 3x3 response and its Monte Carlo error, for one case."""
    R = np.zeros((3, 3))
    dR = np.zeros((3, 3))
    for j, column in enumerate(diffs):
        v = np.array([coadd(d, case) for d in column.values()]) / (-RESPONSE_BETA)
        R[:, j] = v.mean(axis=0)
        if len(v) > 1:
            dR[:, j] = v.std(axis=0, ddof=1) / np.sqrt(len(v))
    return R, dR


# 7b.  Leakage into L >= 2
# The same paired differences, read above L = 1.  Contract the three measured
# columns with the true input vector and you have the coherent field that the
# real boost puts on the sky, mean field already cancelled inside each pair.

def leakage_from_response(diffs, case):
    """Coherent C_L of the boost-induced signal, MC-noise debiased."""
    common = sorted(set(diffs[0]) & set(diffs[1]) & set(diffs[2]))
    if len(common) < 2:
        return None
    S = np.array([
        sum(A_TRUE[j] * coadd_alm(diffs[j][i], case) / (-RESPONSE_BETA)
            for j in range(3))
        for i in common])                       # (n, npack)
    n = len(S)
    mean = S.mean(axis=0)
    var = (S.real.var(axis=0, ddof=1) + S.imag.var(axis=0, ddof=1)) / n
    # <|mean|^2> = |truth|^2 + var/n, so subtract it rather than reporting a
    # leakage that is really just Monte Carlo noise.
    cl = cl_from_power(np.abs(mean) ** 2 - var)

    if n < 4:                       # a jackknife on 2-3 sims means nothing
        return dict(cl=cl, err=np.full(L_OUT + 1, np.inf), n=n)
    tot = S.sum(axis=0)
    jk = np.empty((n, L_OUT + 1))
    for k in range(n):
        rest = np.delete(S, k, axis=0)
        v_k = (rest.real.var(axis=0, ddof=1)
               + rest.imag.var(axis=0, ddof=1)) / (n - 1)
        jk[k] = cl_from_power(np.abs((tot - S[k]) / (n - 1)) ** 2 - v_k)
    err = np.sqrt((n - 1) / n * ((jk - jk.mean(axis=0)) ** 2).sum(axis=0))
    return dict(cl=cl, err=err, n=n)


def spectra_from_sims(mf_vecs, data_vecs, case):
    """Noise floor, mean-field power, and the data-sim view of the leakage."""
    M = np.array([coadd_alm(v, case) for v in mf_vecs])
    D = np.array([coadd_alm(v, case) for v in data_vecs])
    n_m, n_d = len(M), len(D)
    mbar = M.mean(axis=0)
    var_m = (M.real.var(axis=0, ddof=1) + M.imag.var(axis=0, ddof=1))
    var_d = (D.real.var(axis=0, ddof=1) + D.imag.var(axis=0, ddof=1))

    noise = cl_from_power(var_m)                        # per-realisation N_L
    mfcl = cl_from_power(np.abs(mbar) ** 2 - var_m / n_m)
    mean = D.mean(axis=0) - mbar
    datleak = cl_from_power(np.abs(mean) ** 2 - var_d / n_d - var_m / n_m)
    return noise, mfcl, datleak


# 8. Aberrated Data Simulations

def run_data():
    print(f"\n[C] data: {N_DATA} aberrated sims", flush=True)
    todo = [i for i in range(N_DATA)
            if mine(i) and cache_get(f"dat_{i:05d}") is None]
    if todo:
        aberrator = aberration.Aberrator(shape, wcs, dir=BDIR, beta=BETA)
        prog = Progress(len(todo), "sim")
        for i in todo:
            s_sig, s_noi = seeds("data", i)
            boosted = aberrator(cmb_map(s_sig))
            cache_put(f"dat_{i:05d}", reconstruct(boosted + noise_map(s_noi)))
            del boosted
            prog.tick(i)
        del aberrator
    return collect([f"dat_{i:05d}" for i in range(N_DATA)])


def fit(v, mf, Rinv):
    """One reconstructed vector -> (amplitude, direction)."""
    a_hat = Rinv @ (v - mf)
    return ((a_hat @ A_TRUE) / (A_TRUE @ A_TRUE),
            -a_hat / np.linalg.norm(a_hat))


# 9.  Analysis and Reporting

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
        print(f"   {e:4s}  " + " ".join(f"{x:+9.3f}" for x in v)
              + f"   {np.linalg.norm(v):8.3f}   +-{err:.3f} per component")
    for case, ests in CASES.items():
        if len(ests) < 2:
            continue                  # already printed above, as an estimator
        v = np.mean([coadd(d, case) for d in mf_vecs], axis=0) / scale
        print(f"   {case:12s}" + " ".join(f"{x:+9.3f}" for x in v)
              + f"   {np.linalg.norm(v):8.3f}")


def report_leakage(leak, noise, mfcl, c1_recon):
    """Print the leakage table for one case."""
    if leak is None:
        print("\nleakage: not enough matched response sims")
        return
    cl, err = leak["cl"], leak["err"]
    print(f"\nleakage of the L=1 boost into higher L, from {leak['n']} paired "
          f"response sims")
    print(f"   input dipole C_1 = {C1_INPUT:.3e}, recovered C_1 = "
          f"{c1_recon:.3e}  ({c1_recon / C1_INPUT:.3f} of input)")
    print(f"     L      C_L^leak      /C_1^rec     MC err      N_L^recon"
          f"     leak/N_L")
    show = [l for l in range(1, L_OUT + 1) if l <= 8 or l % 5 == 0]
    for l in show:
        flag = "" if cl[l] > 2 * err[l] else "   (< 2 sigma)"
        print(f"   {l:3d}   {cl[l]:+.4e}    {cl[l] / c1_recon:9.5f}   "
              f"{err[l]:.2e}   {noise[l]:.3e}   {cl[l] / noise[l]:9.2e}{flag}")
    tail = np.clip(cl[2:], 0.0, None)
    ratio = (tail * (2 * _ELLS[2:] + 1)).sum() / (c1_recon * 3.0)
    print(f"   total power at L>=2, relative to the recovered L=1 dipole: "
          f"{ratio:.4f}")
    print(f"   mean field power at L=2 is {mfcl[2] / max(cl[2], 1e-300):.1f}x "
          f"the leakage there, which is why this is measured from the "
          f"paired\n   differences and not from the aberrated sims")


def analyse(case, mf_vecs, diffs, data_vecs):
    print(f"\n{'=' * 72}\n {case}\n{'=' * 72}")

    R, dR = response_matrix(diffs, case)
    Rinv = np.linalg.inv(R)
    scale = np.max(np.abs(R))
    print(f"response matrix R, from {len(diffs[0])} paired sims per axis")
    for row in R:
        print("     " + "   ".join(f"{x:+9.5f}" for x in row))
    print(f"   Monte Carlo error   {np.max(dR) / scale * 100:.3f}% of the "
          f"largest element")
    print(f"   |R - I|_max         {np.max(np.abs(R - np.eye(3))):.4f}"
          + ("   <- should be small: this validates the normalisation"
             if args.mask == "none" else "   (mask suppression, expected)"))
    print(f"   condition number    {np.linalg.cond(R):.2f}"
          + ("   <- large: the mask leaves one direction poorly constrained"
             if np.linalg.cond(R) > 5 else ""))
    # Which sky directions the footprint actually constrains.
    evals, evecs = np.linalg.eigh(0.5 * (R + R.T))
    for val, vec in zip(evals[::-1], evecs.T[::-1]):
        ra_e, dec_e = to_radec(vec)
        print(f"   eigenvalue {val:+8.5f}  along ra {ra_e:6.1f}, "
              f"dec {dec_e:+6.1f}")

    mf_all = np.array([coadd(v, case) for v in mf_vecs])
    mf = mf_all.mean(axis=0)
    n_mf = len(mf_all)

    dat = np.array([coadd(v, case) for v in data_vecs])
    amps, dirs = zip(*[fit(v, mf, Rinv) for v in dat])
    amps, dirs = np.array(amps), np.array(dirs)
    a_hats = np.array([Rinv @ (v - mf) for v in dat])
    vel = -a_hats * C_KMS                       # velocity in km/s
    n_dat = len(amps)

    sigma = amps.std(ddof=1)
    err_mean = sigma * np.sqrt(1.0 / n_dat + 1.0 / n_mf)   # includes MF noise

    # Forecast from the analytic normalisation, for a full sky with no mean
    # field: sigma(A) = sqrt(3 A_1 / 4pi) / beta.
    sigma_pred = np.sqrt(3 * CASE_NORM[case] / (4 * np.pi)) / BETA

    # Amplitude with no response correction, to show the mask suppression.
    naive = (dat - mf) @ A_TRUE / (A_TRUE @ A_TRUE)

    ang = np.array([angle_between(d, D_TRUE) for d in dirs])
    mean_dir = -a_hats.mean(axis=0)
    mean_dir /= np.linalg.norm(mean_dir)
    ra_m, dec_m = to_radec(mean_dir)

    print(f"\nmean field / expected signal at L=1  = "
          f"{np.linalg.norm(mf) / np.linalg.norm(R @ A_TRUE):.2f}")
    print(f"\namplitude, from {n_dat} aberrated sims and a "
          f"{n_mf}-sim mean field")
    print(f"   mean A               = {amps.mean():+.4f} +- {err_mean:.4f}"
          f"    (expect 1)")
    print(f"   scatter sigma(A)     = {sigma:.4f}"
          f"       (full-sky forecast {sigma_pred:.4f})")
    print(f"   before R correction  = {naive.mean():+.4f}")
    print(f"   beta_hat             = {BETA * amps.mean():.4e}"
          f"   (input {BETA:.4e})")
    print(f"\nvelocity components, km/s   (input "
          + " ".join(f"{x:+.1f}" for x in V_TRUE_KMS) + ")")
    for k, nm in enumerate("xyz"):
        m = vel[:, k].mean()
        sem = vel[:, k].std(ddof=1) / np.sqrt(n_dat)
        print(f"   v{nm}   = {m:+9.2f} +- {sem:6.2f}   "
              f"scatter {vel[:, k].std(ddof=1):8.2f}   "
              f"residual {m - V_TRUE_KMS[k]:+7.2f} "
              f"({(m - V_TRUE_KMS[k]) / sem:+.1f} sigma)")
    print(f"\ndirection")
    print(f"   mean of {n_dat} sims     = ra {ra_m:6.1f} deg, dec {dec_m:6.1f} deg"
          f"   ({angle_between(mean_dir, D_TRUE):.1f} deg from truth)")
    print(f"   scatter sigma(dir)   = {np.sqrt(np.mean(ang ** 2)):5.1f} deg")
    print(f"   per-sim error        = median {np.median(ang):5.1f} deg, "
          f"best {ang.min():.1f}, worst {ang.max():.1f}")

    # Higher L.
    leak = leakage_from_response(diffs, case)
    noise, mfcl, datleak = spectra_from_sims(mf_vecs, data_vecs, case)
    c1_recon = cl_from_power(np.abs(_l1_to_alm(R @ A_TRUE)) ** 2)[1]
    report_leakage(leak, noise, mfcl, c1_recon)
    if leak is not None:
        print(f"   cross-check from the aberrated sims: C_2 = "
              f"{datleak[2]:+.3e} against {leak['cl'][2]:+.3e} from the "
              f"response")

    # Null test: build a mean field from the first half of the unaberrated sims
    # and apply it to the second half.  They contain no aberration, so A = 0.
    if n_mf >= 8:
        h = n_mf // 2
        null = np.array([fit(v, mf_all[:h].mean(axis=0), Rinv)[0]
                         for v in mf_all[h:]])
        err = null.std(ddof=1) * np.sqrt(1.0 / len(null) + 1.0 / h)
        print(f"\nnull test, split-half mean field on the unaberrated sims")
        print(f"   mean A               = {null.mean():+.4f} +- {err:.4f}"
              f"    (expect 0)")

    return dict(sigma=sigma, amp=amps, dir=dirs, vel=vel, R=R,
                leak=(leak["cl"] if leak else np.zeros(L_OUT + 1)),
                leakerr=(leak["err"] if leak else np.zeros(L_OUT + 1)),
                noise=noise, mfcl=mfcl, datleak=datleak, n_dat=n_dat,
                n_mf=n_mf)


def _l1_to_alm(a):
    """Inverse of l1_vector: a Cartesian L=1 vector as a packed alm array."""
    out = np.zeros(len(_PACK_L), dtype=np.complex128)
    out[1] = a[2] * np.sqrt(4.0 * np.pi / 3.0)
    out[2] = (-a[0] + 1j * a[1]) * np.sqrt(2.0 * np.pi / 3.0)
    return out


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
    for case, r in results.items():
        s = r["sigma"]
        now = s * np.sqrt(1.0 / r["n_dat"] + 1.0 / r["n_mf"])
        row = f"   {case:5s} {s:8.4f} {now:8.4f}   "
        cells = []
        for t in targets:
            n = math.ceil(1.5 * (s / t) ** 2)
            cells.append(f"{n:>6d} /{2 * n:>7d}")
        print(row + " ".join(cells))

    best = results[list(results)[-1]]
    cl, err = best["leak"], best["leakerr"]
    if not np.any(np.isfinite(err) & (err > 0)):
        return
    hi = [l for l in range(2, L_OUT + 1) if cl[l] <= 2 * err[l]]
    print(f"\nresponse sims: {n_resp} paired.  The 2-sigma floor on the "
          f"leakage falls as 1/sqrt(N),")
    if hi:
        print(f"   so L = {hi[0]} upward is currently unresolved; "
              f"{4 * n_resp} sims would halve the floor.")
    else:
        print(f"   and every L out to {L_OUT} is already resolved above it.")


# 10.  Main, Claude helped here, double check

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
        print(f"                 variant {args.dr6_variant}, projected from "
              f"nside {DR6_NSIDE or 'native'}"
              + (f", smoothed {args.dr6_smooth_arcmin:g} arcmin"
                 if args.dr6_smooth_arcmin > 0 else "")
              + f", digest {MASK_DIGEST}")
        print(f"                 {w1 * 41253:.0f} deg^2, "
              f"sqrt(w4)/w2 {np.sqrt(w4) / w2:.3f} "
              f"(the penalty on sigma_A relative to the full sky)")
        if DR6_W_NATIVE is not None:
            print(f"                 healpix w1 {DR6_W_NATIVE[0]:.4f}, "
                  f"w2 {DR6_W_NATIVE[1]:.4f}, w4 {DR6_W_NATIVE[2]:.4f}"
                  f"  ->  CAR is "
                  + ", ".join(
                      f"{100 * (c / n - 1):+.2f}%" for c, n in
                      zip((w1, w2, w4), DR6_W_NATIVE))
                  + " in w1, w2, w4")
            worst = max(abs(c / n - 1) for c, n in
                        zip((w1, w2, w4), DR6_W_NATIVE))
            if worst > 0.01:
                print(f"                 WARNING: {100 * worst:.1f}% off the "
                      f"exact healpix value.  {RES / utils.arcmin:.2f} arcmin "
                      f"pixels are not\n                 resolving the source "
                      f"holes; try a smaller --res (and a larger --dr6-nside)")
    _m_dip = mask_at(BDIR[0], BDIR[1])
    print(f"                 mask at the boost direction = {_m_dip:.3f}")
    if _m_dip < 0.05:
        print(f"                 WARNING: the dipole direction is outside the "
              f"footprint.  R will be\n                 badly conditioned "
              f"along it and A is close to meaningless")
    print(f"sims           = {N_MEANFIELD} mean field, {N_RESPONSE} response, "
          f"{N_DATA} data "
          f"({N_MEANFIELD + N_DATA + N_RESPONSE * (4 if not RESPONSE_CENTRAL else 6)}"
          f" reconstructions)")
    print(f"cache          = {CACHE_DIR}")
    if args.nshards > 1:
        print(f"shard          = {args.shard} of {args.nshards}")
    print(f"estimators     = {', '.join(ESTIMATORS)}")
    print(f"{'=' * 72}")

    mf_vecs = run_meanfield()
    diffs = run_response()
    data_vecs = run_data()

    if len(mf_vecs) < 2 or len(data_vecs) < 2 or not all(len(d) for d in diffs):
        print("\nNot all stages have results yet.  Run the remaining shards, "
              "then rerun\nwith the default --nshards 1 to collect them and "
              "print the report.")
        return

    print(f"\ncollected {len(mf_vecs)} mean-field, {len(diffs[0])} response "
          f"and {len(data_vecs)} data sims")

    per_estimator_meanfield(mf_vecs)

    results = {c: analyse(c, mf_vecs, diffs, data_vecs) for c in CASES}

    print(f"\n{'=' * 72}\n COMPARISON\n{'=' * 72}")
    base = results["T+P"]["sigma"]
    for case, r in results.items():
        pred = np.sqrt(CASE_NORM["T+P"] / CASE_NORM[case])
        print(f"sigma(A):  {case:5s} / T+P = {r['sigma'] / base:6.3f}"
              f"   (analytic forecast {1.0 / pred:6.3f})")

    sim_budget(results, len(diffs[0]))
    write_summary(results)


def write_summary(results):
    """Everything the figures need, in one small file."""
    cases = list(results)
    out = dict(cases=np.array(cases), v_true=V_TRUE_KMS, d_true=D_TRUE,
               beta=float(BETA), c_kms=float(C_KMS), ells=_ELLS, clpp=CLPP,
               w1=float(w1), w2=float(w2), w4=float(w4),
               n_mf=results[cases[0]]["n_mf"], n_resp=int(N_RESPONSE),
               n_data=results[cases[0]]["n_dat"], lmax=int(LMAX),
               lout=int(L_OUT), mask_key=np.array(MASK_KEY),
               fsky_target=float(args.fsky),
               mask_kind=np.array(args.mask),
               mask_digest=np.array(MASK_DIGEST),
               mask_file=np.array(os.path.basename(DR6_PATH)
                                  if DR6_PATH else ""),
               mask_variant=np.array(args.dr6_variant
                                     if args.mask == "dr6" else ""),
               dr6_nside=int(DR6_NSIDE),
               w_native=(DR6_W_NATIVE if DR6_W_NATIVE is not None
                         else np.full(3, np.nan)),
               mask_at_dipole=float(mask_at(BDIR[0], BDIR[1])),
               res_arcmin=float(RES / utils.arcmin))
    for k, case in enumerate(cases):
        r = results[case]
        for name, key in (("vel", "vel"), ("amp", "amp"), ("dir", "dir"),
                          ("R", "R"), ("leak", "leak"), ("leakerr", "leakerr"),
                          ("noise", "noise"), ("mfcl", "mfcl"),
                          ("datleak", "datleak")):
            out[f"{name}_{k}"] = np.asarray(r[key])
    path = os.path.join(CACHE_DIR, "summary.npz")
    np.savez(path, **out)
    print(f"\nwrote {path}")

    if not args.plots:
        return
    try:
        import aberration_plots
        for f in aberration_plots.make_all(aberration_plots.load(path),
                                           PLOT_DIR):
            print(f"wrote {f}")
    except Exception as exc:                               # noqa: BLE001
        print(f"figures skipped ({exc}); rerun with\n"
              f"   python aberration_plots.py {path} --out {PLOT_DIR}")


if __name__ == "__main__":
    main()