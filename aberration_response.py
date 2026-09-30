"""
Response-matrix techniques for the aberration dipole, compared on the same
data and mean-field sims.

aberration_act_mask.py measures the 3x3 response R one way.  This script
measures it eight ways (by default), applies each R to the same aberrated data sims with the
same mean field subtracted, and reports the amplitude A (mean and sd) and the
direction in galactic (l, b) (mean and sd of each) that each one gives:

  axis_central       +beta and -beta along x, y, z, same seed, no noise:
                       R e_j = -[q(+b e_j) - q(-b e_j)] / (2 b)
  axis_onesided      +beta along x, y, z minus the unboosted map, same seed:
                       R e_j = -[q(+b e_j) - q(0)] / b
                     This is what the pipeline itself does by default.
  axis_onesided_x2   the same at 2, 5 and 10 times beta (--rm-beta-mults),
  axis_onesided_x5   still divided by the beta actually used, so any drift
  axis_onesided_x10  with the multiplier is nonlinearity
  axis_noisy_mf      +beta along x, y, z with instrument noise and no paired
                     leg; the mean field of the unaberrated noisy sims (all
                     20000 of them) is subtracted instead
  random_central     +beta and -beta along a random direction n_i per sim
  random_onesided    +beta along the same n_i minus the unboosted map

Every design reduces to pairs (n_k, y_k) with y_k = R n_k + noise, one per
boosted leg, and R is the least-squares solution

    R = (sum_k y_k n_k^T) (sum_k n_k n_k^T)^-1.

For the axis designs n is x, y or z the same number of times each, and this is
exactly the column mean the pipeline takes, so all of them go through one fit.

Errors
------
Each mean is a function of three independent sim sets: the data sims, the mean
field and the response sims.  Their contributions are propagated to first
order and added, as the pipeline does.  The response part uses a cluster-robust
sandwich on the fit, one cluster per CMB seed (for the axis designs this is
the pipeline's own W = V @ u formula).  Every technique uses sim index i with
the same seed("response", i), and every technique is applied to the same data
and mean field, so the difference between two techniques is measured far more
precisely than either one on its own.  The error on each difference is built
from the matched per-seed contributions, and that is the number to read when
asking whether two techniques disagree.

Reuse
-----
The +beta, unboosted and noisy legs at the input beta are written under the
pipeline's own cache names (resp_clean_b..._0p_00000 and so on), so response
sims the pipeline already made are picked up here, and ones made here are
picked up by the pipeline.  Shared legs are only computed once: the +x,y,z leg
serves both axis designs, the unboosted leg serves every one-sided design, and
the +n_i leg serves both random designs.

Usage
-----
Every option not starting with --rm- goes to aberration_act_mask.py, and has
to be what the original run used, so that the imported pipeline keys to the
same cache directory.  The script checks the directory name and stops if it
does not match:

    python aberration_response_methods.py --nproc 8 \\
        <the flags of the original run> [--rm-... options]

Sharding works as in the pipeline: run with --shard k --nshards N to make the
response sims, then once with the default --nshards 1 to collect and report.

Outputs, under <cache>/plots/response_methods unless --rm-out says otherwise:

    summary.md                         the document: every technique compared
    comparison_amplitude_direction.png A and (l, b) for every technique
    comparison_response_matrix.png     R - R_ref element by element
    results.csv, results.npz           every number, for further work
    <technique>/summary.md             that technique on its own
    <technique>/amplitude_direction.png
"""

import os
import sys
import argparse


# 0. Command line
#
# This has to come before numpy is imported.  The pipeline sets the thread
# variables at import time, above its own imports, because BLAS and the SHT
# backends read them when they load; importing numpy first would lock in the
# wrong thread count for every worker.

_rm = argparse.ArgumentParser(
    prog="aberration_response_methods.py",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=__doc__.split("\nErrors\n")[0],
    epilog="Every option not listed here is handed to aberration_act_mask.py "
           "and must match\nthe original run.  See its own --help for those.",
    # Without this, a pipeline flag could be read as an abbreviation of one
    # of these and quietly swallowed.
    allow_abbrev=False)
_rm.add_argument("--rm-methods", nargs="+", default=None,
                 help="which techniques to run and compare (default: all); "
                      "names as listed above")
_rm.add_argument("--rm-n-axis", type=int, default=None,
                 help="sims for axis_central and axis_onesided "
                      "(default: the pipeline's --n-response)")
_rm.add_argument("--rm-n-mult", type=int, default=None,
                 help="sims for each axis_onesided_xK (default: --rm-n-axis)")
_rm.add_argument("--rm-n-noisy", type=int, default=None,
                 help="sims for axis_noisy_mf (default: --rm-n-axis).  Nothing "
                      "cancels within a sim here, so its error is roughly "
                      "sigma(A)/sqrt(N) per element; expect to need many more")
_rm.add_argument("--rm-n-random", type=int, default=None,
                 help="sims for the random designs (default: 3 x --rm-n-axis, "
                      "which makes random_central cost the same number of "
                      "reconstructions as axis_central)")
_rm.add_argument("--rm-beta-mults", type=float, nargs="+",
                 default=[2.0, 5.0, 10.0],
                 help="multiples of beta for axis_onesided_xK")
_rm.add_argument("--rm-noisy-beta-mult", type=float, default=1.0,
                 help="multiple of beta for axis_noisy_mf")
_rm.add_argument("--rm-dir-seed", type=int, default=20260930,
                 help="seed for the random boost directions")
_rm.add_argument("--rm-n-data", type=int, default=None,
                 help="use at most this many data sims (default: all found)")
_rm.add_argument("--rm-n-mf", type=int, default=None,
                 help="use at most this many mean-field sims (default: all)")
_rm.add_argument("--rm-expect-cache",
                 default="dr6_baseline_spline1_lmax3000_actact_97cb5711",
                 help="the cache directory the pipeline flags must key to; "
                      "'' turns the check off")
_rm.add_argument("--rm-reference", default="axis_central",
                 help="technique the others are differenced against")
_rm.add_argument("--rm-plot-case", default="T+P",
                 help="estimator combination drawn per technique, or 'all'")
_rm.add_argument("--rm-min-sims", type=int, default=5,
                 help="skip a technique with fewer complete sims than this")
_rm.add_argument("--rm-analyse-only", action="store_true",
                 help="make no new sims; analyse whatever is cached")
_rm.add_argument("--rm-out", default=None,
                 help="output folder (default <cache>/plots/response_methods)")
RM, PIPELINE_ARGV = _rm.parse_known_args()

_stray = [a for a in PIPELINE_ARGV if a.startswith("--rm")]
if _stray:
    _rm.error(f"unrecognised option(s): {' '.join(_stray)}")

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# The pipeline parses sys.argv and builds everything at import time: CAMB,
# the tempura normalisation, the filters, the mask and the cache key.  Handing
# it the pass-through flags is what guarantees the sims here are made exactly
# as the mean-field and data sims were.
sys.argv = [os.path.join(HERE, "aberration_act_mask.py")] + PIPELINE_ARGV
import aberration_act_mask as ab                                   # noqa: E402

import csv                                                         # noqa: E402
import glob                                                        # noqa: E402
import hashlib                                                     # noqa: E402
import time                                                        # noqa: E402

import numpy as np                                                 # noqa: E402
import matplotlib                                                  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402
from matplotlib.patches import Ellipse                             # noqa: E402
from matplotlib.ticker import FuncFormatter                        # noqa: E402

import aberration_plots as ap                                      # noqa: E402

COR, TRUTH, INK, RAW = ap.COR, ap.TRUTH, ap.INK, ap.RAW


# 1. Check that the pipeline keyed to the sims we mean to use

_got = os.path.basename(os.path.normpath(ab.CACHE_DIR))
if RM.rm_expect_cache and _got != RM.rm_expect_cache:
    # The import made this directory.  If it is empty it is ours and can go;
    # rmdir refuses anything with files in it, so nothing real is at risk.
    try:
        os.rmdir(ab.CACHE_DIR)
    except OSError:
        pass
    raise SystemExit(
        f"\nThe pipeline flags key to\n    {ab.CACHE_DIR}\nnot to "
        f"{RM.rm_expect_cache}.\nPass exactly the flags the original run used "
        f"(mask, noise model, lmin, lmax,\nres, beam, --cache, ...), or "
        f"--rm-expect-cache '' to use this directory anyway.")

WORK = os.path.join(ab.CACHE_DIR, "response_methods")
OUT = RM.rm_out or os.path.join(ab.CACHE_DIR, "plots", "response_methods")
os.makedirs(WORK, exist_ok=True)


# 2. Constants and small helpers

BETA = float(ab.BETA)
C_KMS = float(ab.C_KMS)
D_TRUE = np.asarray(ab.D_TRUE, dtype=float)
A_TRUE = np.asarray(ab.A_TRUE, dtype=float)
G_AMP = A_TRUE / float(A_TRUE @ A_TRUE)          # A = G_AMP . a_hat
EST = list(ab.ESTIMATORS)
CASES = list(ab.CASES)
NPACK = int(ab._NPACK)
E3 = np.eye(3)
EQU2GAL = np.asarray(ap.EQU2GAL, dtype=float)


def wrap180(x):
    return (np.asarray(x, dtype=float) + 180.0) % 360.0 - 180.0


def lb_of(v):
    """Galactic (l, b) in degrees of vectors stacked on the last axis."""
    v = np.asarray(v, dtype=float)
    l = np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360.0
    b = np.degrees(np.arctan2(v[..., 2], np.hypot(v[..., 0], v[..., 1])))
    return l, b


def lb_jacobian(v):
    """d(l, b)/dv in degrees per unit of v, for one galactic vector v."""
    x, y, z = np.asarray(v, dtype=float)
    rho2 = x * x + y * y
    rho = np.sqrt(rho2)
    r2 = rho2 + z * z
    dl = np.array([-y / rho2, x / rho2, 0.0])
    db = np.array([-x * z / (rho * r2), -y * z / (rho * r2), rho / r2])
    return np.degrees(np.array([dl, db]))


def radec_of(n):
    """(ra, dec) in radians, the form pixell's Aberrator takes for dir."""
    n = np.asarray(n, dtype=float) / np.linalg.norm(n)
    return (float(np.arctan2(n[1], n[0]) % (2 * np.pi)),
            float(np.arcsin(np.clip(n[2], -1.0, 1.0))))


L_TRUE, B_TRUE = (float(x) for x in lb_of(EQU2GAL @ D_TRUE))


def unwrap_l(l):
    """Longitudes put within 180 deg of the input, so a scatter straddling
    l = 0 is not split across the axis and its sd is not inflated by 360."""
    return L_TRUE + wrap180(np.asarray(l) - L_TRUE)


def l1_rows(P):
    """The pipeline's l1_vector, vectorised over leading axes.

    Checked against ab.l1_vector below rather than trusted, since the two
    have to agree to the last digit for the mean field and the data to be
    handled the same way as the response legs.
    """
    a10 = P[..., 1].real
    a11 = P[..., 2]
    return np.stack([-np.sqrt(3.0 / (2.0 * np.pi)) * a11.real,
                     np.sqrt(3.0 / (2.0 * np.pi)) * a11.imag,
                     np.sqrt(3.0 / (4.0 * np.pi)) * a10], axis=-1)


def case_l1(stack, case):
    """Normalised L=1 vectors of one case, for a whole stack of sims."""
    tot = sum(stack[e] for e in ab.CASES[case])
    return l1_rows(ab.NORM_PACK[case] * tot)


# 3. Techniques

N_AXIS = RM.rm_n_axis or int(ab.N_RESPONSE)
N_MULT = RM.rm_n_mult or N_AXIS
N_NOISY = RM.rm_n_noisy or N_AXIS
N_RAND = RM.rm_n_random or 3 * N_AXIS
DIR_SEED = int(RM.rm_dir_seed)

# 1 is axis_onesided itself, and a repeated multiple would only duplicate.
MULTS = []
for _k in RM.rm_beta_mults:
    if _k > 0 and _k != 1.0 and _k not in MULTS:
        MULTS.append(float(_k))


def _technique_table():
    b = BETA
    T = {}
    T["axis_central"] = dict(
        label="±x,y,z central", kind="axis", mode="central", beta=b,
        noisy=False, n=N_AXIS, legs=6,
        boost="+β and −β along x, y, z; no noise",
        subtract="the −β leg of the same seed, then ÷2",
        desc="Each sim is boosted by +β and by −β along each of x, y and z, "
             "from one CMB seed and without noise.  Column j of R is "
             "−[q(+β e_j) − q(−β e_j)] / (2β).  Everything that does not "
             "depend on the boost (the mean field, the realisation's own "
             "reconstruction noise) cancels inside the pair, and so does "
             "every even power of β, so the relative bias from the "
             "nonlinearity of the boost is O(β²).  Six reconstructions per "
             "sim.  This is the reference the others are differenced "
             "against by default.")
    T["axis_onesided"] = dict(
        label="+x,y,z − unboosted", kind="axis", mode="onesided", beta=b,
        noisy=False, n=N_AXIS, legs=4,
        boost="+β along x, y, z; no noise",
        subtract="the unboosted map of the same seed",
        desc="Each sim is boosted by +β along x, y and z and the unboosted "
             "map from the same seed is subtracted: column j of R is "
             "−[q(+β e_j) − q(0)] / β.  The mean field and the realisation "
             "noise cancel exactly as in the central difference, but the β² "
             "term stays in the numerator, so the relative bias is O(β).  "
             "Four reconstructions per sim.  This is what "
             "aberration_act_mask.py does by default, and at the same N it "
             "reproduces the pipeline's R exactly.")
    for k in MULTS:
        T[f"axis_onesided_x{k:g}"] = dict(
            label=f"+x,y,z ×{k:g}β − unboosted", kind="axis",
            mode="onesided", beta=k * b, noisy=False, n=N_MULT, legs=4,
            mult=k,
            boost=f"+{k:g}β along x, y, z; no noise",
            subtract="the unboosted map of the same seed",
            desc=f"As axis_onesided, with the boost {k:g} times the input β "
                 f"and the difference divided by the β actually used.  In "
                 f"the linear regime this gives the same R; the O(β) bias "
                 f"of a one-sided difference grows in proportion to the "
                 f"multiplier, so the drift of R and of A across 1, "
                 f"{', '.join(f'{m:g}' for m in MULTS)} measures the "
                 f"nonlinearity of the response directly.  The paired "
                 f"scatter is itself linear in β, so a bigger boost does not "
                 f"buy a smaller relative Monte Carlo error.")
    T["axis_noisy_mf"] = dict(
        label="+x,y,z noisy − MF", kind="axis", mode="mf",
        beta=RM.rm_noisy_beta_mult * b, noisy=True, n=N_NOISY, legs=3,
        mult=RM.rm_noisy_beta_mult,
        boost=f"+{RM.rm_noisy_beta_mult:g}β along x, y, z; with noise",
        subtract="the mean field of the unaberrated noisy sims",
        desc="Each sim is boosted along x, y and z with instrument noise "
             "added, as the data are, and no unboosted partner is made.  The "
             "mean field measured from the unaberrated noisy sims is "
             "subtracted instead: column j of R is −[q(+β e_j) − MF] / β.  "
             "Nothing cancels realisation by realisation, so every element "
             "carries the full per-sim reconstruction scatter, about "
             "σ(A)/√N; at equal N this is by far the noisiest technique.  "
             "The same mean field is also what is subtracted from the data, "
             "which couples the two errors; that coupling is included.")
    T["random_central"] = dict(
        label="±random central", kind="random", mode="central", beta=b,
        noisy=False, n=N_RAND, legs=2,
        boost="+β and −β along a random direction per sim; no noise",
        subtract="the −β leg of the same seed, then ÷2",
        desc="Each sim gets its own direction n_i, uniform on the sphere, "
             "and is boosted by +β and −β along it: "
             "y_i = −[q(+β n_i) − q(−β n_i)] / (2β) = R n_i.  R is the "
             "least-squares solution over all sims.  Two reconstructions per "
             "sim instead of six, but each sim constrains R along one "
             "direction only; by default it gets three times the sims of "
             "axis_central, which is the same number of reconstructions.")
    T["random_onesided"] = dict(
        label="+random − unboosted", kind="random", mode="onesided", beta=b,
        noisy=False, n=N_RAND, legs=2,
        boost="+β along the same random directions; no noise",
        subtract="the unboosted map of the same seed",
        desc="The same random directions, boosted by +β only, with the "
             "unboosted map of the same seed subtracted: "
             "y_i = −[q(+β n_i) − q(0)] / β, then least squares.  The "
             "one-sided counterpart of random_central, with the same O(β) "
             "bias as axis_onesided.")
    return T


TECH = _technique_table()

if RM.rm_methods:
    _bad = [m for m in RM.rm_methods if m not in TECH]
    if _bad:
        raise SystemExit(f"unknown technique(s) {_bad}; choose from "
                         f"{', '.join(TECH)}")
    SELECTED = [m for m in TECH if m in RM.rm_methods]
else:
    SELECTED = list(TECH)

if RM.rm_plot_case != "all" and RM.rm_plot_case not in CASES:
    raise SystemExit(f"--rm-plot-case {RM.rm_plot_case!r}: choose from "
                     f"{', '.join(CASES)} or all")
PLOT_CASES = CASES if RM.rm_plot_case == "all" else [RM.rm_plot_case]
# The headline case of the document is the one drawn, so that every figure it
# links to exists; with 'all' that is T+P.
if RM.rm_plot_case != "all":
    MAIN_CASE = RM.rm_plot_case
else:
    MAIN_CASE = "T+P" if "T+P" in CASES else CASES[-1]


# 4. Cache names
#
# The legs at the input beta use the pipeline's own names, so the two codes
# share them.  The pipeline tags a response file with
#     resp_{clean|noisy}_b{beta:g}[_ctr]_{leg}_{i:05d}
# where _ctr appears whenever beta is not the input beta, and the unboosted
# leg is the same map whatever beta is, so one name serves every multiple.

def _btag(b):
    return f"{b:g}"


def tag_off(i):
    return f"resp_clean_b{_btag(BETA)}_off_{i:05d}"


def tag_axis(j, sgn, beta, noisy, i):
    suffix = ("noisy" if noisy else "clean") + f"_b{_btag(beta)}"
    if beta != BETA:
        suffix += "_ctr"
    return f"resp_{suffix}_{j}{'p' if sgn > 0 else 'm'}_{i:05d}"


def tag_rand(sgn, beta, i):
    return (f"rmrand_clean_b{_btag(beta)}_d{DIR_SEED}_"
            f"{'p' if sgn > 0 else 'm'}_{i:05d}")


def random_direction(i):
    """Uniform on the sphere, reproducible per sim and independent of N."""
    rng = np.random.default_rng([DIR_SEED, int(i)])
    z = rng.uniform(-1.0, 1.0)
    phi = rng.uniform(0.0, 2.0 * np.pi)
    s = np.sqrt(1.0 - z * z)
    return np.array([s * np.cos(phi), s * np.sin(phi), z])


# 5. Making the response sims
#
# A leg is (tag, sim, dir as (ra, dec) or None, signed beta, noisy, the unit
# direction to store with it or None, aberrator key).  Legs are grouped so
# that every axis leg shares one Aberrator per worker, as the pipeline does;
# the random legs each need their own.

def _leg_axis(j, sgn, beta, noisy, i):
    return (tag_axis(j, sgn, beta, noisy, i), i, ab.RESPONSE_DIRS[j],
            sgn * beta, noisy, None, ("rm_axis", j, sgn * beta))


def _leg_off(i):
    return (tag_off(i), i, None, 0.0, False, None, None)


def _leg_rand(sgn, beta, i):
    n = random_direction(i)
    return (tag_rand(sgn, beta, i), i, radec_of(n), sgn * beta, False,
            tuple(n), ("rm_rand", i, sgn))


def legs_for(t, i):
    out = []
    if t["kind"] == "axis":
        for j in range(3):
            out.append(_leg_axis(j, +1, t["beta"], t["noisy"], i))
            if t["mode"] == "central":
                out.append(_leg_axis(j, -1, t["beta"], t["noisy"], i))
    else:
        out.append(_leg_rand(+1, t["beta"], i))
        if t["mode"] == "central":
            out.append(_leg_rand(-1, t["beta"], i))
    if t["mode"] == "onesided":
        out.append(_leg_off(i))
    return out


def _task_leg(item):
    """One reconstruction.  Module level, so the pool can pickle it."""
    tag, i, radec, beta, noisy, store_dir, key = item
    aber = None
    if radec is not None:
        # The sign rides on beta with dir fixed: a negative beta is exactly
        # the boost along -dir, the same convention the pipeline uses.
        aber = ab.aberrator_for(key, radec, beta)
    s_sig, s_noi = ab.seeds("response", i)
    tqu = ab.cmb_map(s_sig)
    if aber is not None:
        tqu = aber(tqu)
    if noisy:
        tqu = tqu + ab.noise_map(s_noi)
    out = ab.reconstruct(tqu)
    del tqu
    if store_dir is not None:
        out["dir"] = np.asarray(store_dir, dtype=float)
    ab.cache_put(tag, out)
    return i


def _stage_label(leg):
    tag, i, radec, beta, noisy, store_dir, key = leg
    if radec is None:
        return (0, "unboosted")
    if key[0] == "rm_axis":
        j, b = key[1], key[2]
        return (1, f"{'xyz'[j]}{'+' if b > 0 else '-'} "
                   f"{abs(b) / BETA:g}beta {'noisy' if noisy else 'clean'}")
    return (2, "random " + ("+" if beta > 0 else "-"))


def make_sims():
    legs = {}
    for name in SELECTED:
        t = TECH[name]
        for i in range(t["n"]):
            for leg in legs_for(t, i):
                legs.setdefault(leg[0], leg)
    groups = {}
    for leg in legs.values():
        groups.setdefault(_stage_label(leg), []).append(leg)

    total = sum(1 for lg in legs.values() if ab.mine(lg[1]))
    todo_all = sum(1 for lg in legs.values()
                   if ab.mine(lg[1]) and not ab.cache_has(lg[0]))
    print(f"\n[R] response sims for {len(SELECTED)} techniques: {total} "
          f"distinct reconstructions, {total - todo_all} already cached, "
          f"{todo_all} to run", flush=True)
    for key in sorted(groups, key=lambda k: (k[0], k[1])):
        todo = sorted((lg for lg in groups[key]
                       if ab.mine(lg[1]) and not ab.cache_has(lg[0])),
                      key=lambda lg: lg[1])
        if todo:
            print(f"   {key[1]}: {len(todo)} to run", flush=True)
            ab.run_tasks(todo, _task_leg, key[1])
    ab.close_pool()


# 6. Loading the data and mean-field sims
#
# 25000 small files take a while on a shared filesystem, so the stacked
# arrays are kept in one file, named by the exact set of indices it holds.

def sim_indices(prefix):
    pat = os.path.join(ab.CACHE_DIR, f"{prefix}_" + "[0-9]" * 5 + ".npz")
    return sorted(int(os.path.basename(p)[len(prefix) + 1:-4])
                  for p in glob.glob(pat))


def load_stack(prefix, cap):
    idx = sim_indices(prefix)
    if cap:
        idx = idx[:cap]
    if not idx:
        return None, []
    key = hashlib.md5(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()
    path = os.path.join(WORK, f"stack_{prefix}_{len(idx)}_{key[:10]}.npz")
    if os.path.exists(path):
        with np.load(path) as f:
            return {e: f[e] for e in EST}, idx
    print(f"   stacking {len(idx)} {prefix} sims (once; kept in {path})",
          flush=True)
    t0 = time.time()
    out = {e: np.empty((len(idx), NPACK), dtype=np.complex128) for e in EST}
    for a, i in enumerate(idx):
        v = ab.cache_get(f"{prefix}_{i:05d}")
        for e in EST:
            out[e][a] = v[e]
        if (a + 1) % 2000 == 0:
            print(f"      {a + 1}/{len(idx)}  {time.time() - t0:.0f} s",
                  flush=True)
    tmp = f"{path}.tmp{os.getpid()}.npz"
    np.savez(tmp, **out)
    os.replace(tmp, path)
    return out, idx


# 7. Measurements and the fit

_MEMO = {}


def get(tag):
    if tag not in _MEMO:
        _MEMO[tag] = ab.cache_get(tag)
    return _MEMO[tag]


def measurements(t, mf_alm):
    """Every (n_k, y_k, sim) of one technique, complete sims only.

    A sim counts only if every leg it needs is present, so that sims from an
    unfinished shard never enter half-measured.
    """
    ns, cl, diffs = [], [], []
    bad_dir = 0
    for i in range(t["n"]):
        rows = []
        if t["kind"] == "axis":
            for j in range(3):
                plus = get(tag_axis(j, +1, t["beta"], t["noisy"], i))
                if t["mode"] == "central":
                    other = get(tag_axis(j, -1, t["beta"], t["noisy"], i))
                elif t["mode"] == "onesided":
                    other = get(tag_off(i))
                else:
                    other = mf_alm
                rows.append((E3[j], plus, other))
        else:
            plus = get(tag_rand(+1, t["beta"], i))
            if t["mode"] == "central":
                other = get(tag_rand(-1, t["beta"], i))
            else:
                other = get(tag_off(i))
            n = None
            if plus is not None:
                # The direction the sim was actually boosted along, as
                # stored with it; the regenerated one is only a check.
                n = np.asarray(plus["dir"], dtype=float)
                if not np.allclose(n, random_direction(i), atol=1e-12):
                    bad_dir += 1
            rows.append((n, plus, other))
        if any(p is None or o is None for _, p, o in rows):
            continue
        denom = 2.0 if t["mode"] == "central" else 1.0
        for n, p, o in rows:
            ns.append(n)
            cl.append(i)
            diffs.append({e: (p[e] - o[e]) / denom for e in EST})
    if bad_dir:
        print(f"   WARNING: {bad_dir} random sims were boosted along a "
              f"direction other than the one --rm-dir-seed gives now; the "
              f"stored direction is used.", flush=True)
    if not diffs:
        return None
    y = {case: np.array([ab.coadd(d, case) for d in diffs]) / (-t["beta"])
         for case in CASES}
    return dict(n=np.array(ns), cluster=np.array(cl), y=y,
                n_sims=len(set(cl)))


def fit_R(meas, case):
    """Least-squares R from y_k = R n_k, with what the errors need."""
    N = meas["n"]
    Y = meas["y"][case]
    Minv = np.linalg.inv(N.T @ N)
    R = (Y.T @ N) @ Minv
    E = Y - N @ R.T                        # residuals, one row per leg
    uc, inv = np.unique(meas["cluster"], return_inverse=True)
    K, C = len(N), len(uc)
    # CR1, the usual small-sample factor for a clustered sandwich.
    fac = C / max(C - 1, 1) * (K - 1) / max(K - 3, 1)
    return dict(R=R, Minv=Minv, E=E, N=N, uc=uc, inv=inv, fac=fac, K=K, C=C)


def cluster_terms(fit, u):
    """Per-seed contributions to the error on R u, one row per seed.

    R_hat - R = sum_k eps_k n_k^T M^-1, so (R_hat - R) u is a sum over legs
    of eps_k (n_k^T M^-1 u).  Legs of one seed share a CMB realisation and
    are correlated, so they are summed within the seed before squaring.
    """
    w = fit["N"] @ (fit["Minv"] @ u)
    G = np.zeros((fit["C"], 3))
    np.add.at(G, fit["inv"], fit["E"] * w[:, None])
    return G


# 8. Applying one R to the data

def analyse_one(name, t, meas, case, dat_l1, mf_l1, mf_cov):
    fit = fit_R(meas, case)
    R = fit["R"]
    Rinv = np.linalg.inv(R)
    D = dat_l1[case]
    MF = mf_l1[case]
    n_dat, n_mf = len(D), len(MF)
    mf = MF.mean(axis=0)

    ahat = (D - mf) @ Rinv.T
    u = ahat.mean(axis=0)
    amp = ahat @ G_AMP

    # Per-sim directions.  The boost points along -a, as in the pipeline.
    dirs = -ahat / np.linalg.norm(ahat, axis=1)[:, None]
    l_raw, b_sims = lb_of(dirs @ EQU2GAL.T)
    l_sims = unwrap_l(l_raw)
    off = np.degrees(np.arccos(np.clip(dirs @ D_TRUE, -1.0, 1.0)))

    # The direction of the mean vector, and the gradients that carry the
    # errors on the mean vector into (l, b).
    vg = -(EQU2GAL @ u)
    lbar, bbar = (float(x) for x in lb_of(vg))
    J = lb_jacobian(vg) @ (-EQU2GAL)
    F = {"A": G_AMP, "l": J[0], "b": J[1]}

    # Three independent sources of error on the mean a_hat.
    #   data:   the scatter of the data sims about their mean.
    #   MF:     a_hat = R^-1 (v - mf) moves by -R^-1 dmf.  For axis_noisy_mf
    #           the same dmf is inside R as well, dR u = s dmf, which makes it
    #           -(1 + s) R^-1 dmf.
    #   resp:   R -> R + dR moves a_hat by -R^-1 dR u.
    cov_data = np.cov(ahat, rowvar=False) / n_dat
    s = 0.0
    if t["mode"] == "mf":
        s = float(fit["N"].sum(axis=0) @ fit["Minv"] @ u) / t["beta"]
    Kmf = -(1.0 + s) * Rinv
    cov_mf = Kmf @ mf_cov[case] @ Kmf.T / n_mf
    H = -cluster_terms(fit, u) @ Rinv.T
    cov_resp = fit["fac"] * H.T @ H

    err = {}
    for k, f in F.items():
        e = {"data": float(np.sqrt(max(f @ cov_data @ f, 0.0))),
             "mf": float(np.sqrt(max(f @ cov_mf @ f, 0.0))),
             "resp": float(np.sqrt(max(f @ cov_resp @ f, 0.0)))}
        e["total"] = float(np.sqrt(e["data"] ** 2 + e["mf"] ** 2
                                   + e["resp"] ** 2))
        err[k] = e

    # Error on each element of R, and the per-seed pieces for differences.
    Gcol = [cluster_terms(fit, E3[j]) for j in range(3)]
    Rerr_resp = np.stack([np.sqrt(fit["fac"] * (G ** 2).sum(axis=0))
                          for G in Gcol], axis=1)
    Rerr_mf = np.zeros((3, 3))
    if t["mode"] == "mf":
        w = fit["N"].sum(axis=0) @ fit["Minv"] / t["beta"]
        sd_mf = np.sqrt(np.diag(mf_cov[case]) / n_mf)
        Rerr_mf = np.outer(sd_mf, np.abs(w))
    Rerr = np.sqrt(Rerr_resp ** 2 + Rerr_mf ** 2)

    evals, evecs = np.linalg.eigh(0.5 * (R + R.T))
    return dict(
        name=name, case=case, R=R, Rerr=Rerr, Rerr_mf=Rerr_mf,
        cond=float(np.linalg.cond(R)), evals=evals[::-1],
        evecs=evecs[:, ::-1], n_sims=meas["n_sims"], n_legs=fit["K"],
        n_dat=n_dat, n_mf=n_mf, recons=t["legs"] * meas["n_sims"],
        ahat=ahat, amp=amp, A_mean=float(amp.mean()),
        A_sd=float(amp.std(ddof=1)), A_err=err["A"],
        dirs=dirs, l_sims=l_sims, b_sims=b_sims, off=off,
        l_mean=float(l_sims.mean()), l_sd=float(l_sims.std(ddof=1)),
        b_mean=float(b_sims.mean()), b_sd=float(b_sims.std(ddof=1)),
        l_pct=np.percentile(l_sims, [16, 50, 84]),
        b_pct=np.percentile(b_sims, [16, 50, 84]),
        lbar=lbar, bbar=bbar, l_err=err["l"], b_err=err["b"],
        off_bar=float(np.degrees(np.arccos(np.clip(
            (vg / np.linalg.norm(vg)) @ (EQU2GAL @ D_TRUE), -1, 1)))),
        vbar_kms=float(np.linalg.norm(u) * C_KMS),
        F=F, H=H, uc=fit["uc"], fac=fit["fac"], Kmf=Kmf, Gcol=Gcol,
        mf_cov=mf_cov[case], s_mf=s)


def difference(rT, rR, key):
    """Value and error of (technique - reference) for A, l or b.

    Both use the same data and mean-field sims, and their response sims share
    CMB seeds, so each source is differenced sim by sim before squaring.
    """
    if key == "A":
        val = rT["A_mean"] - rR["A_mean"]
    elif key == "l":
        val = float(wrap180(rT["lbar"] - rR["lbar"]))
    else:
        val = rT["bbar"] - rR["bbar"]
    if rT is rR:
        return 0.0, 0.0
    fT, fR = rT["F"][key], rR["F"][key]
    qT = dict(zip(rT["uc"].tolist(), (rT["H"] @ fT).tolist()))
    qR = dict(zip(rR["uc"].tolist(), (rR["H"] @ fR).tolist()))
    v_resp = max(rT["fac"], rR["fac"]) * sum(
        (qT.get(c, 0.0) - qR.get(c, 0.0)) ** 2 for c in set(qT) | set(qR))
    per = rT["ahat"] @ fT - rR["ahat"] @ fR
    v_data = per.var(ddof=1) / len(per)
    k = fT @ rT["Kmf"] - fR @ rR["Kmf"]
    v_mf = float(k @ rT["mf_cov"] @ k) / rT["n_mf"]
    return val, float(np.sqrt(v_resp + v_data + v_mf))


def difference_R(rT, rR):
    """Error on R_T - R_ref element by element, matched seed by seed."""
    if rT is rR:
        return np.zeros((3, 3))
    fac = max(rT["fac"], rR["fac"])
    out = np.zeros((3, 3))
    for j in range(3):
        dT = dict(zip(rT["uc"].tolist(), rT["Gcol"][j]))
        dR = dict(zip(rR["uc"].tolist(), rR["Gcol"][j]))
        z = np.zeros(3)
        tot = sum(((dT.get(c, z) - dR.get(c, z)) ** 2
                   for c in set(dT) | set(dR)), z)
        out[:, j] = np.sqrt(fac * tot)
    return np.sqrt(out ** 2 + rT["Rerr_mf"] ** 2 + rR["Rerr_mf"] ** 2)


# 9. Figures

def _ellipse(ax, xy, cov, nsig, **kw):
    val, vec = np.linalg.eigh(cov)
    val = np.clip(val, 0.0, None)
    ang = np.degrees(np.arctan2(vec[1, -1], vec[0, -1]))
    ax.add_patch(Ellipse(xy, 2 * nsig * np.sqrt(val[-1]),
                         2 * nsig * np.sqrt(val[0]), angle=ang, **kw))


def _mark_mean_sd(ax, m, sd, color):
    ax.axvspan(m - sd, m + sd, color=color, alpha=0.10, lw=0, zorder=0)
    for k in (1, 2):
        for sgn in (-1, 1):
            ax.axvline(m + sgn * k * sd, color=color, lw=1.0,
                       ls=(0, (2, 3)), alpha=0.8, zorder=1)
    ax.axvline(m, color=color, lw=1.2, ls=(0, (3, 2)))


def _bins(x, truth, n=40, lim=(-np.inf, np.inf)):
    lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
    m, sd = x.mean(), x.std(ddof=1)
    lo, hi = min(lo, m - 2.2 * sd), max(hi, m + 2.2 * sd)
    pad = 0.1 * (hi - lo + 1e-9)
    return np.linspace(max(min(lo - pad, truth - pad), lim[0]),
                       min(max(hi + pad, truth + pad), lim[1]), n)


# l is kept unwrapped about the input on the axes, so a scatter across l = 0
# is drawn in one piece; only the labels are taken back into [0, 360).
_L_TICKS = FuncFormatter(lambda v, _: f"{v % 360.0:.0f}")


def plot_amplitude_direction(r, t, path):
    """Amplitude histogram, the recovered directions in (l, b), and the
    marginal distributions of l and b, for one technique and one case.

    The amplitude panel is drawn as the pipeline's amplitude_direction is.
    The direction is shown in galactic coordinates with l increasing to the
    left, as on a sky map, and l is unwrapped about the input so a scatter
    across l = 0 stays in one piece.
    """
    ap._style()
    amp, l, b = r["amp"], r["l_sims"], r["b_sims"]
    counts = (f"$N$: {r['n_dat']} data, {r['n_mf']} mean field, "
              f"{r['n_sims']} response ({r['recons']} recons)")

    fig, axs = plt.subplots(2, 2, figsize=(12.0, 9.2),
                            gridspec_kw=dict(hspace=0.30, wspace=0.22))

    ax = axs[0, 0]
    bins = _bins(amp, 1.0, 34)
    ax.hist(amp, bins=bins, histtype="stepfilled", color=COR, alpha=0.3)
    ax.hist(amp, bins=bins, histtype="step", lw=1.8, color=COR,
            label=f"$R^{{-1}}$ corrected:  {r['A_mean']:+.4f} $\\pm$ "
                  f"{r['A_sd']:.4f} (sd)\nerror on the mean "
                  f"{r['A_err']['total']:.4f}  (response "
                  f"{r['A_err']['resp']:.4f})\n{counts}")
    _mark_mean_sd(ax, r["A_mean"], r["A_sd"], COR)
    ax.axvline(1.0, color=TRUTH, lw=1.8, label="input")
    ax.set_xlabel("amplitude $A$")
    ax.set_ylabel("sims")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.45)
    ax.legend(fontsize=8.8, loc="upper right", frameon=True,
              framealpha=0.95, edgecolor="0.8")

    ax = axs[0, 1]
    ax.scatter(l, b, s=4, color=COR, alpha=0.22, lw=0, rasterized=True)
    lb = np.column_stack([l, b])
    if len(lb) > 2:
        cov = np.cov(lb, rowvar=False)
        for k in (1, 2):
            _ellipse(ax, lb.mean(axis=0), cov, k, fill=False,
                     edgecolor=COR, lw=1.3, ls=(0, (5, 3)), zorder=4)
    ax.plot(L_TRUE, B_TRUE, "*", ms=18, color=TRUTH, mec="white", mew=0.8,
            zorder=7)
    ax.plot(unwrap_l(r["lbar"]), r["bbar"], "X", ms=10, color=INK,
            mec="white", mew=0.8, zorder=8)
    lx = np.percentile(l, [1, 99])
    by = np.percentile(b, [1, 99])
    px = 0.12 * (lx[1] - lx[0] + 1e-6)
    py = 0.12 * (by[1] - by[0] + 1e-6)
    ax.set_xlim(max(lx[1], L_TRUE) + px, min(lx[0], L_TRUE) - px)
    ax.set_ylim(max(-90.0, min(by[0], B_TRUE) - py),
                min(90.0, max(by[1], B_TRUE) + py))
    ax.xaxis.set_major_formatter(_L_TICKS)
    ax.set_xlabel("galactic longitude $l$  [deg], increasing to the left")
    ax.set_ylabel("galactic latitude $b$  [deg]")
    ax.set_title(f"per sim:  $l$ = {r['l_mean'] % 360:.2f} $\\pm$ "
                 f"{r['l_sd']:.2f}$^\\circ$,   $b$ = {r['b_mean']:+.2f} "
                 f"$\\pm$ {r['b_sd']:.2f}$^\\circ$  (sd)", fontsize=10.5)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="none", color=COR, alpha=0.7, ms=5,
               label="data sims"),
        Line2D([], [], color=COR, lw=1.3, ls=(0, (5, 3)), label="1, 2 sd"),
        Line2D([], [], marker="X", ls="none", color=INK, ms=8,
               label=f"mean vector: ({r['lbar']:.2f}$^\\circ$, "
                     f"{r['bbar']:+.2f}$^\\circ$) $\\pm$ "
                     f"({r['l_err']['total']:.2f}, "
                     f"{r['b_err']['total']:.2f})"),
        Line2D([], [], marker="*", ls="none", color=TRUTH, ms=12,
               label=f"input ({L_TRUE:.2f}$^\\circ$, {B_TRUE:+.2f}$^\\circ$)")],
        fontsize=8.4, loc="lower left", frameon=True, framealpha=0.9,
        edgecolor="0.8")

    for ax, x, truth, m, sd, name, lim, pct in (
            (axs[1, 0], l, L_TRUE, r["l_mean"], r["l_sd"], "l",
             (-np.inf, np.inf), r["l_pct"]),
            (axs[1, 1], b, B_TRUE, r["b_mean"], r["b_sd"], "b",
             (-90.0, 90.0), r["b_pct"])):
        bins = _bins(x, truth, 40, lim)
        ax.hist(x, bins=bins, histtype="stepfilled", color=COR, alpha=0.3)
        ax.hist(x, bins=bins, histtype="step", lw=1.8, color=COR,
                label=f"${name}$ = {m % 360 if name == 'l' else m:.2f} "
                      f"$\\pm$ {sd:.2f}$^\\circ$ (sd)\n"
                      f"16$-$84%: {pct[0] % 360 if name == 'l' else pct[0]:.1f}"
                      f" to {pct[2] % 360 if name == 'l' else pct[2]:.1f}"
                      f"$^\\circ$,  median "
                      f"{pct[1] % 360 if name == 'l' else pct[1]:.1f}$^\\circ$\n"
                      f"error on the mean {sd / np.sqrt(len(x)):.2f}"
                      f"$^\\circ$ (data)")
        _mark_mean_sd(ax, m, sd, COR)
        ax.axvline(truth, color=TRUTH, lw=1.8,
                   label=f"input {truth:.2f}$^\\circ$")
        ax.set_xlabel(f"${name}$  [deg]")
        ax.set_ylabel("sims")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.35)
        ax.legend(fontsize=8.8, loc="upper right", frameon=True,
                  framealpha=0.95, edgecolor="0.8")
    axs[1, 0].invert_xaxis()
    axs[1, 0].xaxis.set_major_formatter(_L_TICKS)

    fig.suptitle(f"{r['name']}:  {t['label']}   ({r['case']})",
                 fontsize=13.5, y=0.955)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_comparison(res, names, case, ref, path):
    """A, and the mean-vector l and b, for every technique side by side."""
    ap._style()
    x = np.arange(len(names))
    rr = [res[(n, case)] for n in names]
    fig, axs = plt.subplots(2, 2, figsize=(max(10.0, 1.25 * len(names) + 4),
                                           8.4), sharex=True,
                            gridspec_kw=dict(hspace=0.12, wspace=0.25))

    def bars(ax, vals, tot, resp, truth, ylabel):
        ax.errorbar(x, vals, yerr=tot, fmt="none", ecolor=COR, elinewidth=1.1,
                    capsize=4, zorder=4)
        ax.errorbar(x, vals, yerr=resp, fmt="o", color=COR, mec=INK,
                    mew=0.7, ms=6, ecolor=RAW, elinewidth=4.5, alpha=0.9,
                    zorder=5)
        if truth is not None:
            ax.axhline(truth, color=TRUTH, lw=1.6, zorder=3)
        ax.set_ylabel(ylabel)

    bars(axs[0, 0], [r["A_mean"] for r in rr],
         [r["A_err"]["total"] for r in rr], [r["A_err"]["resp"] for r in rr],
         1.0, "amplitude $\\bar A$")
    bars(axs[1, 0], [r["lbar"] for r in rr],
         [r["l_err"]["total"] for r in rr], [r["l_err"]["resp"] for r in rr],
         L_TRUE, "mean-vector $\\bar l$  [deg]")
    bars(axs[1, 1], [r["bbar"] for r in rr],
         [r["b_err"]["total"] for r in rr], [r["b_err"]["resp"] for r in rr],
         B_TRUE, "mean-vector $\\bar b$  [deg]")

    ax = axs[0, 1]
    rref = res[(ref, case)]
    d = [difference(r, rref, "A") for r in rr]
    ax.errorbar(x, [v for v, _ in d], yerr=[e for _, e in d], fmt="o",
                color=COR, mec=INK, mew=0.7, ms=6, ecolor=COR, elinewidth=1.6,
                capsize=4, zorder=5)
    ax.axhline(0.0, color=TRUTH, lw=1.6, zorder=3)
    ax.set_ylabel("$\\bar A - \\bar A_{\\rm ref}$")

    for ax in axs[1]:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    axs[0, 0].legend(handles=[
        Line2D([], [], color=RAW, lw=4.5, label="response sims only"),
        Line2D([], [], color=COR, lw=1.1,
               label="total: data + mean field + response"),
        Line2D([], [], color=TRUTH, lw=1.6, label="input")],
        fontsize=8.8, loc="best", frameon=True, framealpha=0.95,
        edgecolor="0.8")
    axs[0, 1].set_title(f"minus the reference ({ref}); errors matched "
                        f"seed by seed", fontsize=9.5)
    fig.suptitle(f"response-matrix techniques ({case}); bars are errors on "
                 f"the mean", fontsize=12.5, y=0.95)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_R_comparison(res, names, case, ref, path):
    """R_T - R_ref for each of the nine elements, with the matched error."""
    ap._style()
    x = np.arange(len(names))
    rref = res[(ref, case)]
    fig, axs = plt.subplots(3, 3, figsize=(max(11.0, 1.1 * len(names) + 5),
                                           9.5), sharex=True,
                            gridspec_kw=dict(hspace=0.35, wspace=0.35))
    diffs = [(res[(n, case)]["R"] - rref["R"],
              difference_R(res[(n, case)], rref)) for n in names]
    for i in range(3):
        for j in range(3):
            ax = axs[i, j]
            ax.errorbar(x, [d[0][i, j] for d in diffs],
                        yerr=[d[1][i, j] for d in diffs], fmt="o", color=COR,
                        mec=INK, mew=0.6, ms=5, ecolor=COR, elinewidth=1.4,
                        capsize=3)
            ax.axhline(0.0, color=TRUTH, lw=1.2)
            ax.set_title(f"$R_{{{'xyz'[i]}{'xyz'[j]}}}$ = "
                         f"{rref['R'][i, j]:+.4f} $\\pm$ "
                         f"{rref['Rerr'][i, j]:.4f}", fontsize=9.5)
            ax.tick_params(labelsize=8)
    for ax in axs[2]:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    fig.suptitle(f"$R - R_{{\\rm ref}}$ ({case}), reference {ref} (value in "
                 f"each title)", fontsize=12, y=0.95)
    fig.savefig(path)
    plt.close(fig)
    return path


# 10. Text

def pm(x, e, nd=4, sign=True):
    return (f"{x:+.{nd}f} ± {e:.{nd}f}" if sign else f"{x:.{nd}f} ± {e:.{nd}f}")


def sig(val, err):
    """Significance, capped: two techniques that share legs can differ by
    a deterministic amount with next to no scatter, and 10^15 sigma says
    nothing that >999 sigma does not."""
    if not err > 0:
        return "—" if val == 0 else "exact"
    z = val / err
    return f"{z:+.1f}σ" if abs(z) < 1000 else (">+999σ" if z > 0 else "<−999σ")


def _case_file(case):
    return "amplitude_direction.png" if case == MAIN_CASE else \
        f"amplitude_direction_{case.replace('+', 'P')}.png"


def matrix_md(R, Rerr):
    rows = ["|   | x | y | z |", "|---|---|---|---|"]
    for i in range(3):
        rows.append(f"| **{'xyz'[i]}** | "
                    + " | ".join(f"{R[i, j]:+.5f} ± {Rerr[i, j]:.5f}"
                                 for j in range(3)) + " |")
    return "\n".join(rows)


def amp_table(res, names, case, ref):
    rows = ["| technique | recons | Ā ± error on the mean | from response "
            "sims | sd(A) per sim | Ā − Ā_ref | significance | response "
            "error × √recons |",
            "|---|---:|---|---:|---:|---|---:|---:|"]
    rref = res.get((ref, case))
    for n in names:
        r = res[(n, case)]
        dv, de = difference(r, rref, "A") if rref else (np.nan, 0.0)
        rows.append(
            f"| {n} | {r['recons']} | {pm(r['A_mean'], r['A_err']['total'])} "
            f"| {r['A_err']['resp']:.4f} | {r['A_sd']:.4f} | "
            + ("reference" if r is rref else pm(dv, de)) + " | "
            + ("" if r is rref else sig(dv, de)) + " | "
            f"{r['A_err']['resp'] * np.sqrt(r['recons']):.3f} |")
    return "\n".join(rows)


def dir_table(res, names, case, ref):
    rows = ["| technique | l ± σ_l per sim | b ± σ_b per sim | σ_l cos b | "
            "mean-vector l̄ ± err | mean-vector b̄ ± err | l̄ − l̄_ref | "
            "b̄ − b̄_ref | offset of l̄,b̄ from input |",
            "|---|---|---|---:|---|---|---|---|---:|"]
    rref = res.get((ref, case))
    for n in names:
        r = res[(n, case)]
        if r is rref:
            dl = db = "reference"
        else:
            v, e = difference(r, rref, "l")
            dl = f"{pm(v, e, 3)} ({sig(v, e)})"
            v, e = difference(r, rref, "b")
            db = f"{pm(v, e, 3)} ({sig(v, e)})"
        rows.append(
            f"| {n} | {pm(r['l_mean'] % 360, r['l_sd'], 2, False)}° | "
            f"{pm(r['b_mean'], r['b_sd'], 2)}° | "
            f"{r['l_sd'] * np.cos(np.radians(r['b_mean'])):.2f}° | "
            f"{pm(r['lbar'], r['l_err']['total'], 3, False)}° | "
            f"{pm(r['bbar'], r['b_err']['total'], 3)}° | {dl} | {db} | "
            f"{r['off_bar']:.3f}° |")
    return "\n".join(rows)


def technique_md(name, t, res, ref, extra):
    L = [f"# {name}: {t['label']}", "",
         f"**What is boosted.** {t['boost']}.  ",
         f"**What is subtracted.** {t['subtract']}.", "",
         t["desc"], ""]
    r0 = res[(name, MAIN_CASE)]
    L += [f"**Sims.** {r0['n_sims']} of {t['n']} requested sims complete, "
          f"{r0['recons']} reconstructions ({t['legs']} per sim), "
          f"{r0['n_legs']} boosted legs in the fit.  Boost "
          f"β = {t['beta']:.5e} ({t['beta'] / BETA:g}× the input).  Applied "
          f"to {r0['n_dat']} data sims with the mean field of "
          f"{r0['n_mf']} sims subtracted.", ""]
    L += ["## Figures", ""]
    for case in PLOT_CASES:
        L += [f"![amplitude and direction, {case}]({_case_file(case)})", ""]
    L += ["## Amplitude and direction", "",
          "| case | Ā ± err | err: data / MF / response | sd(A) | "
          "l ± σ_l | b ± σ_b | l 16–50–84% | b 16–50–84% | l̄ ± err | "
          "b̄ ± err | \\|v̄\\| km/s |",
          "|---|---|---|---:|---|---|---|---|---|---|---:|"]
    for case in CASES:
        r = res[(name, case)]
        e = r["A_err"]
        L.append(
            f"| {case} | {pm(r['A_mean'], e['total'])} | "
            f"{e['data']:.4f} / {e['mf']:.4f} / {e['resp']:.4f} | "
            f"{r['A_sd']:.4f} | {pm(r['l_mean'] % 360, r['l_sd'], 2, False)}° | "
            f"{pm(r['b_mean'], r['b_sd'], 2)}° | "
            + "–".join(f"{v % 360:.1f}" for v in r["l_pct"]) + " | "
            + "–".join(f"{v:+.1f}" for v in r["b_pct"]) + " | "
            f"{pm(r['lbar'], r['l_err']['total'], 3, False)}° | "
            f"{pm(r['bbar'], r['b_err']['total'], 3)}° | "
            f"{r['vbar_kms']:.1f} |")
    L += ["", f"Input: A = 1, (l, b) = ({L_TRUE:.3f}°, {B_TRUE:+.3f}°), "
              f"|v| = {BETA * C_KMS:.1f} km/s.", ""]
    if name != ref and (ref, MAIN_CASE) in res:
        L += [f"## Against the reference ({ref})", "",
              "Differences use the same data and mean field and matched "
              "response seeds, so their errors are much smaller than the "
              "errors on either technique alone.", "",
              "| case | Ā − Ā_ref | l̄ − l̄_ref | b̄ − b̄_ref | "
              "largest R − R_ref (element) |",
              "|---|---|---|---|---:|"]
        for case in CASES:
            r, rr = res[(name, case)], res[(ref, case)]
            cells = []
            for k, nd in (("A", 4), ("l", 3), ("b", 3)):
                v, e = difference(r, rr, k)
                cells.append(f"{pm(v, e, nd)} ({sig(v, e)})")
            dR = r["R"] - rr["R"]
            dRe = difference_R(r, rr)
            i, j = np.unravel_index(np.argmax(np.abs(dR)), dR.shape)
            cells.append(f"{pm(dR[i, j], dRe[i, j], 5)} "
                         f"({sig(dR[i, j], dRe[i, j])}, R_{'xyz'[i]}"
                         f"{'xyz'[j]})")
            L.append(f"| {case} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Response matrix", ""]
    for case in CASES:
        r = res[(name, case)]
        L += [f"### {case}", "", matrix_md(r["R"], r["Rerr"]), "",
              f"condition number {r['cond']:.3f};  eigenvalues of (R+Rᵀ)/2: "
              + ", ".join(f"{v:+.5f}" for v in r["evals"]), ""]
    L += extra
    return "\n".join(L)


# 11. Main

def main():
    print(f"{'=' * 72}\n response-matrix techniques\n{'=' * 72}")
    print(f"cache       = {ab.CACHE_DIR}")
    print(f"output      = {OUT}")
    print(f"input       = beta {BETA:.5e}, (l, b) = ({L_TRUE:.3f}, "
          f"{B_TRUE:+.3f}) deg")
    _dg = getattr(ab.aberration, "dir_gal", None)
    if _dg is not None:
        print(f"              pixell dir_gal ({np.degrees(_dg[0]):.3f}, "
              f"{np.degrees(_dg[1]):+.3f}) deg, as a check on the rotation")
    print(f"techniques  = {', '.join(SELECTED)}")
    print(f"sims        = axis {N_AXIS}, multiples {N_MULT} "
          f"(x {', '.join(f'{k:g}' for k in MULTS)}), noisy {N_NOISY}, "
          f"random {N_RAND}")

    if not RM.rm_analyse_only:
        make_sims()
    if ab.args.nshards > 1:
        print("\nThis shard's response sims are done.  When every shard has "
              "finished, rerun\nwith the default --nshards 1 to collect and "
              "compare them.")
        return

    print("\nloading data and mean-field sims", flush=True)
    dat, dat_idx = load_stack("dat", RM.rm_n_data)
    mf, mf_idx = load_stack("mf", RM.rm_n_mf)
    if dat is None or mf is None or len(dat_idx) < 3 or len(mf_idx) < 3:
        raise SystemExit(f"found {len(dat_idx)} data and {len(mf_idx)} "
                         f"mean-field sims in {ab.CACHE_DIR}; need both.")
    print(f"   {len(dat_idx)} data sims (dat_{dat_idx[0]:05d} to "
          f"dat_{dat_idx[-1]:05d}), {len(mf_idx)} mean-field sims")

    # The vectorised L=1 extraction has to agree with the pipeline's own.
    one = {e: dat[e][0] for e in EST}
    for case in CASES:
        if not np.allclose(case_l1({e: dat[e][:1] for e in EST}, case)[0],
                           ab.coadd(one, case), rtol=1e-12, atol=0):
            raise SystemExit("internal check failed: vectorised l1 disagrees "
                             "with aberration_act_mask.coadd")

    dat_l1 = {case: case_l1(dat, case) for case in CASES}
    mf_l1 = {case: case_l1(mf, case) for case in CASES}
    mf_cov = {case: np.cov(mf_l1[case], rowvar=False) for case in CASES}
    mf_alm = {e: mf[e].mean(axis=0) for e in EST}

    res = {}
    names = []
    skipped = []
    for name in SELECTED:
        t = TECH[name]
        meas = measurements(t, mf_alm)
        have = 0 if meas is None else meas["n_sims"]
        if have < max(RM.rm_min_sims, 4):
            skipped.append((name, have))
            print(f"   {name}: {have} complete sims, skipped")
            continue
        for case in CASES:
            res[(name, case)] = analyse_one(name, t, meas, case, dat_l1,
                                            mf_l1, mf_cov)
        names.append(name)
    if not names:
        raise SystemExit("no technique has enough complete sims yet")
    ref = RM.rm_reference if RM.rm_reference in names else names[0]

    # The pipeline's own R, if it wrote a summary: axis_onesided at the same
    # N should reproduce it to rounding.
    extra_main = []
    spath = os.path.join(ab.CACHE_DIR, "summary.npz")
    if os.path.exists(spath) and "axis_onesided" in names:
        try:
            with np.load(spath) as f:
                cases_s = [str(c) for c in f["cases"]]
                k = cases_s.index(MAIN_CASE)
                Rs = f[f"R_{k}"]
                ns = int(f["n_resp_used"]) if "n_resp_used" in f.files else -1
            r = res[("axis_onesided", MAIN_CASE)]
            msg = (f"The pipeline's summary.npz R ({ns} sims) differs from "
                   f"axis_onesided ({r['n_sims']} sims) by at most "
                   f"{np.abs(Rs - r['R']).max():.2e} in {MAIN_CASE}"
                   + ("; with equal N this should be rounding."
                      if ns == r["n_sims"] else "."))
            print("\n" + msg)
            extra_main = ["## Consistency with the pipeline", "", msg, ""]
        except Exception as exc:                            # noqa: BLE001
            print(f"   (could not compare with {spath}: {exc})")

    # Console table.
    for case in CASES:
        print(f"\n{'-' * 72}\n {case}   (reference {ref})\n{'-' * 72}")
        print(f"   {'technique':20s} {'A mean':>9s} {'err':>7s} {'resp':>7s} "
              f"{'sd':>7s} {'dA':>9s} {'l':>8s} {'sd_l':>6s} {'b':>7s} "
              f"{'sd_b':>6s}")
        for n in names:
            r = res[(n, case)]
            dv, de = difference(r, res[(ref, case)], "A")
            print(f"   {n:20s} {r['A_mean']:+9.4f} {r['A_err']['total']:7.4f} "
                  f"{r['A_err']['resp']:7.4f} {r['A_sd']:7.4f} "
                  f"{dv:+9.4f} {r['l_mean']:8.2f} {r['l_sd']:6.2f} "
                  f"{r['b_mean']:+7.2f} {r['b_sd']:6.2f}")

    # Files.
    os.makedirs(OUT, exist_ok=True)
    written = []
    for name in names:
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        for case in PLOT_CASES:
            written.append(plot_amplitude_direction(
                res[(name, case)], TECH[name], os.path.join(d, _case_file(case))))
        with open(os.path.join(d, "summary.md"), "w") as fh:
            fh.write(technique_md(name, TECH[name], res, ref, []))
        written.append(os.path.join(d, "summary.md"))

    for case in PLOT_CASES:
        tag = "" if case == MAIN_CASE else "_" + case.replace("+", "P")
        written.append(plot_comparison(
            res, names, case, ref,
            os.path.join(OUT, f"comparison_amplitude_direction{tag}.png")))
        written.append(plot_R_comparison(
            res, names, case, ref,
            os.path.join(OUT, f"comparison_response_matrix{tag}.png")))

    write_document(res, names, ref, dat_idx, mf_idx, skipped, extra_main)
    written.append(os.path.join(OUT, "summary.md"))
    write_tables(res, names, ref)
    written += [os.path.join(OUT, "results.csv"),
                os.path.join(OUT, "results.npz")]
    print()
    for w in written:
        print(f"wrote {w}")


def write_document(res, names, ref, dat_idx, mf_idx, skipped, extra):
    L = ["# Aberration response matrix: techniques compared", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} by "
         f"aberration_response_methods.py.", "",
         f"Cache: `{ab.CACHE_DIR}`  ",
         f"Data: {len(dat_idx)} aberrated sims; mean field from "
         f"{len(mf_idx)} unaberrated sims, the same for every technique.  ",
         f"Input: β = {BETA:.5e} toward (l, b) = ({L_TRUE:.3f}°, "
         f"{B_TRUE:+.3f}°), |v| = {BETA * C_KMS:.1f} km/s, so A = 1.  ",
         f"Reference for the differences: **{ref}**.", ""]
    L += ["## Techniques", "",
          "| technique | boosted | subtracted | sims | recons |",
          "|---|---|---|---:|---:|"]
    for n in names:
        t = TECH[n]
        r = res[(n, MAIN_CASE)]
        L.append(f"| [{n}]({n}/summary.md) | {t['boost']} | {t['subtract']} "
                 f"| {r['n_sims']} | {r['recons']} |")
    for n, have in skipped:
        L.append(f"| {n} | {TECH[n]['boost']} | {TECH[n]['subtract']} | "
                 f"{have} (skipped) | — |")
    L.append("")
    L += [f"## Amplitude ({MAIN_CASE})", "", amp_table(res, names, MAIN_CASE,
                                                       ref), "",
          f"## Direction ({MAIN_CASE})", "",
          dir_table(res, names, MAIN_CASE, ref), "",
          f"![comparison](comparison_amplitude_direction.png)", "",
          f"![response matrices](comparison_response_matrix.png)", ""]
    L += extra
    L += ["## Each technique", ""]
    for n in names:
        L += [f"### {n}: {TECH[n]['label']}", "", TECH[n]["desc"], "",
              f"![{n}]({n}/{_case_file(MAIN_CASE)})", "",
              f"Details: [{n}/summary.md]({n}/summary.md)", ""]
    L += ["## The other estimator combinations", ""]
    for case in CASES:
        if case == MAIN_CASE:
            continue
        L += [f"### {case}", "", amp_table(res, names, case, ref), "",
              dir_table(res, names, case, ref), ""]
    L += ["## Reading the numbers", "",
          "*Amplitude.* A = â·a_in / |a_in|² with â = R⁻¹(v − MF) for each "
          "data sim, exactly as in the pipeline.  **sd(A)** is the per-sim "
          "scatter and is set almost entirely by the data sims, so it is "
          "nearly the same for every technique.  The **error on the mean** "
          "adds the data, mean-field and response-sim contributions in "
          "quadrature; the **from response sims** column is the part a "
          "technique is responsible for.  **Response error × √recons** puts "
          "the techniques on an equal-cost footing: lower is more efficient.",
          "",
          "*Direction.* Each data sim gives a direction −â/|â|, converted to "
          "galactic (l, b).  **l ± σ_l** and **b ± σ_b** are the mean and sd "
          "of those per-sim angles, with l unwrapped about the input so a "
          "scatter across l = 0 stays whole.  σ_l is stretched by 1/cos b; "
          "**σ_l cos b** is the same scatter as an angle on the sky.  On a "
          "cut sky the per-sim directions are broad and not Gaussian, so the "
          "mean of the per-sim angles is pulled away from the truth; the "
          "**mean-vector** direction, of the mean of â over the data sims, "
          "is the one to compare with the input, and its error comes from "
          "the same three-part propagation as Ā.", "",
          "*Differences.* Every technique is applied to the same data and "
          "mean field, and the response sims of every technique use the "
          "same CMB seed for sim i, so the difference between two "
          "techniques is measured seed by seed and its error is far smaller "
          "than either error on its own.  The significance column is that "
          "difference over its own error: that is where a bias from a "
          "one-sided difference or a large boost shows up.", "",
          "*Response error.* A cluster-robust sandwich on the least-squares "
          "fit y = R n, one cluster per CMB seed.  For the axis designs it "
          "reduces to the pipeline's W = V·u formula; for the random designs "
          "it is the only form that works, since no single sim measures a "
          "whole column.  For axis_noisy_mf the mean field enters R as well "
          "as the data, and that shared error is carried through.", ""]
    with open(os.path.join(OUT, "summary.md"), "w") as fh:
        fh.write("\n".join(L))


def write_tables(res, names, ref):
    cols = ["technique", "case", "n_sims", "recons", "beta_mult", "A_mean",
            "A_sd", "A_err_total", "A_err_data", "A_err_mf", "A_err_resp",
            "dA_ref", "dA_ref_err", "l_mean", "l_sd", "b_mean", "b_sd",
            "lbar", "lbar_err", "bbar", "bbar_err", "dl_ref", "dl_ref_err",
            "db_ref", "db_ref_err", "offset_bar_deg", "cond"]
    cols += [f"R_{a}{b}" for a in "xyz" for b in "xyz"]
    cols += [f"Rerr_{a}{b}" for a in "xyz" for b in "xyz"]
    arrays = dict(names=np.array(names), cases=np.array(CASES),
                  l_true=L_TRUE, b_true=B_TRUE, beta=BETA, ref=np.array(ref))
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for n in names:
            for case in CASES:
                r = res[(n, case)]
                rr = res[(ref, case)]
                dA = difference(r, rr, "A")
                dl = difference(r, rr, "l")
                db = difference(r, rr, "b")
                w.writerow([n, case, r["n_sims"], r["recons"],
                            f"{TECH[n]['beta'] / BETA:g}", r["A_mean"],
                            r["A_sd"], r["A_err"]["total"],
                            r["A_err"]["data"], r["A_err"]["mf"],
                            r["A_err"]["resp"], dA[0], dA[1], r["l_mean"],
                            r["l_sd"], r["b_mean"], r["b_sd"], r["lbar"],
                            r["l_err"]["total"], r["bbar"],
                            r["b_err"]["total"], dl[0], dl[1], db[0], db[1],
                            r["off_bar"], r["cond"]]
                           + list(r["R"].ravel()) + list(r["Rerr"].ravel()))
                p = f"{n}__{case}__"
                for key in ("R", "Rerr", "amp", "l_sims", "b_sims", "off",
                            "dirs", "evals", "evecs"):
                    arrays[p + key] = np.asarray(r[key])
    np.savez(os.path.join(OUT, "results.npz"), **arrays)


if __name__ == "__main__":
    main()