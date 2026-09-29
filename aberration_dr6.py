"""
Doppler boost of the CMB, aberration and modulation, measured with two
quadratic estimators on the ACT DR6 lensing mask with ACT-like noise.  One
fixed configuration; the only inputs are the numbers of simulations and,
optionally, how to split them across jobs.

    python aberration_dr6.py --n-meanfield 800 --n-response 60 --n-data 400

To split the sims across independent jobs (e.g. a SLURM array, --array=0-9),
give every job the same sim counts plus
    --nshards 10 --shard $SLURM_ARRAY_TASK_ID
Each job makes and caches the sims i with i % nshards == shard.  When all
have finished, run once more without --shard/--nshards: that run makes any
sims still missing and writes the complete report and figures.

Estimators, each reduced to its L = 1 vector and scaled so that on the full
sky, for its own effect, it estimates the boost vector u = beta * d:
    aberration  lensing QE, TT+TE+EE          phi = -u . n
    modulation  TT amplitude-modulation QE    f = b u . n,  b = x coth(x/2) - 1
                (Wiener x inverse-variance T product map, tempura "amp"
                normalisation; b = 2.05 at 150 GHz)

Every boost goes through pixell.aberration.boost_map.
    A  mean field  unboosted, masked, noisy sims.
    B  response    noise-free sims boosted by +beta and -beta along x, y, z,
                   once with aberration only and once with modulation only.
                   Half the difference, over beta, gives the 6x6 response K of
                   [aberration QE, modulation QE] to [aberration, modulation].
    C  data        sims with the full boost (pixell's dipole) plus noise.

    separate   u_aber and u_mod solved from K together, so each is free of
               the other effect and they are not assumed equal
    joint      one u for both effects (equal in these sims), weighted by the
               covariance of the six estimator components

Each reconstruction is cached in cache_boost/<CACHE_TAG>/ (one file per
sim, named by stage and index; the seeds follow the index), so an
interrupted run resumes and a larger N only adds sims.  Give CACHE_TAG a new
value whenever a setting changes: each tag keeps its own sims.  Every run, shard or not, then reports on all
sims cached so far (by any shard) and writes summary.npz and the figures
(plots/, drawn by aberration_dr6_plots.py) into that same folder, so a shard
that finishes early leaves a partial snapshot there.  Unlensed Gaussian CMB, homogeneous ACT
f150 noise, noise high-passed below LMIN.

DR6 lensing mask (healpix nside 4096, equatorial, ~0.8 GB):
https://phy-act1.princeton.edu/public/data/dr6_lensing_v1/maps/baseline/mask_act_dr6_lensing_v1_healpix_nside_4096_baseline.fits
"""

import argparse
import os
import time

import numpy as np
import healpy as hp
import camb
import pytempura
from falafel import qe
from pixell import enmap, curvedsky, aberration, reproject, utils

parser = argparse.ArgumentParser()
parser.add_argument("--n-meanfield", type=int, default=800)
parser.add_argument("--n-response", type=int, default=60)
parser.add_argument("--n-data", type=int, default=400)
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--nshards", type=int, default=1)
args = parser.parse_args()

# ---------------------------------------------------------------- settings
MASK_FILE = "mask_act_dr6_lensing_v1_healpix_nside_4096_baseline.fits"
LMIN, LMAX, MLMAX = 600, 3000, 3500
LOUT = 5                                  # highest reconstruction L kept
RES = 3.0 * utils.arcmin                  # fejer1 grid: exact SHTs to l = 3599
NOISE_T, BEAM = 24.0, 1.42                # uK-arcmin (T; P is sqrt2 x), FWHM arcmin
KNEE_T, KNEE_P = 3000.0, 475.0            # 1/f knees
ALPHA_T, ALPHA_P = -3.0, -4.5             # 1/f slopes
FREQ = 150e9                              # Hz; sets the modulation factor b
TCMB, C_KMS = 2.7255e6, 299792.458        # uK, km/s
BETA, BDIR = aberration.beta, aberration.dir_equ     # BDIR is (ra, dec), rad
AXES = [(0.0, 0.0), (np.pi / 2, 0.0), (0.0, np.pi / 2)]   # x, y, z as (ra, dec)
EST = ["TT", "TE", "EE"]
CASES = {"TT": ["TT"], "TE": ["TE"], "EE": ["EE"], "T+P": EST}
ABER_CASE = "T+P"                         # lensing combination used for u_aber
CACHE_TAG = "alphaT-3_alphaP-4.5"         # new value whenever a setting changes
CACHE = os.path.join("cache_boost", CACHE_TAG)     # one .npy per sim

X_NU = utils.h * FREQ / (utils.k * utils.T_cmb)
B_NU = X_NU / np.tanh(X_NU / 2) - 1       # linearised-T modulation factor
D_TRUE = np.array([np.cos(BDIR[1]) * np.cos(BDIR[0]),
                   np.cos(BDIR[1]) * np.sin(BDIR[0]),
                   np.sin(BDIR[1])])
U_TRUE = BETA * D_TRUE
GAL = hp.Rotator(coord=["C", "G"]).mat    # equatorial -> galactic vectors

# pixell's interpol_map (0.32.7 and master) doubles a full-sky map for the
# NUFFT by flipping ra where it should flip dec.  The seam this leaves at the
# poles rings: 26% of the map rms in the polar rows, ~5e-4 at |dec| < 62.
# Doubling the map here makes the boost exact to 1e-11.
_interpol = aberration.interpol_map


def _interpol_fixed(imap, pixs, epsilon=None, nthread=None, ydouble=False):
    if ydouble:
        flip = np.roll(imap[..., ::-1, :], imap.shape[-1] // 2, -1)
        imap = enmap.enmap(np.concatenate([imap, flip], -2), imap.wcs)
    return _interpol(imap, pixs, epsilon=epsilon, nthread=nthread)


aberration.interpol_map = _interpol_fixed

# ---------------------------------------------------------------- spectra
print("CAMB ...", flush=True)
pars = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122, ns=0.965,
                       As=2.1e-9, tau=0.06)
pars.set_for_lmax(MLMAX + 500)
cls = camb.get_results(pars).get_cmb_power_spectra(
    pars, raw_cl=True, spectra=["unlensed_scalar"])["unlensed_scalar"]
cltt, clee, clbb, clte = cls[:MLMAX + 1].T          # dimensionless (dT/T)^2

ell = np.arange(MLMAX + 1.0)
bl2 = hp.gauss_beam(BEAM * utils.arcmin, lmax=MLMAX) ** 2


def act_noise(white, knee, alpha):
    nl = ((white * utils.arcmin / TCMB) ** 2
          * (1.0 + (np.maximum(ell, 1.0) / knee) ** alpha) / bl2)
    nl[LMAX + 1:] = nl[LMAX]
    nl[:LMIN] = 0.0
    return nl


nltt = act_noise(NOISE_T, KNEE_T, ALPHA_T)
nlee = act_noise(NOISE_T * np.sqrt(2.0), KNEE_P, ALPHA_P)

ucls = {"TT": cltt, "EE": clee, "BB": clbb, "TE": clte}
tcls = {"TT": cltt + nltt, "EE": clee + nlee, "BB": clbb + nlee, "TE": clte}


def inverse_variance(total):
    f = np.zeros(MLMAX + 1)
    f[LMIN:LMAX + 1] = 1.0 / total[LMIN:LMAX + 1]
    return f


FILTERS = [inverse_variance(tcls[k]) for k in ("TT", "EE", "BB")]

# Full-sky normalisations.  A lensing case sums the unnormalised estimators
# and divides by sum(1/A_L), the inverse-variance combination.
norms = pytempura.get_norms(EST, ucls, ucls, tcls, LMIN, LMAX, k_ellmax=MLMAX)
NORM = {}
for case, members in CASES.items():
    inv = sum(1.0 / np.asarray(norms[e][0][1:LOUT + 1]) for e in members)
    NORM[case] = np.concatenate([[0.0], 1.0 / inv])       # L = 0 dropped
amp_norm = pytempura.norm_general.qtt("amp", MLMAX, LMIN, LMAX, cltt, cltt,
                                      tcls["TT"])
NORM["MOD"] = np.concatenate([[0.0], np.asarray(amp_norm)[0, 1:LOUT + 1]])

PS_CMB = np.zeros((3, 3, MLMAX + 1))
PS_CMB[0, 0], PS_CMB[1, 1], PS_CMB[2, 2] = cltt, clee, clbb
PS_CMB[0, 1] = PS_CMB[1, 0] = clte
PS_NOISE = np.zeros((3, 3, MLMAX + 1))
PS_NOISE[0, 0], PS_NOISE[1, 1], PS_NOISE[2, 2] = nltt, nlee, nlee

# ---------------------------------------------------------------- mask
shape, wcs = enmap.fullsky_geometry(res=RES)
px = qe.pixelization(shape=shape, wcs=wcs)

print("mask ...", flush=True)
hmask = np.clip(np.nan_to_num(hp.read_map(MASK_FILE, dtype=np.float32)), 0, 1)
# Bilinear lookup at each CAR pixel centre: stays inside [0, 1], unlike "harm".
mask = reproject.healpix2map(hmask, shape, wcs, spin=[0], method="spline",
                             order=1).astype(np.float64)
del hmask
pixarea = enmap.pixsizemap(shape, wcs, broadcastable=True)
W1, W2 = [float((mask ** n * pixarea).sum() / (4 * np.pi)) for n in (1, 2)]

# ---------------------------------------------------------------- sims
PACK_L = np.array([l for l in range(LOUT + 1) for m in range(l + 1)])
PACK_M = np.array([m for l in range(LOUT + 1) for m in range(l + 1)])
PACK_IDX = hp.Alm.getidx(MLMAX, PACK_L, PACK_M)


def cmb(seed):
    return curvedsky.rand_map((3,) + shape, wcs, PS_CMB, lmax=MLMAX, seed=seed)


def noise(seed):
    return curvedsky.rand_map((3,) + shape, wcs, PS_NOISE, lmax=MLMAX,
                              seed=seed)


def boost(tqu, direction, beta, aberrate=True, modulate=True):
    """pixell's Doppler boost.  The maps are dT/T_cmb, hence map_unit=T_cmb.
    With aberrate=False, boost_map modulates its input in place."""
    return aberration.boost_map(tqu, dir=np.asarray(direction, float),
                                beta=beta, freq=FREQ, map_unit=utils.T_cmb,
                                aberrate=aberrate, modulate=modulate)


def reconstruct(tqu):
    """TQU map -> unnormalised TT, TE, EE lensing and TT modulation alm for
    L <= LOUT, shape (4, npack)."""
    alm = curvedsky.map2alm(mask * tqu, lmax=MLMAX)
    f = [qe.filter_alms(a, fl, lmin=LMIN, lmax=LMAX)
         for a, fl in zip(alm, FILTERS)]
    rec = qe.qe_all(px, ucls, MLMAX, fTalm=f[0], fEalm=f[1], fBalm=f[2],
                    estimators=EST)
    # Modulation: the Wiener-filtered T map times the inverse-variance one.
    # (falafel's qe_mask does this too, but returns complex maps on CAR.)
    t_ivf = curvedsky.alm2map(f[0], enmap.empty(shape, wcs))
    t_wf = curvedsky.alm2map(curvedsky.almxfl(f[0], cltt),
                             enmap.empty(shape, wcs))
    mod = curvedsky.map2alm(t_ivf * t_wf, lmax=MLMAX)
    return np.array([rec[e][0][PACK_IDX] for e in EST] + [mod[PACK_IDX]])


def cache_path(label, i):
    return os.path.join(CACHE, f"{label.replace(' ', '_')}_{i:04d}.npy")


RESP_LABELS = [f"{effect} {axis}{sign}" for effect in ("aberration", "modulation")
               for axis in "xyz" for sign in "+-"]
STAGE = {"mf": "mf", "data": "data", **{l: "response" for l in RESP_LABELS}}
TODO = {}           # sims this run still has to make, per label (set in main)


def run(label, n, one_sim):
    """Make and cache this shard's sims out of 0..n-1 (all of them with
    --nshards 1) that are not cached yet.  Each file is written under a
    temporary name and renamed, so no run ever reads one half-written.
    The time left assumes every remaining sim, of any stage, takes as long
    as the average so far in this call."""
    t0, done = time.time(), 0
    for i in range(args.shard, n, args.nshards):
        path = cache_path(label, i)
        if os.path.exists(path):
            continue
        tmp = f"{path}.{os.getpid()}.npy"
        np.save(tmp, one_sim(i))
        os.replace(tmp, path)
        done += 1
        TODO[label] -= 1
        rate = (time.time() - t0) / done
        stage_left = sum(v for l, v in TODO.items() if STAGE[l] == STAGE[label])
        print(f"   {label} {i + 1}/{n}   {rate:.1f} s/sim "
              f"{rate * stage_left / 60:.1f} min left ({STAGE[label]}) "
              f"{rate * sum(TODO.values()) / 60:.1f} min left (total)",
              flush=True)


def collect(labels, n):
    """The sims 0..n-1 cached for every one of labels, by any shard, as an
    array (label, sim, ...).  A response sim counts once all its legs are in."""
    idx = [i for i in range(n)
           if all(os.path.exists(cache_path(l, i)) for l in labels)]
    return np.array([[np.load(cache_path(l, i)) for i in idx] for l in labels])


# ---------------------------------------------------------------- analysis
def l1_vector(a):
    """Cartesian vector of the L = 1 part of packed alm (along the last axis)."""
    return np.stack([-np.sqrt(3 / (2 * np.pi)) * a[..., 2].real,
                     np.sqrt(3 / (2 * np.pi)) * a[..., 2].imag,
                     np.sqrt(3 / (4 * np.pi)) * a[..., 1].real], axis=-1)


def cl_of(power):
    """Per-mode power (packed) -> C_L, counting m < 0."""
    w = np.where(PACK_M == 0, 1.0, 2.0) * power
    return np.array([w[PACK_L == l].sum() / (2 * l + 1)
                     for l in range(LOUT + 1)])


def debiased_cl(x):
    """C_L of the mean of the rows of x, minus its Monte Carlo noise."""
    var = (x.real.var(axis=0, ddof=1) + x.imag.var(axis=0, ddof=1)) / len(x)
    return cl_of(np.abs(x.mean(axis=0)) ** 2 - var)


def est_alm(rec, name):
    """Raw reconstructions (..., 4, npack) -> one estimator's packed alm, in
    units where its L = 1 vector estimates u = beta d on the full sky."""
    if name == "MOD":
        return NORM["MOD"][PACK_L] * rec[..., 3, :] / B_NU
    k = [EST.index(e) for e in CASES[name]]
    return -NORM[name][PACK_L] * rec[..., k, :].sum(axis=-2)      # phi = -u.n


def system(names, mf, diff, dat):
    """L = 1 vectors of two estimators stacked to 6, and their 6x6 response
    K (rows: components, columns: aberration xyz then modulation xyz)."""
    Y = [[est_alm(x, n) for x in (mf, diff, dat)] for n in names]
    y_mf, y_dat = (np.concatenate([l1_vector(y[k]) for y in Y], -1)
                   for k in (0, 2))
    V = np.concatenate([l1_vector(y[1]) for y in Y], -1) / BETA
    K = V.mean(axis=2).reshape(6, 6).T
    Kerr = (V.std(axis=2, ddof=1) / np.sqrt(V.shape[2])).reshape(6, 6).T
    return Y, y_mf, y_dat, K, Kerr


def galactic(v):
    """Equatorial unit vectors (..., 3) -> galactic (l, b) in degrees."""
    g = v @ GAL.T
    return (np.degrees(np.arctan2(g[..., 1], g[..., 0])) % 360,
            np.degrees(np.arcsin(np.clip(g[..., 2], -1, 1))))


STATS = ["A", "v_x", "v_y", "v_z", "l", "b"]        # per-sim statistics


def stats(u):
    """Per-sim u (n, 3) -> amplitude A, velocity in km/s, and the galactic
    l, b of its direction in degrees (l unwrapped around the input)."""
    l_in = galactic(D_TRUE)[0]
    l, b = galactic(u / np.linalg.norm(u, axis=1)[:, None])
    return np.column_stack([u @ U_TRUE / (U_TRUE @ U_TRUE), u * C_KMS,
                            l_in + (l - l_in + 180) % 360 - 180, b])


def report(title, e):
    """Print one estimate and return what the summary keeps of it."""
    mean, sd, err, part = e["mean"], e["sd"], e["err"], e["err_parts"]
    dirs = e["u"] / np.linalg.norm(e["u"], axis=1)[:, None]
    ubar = e["u"].mean(axis=0) / np.linalg.norm(e["u"].mean(axis=0))
    l_in, b_in = galactic(D_TRUE)
    v_in = C_KMS * U_TRUE
    print(f"\n{title}")
    print(f"   A = {mean[0]:+.4f} +- {sd[0]:.4f} (sd), +- {err[0]:.4f} "
          f"(error on the mean)   expect 1")
    print(f"       error on the mean from data {part[0, 0]:.4f}, mean field "
          f"{part[1, 0]:.4f}, response {part[2, 0]:.4f}")
    print(f"   null A = {e['null'].mean():+.4f} +- {e['null_err']:.4f}"
          f"   expect 0")
    print("   v [km/s]  " + "   ".join(
        f"{'xyz'[k]} {mean[k + 1]:+7.1f} +- {err[k + 1]:5.1f} "
        f"(in {v_in[k]:+6.1f})" for k in range(3)))
    print(f"   galactic  l = {mean[4]:.2f} +- {err[4]:.2f} (sd {sd[4]:.2f}), "
          f"b = {mean[5]:+.2f} +- {err[5]:.2f} (sd {sd[5]:.2f});  "
          f"input l = {l_in:.2f}, b = {b_in:+.2f}")
    print(f"   mean velocity "
          f"{np.degrees(np.arccos(np.clip(ubar @ D_TRUE, -1, 1))):.1f} deg "
          f"from the input, single sims median "
          f"{np.median(np.degrees(np.arccos(np.clip(dirs @ D_TRUE, -1, 1)))):.1f} deg")
    return dict(amp=e["stats"][:, 0], vel=e["stats"][:, 1:4], dir=dirs,
                null=e["null"], null_err=e["null_err"], stat_mean=mean,
                stat_sd=sd, stat_err=err, stat_err_parts=part)


def higher_l(Y, own):
    """(L, M) response of one estimator to its own effect, and the coherent
    power the true (full) boost leaves at each L, with a jackknife error."""
    S = Y[1][own] / BETA                               # (axis, sim, npack)
    P = np.einsum("jia,j->ia", (Y[1][0] + Y[1][1]) / BETA, U_TRUE)
    n_r = len(P)
    jk = np.array([debiased_cl(np.delete(P, k, axis=0)) for k in range(n_r)])
    M, D = Y[0], Y[2]
    var_m = M.real.var(axis=0, ddof=1) + M.imag.var(axis=0, ddof=1)
    var_d = D.real.var(axis=0, ddof=1) + D.imag.var(axis=0, ddof=1)
    return dict(Ralm=S.mean(axis=1).T, Ralm_sd=np.abs(S.std(axis=1, ddof=1)).T,
                leak=debiased_cl(P), leakerr=np.sqrt((n_r - 1) * jk.var(axis=0)),
                datleak=cl_of(np.abs(D.mean(axis=0) - M.mean(axis=0)) ** 2
                              - var_d / len(D) - var_m / len(M)),
                noise=cl_of(var_m))


def linear_maps(K, C):
    """3x6 maps from the estimator vectors (mean field removed) to u: rows of
    K^-1 for the separate estimates, and for the joint one the least-squares
    fit of one u to both effects, weighted by the covariance C."""
    Kinv = np.linalg.inv(K)
    Kj = K[:, :3] + K[:, 3:]                   # one u drives both effects
    G = np.linalg.solve(Kj.T @ np.linalg.solve(C, Kj),
                        np.linalg.solve(C, Kj).T)
    return {"aberration": Kinv[:3], "modulation": Kinv[3:], "joint": G}


def solve(names, mf, diff, dat):
    """The three estimates of u from one pair of estimators, with the mean,
    sd and error on the mean of their per-sim statistics.  The error on the
    mean has three independent parts: the scatter of the data sims, and, by
    jackknife, the mean field and K, which every data sim shares."""
    Y, y_mf, y_dat, K, Kerr = system(names, mf, diff, dat)
    n_m, n_r = len(y_mf), diff.shape[2]
    m, C = y_mf.mean(axis=0), np.cov(y_mf, rowvar=False)

    def means(m, C, K):
        L = linear_maps(K, C)
        return {e: stats((y_dat - m) @ L[e].T).mean(axis=0) for e in L}

    jk_mf = [means(r.mean(axis=0), np.cov(r, rowvar=False), K)
             for r in (np.delete(y_mf, k, axis=0) for k in range(n_m))]
    K_jk = [system(names, mf, np.delete(diff, k, axis=2), dat)[3]
            for k in range(n_r)]
    jk_resp = [means(m, C, Kk) for Kk in K_jk]

    L = linear_maps(K, C)
    h = n_m // 2
    null_y = y_mf[h:] - y_mf[:h].mean(axis=0)     # split-half null test
    est = {}
    for e in L:
        u = (y_dat - m) @ L[e].T
        st = stats(u)
        var = np.array([st.var(axis=0, ddof=1) / len(st),
                        (n_m - 1) * np.var([j[e] for j in jk_mf], axis=0),
                        (n_r - 1) * np.var([j[e] for j in jk_resp], axis=0)])
        null = null_y @ L[e].T @ U_TRUE / (U_TRUE @ U_TRUE)
        est[e] = dict(u=u, stats=st, mean=st.mean(axis=0),
                      sd=st.std(axis=0, ddof=1), err=np.sqrt(var.sum(axis=0)),
                      err_parts=np.sqrt(var), null=null,
                      null_err=null.std(ddof=1) * np.sqrt(1 / len(null) + 1 / h))
    return Y, K, Kerr, m, C, est, K_jk


def analyse(mf, diff, dat):
    Y, K, Kerr, m, C, est, K_jk = solve((ABER_CASE, "MOD"), mf, diff, dat)
    Kj = K[:, :3] + K[:, 3:]
    mf_ratio = np.array([np.linalg.norm(m[i:i + 3])
                         / np.linalg.norm(Kj[i:i + 3] @ U_TRUE) for i in (0, 3)])

    print(f"\nresponse K per unit u: rows {ABER_CASE} QE xyz, MOD QE xyz; "
          f"columns aberration xyz, modulation xyz")
    for i in range(6):
        print("   " + " ".join(f"{K[i, j]:+.3f}({Kerr[i, j] * 1e3:3.0f})"
                               for j in range(6)))
    print(f"   (MC error on the mean in units of 1e-3; b = {B_NU:.4f})")
    print(f"   |mean field| / |K u_in|:  {ABER_CASE} {mf_ratio[0]:.1f},  "
          f"MOD {mf_ratio[1]:.1f}")

    titles = {"aberration": "aberration (separate)",
              "modulation": "modulation (separate)",
              "joint": "joint (one u for both effects)"}
    out = {name: report(title, est[name]) for name, title in titles.items()}
    for i, name in enumerate(("aberration", "modulation")):
        blk = np.s_[3 * i:3 * i + 3, 3 * i:3 * i + 3]
        eig = [np.linalg.eigvalsh(0.5 * (Kk[blk] + Kk[blk].T)) for Kk in K_jk]
        out[name].update(higher_l(Y[i], i), R=K[blk], Rerr=Kerr[blk],
                         eig_err=np.sqrt((len(eig) - 1) * np.var(eig, axis=0)))
    sd = np.sqrt(np.diag(C))
    out["system"] = dict(K=K, Kerr=Kerr, corr=C / np.outer(sd, sd),
                         mf_ratio=mf_ratio)

    print("\nseparate aberration amplitude for each lensing combination")
    rows = []
    for case in CASES:
        e = solve((case, "MOD"), mf, diff, dat)[5]["aberration"]
        rows.append([e["mean"][0], e["err"][0], e["sd"][0]])
        print(f"   {case:4s}  A = {e['mean'][0]:+.4f} +- {e['sd'][0]:.4f} (sd), "
              f"+- {e['err'][0]:.4f} (error on the mean)")
    out["lensing_cases"] = dict(names=np.array(list(CASES)), A=np.array(rows))
    return out


def main():
    print(f"lmin {LMIN}, lmax {LMAX}, mlmax {MLMAX}, {RES / utils.arcmin:g}' "
          f"CAR {shape[0]}x{shape[1]}, fsky {W1:.4f}, w2 {W2:.4f}")
    print(f"beta {BETA:.4e} towards ra {np.degrees(BDIR[0]):.2f}, "
          f"dec {np.degrees(BDIR[1]):.2f}; modulation b = {B_NU:.4f} at "
          f"{FREQ / 1e9:g} GHz", flush=True)

    print(f"noise 1/f slopes T {ALPHA_T:g}, P {ALPHA_P:g}; cache {CACHE}/"
          + (f"; shard {args.shard} of {args.nshards}" if args.nshards > 1
             else ""))
    os.makedirs(CACHE, exist_ok=True)
    counts = {"mf": args.n_meanfield, "data": args.n_data,
              **{l: args.n_response for l in RESP_LABELS}}
    TODO.update({l: sum(not os.path.exists(cache_path(l, i))
                        for i in range(args.shard, n, args.nshards))
                 for l, n in counts.items()})
    print(f"this run makes {sum(TODO.values())} sims; the rest are cached "
          f"or belong to other shards")
    print("\n[A] mean field")
    run("mf", args.n_meanfield, lambda i: reconstruct(
        cmb(1_000_000 + 2 * i) + noise(1_000_001 + 2 * i)))

    print("\n[B] response")
    for k, effect in enumerate(("aberration", "modulation")):
        for j, d in enumerate(AXES):
            for s in (1, -1):
                run(f"{effect} {'xyz'[j]}{'+-'[s < 0]}", args.n_response,
                    lambda i: reconstruct(boost(
                    cmb(2_000_000 + i), d, s * BETA,
                    aberrate=k == 0, modulate=k == 1)))

    print("\n[C] data")
    run("data", args.n_data, lambda i: reconstruct(
        boost(cmb(3_000_000 + 2 * i), BDIR, BETA) + noise(3_000_001 + 2 * i)))

    # Report on everything cached so far, whichever shard made it.
    mf = collect(["mf"], args.n_meanfield)[0]
    legs = collect(RESP_LABELS, args.n_response)            # (12 legs, sim, ...)
    diff = (legs[0::2] - legs[1::2]).reshape((2, 3) + legs.shape[1:]) / 2
    dat = collect(["data"], args.n_data)[0]
    counts = (len(mf), diff.shape[2], len(dat))
    print(f"\ncollected {counts[0]}/{args.n_meanfield} mean-field, "
          f"{counts[1]}/{args.n_response} response and "
          f"{counts[2]}/{args.n_data} data sims")
    if counts[0] < 8 or counts[1] < 3 or counts[2] < 2:
        print("too few for a report yet: run the remaining shards")
        return
    if counts != (args.n_meanfield, args.n_response, args.n_data):
        print("partial: the report and figures cover the sims cached so far")
    results = analyse(mf, diff, dat)

    thumb = enmap.downgrade(mask, max(1, round(0.5 * utils.degree / RES)))
    dec, ra = enmap.posaxes(thumb.shape, thumb.wcs)
    out = dict(v_true=C_KMS * U_TRUE, d_true=D_TRUE, beta=BETA, b_nu=B_NU, freq=FREQ, res_arcmin=RES / utils.arcmin,
               noise_t=NOISE_T, beam=BEAM, cache_tag=CACHE_TAG, stat_names=STATS,
               requested=[args.n_meanfield, args.n_response, args.n_data],
               n_mf=len(mf), n_resp=diff.shape[2], n_data=len(dat),
               lmin=LMIN, lmax=LMAX, knee=[KNEE_T, KNEE_P], alpha=[ALPHA_T, ALPHA_P], tcmb=TCMB,
               cl_tt=cltt, cl_ee=clee, nl_tt=nltt, nl_ee=nlee,
               pack_l=PACK_L, pack_m=PACK_M, fsky=W1, aber_case=ABER_CASE,
               mask_thumb=np.asarray(thumb, np.float32),
               mask_dec=np.degrees(dec), mask_ra=np.degrees(ra))
    for name, res in results.items():
        for key, val in res.items():
            out[f"{name}.{key}"] = val
    path = os.path.join(CACHE, "summary.npz")
    tmp = f"{path}.{os.getpid()}.npz"
    np.savez(tmp, **out)
    os.replace(tmp, path)
    print(f"\nwrote {path}")

    import aberration_dr6_plots
    aberration_dr6_plots.make_all(path)             # -> CACHE/plots


if __name__ == "__main__":
    main()