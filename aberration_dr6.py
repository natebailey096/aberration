"""
Doppler boost of the CMB, aberration and modulation, measured with two
quadratic estimators on the ACT DR6 lensing mask with ACT-like noise.  One
fixed configuration; the only inputs are the numbers of simulations.

    python aberration_dr6.py --n-meanfield 800 --n-response 60 --n-data 400

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
value whenever a setting changes: each tag keeps its own sims.  The raw reconstructions also go to dr6_mf<N>_resp<N>_data<N>/reconstructions.npz
as soon as the sims finish, the results to summary.npz beside it, and the
figures are drawn from summary.npz by aberration_dr6_plots.py.  Unlensed Gaussian CMB, homogeneous ACT
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
OUTDIR = f"dr6_mf{args.n_meanfield}_resp{args.n_response}_data{args.n_data}"
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


def run(label, n, one_sim):
    """Sims 0..n-1, loaded from CACHE when there and saved to it when not."""
    out, t0 = [], time.time()
    for i in range(n):
        path = os.path.join(CACHE, f"{label.replace(' ', '_')}_{i:04d}.npy")
        if os.path.exists(path):
            out.append(np.load(path))
        else:
            out.append(one_sim(i))
            np.save(path, out[-1])
        print(f"   {label} {i + 1}/{n}   {(time.time() - t0) / (i + 1):.1f} "
              f"s/sim", flush=True)
    return np.array(out)


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


def report(name, u, null, n_m):
    """Amplitude, velocity and direction of one estimate of u, printed."""
    amp = u @ U_TRUE / (U_TRUE @ U_TRUE)
    dirs = u / np.linalg.norm(u, axis=1)[:, None]
    vel = u * C_KMS
    shrink = np.sqrt(1.0 / len(u) + 1.0 / n_m)     # mean-field error is shared
    ang = np.degrees(np.arccos(np.clip(dirs @ D_TRUE, -1, 1)))
    mean_dir = u.mean(axis=0) / np.linalg.norm(u.mean(axis=0))
    sd_v = vel.std(axis=0, ddof=1)
    h = n_m // 2
    print(f"\n{name}")
    print(f"   A = {amp.mean():+.4f} +- {amp.std(ddof=1):.4f} (sd), "
          f"+- {amp.std(ddof=1) * shrink:.4f} (error on the mean)   expect 1")
    print(f"   null A = {(null @ U_TRUE).mean() / (U_TRUE @ U_TRUE):+.4f} +- "
          f"{(null @ U_TRUE).std(ddof=1) / (U_TRUE @ U_TRUE) * np.sqrt(1 / len(null) + 1 / h):.4f}"
          f"   expect 0")
    print("   v [km/s]  " + "   ".join(
        f"{'xyz'[k]} {vel[:, k].mean():+7.1f} +- {sd_v[k] * shrink:5.1f} "
        f"(in {C_KMS * U_TRUE[k]:+6.1f})" for k in range(3)))
    print(f"   direction: mean {np.degrees(np.arccos(np.clip(mean_dir @ D_TRUE, -1, 1))):.1f} deg "
          f"from input, per sim median {np.median(ang):.1f} deg")
    l_in, b_in = galactic(D_TRUE)
    l, b = galactic(dirs)
    l = l_in + (l - l_in + 180) % 360 - 180        # unwrapped around the input
    print(f"   galactic: l = {l.mean():.2f} +- {l.std(ddof=1):.2f}, "
          f"b = {b.mean():+.2f} +- {b.std(ddof=1):.2f} (mean +- sd);  "
          f"input l = {l_in:.2f}, b = {b_in:+.2f}")
    return dict(amp=amp, dir=dirs, vel=vel, null=null @ U_TRUE / (U_TRUE @ U_TRUE))


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


def analyse(mf, diff, dat):
    Y, y_mf, y_dat, K, Kerr = system((ABER_CASE, "MOD"), mf, diff, dat)
    n_m = len(y_mf)
    m = y_mf.mean(axis=0)
    C = np.cov(y_mf, rowvar=False)
    Kinv = np.linalg.inv(K)
    Kj = K[:, :3] + K[:, 3:]                   # one u drives both effects
    G = np.linalg.solve(Kj.T @ np.linalg.solve(C, Kj),
                        np.linalg.solve(C, Kj).T)
    h = n_m // 2
    null = y_mf[h:] - y_mf[:h].mean(axis=0)
    sep, sep_null = (y_dat - m) @ Kinv.T, null @ Kinv.T

    print(f"\nresponse K per unit u: rows {ABER_CASE} QE xyz, MOD QE xyz; "
          f"columns aberration xyz, modulation xyz")
    for i in range(6):
        print("   " + " ".join(f"{K[i, j]:+.3f}({Kerr[i, j] * 1e3:3.0f})"
                               for j in range(6)))
    print(f"   (MC error on the mean in units of 1e-3; b = {B_NU:.4f})")
    print(f"   |mean field| / |K u_in|:  {ABER_CASE} "
          f"{np.linalg.norm(m[:3]) / np.linalg.norm(Kj[:3] @ U_TRUE):.1f},  "
          f"MOD {np.linalg.norm(m[3:]) / np.linalg.norm(Kj[3:] @ U_TRUE):.1f}")

    out = {"aberration": report("aberration (separate)", sep[:, :3],
                                sep_null[:, :3], n_m),
           "modulation": report("modulation (separate)", sep[:, 3:],
                                sep_null[:, 3:], n_m),
           "joint": report("joint (one u for both effects)",
                           (y_dat - m) @ G.T, null @ G.T, n_m)}
    for i, name in enumerate(("aberration", "modulation")):
        out[name].update(higher_l(Y[i], i), R=K[3 * i:3 * i + 3, 3 * i:3 * i + 3],
                         Rerr=Kerr[3 * i:3 * i + 3, 3 * i:3 * i + 3])
    sd = np.sqrt(np.diag(C))
    out["system"] = dict(K=K, Kerr=Kerr, corr=C / np.outer(sd, sd))

    print("\nseparate aberration amplitude for each lensing combination")
    for case in CASES:
        _, ym, yd, Kc, _ = system((case, "MOD"), mf, diff, dat)
        a = ((yd - ym.mean(axis=0)) @ np.linalg.inv(Kc).T)[:, :3] @ U_TRUE
        a /= U_TRUE @ U_TRUE
        print(f"   {case:4s}  A = {a.mean():+.3f} +- {a.std(ddof=1):.3f} (sd)")
    return out


def main():
    print(f"lmin {LMIN}, lmax {LMAX}, mlmax {MLMAX}, {RES / utils.arcmin:g}' "
          f"CAR {shape[0]}x{shape[1]}, fsky {W1:.4f}, w2 {W2:.4f}")
    print(f"beta {BETA:.4e} towards ra {np.degrees(BDIR[0]):.2f}, "
          f"dec {np.degrees(BDIR[1]):.2f}; modulation b = {B_NU:.4f} at "
          f"{FREQ / 1e9:g} GHz", flush=True)

    print(f"noise 1/f slopes T {ALPHA_T:g}, P {ALPHA_P:g}; cache {CACHE}/")
    os.makedirs(CACHE, exist_ok=True)
    print("\n[A] mean field")
    mf = run("mf", args.n_meanfield, lambda i: reconstruct(
        cmb(1_000_000 + 2 * i) + noise(1_000_001 + 2 * i)))

    print("\n[B] response")
    diff = np.zeros((2, 3, args.n_response) + mf.shape[1:], complex)
    for k, effect in enumerate(("aberration", "modulation")):
        for j, d in enumerate(AXES):
            legs = [run(f"{effect} {'xyz'[j]}{'+-'[s < 0]}", args.n_response,
                        lambda i: reconstruct(boost(
                            cmb(2_000_000 + i), d, s * BETA,
                            aberrate=k == 0, modulate=k == 1)))
                    for s in (1, -1)]
            diff[k, j] = (legs[0] - legs[1]) / 2

    print("\n[C] data")
    dat = run("data", args.n_data, lambda i: reconstruct(
        boost(cmb(3_000_000 + 2 * i), BDIR, BETA) + noise(3_000_001 + 2 * i)))

    os.makedirs(OUTDIR, exist_ok=True)
    np.savez(os.path.join(OUTDIR, "reconstructions.npz"), mf=mf, diff=diff,
             data=dat)
    results = analyse(mf, diff, dat)

    thumb = enmap.downgrade(mask, max(1, round(0.5 * utils.degree / RES)))
    dec, ra = enmap.posaxes(thumb.shape, thumb.wcs)
    out = dict(v_true=C_KMS * U_TRUE, d_true=D_TRUE, beta=BETA, b_nu=B_NU,
               n_mf=len(mf), n_resp=diff.shape[2], n_data=len(dat),
               lmin=LMIN, lmax=LMAX, knee=[KNEE_T, KNEE_P], alpha=[ALPHA_T, ALPHA_P], tcmb=TCMB,
               cl_tt=cltt, cl_ee=clee, nl_tt=nltt, nl_ee=nlee,
               pack_l=PACK_L, pack_m=PACK_M, fsky=W1, aber_case=ABER_CASE,
               mask_thumb=np.asarray(thumb, np.float32),
               mask_dec=np.degrees(dec), mask_ra=np.degrees(ra))
    for name, res in results.items():
        for key, val in res.items():
            out[f"{name}.{key}"] = val
    np.savez(os.path.join(OUTDIR, "summary.npz"), **out)
    print(f"\nwrote {OUTDIR}/summary.npz")

    import aberration_dr6_plots
    aberration_dr6_plots.make_all(os.path.join(OUTDIR, "summary.npz"))


if __name__ == "__main__":
    main()