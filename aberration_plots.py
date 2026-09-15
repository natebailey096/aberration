"""
Figures for the CMB aberration reconstruction.

Kept separate from the pipeline on purpose: it needs nothing but numpy and
matplotlib, so figures can be retuned in seconds from a saved summary without
touching the sims.

    python aberration_plots.py cache_.../summary.npz --out figs/ --case T+P

One case at a time (default T+P).  The pipeline still analyses and prints all
four; --case picks which one gets drawn.

Figures
-------
  velocity_whisker    reconstructed v_x, v_y, v_z against the input, with and
                      without the R^-1 correction
  velocity_planes     the same sims in the xy, xz, yz planes with covariance
                      ellipses: what R^-1 does to the error shape
  response_sky        d.Rd, the misalignment of Rd, and sigma_v, as a function
                      of where on the sky the boost points
  response_matrix     R, R^-1 and the eigenvalues of the symmetric part
  amplitude_direction amplitude before and after correction, and the recovered
                      directions
  leakage_spectrum    coherent power the L=1 boost leaves at L >= 2

Summary keys are indexed by case (0, 1, ... in the order given by `cases`):

    cases, v_true, d_true, beta, c_kms, ells, clpp, w1, w2, w4, patch,
    n_mf, n_resp, n_data, lmax, lout, mask_key, fsky_target
    vel_<k>      (n_data, 3)   reconstructed velocity, km/s
    velraw_<k>   (n_data, 3)   the same without the R^-1 correction
    amp_<k>      (n_data,)     amplitude A
    naive_<k>    (n_data,)     amplitude without the R^-1 correction
    dir_<k>      (n_data, 3)   reconstructed unit direction
    R_<k>        (3, 3)        response matrix
    leak_<k>     (lout+1,)     coherent spurious power C_L from the boost
    leakerr_<k>  (lout+1,)     jackknife error on the above
    noise_<k>    (lout+1,)     per-realisation reconstruction noise N_L
    mfcl_<k>     (lout+1,)     mean-field power
    datleak_<k>  (lout+1,)     same leakage, cross-checked on the data sims
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch

RAW = "#DD8452"          # before the response correction
COR = "#4C72B0"          # after it
TRUTH = "#B22222"
INK = "#111111"
COMPONENTS = [r"$v_x$", r"$v_y$", r"$v_z$"]


def _style():
    plt.rcParams.update({
        "font.size": 10.5,
        "axes.labelsize": 11.5,
        "axes.titlesize": 12,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "figure.dpi": 140,
        "savefig.bbox": "tight",
    })


def load(path, case="T+P"):
    """summary.npz -> dict of shared metadata plus the arrays for one case."""
    with np.load(path, allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    cases = [str(c) for c in d["cases"]]
    if case not in cases:
        raise SystemExit(f"case {case!r} not in {cases}")
    k = cases.index(case)
    out = {name: d[name] for name in d
           if "_" not in name or not name.rsplit("_", 1)[-1].isdigit()}
    out["cases"], out["case"] = cases, case
    for name in ("vel", "velraw", "amp", "naive", "dir", "R", "leak",
                 "leakerr", "noise", "mfcl", "datleak"):
        if f"{name}_{k}" in d:
            out[name] = d[f"{name}_{k}"]
    return out


def _footer(fig, S, extra="", y=0.02):
    txt = (f"{S['case']},  mask {str(S['mask_key'])},  "
           f"$w_1$={float(S['w1']):.3f} $w_2$={float(S['w2']):.3f} "
           f"$w_4$={float(S['w4']):.3f},  "
           f"{int(S['n_data'])} data / {int(S['n_mf'])} mean-field / "
           f"{int(S['n_resp'])} response sims")
    if extra:
        txt += ".   " + extra
    fig.text(0.5, y, txt, ha="center", fontsize=8.8, color="0.35")


def _sky_grid(lon0=0.0, n_lat=181):
    """Plot-coordinate (lon, lat) mesh plus unit vectors at the true longitude.

    The plot coordinate runs -pi..pi so pcolormesh stays monotonic; the sky
    longitude is that plus lon0, which lets the map be centred on the patch
    instead of being cut in half by the +-180 seam.
    """
    lon = np.linspace(-np.pi, np.pi, 2 * n_lat - 1)
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    LON, LAT = np.meshgrid(lon, lat)
    true = LON + np.radians(lon0)
    d = np.stack([np.cos(LAT) * np.cos(true),
                  np.cos(LAT) * np.sin(true),
                  np.sin(LAT)], axis=-1)
    return LON, LAT, d


def _lon0(S):
    """Longitude to centre the sky maps on: the patch, else the dipole."""
    patch = S.get("patch")
    if patch is not None and np.all(np.isfinite(patch)):
        return 0.5 * (float(patch[2]) + float(patch[3]))
    d = np.asarray(S["d_true"], float)
    return np.degrees(np.arctan2(d[1], d[0]))


def _wrap(x):
    return (x + 180.0) % 360.0 - 180.0


def _patch_outline(patch, lon0=0.0, n=200):
    """Boundary of the kept rectangle as lon/lat in radians, nan at the seam."""
    if patch is None or not np.all(np.isfinite(patch)):
        return None, None
    dlo, dhi, rlo, rhi = [float(x) for x in patch]
    ra = np.concatenate([np.linspace(rlo, rhi, n), np.full(n, rhi),
                         np.linspace(rhi, rlo, n), np.full(n, rlo)])
    dec = np.concatenate([np.full(n, dlo), np.linspace(dlo, dhi, n),
                          np.full(n, dhi), np.linspace(dhi, dlo, n)])
    ra = _wrap(ra - lon0)
    # break the line where it jumps across +-180 so it does not draw across
    jump = np.abs(np.diff(ra)) > 180.0
    ra = np.insert(ra.astype(float), np.where(jump)[0] + 1, np.nan)
    dec = np.insert(dec.astype(float), np.where(jump)[0] + 1, np.nan)
    return np.radians(ra), np.radians(dec)


def _mark_sky(ax, S, lon0=0.0, label_truth=True):
    d = np.asarray(S["d_true"], float)
    lon = np.radians(_wrap(np.degrees(np.arctan2(d[1], d[0])) - lon0))
    lat = np.arcsin(np.clip(d[2], -1, 1))
    ax.plot(lon, lat, "*", ms=13, color=TRUTH, mec="white", mew=0.6, zorder=8,
            label="input dipole" if label_truth else None)
    ra_o, dec_o = _patch_outline(S.get("patch"), lon0)
    if ra_o is not None:
        ax.plot(ra_o, dec_o, lw=1.3, color="white", zorder=7)
        ax.plot(ra_o, dec_o, lw=0.7, color=INK, zorder=8)


def _boxes(ax, data, positions, width, color):
    bp = ax.boxplot(data, positions=positions, widths=width,
                    whis=(5, 95), showfliers=False, patch_artist=True,
                    manage_ticks=False, medianprops=dict(color=INK, lw=1.2),
                    boxprops=dict(lw=0.8), whiskerprops=dict(lw=0.9),
                    capprops=dict(lw=0.9))
    for b in bp["boxes"]:
        b.set_facecolor(color)
        b.set_alpha(0.55)
        b.set_edgecolor(color)
    return bp


# ---------------------------------------------------------------- figures

def plot_whisker(S, path):
    """Velocity components, with and without the response correction.

    The uncorrected boxes are what the estimator actually returns on a cut
    sky: suppressed and rotated away from the input.  The corrected ones are
    the same sims after R^-1, which should sit on the input line at the cost
    of a much wider spread.
    """
    _style()
    v_true = np.asarray(S["v_true"], float)
    series = [("no $R^{-1}$", np.asarray(S["velraw"], float), RAW),
              ("$R^{-1}$ corrected", np.asarray(S["vel"], float), COR)]

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(9.2, 7.2), sharex=True,
        gridspec_kw=dict(height_ratios=[2.2, 1.1], hspace=0.09))

    slot = 0.8 / len(series)
    for ci, (label, v, colour) in enumerate(series):
        n = len(v)
        off = (ci - (len(series) - 1) / 2.0) * slot
        for k in range(3):
            pos = [k + off]
            _boxes(ax0, [v[:, k]], pos, slot * 0.72, colour)
            m = v[:, k].mean()
            sem = v[:, k].std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
            ax0.errorbar(pos, [m], yerr=[sem], fmt="o", ms=4.2, color=INK,
                         mfc="white", mew=1.1, elinewidth=1.3, capsize=2.6,
                         zorder=6)
            ax1.errorbar(pos, [m - v_true[k]], yerr=[sem], fmt="o", ms=5.4,
                         color=colour, mec=INK, mew=0.7, elinewidth=1.6,
                         capsize=3.2, zorder=6)

    for k in range(3):
        ax0.hlines(v_true[k], k - 0.47, k + 0.47, color=TRUTH, lw=1.8, zorder=7)
        ax0.text(k + 0.47, v_true[k], f" {v_true[k]:+.0f}", color=TRUTH,
                 va="center", ha="left", fontsize=9)
    ax1.axhline(0.0, color=TRUTH, lw=1.4, zorder=7)

    ax0.set_ylabel("reconstructed velocity  [km s$^{-1}$]")
    ax1.set_ylabel("mean residual\n[km s$^{-1}$]")
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(COMPONENTS)
    ax1.set_xlim(-0.6, 2.6)
    for ax in (ax0, ax1):
        for x in (0.5, 1.5):
            ax.axvline(x, color="0.8", lw=0.7, zorder=0)

    handles = [Patch(facecolor=c, alpha=0.55, edgecolor=c, label=lbl)
               for lbl, _, c in series]
    handles += [plt.Line2D([], [], color=TRUTH, lw=1.8, label="input"),
                plt.Line2D([], [], color=INK, marker="o", ls="none",
                           mfc="white", label="mean $\\pm$ s.e.")]
    ax0.legend(handles=handles, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.13), columnspacing=1.4, handlelength=1.4)
    _footer(fig, S, "boxes: quartiles, whiskers 5-95%", y=0.035)
    fig.savefig(path)
    plt.close(fig)
    return path


def _ellipse(ax, xy, cov, nsig, **kw):
    val, vec = np.linalg.eigh(cov)
    val = np.clip(val, 0, None)
    ang = np.degrees(np.arctan2(vec[1, -1], vec[0, -1]))
    ax.add_patch(Ellipse(xy, 2 * nsig * np.sqrt(val[-1]),
                         2 * nsig * np.sqrt(val[0]), angle=ang, **kw))


def plot_planes(S, path):
    """The sim cloud in the three coordinate planes, with 1 and 2 sigma ellipses.

    R^-1 does not scale the errors evenly: it stretches whichever direction
    the footprint constrains worst, and the tilt of the corrected ellipses is
    the off-diagonal part of R showing up as correlated errors.
    """
    _style()
    v_true = np.asarray(S["v_true"], float)
    vel = np.asarray(S["vel"], float)
    raw = np.asarray(S["velraw"], float)
    pairs = [(0, 1), (0, 2), (1, 2)]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.5))
    for ax, (i, j) in zip(axes, pairs):
        for v, colour, lbl in ((raw, RAW, "no $R^{-1}$"),
                               (vel, COR, "$R^{-1}$ corrected")):
            ax.scatter(v[:, i], v[:, j], s=7, color=colour, alpha=0.3, lw=0)
            c = np.cov(v[:, [i, j]].T)
            mu = v[:, [i, j]].mean(axis=0)
            for ns, a in ((1, 0.9), (2, 0.5)):
                _ellipse(ax, mu, c, ns, fill=False, lw=1.5, ls="-",
                         edgecolor=colour, alpha=a, zorder=5)
            ax.plot(*mu, "o", ms=6, color=colour, mec=INK, mew=0.8, zorder=6,
                    label=lbl)
        ax.plot(v_true[i], v_true[j], "*", ms=17, color=TRUTH, mec="white",
                mew=0.8, zorder=7, label="input")
        ax.axhline(0, color="0.85", lw=0.7, zorder=0)
        ax.axvline(0, color="0.85", lw=0.7, zorder=0)
        ax.set_xlabel(f"$v_{'xyz'[i]}$  [km s$^{{-1}}$]")
        ax.set_ylabel(f"$v_{'xyz'[j]}$  [km s$^{{-1}}$]")
        ax.set_aspect("equal", adjustable="datalim")
    axes[0].legend(fontsize=9, loc="best")
    _footer(fig, S, "ellipses are 1 and 2 sigma of the sim scatter", y=-0.04)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_response_sky(S, path):
    """What the response does as a function of where the boost points.

    All three panels are properties of the measured R (and, for sigma_v, of
    the measured covariance), evaluated for a unit boost along every direction
    on the sky.  The footprint outline and the real dipole are marked.
    """
    _style()
    R = np.asarray(S["R"], float)
    lon0 = _lon0(S)
    LON, LAT, d = _sky_grid(lon0)
    Rd = d @ R.T                                   # reconstructed direction
    amp = np.einsum("...i,...i->...", d, Rd)       # d . R d
    mis = np.degrees(np.arccos(np.clip(
        np.einsum("...i,...i->...", Rd, d)
        / np.linalg.norm(Rd, axis=-1), -1, 1)))
    cov = np.cov(np.asarray(S["vel"], float).T)
    sig = np.sqrt(np.einsum("...i,ij,...j->...", d, cov, d))

    panels = [(amp, r"$\hat{d}\cdot R\,\hat{d}$", "viridis",
               "fraction of the signal recovered\nbefore the $R^{-1}$ correction"),
              (mis, "misalignment of $R\\,\\hat{d}$  [deg]", "magma",
               "how far the mask rotates the\nreconstructed direction"),
              (sig, r"$\sigma_v$ along $\hat{d}$  [km s$^{-1}$]", "cividis",
               "per-sim error on a velocity\npointing that way")]

    fig = plt.figure(figsize=(13.8, 4.6))
    for n, (Z, title, cmap, sub) in enumerate(panels):
        ax = fig.add_subplot(1, 3, n + 1, projection="mollweide")
        im = ax.pcolormesh(LON, LAT, Z, shading="auto", cmap=cmap,
                           rasterized=True)
        _mark_sky(ax, S, lon0, label_truth=(n == 0))
        ax.set_title(title, fontsize=11, pad=12)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, color="0.85", lw=0.4, alpha=0.6)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.06,
                          fraction=0.05, aspect=28)
        cb.ax.tick_params(labelsize=8.5)
        cb.set_label(sub, fontsize=8.5, color="0.3")
    _footer(fig, S, f"outline is the kept footprint, star is the input "
                    f"dipole; maps centred on RA {lon0:.0f}$^\\circ$", y=-0.1)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_response_matrix(S, path):
    """R, its inverse, and the eigen-structure of its symmetric part."""
    _style()
    R = np.asarray(S["R"], float)
    Ri = np.linalg.inv(R)
    val, vec = np.linalg.eigh(0.5 * (R + R.T))
    order = np.argsort(val)[::-1]
    val, vec = val[order], vec[:, order]

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), layout="constrained",
                             gridspec_kw=dict(width_ratios=[1, 1, 1.35]))
    for ax, M, name in ((axes[0], R, "$R$"), (axes[1], Ri, "$R^{-1}$")):
        vmax = np.abs(M).max()
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center",
                        fontsize=9,
                        color="white" if abs(M[i, j]) > 0.62 * vmax else "0.1")
        ax.set_xticks(range(3), ["x", "y", "z"])
        ax.set_yticks(range(3), ["x", "y", "z"])
        ax.set_title(name if M is Ri
                     else f"{name}    (cond {np.linalg.cond(R):.1f})",
                     fontsize=11)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("reconstructed")
    axes[0].set_xlabel("input")

    ax = axes[2]
    lbl = []
    for v in vec.T:
        ra = np.degrees(np.arctan2(v[1], v[0])) % 360.0
        dec = np.degrees(np.arcsin(np.clip(v[2], -1, 1)))
        lbl.append(f"ra {ra:.0f}$^\\circ$\ndec {dec:+.0f}$^\\circ$")
    x = np.arange(3)
    ax.bar(x, val, width=0.6, color=COR, alpha=0.75, edgecolor=COR)
    for xi, v in zip(x, val):
        ax.text(xi, v, f" {v:.3f}\n  ($\\times${1 / v:.1f} noise)", ha="center",
                va="bottom", fontsize=9)
    ax.set_xticks(x, lbl, fontsize=8.5)
    ax.set_ylim(0, max(val.max() * 1.45, 1e-3))
    ax.set_ylabel("eigenvalue of  $(R+R^{T})/2$")
    ax.set_title("directions the footprint constrains", fontsize=11)
    _footer(fig, S, "1/eigenvalue is how much $R^{-1}$ amplifies the noise "
                    "along that axis", y=-0.03)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_amplitude(S, path):
    """Amplitude before and after correction, and the recovered directions."""
    _style()
    d_true = np.asarray(S["d_true"], float)
    amp = np.asarray(S["amp"], float)
    naive = np.asarray(S["naive"], float)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.4, 4.4))

    lo = min(np.percentile(naive, 0.5), np.percentile(amp, 0.5))
    hi = max(np.percentile(naive, 99.5), np.percentile(amp, 99.5))
    pad = 0.1 * (hi - lo + 1e-9)
    bins = np.linspace(lo - pad, hi + pad, 34)
    for a, colour, lbl in ((naive, RAW, "no $R^{-1}$"),
                           (amp, COR, "$R^{-1}$ corrected")):
        ax0.hist(a, bins=bins, histtype="stepfilled", lw=1.6, color=colour,
                 alpha=0.28)
        ax0.hist(a, bins=bins, histtype="step", lw=1.7, color=colour,
                 label=f"{lbl}:  {a.mean():+.3f} $\\pm$ "
                       f"{a.std(ddof=1) / np.sqrt(len(a)):.3f}")
        ax0.axvline(a.mean(), color=colour, lw=1.1, ls=(0, (3, 2)))
    ax0.axvline(1.0, color=TRUTH, lw=1.7, label="input")
    ax0.set_xlabel("amplitude $A$")
    ax0.set_ylabel("sims")
    ax0.set_ylim(top=ax0.get_ylim()[1] * 1.32)
    ax0.legend(fontsize=9, loc="upper right")

    # Azimuthal-equidistant offsets about the input direction: radius is the
    # true angular separation, so it stays honest out to 180 deg.
    t = d_true / np.linalg.norm(d_true)
    east = np.cross([0.0, 0.0, 1.0], t)
    east /= np.linalg.norm(east)
    north = np.cross(t, east)
    d = np.asarray(S["dir"], float)
    d = d / np.linalg.norm(d, axis=1)[:, None]
    sep = np.degrees(np.arccos(np.clip(d @ t, -1, 1)))
    perp = d - np.outer(d @ t, t)
    nrm = np.linalg.norm(perp, axis=1)
    nrm[nrm == 0] = 1.0
    psi = np.arctan2(perp @ east / nrm, perp @ north / nrm)
    ax1.scatter(sep * np.sin(psi), sep * np.cos(psi), s=10, color=COR,
                alpha=0.45, lw=0)
    rmax = np.percentile(sep, 98)
    for r in (5, 10, 20, 30, 60, 90):
        if r > 1.25 * rmax:
            continue
        th = np.linspace(0, 2 * np.pi, 200)
        ax1.plot(r * np.cos(th), r * np.sin(th), lw=0.7, color="0.6", ls=":")
        ax1.text(r * 0.707, r * 0.707, f"{r}$^\\circ$", fontsize=8,
                 color="0.5", zorder=5)
    ax1.plot(0, 0, "*", ms=17, color=TRUTH, zorder=6)
    lim = 1.15 * max(rmax, 1e-3)
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_aspect("equal")
    ax1.set_xlabel("east offset  [deg]")
    ax1.set_ylabel("north offset  [deg]")
    ax1.set_title(f"recovered direction  (median "
                  f"{np.median(sep):.0f}$^\\circ$ from input)", fontsize=11)
    _footer(fig, S, y=-0.04)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_leakage(S, path):
    """Spurious power the L=1 boost puts at L>=2, against what else lives there."""
    _style()
    ells = np.asarray(S["ells"], int)
    beta = float(S["beta"])
    c1_in = 4 * np.pi * beta ** 2 / 9.0        # C_1 of the input dipole
    m = ells >= 1
    leak = np.asarray(S["leak"], float)
    err = np.asarray(S["leakerr"], float)
    noise = np.asarray(S["noise"], float)
    mfcl = np.asarray(S["mfcl"], float)
    datleak = np.asarray(S["datleak"], float)
    clpp = np.asarray(S.get("clpp", np.zeros_like(ells, float)), float)

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(8.6, 7.8), sharex=True,
        gridspec_kw=dict(height_ratios=[1.6, 1.0], hspace=0.1))

    seen = np.concatenate([leak[m], noise[m], clpp[m]])
    seen = seen[np.isfinite(seen) & (seen > 0)]
    top = 10 ** np.ceil(np.log10(max(seen.max() if seen.size else c1_in,
                                     c1_in)) + 0.3)
    floor = top * 1e-7

    det = np.isfinite(err) & (leak > 2 * err)      # resolved above MC noise
    y = np.clip(np.where(det, leak,
                         np.where(np.isfinite(err), 2 * err, top)), floor, top)
    ax0.plot(ells[m], y[m], "-", lw=1.6, color=COR, label="leakage (response)")
    ax0.plot(ells[det & m], np.clip(leak[det & m], floor, top), "o", ms=4,
             color=COR)
    ax0.fill_between(ells[m], np.clip((leak - err)[m], floor, top),
                     np.clip((leak + err)[m], floor, top), where=det[m],
                     color=COR, alpha=0.2, lw=0)
    u = (~det) & m
    if u.any():
        ax0.plot(ells[u], y[u], "v", ms=4.6, mfc="none", color=COR)
    dm = m & (datleak > 0)
    ax0.plot(ells[dm], np.clip(datleak[dm], floor, top), "s", ms=3.6,
             mfc="none", color=RAW, ls="none", label="leakage (data sims)")
    ax0.plot(ells[m], noise[m], ls=(0, (4, 2)), lw=1.4, color="0.35",
             label="recon. noise $N_L$")
    ax0.plot(ells[m], np.clip(mfcl[m], floor, top), ls=(0, (1, 1.6)), lw=1.4,
             color="0.6", label="mean field")
    if np.any(clpp > 0):
        ax0.plot(ells[m], np.clip(clpp[m], floor, top), lw=1.7, color=TRUTH,
                 label=r"lensing $C_L^{\phi\phi}$")
    ax0.axhline(c1_in, color="0.15", lw=1.0, ls=":")
    ax0.text(ells[m][-1], c1_in, "input dipole $C_1$ ", va="bottom",
             ha="right", fontsize=8.5, color="0.15")

    ax0.set_yscale("log")
    ax0.set_ylim(floor, top)
    ax0.set_ylabel(r"$C_L$  [dimensionless $\phi$]")
    ax0.set_title("coherent power the $L=1$ boost leaves at higher $L$", pad=30)
    ax0.legend(ncol=3, fontsize=8.8, loc="upper center",
               bbox_to_anchor=(0.5, 1.16), columnspacing=1.2, handlelength=1.6)

    ref = leak[1] if leak[1] > 0 else np.nan
    with np.errstate(invalid="ignore"):
        r = 100 * np.where(det, leak,
                           np.where(np.isfinite(err), 2 * err, np.nan)) / ref
    sel = m.copy()
    sel[1] = False
    ax1.plot(ells[sel], r[sel], "-", lw=1.5, color=COR)
    ax1.plot(ells[sel & det], r[sel & det], "o", ms=4, color=COR)
    ax1.plot(ells[sel & ~det], r[sel & ~det], "v", ms=4.6, mfc="none",
             color=COR)
    rr = r[sel]
    rr = rr[np.isfinite(rr) & (rr > 0)]
    if rr.size:
        ax1.set_yscale("log")
        ax1.set_ylim(rr.min() / 3.0, rr.max() * 3.0)
    ax1.set_ylabel(r"leakage / recovered $L=1$  [%]")
    ax1.set_xlabel(r"multipole $L$ of the reconstruction")
    ax1.set_xlim(0.6, ells[-1] + 0.4)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    _footer(fig, S, "open triangles are 2$\\sigma$ upper limits")
    fig.savefig(path)
    plt.close(fig)
    return path


def make_all(S, outdir):
    os.makedirs(outdir, exist_ok=True)
    jobs = [(plot_whisker, "velocity_whisker.png"),
            (plot_planes, "velocity_planes.png"),
            (plot_response_sky, "response_sky.png"),
            (plot_response_matrix, "response_matrix.png"),
            (plot_amplitude, "amplitude_direction.png"),
            (plot_leakage, "leakage_spectrum.png")]
    done = []
    for fn, name in jobs:
        try:
            done.append(fn(S, os.path.join(outdir, name)))
        except Exception as exc:                           # noqa: BLE001
            print(f"   ({name} skipped: {exc})")
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="figures from summary.npz")
    ap.add_argument("summary")
    ap.add_argument("--out", default=None)
    ap.add_argument("--case", default="T+P",
                    help="which estimator combination to draw (default T+P)")
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(a.summary) or ".", "plots")
    for f in make_all(load(a.summary, a.case), out):
        print("wrote", f)