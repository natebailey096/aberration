"""
Figures for aberration_dr6.py, drawn from its summary.npz (numpy and
matplotlib only, so they can be redrawn without rerunning anything).

    python aberration_dr6_plots.py dr6_mf800_resp60_data400/summary.npz

    plots/             coverage, noise, response_6x6 (K and the correlation
                       of the six estimator components), comparison (A from
                       the three estimates, and aberration against modulation)
    plots/aberration/  lensing QE, solved without assuming the modulation
    plots/modulation/  modulation QE, solved without assuming the aberration
    plots/joint/       one velocity for both effects

Boxes and ellipses are standard deviations (1 and 2 sd).  Error bars on a
mean include the mean-field term: sd * sqrt(1/N_data + 1/N_mf).
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Ellipse, Rectangle

COR, TRUTH, RAW, INK = "#0072B2", "#C02A2A", "#E69F00", "#1A1A1A"
COLOURS = {"aberration": COR, "modulation": RAW, "joint": "#009E73"}
TITLES = {"aberration": "aberration (lensing QE), separate",
          "modulation": "Doppler modulation (TT QE), separate",
          "joint": "joint: one velocity for both effects"}
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.2,
                     "legend.frameon": False, "figure.dpi": 150,
                     "savefig.bbox": "tight"})


def _rz(t):
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0],
                     [0, 0, 1]])


def _ry(t):
    return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0],
                     [-np.sin(t), 0, np.cos(t)]])


# Equatorial -> galactic (J2000): NGP at ra 192.85948, dec 27.12825, and the
# north celestial pole at l = 122.93192.
EQU2GAL = (_rz(np.radians(122.93192 - 180)) @ _ry(np.radians(27.12825 - 90))
           @ _rz(np.radians(-192.85948)))


def unit(lon, lat):
    lon, lat = np.broadcast_arrays(np.radians(lon), np.radians(lat))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=-1)


def lonlat(v):
    v = np.asarray(v) / np.linalg.norm(v, axis=-1, keepdims=True)
    return (np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360,
            np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1))))


def ellipse(ax, xy, cov, nsig, **kw):
    val, vec = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vec[1, 1], vec[0, 1]))
    ax.add_patch(Ellipse(xy, 2 * nsig * np.sqrt(val[1]),
                         2 * nsig * np.sqrt(val[0]), angle=ang, fill=False,
                         **kw))


def load(path, name=None):
    """Shared entries of summary.npz, plus those of one estimate
    (aberration, modulation, joint or system) with the prefix dropped."""
    d = dict(np.load(path))
    S = {k: v for k, v in d.items() if "." not in k}
    if name:
        S.update({k.split(".", 1)[1]: v for k, v in d.items()
                  if k.startswith(name + ".")})
        S["name"] = name
    return S


def plot_coverage(S, path):
    """Mask in equatorial Mollweide, RA increasing to the left, centred on
    RA 180, with the galactic plane and the dipole axis."""
    def x_of(ra):
        return -np.radians((np.asarray(ra) - 180 + 180) % 360 - 180)

    x = x_of(S["mask_ra"])
    order = np.argsort(x)
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(projection="mollweide")
    W = S["mask_thumb"][:, order]
    pcm = ax.pcolormesh(x[order], np.radians(S["mask_dec"]),
                        np.where(W > 1e-3, W, np.nan), cmap="Blues",
                        vmin=0, vmax=1, shading="auto", rasterized=True)
    ra, dec = lonlat(unit(np.linspace(0, 360, 721), 0.0) @ EQU2GAL)
    xs = x_of(ra)
    cut = np.flatnonzero(np.abs(np.diff(xs)) > np.pi) + 1
    for xx, yy in zip(np.split(xs, cut), np.split(np.radians(dec), cut)):
        ax.plot(xx, yy, color=TRUTH, ls="--", lw=1.5)
    ra_d, dec_d = lonlat(S["d_true"])
    ax.plot(x_of(ra_d), np.radians(dec_d), "*", ms=16, mfc="gold", mec=INK,
            label="input dipole")
    ax.plot(x_of(ra_d + 180), np.radians(-dec_d), "*", ms=12, mfc="none",
            mec=INK, label="antipode")
    ax.plot([], [], color=TRUTH, ls="--", label="galactic plane")
    xt = np.radians(np.arange(-150, 151, 30))
    ax.set_xticks(xt, [f"{(180 - d) % 360:.0f}" for d in np.degrees(xt)],
                  fontsize=8)
    ax.set_title(f"mask, f_sky = {float(S['fsky']):.3f}")
    fig.colorbar(pcm, orientation="horizontal", shrink=0.4, pad=0.06,
                 label="mask weight")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=3,
              fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def plot_noise(S, path):
    """Theory signal and the noise the filters use, as D_l in uK^2."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    for ax, name, knee in zip(axes, ("tt", "ee"), S["knee"]):
        cl, nl = S[f"cl_{name}"], S[f"nl_{name}"]
        ell = np.arange(cl.size)
        dl = ell * (ell + 1) / (2 * np.pi) * float(S["tcmb"]) ** 2
        ax.plot(ell[2:], (dl * cl)[2:], color=COR,
                label=f"$C_\\ell$ {name.upper()}")
        ax.plot(ell[nl > 0], (dl * nl)[nl > 0], color=RAW, ls="--",
                label=f"$N_\\ell$ {name.upper()}")
        ax.axvspan(S["lmin"], S["lmax"], color=COR, alpha=0.07,
                   label="used by the estimators")
        ax.axvline(knee, color=RAW, ls=":", label=r"$\ell_{\rm knee}$")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$ [$\mu$K$^2$]")
        ax.legend(fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def plot_response_6x6(S, path):
    """K per unit boost (rows: the six estimator components, columns: the
    six boosts), and the correlation of the six components over the
    mean-field sims.  Off-diagonal blocks are each QE's response to the
    other effect."""
    labels = ["aber x", "aber y", "aber z", "mod x", "mod y", "mod z"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    for ax, M, title, v in ((axes[0], S["K"], "response $K$ per unit boost",
                             np.abs(S["K"]).max()),
                            (axes[1], S["corr"], "correlation of the estimator "
                             "components (mean-field sims)", 1.0)):
        im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v)
        for i in range(6):
            for j in range(6):
                txt = f"{M[i, j]:+.2f}"
                if M is S["K"]:
                    txt += f"\n$\\pm${S['Kerr'][i, j]:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)
        ax.axhline(2.5, color="k", lw=1.5)
        ax.axvline(2.5, color="k", lw=1.5)
        ax.set_yticks(range(6), [f"{l} QE" for l in labels])
        ax.set_xticks(range(6), labels if M is S["K"] else
                      [f"{l} QE" for l in labels], rotation=45)
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0].set_xlabel("boost: effect and axis")
    fig.savefig(path)
    plt.close(fig)


def plot_comparison(S3, path):
    """A from the three estimates, and aberration against modulation sim by
    sim: their correlation is what the joint fit uses."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.6),
                                   layout="constrained")
    allamp = np.concatenate([S["amp"] for S in S3])
    bins = np.linspace(*np.percentile(allamp, [0.5, 99.5]), 40)
    for S in S3:
        a = S["amp"]
        ax0.hist(a, bins=bins, histtype="step", lw=1.8,
                 color=COLOURS[S["name"]],
                 label=f"{S['name']}: {a.mean():+.3f} $\\pm$ "
                       f"{a.std(ddof=1):.3f} (sd)")
    ax0.axvline(1.0, color=TRUTH, lw=2, label="input")
    ax0.set_xlabel("amplitude $A$")
    ax0.set_ylabel("sims")
    ax0.legend(fontsize=9)
    a, m = S3[0]["amp"], S3[1]["amp"]
    ax1.scatter(a, m, s=8, color=INK, alpha=0.4, lw=0)
    ax1.plot(1, 1, "*", ms=16, color=TRUTH)
    ax1.set_xlabel("$A$ aberration")
    ax1.set_ylabel("$A$ modulation")
    ax1.set_title(f"correlation {np.corrcoef(a, m)[0, 1]:+.2f}")
    fig.savefig(path)
    plt.close(fig)


def plot_whisker(S, path):
    """v_x, v_y, v_z: mean +- 1 sd boxes, +- 2 sd whiskers."""
    vel, v_true = S["vel"], S["v_true"]
    m, sd = vel.mean(axis=0), vel.std(axis=0, ddof=1)
    err = sd * np.sqrt(1 / len(vel) + 1 / int(S["n_mf"]))
    stats = [dict(med=m[k], q1=m[k] - sd[k], q3=m[k] + sd[k],
                  whislo=m[k] - 2 * sd[k], whishi=m[k] + 2 * sd[k],
                  fliers=[]) for k in range(3)]
    col = COLOURS[S["name"]]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                   gridspec_kw=dict(height_ratios=[2.3, 1]))
    bp = ax0.bxp(stats, positions=range(3), widths=0.45, patch_artist=True,
                 medianprops=dict(color=INK))
    for b in bp["boxes"]:
        b.set(facecolor=col, alpha=0.5, edgecolor=col)
    ax0.hlines(v_true, np.arange(3) - 0.35, np.arange(3) + 0.35, color=TRUTH,
               lw=2, label="input", zorder=5)
    ax0.set_ylabel("velocity [km/s]")
    ax0.set_title(TITLES[S["name"]])
    ax0.legend(loc="upper left")
    ax1.errorbar(range(3), m - v_true, yerr=err, fmt="o", color=col,
                 capsize=4, label="mean - input, +- error on the mean")
    ax1.axhline(0, color=TRUTH)
    ax1.set_ylabel("residual [km/s]")
    ax1.set_xticks(range(3), ["$v_x$", "$v_y$", "$v_z$"])
    ax1.legend(fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def plot_planes(S, path):
    """The sims in the xy, xz, yz planes with 1 and 2 sd ellipses."""
    vel, v_true = S["vel"], S["v_true"]
    col = COLOURS[S["name"]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout="constrained")