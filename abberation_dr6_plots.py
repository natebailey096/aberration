"""
Figures for aberration_dr6.py, drawn from its summary.npz (numpy and
matplotlib only, so they can be redrawn without rerunning anything).

    python aberration_dr6_plots.py cache_boost/<CACHE_TAG>/summary.npz

    plots/             coverage, noise, response_6x6 (K and the correlation
                       of the six estimator components), comparison (A from
                       the three estimates, and aberration against modulation)
    plots/aberration/  lensing QE, solved without assuming the modulation
    plots/modulation/  modulation QE, solved without assuming the aberration
    plots/joint/       one velocity for both effects
    plots/report.md    the setup, where every error comes from, and the
                       mean +- error of each quantity in the figures

Boxes and ellipses are standard deviations (1 and 2 sd).  Error bars on a
mean are the pipeline's stat_err: the data, mean-field and response sims
(the last two by jackknife) added in quadrature.  report.md explains each.
"""

import argparse
import os
import time

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
                txt = f"{M[i, j]:+.3f}"
                if M is S["K"]:
                    txt += f"\n$\\pm${S['Kerr'][i, j]:.3f}"
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
    """v_x, v_y, v_z: mean +- 1 sd boxes, +- 2 sd whiskers.  Below, the
    mean minus the input with its error on the mean from all three sets of
    sims (data, mean field, and response by jackknife)."""
    vel, v_true = S["vel"], S["v_true"]
    m, sd = vel.mean(axis=0), vel.std(axis=0, ddof=1)
    err = S["stat_err"][1:4]
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
                 capsize=4, label="mean - input, +- error on the mean "
                 "(data, mean-field and response sims)")
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
    for ax, (i, j) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
        ax.scatter(vel[:, i], vel[:, j], s=6, color=col, alpha=0.35, lw=0)
        mu = vel[:, [i, j]].mean(axis=0)
        for ns in (1, 2):
            ellipse(ax, mu, np.cov(vel[:, [i, j]].T), ns, color=col, lw=1.5)
        ax.plot(*mu, "o", color=col, mec=INK, label="mean of sims")
        ax.plot(v_true[i], v_true[j], "*", ms=16, color=TRUTH, label="input")
        ax.set_xlabel(f"$v_{'xyz'[i]}$ [km/s]")
        ax.set_ylabel(f"$v_{'xyz'[j]}$ [km/s]")
        ax.set_aspect("equal", adjustable="datalim")
    axes[0].legend()
    fig.suptitle(TITLES[S["name"]])
    fig.savefig(path)
    plt.close(fig)


def plot_response_matrix(S, path):
    """The estimator's response to its own effect (its diagonal block of K)
    +- error on the mean, the inverse of that block, and the eigenvalues of
    its symmetric part."""
    R, Rerr = S["R"], S["Rerr"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
    for ax, M, name in ((axes[0], R, "$R$ (own block of $K$)"),
                        (axes[1], np.linalg.inv(R), "$R^{-1}$")):
        v = np.abs(M).max()
        im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v)
        for i in range(3):
            for j in range(3):
                txt = f"{M[i, j]:+.3f}"
                if M is R:
                    txt += f"\n$\\pm${Rerr[i, j]:.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8.5)
        ax.set_xticks(range(3), ["x", "y", "z"])
        ax.set_yticks(range(3), ["x", "y", "z"])
        ax.set_xlabel("boost along")
        ax.set_title(name)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
    val, vec = np.linalg.eigh(0.5 * (R + R.T))
    labels = []
    for v in vec.T:
        ra, dec = lonlat(v if v[2] >= 0 else -v)    # an axis: name its north end
        labels.append(f"axis ra {ra:.0f}\ndec {dec:+.0f}")
    axes[2].bar(range(3), val, color=COLOURS[S["name"]])
    axes[2].set_xticks(range(3), labels, fontsize=8.5)
    axes[2].set_ylabel("eigenvalue of $(R+R^T)/2$")
    fig.suptitle(TITLES[S["name"]])
    fig.savefig(path)
    plt.close(fig)


def plot_amplitude_direction(S, path):
    """Amplitude A, and the recovered directions in galactic coordinates:
    per-sim directions, their mean +- 1 sd in l and b, and the input."""
    amp = S["amp"]
    col = COLOURS[S["name"]]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.8),
                                   layout="constrained")
    sd = amp.std(ddof=1)
    ax0.hist(amp, bins=30, color=col, alpha=0.4,
             label=f"$A$ = {amp.mean():+.3f} $\\pm$ {sd:.3f} (sd)\n"
                   f"N: {len(amp)} data, {int(S['n_mf'])} mf, "
                   f"{int(S['n_resp'])} response")
    for k in (-2, -1, 1, 2):
        ax0.axvline(amp.mean() + k * sd, color=col, ls=":", lw=1)
    ax0.axvline(amp.mean(), color=col, ls="--")
    ax0.axvline(1.0, color=TRUTH, lw=2, label="input")
    ax0.set_xlabel("amplitude $A$")
    ax0.set_ylabel("sims")
    ax0.legend(fontsize=9)

    # l is unwrapped around the input so the cloud never splits at l = 0,
    # and runs right to left as on the sky.  The aspect 1/cos b makes equal
    # distances on the sky look equal; sigma_l is a coordinate spread.
    l_in, b_in = lonlat(S["d_true"] @ EQU2GAL.T)
    l, b = lonlat(S["dir"] @ EQU2GAL.T)
    l = l_in + (l - l_in + 180) % 360 - 180
    lm, bm, sl, sb = l.mean(), b.mean(), l.std(ddof=1), b.std(ddof=1)
    ax1.scatter(l, b, s=14, color=col, alpha=0.55, lw=0, label="per sim")
    for ns in (1, 2):
        ellipse(ax1, (lm, bm), np.cov(l, b), ns, color=col, ls="--", lw=1)
    ax1.errorbar(lm, bm, xerr=sl, yerr=sb, fmt="o", ms=7, color=col,
                 mec=INK, elinewidth=2, capsize=4, zorder=5,
                 label="mean $\\pm$ 1 sd")
    ax1.plot(l_in, b_in, "*", ms=16, color=TRUTH, zorder=6, label="input")
    ax1.invert_xaxis()
    ax1.set_aspect(1 / np.cos(np.radians(b_in)), adjustable="datalim")
    ax1.set_xlabel(r"galactic longitude $\ell$ [deg]")
    ax1.set_ylabel(r"galactic latitude $b$ [deg]")
    ax1.text(0.03, 0.03,
             f"input   $\\ell$ = {l_in:.2f}°,  $b$ = {b_in:+.2f}°\n"
             f"mean   $\\ell$ = {lm:.2f}°,  $b$ = {bm:+.2f}°\n"
             f"sd       $\\sigma_\\ell$ = {sl:.2f}°,  $\\sigma_b$ = {sb:.2f}°",
             transform=ax1.transAxes, fontsize=9, va="bottom",
             bbox=dict(fc="white", ec="0.8", alpha=0.9))
    ax1.legend(fontsize=9, loc="upper right")
    fig.suptitle(TITLES[S["name"]])
    fig.savefig(path)
    plt.close(fig)


def real_basis(A, pl, pm):
    """Packed complex alm (..., npack) -> real-harmonic r_LM, L >= 1,
    M = -L..L: r_L0 = a_L0, r_LM = sqrt2 (-1)^M Re a_LM,
    r_L,-M = -sqrt2 (-1)^M Im a_LM.  Times sqrt(3/4pi), L = 1 is (y, z, x)."""
    rows, cols = [], []
    for l in range(1, pl.max() + 1):
        for m in range(-l, l + 1):
            k = np.flatnonzero((pl == l) & (pm == abs(m)))[0]
            s = np.sqrt(2) * (-1) ** abs(m)
            cols.append(A[..., k].real if m == 0 else
                        s * A[..., k].real if m > 0 else -s * A[..., k].imag)
            rows.append((l, m))
    return rows, np.stack(cols, axis=-1)


def plot_alm_response(S, path):
    """(L, M) response to a unit boost of the estimator's own effect along
    x, y, z.  The boxed L = 1 block is R; everything below it is leakage.
    The linear band of the colour scale is the median MC error on the mean."""
    pl, pm = S["pack_l"], S["pack_m"]
    c = np.sqrt(3 / (4 * np.pi))
    rows, V = real_basis(S["Ralm"].T, pl, pm)
    V = c * V.T
    err = c * np.array([S["Ralm_sd"][(pl == l) & (pm == abs(m))][0]
                        for l, m in rows]) / np.sqrt(int(S["n_resp"]))
    thresh = float(np.median(err))
    vmax = max(np.abs(V).max(), 10 * thresh)
    fig, ax = plt.subplots(figsize=(5.4, 10), layout="constrained")
    im = ax.imshow(V, cmap="RdBu_r", aspect="auto",
                   norm=SymLogNorm(thresh, vmin=-vmax, vmax=vmax))
    for i in range(len(rows)):
        for j in range(3):
            ax.text(j, i, f"{V[i, j]:+.1e}", ha="center", va="center",
                    fontsize=7)
    lr = np.array([l for l, m in rows])
    for e in np.flatnonzero(np.diff(lr)) + 1:
        ax.axhline(e - 0.5, color="white", lw=1.6)
    ax.add_patch(Rectangle((-0.5, -0.5), 3, 3, fill=False, ec=INK, lw=2))
    ax.set_xticks(range(3), ["x", "y", "z"])
    ax.set_yticks(range(len(rows)), [f"{l},{m:+d}" for l, m in rows],
                  fontsize=7.5)
    ax.set_xlabel("boost direction")
    ax.set_ylabel("$(L, M)$")
    ax.set_title(TITLES[S["name"]], fontsize=10)
    ax.grid(False)
    fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.035,
                 label=r"$\sqrt{3/4\pi}\,\langle r_{LM}\rangle$ per unit boost")
    fig.savefig(path)
    plt.close(fig)


def plot_leakage(S, path):
    """Coherent power the true boost (both effects) leaves at each L of this
    estimator, from the paired response sims (+- 1 and 2 jackknife sd), with
    the noisier data sims minus mean field as a cross-check."""
    L = np.arange(1, S["leak"].size)
    y, e, dat = S["leak"][1:], S["leakerr"][1:], S["datleak"][1:]
    fig, ax = plt.subplots(figsize=(7.4, 5))
    ax.errorbar(L, y, yerr=2 * e, fmt="none", ecolor=COR, alpha=0.4, capsize=4)
    ax.errorbar(L, y, yerr=e, fmt="o-", color=COR, lw=1.5, elinewidth=2.5,
                label="paired response sims")
    ax.plot(L, dat, "s", mfc="none", mec="0.4", label="data sims - mean field")
    vals = np.concatenate([y, dat])
    if np.all(vals > 0):
        ax.set_yscale("log")
    else:
        ax.set_yscale("symlog", linthresh=max(np.abs(y).min(),
                                              1e-4 * np.abs(vals).max()))
        ax.axhline(0, color="0.5", lw=1)
    ax.set_xticks(L)
    ax.set_xlabel("reconstruction multipole $L$")
    ax.set_ylabel(r"$C_L$ in units of $u^2$")
    ax.set_title(TITLES[S["name"]])
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def write_report(summary, path):
    """report.md: the setup, where every number and error comes from, and
    the mean +- error of each quantity in the figures, so that a run can be
    read back long after it finished."""
    S, sy, lc = (load(summary, n) for n in (None, "system", "lensing_cases"))
    E = {n: load(summary, n) for n in TITLES}
    n_mf, n_r, n_d = (int(S[k]) for k in ("n_mf", "n_resp", "n_data"))
    req = [int(x) for x in S["requested"]]
    ra_in, dec_in = lonlat(S["d_true"])
    l_in, b_in = lonlat(S["d_true"] @ EQU2GAL.T)
    v_in = S["v_true"]
    comp = ["aber x", "aber y", "aber z", "mod x", "mod y", "mod z"]
    out = []
    w = out.append

    def corr_err(r, n):
        return (1 - r * r) / np.sqrt(n - 1)

    w("# Doppler boost simulations: report\n")
    w(f"Written {time.strftime('%Y-%m-%d %H:%M')} from `summary.npz` (cache tag "
      f"`{S['cache_tag']}`).  Simulations: {n_mf} mean-field, {n_r} response, "
      f"{n_d} data.")
    if [n_mf, n_r, n_d] != req:
        w(f"\n**Partial snapshot.**  {req[0]} / {req[1]} / {req[2]} were "
          f"requested; the rest were not in the cache yet.")

    w("\n## Setup (`coverage.png`, `noise.png`)\n")
    w("| | |\n|---|---|")
    w(f"| multipoles | {int(S['lmin'])} ≤ ℓ ≤ {int(S['lmax'])} |")
    w(f"| maps | {float(S['res_arcmin']):g}′ CAR, mask f_sky = "
      f"{float(S['fsky']):.4f} |")
    w(f"| noise (ACT f150 model) | {float(S['noise_t']):g} μK-arcmin in T, "
      f"×√2 in P; beam {float(S['beam']):g}′ FWHM; 1/f knees "
      f"{S['knee'][0]:g} (T), {S['knee'][1]:g} (P); slopes {S['alpha'][0]:g} "
      f"(T), {S['alpha'][1]:g} (P); zero below ℓ = {int(S['lmin'])} |")
    w(f"| input boost | β = {float(S['beta']):.4e} = "
      f"{np.linalg.norm(v_in):.1f} km/s toward galactic ℓ = {l_in:.2f}°, "
      f"b = {b_in:+.2f}° (ra {ra_in:.2f}°, dec {dec_in:+.2f}°) |")
    w(f"| input velocity | v_x = {v_in[0]:+.1f}, v_y = {v_in[1]:+.1f}, "
      f"v_z = {v_in[2]:+.1f} km/s (equatorial axes) |")
    w(f"| modulation factor | b = {float(S['b_nu']):.4f} at "
      f"{float(S['freq']) / 1e9:g} GHz |")
    w(f"| aberration estimator | lensing quadratic estimator, {S['aber_case']} |")

    w("\n## Where the numbers come from\n")
    w("Each estimate turns the reconstructed dipole into a boost vector "
      "u = β d.  The amplitude A = u·u_in / |u_in|² is 1 when unbiased.  "
      "Velocities are c u in km/s along the equatorial axes; ℓ and b are "
      "the galactic coordinates of each sim's direction, ℓ unwrapped around "
      "the input.\n")
    w("**Three estimates.**  *Aberration* and *modulation* are solved together "
      "from the 6×6 response K, so neither assumes the other has the same "
      "velocity.  *Joint* fits one velocity to both, weighted by the "
      "covariance of the six estimator components over the mean-field sims.\n")
    w("**sd** is the spread of single sims: the error one measurement on one "
      "sky would have.  The boxes, whiskers, ellipses and histograms show it.\n")
    w("**Error on the mean** is how well the average over all the sims is "
      "known, so it sets how precisely this run tests for bias: compare "
      "mean − input with it (last column of the tables).  It has three "
      "independent parts, added in quadrature:\n")
    w(f"1. *data*: the sd of the {n_d} data sims divided by √{n_d}.")
    w(f"2. *mean field*: one mean field, averaged over {n_mf} sims, is "
      "subtracted from every data sim, so its error does not shrink with "
      "more data sims.  It is found by jackknife: leave one mean-field sim "
      "out, redo the estimate (including the joint weights), repeat for each "
      "sim, and take (N − 1)/N times the sum of the squared deviations.")
    w(f"3. *response*: every data sim is corrected with the same K, measured "
      f"from {n_r} response sims; jackknife over the response sims in the "
      f"same way.\n")
    w("The jackknife works for every quantity, including ℓ and b, which are "
      "not linear in the sims.\n")
    w("**Other errors.**\n")
    w(f"- K and its diagonal blocks R: the sd of each element over the {n_r} "
      f"response sims divided by √{n_r}.  Eigenvalues of R: jackknife over the "
      f"response sims.")
    w("- Null test: the second half of the mean-field sims analysed with the "
      "first half as their mean field.  Expect 0; the error is "
      "sd · √(1/N₂ + 1/N₁).")
    w(f"- (L, M) response: the sd over the response sims divided by √{n_r}.")
    w("- Leakage C_L: the power of the mean response minus its Monte Carlo "
      "noise, with a jackknife error over the response sims.  The data-sim "
      "cross-check (data minus mean field) is far noisier; no error is quoted.")
    w("- Correlation coefficients r: approximate standard error "
      "(1 − r²)/√(N − 1).")

    w("\n## Response and correlations (`response_6x6.png`)\n")
    w("K per unit boost, mean ± error on the mean.  Rows: estimator "
      "components.  Columns: boost effect and axis.\n")
    w("| | " + " | ".join(comp) + " |\n|---|" + "---|" * 6)
    for i in range(6):
        w(f"| {comp[i]} QE | " + " | ".join(
            f"{sy['K'][i, j]:+.4f} ± {sy['Kerr'][i, j]:.4f}" for j in range(6))
          + " |")
    w(f"\nMean field relative to the signal, |mean field| / |K u_in|: "
      f"aberration QE {sy['mf_ratio'][0]:.2f}, modulation QE "
      f"{sy['mf_ratio'][1]:.2f}.\n")
    w(f"Correlation of the six components over the {n_mf} mean-field sims "
      f"(error about {1 / np.sqrt(n_mf - 1):.3f} near zero):\n")
    w("| | " + " | ".join(f"{c} QE" for c in comp) + " |\n|---|" + "---|" * 6)
    for i in range(6):
        w(f"| {comp[i]} QE | " + " | ".join(f"{sy['corr'][i, j]:+.3f}"
                                           for j in range(6)) + " |")

    w("\n## The three estimates side by side (`comparison.png`)\n")
    w("| estimate | A mean | error on the mean | sd (one sim) | "
      "(mean − 1)/error |\n|---|---|---|---|---|")
    for n, e in E.items():
        a, er, sd = e["stat_mean"][0], e["stat_err"][0], e["stat_sd"][0]
        w(f"| {n} | {a:+.4f} | {er:.4f} | {sd:.4f} | {(a - 1) / er:+.2f} |")
    r = np.corrcoef(E["aberration"]["amp"], E["modulation"]["amp"])[0, 1]
    w(f"\nCorrelation of A between aberration and modulation, sim by sim: "
      f"r = {r:+.3f} ± {corr_err(r, n_d):.3f}.")

    label = ["A", "v_x [km/s]", "v_y [km/s]", "v_z [km/s]", "ℓ [deg]",
             "b [deg]"]
    truth = [1.0, *v_in, l_in, b_in]
    fmt = [".4f", ".1f", ".1f", ".1f", ".2f", ".2f"]
    for n, e in E.items():
        w(f"\n## {TITLES[n][0].upper() + TITLES[n][1:]} (`{n}/`)\n")
        w("Mean ± error on the mean with its three parts, and the sd of one "
          "sim (`whisker.png`, `planes.png`, `amplitude_direction.png`).\n")
        w("| quantity | input | mean | error on the mean | data | mean field "
          "| response | sd (one sim) | (mean − input)/error |\n"
          "|---|---|---|---|---|---|---|---|---|")
        for k in range(6):
            f, mu, er = fmt[k], e["stat_mean"][k], e["stat_err"][k]
            p = e["stat_err_parts"][:, k]
            w(f"| {label[k]} | {truth[k]:+{f}} | {mu:+{f}} | {er:{f}} | "
              f"{p[0]:{f}} | {p[1]:{f}} | {p[2]:{f}} | {e['stat_sd'][k]:{f}} "
              f"| {(mu - truth[k]) / er:+.2f} |")
        nl, ne = e["null"], float(e["null_err"])
        w(f"| null A | 0 | {nl.mean():+.4f} | {ne:.4f} | | | | "
          f"{nl.std(ddof=1):.4f} | {nl.mean() / ne:+.2f} |")
        ubar = e["vel"].mean(axis=0) / np.linalg.norm(e["vel"].mean(axis=0))
        sep = np.degrees(np.arccos(np.clip(e["dir"] @ S["d_true"], -1, 1)))
        c = np.corrcoef(e["vel"].T)
        w(f"\nThe mean velocity points "
          f"{np.degrees(np.arccos(np.clip(ubar @ S['d_true'], -1, 1))):.2f}° "
          f"from the input; single sims are off by a median of "
          f"{np.median(sep):.1f}°.\n")
        w("Velocity correlations over the data sims (`planes.png`): " + ", ".join(
            f"r(v_{'xyz'[i]}, v_{'xyz'[j]}) = {c[i, j]:+.3f} ± "
            f"{corr_err(c[i, j], n_d):.3f}" for i, j in ((0, 1), (0, 2), (1, 2)))
          + ".")
        if "R" not in e:
            continue
        w("\nResponse to its own effect, the diagonal block R of K "
          "(`response_matrix.png`), mean ± error on the mean:\n")
        w("| | boost x | boost y | boost z |\n|---|---|---|---|")
        for i in range(3):
            w(f"| {'xyz'[i]} | " + " | ".join(
                f"{e['R'][i, j]:+.4f} ± {e['Rerr'][i, j]:.4f}"
                for j in range(3)) + " |")
        val, vec = np.linalg.eigh(0.5 * (e["R"] + e["R"].T))
        axes = [lonlat(v if v[2] >= 0 else -v) for v in vec.T]
        w("\nEigenvalues of (R + Rᵀ)/2 with jackknife errors: " + "; ".join(
            f"{val[k]:.4f} ± {e['eig_err'][k]:.4f} along ra {a[0]:.0f}°, "
            f"dec {a[1]:+.0f}°" for k, a in enumerate(axes)) + ".")
        w("\nLeakage (`leakage.png`): coherent power the input boost leaves "
          "at each L of this estimator, in units of u².  N_L is the "
          "reconstruction noise of one sim.\n")
        w("| L | C_L, response sims | jackknife error | data sims − mean "
          "field | N_L |\n|---|---|---|---|---|")
        for l in range(1, len(e["leak"])):
            w(f"| {l} | {e['leak'][l]:+.3e} | {e['leakerr'][l]:.1e} | "
              f"{e['datleak'][l]:+.3e} | {e['noise'][l]:.3e} |")
        pl, pm = S["pack_l"], S["pack_m"]
        cfac = np.sqrt(3 / (4 * np.pi))
        rows, V = real_basis(e["Ralm"].T, pl, pm)
        V = cfac * V.T
        err = cfac * np.array([e["Ralm_sd"][(pl == l) & (pm == abs(m))][0]
                               for l, m in rows]) / np.sqrt(n_r)
        w("\n(L, M) response to a unit boost of its own effect, real "
          "harmonics times √(3/4π), mean ± error on the mean "
          "(`alm_response.png`).  The L = 1 rows M = −1, 0, +1 are the y, z, "
          "x rows of R.\n")
        w("| (L, M) | boost x | boost y | boost z |\n|---|---|---|---|")
        for i, (l, m) in enumerate(rows):
            w(f"| {l}, {m:+d} | " + " | ".join(
                f"{V[i, j]:+.2e} ± {err[i, j]:.0e}" for j in range(3)) + " |")

    w("\n## Aberration amplitude for each lensing combination\n")
    w("The separate aberration estimate (solved with the modulation "
      "estimator) for each choice of lensing estimators.\n")
    w("| lensing estimators | A mean | error on the mean | sd (one sim) |\n"
      "|---|---|---|---|")
    for name, (a, er, sd) in zip(lc["names"], lc["A"]):
        w(f"| {name} | {a:+.4f} | {er:.4f} | {sd:.4f} |")

    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    os.replace(tmp, path)
    print("wrote", path)


PER_ESTIMATOR = [plot_whisker, plot_planes, plot_response_matrix,
                 plot_amplitude_direction, plot_alm_response, plot_leakage]


def make_all(summary, outdir=None):
    """Every figure, into plots/ beside summary.npz.  Each is drawn under a
    temporary name and renamed, so runs finishing together never leave a
    half-written file."""
    outdir = outdir or os.path.join(os.path.dirname(summary), "plots")

    def save(fn, S, path):
        tmp = f"{path[:-4]}.{os.getpid()}.png"
        fn(S, tmp)
        os.replace(tmp, path)
        print("wrote", path)

    jobs = [(outdir, None, [plot_coverage, plot_noise]),
            (outdir, "system", [plot_response_6x6]),
            (os.path.join(outdir, "aberration"), "aberration", PER_ESTIMATOR),
            (os.path.join(outdir, "modulation"), "modulation", PER_ESTIMATOR),
            (os.path.join(outdir, "joint"), "joint",
             [plot_whisker, plot_planes, plot_amplitude_direction])]
    for folder, name, fns in jobs:
        os.makedirs(folder, exist_ok=True)
        S = load(summary, name)
        for fn in fns:
            save(fn, S, os.path.join(folder, fn.__name__[5:] + ".png"))
    save(plot_comparison, [load(summary, n) for n in TITLES],
         os.path.join(outdir, "comparison.png"))
    write_report(summary, os.path.join(outdir, "report.md"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    make_all(a.summary, a.out)