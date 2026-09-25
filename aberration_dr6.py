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


PER_ESTIMATOR = [plot_whisker, plot_planes, plot_response_matrix,
                 plot_amplitude_direction, plot_alm_response, plot_leakage]


def make_all(summary, outdir=None):
    outdir = outdir or os.path.join(os.path.dirname(summary), "plots")
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
            path = os.path.join(folder, fn.__name__[5:] + ".png")
            fn(S, path)
            print("wrote", path)
    path = os.path.join(outdir, "comparison.png")
    plot_comparison([load(summary, n) for n in TITLES], path)
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    make_all(a.summary, a.out)