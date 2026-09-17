"""
Figures for the CMB aberration reconstruction.

Kept separate from the pipeline on purpose: it needs nothing but numpy and
matplotlib, so figures can be retuned in seconds from a saved summary without
touching the sims, and without needing pixell, healpy or the 0.8 GB DR6 mask.

    python aberration_plots.py cache_.../summary.npz --out figs/ --case T+P

One case at a time (default T+P).  The pipeline still analyses and prints all
four; --case picks which one gets drawn.

Figures
-------
  sky_coverage        the mask that was actually used, with the input dipole
                      and its antipode marked
  velocity_whisker    reconstructed v_x, v_y, v_z against the input, after
                      the R^-1 correction
  velocity_planes     the same sims in the xy, xz, yz planes with covariance
                      ellipses: the error shape R^-1 leaves behind
  response_sky        d.Rd, the misalignment of Rd, and sigma_v, as a function
                      of where on the sky the boost points
  response_matrix     R, R^-1 and the eigenvalues of the symmetric part
  amplitude_direction amplitude before and after correction - the one place
                      the uncorrected estimator is drawn - and the recovered
                      directions
  leakage_spectrum    coherent power the L=1 boost leaves at L >= 2

Sky maps are drawn with RA increasing to the left, the usual convention, and
the tick labels are built from the axis values rather than typed out, since a
hand-written list is one sign error away from a 12 hour shift.

Summary keys are indexed by case (0, 1, ... in the order given by `cases`):

    cases, v_true, d_true, dipole_radec, beta, c_kms, ells, clpp,
    w1, w2, w4, n_mf, n_resp, n_data, lmax, lout, res_arcmin,
    mask_key, mask_kind, mask_file, mask_variant, mask_digest, mask_at_dipole,
    mask_thumb (ndec, nra), mask_thumb_dec (ndec,), mask_thumb_ra (nra,)
    vel_<k>      (n_data, 3)   reconstructed velocity, km/s
    velraw_<k>   (n_data, 3)   the same without the R^-1 correction; stored
                               but no longer drawn, since only the amplitude
                               figure shows the uncorrected case
    amp_<k>      (n_data,)     amplitude A
    naive_<k>    (n_data,)     amplitude without the R^-1 correction
    dir_<k>      (n_data, 3)   reconstructed unit direction
    R_<k>        (3, 3)        response matrix
    leak_<k>     (lout+1,)     coherent spurious power C_L from the boost
    leakerr_<k>  (lout+1,)     jackknife error on the above
    noise_<k>    (lout+1,)     per-realisation reconstruction noise N_L
    mfcl_<k>     (lout+1,)     mean-field power
    datleak_<k>  (lout+1,)     same leakage, cross-checked on the data sims

AI Statement: written with Claude.  On the sky maps, check the dipole marker
against the RA and dec printed in the pipeline header (ra 167.9, dec -6.9)
before trusting the orientation of anything else on them.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch

# One palette for every figure, chosen to stay legible in greyscale and to
# work for the common colour deficiencies: amber and blue differ in lightness
# as well as hue, and nothing relies on red against green.
RAW = "#E69F00"          # before the response correction
COR = "#0072B2"          # after it
TRUTH = "#C02A2A"        # the input value
STAR = "#FFD400"         # the dipole marker, so it shows on a dark map
INK = "#1A1A1A"
GRID = "0.75"
COMPONENTS = [r"$v_x$", r"$v_y$", r"$v_z$"]


def _style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12.5,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "figure.dpi": 150,
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


# ---------------------------------------------------------------- helpers

def _scalar(S, key, default=np.nan):
    """A float from a summary entry, whatever shape it was stored as."""
    if key not in S:
        return default
    try:
        return float(np.asarray(S[key]).reshape(-1)[0])
    except (ValueError, TypeError):
        return default


def _text(S, key, default=""):
    if key not in S:
        return default
    return str(np.asarray(S[key]).reshape(-1)[0])


def _wrap(x):
    return (np.asarray(x, dtype=float) + 180.0) % 360.0 - 180.0


def to_radec(vec):
    """(ra, dec) in degrees for a Cartesian vector."""
    v = np.asarray(vec, dtype=float)
    v = v / np.linalg.norm(v)
    return (np.degrees(np.arctan2(v[1], v[0]) % (2 * np.pi)),
            np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0))))


# On every sky map the plot x coordinate is -(RA - lon0), so that RA increases
# to the left and the map is centred on the footprint rather than being cut in
# half by the seam.  These functions are the only places that convention lives.

def _sky_x(ra_deg, lon0=0.0):
    return -np.radians(_wrap(np.asarray(ra_deg, dtype=float) - lon0))


def _sky_point(ax, ra_deg, dec_deg, lon0=0.0, **kw):
    return ax.plot(_sky_x(ra_deg, lon0), np.radians(dec_deg), **kw)


def _lon0(S):
    """RA to centre the sky maps on: the footprint, else the dipole.

    The weighted circular mean of the kept sky.  A patch centred near RA 180
    (the default act patch, and much of the DR6 footprint) would otherwise be
    split down both edges of the map.
    """
    mesh = _mask_mesh_raw(S)
    if mesh is not None:
        W, dec, ra = mesh
        w = (W * np.cos(np.radians(dec))[:, None]).sum(axis=0)
        if w.sum() > 0:
            c, s = (w * np.cos(np.radians(ra))).sum(), \
                   (w * np.sin(np.radians(ra))).sum()
            # A footprint symmetric in RA gives no meaningful centre, so fall
            # through to the dipole rather than picking an arbitrary angle.
            # Rounded to a multiple of 30, which is the tick spacing: the
            # centring only has to be approximate, and an exact centroid of
            # 25 deg would label every meridian 175, 145, 115 ...
            if np.hypot(c, s) > 0.05 * w.sum():
                return float(30.0 * round(np.degrees(np.arctan2(s, c)) / 30.0))
    if "d_true" in S:
        return float(30.0 * round(to_radec(S["d_true"])[0] / 30.0))
    return 0.0


def _sky_mesh(lon0=0.0, n_lat=121):
    """Plot mesh for a full-sky map, plus the unit vector at every cell.

    The relative longitude descends so that the plot coordinate ascends,
    which keeps pcolormesh and contour happy.
    """
    rel = np.linspace(180.0, -180.0, 2 * n_lat - 1)
    dec = np.linspace(-90.0, 90.0, n_lat)
    REL, DEC = np.meshgrid(np.radians(rel), np.radians(dec))
    RA = REL + np.radians(lon0)
    d = np.stack([np.cos(DEC) * np.cos(RA),
                  np.cos(DEC) * np.sin(RA),
                  np.sin(DEC)], axis=-1)
    return -REL, DEC, d


def _mask_mesh_raw(S):
    """(W, dec, ra) of the stored mask thumbnail, in degrees, or None."""
    for key in ("mask_thumb", "mask_thumb_dec", "mask_thumb_ra"):
        if key not in S:
            return None
    W = np.asarray(S["mask_thumb"], dtype=float)
    dec = np.asarray(S["mask_thumb_dec"], dtype=float)
    ra = _wrap(S["mask_thumb_ra"])
    if W.shape != (dec.size, ra.size):
        return None
    return W, dec, ra


def _mask_mesh(S, lon0=0.0):
    """(X, Y, W) of the stored mask thumbnail, ready to draw, or None."""
    raw = _mask_mesh_raw(S)
    if raw is None:
        return None
    W, dec, ra = raw
    order = np.argsort(-_wrap(ra - lon0))        # plot x comes out ascending
    X, Y = np.meshgrid(_sky_x(ra[order], lon0), np.radians(dec))
    return X, Y, W[:, order]


def _dress_sky(ax, lon0=0.0, label_size=8, ra_labels=True):
    """Graticule and positive RA tick labels on a mollweide axis.

    The labels are built from the axis values rather than written out by hand:
    x is -(RA - lon0), and a hand-typed list is one sign error away from a 12
    hour shift.  They get a light background because the footprint fill can
    sit right under them.
    """
    # Above the mesh, not below it: the footprint fill crosses the equator,
    # which is exactly where mollweide puts the RA labels.
    ax.set_axisbelow(False)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.8)
    xt = np.radians(np.arange(-150, 151, 30))
    ax.set_xticks(xt)
    if not ra_labels:
        # On a small panel filled edge to edge with colour, a row of labels
        # across the equator hides more than it explains; the graticule, the
        # footprint outline and the dipole marker carry the orientation, and
        # every sky map here shares the same centring.
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels(
            [f"{int(round(-np.degrees(t) + lon0)) % 360}$\\degree$"
             for t in xt], fontsize=label_size, color="0.3")
        for lab in ax.get_xticklabels():
            lab.set_bbox(dict(fc="white", ec="none", alpha=0.7, pad=1.0))
    ax.set_yticks(np.radians([-60, -30, 0, 30, 60]))
    ax.tick_params(labelsize=label_size)


def _mask_outline(ax, S, lon0=0.0, coarsen_deg=2.0, **kw):
    """Outline of the footprint, as a thin dark line at half weight.

    The mask is block-averaged to `coarsen_deg` first.  Contouring the full
    thumbnail instead would draw a ring around each of the hundreds of
    point-source holes, which buries the boundary that the outline is for.
    """
    mesh = _mask_mesh(S, lon0)
    if mesh is None:
        return False
    X, Y, W = mesh
    f = max(1, int(round(coarsen_deg * (W.shape[0] - 1) / 180.0)))
    if f > 1:
        ny, nx = (W.shape[0] // f) * f, (W.shape[1] // f) * f
        if ny >= 2 * f and nx >= 2 * f:
            W = W[:ny, :nx].reshape(ny // f, f, nx // f, f).mean(axis=(1, 3))
            X = X[:ny, :nx].reshape(ny // f, f, nx // f, f).mean(axis=(1, 3))
            Y = Y[:ny, :nx].reshape(ny // f, f, nx // f, f).mean(axis=(1, 3))
    if not (W.min() < 0.5 < W.max()):
        return False
    opts = dict(levels=[0.5], colors=[INK], linewidths=0.9, alpha=0.8)
    opts.update(kw)
    ax.contour(X, Y, W, **opts)
    return True


def _mark_dipole(ax, S, lon0=0.0, ms=16, antipode=True):
    """The input dipole axis: a filled star, and an open one at the antipode."""
    if "d_true" not in S:
        return []
    ra_d, dec_d = to_radec(S["d_true"])
    _sky_point(ax, ra_d, dec_d, lon0, marker="*", ms=ms, mfc=STAR, mec=INK,
               mew=0.9, ls="none", zorder=8)
    handles = [Line2D([], [], marker="*", ms=13, mfc=STAR, mec=INK,
                      ls="none", label="input dipole")]
    if antipode:
        _sky_point(ax, ra_d + 180.0, -dec_d, lon0, marker="*", ms=ms * 0.7,
                   mfc="none", mec=INK, mew=1.1, ls="none", zorder=8)
        handles.append(Line2D([], [], marker="*", ms=10, mfc="none", mec=INK,
                              ls="none", label="antipode"))
    return handles


def _boxes(ax, data, positions, width, color):
    bp = ax.boxplot(data, positions=positions, widths=width,
                    whis=(5, 95), showfliers=False, patch_artist=True,
                    manage_ticks=False, medianprops=dict(color=INK, lw=1.3),
                    boxprops=dict(lw=0.9), whiskerprops=dict(lw=1.0),
                    capprops=dict(lw=1.0))
    for b in bp["boxes"]:
        b.set_facecolor(color)
        b.set_alpha(0.6)
        b.set_edgecolor(color)
    return bp


def _ellipse(ax, xy, cov, nsig, **kw):
    val, vec = np.linalg.eigh(cov)
    val = np.clip(val, 0, None)
    ang = np.degrees(np.arctan2(vec[1, -1], vec[0, -1]))
    ax.add_patch(Ellipse(xy, 2 * nsig * np.sqrt(val[-1]),
                         2 * nsig * np.sqrt(val[0]), angle=ang, **kw))


# ---------------------------------------------------------------- figures

def plot_coverage(S, path):
    """The mask that was used, with the input dipole axis on it.

    Everything here comes out of summary.npz, so this draws the real DR6
    footprint as readily as the analytic patch, and needs neither pixell nor
    the released mask file.  Both ends of the dipole axis are marked: the
    estimator is sensitive to the axis, so how much sky sits at each end is
    what matters.
    """
    _style()
    lon0 = _lon0(S)
    mesh = _mask_mesh(S, lon0)
    if mesh is None:
        raise ValueError("summary.npz has no mask thumbnail; rerun the "
                         "pipeline to write one")
    X, Y, W = mesh

    fig = plt.figure(figsize=(10.0, 5.8))
    ax = fig.add_subplot(111, projection="mollweide")

    # Zero weight is left as background rather than coloured, so the apodised
    # edge reads as a fade into blank sky instead of as a dark border.  No
    # outline on top of it: the colour already shows where the footprint is,
    # and a contour here would trace every source hole.
    pcm = ax.pcolormesh(X, Y, np.where(W > 1e-4, W, np.nan),
                        cmap=plt.get_cmap("Blues"), vmin=0.0, vmax=1.0,
                        shading="auto", rasterized=True, zorder=0)
    _dress_sky(ax, lon0)
    handles = _mark_dipole(ax, S, lon0, ms=18)

    cb = fig.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.12,
                      shrink=0.5, aspect=36)
    cb.set_label("mask weight", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    # Under the map rather than in a corner of it: the corners of a mollweide
    # frame are where the declination labels live.
    if handles:
        ax.legend(handles=handles, loc="upper center", ncol=2, fontsize=9.5,
                  bbox_to_anchor=(0.5, 0.02), columnspacing=2.0)

    w1, w2, w4 = (_scalar(S, k) for k in ("w1", "w2", "w4"))
    md = _scalar(S, "mask_at_dipole")
    bits = []
    if np.isfinite(w1):
        bits.append(f"$f_{{\\rm sky}}$ {w1:.3f} ({w1 * 41253:.0f} deg$^2$)")
    if np.isfinite(w4) and np.isfinite(w2) and w2 > 0:
        bits.append(f"$\\sqrt{{w_4}}/w_2$ {np.sqrt(w4) / w2:.2f}")
    if np.isfinite(md):
        bits.append(f"weight at the dipole {md:.2f}")
    bits.append(f"centred on RA {lon0 % 360:.0f}$^\\circ$")
    ax.set_title(f"sky coverage:  {_text(S, 'mask_key', 'mask')}\n"
                 + "     ".join(bits), fontsize=11, linespacing=1.5)

    fig.savefig(path)
    plt.close(fig)
    return path


def plot_whisker(S, path):
    """Velocity components against the input, after the R^-1 correction.

    The corrected values should sit on the input line; the spread is what a
    cut sky costs.  The uncorrected estimator is not drawn here - the
    amplitude figure already shows what R^-1 is correcting for.
    """
    _style()
    v_true = np.asarray(S["v_true"], float)
    vel = np.asarray(S["vel"], float)
    n = len(vel)

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(8.0, 6.6), sharex=True,
        gridspec_kw=dict(height_ratios=[2.3, 1.0], hspace=0.08))

    for k in range(3):
        _boxes(ax0, [vel[:, k]], [k], 0.42, COR)
        m = vel[:, k].mean()
        sem = vel[:, k].std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        ax0.errorbar([k], [m], yerr=[sem], fmt="o", ms=5.0, color=INK,
                     mfc="white", mew=1.2, elinewidth=1.4, capsize=2.8,
                     zorder=6)
        ax1.errorbar([k], [m - v_true[k]], yerr=[sem], fmt="o", ms=6.0,
                     color=COR, mec=INK, mew=0.7, elinewidth=1.8,
                     capsize=3.6, zorder=6)
        ax0.hlines(v_true[k], k - 0.45, k + 0.45, color=TRUTH, lw=2.0,
                   zorder=7)
        ax0.text(k + 0.46, v_true[k], f" {v_true[k]:+.0f}", color=TRUTH,
                 va="center", ha="left", fontsize=9.5)
    ax1.axhline(0.0, color=TRUTH, lw=1.6, zorder=7)

    ax0.set_ylabel("reconstructed velocity  [km s$^{-1}$]")
    ax1.set_ylabel("residual\n[km s$^{-1}$]")
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(COMPONENTS)
    ax1.set_xlim(-0.6, 2.6)
    for ax in (ax0, ax1):
        for x in (0.5, 1.5):
            ax.axvline(x, color="0.9", lw=0.8, zorder=0)

    handles = [Patch(facecolor=COR, alpha=0.6, edgecolor=COR,
                     label="$R^{-1}$ corrected"),
               Line2D([], [], color=TRUTH, lw=2.0, label="input"),
               Line2D([], [], color=INK, marker="o", ls="none",
                      mfc="white", label="mean $\\pm$ s.e.")]
    ax0.legend(handles=handles, ncol=1, loc="upper left", fontsize=9.5,
               frameon=True, framealpha=0.9, borderpad=0.6,
               title=f"{S['case']}:  boxes are quartiles, whiskers 5-95%",
               title_fontsize=8.5)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_planes(S, path):
    """The corrected sim cloud in the three coordinate planes, with 1 and 2
    sigma ellipses.

    R^-1 does not scale the errors evenly: it stretches whichever direction
    the footprint constrains worst, and the tilt of the ellipses is the
    off-diagonal part of R showing up as correlated errors.
    """
    _style()
    v_true = np.asarray(S["v_true"], float)
    vel = np.asarray(S["vel"], float)
    pairs = [(0, 1), (0, 2), (1, 2)]

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.4),
                             layout="constrained")
    for ax, (i, j) in zip(axes, pairs):
        ax.scatter(vel[:, i], vel[:, j], s=8, color=COR, alpha=0.35, lw=0)
        c = np.cov(vel[:, [i, j]].T)
        mu = vel[:, [i, j]].mean(axis=0)
        for ns, a in ((1, 0.95), (2, 0.5)):
            _ellipse(ax, mu, c, ns, fill=False, lw=1.6, ls="-",
                     edgecolor=COR, alpha=a, zorder=5)
        ax.plot(*mu, "o", ms=7.0, color=COR, mec=INK, mew=0.8, zorder=6,
                label="mean of the sims")
        ax.plot(v_true[i], v_true[j], "*", ms=18, color=TRUTH, mec="white",
                mew=0.8, zorder=7, label="input")
        ax.axhline(0, color="0.9", lw=0.8, zorder=0)
        ax.axvline(0, color="0.9", lw=0.8, zorder=0)
        ax.set_xlabel(f"$v_{'xyz'[i]}$  [km s$^{{-1}}$]")
        ax.set_ylabel(f"$v_{'xyz'[j]}$  [km s$^{{-1}}$]")
        ax.set_aspect("equal", adjustable="datalim")
    axes[0].legend(fontsize=9.5, loc="best",
                   title=f"{S['case']}, $R^{{-1}}$ corrected\n"
                         f"ellipses: 1 and 2$\\sigma$",
                   title_fontsize=8.5)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_response_sky(S, path):
    """What the response does as a function of where the boost points.

    All three panels are properties of the measured R (and, for sigma_v, of
    the measured covariance), evaluated for a unit boost along every direction
    on the sky.  The footprint outline and the input dipole are marked.
    """
    _style()
    R = np.asarray(S["R"], float)
    lon0 = _lon0(S)
    X, Y, d = _sky_mesh(lon0)
    Rd = d @ R.T                                   # reconstructed direction
    amp = np.einsum("...i,...i->...", d, Rd)       # d . R d
    mis = np.degrees(np.arccos(np.clip(
        np.einsum("...i,...i->...", Rd, d)
        / np.linalg.norm(Rd, axis=-1), -1, 1)))
    cov = np.cov(np.asarray(S["vel"], float).T)
    sig = np.sqrt(np.einsum("...i,ij,...j->...", d, cov, d))

    panels = [(amp, r"$\hat{d}\cdot R\,\hat{d}$", "viridis",
               "signal recovered before $R^{-1}$"),
              (mis, "misalignment of $R\\,\\hat{d}$  [deg]", "magma",
               "rotation of the reconstructed direction"),
              (sig, r"$\sigma_v$ along $\hat{d}$  [km s$^{-1}$]", "cividis",
               "per-sim error for a boost that way")]

    fig = plt.figure(figsize=(13.2, 4.6))
    for n, (Z, title, cmap, sub) in enumerate(panels):
        ax = fig.add_subplot(1, 3, n + 1, projection="mollweide")
        im = ax.pcolormesh(X, Y, Z, shading="auto", cmap=cmap,
                           rasterized=True, zorder=0)
        # White underneath, dark on top, so the outline reads on any colormap.
        _mask_outline(ax, S, lon0, colors=["white"], linewidths=1.6,
                      alpha=0.9)
        _mask_outline(ax, S, lon0, linewidths=0.7)
        handles = _mark_dipole(ax, S, lon0, ms=14, antipode=False)
        _dress_sky(ax, lon0, label_size=7, ra_labels=False)
        ax.set_title(title, fontsize=11, pad=12)
        if n == 0 and handles and "d_true" in S:
            # Labelled in place rather than with a legend: a legend inside a
            # mollweide axis lands on the declination labels, and one in
            # figure coordinates pins the tight bounding box open.
            ra_d, dec_d = to_radec(S["d_true"])
            ax.annotate("dipole",
                        xy=(_sky_x(ra_d, lon0), np.radians(dec_d)),
                        xytext=(9, -13), textcoords="offset points",
                        fontsize=8.5, color=INK,
                        bbox=dict(fc="white", ec="none", alpha=0.75,
                                  pad=1.5))
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05,
                          fraction=0.05, aspect=28)
        cb.ax.tick_params(labelsize=8.5)
        cb.set_label(sub, fontsize=9, color="0.3")
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

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), layout="constrained",
                             gridspec_kw=dict(width_ratios=[1, 1, 1.35]))
    for ax, M, name in ((axes[0], R, "$R$"), (axes[1], Ri, "$R^{-1}$")):
        vmax = np.abs(M).max()
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center",
                        fontsize=9.5,
                        color="white" if abs(M[i, j]) > 0.62 * vmax else "0.1")
        ax.set_xticks(range(3), ["x", "y", "z"])
        ax.set_yticks(range(3), ["x", "y", "z"])
        ax.set_title(name if M is Ri
                     else f"{name}    (cond {np.linalg.cond(R):.1f})",
                     fontsize=11.5)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("reconstructed")
    axes[0].set_xlabel("input")

    ax = axes[2]
    lbl = []
    for v in vec.T:
        ra, dec = to_radec(v)
        lbl.append(f"ra {ra:.0f}$^\\circ$\ndec {dec:+.0f}$^\\circ$")
    x = np.arange(3)
    ax.bar(x, val, width=0.6, color=COR, alpha=0.85, edgecolor=COR)
    for xi, v in zip(x, val):
        ax.text(xi, v, f" {v:.3f}\n  ($\\times${1 / v:.1f} noise)",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, lbl, fontsize=8.5)
    ax.set_ylim(0, max(val.max() * 1.45, 1e-3))
    ax.set_ylabel("eigenvalue of  $(R+R^{T})/2$")
    ax.set_title("directions the footprint constrains\n"
                 "1/eigenvalue is the noise $R^{-1}$ adds along that axis",
                 fontsize=10.5, linespacing=1.4)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_amplitude(S, path):
    """Amplitude before and after correction, and the recovered directions."""
    _style()
    d_true = np.asarray(S["d_true"], float)
    amp = np.asarray(S["amp"], float)
    naive = np.asarray(S["naive"], float)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    lo = min(np.percentile(naive, 0.5), np.percentile(amp, 0.5))
    hi = max(np.percentile(naive, 99.5), np.percentile(amp, 99.5))
    pad = 0.1 * (hi - lo + 1e-9)
    bins = np.linspace(lo - pad, hi + pad, 34)
    for a, colour, lbl in ((naive, RAW, "no $R^{-1}$"),
                           (amp, COR, "$R^{-1}$ corrected")):
        ax0.hist(a, bins=bins, histtype="stepfilled", lw=1.6, color=colour,
                 alpha=0.3)
        ax0.hist(a, bins=bins, histtype="step", lw=1.8, color=colour,
                 label=f"{lbl}:  {a.mean():+.3f} $\\pm$ "
                       f"{a.std(ddof=1) / np.sqrt(len(a)):.3f}")
        ax0.axvline(a.mean(), color=colour, lw=1.2, ls=(0, (3, 2)))
    ax0.axvline(1.0, color=TRUTH, lw=1.8, label="input")
    ax0.set_xlabel("amplitude $A$")
    ax0.set_ylabel("sims")
    ax0.set_ylim(top=ax0.get_ylim()[1] * 1.3)
    ax0.legend(fontsize=9.5, loc="upper right")

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
    ax1.scatter(sep * np.sin(psi), sep * np.cos(psi), s=11, color=COR,
                alpha=0.5, lw=0)
    rmax = np.percentile(sep, 98)
    # Rings chosen to suit the spread rather than from a fixed list: on a
    # cut sky the scatter can be tens of degrees, and a fixed 5/10/20 set
    # then piles all its labels on top of each other near the centre.
    cand = np.array([1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 150])
    rings = cand[(cand > 0.22 * rmax) & (cand < 1.2 * rmax)]
    if rings.size > 3:
        rings = rings[np.linspace(0, rings.size - 1, 3).round().astype(int)]
    th = np.linspace(0, 2 * np.pi, 200)
    for r in rings:
        ax1.plot(r * np.cos(th), r * np.sin(th), lw=0.8, color="0.7", ls=":")
        ax1.text(0.0, r, f"{r}$^\\circ$", fontsize=8.5, color="0.4",
                 ha="center", va="center", zorder=5,
                 bbox=dict(fc="white", ec="none", pad=1.2))
    ax1.plot(0, 0, "*", ms=18, color=TRUTH, zorder=6)
    lim = 1.15 * max(rmax, 1e-3)
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_aspect("equal")
    ax1.grid(False)
    ax1.set_xlabel("east offset  [deg]")
    ax1.set_ylabel("north offset  [deg]")
    ax1.set_title(f"recovered direction  (median "
                  f"{np.median(sep):.0f}$^\\circ$ from input)", fontsize=11)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_leakage(S, path):
    """Spurious power the L=1 boost puts at L>=2, against what else lives there."""
    _style()
    ells = np.asarray(S["ells"], int)
    beta = float(_scalar(S, "beta"))
    c1_in = 4 * np.pi * beta ** 2 / 9.0        # C_1 of the input dipole
    m = ells >= 1
    leak = np.asarray(S["leak"], float)
    err = np.asarray(S["leakerr"], float)
    noise = np.asarray(S["noise"], float)
    mfcl = np.asarray(S["mfcl"], float)
    datleak = np.asarray(S["datleak"], float)
    clpp = np.asarray(S.get("clpp", np.zeros_like(ells, float)), float)

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(8.4, 7.4), sharex=True,
        gridspec_kw=dict(height_ratios=[1.6, 1.0], hspace=0.09))

    seen = np.concatenate([leak[m], noise[m], mfcl[m], clpp[m]])
    seen = seen[np.isfinite(seen) & (seen > 0)]
    top = 10 ** np.ceil(np.log10(max(seen.max() if seen.size else c1_in,
                                     c1_in)) + 0.3)
    floor = top * 1e-7

    det = np.isfinite(err) & (leak > 2 * err)      # resolved above MC noise
    y = np.clip(np.where(det, leak,
                         np.where(np.isfinite(err), 2 * err, top)), floor, top)
    ax0.plot(ells[m], y[m], "-", lw=1.8, color=COR, label="leakage (response)")
    ax0.plot(ells[det & m], np.clip(leak[det & m], floor, top), "o", ms=4.5,
             color=COR)
    ax0.fill_between(ells[m], np.clip((leak - err)[m], floor, top),
                     np.clip((leak + err)[m], floor, top), where=det[m],
                     color=COR, alpha=0.2, lw=0)
    u = (~det) & m
    if u.any():
        ax0.plot(ells[u], y[u], "v", ms=5.0, mfc="none", color=COR,
                 ls="none", label="2$\\sigma$ upper limit")
    dm = m & (datleak > 0)
    ax0.plot(ells[dm], np.clip(datleak[dm], floor, top), "s", ms=4.0,
             mfc="none", color=RAW, ls="none", label="leakage (data sims)")
    ax0.plot(ells[m], noise[m], ls=(0, (4, 2)), lw=1.5, color="0.3",
             label="recon. noise $N_L$")
    ax0.plot(ells[m], np.clip(mfcl[m], floor, top), ls=(0, (1, 1.6)), lw=1.5,
             color="0.6", label="mean field")
    if np.any(clpp > 0):
        ax0.plot(ells[m], np.clip(clpp[m], floor, top), lw=1.8, color=TRUTH,
                 label=r"lensing $C_L^{\phi\phi}$")
    ax0.axhline(c1_in, color=INK, lw=1.0, ls=":")
    ax0.text(ells[m][-1], c1_in, "input dipole $C_1$ ", va="bottom",
             ha="right", fontsize=9, color=INK)

    ax0.set_yscale("log")
    ax0.set_ylim(floor, top)
    ax0.set_ylabel(r"$C_L$  [dimensionless $\phi$]")
    ax0.set_title("coherent power the $L=1$ boost leaves at higher $L$",
                  pad=12)
    ax0.legend(ncol=2, fontsize=9, loc="lower left", frameon=True,
               framealpha=0.9, borderpad=0.6, columnspacing=1.3,
               handlelength=1.7)

    ref = leak[1] if leak[1] > 0 else np.nan
    with np.errstate(invalid="ignore"):
        r = 100 * np.where(det, leak,
                           np.where(np.isfinite(err), 2 * err, np.nan)) / ref
    sel = m.copy()
    sel[1] = False
    ax1.plot(ells[sel], r[sel], "-", lw=1.6, color=COR)
    ax1.plot(ells[sel & det], r[sel & det], "o", ms=4.5, color=COR)
    ax1.plot(ells[sel & ~det], r[sel & ~det], "v", ms=5.0, mfc="none",
             color=COR, ls="none")
    rr = r[sel]
    rr = rr[np.isfinite(rr) & (rr > 0)]
    if rr.size:
        ax1.set_yscale("log")
        ax1.set_ylim(rr.min() / 3.0, rr.max() * 3.0)
    ax1.set_ylabel(r"leakage / recovered $L=1$  [%]")
    ax1.set_xlabel(r"multipole $L$ of the reconstruction")
    ax1.set_xlim(0.6, ells[-1] + 0.4)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.savefig(path)
    plt.close(fig)
    return path


def make_all(S, outdir):
    os.makedirs(outdir, exist_ok=True)
    jobs = [(plot_coverage, "sky_coverage.png"),
            (plot_whisker, "velocity_whisker.png"),
            (plot_planes, "velocity_planes.png"),
            (plot_response_sky, "response_sky.png"),
            (plot_response_matrix, "response_matrix.png"),
            (plot_amplitude, "amplitude_direction.png"),
            (plot_leakage, "leakage_spectrum.png")]
    done = []
    for fn, name in jobs:
        try:
            done.append(fn(S, os.path.join(outdir, name)))
        except KeyError as exc:
            # Almost always a summary.npz written by an older version of the
            # pipeline, so say so rather than printing a bare key name.
            print(f"   ({name} skipped: summary.npz has no {exc}; rerun the "
                  f"pipeline to write an up-to-date one)")
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