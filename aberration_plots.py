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
  sky_coverage        the mask that was actually used, drawn twice: once in
                      equatorial and once in galactic coordinates.  Both
                      panels carry both reference planes, each in one style
                      throughout (galactic plane red dashed, celestial
                      equator amber dash-dot), and the title reports what
                      fraction of the galactic plane lies in the footprint
                      and the smallest |b| the footprint reaches: measured
                      once, from the mask alone, so the two projections
                      cannot be read as disagreeing
  velocity_whisker    reconstructed v_x, v_y, v_z against the input, after
                      the R^-1 correction
  velocity_planes     the same sims in the xy, xz, yz planes with covariance
                      ellipses: the error shape R^-1 leaves behind
  response_matrix     R, R^-1 and the eigenvalues of the symmetric part
  amplitude_direction the R^-1 corrected amplitude and the recovered
                      directions; the uncorrected estimator is stored but
                      no longer drawn anywhere
  alm_response        the response to a boost along x, y and z at every
                      (L, M) out to LEAK_LMAX, one signed number per mode in
                      the real spherical-harmonic basis, scaled so that the
                      boxed L=1 block is the response matrix itself; the rest
                      is leakage.  Reported as the coefficients themselves
                      rather than collapsed into a power spectrum
  noise_spectra_tt    the theory signal and the CMB noise the filters were
  noise_spectra_ee    built from, both as D_l, temperature and polarisation
                      on separate figures, always out to NOISE_PLOT_LMAX
  leakage_spectrum    coherent power the boost leaves at each L out to
                      LEAK_LMAX, measured from the aberrated data sims
                      (mean field subtracted), with 1 and 2 sd bars

Scatter is quoted as a standard deviation throughout: the amplitude histogram
marks the mean +- 1 and 2 sd, the velocity boxes span the mean +- 1 sd with
whiskers at +- 2, and the covariance ellipses are 1 and 2 sigma.  Where the
error on a *mean* is the meaningful quantity, as in the residual panel and the
response matrix, it is drawn as well and labelled as such.

Sky maps are drawn with RA increasing to the left, the usual convention, and
the tick labels are built from the axis values rather than typed out, since a
hand-written list is one sign error away from a 12 hour shift.

Summary keys are indexed by case (0, 1, ... in the order given by `cases`):

    cases, v_true, d_true, dipole_radec, beta, c_kms, ells, clpp,
    w1, w2, w4, n_mf, n_resp, n_data, lmax, lout, res_arcmin,
    mask_key, mask_kind, mask_file, mask_variant, mask_digest, mask_at_dipole,
    mask_thumb (ndec, nra), mask_thumb_dec (ndec,), mask_thumb_ra (nra,)
    pack_l, pack_m           (npack,) the (L, M) of each stored coefficient
    n_resp_used              paired response sims behind the alm response
    mask_method              how the healpix mask was put on the CAR grid
    vel_<k>      (n_data, 3)   reconstructed velocity, km/s
    velraw_<k>   (n_data, 3)   the same without the R^-1 correction; stored
                               but not drawn
    amp_<k>      (n_data,)     amplitude A
    naive_<k>    (n_data,)     amplitude without the R^-1 correction;
                               stored but not drawn
    dir_<k>      (n_data, 3)   reconstructed unit direction
    R_<k>        (3, 3)        response matrix
    Rsd_<k>      (3, 3)        per-sim scatter of its elements
    Rerr_<k>     (3, 3)        error on the mean of its elements
    Ralm_<k>     (npack, 3)    (L,M) response to a unit boost along x, y, z
    Ralm_sd_re_<k>, Ralm_sd_im_<k>
                 (npack, 3)    per-sim scatter of its real and imaginary parts
    leak_<k>     (lout+1,)     coherent spurious power C_L from the boost
    leakerr_<k>  (lout+1,)     jackknife error on the above
    noise_<k>    (lout+1,)     per-realisation reconstruction noise N_L
    mfcl_<k>     (lout+1,)     mean-field power
    datleak_<k>  (lout+1,)     same leakage, cross-checked on the data sims

AI Statement: written with Claude.  On the sky maps, check the dipole marker
against the RA and dec printed in the pipeline header (ra 167.9, dec -6.9)
before trusting the orientation of anything else on them.  The galactic panel
is worth the same check: that dipole should land near l = 264, b = +48, which
is what pixell's aberration.dir_gal says, and the rotation here is built from
the three defining angles rather than copied in as a matrix so it can be
re-derived if it is ever doubted.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import patheffects
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Ellipse, Patch, Rectangle

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

# Overlays on the sky maps.  Each reference plane has one style, used in every
# panel, so the eye can follow the same line from one frame to the other, and
# each carries a white halo.  The footprint fill runs to dark navy, and a thin
# dark line laid over it all but vanishes: that is how the galactic plane once
# looked as if it went round the patch in the equatorial panel while plainly
# crossing it in the galactic one, when it was cutting across it in both.
HALO = [patheffects.Stroke(linewidth=4.4, foreground="white", alpha=0.95),
        patheffects.Normal()]
HALO_THIN = [patheffects.Stroke(linewidth=3.0, foreground="white",
                                alpha=0.95),
             patheffects.Normal()]
GAL_STYLE = dict(color=TRUTH, lw=1.8, ls=(0, (5, 3)), zorder=6,
                 path_effects=HALO)
GAL_BAND_STYLE = dict(color=TRUTH, lw=1.1, ls=(0, (1, 2.4)), zorder=6,
                      path_effects=HALO_THIN)
EQU_STYLE = dict(color=RAW, lw=1.8, ls=(0, (7, 2, 1.5, 2)), zorder=6,
                 path_effects=HALO)
# Where a footprint counts as covering a point.  Half weight is the edge of an
# apodised mask, the same level the outline contour elsewhere uses.
FOOTPRINT_LEVEL = 0.5

# Highest L drawn on the higher-L figures.  The pipeline reconstructs out to
# --lout, 5 by default, and the alm response is reported over all of it.
LEAK_LMAX = 5

# Highest multipole the noise figures draw.  Fixed rather than taken from
# lmax, so that runs with different lmax are drawn on the same axis.
NOISE_PLOT_LMAX = 3000

# Equatorial -> galactic rotation, built from the three angles that define the
# frame rather than typed in as nine numbers: the north galactic pole, and the
# galactic longitude of the north celestial pole (J2000/ICRS).  Applied to
# pixell's aberration.dir_equ it returns its dir_gal to every printed digit,
# which is the check worth doing if this is ever edited.
_A_NGP, _D_NGP, _L_NCP = 192.85948, 27.12825, 122.93192


def _rot_z(t):
    c, s_ = np.cos(t), np.sin(t)
    return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_y(t):
    c, s_ = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])


# spin the NGP meridian to longitude 0, tip the NGP onto the pole, then spin
# again so that the north celestial pole lands at l = _L_NCP.
EQU2GAL = (_rot_z(np.radians(_L_NCP - 180.0))
           @ _rot_y(np.radians(_D_NGP - 90.0))
           @ _rot_z(np.radians(-_A_NGP)))
GAL2EQU = EQU2GAL.T


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
    for name in ("vel", "velraw", "amp", "naive", "dir", "R", "Rsd", "Rerr",
                 "Ralm", "Ralm_sd_re", "Ralm_sd_im", "leak",
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


def _to_vec(lon_deg, lat_deg):
    """Unit vectors from angles in degrees, stacked along the last axis."""
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    return np.stack([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=-1)


def _from_vec(v):
    """(lon, lat) in degrees for unit vectors stacked along the last axis."""
    v = np.asarray(v, dtype=float)
    return (np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360.0,
            np.degrees(np.arcsin(np.clip(v[..., 2], -1.0, 1.0))))


def equ_to_gal(ra_deg, dec_deg):
    """Equatorial (ra, dec) -> galactic (l, b), both in degrees."""
    return _from_vec(_to_vec(ra_deg, dec_deg) @ EQU2GAL.T)


def gal_to_equ(l_deg, b_deg):
    """Galactic (l, b) -> equatorial (ra, dec), both in degrees."""
    return _from_vec(_to_vec(l_deg, b_deg) @ GAL2EQU.T)


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


def _sample_thumb(W, dec, ra, dec_q, ra_q):
    """Bilinear sample of the mask thumbnail at arbitrary (dec, ra), degrees.

    RA wraps and dec clamps at the poles.  This is what lets the galactic panel
    be drawn from the same stored thumbnail as the equatorial one, with no
    pixell and no second copy of the mask: every galactic grid point is turned
    back into (ra, dec) and read off here.
    """
    ra_e = np.concatenate([ra, [ra[0] + 360.0]])
    W_e = np.concatenate([W, W[:, :1]], axis=1)

    dq = np.clip(np.asarray(dec_q, float), dec[0], dec[-1])
    rq = ra[0] + np.mod(np.asarray(ra_q, float) - ra[0], 360.0)

    iy = np.clip(np.searchsorted(dec, dq) - 1, 0, dec.size - 2)
    ix = np.clip(np.searchsorted(ra_e, rq) - 1, 0, ra_e.size - 2)
    ty = np.clip((dq - dec[iy]) / np.maximum(dec[iy + 1] - dec[iy], 1e-12),
                 0.0, 1.0)
    tx = np.clip((rq - ra_e[ix]) / np.maximum(ra_e[ix + 1] - ra_e[ix], 1e-12),
                 0.0, 1.0)
    return ((1 - ty) * ((1 - tx) * W_e[iy, ix] + tx * W_e[iy, ix + 1])
            + ty * ((1 - tx) * W_e[iy + 1, ix] + tx * W_e[iy + 1, ix + 1]))


def _sky_curve(ax, lon_deg, lat_deg, lon0=0.0, **kw):
    """A curve on a sky map, split where it crosses the seam.

    Without the split the line runs the whole way back across the map every
    time the longitude wraps.
    """
    x = _sky_x(lon_deg, lon0)
    y = np.radians(np.asarray(lat_deg, dtype=float))
    cut = np.where(np.abs(np.diff(x)) > np.pi)[0] + 1
    for xs, ys in zip(np.split(x, cut), np.split(y, cut)):
        if xs.size > 1:
            ax.plot(xs, ys, **kw)


def _dress_sky(ax, lon0=0.0, label_size=8, ra_labels=True):
    """Graticule and positive longitude tick labels on a mollweide axis.

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


def _mark_dipole(ax, S, lon0=0.0, ms=16, antipode=True, frame="equ"):
    """The input dipole axis: a filled star, and an open one at the antipode."""
    if "d_true" not in S:
        return []
    ra_d, dec_d = to_radec(S["d_true"])
    if frame == "gal":
        ra_d, dec_d = equ_to_gal(ra_d, dec_d)
    _sky_point(ax, ra_d, dec_d, lon0, marker="*", ms=ms, mfc=STAR, mec=INK,
               mew=0.9, ls="none", zorder=8, path_effects=HALO_THIN)
    handles = [Line2D([], [], marker="*", ms=13, mfc=STAR, mec=INK,
                      ls="none", label="input dipole")]
    if antipode:
        # The antipode of (l, b) is (l+180, -b) in any spherical frame, so this
        # is right in both without a second rotation.
        _sky_point(ax, ra_d + 180.0, -dec_d, lon0, marker="*", ms=ms * 0.7,
                   mfc="none", mec=INK, mew=1.1, ls="none", zorder=8,
                   path_effects=HALO_THIN)
        handles.append(Line2D([], [], marker="*", ms=10, mfc="none", mec=INK,
                              ls="none", label="antipode"))
    return handles


def _boxes(ax, data, positions, width, color):
    """Box and whisker drawn on standard deviations, not quartiles.

    The box spans the mean +- 1 sd and the whiskers the mean +- 2 sd, with the
    central line at the mean.  matplotlib's boxplot has no option for that, so
    the statistics are computed here and handed to bxp directly.  For a
    Gaussian the box would hold 68% and the whiskers 95%, which is close to the
    quartile and 5-95% box this replaces, but these are the numbers quoted in
    the text, so the figure and the printout now agree by construction.
    """
    stats = []
    for x in data:
        x = np.asarray(x, dtype=float)
        m = x.mean()
        sd = x.std(ddof=1) if x.size > 1 else 0.0
        stats.append(dict(label="", mean=m, med=m, q1=m - sd, q3=m + sd,
                          whislo=m - 2 * sd, whishi=m + 2 * sd, fliers=[]))
    bp = ax.bxp(stats, positions=positions, widths=width, showfliers=False,
                patch_artist=True, manage_ticks=False, shownotches=False,
                medianprops=dict(color=INK, lw=1.3),
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

def _plane_overlap(raw):
    """How much of the galactic plane the footprint covers, and how near it
    comes.  Computed once and quoted once, for both panels together.

    Returns (fraction of the plane's length inside the footprint, the smallest
    |b| of any footprint cell in degrees).  The fraction samples the plane at
    uniform l, which is uniform arc length on a great circle.  The |b| comes
    from converting every thumbnail cell above FOOTPRINT_LEVEL straight to
    galactic coordinates, with no resampling, so it is good to the thumbnail
    cell size.  Both are properties of the mask alone: neither panel's
    projection enters, which is the point.
    """
    W, dec, ra = raw
    l = np.linspace(0.0, 360.0, 7200, endpoint=False)
    r, d = gal_to_equ(l, np.zeros_like(l))
    frac = float(np.mean(_sample_thumb(W, dec, ra, d, r) > FOOTPRINT_LEVEL))
    keep = W > FOOTPRINT_LEVEL
    if not keep.any():
        return frac, np.nan
    DEC, RA = np.meshgrid(dec, ra, indexing="ij")
    _, b = equ_to_gal(RA[keep], DEC[keep])
    return frac, float(np.abs(b).min())


def _coverage_panel(fig, gs_index, S, frame, lon0, raw):
    """One mollweide panel of the footprint, in the given frame."""
    W, dec, ra = raw
    ax = fig.add_subplot(1, 2, gs_index, projection="mollweide")

    if frame == "equ":
        # Draw the stored grid directly: no resampling, so this panel shows the
        # thumbnail exactly as the pipeline wrote it.
        order = np.argsort(-_wrap(ra - lon0))
        X, Y = np.meshgrid(_sky_x(ra[order], lon0), np.radians(dec))
        Z = W[:, order]
    else:
        # A regular galactic grid, each cell read back out of the equatorial
        # thumbnail.  Step chosen to match the thumbnail so nothing is invented.
        step = min(abs(dec[1] - dec[0]), 0.5) if dec.size > 1 else 0.5
        # Relative longitude, descending from +180 so that the plot x
        # coordinate -radians(rel) comes out ascending.  Built directly rather
        # than through _sky_x: that wraps, and wrapping maps both ends of the
        # grid onto +180, which leaves pcolormesh with a non-monotonic axis.
        nlon = int(round(360.0 / step)) + 1
        rel = np.linspace(180.0, -180.0, nlon)
        lat = np.linspace(-90.0, 90.0, int(round(180.0 / step)) + 1)
        LON, LAT = np.meshgrid(rel + lon0, lat)
        ra_q, dec_q = gal_to_equ(LON, LAT)
        Z = _sample_thumb(W, dec, ra, dec_q, ra_q)
        X, Y = np.meshgrid(-np.radians(rel), np.radians(lat))

    pcm = ax.pcolormesh(X, Y, np.where(Z > 1e-4, Z, np.nan),
                        cmap=plt.get_cmap("Blues"), vmin=0.0, vmax=1.0,
                        shading="auto", rasterized=True, zorder=0)
    _dress_sky(ax, lon0)

    # Both reference planes on both panels, each in the one style it has
    # everywhere: the galactic plane (with |b| = 30 dotted either side) and
    # the celestial equator.  In its own frame each is a straight line along
    # the map's equator; in the other frame it is the curve.  Drawn densely
    # because a great circle far from the map's equator bends sharply.
    t = np.linspace(0.0, 360.0, 2881)
    zero = np.zeros_like(t)
    if frame == "equ":
        gal = [gal_to_equ(t, zero + b) for b in (0.0, 30.0, -30.0)]
        equ = (t, zero)
        gc = gal_to_equ(0.0, 0.0)
    else:
        gal = [(t, zero + b) for b in (0.0, 30.0, -30.0)]
        equ = equ_to_gal(t, zero)
        gc = (0.0, 0.0)
    for c in gal[1:]:
        _sky_curve(ax, *c, lon0, **GAL_BAND_STYLE)
    _sky_curve(ax, *equ, lon0, **EQU_STYLE)
    _sky_curve(ax, *gal[0], lon0, **GAL_STYLE)

    ax.plot(_sky_x(gc[0], lon0), np.radians(gc[1]), marker="+", ms=10,
            mew=1.8, color=TRUTH, ls="none", zorder=7, path_effects=HALO)
    _mark_dipole(ax, S, lon0, ms=18, frame=frame)

    ax.set_title("equatorial" if frame == "equ" else "galactic", fontsize=11)
    return ax, pcm


def plot_coverage(S, path):
    """The mask that was used, in equatorial and in galactic coordinates.

    Everything here comes out of summary.npz, so this draws the real DR6
    footprint as readily as the analytic patch, and needs neither pixell nor
    the released mask file.  Both ends of the dipole axis are marked: the
    estimator is sensitive to the axis, so how much sky sits at each end is
    what matters.

    The galactic panel is the same thumbnail read on a galactic grid, not a
    second mask, so the two panels cannot disagree about the footprint.  Both
    panels carry both planes in matching styles, and the overlap between the
    footprint and the galactic plane is measured once and printed once,
    rather than left to be judged by eye from two different projections.
    """
    _style()
    raw = _mask_mesh_raw(S)
    if raw is None:
        raise ValueError("summary.npz has no mask thumbnail; rerun the "
                         "pipeline to write one")

    fig = plt.figure(figsize=(14.0, 5.8))
    ax_e, pcm = _coverage_panel(fig, 1, S, "equ", _lon0(S), raw)
    # The galactic panel is centred on the galactic centre, the usual view.
    ax_g, _ = _coverage_panel(fig, 2, S, "gal", 0.0, raw)

    cb = fig.colorbar(pcm, ax=[ax_e, ax_g], orientation="horizontal",
                      pad=0.10, shrink=0.35, aspect=36)
    cb.set_label("mask weight", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    def _legend_line(style, label):
        keep = {k: v for k, v in style.items()
                if k in ("color", "lw", "ls")}
        return Line2D([], [], label=label, **keep)

    handles = [
        _legend_line(GAL_STYLE, "galactic plane"),
        _legend_line(GAL_BAND_STYLE, "$|b| = 30^\\circ$"),
        _legend_line(EQU_STYLE, "celestial equator"),
        Line2D([], [], color=TRUTH, marker="+", ms=9, mew=1.8, ls="none",
               label="galactic centre"),
        Line2D([], [], marker="*", ms=13, mfc=STAR, mec=INK, ls="none",
               label="input dipole"),
        Line2D([], [], marker="*", ms=10, mfc="none", mec=INK, ls="none",
               label="antipode"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.03), columnspacing=1.8, frameon=False,
               handlelength=3.2)

    # The numbers that used to sit in the title go to the terminal instead,
    # so the figure stays clean and the overlap check is still on record.
    w1 = _scalar(S, "w1")
    frac, bmin = _plane_overlap(raw)
    print(f"   sky coverage: {_text(S, 'mask_key', 'mask')}"
          + (f", fsky {w1:.3f}" if np.isfinite(w1) else "")
          + f", galactic plane {100 * frac:.0f}% inside the footprint"
          + (f", footprint reaches |b| = {bmin:.1f} deg"
             if np.isfinite(bmin) else ""))
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_whisker(S, path):
    """Velocity components against the input, after the R^-1 correction.

    The corrected values should sit on the input line; the spread is what a
    cut sky costs.  Boxes span the mean +- 1 standard deviation and whiskers
    the mean +- 2, so the figure quotes the same numbers as the printout
    rather than quartiles.  The uncorrected estimator is not drawn here, nor
    on any other figure; the pipeline still prints it.
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
        # The residual panel is about whether the mean sits on the input, so
        # its bar is the error on that mean; the sd is the box above.
        ax1.errorbar([k], [m - v_true[k]], yerr=[sem], fmt="o", ms=6.0,
                     color=COR, mec=INK, mew=0.7, elinewidth=2.6,
                     capsize=3.6, zorder=6)
        ax0.hlines(v_true[k], k - 0.45, k + 0.45, color=TRUTH, lw=2.0,
                   zorder=7)
    ax1.axhline(0.0, color=TRUTH, lw=1.6, zorder=7)

    # Room above the tallest whisker for the legend, which otherwise lands on
    # whichever component happens to sit highest.
    top = max((vel[:, k].mean() + 2 * vel[:, k].std(ddof=1)) for k in range(3))
    bot = min((vel[:, k].mean() - 2 * vel[:, k].std(ddof=1)) for k in range(3))
    span = top - bot
    ax0.set_ylim(bot - 0.06 * span, top + 0.42 * span)

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
                      mfc="white", label="mean $\\pm$ error on the mean")]
    ax0.legend(handles=handles, ncol=1, loc="upper left", fontsize=9.5,
               frameon=True, framealpha=0.9, borderpad=0.6)
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
    axes[0].legend(fontsize=9.5, loc="best")
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
    Rerr = np.asarray(S["Rerr"], float) if "Rerr" in S else None
    for ax, M, name in ((axes[0], R, "$R$"), (axes[1], Ri, "$R^{-1}$")):
        vmax = np.abs(M).max()
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        for i in range(3):
            for j in range(3):
                txt = f"{M[i, j]:+.3f}"
                # Only R has a measured error.  R^-1 is a nonlinear function
                # of it, so quoting the same number there would be wrong.
                if M is R and Rerr is not None:
                    txt += f"\n$\\pm${Rerr[i, j]:.3f}"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8.5, linespacing=1.3,
                        color="white" if abs(M[i, j]) > 0.62 * vmax else "0.1")
        ax.set_xticks(range(3), ["x", "y", "z"])
        ax.set_yticks(range(3), ["x", "y", "z"])
        ax.set_title(name, fontsize=11.5)
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
    ax.set_xticks(x, lbl, fontsize=8.5)
    ax.set_ylim(0, max(val.max() * 1.15, 1e-3))
    ax.set_ylabel("eigenvalue of  $(R+R^{T})/2$")
    fig.savefig(path)
    plt.close(fig)
    return path


def _sim_counts(S, n_amp):
    """The three sim counts behind an estimate, as one short line."""
    n_mf = int(_scalar(S, "n_mf", 0))
    n_rs = int(_scalar(S, "n_resp_used", 0)) or int(_scalar(S, "n_resp", 0))
    return (f"$N$: {n_amp} data, {n_mf} mean field, {n_rs} response")


def plot_amplitude(S, path):
    """Corrected amplitude and the recovered directions.

    Only the R^-1 corrected amplitude is drawn.  The uncorrected estimator is
    still stored in summary.npz as naive_<k> and still printed by the
    pipeline; it is just not plotted, so the histogram can be binned on the
    corrected values alone and sits around 1 instead of being pushed to one
    side by a second distribution an order of magnitude away.
    """
    _style()
    d_true = np.asarray(S["d_true"], float)
    amp = np.asarray(S["amp"], float)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    lo, hi = np.percentile(amp, 0.5), np.percentile(amp, 99.5)
    # Keep the 2 sd markers on the axis even when the tails are short.
    lo = min(lo, amp.mean() - 2.2 * amp.std(ddof=1))
    hi = max(hi, amp.mean() + 2.2 * amp.std(ddof=1))
    pad = 0.1 * (hi - lo + 1e-9)
    # The input has to stay on the axis even if the sims cluster away from it.
    bins = np.linspace(min(lo - pad, 1.0 - pad), max(hi + pad, 1.0 + pad), 34)
    ax0.hist(amp, bins=bins, histtype="stepfilled", lw=1.6, color=COR,
             alpha=0.3)
    sd = amp.std(ddof=1)
    ax0.hist(amp, bins=bins, histtype="step", lw=1.8, color=COR,
             label=f"$R^{{-1}}$ corrected:  {amp.mean():+.3f} $\\pm$ "
                   f"{sd:.3f}\n{_sim_counts(S, len(amp))}")
    # Mean, and the +-1 and +-2 sd bands the whisker figure also draws.
    ax0.axvspan(amp.mean() - sd, amp.mean() + sd, color=COR, alpha=0.10,
                lw=0, zorder=0)
    for k in (1, 2):
        for sgn in (-1, 1):
            ax0.axvline(amp.mean() + sgn * k * sd, color=COR, lw=1.0,
                        ls=(0, (2, 3)), alpha=0.8, zorder=1)
    ax0.axvline(amp.mean(), color=COR, lw=1.2, ls=(0, (3, 2)))
    ax0.axvline(1.0, color=TRUTH, lw=1.8, label="input")
    ax0.set_xlabel("amplitude $A$")
    ax0.set_ylabel("sims")
    ax0.set_ylim(top=ax0.get_ylim()[1] * 1.3)
    ax0.legend(fontsize=9.5, loc="upper right", frameon=True,
               framealpha=0.95, edgecolor="0.8")

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
    # The scatter's own 1 and 2 sd contours, so the spread on the direction is
    # drawn and not only quoted: the same 1 and 2 sd the amplitude panel marks
    # and the velocity boxes span.
    off = np.column_stack([sep * np.sin(psi), sep * np.cos(psi)])
    if off.shape[0] > 2:
        cov = np.cov(off, rowvar=False)
        for k in (1, 2):
            _ellipse(ax1, off.mean(axis=0), cov, k, fill=False,
                     edgecolor=COR, lw=1.3, ls=(0, (5, 3)), zorder=4)
    ax1.plot(0, 0, "*", ms=18, color=TRUTH, zorder=6)
    # The same mean +- sd the amplitude panel quotes, for the angle off the
    # input axis.  The rings give the scale; this gives the number.
    ax1.legend(handles=[Line2D([], [], marker="o", ls="none", color=COR,
                               alpha=0.7, ms=5,
                               label=f"offset  {sep.mean():.1f}$^\\circ$ "
                                     f"$\\pm$ {sep.std(ddof=1):.1f}$^\\circ$"),
                        Line2D([], [], color=COR, lw=1.3, ls=(0, (5, 3)),
                               label="1, 2 sd")],
               fontsize=9.5, loc="upper right", frameon=True,
               framealpha=0.95, edgecolor="0.8")
    lim = 1.15 * max(rmax, 1e-3)
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_aspect("equal")
    ax1.grid(False)
    ax1.set_xlabel("east offset  [deg]")
    ax1.set_ylabel("north offset  [deg]")
    fig.savefig(path)
    plt.close(fig)
    return path


def _fmt_cell(v):
    """Compact fixed or exponential, whichever reads better at this size."""
    if v == 0 or not np.isfinite(v):
        return "0"
    return f"{v:+.3f}" if abs(v) >= 5e-3 else f"{v:+.0e}".replace("e-0", "e-")


def _real_rows(pl, pm, lmax):
    """The (L, M) rows, M from -L to L, present in a packed alm array."""
    return [(l, m) for l in range(1, lmax + 1) for m in range(-l, l + 1)
            if np.any((pl == l) & (pm == abs(m)))]


def _to_real(A, pl, pm, lmax):
    """Packed complex alm (M >= 0) -> real spherical-harmonic coefficients.

    Acts on the last axis, so it takes one alm vector, a stack of them, or the
    transposed response matrix without changing shape anywhere else.  For a
    real field every complex a_LM with M > 0 carries two real numbers, and the
    real harmonics hold them as two signed coefficients:

        r_L0 = a_L0,  r_LM = sqrt2 (-1)^M Re a_LM,  r_L,-M = -sqrt2 (-1)^M Im a_LM.

    That is a rotation of the same information, not a summary of it: one signed
    number per mode, and sum_M r_LM^2 is the power the complex coefficients
    carry.  It is also the basis in which L = 1 is Cartesian, which is what
    lets the L = 1 block of the response be read as the matrix R.
    """
    rows = _real_rows(pl, pm, lmax)
    out = np.empty(np.shape(A)[:-1] + (len(rows),), dtype=float)
    for i, (l, m) in enumerate(rows):
        k = int(np.flatnonzero((pl == l) & (pm == abs(m)))[0])
        if m == 0:
            out[..., i] = A[..., k].real
        elif m > 0:
            out[..., i] = np.sqrt(2) * (-1) ** m * A[..., k].real
        else:
            out[..., i] = -np.sqrt(2) * (-1) ** m * A[..., k].imag
    return rows, out


def _real_basis(R, sd_re, sd_im, pl, pm, lmax):
    """The response and its per-sim scatter in the real-harmonic basis."""
    rows, V = _to_real(np.asarray(R).T, pl, pm, lmax)
    # The same map applied to the scatter, whose signs carry no meaning.
    _, E = _to_real((np.asarray(sd_re) + 1j * np.asarray(sd_im)).T,
                    pl, pm, lmax)
    return rows, V.T, np.abs(E).T


def plot_alm_response(S, path):
    """The (L,M) response to a unit boost along x, y, z.

    The direct analogue of the response-matrix figure, built from the same
    three boosts and the same normalisation, but read at every reconstruction
    multipole instead of only L=1.  Each column is what one unit of boost along
    that axis puts into each mode, averaged over the paired response sims.

    Drawn in the real spherical-harmonic basis, so each mode is one signed
    number rather than a real and an imaginary part: nothing is dropped, the
    complex pair at +-M is just rewritten as two real coefficients.  Everything
    is scaled by sqrt(3/4pi), a single constant, chosen so that the boxed L=1
    block is the response matrix R itself (rows M = -1, 0, +1 are the y, z and
    x components).  Everything below that block is leakage.

    The colour scale is symmetric log with its linear region set to the Monte
    Carlo error on the mean, so anything the sims cannot resolve sits in the
    pale band around zero and only resolved structure takes on colour.
    """
    _style()
    R = np.asarray(S["Ralm"])
    pl = np.asarray(S["pack_l"], int)
    pm = np.asarray(S["pack_m"], int)
    n = max(int(_scalar(S, "n_resp_used", 1)), 1)
    sd_re = np.asarray(S["Ralm_sd_re"], float)
    sd_im = np.asarray(S["Ralm_sd_im"], float)

    lmax = int(min(LEAK_LMAX, pl.max()))
    if lmax < 1:
        raise ValueError("summary.npz has no usable alm response")
    c = np.sqrt(3.0 / (4.0 * np.pi))
    rows, V, SD = _real_basis(R, sd_re, sd_im, pl, pm, lmax)
    V = c * V
    ERR = c * SD / np.sqrt(n)          # error on the mean, what the cells are
    nrow = len(rows)
    lr = np.array([r[0] for r in rows])
    lab = [f"{l},{m:+d}" if m else f"{l},0" for l, m in rows]
    edges = [i for i in range(1, nrow) if lr[i] != lr[i - 1]]

    fig, ax = plt.subplots(figsize=(5.4, 10.0), layout="constrained")
    thresh = max(float(np.median(ERR)), 1e-12)
    vmax = max(float(np.abs(V).max()), 10 * thresh)
    norm = SymLogNorm(linthresh=thresh, vmin=-vmax, vmax=vmax, base=10)
    im = ax.imshow(V, cmap="RdBu_r", norm=norm, aspect="auto")
    for i in range(nrow):
        for j in range(3):
            ax.text(j, i, _fmt_cell(V[i, j]), ha="center", va="center",
                    fontsize=7.2,
                    color="white" if abs(V[i, j]) > 0.25 * vmax else "0.15")
    for e in edges:
        ax.axhline(e - 0.5, color="white", lw=1.6)
    n1 = int((lr == 1).sum())
    ax.add_patch(Rectangle((-0.5, -0.5), 3, n1, fill=False, edgecolor=INK,
                           lw=1.9, zorder=5))
    ax.set_xticks(range(3), ["x", "y", "z"])
    ax.set_yticks(range(nrow), lab, fontsize=7.5)
    ax.set_xlabel("boost direction")
    ax.set_ylabel("$(L, M)$")
    ax.grid(False)

    e0 = int(np.ceil(np.log10(thresh)))
    e1 = int(np.floor(np.log10(vmax)))
    ticks = [0.0] + [sg * 10.0 ** e for e in range(e0, e1 + 1)
                     for sg in (1, -1) if 10.0 ** e > 2.0 * thresh]
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.035,
                      pad=0.02, aspect=30)
    if len(ticks) > 1:
        cb.set_ticks(sorted(ticks))
    cb.ax.tick_params(labelsize=8)
    cb.set_label(r"$\sqrt{3/4\pi}\;\langle r_{LM}\rangle / (-\beta)$",
                 fontsize=10)
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_noise(S, path, field):
    """Theory signal against the noise the filters were built from.

    One field per figure, both as D_l, always out to NOISE_PLOT_LMAX whatever
    lmax happens to be, so the two are drawn on the same axis and runs with
    different lmax stay comparable.  Where the noise crosses the signal is the
    scale beyond which that field stops carrying information; the dotted line
    is the atmospheric knee, where N_l is twice its white level by definition.
    """
    _style()
    name, ckey, nkey, ki = field
    cl = np.asarray(S[ckey], float)
    nl = np.asarray(S[nkey], float)
    t2 = _scalar(S, "tcmb_uk", 1.0) ** 2          # dimensionless -> uK^2
    ell = np.arange(cl.size, dtype=float)
    dfac = ell * (ell + 1.0) / (2.0 * np.pi)
    lmin = int(_scalar(S, "lmin", 2)) or 2
    lmax = int(_scalar(S, "lmax", cl.size - 1))
    hi = min(NOISE_PLOT_LMAX, cl.size - 1)
    sl = slice(2, hi + 1)

    fig, ax = plt.subplots(figsize=(7.6, 5.0), layout="constrained")
    ax.axvspan(lmin, min(lmax, hi), color=COR, alpha=0.07, lw=0, zorder=0)
    ax.plot(ell[sl], (dfac * cl * t2)[sl], lw=1.8, color=COR,
            label=f"$C_\\ell^{{{name}}}$")
    ax.plot(ell[sl], (dfac * nl * t2)[sl], lw=1.8, color=RAW,
            ls=(0, (5, 3)), label=f"$N_\\ell^{{{name}}}$")

    knee = np.atleast_1d(np.asarray(S.get("ell_knee", []), float)).ravel()
    if knee.size > ki and np.isfinite(knee[ki]) and 2 < knee[ki] < hi:
        ax.axvline(knee[ki], color=RAW, lw=1.1, ls=(0, (1, 3)), alpha=0.9,
                   label=r"$\ell_{\rm knee}$")

    ax.set_yscale("log")
    ax.set_xlim(0, hi)
    d_cl = (dfac * cl * t2)[sl]
    d_nl = (dfac * nl * t2)[sl]
    ax.set_ylim(d_cl.max() * 1e-4, max(d_cl.max(), d_nl.max()) * 3.0)
    ax.set_xlabel(r"multipole $\ell$")
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$  [$\mu$K$^2$]")
    ax.legend(fontsize=9.5, loc="upper right", frameon=True, framealpha=0.95,
              edgecolor="0.8")
    ax.grid(True, which="major", color=GRID, lw=0.6, alpha=0.35)
    ax.grid(True, which="minor", axis="y", color=GRID, lw=0.4, alpha=0.15)

    fig.savefig(path)
    plt.close(fig)
    return path


def plot_noise_tt(S, path):
    """Temperature signal and noise."""
    return _plot_noise(S, path, ("TT", "cl_tt", "nl_tt", 0))


def plot_noise_ee(S, path):
    """Polarisation signal and noise."""
    return _plot_noise(S, path, ("EE", "cl_ee", "nl_ee", 1))


def plot_leakage(S, path):
    """Spurious power the boost leaves at each L, from the aberrated sims.

    One panel, L = 1 to LEAK_LMAX, one series: datleak, the mean of the
    aberrated data sims with the mean field subtracted and the Monte Carlo
    variance of both removed.

    That estimator is a difference of two noisy means, so it can come out
    negative wherever the leakage is below the noise.  If every point drawn is
    positive the axis is a plain log; if any is not, it switches to symlog so
    the negative ones can be seen, with the linear region set just wide enough
    to hold the smallest point.  Negative points are drawn open so the sign
    survives in greyscale and in print.
    """
    _style()
    ells = np.asarray(S["ells"], int)
    dat = np.asarray(S["datleak"], float)

    m = (ells >= 1) & (ells <= LEAK_LMAX) & np.isfinite(dat)
    if m.sum() < 2:
        raise ValueError("summary.npz has no usable datleak")
    L, y = ells[m], dat[m]

    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    err = None
    if "leakerr" in S:
        e = np.asarray(S["leakerr"], float)
        if e.shape == dat.shape:
            err = e[m]

    pos, neg = y > 0, y <= 0
    ax.plot(L, y, "-", lw=1.6, color=COR, alpha=0.8, zorder=2)
    if err is not None:
        # Jackknife error on the debiased estimator, drawn at 1 and 2 sd.
        ax.errorbar(L, y, yerr=2 * err, fmt="none", ecolor=COR, alpha=0.4,
                    elinewidth=1.0, capsize=3.5, zorder=3)
        ax.errorbar(L, y, yerr=err, fmt="none", ecolor=COR, alpha=0.9,
                    elinewidth=2.0, capsize=0, zorder=3)
    ax.plot(L[pos], y[pos], "o", ms=6.5, color=COR, mec=INK, mew=0.6,
            ls="none", zorder=4, label="positive")
    if neg.any():
        ax.plot(L[neg], y[neg], "o", ms=6.5, mfc="white", mec=COR, mew=1.5,
                ls="none", zorder=4, label="negative after debiasing")

    small, big = np.abs(y[y != 0]).min(), np.abs(y).max()
    if err is not None and err.size:
        big = max(big, float(np.abs(y + 2 * err).max()))
    if neg.any():
        # Four decades below the largest point, or the smallest point if that
        # is larger.  Taking the smallest point itself would squeeze the
        # linear region to nothing and pile the 0 tick on top of its
        # neighbours; anything below the threshold is noise either way.
        lt = max(small, big * 1e-4)
        ax.axhline(0.0, color="0.55", lw=1.0, zorder=1)
        ax.set_yscale("symlog", linthresh=lt, linscale=0.7)
        ax.set_ylim(min(-3.0 * lt, 1.8 * y.min()), 4.0 * max(y.max(), lt))
        # Ticks written out rather than left to the automatic locator, which
        # puts decades inside the linear region as well and lands them on top
        # of the 0 label.  One per decade from the threshold outward, plus 0.
        lo, hi = ax.get_ylim()
        e0 = int(np.ceil(np.log10(lt)))
        e1 = int(np.ceil(np.log10(max(abs(lo), abs(hi)))))
        ticks = [0.0] + [s * 10.0 ** e for e in range(e0, e1 + 1)
                         for s in (1, -1) if lo <= s * 10.0 ** e <= hi]
        ax.set_yticks(sorted(ticks))
        ax.legend(fontsize=9.5, loc="upper right", frameon=True,
                  framealpha=0.9, borderpad=0.6).set_zorder(6)
    else:
        ax.set_yscale("log")
        ax.set_ylim(small / 3.0, big * 4.0)

    ax.set_xlim(0.6, L[-1] + 0.4)
    ax.set_xticks(L)
    ax.set_xlabel(r"multipole $L$ of the reconstruction")
    ax.set_ylabel(r"$C_L$  [dimensionless $\phi$]")
    ax.grid(True, which="major", color=GRID, lw=0.6, alpha=0.35)
    ax.grid(True, which="minor", axis="y", color=GRID, lw=0.4, alpha=0.18)

    fig.savefig(path)
    plt.close(fig)
    return path


def make_all(S, outdir):
    os.makedirs(outdir, exist_ok=True)
    jobs = [(plot_coverage, "sky_coverage.png"),
            (plot_whisker, "velocity_whisker.png"),
            (plot_planes, "velocity_planes.png"),
            (plot_response_matrix, "response_matrix.png"),
            (plot_amplitude, "amplitude_direction.png"),
            (plot_alm_response, "alm_response.png"),
            (plot_noise_tt, "noise_spectra_tt.png"),
            (plot_noise_ee, "noise_spectra_ee.png"),
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