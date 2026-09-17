"""Figure 06: every transfer function of a site on one page.

The panel conventions are ported from Processing_Run/wamt_tools.py:465-488 plot_tf
(D:/BEN/MTH5_Aurora_mt-io_2026): period on a log x axis labelled `period (s)`; apparent resistivity on a log
axis in Ohm.m; phase 0-90 deg with the yx panel labelled `+ 180 deg`; the tipper -0.8..0.8 with the real part
as filled circles and the imaginary part as open triangles on a dotted line over a zero line; grids at
alpha 0.25; the series in C0..C9.

The page layout is ported from scripts/processing/vic_site_figure.py: every transfer function of the site on
one page, the phase folded, the y limits taken from the data. Kinds are the colours, rates are the line style
(1 Hz solid, 10 Hz dashed) and runs are the marker, so three axes of one page are readable at once. The
header lines the files carry are written into the caption.

The y limits: the 2nd to 98th percentile of every rho curve on the panel, padded by half a decade each way
and never narrower than a decade; the phase is fixed at 0-90 deg.

The page is finished by figures.common.finish, which sets a short title, wraps the caption under the axes and
saves at dpi 110.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..process import KINDS, KIND_WORD
from ..transfer_functions import rho_phase, tipper_parts
from .common import finish

RHO_PAD_DEX = 0.5             # half a decade of padding above and below the data
RHO_MIN_DEX = 1.0             # a panel never spans less than a decade
PHASE_LIM = (0, 90)
TIPPER_LIM = (-0.8, 0.8)
GRID_ALPHA = 0.25

KIND_COLOUR = {k: "C%d" % i for i, k in enumerate(KINDS)}
RATE_STYLE = {1: "-", 10: "--"}
RUN_MARKER = ("o", "s", "^", "D", "v", "P")


def rho_limits(curves) -> tuple:
    """(low, high) for a log rho axis: the 2nd-98th percentile of every curve, padded half a decade."""
    vals = np.concatenate([np.asarray(c, float)[np.isfinite(c) & (np.asarray(c, float) > 0)]
                           for c in curves if c is not None and len(c)]) if curves else np.zeros(0)
    if not len(vals):
        return 0.1, 1000.0
    lo = np.log10(np.percentile(vals, 2)) - RHO_PAD_DEX
    hi = np.log10(np.percentile(vals, 98)) + RHO_PAD_DEX
    if hi - lo < RHO_MIN_DEX:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 0.5 * RHO_MIN_DEX, mid + 0.5 * RHO_MIN_DEX
    return 10.0 ** lo, 10.0 ** hi


def _style(kind, rate_hz, run_index):
    """The three axes of a page: the kind is the colour, the rate the line style, the run the marker."""
    return dict(colour=KIND_COLOUR.get(kind, "C7"), ls=RATE_STYLE.get(int(rate_hz), ":"),
                marker=RUN_MARKER[run_index % len(RUN_MARKER)])


def tf_panels(ax_rho_xy, ax_rho_yx, ax_ph_xy, ax_ph_yx, ax_tzx, ax_tzy, tf, label, colour="C0",
              ls="-", marker="o", lw=0.7, ms=3, alpha=0.85, zorder=3, period_range=None, bars=True):
    """One transfer function drawn on the six panels of a page. Returns the rho arrays it drew."""
    p = np.asarray(tf.period, float)
    keep = (np.ones(len(p), bool) if period_range is None
            else ((p >= period_range[0]) & (p <= period_range[1])))
    drawn = []
    for comp, ax_r, ax_p in (("xy", ax_rho_xy, ax_ph_xy), ("yx", ax_rho_yx, ax_ph_yx)):
        rho, rho_err, ph, ph_err = rho_phase(tf.period, tf.z, tf.z_err, comp)
        m = keep & np.isfinite(rho) & (rho > 0)
        if not m.any():
            continue
        ax_r.errorbar(p[m], rho[m], yerr=(rho_err[m] if bars else None), fmt=marker, ls=ls, ms=ms, lw=lw,
                      elinewidth=0.7, color=colour, alpha=alpha, zorder=zorder,
                      label=(label if comp == "xy" else None))
        ax_p.errorbar(p[m], ph[m], yerr=(ph_err[m] if bars else None), fmt=marker, ls=ls, ms=ms, lw=lw,
                      elinewidth=0.7, color=colour, alpha=alpha, zorder=zorder)
        drawn.append(rho[m])
    for comp, ax in (("zx", ax_tzx), ("zy", ax_tzy)):
        if ax is None:
            continue
        re_, im_, err = tipper_parts(tf, comp)
        if re_ is None:
            continue
        m = keep & np.isfinite(re_)
        if not m.any():
            continue
        ax.errorbar(p[m], re_[m], yerr=(err[m] if bars else None), fmt=marker, ls=ls, ms=ms, lw=lw,
                    elinewidth=0.7, color=colour, alpha=alpha, zorder=zorder)
        ax.errorbar(p[m], im_[m], yerr=(err[m] if bars else None), fmt="^", ls=":", ms=ms, lw=lw,
                    elinewidth=0.7, mfc="none", color=colour, alpha=alpha, zorder=zorder)
    return drawn


def _dress(axes, rho_curves, period_range, comps=("xy", "yx")):
    """The panel limits and titles. `comps` names the two components the panels carry, so a page drawn in a
    turned frame titles them x'y' and y'x' and a page in the site's own frame titles them xy and yx."""
    ax_rho_xy, ax_rho_yx, ax_ph_xy, ax_ph_yx, ax_tzx, ax_tzy = axes
    c_xy, c_yx = comps
    lo, hi = rho_limits(rho_curves)
    for ax, title, ylabel in ((ax_rho_xy, "rho %s" % c_xy, "rho (Ohm.m)"),
                              (ax_rho_yx, "rho %s" % c_yx, "")):
        ax.set(xscale="log", yscale="log", ylim=(lo, hi), title=title, ylabel=ylabel)
    for ax, title, ylabel in ((ax_ph_xy, "phase %s" % c_xy, "phase (deg)"),
                              (ax_ph_yx, "phase %s + 180 deg" % c_yx, "")):
        ax.set(xscale="log", ylim=PHASE_LIM, title=title, ylabel=ylabel)
    for ax, title, ylabel in ((ax_tzx, "Tzx: real filled, imaginary open on dotted", "tipper"),
                              (ax_tzy, "Tzy: real filled, imaginary open on dotted", "")):
        if ax is None:
            continue
        ax.set(xscale="log", ylim=TIPPER_LIM, title=title, ylabel=ylabel)
        ax.axhline(0, color="0.6", lw=0.6)
    for ax in axes:
        if ax is None:
            continue
        ax.grid(alpha=GRID_ALPHA, which="both")
        if period_range:
            ax.set_xlim(period_range)
    for ax in (ax_tzx or ax_ph_xy, ax_tzy or ax_ph_yx):
        ax.set_xlabel("period (s)")


def site_page(site, tfs, read, out, title="", caption="", header_lines=(), period_range=None,
              figsize=(13.0, 12.5), tipper=True):
    """Every transfer function of one site on one page, written to `out`.

    `tfs` is the site's rows of find_transfer_functions and `read` turns a path into a TFData. `title` is one
    short line and `caption` the sentence under the axes; the processing_parameters lines in `header_lines`
    are added to the caption, because they are the parameters the page was drawn on. Returns (the path, the
    index rows it drew).
    """
    import matplotlib.pyplot as plt

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    nrow = 3 if tipper else 2
    fig, ax = plt.subplots(nrow, 2, figsize=figsize, squeeze=False)
    panels = (ax[0][0], ax[0][1], ax[1][0], ax[1][1],
              ax[2][0] if tipper else None, ax[2][1] if tipper else None)

    rho_curves, index = [], []
    runs = sorted({(r.run, r.stamp) for r in tfs.itertuples()})
    for r in tfs.itertuples():
        if not r.on_disk:
            continue
        try:
            tf = read(r.path)
        except Exception:
            continue
        st = _style(r.kind, r.rate_hz, runs.index((r.run, r.stamp)))
        label = "%s (%s), %g Hz%s" % (KIND_WORD.get(r.kind, r.kind), r.kind, r.rate_hz,
                                      "" if len(runs) < 2 else ", %s" % r.run)
        rho_curves += tf_panels(*panels, tf, label, period_range=period_range, **st)
        index.append(dict(site=site, curve="transfer_function", label=label, path=str(r.path),
                          colour=st["colour"], kind=r.kind, rate_hz=r.rate_hz, run=r.run))
    _dress(panels, rho_curves, period_range)
    handles, labels = ax[0][0].get_legend_handles_labels()
    if handles:
        ax[0][0].legend(handles, labels, fontsize=6.5, loc="best", ncol=2, framealpha=0.85)
    lines = [str(x) for x in header_lines]
    text = "; ".join(lines)
    full = "%s %s" % (str(caption).strip(), text) if caption else text
    return finish(fig, title or "%s: every transfer function" % site, full.strip(), out), index

