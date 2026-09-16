"""Figure 06 and 07: every product of a site on one page, and the gallery of a survey.

The panel conventions are ported from Processing_Run/wamt_tools.py:465-488 plot_tf
(D:/BEN/MTH5_Aurora_mt-io_2026): period on a log x axis labelled `period (s)`; apparent resistivity on a log
axis in Ohm.m; phase 0-90 deg with the yx panel labelled `+ 180 deg`; the tipper -0.8..0.8 with the real part
as filled circles and the imaginary part as open triangles on a dotted line over a zero line; grids at
alpha 0.25; the series in C0..C9; a comparison source black with its own error bars, drawn first at zorder 1.

The page layout is ported from scripts/processing/vic_site_figure.py: every product of the site on one page,
the comparison behind, the phase folded, the y limits taken from the data. Kinds are the colours, rates are
the line style (1 Hz solid, 10 Hz dashed) and runs are the marker, so three axes of one page are readable at
once. The header lines the products carry are printed under the title.

The gallery is the simpler form of scripts/processing/vic_gallery_all.py: N sites a page (6 by default), rho
and phase for xy and yx per site, one legend on the first page, and an index CSV naming every curve drawn.

The y limits follow the same rule in both: the 2nd to 98th percentile of every rho curve on the panel, padded
by half a decade each way and never narrower than a decade; the phase is fixed at 0-90 deg.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..process import KINDS, KIND_WORD
from ..products import OFF_DIAGONAL, rho_phase, tipper_parts

DPI = 110
RHO_PAD_DEX = 0.5             # half a decade of padding above and below the data
RHO_MIN_DEX = 1.0             # a panel never spans less than a decade
PHASE_LIM = (0, 90)
TIPPER_LIM = (-0.8, 0.8)
GRID_ALPHA = 0.25

KIND_COLOUR = {k: "C%d" % i for i, k in enumerate(KINDS)}
RATE_STYLE = {1: "-", 10: "--"}
RUN_MARKER = ("o", "s", "^", "D", "v", "P")
COMPARISON_COLOUR = "k"


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


def _dress(axes, rho_curves, period_range):
    ax_rho_xy, ax_rho_yx, ax_ph_xy, ax_ph_yx, ax_tzx, ax_tzy = axes
    lo, hi = rho_limits(rho_curves)
    for ax, title, ylabel in ((ax_rho_xy, "rho xy", "rho (Ohm.m)"), (ax_rho_yx, "rho yx", "")):
        ax.set(xscale="log", yscale="log", ylim=(lo, hi), title=title, ylabel=ylabel)
    for ax, title, ylabel in ((ax_ph_xy, "phase xy", "phase (deg)"),
                              (ax_ph_yx, "phase yx + 180 deg", "")):
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


def site_page(site, products, read, out, comparisons=(), title="", header_lines=(), period_range=None,
              figsize=(13.0, 12.5), tipper=True):
    """Every product of one site on one page, the comparisons behind, written to `out`.

    `products` is the site's rows of find_products, `read` turns a path into a TFData, and `comparisons` is a
    list of (label, TFData) already in our frame. Returns (the path, the index rows it drew).
    """
    import matplotlib.pyplot as plt

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    nrow = 3 if tipper else 2
    fig = plt.figure(figsize=figsize)
    # the header band is sized from the number of lines it holds, so a product with a long members line does
    # not push the table down over the top panels
    band = 0.0085 * max(len(list(header_lines)), 8) + 0.045 if header_lines else 0.0
    top = 0.945 - band
    gs = fig.add_gridspec(nrow, 2, top=top, bottom=0.05, left=0.07, right=0.98, hspace=0.32, wspace=0.16)
    ax = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(nrow)]
    ax_tzx = ax[2][0] if tipper else None
    ax_tzy = ax[2][1] if tipper else None
    panels = (ax[0][0], ax[0][1], ax[1][0], ax[1][1], ax_tzx, ax_tzy)

    rho_curves, index = [], []
    for k, (label, tf) in enumerate(comparisons):
        rho_curves += tf_panels(*panels, tf, "%s (comparison, not truth)" % label,
                                colour=COMPARISON_COLOUR, ls="-", marker=".", lw=1.0, ms=3, alpha=0.9,
                                zorder=1, period_range=period_range)
        index.append(dict(site=site, curve="comparison", label=label,
                          path=str(tf.meta.get("path", "")), colour=COMPARISON_COLOUR))
    runs = sorted({(r.run, r.stamp) for r in products.itertuples()})
    for r in products.itertuples():
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
        index.append(dict(site=site, curve="product", label=label, path=str(r.path),
                          colour=st["colour"], kind=r.kind, rate_hz=r.rate_hz, run=r.run))
    _dress(panels, rho_curves, period_range)
    handles, labels = ax[0][0].get_legend_handles_labels()
    fig.suptitle(title or "%s: every product on one page" % site, y=0.99, fontsize=12)
    if header_lines:
        fig.text(0.02, 0.965, "\n".join(str(x) for x in header_lines), fontsize=6.5, va="top",
                 family="monospace")
    # the legend sits in the header band, not over a panel: a page of ten curves loses a decade of rho
    # underneath a legend drawn inside the axes
    if handles:
        fig.legend(handles, labels, fontsize=7, loc="upper right",
                   bbox_to_anchor=(0.995, 0.965), ncol=2, framealpha=0.9)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out, index


def gallery(sites, products, read, out_dir, stem="gallery", per_page=6, period_range=None,
            title="", figsize=(12.0, 2.1)):
    """rho xy/yx and phase xy/yx per site, `per_page` sites a page, one legend on the first page.

    Writes <out_dir>/<stem>_p<k>.png and <out_dir>/<stem>_index.csv. Returns (the pages, the index frame).
    """
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pages, index = [], []
    chunks = [list(sites)[i:i + int(per_page)] for i in range(0, len(sites), int(per_page))]
    for page, chunk in enumerate(chunks, start=1):
        fig, axes = plt.subplots(len(chunk), 4, figsize=(figsize[0], figsize[1] * len(chunk)),
                                 squeeze=False)
        for row, site in enumerate(chunk):
            grp = products[products.site == site]
            runs = sorted({(r.run, r.stamp) for r in grp.itertuples()})
            rho_curves = []
            for r in grp.itertuples():
                if not r.on_disk:
                    continue
                try:
                    tf = read(r.path)
                except Exception:
                    continue
                st = _style(r.kind, r.rate_hz, runs.index((r.run, r.stamp)))
                label = "%s, %g Hz" % (KIND_WORD.get(r.kind, r.kind), r.rate_hz)
                p = np.asarray(tf.period, float)
                keep = (np.ones(len(p), bool) if period_range is None
                        else ((p >= period_range[0]) & (p <= period_range[1])))
                for col, comp in enumerate(OFF_DIAGONAL):
                    rho, _re, ph, _pe = rho_phase(tf.period, tf.z, tf.z_err, comp)
                    m = keep & np.isfinite(rho) & (rho > 0)
                    if not m.any():
                        continue
                    axes[row][col].plot(p[m], rho[m], st["marker"], ls=st["ls"], ms=2, lw=0.6,
                                        color=st["colour"], alpha=0.85,
                                        label=(label if (row == 0 and col == 0) else None))
                    axes[row][2 + col].plot(p[m], ph[m], st["marker"], ls=st["ls"], ms=2, lw=0.6,
                                            color=st["colour"], alpha=0.85)
                    rho_curves.append(rho[m])
                    index.append(dict(page=page, row=row, site=site, kind=r.kind, rate_hz=r.rate_hz,
                                      run=r.run, component=comp, n_periods=int(m.sum()), path=str(r.path)))
            lo, hi = rho_limits(rho_curves)
            for col, comp in enumerate(OFF_DIAGONAL):
                axes[row][col].set(xscale="log", yscale="log", ylim=(lo, hi))
                axes[row][2 + col].set(xscale="log", ylim=PHASE_LIM)
            axes[row][0].set_ylabel("%s\nrho (Ohm.m)" % site, fontsize=8)
            axes[row][2].set_ylabel("phase (deg)", fontsize=8)
            for col, name in enumerate(("rho xy", "rho yx", "phase xy", "phase yx + 180 deg")):
                axes[row][col].grid(alpha=GRID_ALPHA, which="both")
                axes[row][col].tick_params(labelsize=7)
                if row == 0:
                    axes[row][col].set_title(name, fontsize=9)
                if row == len(chunk) - 1:
                    axes[row][col].set_xlabel("period (s)", fontsize=8)
                if period_range:
                    axes[row][col].set_xlim(period_range)
        if page == 1:
            axes[0][0].legend(fontsize=6, loc="best")
        fig.suptitle("%s -- page %d of %d" % (title or "the gallery", page, len(chunks)), fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        path = out_dir / ("%s_p%d.png" % (stem, page))
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        pages.append(path)
    idx = pd.DataFrame(index)
    idx_path = out_dir / ("%s_index.csv" % stem)
    idx.to_csv(idx_path, index=False)
    return pages, idx, idx_path
