"""One site's products, the rule's proposal, the delivered file, the comparison, and the survey's gallery.

The panel conventions are the ones figures/products.py sets: period on a log x axis, apparent resistivity on
a log axis in Ohm.m, the phase 0-90 deg with the yx panel labelled `+ 180 deg`, the tipper -0.8..0.8 with the
real part as filled circles and the imaginary part as open triangles on a dotted line.

A figure is drawn where it shows a curve, a record or a distribution over many items (Ben, 2026-09-17). A
criterion that yields one or two numbers per site is a line in a table and in a verdict, not a figure, so
the split-half departures and the per-row step at the join are printed and scored and never plotted.

`curves_page` draws every four- or six-panel page of workbook 06 and the sections differ only in what they
hand it: the products of a site coloured by reference kind, the rule's proposal against the products it
rejected, and the delivered file with its error bars and any comparison behind it. A curve a response test
refused is drawn grey and carries the test's name at its long end, and it does not set the y limits: a
refused row is often the one that leaves the panel, and letting it set the axes squeezes the curve the page
is about into a line.

Every figure is finished by figures.common.finish, which sets a short title, wraps the caption under the
axes and saves at dpi 110.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..process import KIND_WORD
from ..products import OFF_DIAGONAL, rho_phase
from ..readings import KINDS
from .common import finish
from .products import COMPARISON_COLOUR, GRID_ALPHA, PHASE_LIM, _dress, rho_limits, tf_panels

FINAL_COLOUR = "C3"                # the delivered curve, drawn on top
REJECTED_COLOUR = "0.65"           # a product the rule did not choose, and a quantity scored by nothing
FAILED_COLOUR = "0.80"             # ... and one a response test refused
FIGSIZE = (13.0, 12.5)

# the four reference kinds a product may be delivered on; the colours are workbook 04's
KIND_COLOUR = {k: "C%d" % i for i, k in enumerate(KINDS)}
RATE_STYLE = {1: "-", 10: "--"}    # a form is drawn dotted whatever its rate


def kind_colour(kind) -> str:
    return KIND_COLOUR.get(str(kind), "C7")


def curve_style(kind, rate_hz, form="", passes=True) -> dict:
    """The colour and line style of one product: the kind is the colour, the rate the line, a form dotted."""
    return dict(colour=(kind_colour(kind) if passes else FAILED_COLOUR),
                ls=(":" if str(form or "") else RATE_STYLE.get(int(float(rate_hz)), "-.")))


def _note_at(ax, tf, comp, text, colour):
    """The name of a failed test written at the long end of that curve, on the rho panel."""
    rho, _re, _ph, _pe = rho_phase(tf.period, tf.z, tf.z_err, comp)
    p = np.asarray(tf.period, float)
    m = np.isfinite(rho) & (rho > 0)
    if not m.any() or not text:
        return
    k = int(np.argmax(p[m]))
    ax.annotate(str(text), (p[m][k], rho[m][k]), fontsize=6, color=colour, va="center",
                ha="right", xytext=(-4, 0), textcoords="offset points")


def curves_page(site, curves, out, title="", caption="", period_range=None, tipper=True,
                figsize=FIGSIZE, legend_ncol=2, join_s=None):
    """Every curve of one page on four or six panels, finished with a short title and a caption.

    `curves` is a list of mappings: `tf` the TFData, `label` the legend entry or None, `colour`, `ls`,
    `marker`, `lw`, `ms`, `alpha`, `zorder`, `bars` whether the error bars are drawn, `limits` whether the
    curve sets the rho y limits, and `note` a short string written at the curve's long end.
    """
    import matplotlib.pyplot as plt

    nrow = 3 if tipper else 2
    fig, ax = plt.subplots(nrow, 2, figsize=figsize, squeeze=False)
    panels = (ax[0][0], ax[0][1], ax[1][0], ax[1][1],
              ax[2][0] if tipper else None, ax[2][1] if tipper else None)
    rho_curves = []
    for c in curves:
        tf = c.get("tf")
        if tf is None:
            continue
        drawn = tf_panels(*panels, tf, c.get("label"), colour=c.get("colour", "C0"),
                          ls=c.get("ls", "-"), marker=c.get("marker", "o"), lw=c.get("lw", 0.8),
                          ms=c.get("ms", 3), alpha=c.get("alpha", 0.85), zorder=c.get("zorder", 3),
                          period_range=period_range, bars=bool(c.get("bars", False)))
        if c.get("limits", True):
            rho_curves += drawn
        if c.get("note"):
            _note_at(ax[0][0], tf, "xy", c["note"], c.get("colour", "0.5"))
            _note_at(ax[0][1], tf, "yx", c["note"], c.get("colour", "0.5"))
    _dress(panels, rho_curves, period_range)
    if join_s:
        # the join is where the delivered row changes hands, so the reader has to see which side of the
        # line each point came from
        for a in panels:
            if a is not None:
                a.axvline(float(join_s), color="0.4", ls="--", lw=0.8, zorder=0)
        ax[0][0].plot([], [], ls="--", lw=0.8, color="0.4",
                      label="the join at %g s: the 10 Hz row below it" % float(join_s))
    handles, labels = ax[0][0].get_legend_handles_labels()
    if handles:
        ax[0][0].legend(handles, labels, fontsize=6.5, loc="best", ncol=legend_ncol, framealpha=0.85)
    return finish(fig, title or "%s: the transfer functions" % site, caption, out)


def products_page(site, rows, out, title="", caption="", period_range=None, tipper=True,
                  figsize=FIGSIZE):
    """Every product of one site: colour by kind, 10 Hz dashed, forms dotted, a refused product grey.

    `rows` is a list of mappings carrying `tf`, `label`, `kind`, `rate_hz`, `form`, `passes` and `fails`.
    """
    curves = []
    for r in sorted(rows, key=lambda d: (not d.get("passes", True), str(d.get("label", "")))):
        st = curve_style(r.get("kind", ""), r.get("rate_hz", 1), r.get("form", ""),
                         bool(r.get("passes", True)))
        curves.append(dict(tf=r.get("tf"), label=r.get("label"), kind=r.get("kind", ""),
                           colour=st["colour"], ls=st["ls"], marker="o", lw=0.7, ms=2.5,
                           alpha=(0.85 if r.get("passes", True) else 0.6),
                           zorder=(3 if r.get("passes", True) else 2), bars=False,
                           limits=bool(r.get("passes", True)),
                           note=("" if r.get("passes", True) else r.get("fails", ""))))
    return curves_page(site, curves, out, title=(title or "%s: every product of this site" % site),
                       caption=caption, period_range=period_range, tipper=tipper, figsize=figsize)


def over_rejected(site, chosen, rejected, out, title="", caption="", period_range=None, tipper=True,
                  bars=True, figsize=FIGSIZE):
    """The chosen curves over the products they were chosen against, each rejected curve named in grey.

    `chosen` is a list of (label, kind, TFData) and `rejected` a list of (label, TFData).
    """
    curves = [dict(tf=tf, label=None, colour=REJECTED_COLOUR, ls="-", marker=".", lw=0.6, ms=2,
                   alpha=0.8, zorder=2, bars=False, limits=False, note=label)
              for label, tf in rejected]
    for label, kind, tf in chosen:
        curves.append(dict(tf=tf, label=label, kind=kind, colour=kind_colour(kind), ls="-", marker="o",
                           lw=1.4, ms=3.5, alpha=1.0, zorder=5, bars=bars, limits=True))
    return curves_page(site, curves, out,
                       title=(title or "%s: the choice over the products it was made against" % site),
                       caption=caption, period_range=period_range, tipper=tipper, figsize=figsize)


def delivered_page(site, final, out, title="", caption="", comparisons=(), period_range=None,
                   tipper=True, figsize=FIGSIZE, join_s=None):
    """The delivered curve with its error bars, the join marked, and any comparison in black behind it."""
    curves = [dict(tf=tf, label="%s (a comparison, not truth)" % label, colour=COMPARISON_COLOUR, ls="-",
                   marker=".", lw=1.0, ms=3, alpha=0.9, zorder=1, bars=True, limits=False)
              for label, tf in comparisons]
    curves.append(dict(tf=final, label="the delivered file", colour=FINAL_COLOUR, ls="-", marker="o",
                       lw=1.6, ms=4, alpha=1.0, zorder=5, bars=True, limits=True))
    return curves_page(site, curves, out, title=(title or "%s: the delivered transfer function" % site),
                       caption=caption, period_range=period_range, tipper=tipper, figsize=figsize,
                       join_s=join_s)


# ------------------------------------------------------------------ the gallery

def final_gallery(sites, finals: dict, out_dir, stem="gallery_final", per_page=6, period_range=None,
                  title="", caption="", comparisons=None, reasons=None, figsize=(12.0, 2.1)):
    """rho xy/yx and phase xy/yx of every delivered file, `per_page` sites a page, one legend on page one.

    `finals` is {site: TFData}, `comparisons` an optional {site: [(label, TFData)]} drawn in black behind,
    and `reasons` an optional {site: why} printed in the panel of a site with no delivery. Writes
    <out_dir>/<stem>_p<k>.png and <out_dir>/<stem>_index.csv.
    """
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comparisons = comparisons or {}
    reasons = reasons or {}
    pages, index = [], []
    chunks = [list(sites)[i:i + int(per_page)] for i in range(0, len(sites), int(per_page))]
    for page, chunk in enumerate(chunks, start=1):
        fig, axes = plt.subplots(len(chunk), 4, figsize=(figsize[0], figsize[1] * len(chunk)),
                                 squeeze=False)
        for row, site in enumerate(chunk):
            rho_curves = []
            drawn = [(lab, tf, COMPARISON_COLOUR, 1) for lab, tf in comparisons.get(site, [])]
            if finals.get(site) is not None:
                drawn.append(("the delivered file", finals[site], FINAL_COLOUR, 3))
            for label, tf, colour, z in drawn:
                p = np.asarray(tf.period, float)
                keep = (np.ones(len(p), bool) if period_range is None
                        else ((p >= period_range[0]) & (p <= period_range[1])))
                for col, comp in enumerate(OFF_DIAGONAL):
                    rho, _re, ph, _pe = rho_phase(tf.period, tf.z, tf.z_err, comp)
                    m = keep & np.isfinite(rho) & (rho > 0)
                    if not m.any():
                        continue
                    axes[row][col].plot(p[m], rho[m], "o", ls="-", ms=2, lw=0.7, color=colour,
                                        alpha=0.9, zorder=z,
                                        label=(label if (row == 0 and col == 0) else None))
                    axes[row][2 + col].plot(p[m], ph[m], "o", ls="-", ms=2, lw=0.7, color=colour,
                                            alpha=0.9, zorder=z)
                    # the limits come from the delivered curve, which is what the page is about; a
                    # comparison three decades away would otherwise squeeze every curve on the page into a
                    # line, and it stays drawn, running off the panel where it disagrees that far
                    if colour == FINAL_COLOUR:
                        rho_curves.append(rho[m])
                    index.append(dict(page=page, row=row, site=site, curve=label, component=comp,
                                      n_periods=int(m.sum()), path=str(tf.meta.get("path", ""))))
            if finals.get(site) is None:
                axes[row][0].text(0.5, 0.5, str(reasons.get(site, "no delivered file"))[:150],
                                  transform=axes[row][0].transAxes, ha="center", va="center",
                                  fontsize=7, color="0.3", wrap=True)
            lo, hi = rho_limits(rho_curves)
            for col, _comp in enumerate(OFF_DIAGONAL):
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
        pages.append(finish(fig, "%s -- page %d of %d" % (title or "the delivered files", page,
                                                          len(chunks)),
                            caption, out_dir / ("%s_p%d.png" % (stem, page))))
    idx = pd.DataFrame(index)
    idx_path = out_dir / ("%s_index.csv" % stem)
    idx.to_csv(idx_path, index=False)
    return pages, idx, idx_path


def kind_key(kinds=KINDS) -> pd.DataFrame:
    """The colour key a page's legend abbreviates: the word, the code key and the colour of each kind."""
    return pd.DataFrame([dict(word=KIND_WORD.get(k, k), key=k, colour=kind_colour(k)) for k in kinds])
