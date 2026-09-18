"""One site's transfer functions, the rule's proposal, and the delivered file: the three pages of workbook 06.

The panel conventions are the ones figures/transfer_functions.py sets: period on a log x axis, apparent
resistivity on a log axis in Ohm.m, the phase 0-90 deg with the yx panel labelled `+ 180 deg`, the tipper
-0.8..0.8 with the real part as filled circles and the imaginary part as open triangles on a dotted line.

A figure is drawn where it shows a curve, a record or a distribution over many items (Ben, 2026-09-17). A
criterion that yields one or two numbers per site is a line in a table and in a verdict, not a figure, so the
split-half departures and the per-row step at the join are printed and scored and never plotted.

Workbook 06 delivers one site per run, so nothing here draws a survey.

`curves_page` draws every four- or six-panel page and the three callers differ only in what they hand it: the
transfer functions of a site coloured by reference kind, the rule's proposal against the curves it rejected,
and the delivered file with its error bars and the join marked. A curve a response test refused is drawn grey
and carries the test's name at its long end, and it does not set the y limits: a refused row is often the one
that leaves the panel, and letting it set the axes squeezes the curve the page is about into a line.

Every figure is finished by figures.common.finish, which sets a short title, wraps the caption under the axes
and saves at dpi 110.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import numpy as np

from ..transfer_functions import rho_phase
from ..readings import KINDS
from .common import finish, legend_below
from .transfer_functions import _dress, tf_panels

FINAL_COLOUR = "C3"                # the delivered curve, drawn on top
REJECTED_COLOUR = "0.65"           # a curve the rule did not choose, and a quantity scored by nothing
FAILED_COLOUR = "0.80"             # ... and one a response test refused
FIGSIZE = (13.0, 12.5)

# the four reference kinds a transfer function may be delivered on; the colours are workbook 04's
KIND_COLOUR = {k: "C%d" % i for i, k in enumerate(KINDS)}
RATE_STYLE = {1: "-", 10: "--"}    # a form is drawn dotted whatever its rate


def kind_colour(kind) -> str:
    return KIND_COLOUR.get(str(kind), "C7")


def curve_style(kind, rate_hz, form="", passes=True) -> dict:
    """The colour and line style of one curve: the kind is the colour, the rate the line, a form dotted."""
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

    The key goes under the panels where there are more than common.LEGEND_IN_PANEL_MAX entries, which is
    every page that draws a site's transfer functions; `legend_ncol` is the corner key's columns and is not
    read once the key has left the axes, where the figure's own width sets them.
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
    below = bool(handles) and legend_below(labels)
    if handles and not below:
        ax[0][0].legend(handles, labels, fontsize=6.5, loc="best", ncol=legend_ncol, framealpha=0.85)
    return finish(fig, title or "%s: the transfer functions" % site, caption, out,
                  legend=((handles, labels) if below else None))


def transfer_functions_page(site, rows, out, title="", caption="", period_range=None, tipper=True,
                  figsize=FIGSIZE):
    """Every transfer function of one site: colour by kind, 10 Hz dashed, forms dotted, a refused row grey.

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
    return curves_page(site, curves, out, title=(title or "%s: every transfer function" % site),
                       caption=caption, period_range=period_range, tipper=tipper, figsize=figsize)


def over_rejected(site, chosen, rejected, out, title="", caption="", period_range=None, tipper=True,
                  bars=True, figsize=FIGSIZE):
    """The chosen curves over the ones they were chosen against, each rejected curve named in grey.

    `chosen` is a list of (label, kind, TFData) and `rejected` a list of (label, TFData).
    """
    curves = [dict(tf=tf, label=None, colour=REJECTED_COLOUR, ls="-", marker=".", lw=0.6, ms=2,
                   alpha=0.8, zorder=2, bars=False, limits=False, note=label)
              for label, tf in rejected]
    for label, kind, tf in chosen:
        curves.append(dict(tf=tf, label=label, kind=kind, colour=kind_colour(kind), ls="-", marker="o",
                           lw=1.4, ms=3.5, alpha=1.0, zorder=5, bars=bars, limits=True))
    return curves_page(site, curves, out,
                       title=(title or "%s: the choice and the curves it was made against" % site),
                       caption=caption, period_range=period_range, tipper=tipper, figsize=figsize)


def delivered_page(site, final, out, title="", caption="", period_range=None, tipper=True,
                   figsize=FIGSIZE, join_s=None):
    """The delivered curve alone with its error bars and the join marked."""
    curves = [dict(tf=final, label="the delivered file", colour=FINAL_COLOUR, ls="-", marker="o",
                   lw=1.6, ms=4, alpha=1.0, zorder=5, bars=True, limits=True)]
    return curves_page(site, curves, out, title=(title or "%s: the delivered transfer function" % site),
                       caption=caption, period_range=period_range, tipper=tipper, figsize=figsize,
                       join_s=join_s)
