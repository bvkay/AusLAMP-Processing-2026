"""The final transfer function of a site on one page, and the gallery of the survey's finals.

The panel conventions are the ones figures/products.py sets: period on a log x axis, apparent resistivity on
a log axis in Ohm.m, the phase 0-90 deg with the yx panel labelled `+ 180 deg`, the tipper -0.8..0.8 with the
real part as filled circles and the imaginary part as open triangles on a dotted line.

What a page carries, drawn back to front: the comparison sources in black with their own error bars, the
earths the rule rejected in grey, the products of record in the kind colours of workbook 04, and the final
itself on top as a heavy line. The final and the products it came from are the same numbers where the merge
took a whole row, so the page reads as one curve with its sources under it; where they part, a row came from
somewhere else and the page says which.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..process import KIND_WORD
from ..products import OFF_DIAGONAL, rho_phase
from .products import (COMPARISON_COLOUR, DPI, GRID_ALPHA, KIND_COLOUR, PHASE_LIM, _dress, rho_limits,
                       tf_panels)

FINAL_COLOUR = "C3"                # the delivered curve, drawn on top
REJECTED_COLOUR = "0.65"           # an earth the rule did not choose, and a quantity scored by nothing
PASS_COLOUR = "C2"                 # a row that holds the clause a panel is about
FAIL_COLOUR = "C1"                 # ... and a row that does not
FIGSIZE = (13.0, 12.5)


def final_page(site, final, sources=(), rejected=(), comparisons=(), out=None, title="",
               header_lines=(), period_range=None, figsize=FIGSIZE, tipper=True):
    """One site's final with its sources behind it. Returns (the path, the index rows it drew).

    `final` is the delivered TFData, `sources` a list of (label, kind, TFData) of the products of record,
    `rejected` a list of (label, TFData) of the earths the rule did not choose, and `comparisons` a list of
    (label, TFData) already in our frame.
    """
    import matplotlib.pyplot as plt

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    nrow = 3 if tipper else 2
    fig = plt.figure(figsize=figsize)
    band = 0.0085 * max(len(list(header_lines)), 8) + 0.045 if header_lines else 0.0
    top = 0.945 - band
    gs = fig.add_gridspec(nrow, 2, top=top, bottom=0.05, left=0.07, right=0.98, hspace=0.32, wspace=0.16)
    ax = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(nrow)]
    panels = (ax[0][0], ax[0][1], ax[1][0], ax[1][1],
              ax[2][0] if tipper else None, ax[2][1] if tipper else None)

    rho_curves, index = [], []
    for label, tf in comparisons:
        rho_curves += tf_panels(*panels, tf, "%s (comparison, not truth)" % label,
                                colour=COMPARISON_COLOUR, ls="-", marker=".", lw=1.0, ms=3, alpha=0.9,
                                zorder=1, period_range=period_range)
        index.append(dict(site=site, curve="comparison", label=label,
                          path=str(tf.meta.get("path", ""))))
    for label, tf in rejected:
        # drawn, but kept out of the y limits: a row the rule refused is often the one that leaves the
        # panel, and letting it set the axes would squeeze the curve the page is about into a line
        tf_panels(*panels, tf, None, colour=REJECTED_COLOUR, ls="-", marker=".", lw=0.6,
                  ms=2, alpha=0.7, zorder=2, period_range=period_range, bars=False)
        index.append(dict(site=site, curve="rejected earth", label=label,
                          path=str(tf.meta.get("path", ""))))
    if rejected:
        ax[0][0].plot([], [], ".", ls="-", color=REJECTED_COLOUR, ms=2, lw=0.6,
                      label="an earth the rule did not choose (%d)" % len(rejected))
    for label, kind, tf in sources:
        rho_curves += tf_panels(*panels, tf, label, colour=KIND_COLOUR.get(kind, "C7"), ls="-",
                                marker="o", lw=0.7, ms=3, alpha=0.8, zorder=3,
                                period_range=period_range)
        index.append(dict(site=site, curve="product of record", label=label, kind=kind,
                          path=str(tf.meta.get("path", ""))))
    if final is not None:
        rho_curves += tf_panels(*panels, final, "the final", colour=FINAL_COLOUR, ls="-", marker="o",
                                lw=1.6, ms=4, alpha=1.0, zorder=5, period_range=period_range)
        index.append(dict(site=site, curve="final", label="the final",
                          path=str(final.meta.get("path", ""))))
    _dress(panels, rho_curves, period_range)
    handles, labels = ax[0][0].get_legend_handles_labels()
    fig.suptitle(title or "%s: the final transfer function" % site, y=0.99, fontsize=12)
    if header_lines:
        fig.text(0.02, 0.965, "\n".join(str(x) for x in header_lines), fontsize=6.5, va="top",
                 family="monospace")
    if handles:
        fig.legend(handles, labels, fontsize=7, loc="upper right", bbox_to_anchor=(0.995, 0.965),
                   ncol=2, framealpha=0.9)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out, index


def final_gallery(sites, finals: dict, out_dir, stem="gallery_final", per_page=6, period_range=None,
                  title="", comparisons=None, figsize=(12.0, 2.1)):
    """rho xy/yx and phase xy/yx of every final, `per_page` sites a page, one legend on the first page.

    `finals` is {site: TFData} and `comparisons` an optional {site: [(label, TFData)]} drawn in black behind.
    Writes <out_dir>/<stem>_p<k>.png and <out_dir>/<stem>_index.csv.
    """
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comparisons = comparisons or {}
    pages, index = [], []
    chunks = [list(sites)[i:i + int(per_page)] for i in range(0, len(sites), int(per_page))]
    for page, chunk in enumerate(chunks, start=1):
        fig, axes = plt.subplots(len(chunk), 4, figsize=(figsize[0], figsize[1] * len(chunk)),
                                 squeeze=False)
        for row, site in enumerate(chunk):
            rho_curves = []
            drawn = [(lab, tf, COMPARISON_COLOUR, 1) for lab, tf in comparisons.get(site, [])]
            if finals.get(site) is not None:
                drawn.append(("the final", finals[site], FINAL_COLOUR, 3))
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
                    # comparison three decades away would otherwise squeeze every final on the page into a
                    # line, and it stays drawn, running off the panel where it disagrees that far
                    if colour == FINAL_COLOUR:
                        rho_curves.append(rho[m])
                    index.append(dict(page=page, row=row, site=site, curve=label, component=comp,
                                      n_periods=int(m.sum()), path=str(tf.meta.get("path", ""))))
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
        fig.suptitle("%s -- page %d of %d" % (title or "the finals", page, len(chunks)), fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        path = out_dir / ("%s_p%d.png" % (stem, page))
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        pages.append(path)
    idx = pd.DataFrame(index)
    idx_path = out_dir / ("%s_index.csv" % stem)
    idx.to_csv(idx_path, index=False)
    return pages, idx, idx_path


def _strip(ax, x, y, ok, title, ylabel, criterion=None, side="below", log=False, labels=None):
    """One clause as its statistic against its criterion line: the rows that hold it, and the rows that do not.

    `side` says which way the criterion is read -- "below" where the statistic must sit under the line,
    "above" where it must sit over it -- so the shaded half of the panel is the half the clause refuses.
    """
    if log:
        ax.set_yscale("log")
    ax.plot(np.asarray(x)[ok], np.asarray(y)[ok], "o", ms=3, color=PASS_COLOUR, alpha=0.8, zorder=3,
            label="holds this clause")
    ax.plot(np.asarray(x)[~ok], np.asarray(y)[~ok], "o", ms=3, color=FAIL_COLOUR, alpha=0.9, zorder=4,
            label="fails it")
    # the limits are taken before the shading, and put back after it: an axhspan reaching to the current
    # edge makes that edge the data and matplotlib rescales the axis around it
    lo, hi = ax.get_ylim()
    if criterion is not None:
        # the criterion line has to be inside the panel: a clause every row holds comfortably would
        # otherwise draw its line off the top and leave the reader nothing to read the points against
        if log:
            lo, hi = min(lo, criterion / 1.6), max(hi, criterion * 1.6)
        else:
            pad = 0.08 * max(hi - lo, abs(criterion))
            lo, hi = min(lo, criterion - pad), max(hi, criterion + pad)
        ax.axhline(criterion, color="C3", ls="--", lw=1.0, zorder=2)
        ax.axhspan(criterion, hi if side == "below" else lo, color="C3", alpha=0.07, zorder=1)
        ax.set_ylim(lo, hi)
    ax.set(title=title, ylabel=ylabel)
    ax.grid(alpha=GRID_ALPHA, which="both")
    ax.tick_params(labelsize=7)
    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)


def rule_page(readings, out, bar_max=1.0, z_slope_cut=-0.75, quadrant_frac=0.70, slope_tol=1.2,
              title="", figsize=(13.0, 8.5)):
    """The earth rule as five clauses, each as its own statistic against its own criterion line.

    One point per product and component, ordered as the readings table is. A point below the shaded half of
    a panel holds that clause; a point inside it is refused by that clause, and the earth flag is the AND of
    the five.
    """
    import matplotlib.pyplot as plt

    d = readings[readings.status == "ok"].reset_index(drop=True)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    x = np.arange(len(d))
    bar = np.asarray(d.bar_10_1000, float)
    zs = np.asarray(d.z_slope, float)
    qf = np.asarray(d.quadrant_frac, float)
    hn = np.asarray(d.held_n, float)
    ss = np.abs(np.asarray(d.short_slope, float))
    _strip(axes[0][0], x, bar, np.isfinite(bar) & (bar <= bar_max), "the bar over 10-1000 s",
           "median error over |Z|", bar_max, "below", log=True)
    _strip(axes[0][1], x, zs, ~(np.isfinite(zs) & (zs < z_slope_cut)),
           "the impedance slope", "d log|Z| / d log T", z_slope_cut, "above")
    _strip(axes[0][2], x, qf, np.isfinite(qf) & (qf >= quadrant_frac), "the quadrant",
           "fraction of the band in (0, 90) deg", quadrant_frac, "above")
    _strip(axes[1][0], x, np.maximum(hn, 0.5), hn > 0, "the held range", "periods held", 1.0, "above",
           log=True)
    if np.isfinite(ss).any():
        ok_ss = ~(np.isfinite(ss) & (ss > slope_tol))
        _strip(axes[1][1], x[np.isfinite(ss)], ss[np.isfinite(ss)], ok_ss[np.isfinite(ss)],
               "the short slope", "|d log rho / d log T|", slope_tol, "below")
    else:
        axes[1][1].text(0.5, 0.5, "UNJUDGED at every row:\nthe live band leaves no part of 2-20 s\nwith "
                                  "four periods to fit", ha="center", va="center", fontsize=9,
                        transform=axes[1][1].transAxes)
        axes[1][1].set(title="the short slope", xticks=[], yticks=[])
    earth = d.earth.astype(bool).values
    axes[1][2].bar(["an earth", "not an earth"], [int(earth.sum()), int((~earth).sum())],
                   color=[PASS_COLOUR, FAIL_COLOUR])
    axes[1][2].set(title="the rule's verdict", ylabel="rows")
    axes[1][2].grid(alpha=GRID_ALPHA, axis="y")
    for ax in list(axes[0]) + [axes[1][0]]:
        ax.set_xlabel("product and component, in table order", fontsize=8)
    axes[0][0].legend(fontsize=7, loc="best")
    fig.suptitle(title or "the earth rule: each clause against its own criterion line", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def record_page(readings, record, out, title="", figsize=(13.0, 7.0)):
    """The choice as it was made: every earth's bar per site and component, the chosen one ringed.

    A site whose column holds points but no ring is a component whose earths corroborate nothing; a site
    with no point at all has no earth to choose from.
    """
    import matplotlib.pyplot as plt

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sites = sorted(set(record.site))
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    for ax, comp in zip(axes, ("xy", "yx")):
        g = readings[(readings.status == "ok") & (readings.component == comp)]
        chosen = record[record.component == comp].set_index("site")
        for k, site in enumerate(sites):
            rows = g[(g.site == site) & g.earth.astype(bool)]
            other = g[(g.site == site) & ~g.earth.astype(bool)]
            ax.plot([k] * len(other), np.asarray(other.bar_10_1000, float), "x", ms=3,
                    color=REJECTED_COLOUR, alpha=0.7,
                    label=("not an earth" if k == 0 else None))
            agree = rows[rows.agree_n > 0]
            alone = rows[rows.agree_n == 0]
            ax.plot([k] * len(alone), np.asarray(alone.bar_10_1000, float), "o", ms=4, mfc="none",
                    color=FAIL_COLOUR, label=("an earth that corroborates nothing" if k == 0 else None))
            ax.plot([k] * len(agree), np.asarray(agree.bar_10_1000, float), "o", ms=4,
                    color=PASS_COLOUR, label=("an earth agreeing with another kind" if k == 0 else None))
            if site in chosen.index and str(chosen.loc[site, "product"]) != "none":
                ax.plot([k], [float(chosen.loc[site, "bar_10_1000"])], "o", ms=11, mfc="none",
                        mew=1.6, color=FINAL_COLOUR,
                        label=("the product of record" if k == 0 else None))
        ax.set(yscale="log", ylabel="bar over 10-1000 s", title="component %s" % comp)
        ax.grid(alpha=GRID_ALPHA, which="both")
    axes[0].legend(fontsize=7, loc="best", ncol=2)
    axes[1].set_xticks(range(len(sites)))
    axes[1].set_xticklabels(sites, rotation=90, fontsize=7)
    fig.suptitle(title or "the product of record: the smallest bar among the agreeing earths", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def halves_page(halves, out, agree_rho=0.20, agree_phase=5.0, title="", figsize=(13.0, 6.5)):
    """The split-half departures against the two criterion lines, one point per candidate and component."""
    import matplotlib.pyplot as plt

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    d = halves[halves.reproducible != "UNJUDGED"].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    labels = ["%s %s" % (r.site, r.component) for r in d.itertuples()]
    x = np.arange(len(d))
    ok = np.asarray(d.reproducible == "yes")
    _strip(axes[0], x, np.asarray(d.rho_dev, float), ok, "apparent resistivity",
           "median |rho ratio - 1| between the halves", agree_rho, "below", labels=labels)
    _strip(axes[1], x, np.asarray(d.phase_dev_deg, float), ok, "phase",
           "median |phase difference| between the halves, deg", agree_phase, "below", labels=labels)
    if len(d):
        axes[0].legend(fontsize=7, loc="best")
    fig.suptitle(title or "reproducibility: each half against the other, over 10-1000 s", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def halves_curves(site, whole, first, second, out, period_range=None, title="", figsize=(11.0, 5.0)):
    """One candidate's two half products drawn against the whole-record product they were split from."""
    import matplotlib.pyplot as plt

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    for col, comp in enumerate(OFF_DIAGONAL):
        for tf, label, colour, lw in ((whole, "the whole record", FINAL_COLOUR, 1.6),
                                      (first, "the first half", "C0", 0.8),
                                      (second, "the second half", "C1", 0.8)):
            if tf is None:
                continue
            p = np.asarray(tf.period, float)
            keep = (np.ones(len(p), bool) if period_range is None
                    else ((p >= period_range[0]) & (p <= period_range[1])))
            rho, _re, ph, _pe = rho_phase(tf.period, tf.z, tf.z_err, comp)
            m = keep & np.isfinite(rho) & (rho > 0)
            if not m.any():
                continue
            axes[0][col].plot(p[m], rho[m], "o", ls="-", ms=2.5, lw=lw, color=colour, alpha=0.9,
                              label=(label if col == 0 else None))
            axes[1][col].plot(p[m], ph[m], "o", ls="-", ms=2.5, lw=lw, color=colour, alpha=0.9)
        axes[0][col].set(xscale="log", yscale="log", title="rho %s" % comp)
        axes[1][col].set(xscale="log", ylim=PHASE_LIM, title="phase %s" % comp, xlabel="period (s)")
        for ax in (axes[0][col], axes[1][col]):
            ax.grid(alpha=GRID_ALPHA, which="both")
    axes[0][0].set_ylabel("rho (Ohm.m)")
    axes[1][0].set_ylabel("phase (deg)")
    axes[0][0].legend(fontsize=7, loc="best")
    fig.suptitle(title or "%s: the whole record and its two halves" % site, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def splice_page(scored, out, max_step_pct=2.0, guard=(18.0, 36.0), clip_pct=12.0, title="",
                figsize=(13.0, 6.5)):
    """Every 10 Hz row that was scored, on both ruled bands against the criterion, the chosen rows ringed.

    `scored` is one row per candidate with `step_below_pct`, `step_above_pct`, `guard_pct` and `chosen`.
    A row further than `clip_pct` from zero is drawn at the edge with an arrow marker, so one noise-biased
    single-station row does not set the scale the 2 per cent criterion has to be read against. The guard
    band carries no criterion line: it is measured, printed and scored by nothing.
    """
    import matplotlib.pyplot as plt

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    d = scored[scored.step_below_pct.notna() | scored.step_above_pct.notna()].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    labels = ["%s %s %s/%s" % (r.site, r.component, r.kind, r.selection) for r in d.itertuples()]
    x = np.arange(len(d))
    chosen = np.asarray(d.chosen, bool) if "chosen" in d.columns else np.zeros(len(d), bool)

    def draw(ax, v, colour, name):
        v = np.asarray(v, float)
        inside = np.isfinite(v) & (np.abs(v) <= clip_pct)
        out_hi = np.isfinite(v) & (v > clip_pct)
        out_lo = np.isfinite(v) & (v < -clip_pct)
        ax.plot(x[inside], v[inside], "o", ms=4, alpha=0.9, color=colour, label=name)
        ax.plot(x[out_hi], np.full(out_hi.sum(), clip_pct), "^", ms=5, color=colour, alpha=0.9)
        ax.plot(x[out_lo], np.full(out_lo.sum(), -clip_pct), "v", ms=5, color=colour, alpha=0.9)

    draw(axes[0], d.step_below_pct, PASS_COLOUR, "8-16 s, below the join")
    draw(axes[0], d.step_above_pct, "C0", "32-100 s, above the join")
    if chosen.any():
        axes[0].plot(x[chosen], np.zeros(chosen.sum()), "o", ms=12, mfc="none", mew=1.4,
                     color=FINAL_COLOUR, label="the row that was joined")
    for y in (max_step_pct, -max_step_pct):
        axes[0].axhline(y, color="C3", ls="--", lw=1.0)
    axes[0].set(title="the step at the join, the worse band governing", ylim=(-clip_pct * 1.1,
                                                                             clip_pct * 1.1),
                ylabel="the 10 Hz level over the 1 Hz level, per cent")
    axes[0].legend(fontsize=7, loc="best")
    draw(axes[1], d.guard_pct, REJECTED_COLOUR, "measured, scored by nothing")
    axes[1].set(title="the %g-%g s guard band: the 20.6 s instrument line" % guard,
                ylim=(-clip_pct * 1.1, clip_pct * 1.1),
                ylabel="the 10 Hz level over the 1 Hz level, per cent")
    axes[1].legend(fontsize=7, loc="best")
    for ax in axes:
        ax.grid(alpha=GRID_ALPHA)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
    fig.suptitle(title or "the splice: the step at the join against the %.1f per cent criterion"
                 % max_step_pct, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def kind_key(kinds=()) -> pd.DataFrame:
    """The colour key a page's legend abbreviates: the word, the code key and the colour of each kind."""
    return pd.DataFrame([dict(word=KIND_WORD.get(k, k), key=k, colour=KIND_COLOUR.get(k, "C7"))
                         for k in kinds])
