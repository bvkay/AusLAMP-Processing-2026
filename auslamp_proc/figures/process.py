"""The figures of workbook 03: what each step of one site's processing did, beside the table that decided it.

One helper per figure, each taking what its section already computed and writing a PNG at dpi 110 under
the run's work root. Nothing is estimated here: every number drawn is one the section printed.

Each figure carries a SHORT title -- the site and what the figure is, one line that fits at any
width -- and a CAPTION under the axes in smaller text, wrapped to the figure's width, naming the parameters a
student would change and what changing them does. Both go through figures.common.finish, which reserves the
caption's space before the save.

The conventions are the package's. Period is a log x axis labelled `period (s)`; apparent resistivity is log
in Ohm.m; phase runs 0-90 deg with the yx panel labelled `+ 180 deg`; time series carry `channel (unit)` in
nT and mV/km against `days from <t0> UTC`; series are C0..C9, refused or excluded items grey, masked spans
grey; grids at alpha 0.25. The transfer-function panels are drawn through figures.transfer_functions.tf_panels and
dressed by its own _dress, so every transfer-function panel of the series reads the same way.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from .common import finish
from .transfer_functions import _dress, tf_panels
from .record import minute_stats

DPI = 110
DAY = 86400.0
GRID_ALPHA = 0.25
UNIT = {"Hx": "nT", "Hy": "nT", "Hz": "nT", "Ex": "mV/km", "Ey": "mV/km"}


def _fig(nrows=1, ncols=1, figsize=(13, 6), **kw):
    import matplotlib.pyplot as plt
    return plt.subplots(nrows, ncols, figsize=figsize, **kw)


def _iso(t) -> str:
    return datetime.fromtimestamp(float(t), timezone.utc).strftime("%Y-%m-%d %H:%M")


def _num(v, default=np.nan) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _txt(v, fmt="%.0f") -> str:
    """A number for a caption, or `not measured` where the sidecar carries none."""
    x = _num(v)
    return (fmt % x) if np.isfinite(x) else "not measured"


# ---------------------------------------------------------------- 1 the frame

def frame_hodogram(h_before, h_after, angle, site, table, out, fs=1.0, figsize=(13, 5.5)):
    """The horizontal pair before and after the turn, with the angles of a set of sites beside it.

    `h_before` and `h_after` are {'Hx','Hy'} of the site as read and as turned; `table` carries one row per
    site with site, angle_deg and declination_deg. A table of one site carries no distribution to draw, so
    the bars are dropped and the hodogram takes the whole figure; the angle and the declination are then a
    line of the caption and of the table above the figure.
    """
    import matplotlib.pyplot as plt
    bars = len(table) > 1
    if bars:
        fig, ax = _fig(1, 2, figsize=figsize, gridspec_kw=dict(width_ratios=[1, 1.6]))
    else:
        fig = plt.figure(figsize=(figsize[0] * 0.55, figsize[1]))
        ax = [fig.add_subplot(1, 1, 1), None]
    per = max(1, int(round(60 * float(fs))))
    xs, ys, means = [], [], []
    for h, label, colour in ((h_before, "as read", "C7"), (h_after, "turned", "C0")):
        x = minute_stats(np.asarray(h["Hx"], float), per)[0]
        y = minute_stats(np.asarray(h["Hy"], float), per)[0]
        g = np.isfinite(x) & np.isfinite(y)
        ax[0].plot(x[g], y[g], ".", ms=1.4, color=colour, alpha=0.5, label=label)
        mx, my = float(np.nanmean(x[g])), float(np.nanmean(y[g]))
        ax[0].plot([mx], [my], "o", ms=6, color=colour, mec="k", mew=0.6)
        xs.append(x[g])
        ys.append(y[g])
        means.append((mx, my, colour))
    # the mean vector runs from the origin, which sits tens of thousands of nT away from the cloud, so it
    # is drawn as the ray through the mean rather than as an arrow that would fill the whole panel
    allx = np.concatenate(xs) if xs else np.zeros(1)
    ally = np.concatenate(ys) if ys else np.zeros(1)
    px = 0.08 * max(1.0, float(np.ptp(allx)))
    py = 0.15 * max(1.0, float(np.ptp(ally)))
    x_lo, x_hi = float(allx.min()) - px, float(allx.max()) + px
    for mx, my, colour in means:
        if mx:
            ax[0].plot([x_lo, x_hi], [my / mx * x_lo, my / mx * x_hi], ls="--", lw=1.1, color=colour)
    a_txt = ("%+.2f deg" % angle) if isinstance(angle, (int, float)) and np.isfinite(angle) \
        else "%s regime(s)" % (len(angle) if hasattr(angle, "__len__") else "several")
    ax[0].axhline(0, color="0.5", lw=0.8)
    ax[0].set(xlabel="Hx (nT)", ylabel="Hy (nT)", title="%s: theta = %s" % (site, a_txt),
              xlim=(x_lo, x_hi), ylim=(float(ally.min()) - py, float(ally.max()) + py))
    ax[0].grid(alpha=GRID_ALPHA)
    ax[0].legend(fontsize=8, loc="best", markerscale=5)

    sites = list(table.site)
    ang = np.array([_num(v) for v in table.angle_deg], float)
    dec = np.array([_num(v) for v in table.declination_deg], float)
    xi = np.arange(len(sites))
    if not bars:
        n_reg = int(sum(1 for v in table.angle_deg if not np.isfinite(_num(v))))
        return finish(fig, "%s: the horizontal pair before and after the turn" % site,
                      "The per-minute mean of the pair as read (grey) and turned (blue), each with its own "
                      "mean marked and the ray from the origin through that mean drawn dashed; the angle "
                      "between the two rays is the turn, theta = %s, and it puts the mean Hy at zero. The "
                      "turn is rigid, so the cloud moves and does not change shape. The IGRF declination "
                      "here is %s: the angle is the sensor's own misalignment and the declination is the "
                      "field's, and neither is the other. What to change: rot_regimes in decisions.csv "
                      "splits the record where the sensor was moved and one angle no longer fits, and "
                      "rot_drop trims the stretches the angle is not measured over; this site carries %d "
                      "regime list(s) rather than one angle."
                      % (a_txt, ("%+.2f deg" % dec[0]) if np.isfinite(dec).any() else "not computed",
                         n_reg), out)
    ax[1].bar(xi, ang, color=["C3" if s == site else "C0" for s in sites], width=0.72,
              label="the rotation angle (the sensor's)")
    ax[1].plot(xi, dec, "kv", ms=5, label="the IGRF declination (the field's)")
    ax[1].axhline(0, color="0.4", lw=0.8)
    both = np.concatenate([ang[np.isfinite(ang)], dec[np.isfinite(dec)]]) if len(sites) else np.zeros(1)
    lo, hi = (np.percentile(both, 2), np.percentile(both, 98)) if len(both) else (-1.0, 1.0)
    pad = max(2.0, 0.35 * (hi - lo))
    ax[1].set(ylabel="angle (deg)", xticks=xi, ylim=(lo - pad, hi + pad))
    # a sensor laid back to front sits at about 180 deg and would flatten every other bar, so it is written
    # at the edge instead of setting the scale
    for i, v in enumerate(ang):
        if np.isfinite(v) and not (lo - pad <= v <= hi + pad):
            ax[1].annotate("%+.0f" % v, (i, lo - pad if v < lo else hi + pad), fontsize=6, color="C3",
                           ha="center", va=("bottom" if v < lo else "top"))
    ax[1].set_xticklabels(sites, rotation=90, fontsize=7)
    ax[1].grid(alpha=GRID_ALPHA, axis="y")
    ax[1].legend(fontsize=8, loc="best")
    n_reg = int(sum(1 for v in table.angle_deg if not np.isfinite(_num(v))))
    return finish(fig, "%s: the horizontal pair before and after the turn" % site,
                  "Left: the per-minute mean of the pair as read (grey) and turned (blue), each with its "
                  "own mean marked and the ray from the origin through that mean drawn dashed; the angle "
                  "between the two rays is the turn, theta = %s, and it puts the mean Hy at zero. The turn "
                  "is rigid, so the cloud moves and does not change shape. Right: one bar per site of its "
                  "rotation angle with "
                  "the IGRF declination as a marker; the angle is the sensor's own misalignment and the "
                  "declination is the field's, and neither is the other. Over %d site(s) the angles run "
                  "%.2f to %.2f deg and the declinations %.2f to %.2f deg. What to change: rot_regimes in "
                  "decisions.csv splits the record where the sensor was moved and one angle no longer "
                  "fits, and rot_drop trims the stretches the angle is not measured over; %d site(s) here "
                  "carry a regime list rather than one angle."
                  % (a_txt, len(sites), np.nanmin(ang) if np.isfinite(ang).any() else np.nan,
                     np.nanmax(ang) if np.isfinite(ang).any() else np.nan,
                     np.nanmin(dec) if np.isfinite(dec).any() else np.nan,
                     np.nanmax(dec) if np.isfinite(dec).any() else np.nan, n_reg), out)


# ---------------------------------------------------------------- 2 the remote site

def remote_scatter(scores, choice, site, out, branches=None, coh_min=0.5, coh_relax=0.3,
                   overlap_fraction=0.9, named=None, figsize=(13, 5.5)):
    """The candidates of one target as coherence against overlap, and the branch taken over every site.

    `scores` is the candidate table, `choice` the store's chosen remote dict, `branches` a mapping of branch
    number to the number of sites that took it, and `named` the decisions.csv remote where one differs.
    """
    fig, ax = _fig(1, 2, figsize=figsize, gridspec_kw=dict(width_ratios=[1.7, 1]))
    t = scores[scores.overlap_days > 0] if len(scores) else scores
    own = _num(t.own_days.iloc[0]) if len(t) else np.nan
    if len(t):
        km = np.array([_num(v) for v in t.km], float)
        size = 20 + 180 * (1.0 - (km - np.nanmin(km)) / max(1e-9, np.nanmax(km) - np.nanmin(km)))
        for clean, face, label in ((True, None, "clean"), (False, "none", "not clean")):
            m = np.array([bool(v) == clean for v in t.clean], bool)
            if not m.any():
                continue
            ax[0].scatter(np.array([_num(v) for v in t.overlap_days])[m],
                          np.array([_num(v) for v in t.coh])[m], s=size[m], facecolors=face,
                          edgecolors="C0", linewidths=1.1, alpha=0.85, label=label)
        for j, r in enumerate(t.itertuples()):
            if np.isfinite(_num(r.coh)):
                ax[0].annotate(r.name, (_num(r.overlap_days), _num(r.coh)), fontsize=6,
                               xytext=(3, 4 + 7 * (j % 3)), textcoords="offset points", color="0.35")
    ax[0].axhline(coh_min, color="C3", ls="--", lw=1.1, label="COH_MIN = %g" % coh_min)
    ax[0].axhline(coh_relax, color="C1", ls=":", lw=1.1, label="COH_RELAX = %g" % coh_relax)
    if np.isfinite(own):
        ax[0].axvline(overlap_fraction * own, color="C2", ls="-.", lw=1.1,
                      label="%.0f %% of the %.1f d the target can use" % (100 * overlap_fraction, own))
    for name, marker, colour, label in ((choice.get("rule_name"), "o", "C3", "the rule's choice"),
                                        (named, "s", "C4", "decisions.csv")):
        if not name or not len(t):
            continue
        row = t[t.name == name]
        if not len(row):
            continue
        ax[0].scatter([_num(row.overlap_days.iloc[0])], [_num(row.coh.iloc[0])], s=300, marker=marker,
                      facecolors="none", edgecolors=colour, linewidths=1.8, label=label)
    if named and (not len(t) or named not in set(t.name)):
        ax[0].text(0.02, 0.86, "decisions.csv names %s, which is not in the candidate table" % named,
                   transform=ax[0].transAxes, fontsize=7, color="C4")
    ax[0].set(xlabel="usable overlap (days)", ylabel="20-200 s coherence", ylim=(0, 1),
              title="%s: %d candidate(s), marker size by distance" % (site, len(t)))
    ax[0].grid(alpha=GRID_ALPHA)
    ax[0].legend(fontsize=7, loc="upper left")

    b = dict(branches or {})
    keys = sorted(b)
    ax[1].bar([str(k) for k in keys], [b[k] for k in keys], color="C0", width=0.6)
    ax[1].set(xlabel="branch", ylabel="sites", title="the branch the rule took")
    ax[1].grid(alpha=GRID_ALPHA, axis="y")
    for i, k in enumerate(keys):
        ax[1].text(i, b[k], str(b[k]), ha="center", va="bottom", fontsize=8)
    return finish(fig, "%s: the remote-site candidates and the branch taken" % site,
                  "Left: every candidate of %s scored on its event-free 20-200 s coherence against the "
                  "usable overlap it offers, the marker sized by distance (large is near), filled where the "
                  "candidate is in the clean pool and open where it is not, with the rule's choice ringed. "
                  "Right: the branch the five-branch rule took over every site -- 1 clean, coh >= COH_MIN "
                  "and overlap >= %.0f per cent, the nearest of them; 2 clean and coh >= COH_MIN, the "
                  "longest overlap; 3 clean and coh >= COH_RELAX, the most coherent; 4 coh >= COH_MIN but "
                  "not clean, the fewest events; 5 none of the above, the nearest by km; 0 is a remote "
                  "decisions.csv names. What to change: COH_MIN = %g and COH_RELAX = %g move the branch, "
                  "MIN_OVERLAP_DAYS the vertical line that decides which candidates are scored at all, and "
                  "a name in decisions.csv remote_site overrides all of them."
                  % (site, 100 * overlap_fraction, coh_min, coh_relax), out)


# ---------------------------------------------------------------- 3 the fleet stack

def stack_weights(kept, refused, lags, site, out, series=(), t_start=None, cutoff=0.5, n_max=8, n_min=2,
                  fs=1.0, hours=6.0, bridged=(), steps=None, screen_k=30.0, screen_floor_nt=3.0,
                  edge_s=30.0, step_nt=2.0, level_win_s=3600.0, figsize=(13, 5.5)):
    """The member weights against the cutoff, and the target, two members and the stack over a few hours.

    `series` is [(label, array, colour)] already cut to the window, each demeaned by the caller or here.
    `bridged` is [(start, end)] in samples from the window start where a member sample was bridged by the
    spike screen, and `steps` the sidecar's steps_per_million record, whose 'stack' and
    'members_median_screened' entries are the two numbers the check scores.
    """
    fig, ax = _fig(1, 2, figsize=figsize, gridspec_kw=dict(width_ratios=[1, 1.4]))
    names = list(kept) + sorted(refused)
    xi = np.arange(len(names))
    ax[0].bar(xi[:len(kept)], [float(kept[d]) for d in kept], color="C0", width=0.66,
              label="member weight")
    if refused:
        ax[0].bar(xi[len(kept):], [cutoff] * len(refused), color="0.8", width=0.66, hatch="//",
                  edgecolor="0.5", label="refused")
    ax[0].axhline(cutoff, color="C3", ls="--", lw=1.2, label="STACK_CUTOFF = %g" % cutoff)
    for i, d in enumerate(kept):
        ax[0].text(i, float(kept[d]), "%+.2f s" % float((lags or {}).get(d, 0.0)), ha="center",
                   va="bottom", fontsize=6, rotation=90)
    ax[0].set(ylabel="fleet coherence at 100-1000 s", xticks=xi, ylim=(0, 1.15))
    ax[0].set_xticklabels(names, rotation=90, fontsize=7)
    ax[0].grid(alpha=GRID_ALPHA, axis="y")
    ax[0].legend(fontsize=7, loc="upper right")

    n_drawn = 0
    for label, arr, colour in series:
        v = np.asarray(arr, float)
        g = np.isfinite(v)
        if not g.any():
            continue
        h = np.arange(len(v)) / (3600.0 * float(fs))
        ax[1].plot(h, v - np.nanmean(v), lw=0.7, color=colour, alpha=0.85, label=label)
        n_drawn += 1
    ax[1].set(xlabel=("hours from %s UTC" % _iso(t_start)) if t_start else "hours",
              ylabel="Hx, demeaned (nT)", title="%g h of one quiet day" % hours)
    ax[1].grid(alpha=GRID_ALPHA)
    marks = [(float(a), float(b)) for a, b in (bridged or ())]
    if marks and n_drawn:
        lo, hi = ax[1].get_ylim()
        span = hi - lo
        band = lo - 0.10 * span
        ax[1].set_ylim(band - 0.05 * span, hi)
        # a bridged span is a handful of samples in six hours and is sub-pixel drawn to scale, so each one
        # is a tick of its own at the span's centre
        ax[1].vlines([0.5 * (a0 + b0) / (3600.0 * float(fs)) for a0, b0 in marks],
                     band - 0.025 * span, band + 0.025 * span, lw=0.9, color="C4",
                     label="a member sample bridged")
        ax[1].axhline(band, color="C4", lw=0.4, alpha=0.35)
    if n_drawn:
        ax[1].legend(fontsize=7, loc="upper right", ncol=2)
    sk = ((steps or {}).get("stack") or {})
    md = ((steps or {}).get("members_median_screened") or {})
    stat = ("The stack reads %s steps above %g nT per million samples on Hx and %s on Hy, against the "
            "median of its screened members, %s and %s: the point of averaging several members is a "
            "reference no rougher than a typical one of them, and the check below the figure scores it."
            % (_txt(sk.get("Hx")), step_nt, _txt(sk.get("Hy")), _txt(md.get("Hx")), _txt(md.get("Hy")))
            if sk or md else
            "The stack's steps above %g nT per million samples and its screened members' median are written "
            "to the sidecar and scored by the check below the figure." % step_nt)
    return finish(fig, "%s: the stack members and what the stack looks like" % site,
                  "Left: each member's weight, which is its median coherence with the fleet at 100-1000 s "
                  "and never its coherence with the target, against the cutoff; the refused members are "
                  "hatched and each kept bar carries the lag the alignment measured, in seconds. Right: %g "
                  "hours of one quiet day with the target's Hx, two members' Hx and the stack's Hx drawn "
                  "over each other after each is demeaned, and under them a mark at every sample where the "
                  "spike screen bridged a member -- %d span(s) in this window. %s %d member(s) are kept and "
                  "%d refused. What to change: STACK_CUTOFF = %g refuses a member below it, STACK_MAX = %d "
                  "caps how many enter, STACK_MIN = %d refuses the stack outright below it because a "
                  "one-member stack is a remote site renamed; SCREEN_K = %g and SCREEN_FLOOR_NT = %g nT set "
                  "how far a member's first difference may depart from the median over members before it is "
                  "bridged, EDGE_S = %g s is what is cut inside every record end and gap end, where the "
                  "logger's settling sits, and LEVEL_WIN_S = %g s is the window each member's offset from "
                  "the fleet's level is averaged over, which is what keeps the composite from stepping when "
                  "a member drops in or out. The alignment is measured on 5-20 s, where a lag shows."
                  % (hours, len(marks), stat, len(kept), len(refused), cutoff, n_max, n_min,
                     screen_k, screen_floor_nt, edge_s, level_win_s), out)


# ---------------------------------------------------------------- 4 the observatory

def observatory_bars(table, out, series=(), t_start=None, code="", site="", fs=1.0, hours=6.0,
                     figsize=(13, 5.5)):
    """The site-observatory coherence at 100-1000 s per site, and the two Hx records over a few hours."""
    t = table.sort_values("site")
    sites = list(t.site)
    coh = np.array([_num(v) for v in t.coh_100_1000s], float)
    km = np.array([_num(v) for v in t.km], float)
    xi = np.arange(len(sites))
    fig, ax = _fig(1, 2, figsize=figsize, gridspec_kw=dict(width_ratios=[1.4, 1.2]))
    ax[0].bar(xi, coh, color=["C3" if s == site else "C0" for s in sites], width=0.72)
    top = np.nanmax(coh) if np.isfinite(coh).any() else 1.0
    for i, d in enumerate(km):
        if np.isfinite(d):
            ax[0].text(i, 0.02 * top, "%.0f km" % d, ha="center", va="bottom", fontsize=6, rotation=90,
                       color="0.2")
    ax[0].set(ylabel="site-%s coherence at 100-1000 s" % (code or "observatory"), xticks=xi, ylim=(0, 1))
    ax[0].set_xticklabels(sites, rotation=90, fontsize=7)
    ax[0].grid(alpha=GRID_ALPHA, axis="y")
    for label, arr, colour in series:
        v = np.asarray(arr, float)
        if not np.isfinite(v).any():
            continue
        h = np.arange(len(v)) / (3600.0 * float(fs))
        ax[1].plot(h, v - np.nanmean(v), lw=0.7, color=colour, alpha=0.9, label=label)
    ax[1].set(xlabel=("hours from %s UTC" % _iso(t_start)) if t_start else "hours",
              ylabel="Hx, demeaned (nT)", title="%g h of one quiet day" % hours)
    ax[1].grid(alpha=GRID_ALPHA)
    if len(series):
        ax[1].legend(fontsize=7, loc="upper right")
    good = coh[np.isfinite(coh)]
    return finish(fig, "%s against the sites at 100-1000 s" % (code or "the observatory"),
                  "Left: one bar per site of its coherence with the %s one-second record over 100-1000 s, "
                  "with the distance to the observatory written on each; over %d scored site(s) it runs "
                  "%.2f to %.2f. Right: %g hours of one quiet day with the site's Hx and the observatory's "
                  "Hx, the observatory turned into its own mean-field frame like any other member and both "
                  "demeaned. This is what the observatory kind can buy and what it cannot: at 100-1000 s "
                  "the two records carry the same field, and at the short end a reference several hundred "
                  "kilometres away shares almost nothing. The observatory is never shifted, because a delay "
                  "on a single reference cancels exactly in Z. What to change: the observatory code and its "
                  "archive in survey.yaml choose which record this is; nothing here has a threshold."
                  % (code or "observatory", len(good), np.min(good) if len(good) else np.nan,
                     np.max(good) if len(good) else np.nan, hours), out)


# ---------------------------------------------------------------- 5 the bands

def band_ladder(tables, files, out, params="", figsize=(13, 3.6)):
    """One panel per band file: each band as a bar over its own period range, at its decimation level.

    `tables` is {key: the band table aurora_run.band_table returns} and `files` is {key: the file name}.
    """
    fig, axes = _fig(1, len(tables), figsize=figsize)
    axes = np.atleast_1d(axes)
    for ax, (k, tab) in zip(axes, sorted(tables.items())):
        for r in tab.itertuples():
            ax.plot([r.lower_s, r.upper_s], [r.level, r.level], lw=6, solid_capstyle="butt",
                    color="C%d" % (r.level % 10), alpha=0.8)
            ax.plot(r.centre_s, r.level, "k|", ms=8)
        ax.set(xscale="log", xlabel="period (s)", ylabel="decimation level",
               title="%s: %s, %d bands" % (k, files.get(k, ""), len(tab)),
               yticks=range(tab.level.nunique()))
        ax.grid(alpha=GRID_ALPHA, which="both")
    return finish(fig, "the band files, level by level",
                  "Each band drawn as a bar over its own lower and upper period at the decimation level it "
                  "belongs to, with its centre marked. The lines of a band file are FFT harmonics of the "
                  "window and not of the record, so the file, the level count and the window are one object "
                  "in the package and these tables are read through Aurora's own band machinery rather than "
                  "off the text file. The cascade is [1] + [4] x (levels - 1): the rate falls by four at "
                  "each level after the first. What to change: PARAMS = %s selects the Aurora parameter set, "
                  "and the band file of a rate is fixed with it." % (params or "the parameter set"), out)


# ---------------------------------------------------------------- 6 the mask

def mask_record(t0, arrays, keep, out, site="", spans=(), fs=1.0, min_segment_s=3600.0,
                channels=("Hx", "Ex"), figsize=(13, 6.5)):
    """The record with the transient and E-burst spans over it, and the kept runs as a row of bars.

    `spans` is [(label, colour, [(t_a, t_b), ...])] in unix seconds.
    """
    n = len(arrays[channels[0]])
    per = max(1, int(round(60 * float(fs))))
    days = (np.arange(n // per) * per) / (float(fs) * DAY)
    fig, ax = _fig(len(channels) + 1, 1, figsize=figsize, sharex=True,
                   gridspec_kw=dict(height_ratios=[3] * len(channels) + [1]))
    for k, ch in enumerate(channels):
        v = np.asarray(arrays[ch], float)
        m, lo, hi = minute_stats(v, per)
        a = ax[k]
        a.fill_between(days[:len(m)], lo, hi, color="0.8", lw=0)
        a.plot(days[:len(m)], m, color="C%d" % k, lw=0.6)
        g = np.isfinite(v)
        if g.any():
            a.set_ylim(np.percentile(v[g], 1), np.percentile(v[g], 99))
        a.set(ylabel="%s (%s)" % (ch, UNIT.get(ch, "")))
        a.grid(alpha=GRID_ALPHA)
    for label, colour, items in spans:
        for j, (ta, tb) in enumerate(items):
            for a in ax[:len(channels)]:
                a.axvspan((ta - t0) / DAY, (tb - t0) / DAY, color=colour, alpha=0.25, lw=0,
                          label=(label if j == 0 and a is ax[0] else None))
    if spans:
        ax[0].legend(fontsize=7, loc="upper right", ncol=len(spans))
    k = np.asarray(keep, bool)
    d = np.diff(np.concatenate(([0], k.view(np.int8), [0])))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    floor = int(round(float(min_segment_s) * float(fs)))
    n_runs = n_short = written = 0
    for a0, b0 in zip(st, en):
        long_enough = (b0 - a0) >= floor
        n_runs += int(long_enough)
        n_short += int(not long_enough)
        written += int(b0 - a0) if long_enough else 0
        ax[-1].axvspan(a0 / (fs * DAY), b0 / (fs * DAY), color=("C2" if long_enough else "C3"),
                       alpha=0.9, lw=0)
    ax[-1].set(ylabel="runs", yticks=[], xlabel="days from %s UTC" % _iso(t0))
    ax[-1].set_ylim(0, 1)
    ax[-1].grid(alpha=GRID_ALPHA, axis="x")
    return finish(fig, "%s: the mask and the runs it leaves" % site,
                  "The record with every masked interval drawn over it and, underneath, the stretches the "
                  "mask keeps: green where the stretch is at least %g s and becomes one Aurora run, red "
                  "where it is shorter and is dropped because it holds no complete window at the deep "
                  "decimation levels. The record is never cut and nothing is concatenated: a mask is "
                  "applied as one run per kept stretch, because joining the kept pieces puts a step at each "
                  "join that is common to E, H and the reference and a robust regression fits such a step "
                  "rather than down-weighting it. The mask keeps %.2f per cent of the record over %d run(s), "
                  "and the floor then drops %d short stretch(es), a further %.2f per cent of the whole "
                  "record: two events forty minutes apart cost the forty minutes between them as well as "
                  "themselves. What to change: MIN_SEGMENT_S = %g sets the floor, and the pool thresholds "
                  "own_ratio, fleet_excess and e_burst_ratio in survey.yaml decide which intervals are "
                  "masked at all."
                  % (min_segment_s, 100 * float(k.mean()), n_runs, n_short,
                     100 * (int(k.sum()) - written) / max(1, len(k)), min_segment_s), out)


# ---------------------------------------------------------------- 7 the 10 Hz stretch

NO_TEN_HZ = "the 10 Hz pass was not run for this site: RATES holds no 10 Hz"


def hour_selection(table, sel, curves, site, out, join_s=16.0, bands=((8.0, 16.0), (32.0, 100.0)),
                   period_range=(0.6, 2000.0), band_s=(20.0, 200.0), coh_min=0.5, seed=20260916,
                   figsize=(13, 11), no_ten_hz=NO_TEN_HZ):
    """The two hourly coherences with the stretch and its control as spans, and what they estimated.

    `table` is the hour score table, `sel` the selection record keyed by tag (`stretch` and `control`), and
    `curves` is [(label, TFData, colour, linestyle)] with the 1 Hz baseline solid and the 10 Hz stretches
    dashed.

    Where `curves` holds no dashed curve the site has no 10 Hz transfer function of the stretch, and the
    lower panels carry `no_ten_hz` in place of a curve rather than the 1 Hz row alone.
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 1])
    ax_s = fig.add_subplot(gs[0, :])
    ax_rho_xy, ax_rho_yx = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    ax_ph_xy, ax_ph_yx = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
    t0 = float(np.min(table.t_start)) if len(table) else 0.0
    d = (np.asarray(table.t_start, float) - t0) / DAY
    for col, colour, label in (("coh_xy", "C0", "Ex with Hy"), ("coh_yx", "C1", "Ey with Hx")):
        if col in table.columns:
            ax_s.plot(d, np.asarray(table[col], float), color=colour, lw=0.5, label=label)
    ax_s.axhline(float(coh_min), color="0.3", ls="--", lw=0.9,
                 label="the threshold both lines must hold, %.2f" % float(coh_min))
    top = 1.0
    tags = [k for k in ("stretch", "control") if k in (sel or {})]
    row_h, gap = 0.10 * top, 0.02 * top
    for j, tag in enumerate(tags):
        item = sel[tag]
        colour = "0.45" if item.get("random") else "C2"
        lo = top * 1.06 + j * (row_h + gap)
        hi = lo + row_h
        first = True
        for ta, tb in item["hours"]:
            ax_s.fill_between([(ta - t0) / DAY, (tb - t0) / DAY], lo, hi, color=colour, lw=0.4,
                              edgecolor=colour, hatch=("//" if item.get("random") else None),
                              label=("%s: %d h" % (tag, item["n_hours"]) if first else None))
            first = False
        ax_s.text(1.005, 0.5 * (lo + hi), tag, transform=ax_s.get_yaxis_transform(), fontsize=7,
                  va="center", ha="left", color=colour)
    ax_s.set(ylabel="%g-%g s E-H coherence" % band_s, xlabel="days from %s UTC" % _iso(t0),
             ylim=(0, top * 1.06 + max(1, len(tags)) * (row_h + gap)))
    ax_s.grid(alpha=GRID_ALPHA)
    ax_s.legend(fontsize=7, loc="lower right", ncol=4)
    lower = (ax_rho_xy, ax_rho_yx, ax_ph_xy, ax_ph_yx)
    drawn = [c for c in curves if c[1] is not None and c[3] == "--"]
    if not drawn:
        for a in lower:
            a.text(0.5, 0.5, no_ten_hz, transform=a.transAxes, ha="center", va="center", fontsize=9,
                   color="0.35", wrap=True)
            a.set(xticks=[], yticks=[])
            for side in a.spines.values():
                side.set_color("0.8")
    else:
        rho = []
        for label, tf, colour, ls in curves:
            if tf is None:
                continue
            rho += tf_panels(ax_rho_xy, ax_rho_yx, ax_ph_xy, ax_ph_yx, None, None, tf, label,
                             colour=colour, ls=ls, marker="o", period_range=period_range, bars=True)
        _dress((ax_rho_xy, ax_rho_yx, ax_ph_xy, ax_ph_yx, None, None), rho, period_range)
        for a in lower:
            for lo, hi in bands:
                a.axvspan(lo, hi, color="C7", alpha=0.13, lw=0)
            a.axvline(join_s, color="0.3", ls="-.", lw=1)
        ax_rho_xy.legend(fontsize=7, loc="best", ncol=2)
    kept = ", ".join("%s %d h" % (k, sel[k]["n_hours"]) for k in tags)
    return finish(fig, "%s: the hours the 10 Hz pass was run on" % site,
                  "Top: the squared coherence of Ex with Hy and of Ey with Hx over %g-%g s, one value per "
                  "whole UTC hour of the 1 Hz cache, with the threshold both lines must hold drawn as a "
                  "dashed line and the chosen stretch and its control drawn as rows of spans above, each "
                  "named at the right. The hours kept are %s. Bottom: %s. What to change: WINDOW_COH sets "
                  "the threshold, STRETCH_MAX_H the longest stretch taken, and SEED the control's draw. A "
                  "kept hour is 3,600 s = 36,000 samples at 10 Hz, exactly the run floor, so an isolated "
                  "kept hour survives as one Aurora run and adjacent kept hours merge into one longer one."
                  % (band_s[0], band_s[1], kept or "none",
                     (("the transfer functions the stretch and its control made, dashed, against the 1 Hz "
                       "row of the same kind, solid, with the %g s join marked and the two bands a splice "
                       "step is scored on shaded. A stretch that does not beat its control buys efficiency "
                       "and not a different answer, which is what the control is there to show. The "
                       "departure of the 10 Hz row from the 1 Hz row is the survey's own, measured on its "
                       "whole-record passes and written into every 10 Hz file, so an offset common to both "
                       "is the rate and not the choosing" % join_s) if drawn else
                      ("%s, so the panels carry no curve. The hours above are scored and the stretch chosen "
                       "whatever the rate asked for, because the score is a property of the record"
                       % no_ten_hz))),
                  out)
