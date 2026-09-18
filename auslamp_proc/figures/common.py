"""What every figure of the package does last: a short title, a key, a wrapped caption, and the file.

A title that carries the method, its parameters and the reading runs off both ends of the canvas at any
figure width, so the two are separated: the suptitle is one short line naming the site and what the figure
is, and the caption sits under the axes in smaller text, wrapped to the figure's own width, carrying what was
done and with which values.

The wrap width is taken from the figure: at CAPTION_SIZE = 9 pt a proportional face averages about
CHARS_PER_INCH = 12.5 characters to the inch, so a 13 in figure takes about 162 characters a line. The space
is reserved by growing the canvas -- CAPTION_LINE_IN per caption line at the bottom, TITLE_PAD_IN at the top
-- and putting the axes back where tight_layout left them in inches, so neither the panels nor an axes title
is squeezed and the suptitle sits in a strip no axes reaches into.

The legend is the third strip. A key of more than LEGEND_IN_PANEL_MAX entries covers the curves whatever
corner of the axes it is put in (Ben, 2026-09-18: the legend here covers the transfer function), so above
that count it leaves the axes entirely and is drawn between the panels and the caption, over as many columns
as the figure width fits. `legend_below` states that rule once and every page asks it.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import math
import textwrap
from pathlib import Path

DPI = 110
TITLE_SIZE = 11
CAPTION_SIZE = 9
CHARS_PER_INCH = 12.5           # characters a line at CAPTION_SIZE in a proportional face
CAPTION_LINE_IN = 0.20          # the height one caption line is given, in inches
CAPTION_PAD_IN = 0.12           # the gap between the axes and the first caption line
TITLE_PAD_IN = 0.34             # the height the suptitle is given

LEGEND_IN_PANEL_MAX = 6         # above this many entries the key leaves the axes
LEGEND_SIZE = 7
LEGEND_NCOL_MIN = 4
LEGEND_NCOL_MAX = 6
LEGEND_LINE_IN = 0.20           # the height one legend row is given, in inches
LEGEND_PAD_IN = 0.14            # the gap between the axes and the legend
LEGEND_HANDLE_IN = 0.50         # the line sample and its padding, in inches, in front of each entry


def wrap(caption, width_in, chars_per_inch=CHARS_PER_INCH) -> list:
    """The caption as a list of lines wrapped to the figure's width."""
    n = max(40, int(float(width_in) * float(chars_per_inch)))
    out = []
    for para in str(caption or "").split("\n"):
        out += textwrap.wrap(para, n) or [""]
    return [ln for ln in out if ln != ""] or []


def legend_columns(labels, width_in, chars_per_inch=CHARS_PER_INCH) -> int:
    """How many columns a key of these labels is spread over, between LEGEND_NCOL_MIN and LEGEND_NCOL_MAX.

    A column has to hold the longest entry and its line sample, so the width one column needs is the longest
    label at LEGEND_SIZE plus LEGEND_HANDLE_IN. The count is what the figure's own width fits, held inside
    the two bounds and never more than there are entries.
    """
    labels = [str(x) for x in labels]
    if not labels:
        return LEGEND_NCOL_MIN
    per_in = float(chars_per_inch) * CAPTION_SIZE / float(LEGEND_SIZE)
    col_in = max(0.6, max(len(x) for x in labels) / per_in + LEGEND_HANDLE_IN)
    fits = int(max(1, float(width_in) // col_in))
    return int(min(max(min(fits, LEGEND_NCOL_MAX), LEGEND_NCOL_MIN), len(labels)))


def legend_below(labels, max_in_panel=LEGEND_IN_PANEL_MAX) -> bool:
    """True where this many entries would cover the curves and the key belongs under the panels instead."""
    return len(list(labels)) > int(max_in_panel)


def finish(fig, title, caption, out, dpi=DPI, chars_per_inch=CHARS_PER_INCH, legend=None):
    """Set a short suptitle, put the key and a wrapped caption under the axes, save and return the path.

    `title` is one line -- the site and what the figure is. `caption` is the sentence that would otherwise
    have been the title: what was done, with which parameters, and what the numbers were. `legend` is
    (handles, labels) or (handles, labels, ncol): the key is then drawn in its own strip between the panels
    and the caption, outside every axes, and no axes carries one.
    """
    # a figure built with a layout engine is laid out once and the engine is then turned off, so the strips
    # reserved below are not re-laid over; a figure without one is laid out by tight_layout here
    engine = getattr(fig, "get_layout_engine", lambda: None)()
    if engine is None:
        fig.tight_layout()
    else:
        fig.canvas.draw()
        fig.set_layout_engine("none")
    w_in, h_in = fig.get_size_inches()
    lines = wrap(caption, w_in, chars_per_inch)
    handles, labels, ncol = [], [], LEGEND_NCOL_MIN
    if legend:
        handles, labels = list(legend[0]), [str(x) for x in legend[1]]
        ncol = int(legend[2]) if len(legend) > 2 and legend[2] else legend_columns(labels, w_in,
                                                                                  chars_per_inch)
        ncol = max(1, min(ncol, max(1, len(labels))))
    n_rows = int(math.ceil(len(labels) / float(ncol))) if handles else 0
    # every axes keeps the size and the position the layout gave it, in inches: the figure grows by the two
    # strips the caption and the key need at the bottom and the strip the title needs at the top, and each
    # axes is placed back where it was inside the taller canvas. Nothing is squashed, an axes title cannot
    # meet the suptitle, and the key sits where it covers no curve.
    cap_b = (CAPTION_PAD_IN + CAPTION_LINE_IN * len(lines)) if lines else 0.0
    leg_b = (LEGEND_PAD_IN + LEGEND_LINE_IN * n_rows) if n_rows else 0.0
    pad_b = cap_b + leg_b
    pad_t = TITLE_PAD_IN if title else 0.0
    h2 = h_in + pad_b + pad_t
    if pad_b or pad_t:
        boxes = [(a, a.get_position()) for a in fig.axes]
        fig.set_size_inches(w_in, h2, forward=True)
        for ax, box in boxes:
            y0 = (pad_b + box.y0 * h_in) / h2
            y1 = (pad_b + box.y1 * h_in) / h2
            ax.set_position([box.x0, y0, box.width, y1 - y0])
    if title:
        fig.suptitle(str(title), fontsize=TITLE_SIZE, y=1.0 - 0.3 * pad_t / h2, va="top")
    if n_rows:
        # the key's own strip, between the axes above it and the caption below it
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, cap_b / h2), ncol=ncol,
                   fontsize=LEGEND_SIZE, frameon=False, borderaxespad=0.0)
    if lines:
        y = (cap_b - CAPTION_PAD_IN) / h2
        for k, line in enumerate(lines):
            fig.text(0.01, y - k * CAPTION_LINE_IN / h2, line, fontsize=CAPTION_SIZE, va="top",
                     ha="left", color="0.25")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out
