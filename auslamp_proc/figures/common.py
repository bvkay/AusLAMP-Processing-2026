"""What every figure of the package does last: a short title, a wrapped caption, and the file.

A title that carries the method, its parameters and the reading runs off both ends of the canvas at any
figure width, so the two are separated: the suptitle is one short line naming the site and what the figure
is, and the caption sits under the axes in smaller text, wrapped to the figure's own width, carrying what was
done and with which values.

The wrap width is taken from the figure: at CAPTION_SIZE = 9 pt a proportional face averages about
CHARS_PER_INCH = 12.5 characters to the inch, so a 13 in figure takes about 162 characters a line. The space
is reserved by growing the canvas -- CAPTION_LINE_IN per caption line at the bottom, TITLE_PAD_IN at the top
-- and putting the axes back where tight_layout left them in inches, so neither the panels nor an axes title
is squeezed and the suptitle sits in a strip no axes reaches into.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import textwrap
from pathlib import Path

DPI = 110
TITLE_SIZE = 11
CAPTION_SIZE = 9
CHARS_PER_INCH = 12.5           # characters a line at CAPTION_SIZE in a proportional face
CAPTION_LINE_IN = 0.20          # the height one caption line is given, in inches
CAPTION_PAD_IN = 0.12           # the gap between the axes and the first caption line
TITLE_PAD_IN = 0.34             # the height the suptitle is given


def wrap(caption, width_in, chars_per_inch=CHARS_PER_INCH) -> list:
    """The caption as a list of lines wrapped to the figure's width."""
    n = max(40, int(float(width_in) * float(chars_per_inch)))
    out = []
    for para in str(caption or "").split("\n"):
        out += textwrap.wrap(para, n) or [""]
    return [ln for ln in out if ln != ""] or []


def finish(fig, title, caption, out, dpi=DPI, chars_per_inch=CHARS_PER_INCH):
    """Set a short suptitle, put a wrapped caption under the axes, reserve the space, save and return the path.

    `title` is one line -- the site and what the figure is. `caption` is the sentence that would otherwise
    have been the title: what was done, with which parameters, and what the numbers were.
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
    # every axes keeps the size and the position the layout gave it, in inches: the figure grows by the strip
    # the caption needs at the bottom and the strip the title needs at the top, and each axes is placed back
    # where it was inside the taller canvas. Nothing is squashed, and an axes title cannot meet the suptitle
    # because the suptitle is drawn in a strip no axes reaches into.
    pad_b = (CAPTION_PAD_IN + CAPTION_LINE_IN * len(lines)) if lines else 0.0
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
    if lines:
        y = (pad_b - CAPTION_PAD_IN) / h2
        for k, line in enumerate(lines):
            fig.text(0.01, y - k * CAPTION_LINE_IN / h2, line, fontsize=CAPTION_SIZE, va="top",
                     ha="left", color="0.25")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out
