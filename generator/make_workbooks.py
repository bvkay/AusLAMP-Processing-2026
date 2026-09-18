r"""Write the workbooks. Each is a Python list of ("md", text) and ("code", source) held in its own module.

    python generator/make_workbooks.py            all
    python generator/make_workbooks.py 01         some, by prefix

Then execute them, which is what puts the outputs in:

    python generator/run_workbooks.py 01

A notebook is never edited in place; the generator is edited and re-run.

One module per notebook holds its parameter, rule and setup strings and its cell list: wb01_survey,
wb02_records, wb03_site, wb04_site and wb05_final. This module imports the five lists, names the notebook
each writes, and writes them.

Prose rules the cells follow: declarative, present tense; every number with its unit and its band; every rule
with its value and the function that applies it; a markdown cell says what to look for in the cell below it;
every check states its failure criterion in bold before the cell runs and the cell prints a verdict.

@author: ben kay (ben@auscope.org.au)
"""
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent          # the generator modules
WORKBOOKS = HERE.parent / "workbooks"            # where the notebooks are written
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from wb01_survey import WB01          # noqa: E402
from wb02_records import WB02         # noqa: E402
from wb03_site import WB03            # noqa: E402
from wb04_site import WB04            # noqa: E402
from wb05_final import WB05           # noqa: E402

KERNEL = "auslamp-processing-2026"

NOTEBOOKS = {"01_survey.ipynb": WB01, "02_records.ipynb": WB02, "03_site.ipynb": WB03,
             "04_site.ipynb": WB04, "05_final.ipynb": WB05}


def nb(cells):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": KERNEL, "display_name": "Python (%s)" % KERNEL, "language": "python"}
    n.metadata["language_info"] = {"name": "python"}
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    return n


def main(argv):
    names = sorted(NOTEBOOKS)
    if argv:
        names = [n for n in names if any(n.startswith(a) for a in argv)]
    for name in names:
        path = WORKBOOKS / name
        nbf.write(nb(NOTEBOOKS[name]), path)
        print("wrote %s (%d cells)" % (path, len(NOTEBOOKS[name])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
