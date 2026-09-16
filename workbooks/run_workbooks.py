"""Execute the workbooks in place and report every error cell.

    python workbooks/run_workbooks.py 01                              by prefix, in place
    python workbooks/run_workbooks.py                                 all, in order
    python workbooks/run_workbooks.py 01 --survey queensland_phase2   a copy, on another survey
    python workbooks/run_workbooks.py 02 --survey victoria --set SITES='"largest"' --set MAX_SITES=13

With --survey the workbook is copied to workbooks/examples/<survey>/<name>.ipynb with its outputs cleared and
its SURVEY assignment rewritten, and that copy is executed. The workbook in workbooks/ is not touched, so the
survey its parameter cell names stays the one the repository ships executed.

--set NAME=VALUE rewrites one more module-level assignment in the copy, the right-hand side taken as the
Python source it is given. It applies only with --survey, because rewriting a parameter of the workbook the
repository ships would leave that workbook executed under parameters its own cell does not carry. A name the
workbook does not assign raises.

This check fails if nbconvert exits non-zero, if any executed cell carries an error output, or if a code cell
has no execution count.

Run it with the environment's own interpreter:

    C:/Users/joint/anaconda3/envs/auslamp-processing-2026/python.exe workbooks/run_workbooks.py 01

The module calls `python -m nbconvert`, not `python -m jupyter nbconvert`. The `jupyter` dispatcher resolves
jupyter-nbconvert.EXE from PATH, which on a machine with a base Anaconda install is a different nbconvert
(7.16.6) from the environment's (7.17.1). `python -m nbconvert` stays inside the interpreter it was launched
with.

@author: ben kay (ben@auscope.org.au)
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
PY = sys.executable
KERNEL = "auslamp-processing-2026"

SURVEY_LINE = re.compile(r'^SURVEY\s*=\s*"[^"]*"(.*)$', re.M)


def execute(nb_path, timeout=7200):
    cmd = [PY, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
           "--ExecutePreprocessor.kernel_name=%s" % KERNEL,
           "--ExecutePreprocessor.timeout=%d" % timeout,
           "--ExecutePreprocessor.allow_errors=True", str(nb_path)]
    r = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True)
    return r.returncode, (r.stderr or "")[-1500:]


def for_survey(nb_path, survey, overrides=()):
    """A copy of the workbook under examples/<survey>/, outputs cleared and SURVEY rewritten.

    The assignment is rewritten rather than the whole parameter cell, so every other parameter and the inline
    comment that names the alternatives survive. A workbook whose parameter cell holds no SURVEY assignment
    raises, because executing it would silently run the survey the copy was made from. `overrides` is a list
    of (name, source) pairs rewritten the same way.
    """
    n = nbf.read(nb_path, as_version=4)
    hits = {"SURVEY": 0}
    pats = [("SURVEY", SURVEY_LINE, 'SURVEY = "%s"\\1' % survey)]
    for name, value in overrides:
        hits[name] = 0
        pats.append((name, re.compile(r'^%s\s*=\s*[^#\n]*(.*)$' % re.escape(name), re.M),
                     "%s = %s\\1" % (name, value)))
    for c in n.cells:
        if c.cell_type != "code":
            continue
        c["outputs"] = []
        c["execution_count"] = None
        for name, pat, repl in pats:
            new, k = pat.subn(repl, c["source"])
            if k:
                c["source"] = new
                hits[name] += k
    for name, k in hits.items():
        if not k:
            raise ValueError("%s holds no %s assignment to rewrite" % (nb_path.name, name))
    out = HERE / "examples" / survey / nb_path.name
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(n, out)
    return out


def errors(nb_path):
    n = nbf.read(nb_path, as_version=4)
    out = []
    for i, c in enumerate(n.cells):
        if c.cell_type != "code":
            continue
        if c.get("execution_count") is None:
            out.append((i, "NOT EXECUTED", ""))
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                out.append((i, o.get("ename"), " ".join(o.get("traceback", []))[-1200:]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", nargs="*", help="workbook name prefixes, e.g. 01")
    ap.add_argument("--survey", default="", help="run a copy under examples/<survey>/ on that survey")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="NAME=VALUE",
                    help="rewrite one more parameter in the copy; needs --survey")
    a = ap.parse_args(argv)
    overrides = [tuple(s.split("=", 1)) for s in a.sets]
    if overrides and not a.survey:
        print("--set needs --survey: a parameter of the shipped workbook is not rewritten in place")
        return 1

    names = sorted(p.name for p in HERE.glob("*.ipynb"))
    if a.prefix:
        names = [n for n in names if any(n.startswith(x) for x in a.prefix)]
    if not names:
        print("no workbook matched %s in %s" % (a.prefix, HERE))
        return 1
    bad = 0
    for name in names:
        path = for_survey(HERE / name, a.survey, overrides) if a.survey else HERE / name
        rc, err = execute(path)
        errs = errors(path)
        status = "PASS" if rc == 0 and not errs else "FAIL"
        print("%s  %s  (nbconvert exit %d, %d error cell(s))"
              % (status, path.relative_to(HERE), rc, len(errs)))
        for i, ename, tb in errs:
            print("   cell %d: %s\n      %s" % (i, ename, tb.replace("\n", "\n      ")[-900:]))
        if rc != 0 and not errs:
            print("   nbconvert stderr:", err[-800:])
        bad += status == "FAIL"
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
