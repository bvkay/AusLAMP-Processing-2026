"""Execute the workbooks and report every error cell and every verdict they printed.

    python generator/run_workbooks.py 01                              by prefix, in place
    python generator/run_workbooks.py                                 all, in order
    python generator/run_workbooks.py 01 --survey queensland_phase2   a copy, on another survey
    python generator/run_workbooks.py 02 --survey victoria --set SITES='"largest"' --set MAX_SITES=13
    python generator/run_workbooks.py 03 --timeout 21600              a workbook that processes

Two ways to set a parameter, and each is right for whoever is using it.

A student edits the parameter cell of the notebook in `workbooks/` and runs it here, in place, which is what
the README describes: the notebook is the student's working copy and its parameter cell is the record of
what was run. A maintainer keeping the shipped notebooks as the repository executed them uses --survey
instead, which copies the workbook to workbooks/examples/<survey>/<name>.ipynb with its outputs cleared and
its SURVEY assignment rewritten, and executes that copy. The workbook in `workbooks/` is then not touched,
so the survey its parameter cell names stays the one the repository ships executed.

--set NAME=VALUE rewrites one more module-level assignment in the copy. It applies only with --survey, for
the same reason: a parameter of the shipped workbook is not rewritten behind its own cell's back. A name the
workbook does not assign raises.

The right-hand side is read as Python where it parses as a Python literal and as a string where it does not,
so all of these set the same value:

    --set SITES='"largest"'      a POSIX shell keeps the inner quotes and it arrives as a Python string
    --set SITES=largest          PowerShell strips them, so a bare word arrives; a bare word is a string
    --set MAX_SITES=13           a literal int; a list such as ["Q32","Q33"] is a literal too

Whatever arrives, what is written into the copy is the repr of what was read, and the runner prints each
assignment it wrote. A bare word meant as a NAME and not as a string is the one case this cannot serve.

--timeout is the per-cell limit in seconds, 7200 by default. A workbook that processes needs more: the run
cell of workbook 03 is one cell holding every site's pass, and nbconvert stops a cell at the limit and
reports a timeout, which is indistinguishable from a fault.

What the runner reports is whether a notebook EXECUTED, not whether it passed. It counts a non-zero
nbconvert exit, a cell carrying an error output and a code cell with no execution count, and the exit code
is a statement about execution alone. The verdicts are the workbooks' own, so the line beside each notebook
counts them -- N PASS, M FAIL, K UNJUDGED -- and names the failing ones. A workbook that runs to the end
carrying a FAIL verdict is a survey with something in it rather than a broken run, and that survey's
`checks.retained_failures` is where such a verdict is named.

Run it with the environment's own interpreter:

    C:/Users/joint/anaconda3/envs/auslamp-processing-2026/python.exe workbooks/run_workbooks.py 01

The module calls `python -m nbconvert`, not `python -m jupyter nbconvert`. The `jupyter` dispatcher resolves
jupyter-nbconvert.EXE from PATH, which on a machine with a base Anaconda install is a different nbconvert
(7.16.6) from the environment's (7.17.1). `python -m nbconvert` stays inside the interpreter it was launched
with.

@author: ben kay (ben@auscope.org.au)
"""
import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent.parent / "workbooks"   # the notebooks, one folder up
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
    # a function replacement, not a string: a value carrying a backslash -- a Windows path -- would
    # otherwise be read as a regex escape and written back mangled
    def _repl(assignment):
        return lambda m: "%s%s" % (assignment, m.group(1))
    pats = [("SURVEY", SURVEY_LINE, _repl('SURVEY = "%s"' % survey))]
    for name, value in overrides:
        hits[name] = 0
        pats.append((name, re.compile(r'^%s\s*=\s*[^#\n]*(.*)$' % re.escape(name), re.M),
                     _repl("%s = %s" % (name, value))))
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


def verdicts(nb_path):
    """Every VERDICT line the executed notebook printed, in order.

    The workbooks state their own criteria and print their own verdicts; this reads them back off the
    executed file so that the runner's one line per notebook is not silent about what the notebook said.
    """
    n = nbf.read(nb_path, as_version=4)
    out = []
    for c in n.cells:
        for o in c.get("outputs", []) or []:
            if o.get("output_type") != "stream":
                continue
            text = o.get("text") or ""
            if isinstance(text, list):
                text = "".join(text)
            out += [ln.strip() for ln in text.splitlines() if ln.strip().startswith("VERDICT:")]
    return out


def verdict_counts(lines):
    """(n_pass, n_fail, n_unjudged) over VERDICT lines, plus the failing ones cut to one line each."""
    n_pass = sum(1 for ln in lines if ln.startswith("VERDICT: PASS"))
    n_fail = sum(1 for ln in lines if ln.startswith("VERDICT: FAIL"))
    n_unj = sum(1 for ln in lines if ln.startswith("VERDICT: UNJUDGED"))
    bad = [ln[:200] for ln in lines if not ln.startswith("VERDICT: PASS")]
    return n_pass, n_fail, n_unj, bad


def parameter_value(text):
    """The source to write for a --set right-hand side: a Python literal where it is one, else a string.

    PowerShell strips the inner quotes of --set SITES='"largest"', so the value arrives as the bare word
    `largest`, which written straight into the copy is a NameError at execution rather than a refusal at the
    command line. A bare word is read as the string it plainly is, and the runner prints what it wrote.
    """
    s = str(text).strip()
    try:
        return repr(ast.literal_eval(s))
    except (ValueError, SyntaxError):
        return repr(s)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", nargs="*", help="workbook name prefixes, e.g. 01")
    ap.add_argument("--survey", default="", help="run a copy under examples/<survey>/ on that survey")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="NAME=VALUE",
                    help="rewrite one more parameter in the copy; needs --survey")
    ap.add_argument("--timeout", type=int, default=7200, help="per-cell limit in seconds")
    a = ap.parse_args(argv)
    overrides = []
    for s in a.sets:
        name, _, value = s.partition("=")
        overrides.append((name.strip(), parameter_value(value)))
    if overrides and not a.survey:
        print("--set needs --survey: a parameter of the shipped workbook is not rewritten in place")
        return 1
    for name, src in overrides:
        print("--set          %s = %s" % (name, src))

    names = sorted(p.name for p in HERE.glob("*.ipynb"))
    if a.prefix:
        names = [n for n in names if any(n.startswith(x) for x in a.prefix)]
    if not names:
        print("no workbook matched %s in %s" % (a.prefix, HERE))
        return 1
    bad = 0
    for name in names:
        path = for_survey(HERE / name, a.survey, overrides) if a.survey else HERE / name
        rc, err = execute(path, timeout=a.timeout)
        errs = errors(path)
        n_pass, n_fail, n_unj, failing = verdict_counts(verdicts(path))
        # "executed" is a statement about the execution and never about the verdicts: a workbook that runs
        # to the end carrying a FAIL is the survey saying something, and the counts beside it say what
        print("%-12s %s  (nbconvert exit %d, %d error cell(s); %d PASS, %d FAIL, %d UNJUDGED)"
              % ("executed" if rc == 0 and not errs else "NOT EXECUTED", path.relative_to(HERE), rc,
                 len(errs), n_pass, n_fail, n_unj))
        for i, ename, tb in errs:
            print("   cell %d: %s\n      %s" % (i, ename, tb.replace("\n", "\n      ")[-900:]))
        if rc != 0 and not errs:
            print("   nbconvert stderr:", err[-800:])
        for ln in failing:
            print("   %s" % ln)
        bad += rc != 0 or bool(errs)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
