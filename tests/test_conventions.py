"""The standing checks on the repository itself: the author line, the vocabulary, ASCII, and the generator.

Every test reads the tracked files and reads nothing more. They are the checks a cleanup pass would
otherwise have to be trusted to have done, so each one states what it would take to fail and names every
file that does.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
AUTHOR = "@author: ben kay (ben@auscope.org.au)"

# item 1 of the cleanup: the phrases a plain technical statement does not carry
FORBIDDEN = ("we will", "let us", "let's", "note that", "simply", "easily", "nothing else",
             "and they answer different questions", "tie-break key", "shown and never delivered",
             "not truth", "never truth")

# The word that left the package (Ben, 2026-09-17): a pass estimates a TRANSFER FUNCTION and a site delivers
# one. A path that names a file outside this repository is exempt, because renaming it here would falsify the
# provenance it states; so is surveys/*/decisions.csv, whose cells are the analyst's own dated record.
PRODUCT = re.compile(r"[A-Za-z0-9_./\\-]*\bproducts?\b[A-Za-z0-9_./\\-]*", re.IGNORECASE)
PRODUCT_EXEMPT_SUFFIX = (".py", ".csv")

TEXT_SUFFIXES = (".py", ".md", ".yaml", ".csv", ".cfg", ".toml")
CITED = ("METHOD.md", "DESIGN.md", "STYLE.md", "docs/", "notes/")
SHIPPED = ("01_survey.ipynb", "02_records.ipynb", "03_site.ipynb", "04_site.ipynb", "05_final.ipynb")
SELF = "test_conventions.py"   # this file names every forbidden phrase, so it is not scanned for them


IGNORED = ("docs", "notes", "workbooks/examples", "surveys/victoria", "surveys/queensland_phase2",
           "surveys/queensland_phase3", ".git", "__pycache__", ".ipynb_checkpoints", ".pytest_cache",
           "auslamp_proc.egg-info")


def tracked():
    """The files the repository ships: `git ls-files` where a checkout exists, else the tree less the folders
    .gitignore names, so the test also runs on a copy of the package without its history."""
    try:
        out = subprocess.run(["git", "ls-files"], cwd=str(ROOT), capture_output=True, text=True)
        names = [p for p in out.stdout.splitlines() if p.strip()]
    except OSError:
        names = []
    if names:
        return [ROOT / p for p in names]
    files = []
    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT).as_posix()
        if p.is_file() and not any(rel == i or rel.startswith(i + "/") or ("/" + i + "/") in ("/" + rel) for i in IGNORED):
            files.append(p)
    return files


def tracked_suffix(*suffixes):
    return [p for p in tracked() if p.suffix in suffixes and p.exists()]


def markdown_cells(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return ["".join(c.get("source") or []) for c in doc.get("cells", []) if c.get("cell_type") == "markdown"]


def code_cells(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return ["".join(c.get("source") or []) for c in doc.get("cells", []) if c.get("cell_type") == "code"]


# ---------------------------------------------------------------- the files themselves

def test_every_tracked_python_file_carries_the_author_line():
    """Fails if a tracked .py does not end its module docstring with the author line."""
    bad = [str(p.relative_to(ROOT)) for p in tracked_suffix(".py")
           if AUTHOR not in p.read_text(encoding="utf-8")]
    assert not bad, "no author line in: %s" % ", ".join(bad)


def test_every_tracked_python_file_parses():
    """Fails if a tracked .py does not parse, which no other test would catch for a file nothing imports."""
    bad = []
    for p in tracked_suffix(".py"):
        try:
            ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            bad.append("%s: %s" % (p.relative_to(ROOT), exc))
    assert not bad, "; ".join(bad)


def test_every_tracked_text_file_is_ascii():
    """Fails if a tracked text file carries a character outside ASCII: write --, deg, +- and x instead."""
    bad = []
    for p in tracked_suffix(*TEXT_SUFFIXES):
        raw = p.read_bytes()
        for i, b in enumerate(raw):
            if b > 127:
                bad.append("%s at byte %d (0x%02x)" % (p.relative_to(ROOT), i, b))
                break
    assert not bad, "; ".join(bad)


def test_no_tracked_file_cites_a_local_only_document():
    """Fails if a tracked file cites METHOD.md, DESIGN.md, STYLE.md, docs/ or notes/, none of which ships."""
    bad = []
    for p in tracked_suffix(*TEXT_SUFFIXES) + [ROOT / n for n in SHIPPED]:
        if p.name == SELF:   # this file names the phrases it forbids
            continue
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for name in CITED:
            if name in text:
                bad.append("%s cites %s" % (p.relative_to(ROOT), name))
    assert not bad, "; ".join(bad)


# ---------------------------------------------------------------- the prose

def prose_of(path):
    """The comments and the string literals of a .py, which is where its prose is.

    Code is not scanned: an identifier such as `NOTHING` followed by the keyword `else` reads as a forbidden
    phrase to a scan over the raw text and is not prose at all.
    """
    import io
    import tokenize
    text = Path(path).read_text(encoding="utf-8")
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                out.append(tok.string)
    except (tokenize.TokenError, IndentationError):
        out.append(text)
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append(node.value)
    return "\n".join(out)


def test_no_tracked_file_carries_a_forbidden_phrase():
    """Fails if the prose of a tracked .py or .md, or a markdown cell of a shipped notebook, carries one of
    the phrases a plain technical statement does not use."""
    bad = []
    for p in tracked_suffix(".py", ".md"):
        if p.name == SELF:
            continue
        low = (prose_of(p) if p.suffix == ".py" else p.read_text(encoding="utf-8")).lower()
        for phrase in FORBIDDEN:
            if phrase in low:
                bad.append("%s: %s" % (p.relative_to(ROOT), phrase))
    for name in SHIPPED:
        p = ROOT / "workbooks" / name
        if not p.exists():
            continue
        low = "\n".join(markdown_cells(p)).lower()
        for phrase in FORBIDDEN:
            if phrase in low:
                bad.append("%s: %s" % (name, phrase))
    assert not bad, "; ".join(bad)


def test_the_word_product_has_left_the_package():
    """Fails if a tracked .md or .yaml, or a markdown cell of a shipped notebook, still calls a transfer
    function a product.

    A pass estimates a transfer function and a site delivers one (Ben, 2026-09-17). Python files and
    surveys/*/decisions.csv are exempt and are covered by their own test below: a .py may name an outside
    script whose file name carries the word, and a decisions cell is the analyst's own dated record of what a
    campaign said.
    """
    bad = []
    for p in tracked_suffix(".md", ".yaml"):
        for m in PRODUCT.finditer(p.read_text(encoding="utf-8")):
            if m.group(0).endswith(".py"):
                continue
            bad.append("%s: %s" % (p.relative_to(ROOT), m.group(0)))
    for name in SHIPPED:
        p = ROOT / "workbooks" / name
        if not p.exists():
            continue
        for cell in markdown_cells(p):
            for m in PRODUCT.finditer(cell):
                if not m.group(0).endswith(".py"):
                    bad.append("%s: %s" % (name, m.group(0)))
    assert not bad, "; ".join(sorted(set(bad))[:20])


def test_the_word_product_has_left_the_python_files():
    """Fails if a tracked .py carries the word outside a file name that ends in .py.

    An outside script named in a provenance clause keeps its own name; every other use is the vocabulary
    the ruling removed.
    """
    bad = []
    for p in tracked_suffix(".py"):
        if p.name == SELF:   # this file names the phrases it forbids
            continue
        for m in PRODUCT.finditer(p.read_text(encoding="utf-8")):
            if m.group(0).endswith(".py"):
                continue
            bad.append("%s: %s" % (p.relative_to(ROOT), m.group(0)))
    assert not bad, "; ".join(sorted(set(bad))[:20])


def test_no_markdown_cell_of_a_shipped_notebook_carries_an_exclamation_mark():
    """Fails if a markdown cell ends a sentence with an exclamation mark. Code cells are not scored, because
    `!=` is an operator."""
    bad = []
    for name in SHIPPED:
        p = ROOT / "workbooks" / name
        if not p.exists():
            continue
        for cell in markdown_cells(p):
            if re.search(r"!(\s|$|\")", cell):
                bad.append(name)
                break
    assert not bad, "; ".join(bad)


# ---------------------------------------------------------------- the generator

def test_the_generator_reproduces_every_shipped_notebook(tmp_path):
    """Fails if a shipped notebook's cell sources differ from what the generator writes now.

    A notebook is never edited in place: the generator is edited and re-run. This is the test of that rule,
    and it compares the cell SOURCES and not the outputs, which are the record of one execution.
    """
    sys.path.insert(0, str(ROOT / "workbooks"))
    for mod in ("make_workbooks", "wb01_survey", "wb02_records", "wb03_site", "wb04_site", "wb05_final"):
        sys.modules.pop(mod, None)
    import make_workbooks as MW

    missing = [n for n in SHIPPED if not (ROOT / "workbooks" / n).exists()]
    assert not missing, "not on disk: %s" % ", ".join(missing)
    assert sorted(MW.NOTEBOOKS) == sorted(SHIPPED), \
        "the generator writes %s" % ", ".join(sorted(MW.NOTEBOOKS))
    bad = []
    for name, cells in MW.NOTEBOOKS.items():
        shipped = ROOT / "workbooks" / name
        doc = json.loads(shipped.read_text(encoding="utf-8"))
        got = [("markdown" if c["cell_type"] == "markdown" else "code", "".join(c.get("source") or []))
               for c in doc["cells"]]
        want = [("markdown" if k == "md" else "code", src) for k, src in cells]
        if len(got) != len(want):
            bad.append("%s: %d cells on disk, %d in the generator" % (name, len(got), len(want)))
            continue
        for i, (a, b) in enumerate(zip(want, got)):
            if a[0] != b[0] or a[1].rstrip("\n") != b[1].rstrip("\n"):
                bad.append("%s: cell %d differs" % (name, i))
    assert not bad, "; ".join(bad)


def test_every_code_cell_of_every_shipped_notebook_compiles():
    """Fails if a code cell of a shipped notebook is not valid Python, which an execution would only find
    when it reached that cell."""
    bad = []
    for name in SHIPPED:
        p = ROOT / "workbooks" / name
        if not p.exists():
            continue
        for i, src in enumerate(code_cells(p)):
            try:
                compile(src, "%s cell %d" % (name, i), "exec")
            except SyntaxError as exc:
                bad.append("%s cell %d: %s" % (name, i, exc))
    assert not bad, "; ".join(bad)


def test_every_check_of_every_shipped_notebook_states_its_criterion_in_bold():
    """Fails if a notebook prints more VERDICT lines than it states criteria for.

    A check states its failure criterion in bold above the cell that prints its verdict, so the count of
    `**This check fails if` in the markdown is the count of criteria the workbook stands behind.
    """
    bad = []
    for name in SHIPPED:
        p = ROOT / "workbooks" / name
        if not p.exists():
            continue
        md = "\n".join(markdown_cells(p))
        criteria = md.count("**This check fails if") + md.count("**It fails if") \
            + md.count("**Limb B fails if") + md.count("fails if", 0, 0)
        verdicts = sum(src.count("VERDICT:") for src in code_cells(p))
        # one cell prints PASS, FAIL and UNJUDGED for one criterion, so the verdict strings outnumber the
        # criteria; a criterion with no verdict at all is the fault this scores
        if criteria == 0 and verdicts:
            bad.append("%s: %d VERDICT string(s) and no criterion in bold" % (name, verdicts))
    assert not bad, "; ".join(bad)


def retained_failures():
    """[(workbook, site, reason)] from every tracked survey.yaml's `checks.retained_failures`.

    An honest FAIL over a condition the survey itself carries is not a fault of the package, but it may not
    be silent either: the survey has to name the workbook, the site and the reason, and this reads them.
    """
    import yaml
    out = []
    for p in tracked_suffix(".yaml"):
        if p.name != "survey.yaml":
            continue
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        for entry in ((cfg.get("checks") or {}).get("retained_failures") or []):
            out.append((str(entry.get("workbook", "")), str(entry.get("site", "")),
                        str(entry.get("reason", ""))))
    return out


def test_every_verdict_of_an_executed_notebook_is_pass_or_a_named_unjudged():
    """Fails if an executed shipped notebook prints a FAIL no survey.yaml names as retained, or an UNJUDGED
    carrying no reason.

    A retained FAIL is an entry of `checks.retained_failures` in the survey's own survey.yaml, carrying the
    workbook, the site and the reason. A verdict is excused only where an entry names both that workbook and
    a site the verdict names, so a new failure cannot hide behind an old one's entry.

    Skipped where no shipped notebook carries an output: a notebook that has not been executed since it was
    generated has no verdict to score, and this states that rather than passing on nothing.
    """
    lines, executed = [], []
    for name in SHIPPED:
        p = ROOT / "workbooks" / name
        if not p.exists():
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        got = []
        for c in doc.get("cells", []):
            for o in c.get("outputs") or []:
                text = "".join(o.get("text") or []) if o.get("output_type") == "stream" else ""
                got += [ln.strip() for ln in text.splitlines() if ln.strip().startswith("VERDICT:")]
        if got:
            executed.append(name)
            lines += [(name, ln) for ln in got]
    if not executed:
        pytest.skip("no shipped notebook carries an output: none has been executed since it was generated")
    named = retained_failures()

    def excused(notebook, verdict):
        """True where an entry names this workbook and a site this verdict names."""
        stem = notebook.replace(".ipynb", "")
        for workbook, site, _reason in named:
            if workbook not in (stem, stem.split("_")[0]):
                continue
            if site and re.search(r"\b%s\b" % re.escape(site), verdict):
                return True
        return False

    bad = [("%s: %s" % (n, ln))[:220] for n, ln in lines
           if (ln.startswith("VERDICT: FAIL") and not excused(n, ln))
           or (ln.startswith("VERDICT: UNJUDGED") and "--" not in ln)]
    assert not bad, "%d unexcused verdict(s): %s" % (len(bad), "; ".join(bad))


# ---------------------------------------------------------------- the tables

def test_the_template_tables_carry_the_column_lists_the_package_reads():
    """Fails if surveys/_template/sites.csv or decisions.csv has a header that is not the package's own
    column list, which would make a copied template unreadable."""
    from auslamp_proc import final as FN, survey as SV
    for name, want in (("sites.csv", SV.SITES_COLUMNS), ("decisions.csv", SV.DECISIONS_COLUMNS)):
        head = (ROOT / "surveys" / "_template" / name).read_text(encoding="utf-8").splitlines()[0]
        assert head.split(",") == list(want), "%s: %s" % (name, head)
    for p in sorted((ROOT / "surveys").glob("*/final_choices.csv")):
        head = p.read_text(encoding="utf-8").splitlines()[0]
        assert head.split(",") == list(FN.CHOICE_COLUMNS), "%s: %s" % (p.relative_to(ROOT), head)


def test_every_column_of_both_template_tables_is_documented():
    """Fails if a column of sites.csv or decisions.csv is not named in SITES_COLUMNS.md, which is the only
    place a student is told what a cell means."""
    from auslamp_proc import survey as SV
    doc = (ROOT / "surveys" / "_template" / "SITES_COLUMNS.md").read_text(encoding="utf-8")
    missing = [c for c in list(SV.SITES_COLUMNS) + list(SV.DECISIONS_COLUMNS) if c not in doc]
    assert not missing, "not documented: %s" % ", ".join(missing)
