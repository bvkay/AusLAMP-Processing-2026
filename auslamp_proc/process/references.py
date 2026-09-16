"""The five reference kinds and the store they are written to.

    single station        the site's own H and E; no reference is written
    remote site           one other site's H, whole, on the target's grid
    fleet stack           a coherence-weighted mean of several sites' H, aligned and demeaned
    observatory           an INTERMAGNET one-second record in its own mean-field frame
    stack + observatory   the stack with the observatory as one more member

What each is for. The remote site cancels the noise that is uncorrelated between the two sites, and costs the
variance of one more record. The stack averages down each member's own noise and needs the members aligned,
because delays that differ by 2-3 s partly cancel each other in the sum. The observatory is far and quiet and
reaches the long periods, and is never shifted: a delay on a single reference cancels exactly in Z.

The pool. A member is drawn from the clean pool (transients.clean_row) restricted to the target's overlap
group in <work_root>/survey/deployment_groups.csv and to the sites that have a cache. The remote for a site
is decisions.csv `remote_site` where that cell names a site, and otherwise the five-branch rule.

The five-branch remote-site rule (ported from wamt_remotes.partner :941-1029, itself the Queensland campaign's
rule), with the branch and the reason recorded in every product:

    1  clean, coh >= COH_MIN, overlap >= 90 % of the target's record -> the nearest of them
    2  clean, coh >= COH_MIN                                          -> the longest overlap
    3  clean, coh >= COH_RELAX                                        -> the most coherent
    4  coh >= COH_MIN, not clean                                      -> the fewest events
    5  none of the above                                              -> the nearest by km

A candidate is scored only where its usable overlap reaches overlap_floor = min(MIN_OVERLAP_DAYS,
0.75 x the days the target can use); the coherence is the event-free 20-200 s chunk median (coherence.pair_coh).

The stack (ported from wamt_remotes.stack_weights / accumulate / _finish :1188-1356). The weight is the
member's median coherence with the FLEET at 100-1000 s and never its coherence with the target
(Ben's rule, 2026-09-06): whether a reference is a good measurement of the regional field is a
question about the reference and the field. Members below STACK_CUTOFF = 0.5 are excluded, the best
STACK_MAX = 8 are kept, and a stack with fewer than STACK_MIN = 2 members is refused -- a one-member stack is
a remote site renamed (STACK_MIN_MEMBERS, qld_p1_remotes_v2.py:115). Each member is demeaned over its own
finite samples and a sample where a member is NaN does not add to that member's weight there, so the mean is
over whoever is sound and the weights renormalise per sample. Where no member is sound the stack is zero and
the mask is False: a zero reference contributes nothing to either the cross- or the auto-spectrum, so those
windows drop out of the estimate instead of biasing it.

The store is <work_root>/references/<rate>hz/<kind>_<site>.npz (t0, Hx, Hy, mask, coverage) beside
<kind>_<site>.json (members, weights, lags, roles, refusals, frame, built_at), with pool.json,
fleet_weights.json, fleet_pairs.json and candidates_<site>.json for the tables the workbook prints. A 10 Hz
store re-reads the 1 Hz specification -- the same pool, the same remote, the same members, weights and lags --
on the 10 Hz grid after the gap-edge screen.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import geo, observatory
from ..raw import cache
from . import align, coherence as COH, frame as FR, transients as TR

COH_MIN = 0.5
COH_RELAX = 0.3
MIN_OVERLAP_DAYS = 20.0
STACK_CUTOFF = 0.5
STACK_MAX = 8
STACK_MIN = 2
GAP_EDGE_S = 1.0                      # widened either side of every gap above 1 Hz
H = ("Hx", "Hy")
KINDS_WITH_STORE = ("remote", "stack", "obs", "stack_obs")


class NoMembers(RuntimeError):
    """A stack was asked for with nothing to stack."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _iso(t) -> str:
    return datetime.fromtimestamp(float(t), timezone.utc).isoformat(timespec="seconds")


def gap_edge_screen(x, fs, edge_s=GAP_EDGE_S) -> np.ndarray:
    """Widen every gap by `edge_s` seconds at each end, above 1 Hz.

    The samples either side of a gap carry the recorder's settling, which the decimation to 1 Hz averages
    away and the 10 Hz grid does not. The screen widens what is already missing and so cannot break a
    continuous stretch into pieces.

    A per-sample amplitude screen was tried here first -- NaN wherever the absolute first difference
    exceeds 10x the record's median -- and is not used: measured on Q60N at 10 Hz on 2026-09-16 it NaNs
    865,078 of 52.5 M Hx samples scattered through a gapless record, which leaves 248,970 pieces of a
    median 1 s, and the 3,600 s run floor then throws away 64.9 per cent of the record. A screen whose cost
    is paid by the run floor is not a screen. Outliers are handled where they belong, inside the estimator:
    the robust regression down-weights them and the window screen drops a window whose rms first difference
    exceeds 10x the median.
    """
    v = np.asarray(x, float).copy()
    bad = ~np.isfinite(v)
    if not bad.any() or bad.all() or float(fs) <= 1.0:
        return v
    k = int(round(float(edge_s) * float(fs)))
    if k < 1:
        return v
    d = np.diff(np.concatenate(([0], bad.view(np.int8), [0])))
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        v[max(0, a - k):min(len(v), b + k)] = np.nan
    return v


class Store:
    """One survey's references at one rate, with the caches the scoring needs.

    `spec_rate` is the rate the decisions are measured at. A 10 Hz store is built with spec_rate 1: the pool,
    the remote, the members, the weights and the lags come from the 1 Hz store and only the arrays are read
    on the 10 Hz grid.
    """

    def __init__(self, survey, sites=None, rate=1, spec_rate=None, rot_cache=None, coh_min=COH_MIN,
                 coh_relax=COH_RELAX, min_overlap_days=MIN_OVERLAP_DAYS, cutoff=STACK_CUTOFF,
                 n_max=STACK_MAX, n_min=STACK_MIN):
        self.sv = survey
        self.cfg = survey.cfg
        self.work = Path(survey.cfg["work_root"])
        self.rate = int(rate)
        self.fs = float(rate)
        self.spec_rate = int(spec_rate if spec_rate is not None else rate)
        self.sites = list(sites) if sites is not None else list(survey.sites.site)
        self.dir = self.work / "references" / ("%dhz" % self.rate)
        self.coh_min = float(coh_min)
        self.coh_relax = float(coh_relax)
        self.min_overlap_days = float(min_overlap_days)
        self.cutoff = float(cutoff)
        self.n_max = int(n_max)
        self.n_min = int(n_min)
        self._rot: dict = {}
        self._order: list = []
        self._max = int(rot_cache if rot_cache is not None else (24 if self.rate == 1 else 2))
        self._events: dict = {}
        self._scores: dict = {}
        self._pool = None
        self._fleet = None
        self._groups = None

    # -------------------------------------------------------------- the record

    def has_cache(self, site: str) -> bool:
        return (self.work / ("cache_%dhz" % self.rate) / ("%s.npz" % site)).exists()

    def rotated(self, site: str):
        """(t0, {'Hx','Hy'} in the site's own mean-field frame, angle, n) at this store's rate.

        Signs from decisions.csv are applied first, then the rotation; rot_drop days come back NaN and are
        excluded from the mean. Above 1 Hz the gap-edge screen runs before the rotation.
        """
        if site in self._rot:
            self._order.remove(site)
            self._order.append(site)
            return self._rot[site]
        path = self.work / ("cache_%dhz" % self.rate) / ("%s.npz" % site)
        z = np.load(path, allow_pickle=False)
        t0 = int(z["t0"][0])
        arr = {c: np.asarray(z[c], float) for c in H}
        z.close()
        if self.rate > 1:
            arr = {c: gap_edge_screen(arr[c], self.fs) for c in H}
        dec = self._decision(site)
        arr, _applied, _und = FR.apply_signs(arr, dec)
        regimes = FR.parse_regimes(dec.get("rot_regimes") if dec is not None else "")
        drop = FR.parse_regimes(dec.get("rot_drop") if dec is not None else "")
        arr, ang = FR.rotate_to_mean_field(arr, regimes=regimes, drop=drop, fs=self.fs)
        got = (t0, {c: np.asarray(arr[c], np.float32) for c in H}, ang, len(arr["Hx"]))
        self._rot[site] = got
        self._order.append(site)
        while len(self._order) > self._max:
            self._rot.pop(self._order.pop(0), None)
        return got

    def _decision(self, site: str):
        try:
            return self.sv.decision(site)
        except KeyError:
            return None

    def window(self, site: str) -> tuple:
        """(t0, n) of the site's cache at this rate, read from the npz header alone."""
        path = self.work / ("cache_%dhz" % self.rate) / ("%s.npz" % site)
        z = np.load(path, allow_pickle=False)
        t0, n = int(z["t0"][0]), int(len(z["Hx"]))
        z.close()
        return t0, n

    def usable_days(self, site: str) -> float:
        _t0, h, _a, _n = self.rotated(site)
        return float(np.isfinite(h["Hx"]).sum()) / (86400.0 * self.fs)

    def events(self, site: str) -> list:
        if site not in self._events:
            self._events[site] = TR.site_events(site, self.sites, self.work, self.cfg)
        return self._events[site]

    def position(self, site: str) -> tuple:
        r = self.sv.site(site)
        return float(r.lat), float(r.lon)

    def distance_km(self, a: str, b: str) -> float:
        return geo.distance_km(self.position(a), self.position(b))

    # ---------------------------------------------------------------- the pool

    def groups(self) -> dict:
        """{site: group} from <work_root>/survey/deployment_groups.csv, written by workbook 01."""
        if self._groups is None:
            p = self.work / "survey" / "deployment_groups.csv"
            if not p.exists():
                raise FileNotFoundError("%s has not been written; run workbook 01 first" % p)
            g = pd.read_csv(p)
            self._groups = {m: r.group for r in g.itertuples() for m in str(r.members).split()}
        return self._groups

    def clean_pool(self, force=False) -> tuple:
        """(the pool, one row per site of the clean test with its reason).

        A site is in the pool where its tail scan is clean and it has a cache at this store's rate. The rows
        name every exclusion.
        """
        if self._pool is not None and not force:
            return self._pool
        rows = []
        for s in self.sites:
            r = TR.clean_row(s, self.sites, self.work, self.cfg)
            if not self.has_cache(s):
                r["clean"] = False
                r["reason"] = "; ".join(x for x in (r["reason"], "no %d Hz cache" % self.rate) if x)
            rows.append(r)
        table = pd.DataFrame(rows)
        pool = [r["site"] for r in rows if r["clean"]]
        self._pool = (pool, table)
        return self._pool

    def members_for(self, target: str) -> tuple:
        """(the candidate members of a target, one line saying where the set came from).

        decisions.csv `stack_members` where the cell names sites, otherwise the clean pool restricted to the
        target's own overlap group.
        """
        dec = self._decision(target)
        cell = str(dec.get("stack_members", "") if dec is not None else "").strip()
        if cell and cell.lower() not in ("decide", "nan", "none"):
            named = [s for s in cell.replace(",", " ").split() if s != target]
            return named, "decisions.csv stack_members"
        pool, _t = self.clean_pool()
        grp = self.groups()
        mine = grp.get(target)
        out = [s for s in pool if s != target and grp.get(s) == mine and self.has_cache(s)]
        return out, "the clean pool inside overlap group %s" % (mine or "-")

    # ------------------------------------------------------------- the scoring

    def overlap(self, a: str, b: str):
        ta, na = self.window(a)
        tb, nb = self.window(b)
        lo, hi = max(ta, tb), min(ta + na / self.fs, tb + nb / self.fs)
        return (lo, hi) if hi > lo else None

    def overlap_floor(self, target: str, own_days: float) -> float:
        """min(MIN_OVERLAP_DAYS, 0.75 x the days the target can use).

        The days the target can use, not its cache length: a cache is as long as the logger ran, not as long
        as the magnetics are sound, and asking a candidate to cover 9.8 days of a 2.6-day usable record
        admits nobody (Ben, 2026-09-11).
        """
        try:
            u = float(self.usable_days(target))
        except Exception:
            u = float("nan")
        if not np.isfinite(u) or u <= 0:
            u = own_days
        return float(min(self.min_overlap_days, 0.75 * min(u, own_days)))

    def _on_grid(self, site: str, t0, n, drop_events=True, lag_s=None) -> dict:
        """A member's rotated H on the target grid; NaN where it has nothing to say.

        `drop_events` NaNs the member's own transient chunks, which is what a stack member needs. A remote
        site is passed through whole instead, because the single-reference pass cuts local and remote
        together on the union of the two masks at processing time and NaNing here would hide that cut.
        """
        td, hd, _a, nd = self.rotated(site)
        out = {c: np.full(int(n), np.nan) for c in H}
        fs = self.fs
        lo, hi = max(t0, td), min(t0 + n / fs, td + nd / fs)
        if hi <= lo:
            return out
        i0 = int(round((lo - t0) * fs))
        j0 = int(round((lo - td) * fs))
        L = max(0, min(int(round((hi - lo) * fs)), int(n) - i0, nd - j0))
        if L == 0:
            return out
        for c in H:
            out[c][i0:i0 + L] = hd[c][j0:j0 + L]
        if lag_s is not None and np.isfinite(lag_s) and lag_s != 0.0:
            out = {c: align.shift(out[c], float(lag_s), fs) for c in H}
        if drop_events:
            keep = TR.keep_mask(td, nd, fs, self.events(site))
            for c in H:
                out[c][i0:i0 + L][~keep[j0:j0 + L]] = np.nan
        return out

    def pair_coherence(self, a: str, b: str, band=COH.PAIR_BAND, nperseg=COH.PAIR_NPERSEG,
                       min_segments=COH.PAIR_MIN_SEGMENTS) -> dict:
        """coherence.pair_coh over the two sites' overlap, both event lists masked."""
        ta, ha, _aa, na = self.rotated(a)
        tb, hb, _ab, nb = self.rotated(b)
        ov = self.overlap(a, b)
        if ov is None:
            return dict(coh=float("nan"), chunks=0, chunk_days=0.0, clean_days=0.0, mask_frac=float("nan"),
                        overlap_s=0)
        lo, hi = ov
        fs = self.fs
        ia, ib = int(round((lo - ta) * fs)), int(round((lo - tb) * fs))
        L = int(round((hi - lo) * fs))
        keep = TR.keep_mask(lo, L, fs, self.events(a) + self.events(b))
        cd = float(((self.cfg.get("floors") or {}).get("member_chunk_days")) or COH.CHUNK_DAYS)
        mc = 1 if cd < COH.CHUNK_DAYS else COH.MIN_CHUNKS
        ms = max(4, int(round(min_segments * min(1.0, cd / COH.CHUNK_DAYS))))
        r = COH.pair_coh({c: ha[c][ia:ia + L] for c in H}, {c: hb[c][ib:ib + L] for c in H}, fs, keep,
                         band=band, nperseg=COH.nperseg_for(nperseg, fs), min_segments=ms,
                         chunk_days=cd, min_chunks=mc)
        r["overlap_s"] = int(hi - lo)
        return r

    def score_candidates(self, target: str, pool=None, verbose=False, force=False) -> pd.DataFrame:
        """One row per candidate: km, overlap, event fraction, baseline, clean, and the 20-200 s coherence.

        A candidate below the overlap floor is reported with the reason and not scored for coherence. The
        table is kept per target, because the remote-site rule and the stack weights both ask for it.
        """
        if pool is None and target in self._scores and not force:
            return self._scores[target]
        cached = pool is None
        pool = list(pool) if pool is not None else self.members_for(target)[0]
        ta, na = self.window(target)
        own_days = max(na / (86400.0 * self.fs), 1e-9)
        floor = self.overlap_floor(target, own_days)
        clean_tab = self.clean_pool()[1].set_index("site")
        rows = []
        for d in pool:
            if d == target or not self.has_cache(d):
                continue
            ov = self.overlap(target, d)
            row = dict(name=d, km=round(self.distance_km(target, d), 1),
                       own_days=round(own_days, 1), min_overlap_days=round(floor, 1))
            if ov is None:
                row.update(overlap_days=0.0, overlap_frac=0.0, ok=False, why="no overlap",
                           coh=None, chunks=0, clean=False, event_frac=None, baseline=None)
                rows.append(row)
                continue
            ovd = (ov[1] - ov[0]) / 86400.0
            cr = clean_tab.loc[d] if d in clean_tab.index else None
            row.update(overlap_days=round(ovd, 1), overlap_frac=round(ovd / own_days, 3),
                       event_frac=(None if cr is None or cr.event_frac is None else round(float(cr.event_frac), 4)),
                       baseline=(None if cr is None or cr.baseline is None else round(float(cr.baseline), 1)),
                       clean=bool(cr.clean) if cr is not None else False,
                       ok=bool(ovd >= floor))
            if not row["ok"]:
                row.update(why="overlap %.1f d < floor %.1f d" % (ovd, floor), coh=None, chunks=0)
                rows.append(row)
                continue
            r = self.pair_coherence(target, d)
            row.update(why="", coh=(None if not np.isfinite(r["coh"]) else round(r["coh"], 3)),
                       chunks=int(r["chunks"]), chunk_days=r["chunk_days"],
                       clean_days=r["clean_days"], mask_frac=round(float(r["mask_frac"]), 4))
            rows.append(row)
            if verbose:
                print("      %-8s %6.0f km  ov %5.1f d  coh %s" % (d, row["km"], row["overlap_days"],
                                                                   row["coh"]), flush=True)
        cols = ["name", "km", "overlap_days", "overlap_frac", "event_frac", "baseline", "clean", "coh",
                "chunks", "ok", "why", "own_days", "min_overlap_days"]
        t = pd.DataFrame(rows)
        t = (t.reindex(columns=[c for c in cols if c in t.columns] +
                       [c for c in t.columns if c not in cols]) if len(t) else t)
        if cached:
            self._scores[target] = t
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / ("candidates_%s.json" % target)).write_text(
                json.dumps(dict(target=target, band_s=[20, 200], built_at=_now(),
                                rows=t.to_dict("records")), indent=1, default=str), encoding="utf-8")
        return t

    # ------------------------------------------------------------ the remote site

    def remote_site(self, target: str, scores=None, coh_min=None, coh_relax=None) -> dict:
        """The chosen remote, its branch and the reason, or decisions.csv's named site where there is one."""
        coh_min = self.coh_min if coh_min is None else float(coh_min)
        coh_relax = self.coh_relax if coh_relax is None else float(coh_relax)
        dec = self._decision(target)
        named = str(dec.get("remote_site", "") if dec is not None else "").strip()
        scores = self.score_candidates(target) if scores is None else scores
        cands = scores[scores.ok].to_dict("records") if len(scores) else []
        cands.sort(key=lambda c: c["km"])
        rule = self._rule_choice(target, cands, scores, coh_min, coh_relax)
        if named and named.lower() not in ("decide", "nan", "none"):
            row = next((c for c in cands if c["name"] == named), {})
            src = str(dec.get("remote_source", "") if dec is not None else "") or "decisions.csv remote_site"
            return dict(target=target, name=named, branch=0, coh=row.get("coh"),
                        source="decisions.csv", source_detail=src,
                        reason="decisions.csv names %s as the remote (%s); the rule would have chosen %s "
                               "on %s" % (named, src[:120], rule["name"], rule["reason"].split(":")[0]),
                        rule_name=rule["name"], rule_branch=rule["branch"], rule_coh=rule["coh"],
                        candidates=cands)
        rule.update(source="the five-branch rule", source_detail="", rule_name=rule["name"],
                    rule_branch=rule["branch"], rule_coh=rule["coh"], candidates=cands)
        return rule

    def _rule_choice(self, target, cands, scores, coh_min, coh_relax) -> dict:
        n_considered = len(scores)
        n_short = int((~scores.ok).sum()) if len(scores) else 0
        own_days = float(scores.own_days.iloc[0]) if len(scores) else float("nan")
        floor = float(scores.min_overlap_days.iloc[0]) if len(scores) else float("nan")

        def nearest():
            pool = self.members_for(target)[0]
            near = sorted((d for d in pool if d != target and self.has_cache(d)),
                          key=lambda d: self.distance_km(target, d))
            return near[0] if near else None

        return branch_rule(target, cands, n_considered, n_short, own_days, floor,
                           coh_min=coh_min, coh_relax=coh_relax, nearest=nearest)

    # ----------------------------------------------------------- the fleet weights

    def fleet_weights(self, force=False, verbose=False) -> tuple:
        """({member: median 100-1000 s coherence with the rest of the pool}, the per-pair table).

        The quantity the stack weights on. Computed once per store and kept as fleet_weights.json with the
        pairs in fleet_pairs.json, so the workbook can recompute one site's weights from the table it prints.
        """
        if self._fleet is not None and not force:
            return self._fleet
        fw = self.dir / "fleet_weights.json"
        fp = self.dir / "fleet_pairs.json"
        if fw.exists() and fp.exists() and not force:
            self._fleet = (json.loads(fw.read_text(encoding="utf-8")),
                           json.loads(fp.read_text(encoding="utf-8")))
            return self._fleet
        pool, _t = self.clean_pool()
        pairs = {}
        for i, a in enumerate(pool):
            for b in pool[i + 1:]:
                r = self.pair_coherence(a, b, band=COH.WEIGHT_BAND, nperseg=COH.WEIGHT_NPERSEG,
                                        min_segments=COH.WEIGHT_MIN_SEGMENTS)
                pairs["%s:%s" % (a, b)] = dict(
                    coh=(None if not np.isfinite(r["coh"]) else round(float(r["coh"]), 4)),
                    chunks=int(r["chunks"]))
                if verbose:
                    print("   fleet pair %s-%s coh %s over %d chunks"
                          % (a, b, pairs["%s:%s" % (a, b)]["coh"], r["chunks"]), flush=True)
        weights = COH.fleet_weight_table(pairs, pool)
        self.dir.mkdir(parents=True, exist_ok=True)
        fw.write_text(json.dumps(weights, indent=1), encoding="utf-8")
        fp.write_text(json.dumps(pairs, indent=1), encoding="utf-8")
        self._fleet = (weights, pairs)
        return self._fleet

    # ---------------------------------------------------------------- the stack

    def stack_members(self, target: str, cutoff=None, n_max=None, align_members=True,
                      verbose=False) -> tuple:
        """({member: weight}, {member: lag_s}, {member: refusal}, {member: alignment note}).

        The weight is the member's fleet coherence at 100-1000 s. A member is refused where it has no fleet
        coherence, where that coherence is below the cutoff, or where it falls outside the best n_max.
        """
        cutoff = self.cutoff if cutoff is None else float(cutoff)
        n_max = self.n_max if n_max is None else int(n_max)
        cand, _why = self.members_for(target)
        weights, _pairs = self.fleet_weights()
        # the default pool, so the candidate table kept for the remote-site rule answers here as well
        scores = self.score_candidates(target)
        ok = set(scores.name[scores.ok]) if len(scores) else set()
        scored, refused, lags, notes = [], {}, {}, {}
        for d in cand:
            if d not in ok:
                row = scores[scores.name == d]
                refused[d] = str(row.why.iloc[0]) if len(row) else "not scored"
                continue
            w = weights.get(d)
            if w is None or not np.isfinite(w) or w <= 0:
                refused[d] = "no fleet coherence at 100-1000 s (fewer than 3 scored pairs)"
                continue
            if w < cutoff:
                refused[d] = "fleet coherence %.3f < %s" % (w, cutoff)
                continue
            scored.append((float(w), d))
        scored.sort(reverse=True)
        for w, d in scored[n_max:]:
            refused[d] = "fleet coherence %.3f, outside the best %d" % (w, n_max)
        kept = {d: round(w, 4) for w, d in scored[:n_max]}
        if align_members and kept:
            t0, n = self.window(target)
            tt, th, _a, _n = self.rotated(target)
            keep = TR.keep_mask(t0, n, self.fs, self.events(target))
            fb = (self.cfg.get("floors") or {}).get("align_fallback_s")
            for d in list(kept):
                g = self._on_grid(d, t0, n, drop_events=True)
                l, why = align.shift_for({c: np.asarray(th[c], float) for c in H}, g, self.fs, keep,
                                         fallback_s=fb)
                lags[d] = round(float(l), 3)
                if why:
                    notes[d] = "kept at lag 0: %s" % why
                del g
        else:
            lags = {d: 0.0 for d in kept}
        if verbose and kept:
            print("   %s stack members: %s" % (target, ", ".join("%s %.3f (%+.2f s)"
                                                                 % (d, w, lags.get(d, 0.0))
                                                                 for d, w in kept.items())), flush=True)
        return kept, lags, refused, notes

    def accumulate(self, target: str, weights: dict, lags: dict) -> tuple:
        """(t0, n, num, den, count) of the weighted member sum, one read per member."""
        t0, n = self.window(target)
        num = {c: np.zeros(n) for c in H}
        den = {c: np.zeros(n) for c in H}
        cnt = np.zeros(n, np.int16)
        for d, w in weights.items():
            g = self._on_grid(d, t0, n, drop_events=True, lag_s=(lags or {}).get(d))
            for c in H:
                x = np.asarray(g[c], float)
                m = np.isfinite(x)
                if m.any():
                    x = x.copy()
                    x[m] -= x[m].mean()
                num[c][m] += w * x[m]
                den[c][m] += w
            cnt += np.isfinite(g["Hx"])
            del g
        return t0, n, num, den, cnt

    @staticmethod
    def finish(num, den) -> tuple:
        """(H, coverage per channel, mask): zero and mask False where no member is sound."""
        h = {c: np.where(den[c] > 0, num[c] / np.maximum(den[c], 1e-12), 0.0) for c in H}
        cov = {c: round(float((den[c] > 0).mean()), 4) for c in H}
        mask = (den["Hx"] > 0) & (den["Hy"] > 0)
        return h, cov, mask

    # ---------------------------------------------------------- the observatory

    def observatory_member(self, code: str, t0, n) -> tuple:
        """(H in the observatory's own mean-field frame, its real-data mask, the angle used).

        The archive holds geographic X, Y, Z; the pair is turned into the mean-field frame of the window,
        like every member. The observatory is never shifted.

        The archive is one-second data whatever this store's rate is. On a 10 Hz grid each sample is held
        for ten, which is a zero-order hold and puts images above 0.5 Hz: the observatory kinds are built at
        1 Hz, where the archive is native, and a 10 Hz run takes the single station, the remote site and the
        fleet stack.
        """
        archive = (self.cfg.get("observatory") or {}).get("archive", "")
        n1 = int(round(n / self.fs))                       # the archive is 1 Hz whatever this store's rate is
        d = observatory.load(code, int(t0), n1, archive)
        arr = {c: np.asarray(d[c], float) for c in H}
        turned, ang = FR.rotate_to_mean_field(arr, fs=1.0)
        mask = np.asarray(d["mask"], bool)
        if self.rate != 1:
            rep = int(round(self.fs))
            turned = {c: np.repeat(turned[c], rep)[:int(n)] for c in H}
            mask = np.repeat(mask, rep)[:int(n)]
            for c in H:
                if len(turned[c]) < int(n):
                    turned[c] = np.concatenate([turned[c], np.full(int(n) - len(turned[c]), np.nan)])
            if len(mask) < int(n):
                mask = np.concatenate([mask, np.zeros(int(n) - len(mask), bool)])
        return turned, mask, (ang if isinstance(ang, float) else float("nan"))

    def site_observatory_coh(self, site: str, code: str) -> dict:
        """One site's event-free coherence with the observatory at 100-1000 s, on the same rule as a pair.

        A reading: it says how much of the site's long-period field the observatory shares from several
        hundred kilometres away, which is what the observatory kind can buy and what it cannot.
        """
        t0, n = self.window(site)
        o, omask, _a = self.observatory_member(code, t0, n)
        _ts, hs, _as, _ns = self.rotated(site)
        keep = TR.keep_mask(t0, n, self.fs, self.events(site)) & omask
        r = COH.pair_coh({c: np.asarray(hs[c], float) for c in H}, o, self.fs, keep,
                         band=COH.WEIGHT_BAND, nperseg=COH.nperseg_for(COH.WEIGHT_NPERSEG, self.fs),
                         min_segments=COH.WEIGHT_MIN_SEGMENTS)
        del o
        return r

    def observatory_weight(self, code: str, force=False) -> tuple:
        """(the observatory's fleet coherence at 100-1000 s, the pairs it was measured over).

        Measured on the same rule as a site member and against the same pool, so the observatory enters the
        stack at a weight on the members' own scale rather than at its agreement with any one target.
        """
        p = self.dir / "observatory_weight.json"
        if p.exists() and not force:
            return tuple(json.loads(p.read_text(encoding="utf-8")).values())[:2]
        pool, _t = self.clean_pool()
        vals, pairs = [], {}
        for s in pool:
            t0, n = self.window(s)
            o, omask, _a = self.observatory_member(code, t0, n)
            _ts, hs, _as, _ns = self.rotated(s)
            keep = TR.keep_mask(t0, n, self.fs, self.events(s)) & omask
            r = COH.pair_coh({c: np.asarray(hs[c], float) for c in H}, o, self.fs, keep,
                             band=COH.WEIGHT_BAND, nperseg=COH.nperseg_for(COH.WEIGHT_NPERSEG, self.fs),
                             min_segments=COH.WEIGHT_MIN_SEGMENTS)
            pairs[s] = None if not np.isfinite(r["coh"]) else round(float(r["coh"]), 4)
            if pairs[s] is not None:
                vals.append(pairs[s])
            del o
        w = round(float(np.median(vals)), 4) if vals else None
        self.dir.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(weight=w, pairs=pairs, code=code, band_s=[100, 1000],
                                     built_at=_now()), indent=1), encoding="utf-8")
        return w, pairs

    # ----------------------------------------------------------------- the store

    def path(self, kind: str, site: str) -> Path:
        return self.dir / ("%s_%s.npz" % (kind, site))

    def _write(self, kind, site, t0, h, mask, info) -> tuple:
        """Write the npz and its sidecar, and return (path, the sidecar as written)."""
        p = self.path(kind, site)
        p.parent.mkdir(parents=True, exist_ok=True)
        cov = float(np.asarray(mask, bool).mean())
        np.savez(p, t0=np.array([int(t0)], np.int64), fs=np.array([float(self.fs)]),
                 Hx=np.asarray(h["Hx"], np.float32), Hy=np.asarray(h["Hy"], np.float32),
                 mask=np.asarray(mask, bool), coverage=np.array([cov]))
        info = dict(info)
        info.update(kind=kind, target=site, rate_hz=self.fs, npz=p.name, t0=int(t0),
                    n=int(len(h["Hx"])), coverage=round(cov, 4), built_at=_now())
        p.with_suffix(".json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
        return p, info

    def build_remote(self, target: str, choice=None) -> tuple:
        """The chosen remote's rotated H on the target grid, finite everywhere, zero outside its record.

        The remote's own transient intervals are left in and written to the sidecar: the single-reference
        pass cuts local and remote together on the union of the two masks, and NaNing here would hide that
        cut from the code that builds the keep mask.
        """
        choice = self.remote_site(target) if choice is None else choice
        name = choice.get("name")
        if not name:
            raise NoMembers("%s: no remote site could be chosen -- %s" % (target, choice.get("reason", "")))
        t0, n = self.window(target)
        h = self._on_grid(name, t0, n, drop_events=False)
        mask = np.isfinite(h["Hx"]) & np.isfinite(h["Hy"])
        for c in H:
            h[c] = np.where(mask, h[c], 0.0)
        ev = [(float(a), float(b)) for a, b in self.events(name) if b > t0 and a < t0 + n / self.fs]
        info = dict(remote=name, branch=choice.get("branch"), coh=choice.get("coh"),
                    source=choice.get("source"), source_detail=choice.get("source_detail"),
                    reason=choice.get("reason"), band_s=[20, 200],
                    rule_name=choice.get("rule_name"), rule_branch=choice.get("rule_branch"),
                    rule_coh=choice.get("rule_coh"),
                    members=[dict(name=name, weight=1.0, lag_s=0.0, role="remote")],
                    weights={name: 1.0}, lags={name: 0.0}, refusals={},
                    remote_events=[[_iso(a), _iso(b)] for a, b in ev],
                    remote_event_frac_in_window=round(
                        float(sum(min(b, t0 + n / self.fs) - max(a, t0) for a, b in ev)) /
                        max(n / self.fs, 1.0), 5),
                    n_candidates=len(choice.get("candidates", [])),
                    frame="mean-horizontal-field per member (process.frame.rotate_to_mean_field)",
                    note="finite everywhere; zero and mask False outside the remote's own record")
        return self._write("remote", target, t0, h, mask, info)

    def build_stack(self, target: str, members=None, cutoff=None, n_max=None, n_min=None, acc=None,
                    verbose=False) -> tuple:
        cutoff = self.cutoff if cutoff is None else float(cutoff)
        n_max = self.n_max if n_max is None else int(n_max)
        n_min = self.n_min if n_min is None else int(n_min)
        weights, lags, refused, notes = (members if members is not None
                                         else self.stack_members(target, cutoff, n_max, verbose=verbose))
        if len(weights) < n_min:
            raise NoMembers("%s: the fleet stack has %d member(s) and the floor is %d -- a one-member stack "
                            "is a remote site renamed. %s"
                            % (target, len(weights), n_min,
                               " | ".join("%s: %s" % kv for kv in sorted(refused.items())) or
                               "(no candidate scored)"))
        t0, n, num, den, cnt = self.accumulate(target, weights, lags) if acc is None else acc
        h, cov, mask = self.finish(num, den)
        info = dict(weights=weights, lags=lags, refusals=refused, alignment_notes=notes,
                    members=_member_rows(weights, lags, refused, notes),
                    weight_rule=COH.WEIGHT_RULE, weight_band_s=[100, 1000], align_band_s=[5, 20],
                    cutoff=cutoff, n_max=n_max, n_min=n_min,
                    coverage_by_channel=cov, member_count=_count_profile(cnt, len(weights)),
                    frame="mean-horizontal-field per member (process.frame.rotate_to_mean_field)",
                    note="members demeaned over their own finite samples and time-aligned to the target "
                         "before averaging; zero and mask False where no member is sound")
        return self._write("stack", target, t0, h, mask, info)

    def build_obs(self, target: str, code=None) -> tuple:
        code = code or (self.cfg.get("observatory") or {}).get("code", "")
        t0, n = self.window(target)
        o, omask, ang = self.observatory_member(code, t0, n)
        h = {c: np.where(omask & np.isfinite(o[c]), o[c], 0.0) for c in H}
        w, _pairs = self.observatory_weight(code)
        info = dict(observatory=code, angle_deg=ang, fleet_coh=w,
                    distance_km=round(geo.distance_km(self.position(target), code), 1),
                    members=[dict(name=code, weight=w, lag_s=0.0, role="observatory")],
                    weights={code: w}, lags={code: 0.0}, refusals={},
                    alignment_notes={code: "never shifted: a delay on a single reference cancels in Z"},
                    frame="mean-horizontal-field of the observatory record (geographic X, Y turned)",
                    note="zero and mask False outside the archive's real samples")
        return self._write("obs", target, t0, h, omask, info)

    def build_stack_obs(self, target: str, code=None, members=None, acc=None, cutoff=None, n_max=None,
                        n_min=None) -> tuple:
        """The stack with the observatory as one more member at its own fleet weight, never shifted."""
        cutoff = self.cutoff if cutoff is None else float(cutoff)
        n_max = self.n_max if n_max is None else int(n_max)
        n_min = self.n_min if n_min is None else int(n_min)
        code = code or (self.cfg.get("observatory") or {}).get("code", "")
        weights, lags, refused, notes = (members if members is not None
                                         else self.stack_members(target, cutoff, n_max))
        if len(weights) < n_min:
            raise NoMembers("%s: stack + observatory has %d site member(s) and the floor is %d. %s"
                            % (target, len(weights), n_min,
                               " | ".join("%s: %s" % kv for kv in sorted(refused.items())) or
                               "(no candidate scored)"))
        t0, n, num0, den0, cnt = self.accumulate(target, weights, lags) if acc is None else acc
        num = {c: num0[c].copy() for c in H}
        den = {c: den0[c].copy() for c in H}
        w_obs, _pairs = self.observatory_weight(code)
        o, omask, ang = self.observatory_member(code, t0, n)
        if w_obs is not None and np.isfinite(w_obs) and w_obs > 0:
            for c in H:
                x = np.asarray(o[c], float)
                m = np.isfinite(x) & omask
                if m.any():
                    x = x.copy()
                    x[m] -= x[m].mean()
                num[c][m] += w_obs * x[m]
                den[c][m] += w_obs
        h, cov, mask = self.finish(num, den)
        allw = dict(weights)
        allw[code] = w_obs
        info = dict(weights=allw, lags=dict(lags, **{code: 0.0}), refusals=refused,
                    alignment_notes=dict(notes, **{code: "never shifted"}),
                    members=_member_rows(weights, lags, refused, notes, obs=code, w_obs=w_obs),
                    observatory=code, observatory_weight=w_obs, observatory_angle_deg=ang,
                    observatory_coverage=round(float(omask.mean()), 4),
                    weight_rule=COH.WEIGHT_RULE, weight_band_s=[100, 1000], align_band_s=[5, 20],
                    cutoff=cutoff, n_max=n_max, n_min=n_min,
                    coverage_by_channel=cov, member_count=_count_profile(cnt, len(weights)),
                    frame="mean-horizontal-field per member (process.frame.rotate_to_mean_field)",
                    note="the observatory enters as one more member at its own fleet weight and is never "
                         "shifted; zero and mask False where no member is sound")
        return self._write("stack_obs", target, t0, h, mask, info)

    def build_site(self, target: str, kinds=KINDS_WITH_STORE, force=False, verbose=False) -> dict:
        """Every asked-for reference for one target, with one member-scoring pass and one member read pass."""
        out = {}
        if "remote" in kinds:
            try:
                if force or not self.path("remote", target).exists():
                    _p, info = self.build_remote(target)
                else:
                    info = json.loads(self.path("remote", target).with_suffix(".json")
                                      .read_text(encoding="utf-8"))
                out["remote"] = info
            except Exception as exc:
                out["remote"] = dict(error="%s: %s" % (type(exc).__name__, str(exc)[:400]))
        want = [k for k in ("stack", "stack_obs") if k in kinds]
        if want:
            need = force or any(not self.path(k, target).exists() for k in want)
            if need:
                members = self.stack_members(target, verbose=verbose)
                acc = None
                if len(members[0]) >= self.n_min:
                    acc = self.accumulate(target, members[0], members[1])
                for k in want:
                    try:
                        if force or not self.path(k, target).exists():
                            builder = self.build_stack if k == "stack" else self.build_stack_obs
                            _p, info = builder(target, members=members, acc=acc)
                        else:
                            info = json.loads(self.path(k, target).with_suffix(".json")
                                              .read_text(encoding="utf-8"))
                        out[k] = info
                    except Exception as exc:
                        out[k] = dict(error="%s: %s" % (type(exc).__name__, str(exc)[:400]))
                del acc
            else:
                for k in want:
                    out[k] = json.loads(self.path(k, target).with_suffix(".json")
                                        .read_text(encoding="utf-8"))
        if "obs" in kinds:
            try:
                if force or not self.path("obs", target).exists():
                    _p, info = self.build_obs(target)
                else:
                    info = json.loads(self.path("obs", target).with_suffix(".json")
                                      .read_text(encoding="utf-8"))
                out["obs"] = info
            except Exception as exc:
                out["obs"] = dict(error="%s: %s" % (type(exc).__name__, str(exc)[:400]))
        return out


def branch_rule(target, cands, n_considered, n_short, own_days, floor, coh_min=COH_MIN,
                coh_relax=COH_RELAX, nearest=None) -> dict:
    """The five-branch remote-site rule on a scored candidate list, with its branch and its reason.

    `cands` are the candidates that cleared the overlap floor, sorted by km. `nearest` is called only where
    nothing cleared the floor, and answers with the nearest site in the pool.
    """
    def desc(c):
        return ("%.0f km, overlap %.1f d (%.0f %% of %.1f d), events %s, coh %s over %d chunks"
                % (c["km"], c["overlap_days"], 100 * c["overlap_frac"], own_days, c["event_frac"],
                   c["coh"], c.get("chunks", 0)))

    if not cands:
        return dict(target=target, name=(nearest() if nearest else None), branch=5, coh=None,
                    reason="branch 5 (nearest by km): no candidate scored -- %d of %d candidates fall "
                           "below the %.1f d overlap floor for a %.1f d record"
                           % (n_short, n_considered, floor, own_days))
    clean = [c for c in cands if c["clean"]]
    coh_ok = {c["name"] for c in cands if c["coh"] is not None and c["coh"] >= coh_min}
    good = [c for c in clean if c["name"] in coh_ok]
    best_clean = max((c["coh"] for c in clean if c["coh"] is not None), default=None)
    full = [c for c in good if c["overlap_frac"] >= 0.9]
    if full:
        b = min(full, key=lambda c: c["km"])
        return dict(target=target, name=b["name"], branch=1, coh=b["coh"],
                    reason="branch 1 (clean, coh >= %s, overlap >= 90 %%): nearest of %d: %s"
                           % (coh_min, len(full), desc(b)))
    if good:
        b = max(good, key=lambda c: c["overlap_days"])
        return dict(target=target, name=b["name"], branch=2, coh=b["coh"],
                    reason="branch 2 (clean, coh >= %s, longest overlap): branch 1 empty, the best overlap "
                           "among the coherent clean candidates is %.0f %%; %s"
                           % (coh_min, 100 * max(c["overlap_frac"] for c in good), desc(b)))
    relaxed = [c for c in clean if c["coh"] is not None and c["coh"] >= coh_relax]
    if relaxed:
        b = max(relaxed, key=lambda c: c["coh"])
        return dict(target=target, name=b["name"], branch=3, coh=b["coh"],
                    reason="branch 3 (clean, gate relaxed %s -> %s): none of the %d clean candidates "
                           "reaches %s (best %s); %s"
                           % (coh_min, coh_relax, len(clean), coh_min, best_clean, desc(b)))
    why12 = ("none of the %d candidates is clean" % len(cands) if not clean else
             "coherence unmeasurable for every clean candidate" if best_clean is None else
             "no clean candidate reaches even %s (best %s)" % (coh_relax, best_clean))
    coherent = [c for c in cands if c["name"] in coh_ok]
    if coherent:
        b = min(coherent, key=lambda c: (c["event_frac"] if c["event_frac"] is not None else 1.0))
        return dict(target=target, name=b["name"], branch=4, coh=b["coh"],
                    reason="branch 4 (fewest events among the %d at coh >= %s): branches 1-3 empty because "
                           "%s; %s" % (len(coherent), coh_min, why12, desc(b)))
    b = cands[0]
    return dict(target=target, name=b["name"], branch=5, coh=b["coh"],
                reason="branch 5 (nearest by km): branches 1-4 empty because %s and no candidate reaches "
                       "coherence %s, and decisions.csv names no remote; %s" % (why12, coh_min, desc(b)))


def _member_rows(weights, lags, refusals, notes, obs=None, w_obs=None) -> list:
    rows = [dict(name=d, weight=w, lag_s=(lags or {}).get(d), role="member",
                 note=(notes or {}).get(d)) for d, w in weights.items()]
    if obs is not None:
        rows.append(dict(name=obs, weight=w_obs, lag_s=0.0, role="observatory",
                         note="never shifted"))
    rows += [dict(name=d, weight=None, lag_s=None, role="refused", note=r)
             for d, r in sorted((refusals or {}).items())]
    return rows


def _count_profile(cnt, n_members) -> dict:
    """How ragged the member coverage is, which is what costs a stack.

    Each member is demeaned over its own record, so when the member set changes mid-record the composite's
    baseline steps by the difference of the members' local offsets. Under remote reference that costs
    efficiency and not bias, but a target with a ragged profile is one where the single remote site may beat
    the stack, so the profile is written out.
    """
    c = np.asarray(cnt)
    return dict(n_members=int(n_members), min=int(c.min()), median=int(np.median(c)),
                frac_zero=round(float((c == 0).mean()), 4),
                frac_lt_half=round(float((c < max(1, n_members // 2)).mean()), 4),
                frac_full=round(float((c == n_members).mean()), 4))


def build_store(survey, sites=None, rate=1, kinds=KINDS_WITH_STORE, force=False, verbose=True,
                spec_store=None, store=None) -> dict:
    """Build every reference of a survey at one rate. Returns {site: {kind: sidecar or error}}.

    `spec_store` is a 1 Hz store whose decisions a higher-rate store re-uses: the pool, the remote, the
    members, the weights and the lags come from it and only the arrays are read on this store's grid.
    `store` is an existing Store to build into, so a caller that set its own thresholds keeps them.
    """
    st = store if store is not None else Store(survey, sites, rate)
    st.dir.mkdir(parents=True, exist_ok=True)
    pool, table = st.clean_pool()
    (st.dir / "pool.json").write_text(json.dumps(dict(
        pool=pool, rate_hz=st.fs, thresholds=TR.params(st.cfg), built_at=_now(),
        rows=table.to_dict("records")), indent=1, default=str), encoding="utf-8")
    out = {}
    for s in (sites if sites is not None else st.sites):
        if verbose:
            print("   %s" % s, flush=True)
        if spec_store is not None:
            out[s] = _build_from_spec(st, spec_store, s, kinds, force)
        else:
            out[s] = st.build_site(s, kinds=kinds, force=force, verbose=verbose)
    return out


def _build_from_spec(st: Store, spec: Store, target: str, kinds, force) -> dict:
    """One site's references at st's rate, with every decision taken from the 1 Hz store's sidecars."""
    out = {}
    for kind in kinds:
        p = st.path(kind, target)
        if p.exists() and not force:
            out[kind] = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
            continue
        src = spec.path(kind, target).with_suffix(".json")
        if not src.exists():
            out[kind] = dict(error="the %g Hz store holds no %s reference for %s" % (spec.fs, kind, target))
            continue
        spec_info = json.loads(src.read_text(encoding="utf-8"))
        try:
            if kind == "remote":
                choice = dict(name=spec_info["remote"], branch=spec_info.get("branch"),
                              coh=spec_info.get("coh"), source=spec_info.get("source"),
                              source_detail=spec_info.get("source_detail"),
                              reason=spec_info.get("reason"), rule_name=spec_info.get("rule_name"),
                              rule_branch=spec_info.get("rule_branch"),
                              rule_coh=spec_info.get("rule_coh"), candidates=[])
                _q, info = st.build_remote(target, choice=choice)
            elif kind == "obs":
                _q, info = st.build_obs(target)
            else:
                weights = {k: v for k, v in (spec_info.get("weights") or {}).items()
                           if v is not None and k != spec_info.get("observatory")}
                lags = {k: float(v or 0.0) for k, v in (spec_info.get("lags") or {}).items()
                        if k in weights}
                members = (weights, lags, spec_info.get("refusals") or {},
                           spec_info.get("alignment_notes") or {})
                builder = st.build_stack if kind == "stack" else st.build_stack_obs
                _q, info = builder(target, members=members)
            info["specification_from"] = "%g Hz store" % spec.fs
            p.with_suffix(".json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
            out[kind] = info
        except Exception as exc:
            out[kind] = dict(error="%s: %s" % (type(exc).__name__, str(exc)[:400]))
    return out


def load_reference(kind: str, site: str, rate, work_root) -> tuple:
    """(t0, {'Hx','Hy'}, mask, sidecar) for one reference from the store, or Nones for the single station."""
    if kind == "single":
        return None, None, None, dict(kind="single")
    p = Path(work_root) / "references" / ("%dhz" % int(rate)) / ("%s_%s.npz" % (kind, site))
    if not p.exists():
        raise FileNotFoundError("%s has not been built; run the reference store first" % p)
    z = np.load(p, allow_pickle=False)
    t0 = int(z["t0"][0])
    h = {c: np.asarray(z[c], float) for c in H}
    mask = np.asarray(z["mask"], bool)
    z.close()
    info = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    return t0, h, mask, info


def reference_station_id(kind: str, info: dict, survey_code="") -> str:
    """The station id a reference is written into the MTH5 under: REMOTE_<site>, STACK, OBS_<code>, STACK_OBS."""
    if kind == "remote":
        return "REMOTE_%s" % info.get("remote", "")
    if kind == "stack":
        return "STACK"
    if kind == "obs":
        return "OBS_%s" % (info.get("observatory") or survey_code or "")
    if kind == "stack_obs":
        return "STACK_OBS"
    return ""


def cache_sidecar(site, work_root) -> dict:
    return cache.sidecar(site, work_root)
