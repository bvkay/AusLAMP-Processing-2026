"""The three screens a fleet stack member passes: the edge screen, the Lanczos shift and the spike screen.

Every test states what would make it fail. The records are built in the test, so a failure names the code
and not the data. The values under test are align.LANCZOS_A = 16, references.EDGE_S = 120 s,
references.SCREEN_K = 30, references.SCREEN_FLOOR_NT = 3.0 nT and references.SCREEN_MAX_SPAN_S = 12 s.

@author: ben kay (ben@auscope.org.au)
"""
from __future__ import annotations

import numpy as np

from auslamp_proc.process import align, references as REF


# ------------------------------------------------------------- the Lanczos shift

def test_lanczos_shift_matches_an_analytic_sine():
    """Fails if the shifted sine departs from the analytic one by more than 0.4 per cent of the amplitude at
    a 2.3 s period, 0.1 per cent at 5 s or 0.05 per cent at 20 s.

    2.3 s is the shortest band the 1 Hz parameter set estimates, so it is the worst case the stack is asked
    for. The error falls with period because the interpolation is a low-pass one.
    """
    fs, n, d = 1.0, 20000, 0.37
    t = np.arange(n) / fs
    for period, bound in ((2.3, 4e-3), (5.0, 1e-3), (20.0, 5e-4)):
        x = np.sin(2 * np.pi * t / period)
        want = np.sin(2 * np.pi * (t + d) / period)[100:-100]
        got = align.shift(x, d, fs)[100:-100]
        rms = float(np.sqrt(np.nanmean((got - want) ** 2)))
        assert np.isfinite(got).all()
        assert rms < bound, "period %.1f s: rms error %.2e against a bound of %.0e" % (period, rms, bound)


def test_lanczos_shift_does_not_spread_a_spike():
    """Fails if one impulsive sample shifted 0.37 s moves anything beyond LANCZOS_A samples either side.

    This is the defect the Fourier phase ramp had: over a finite run it turned one sample into a sinc train
    decaying as 1/n over the whole run, and Q45 went from 78 to 4,384 steps above 2 nT per million samples
    under its own 0.172 s shift.
    """
    a = align.LANCZOS_A
    x = np.zeros(400)
    x[200] = 1.0
    y = align.shift(x, 0.37, 1.0)
    moved = np.flatnonzero(np.abs(np.nan_to_num(y)) > 1e-12)
    assert moved.min() >= 200 - a and moved.max() <= 200 + a
    assert len(moved) <= 2 * a


def test_a_nan_reaches_only_the_taps_around_it():
    """Fails if one NaN costs more than 2 x LANCZOS_A samples away from the record ends."""
    a = align.LANCZOS_A
    x = np.arange(400.0)
    x[200] = np.nan
    y = align.shift(x, 0.37, 1.0)
    bad = np.flatnonzero(~np.isfinite(y))
    inner = bad[(bad > 4 * a) & (bad < 400 - 4 * a)]
    assert len(inner) == 2 * a
    assert inner.min() == 200 - a and inner.max() == 200 + a - 1
    assert np.isfinite(y[200 - a - 1]) and np.isfinite(y[200 + a])


def test_the_shift_carries_an_integer_part():
    """Fails if a whole-sample lag does not move the record by exactly that many samples."""
    x = np.arange(100.0)
    y = align.shift(x, 3.0, 1.0)
    assert np.allclose(y[:97], x[3:])
    assert np.all(~np.isfinite(y[97:]))


# --------------------------------------------------------------- the edge screen

def test_edge_screen_cuts_both_ends_of_every_run():
    """Fails if the screen leaves the 120 s inside a record end or either side of a gap, or cuts more.

    120 s and not 30: measured over the two Queensland phases, a member's first minute after a break carries
    37x the record's own rate of steps above 2 nT and its second under 2x, so 30 s leaves most of it in.
    """
    assert REF.EDGE_S == 120.0
    x = np.arange(2000.0)
    x[1000] = np.nan
    y = REF.edge_screen(x, 1.0)
    finite = np.flatnonzero(np.isfinite(y))
    assert finite.min() == 120 and finite.max() == 1879
    assert 880 not in finite and 1120 not in finite
    assert 879 in finite and 1121 in finite
    assert len(finite) == 2000 - 1 - 4 * 120

    # the same 120 s is 1,200 samples at 10 Hz
    z = np.arange(20000.0)
    w = REF.edge_screen(z, 10.0)
    fin10 = np.flatnonzero(np.isfinite(w))
    assert fin10.min() == 1200 and fin10.max() == 18799


# -------------------------------------------------------------- the spike screen

def _four_members(n=20000, seed=4):
    """Four members carrying one common field and their own small noise, on one grid."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    field = 30.0 * np.sin(2 * np.pi * t / 900.0) + 8.0 * np.sin(2 * np.pi * t / 137.0)
    return {d: 25000.0 + field + 0.05 * rng.standard_normal(n) for d in ("A", "B", "C", "D")}


def _weighted_stack(members, weights):
    """The stack's own arithmetic: the weights renormalised per sample, NaN where no member is sound.

    No demeaning here, because the level match has already put every member on the fleet's level; a member
    demeaned over its own record is the rule that was replaced.
    """
    n = len(next(iter(members.values())))
    num, den = np.zeros(n), np.zeros(n)
    for d, w in weights.items():
        x = np.asarray(members[d], float)
        m = np.isfinite(x)
        num[m] += w * x[m]
        den[m] += w
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def test_screen_bridges_one_member_spike_and_leaves_a_storm_alone():
    """Fails if a 20 nT spike in one of four members is not flagged and bridged, if the bridge leaves the
    member more than 1 nT from what it was before the spike, or if a 20 nT step common to all four is
    flagged at all.

    The second limb is the one that matters: the departure from the median over members is what is measured,
    so a substorm that lifts the whole fleet within the same second is carried by the median and is signal.
    """
    before = _four_members()
    members = {d: v.copy() for d, v in before.items()}
    members["A"][5000] += 20.0
    for d in members:                                     # a storm: the same step in every member
        members[d][9000:] += 20.0
        before[d][9000:] += 20.0
    got, counts = REF.member_screen(members, 1.0)

    assert counts["A"]["flagged"] >= 2 and counts["A"]["bridged"] >= 1
    assert counts["A"]["nan_spans"] == 0
    assert counts["A"]["threshold_nt"] == REF.SCREEN_FLOOR_NT     # the floor governs at this noise level
    assert abs(float(got["A"][5000]) - float(before["A"][5000])) < 1.0
    assert float(np.nanmax(np.abs(np.diff(got["A"][4900:5100])))) < 2.0

    for d in ("B", "C", "D"):
        assert counts[d]["flagged"] == 0, "%s was flagged by another member's spike" % d
    for d in members:
        assert np.allclose(got[d][8950:9050], before[d][8950:9050]), "%s: the storm was screened" % d


def test_the_stack_is_smoother_than_its_spiking_member():
    """Fails if the stack of four members, one of them spiking, carries more steps above 2 nT per million
    samples than that member does, or if the screen does not cut the spiking member's own rate."""
    members = _four_members(seed=9)
    rng = np.random.default_rng(3)
    for i in rng.choice(np.arange(200, 19800), size=40, replace=False):
        members["A"][int(i)] += 20.0
    weights = {"A": 0.8, "B": 0.7, "C": 0.6, "D": 0.6}

    spiking = REF.steps_per_million(members["A"])
    unscreened = REF.steps_per_million(_weighted_stack(REF.level_match(members, 1.0)[0], weights))
    got, counts = REF.member_screen(members, 1.0)
    screened = REF.steps_per_million(_weighted_stack(REF.level_match(got, 1.0)[0], weights))

    assert spiking > 1000.0, "the spiking member is not rough enough to test against (%.0f)" % spiking
    assert unscreened > 0.0, "the unscreened stack already carries no step, so the screen is untested"
    assert screened < spiking
    assert screened < unscreened
    assert REF.steps_per_million(got["A"]) < spiking
    assert counts["A"]["bridged"] >= 30


def test_screen_leaves_a_long_span_nan_rather_than_bridging_it():
    """Fails if a flagged span longer than SCREEN_MAX_SPAN_S is bridged instead of left NaN."""
    members = _four_members(seed=12)
    members["A"][5000:5040] += np.linspace(0, 400, 40)    # 10 nT a sample for 40 s, well over the 12 s span
    got, counts = REF.member_screen(members, 1.0)
    assert counts["A"]["nan_spans"] >= 1
    assert not np.isfinite(got["A"][5010])
    for d in ("B", "C", "D"):
        assert counts[d]["nan_spans"] == 0


def test_a_sample_carried_by_one_member_is_masked(tmp_path):
    """Fails if a sample fewer than STACK_MIN members are finite at is left in the stack with its mask True.

    One member is a member renamed, per sample as per list. Measured at Q50, whose last two days rest on
    Q53N alone: setting those samples aside took it from 383 to 207 steps above 2 nT per million on Hx and
    from 185 to 18 on Hy (2026-09-17).
    """
    n = 1000
    den = {c: np.full(n, 1.8) for c in ("Hx", "Hy")}
    num = {c: np.full(n, 18.0) for c in ("Hx", "Hy")}
    cnt = np.full(n, 2, np.int16)
    cnt[400:450] = 1                                      # one member over fifty samples
    cnt[900:910] = 0
    for c in ("Hx", "Hy"):
        den[c][900:910] = 0.0

    h0, cov0, mask0 = REF.Store.finish(num, den)
    assert mask0[400] and cov0["Hx"] == 0.99               # without the floor the thin stretch passes

    thin = cnt < REF.STACK_MIN
    h, cov, mask = REF.Store.finish(num, den, thin=thin)
    assert not mask[400:450].any() and not mask[900:910].any()
    assert mask[:400].all() and mask[450:900].all()
    assert np.all(h["Hx"][400:450] == 0.0)
    assert cov["Hx"] == round(1.0 - 60 / n, 4)
    rows = REF._floor_rows(thin, REF.STACK_MIN)
    assert rows["samples_below"] == 60 and rows["n_min"] == REF.STACK_MIN
    assert rows["frac_below"] == 0.06


def test_the_floor_counts_site_members_and_not_the_observatory():
    """Fails if a stretch carried by one site member and the observatory survives in stack + observatory, or
    if the control that counts the observatory does not keep that same stretch.

    One site member and the observatory is a remote site and an observatory, not a stack and one, on the
    same sentence the member list is judged by. At Q50 that stretch was 2.54 per cent of the record and its
    lone member read 1,063 steps above 2 nT per million against 21-190 for the other six (2026-09-17).
    """
    n, a, b = 1000, 400, 450
    cnt = np.full(n, 2, np.int16)
    cnt[a:b] = 1                                          # one site member over the stretch
    obs_here = np.ones(n, bool)                            # and the observatory throughout
    den = {c: np.full(n, 2.7) for c in ("Hx", "Hy")}       # two site members plus the observatory
    num = {c: np.full(n, 27.0) for c in ("Hx", "Hy")}

    # the rule: the observatory does not count toward the floor
    _h, cov, mask = REF.Store.finish(num, den, thin=(cnt < REF.STACK_MIN))
    assert not mask[a:b].any() and mask[:a].all() and mask[b:].all()
    assert cov["Hx"] == round(1.0 - (b - a) / n, 4)
    rows = REF._floor_rows(cnt < REF.STACK_MIN, REF.STACK_MIN)
    assert rows["samples_below"] == b - a and rows["counts"] == "site members only"

    # the control: counting the observatory, as it was, keeps the stretch
    _h2, _c2, kept = REF.Store.finish(num, den, thin=((cnt + obs_here.astype(np.int16)) < REF.STACK_MIN))
    assert kept[a:b].all(), "the control does not keep the stretch, so the test proves nothing"

    # a stretch with two site members is kept either way
    _h3, _c3, all_kept = REF.Store.finish(num, den, thin=(np.full(n, 2, np.int16) < REF.STACK_MIN))
    assert all_kept.all()


def test_the_npz_gains_the_spans_only_where_a_kind_carries_them(tmp_path):
    """Fails if a kind that carries no bridged spans writes anything but the six arrays it always had, or if
    the spans a stack carries do not read back as the pairs they were written from.

    The first limb is what keeps the remote site and the observatory byte-identical across this change.
    """
    import pandas as pd

    class _Survey:
        cfg = dict(work_root="", name="test")
        sites = pd.DataFrame(dict(site=["A"]))

        def decision(self, name):
            raise KeyError(name)

    sv = _Survey()
    sv.cfg = dict(work_root=str(tmp_path), name="test")
    st = REF.Store(sv, ["A"], rate=1)
    h = {c: np.arange(100.0) for c in ("Hx", "Hy")}
    mask = np.ones(100, bool)

    p, _info = st._write("remote", "A", 1_700_000_000, h, mask, dict(note="no spans"))
    z = np.load(p, allow_pickle=False)
    assert sorted(z.files) == ["Hx", "Hy", "coverage", "fs", "mask", "t0"]
    z.close()

    spans = {"bridged_Hx": np.asarray([[10, 14], [40, 42]], np.int32),
             "bridged_Hy": np.zeros((0, 2), np.int32)}
    st._write("stack", "A", 1_700_000_000, h, mask, dict(note="spans"), extra=spans)
    back = REF.bridged_spans("stack", "A", 1, tmp_path, "Hx")
    assert back.shape == (2, 2) and back.tolist() == [[10, 14], [40, 42]]
    assert REF.bridged_spans("stack", "A", 1, tmp_path, "Hy").shape == (0, 2)
    assert REF.bridged_spans("remote", "A", 1, tmp_path, "Hx").shape == (0, 2)
    assert REF.bridged_spans("stack", "MISSING", 1, tmp_path, "Hx").shape == (0, 2)


def test_a_two_member_stack_is_screened_by_the_pair_rule():
    """Fails if a spike in one of only two members is not flagged and bridged, or if the sound member is
    flagged for its neighbour's spike.

    With two members the median of the first differences does not name the culprit, so the rule is the one
    whose own first difference clears the bar while the other's stays under half the floor.
    """
    assert REF.SCREEN_MIN_MEMBERS == 2
    members = _four_members(seed=5)
    before = {d: members[d].copy() for d in ("A", "B")}
    two = {d: members[d].copy() for d in ("A", "B")}
    two["A"][5000] += 20.0
    got, counts = REF.member_screen(two, 1.0)
    assert counts["A"]["flagged"] >= 2 and counts["A"]["flagged_pair"] >= 2
    assert counts["A"]["bridged"] >= 1
    assert counts["B"]["flagged"] == 0, "the sound member was flagged for its neighbour's spike"
    assert abs(float(got["A"][5000]) - float(before["A"][5000])) < 1.0


def test_a_step_common_to_both_of_two_members_is_not_flagged():
    """Fails if a 20 nT step both members take at the same sample is screened out of either of them.

    A step both records take is the field, and with only two members the pair rule has to say so without a
    median to lean on.
    """
    members = _four_members(seed=8)
    two = {d: members[d].copy() for d in ("A", "B")}
    for d in two:
        two[d][9000:] += 20.0
    got, counts = REF.member_screen(two, 1.0)
    assert counts["A"]["flagged"] == 0 and counts["B"]["flagged"] == 0
    for d in two:
        assert np.allclose(got[d], two[d]), "%s was changed by the screen" % d


# -------------------------------------------------------------- the level match

def _demeaned(members):
    """Each member demeaned over its own finite samples, which is the rule the level match replaced."""
    out = {}
    for d, v in members.items():
        x = np.asarray(v, float).copy()
        g = np.isfinite(x)
        x[g] -= x[g].mean()
        out[d] = x
    return out


def _four_with_a_late_member(n=20000, entry=10000, offset=10.0, drift=200.0, seed=17):
    """Four members of one slow drifting field, the fourth carrying its own offset and covering only the
    tail.

    The drift is what makes a per-record demeaning fail: a member that covers the second half alone is
    demeaned by the second half's own mean, which sits a quarter of the drift above the mean the other three
    were demeaned by, and the composite steps by that difference the moment the member appears.

    The field is slower here than in the other tests -- one 3,000 s component, so it moves at most
    0.07 nT a sample -- because the criterion below is an absolute 0.5 nT and a field that moves faster than
    that between samples would be scored as a step of its own.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    field = 25000.0 + 30.0 * np.sin(2 * np.pi * t / 3000.0) + drift * t / n
    members = {d: field + 0.05 * rng.standard_normal(n) for d in ("A", "B", "C", "D")}
    members["D"] = members["D"] + offset
    members["D"][:entry] = np.nan
    return members


def test_level_match_removes_a_member_entering_at_an_offset():
    """Fails if the stack steps by more than 0.5 nT where a fourth member carrying a 10 nT offset joins it,
    or if the control does not step by at least 5 nT in the first place.
    """
    n, entry = 20000, 10000
    members = _four_with_a_late_member(n=n, entry=entry)
    weights = {"A": 0.9, "B": 0.8, "C": 0.8, "D": 0.8}

    old = _weighted_stack(_demeaned(members), weights)
    got, rms, _lv = REF.level_match(members, 1.0)
    new = _weighted_stack(got, weights)

    stepped = abs(float(old[entry] - old[entry - 1]))
    assert stepped > 5.0, "the control does not step, so the test proves nothing (%.3f nT)" % stepped
    assert abs(float(new[entry] - new[entry - 1])) < 0.5
    assert rms["D"] > 5.0, "D's offset from the fleet was not measured (%s nT)" % rms["D"]
    # the offset is slow, so it takes nothing out of a member at the periods the stack is a reference for
    assert float(np.nanmax(np.abs(np.diff(got["A"] - members["A"])))) < 0.05
    assert REF.LEVEL_WIN_S == 3600.0


def test_level_match_holds_the_offset_beyond_its_own_ends():
    """Fails if a member is left unmatched at the record's ends, where the window is too empty to measure."""
    n = 20000
    members = _four_members(n=n, seed=21)
    members["D"] = members["D"] + 10.0
    got, _rms, _lv = REF.level_match(members, 1.0)
    for i in (0, 5, n - 1):
        assert abs(float(got["D"][i] - got["A"][i])) < 1.0, "D is still offset from A at sample %d" % i


def test_the_offset_is_not_dragged_by_an_excursion_at_a_run_edge():
    """Fails if a 2,000 nT excursion lasting 20 s at the start of a member's run moves that member's offset
    by more than 0.5 nT, or if the mean it replaced is not dragged by at least 5 nT.

    This is the defect the two medians are for: a mean takes such an excursion at its own area over the
    window, which put the member 13-22 nT off the fleet across a plateau ending at half the window, while in
    the interior of a run the members agree to 0.08 nT (Queensland Phases 1 and 3, 2026-09-17).
    """
    n, edge, spike = 20000, 8000, 8300
    members = _four_members(n=n, seed=31)
    clean = {d: v.copy() for d, v in members.items()}
    members["D"][:edge] = np.nan
    clean["D"][:edge] = np.nan
    # 20 s of it, inside the run and so past the 120 s the edge screen takes: this is what survives
    members["D"][spike:spike + 20] += 2000.0

    got, _rms, _lv = REF.level_match(members, 1.0)
    ref, _r2, _l2 = REF.level_match(clean, 1.0)
    # the sample mean this replaced: 2,000 nT over 20 s of a 3,600 s window is 11 nT of offset
    dragged = 20 * 2000.0 / REF.LEVEL_WIN_S
    assert dragged > 5.0, "the excursion is too small to drag a mean, so the test proves nothing"

    # away from the excursion itself the member has to sit where it would without it
    lo, hi = spike + 120, spike + 2000
    moved = float(np.nanmax(np.abs(got["D"][lo:hi] - ref["D"][lo:hi])))
    assert moved < 0.5, "the excursion moved the offset by %.2f nT" % moved
    for d in ("A", "B", "C"):
        assert float(np.nanmax(np.abs(got[d] - ref[d]))) < 0.5, "%s was moved by D's excursion" % d
    assert REF.LEVEL_BLOCK_S == 60.0


def test_the_offset_carries_no_step_of_its_own():
    """Fails if a member's offset moves by more than 0.5 nT from one sample to the next where the member set
    changes, or if the single-pass form does not show that step, which is what the first pass is for.

    The fleet level m is a median over whoever is present, and the members' own levels are hundreds of nT
    apart, so m jumps whenever a member joins or leaves. A running median follows that jump; the
    interpolation back to the sample grid then lays it into every member as a ramp of tens of nT a second.
    """
    n, out_a, out_b = 20000, 9000, 12000
    members = _four_members(n=n, seed=41)
    for d, dc in (("A", 0.0), ("B", 600.0), ("C", -900.0), ("D", 1500.0)):
        members[d] = members[d] + dc
    members["D"][out_a:out_b] = np.nan                    # the member set changes twice

    def offset_steps(passes):
        got, _rms, _lv = REF.level_match(members, 1.0, passes=passes)
        worst = 0.0
        for d in members:
            o = np.asarray(members[d], float) - got[d]     # the offset, up to the common constant
            worst = max(worst, float(np.nanmax(np.abs(np.diff(o)))))
        return worst

    assert REF.LEVEL_PASSES == 2
    one = offset_steps(1)
    two = offset_steps(2)
    assert one > 2.0, "the single pass does not step, so the test proves nothing (%.3f nT)" % one
    assert two < 0.5, "the offset still steps by %.3f nT a sample" % two


# -------------------------------------------------------------- the observatory

def test_the_observatory_is_matched_one_sided_and_does_not_step_at_its_mask_edges():
    """Fails if an observatory sitting thousands of nT off the members steps the composite by more than
    0.5 nT where its mask opens or closes, or if either control does not step.

    The observatory is far away and its own level is nothing like the members'. Demeaned over its own record
    it stepped the composite at every one of the archive's 42-70 mask edges over a Queensland window; pooled
    into the fleet median it drags the very level it is supposed to be matched against.
    """
    n = 20000
    rng = np.random.default_rng(51)
    t = np.arange(n, dtype=float)
    # a slow field, as in the other level test: the criterion below is an absolute 0.5 nT and a field that
    # moves faster than that between samples would be scored as a step of its own
    field = 30.0 * np.sin(2 * np.pi * t / 3000.0)
    members = {d: 25000.0 + field + 0.05 * rng.standard_normal(n) for d in ("A", "B", "C", "D")}
    # a stretch where one member is left: there the median of a pair is their mean, so an observatory
    # pooled into it drags the level by half its own distance from the fleet
    for d in ("B", "C", "D"):
        members[d][16000:17000] = np.nan
    lev, _rms, level = REF.level_match(members, 1.0)
    weights = {"A": 0.9, "B": 0.8, "C": 0.8, "D": 0.8}
    w_obs = 0.87

    # the same field the members carry, on its own level, with its own slow drift against them: the level
    # and the drift are the only differences, and they are what the match is for
    obs = 4200.0 + 150.0 * t / n + field + 0.05 * rng.standard_normal(n)
    omask = np.ones(n, bool)
    edges = [(5000, 5600), (9000, 9200), (14000, 15000)]     # the archive opening and closing
    for a, b in edges:
        omask[a:b] = False
    x = np.where(omask, obs, np.nan)

    def composite(add):
        num = np.zeros(n)
        den = np.zeros(n)
        for d, w in weights.items():
            g = np.isfinite(lev[d])
            num[g] += w * lev[d][g]
            den[g] += w
        g = np.isfinite(add)
        num[g] += w_obs * add[g]
        den[g] += w_obs
        return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)

    def worst_at_edges(h):
        return max(abs(float(h[i] - h[i - 1])) for a, b in edges for i in (a, b)
                   if np.isfinite(h[i]) and np.isfinite(h[i - 1]))

    # control one: demeaned over its own record, the rule this replaced
    demeaned = x.copy()
    good = np.isfinite(demeaned)
    demeaned[good] -= demeaned[good].mean()
    ctl = worst_at_edges(composite(demeaned))
    # control two: pooled into the fleet median, so it drags the level it is matched against
    pooled, _r2, _l2 = REF.level_match(dict(lev, CTA=x), 1.0)
    pol = float(np.nanmax(np.abs(np.diff(composite(pooled["CTA"])))))

    off = REF.one_sided_offset(x, level, 1.0)
    got = worst_at_edges(composite(x - off))

    assert ctl > 5.0, "the demeaned control does not step, so the test proves nothing (%.3f nT)" % ctl
    assert pol > 2.0, "the pooled-median control does not step, so it is not a control (%.3f nT)" % pol
    assert got < 0.5, "the matched observatory steps by %.3f nT at its mask edges" % got
    # matched, not flattened: its own variation survives
    assert float(np.nanstd(x - off)) > 10.0


def test_the_block_median_and_the_running_statistics_keep_their_shape():
    """Fails if a block whose samples are mostly NaN carries a value, or if a running statistic is not
    centred on the blocks it is taken over."""
    y = np.arange(600.0)
    b = REF._block_median(y, 60)
    assert len(b) == 10 and b[0] == np.median(np.arange(60.0))
    y2 = y.copy()
    y2[60:110] = np.nan                                   # 10 of 60 finite, under the third
    assert not np.isfinite(REF._block_median(y2, 60)[1])
    y3 = y.copy()
    y3[120:140] = np.nan                                  # 40 of 60 finite, over the third
    assert np.isfinite(REF._block_median(y3, 60)[2])

    blocks = np.arange(200.0)
    for stat in (REF._running_median, REF._running_mean):
        r = stat(blocks, 60, 6)
        assert not np.isfinite(r[0]) and not np.isfinite(r[199])
        assert r[30] == np.mean(np.arange(60.0))          # centred on its own window
        assert np.isfinite(r[30]) and np.isfinite(r[169])
    # the median holds still and then leaps where the series steps; the mean does not
    stepped = np.concatenate([np.zeros(100), np.full(100, 500.0)])
    assert np.nanmax(np.abs(np.diff(REF._running_median(stepped, 60, 6)))) > 100.0
    assert np.nanmax(np.abs(np.diff(REF._running_mean(stepped, 60, 6)))) < 10.0


def test_the_alignment_target_is_the_pair_the_pass_runs_on(tmp_path):
    """Fails if a member's lag is measured against a record the pass does not read.

    A target with decisions.csv h_lender runs on the lender's pair placed on its own grid, so that is what a
    member has to line up with; a target that borrows nothing runs on its own pair. The control is the
    second half: the borrowing site's own record is NOT what comes back, and the non-borrowing site's is
    exactly what Store.rotated returns, so the branch cannot pass by returning one record for both.
    """
    import pandas as pd

    n, t0 = 4 * 3600, 1_700_000_000
    rng = np.random.default_rng(4)
    own = np.cumsum(rng.normal(size=n)) + 40000.0          # the borrower's own, distinct record
    lent = np.cumsum(rng.normal(size=n)) + 40000.0         # the lender's, a different one
    cache_dir = tmp_path / "cache_1hz"
    cache_dir.mkdir(parents=True)
    for site, hx in (("A", own), ("B", lent)):
        np.savez(cache_dir / ("%s.npz" % site), t0=np.array([t0]), fs=np.array([1.0]),
                 layout=np.array(["edl_L"]), chan_x=np.array(["Ex"]), chan_y=np.array(["Ey"]),
                 dipole_n_m=np.array([100.0]), dipole_e_m=np.array([100.0]), e_gain=np.array([1.0]),
                 h_uv_per_nt=np.array([1.0]), bz_divider=np.array([1.0]),
                 Hx=hx, Hy=np.full(n, 10.0), Hz=np.zeros(n), Ex=np.zeros(n), Ey=np.zeros(n))

    def row(**kw):
        """One decisions.csv row with every cell of the grammar at its neutral value."""
        cells = {c: "decide" for c in ("sign_hx", "sign_hy", "sign_hz", "sign_ex", "sign_ey")}
        cells.update(sign_hx="+1", sign_hy="+1", sign_hz="+1", sign_ex="+1", sign_ey="+1",
                     e_exchange="no", e_shift_s="0", h_exchange="no", h_gain="1", h_lender="none",
                     h_lender_channels="none", rot_regimes="", rot_drop="")
        cells.update(kw)
        return pd.Series(cells)

    rows = {"A": row(h_lender="B", h_lender_channels="Hx Hy Hz"), "B": row()}

    class _Survey:
        def __init__(self):
            self.cfg = dict(work_root=str(tmp_path), name="test")
            self.sites = pd.DataFrame(dict(site=["A", "B"]))

        def decision(self, name):
            return rows[name]

        def site(self, name):
            raise KeyError(name)

    st = REF.Store(_Survey(), ["A", "B"], rate=1)

    # the borrower: the pair that comes back is the lender's and not its own
    _ta, ha = st.alignment_target("A")
    _tr, hr, _ang, _n = st.rotated("A")
    _tb, hb = st.alignment_target("B")
    got = np.asarray(ha["Hx"], float)
    assert np.corrcoef(got, np.asarray(hb["Hx"], float))[0, 1] > 0.999, "A is not aligned to B's record"
    assert np.corrcoef(got, np.asarray(hr["Hx"], float))[0, 1] < 0.5, "A came back on its own record"

    # the control: a site that borrows nothing gets exactly what rotated returns, so no lag can move there
    _t2, hb2, _a2, _n2 = st.rotated("B")
    for c in ("Hx", "Hy"):
        assert np.array_equal(np.asarray(hb[c], float), np.asarray(hb2[c], float)), c
