#!/usr/bin/env python3
"""Checks for evalinfra. Pure stdlib, self-contained, no pytest.

    python3 -I tests/test_evalinfra.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalinfra import assertions, gate, pairwise, passk, retrieval, trajectory
from evalinfra.trajectory import Call, Trajectory

PASS = 0
FAIL: list[str] = []


def ok(cond: bool, label: str) -> None:
    global PASS
    if cond:
        PASS += 1
    else:
        FAIL.append(label)


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def raises(fn, exc=Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


# ==========================================================================
# passk
# ==========================================================================

def test_passk() -> None:
    # pass@k and pass^k are exact mirrors on the extremes
    ok(close(passk.pass_at_k(10, 10, 3), 1.0), "pass@k all-success is 1")
    ok(close(passk.pass_at_k(10, 0, 3), 0.0), "pass@k no-success is 0")
    ok(close(passk.pass_hat_k(10, 10, 3), 1.0), "pass^k all-success is 1")
    ok(close(passk.pass_hat_k(10, 0, 3), 0.0), "pass^k no-success is 0")
    ok(close(passk.pass_hat_k(10, 2, 3), 0.0), "pass^k with c<k is 0")

    # k=1 collapses both to the raw rate
    ok(close(passk.pass_at_k(10, 7, 1), 0.7), "pass@1 is the raw rate")
    ok(close(passk.pass_hat_k(10, 7, 1), 0.7), "pass^1 is the raw rate")

    # hand-computed: n=5, c=3, k=2. pass@2 = 1 - C(2,2)/C(5,2) = 1 - 1/10
    ok(close(passk.pass_at_k(5, 3, 2), 0.9), "pass@2 hand value")
    # pass^2 = C(3,2)/C(5,2) = 3/10
    ok(close(passk.pass_hat_k(5, 3, 2), 0.3), "pass^2 hand value")

    # pass@k >= pass^k always
    for n, c, k in [(20, 13, 3), (50, 40, 5), (12, 6, 2), (100, 80, 4)]:
        ok(
            passk.pass_at_k(n, c, k) >= passk.pass_hat_k(n, c, k),
            f"pass@k >= pass^k at ({n},{c},{k})",
        )

    # the thread's headline figure
    ok(close(passk.pass_hat_k_plugin(0.8, 5), 0.32768), "0.8^5 = 0.32768")
    ok(
        abs(passk.pass_hat_k_plugin(0.8, 5) - 0.33) < 0.01,
        "plug-in pass^5 at p=0.8 is about 33%",
    )

    # inversion round-trips
    for target, k in [(0.95, 5), (0.5, 3), (0.99, 10)]:
        p = passk.required_per_run_rate(target, k)
        ok(close(p ** k, target, 1e-12), f"required rate inverts at ({target},{k})")
    ok(
        passk.required_per_run_rate(0.95, 5) > 0.98,
        "pass^5 >= 0.95 needs p > 0.98",
    )

    # THE headline result: n=100 at 80% cannot certify pass^5 >= 0.30
    rep = passk.assess(100, 80, 5, target=0.30)
    ok(rep.pass_hat_k > 0.30, "point estimate of pass^5 clears 0.30")
    ok(rep.pass_hat_k_lo < 0.30, "lower bound of pass^5 does not clear 0.30")
    ok(rep.certified is False, "n=100 at 80% is NOT certified for pass^5>=0.30")
    ok("NOT CERTIFIED" in rep.summary(), "summary states non-certification")

    # amplification: k * p^(k-1) at p=0.8, k=5 is 2.048
    ok(close(rep.amplification, 5 * 0.8 ** 4, 1e-9), "amplification formula")
    ok(rep.amplification > 2.0, "pass^5 amplifies estimate error more than 2x")

    # interval ordering and containment
    ok(rep.p_lo < rep.p_hat < rep.p_hi, "p interval brackets the estimate")
    ok(
        rep.pass_hat_k_lo <= rep.pass_hat_k_plugin <= rep.pass_hat_k_hi,
        "pass^k interval brackets the plug-in",
    )

    # a target the point estimate misses
    rep2 = passk.assess(100, 80, 5, target=0.60)
    ok(rep2.certified is False, "unreachable target is not certified")
    ok("below target" in rep2.summary(), "summary says the estimate is below target")

    # no target means no verdict
    ok(passk.assess(50, 45, 3).certified is None, "no target gives no verdict")

    # planning: refuses when the system cannot meet the target at all
    need = passk.required_per_run_rate(0.30, 5)
    ok(
        passk.n_for_pass_hat_k(0.30, 5, assumed_p=need - 0.01) is None,
        "planning refuses when assumed p is below the required rate",
    )
    ok(
        passk.n_for_pass_hat_k(0.30, 5, assumed_p=need) is None,
        "planning refuses at exactly the required rate",
    )
    n_needed = passk.n_for_pass_hat_k(0.30, 5, assumed_p=0.85)
    ok(n_needed is not None and n_needed > 100, "certifying pass^5 needs n>100 at p=0.85")

    # monotone: a higher assumed p needs no more data
    n_hi = passk.n_for_pass_hat_k(0.30, 5, assumed_p=0.95)
    ok(
        n_hi is not None and n_needed is not None and n_hi <= n_needed,
        "higher p needs no more runs",
    )

    # suite compounding
    ok(close(passk.suite_all_pass([1.0] * 10), 1.0), "perfect suite compounds to 1")
    ok(close(passk.suite_all_pass([0.99] * 40), 0.99 ** 40), "suite compounding")
    ok(passk.suite_all_pass([0.99] * 40) < 0.70, "40 cases at 0.99 is under 0.70")
    ok(close(passk.suite_all_pass([]), 1.0), "empty suite is vacuously 1")

    # input validation
    ok(raises(lambda: passk.pass_at_k(5, 6, 2), ValueError), "c>n rejected")
    ok(raises(lambda: passk.pass_at_k(5, 3, 6), ValueError), "k>n rejected")
    ok(raises(lambda: passk.pass_at_k(0, 0, 1), ValueError), "n=0 rejected")
    ok(raises(lambda: passk.pass_at_k(5, 3, 0), ValueError), "k=0 rejected")
    ok(raises(lambda: passk.pass_hat_k_plugin(1.5, 2), ValueError), "p>1 rejected")
    ok(
        raises(lambda: passk.required_per_run_rate(0.0, 2), ValueError),
        "target=0 rejected",
    )
    ok(
        raises(lambda: passk.suite_all_pass([0.5, 1.2]), ValueError),
        "out-of-range suite probability rejected",
    )
    # the k>n message must say why
    try:
        passk.pass_hat_k(3, 3, 5)
    except ValueError as e:
        ok("at least 5 runs" in str(e), "k>n message explains the requirement")


# ==========================================================================
# pairwise
# ==========================================================================

def _cmp(pid, fwd, rev, la=None, lb=None):
    return pairwise.Comparison(pid, fwd, rev, la, lb)


def test_pairwise() -> None:
    # consistency
    ok(_cmp("1", "a", "a").consistent, "same verdict both orders is consistent")
    ok(not _cmp("1", "a", "b").consistent, "flipped verdict is inconsistent")
    ok(_cmp("1", "a", "a").agreed_winner == "a", "agreed winner")
    ok(_cmp("1", "a", "b").agreed_winner is None, "no agreed winner when flipped")

    # position bias: a judge that always picks what it saw first.
    # forward -> "a" (a was first), reverse -> "b" (b was first)
    biased = [_cmp(str(i), "a", "b") for i in range(20)]
    pb = pairwise.position_bias(biased)
    ok(close(pb.rate, 1.0), "pure first-position judge has rate 1.0")
    ok(pb.p_value is not None and pb.p_value < 0.001, "pure bias is significant")
    ok("FIRST-POSITION" in pb.note, "first-position preference named")
    ok(pb.decisive == 40, "20 pairs give 40 decisive presentations")

    # an unbiased, consistent judge shows no position effect
    clean = [_cmp(str(i), "a", "a") for i in range(10)] + [
        _cmp(str(i + 10), "b", "b") for i in range(10)
    ]
    pb2 = pairwise.position_bias(clean)
    ok(close(pb2.rate, 0.5), "balanced consistent judge sits at 0.5")
    ok(pb2.p_value is not None and pb2.p_value > 0.05, "no bias detected on clean data")
    ok("no detectable position preference" in pb2.note, "clean note")

    # all ties -> undefined, not zero
    allties = [_cmp(str(i), "tie", "tie") for i in range(5)]
    pb3 = pairwise.position_bias(allties)
    ok(pb3.rate is None, "all-ties position bias is undefined, not 0")
    ok(pb3.p_value is None, "all-ties has no p-value")

    # length confound: a is longer and always wins
    verbose = [_cmp(str(i), "a", "a", la=400, lb=100) for i in range(15)]
    lc = pairwise.length_confound(verbose)
    ok(lc.rate is not None and close(lc.rate, 1.0), "longer always won")
    ok(lc.p_value is not None and lc.p_value < 0.001, "verbosity effect significant")
    ok("VERBOSITY" in lc.note, "verbosity preference named")
    ok("A much longer" in lc.stratified, "stratification band present")

    # near-identical lengths carry no length signal
    samelen = [_cmp(str(i), "a", "a", la=100, lb=103) for i in range(10)]
    lc2 = pairwise.length_confound(samelen)
    ok(lc2.rate is None, "near-identical lengths give no length rate")
    ok("near-identical" in lc2.note, "near-identical note")

    # missing lengths are not invented
    lc3 = pairwise.length_confound([_cmp(str(i), "a", "a") for i in range(5)])
    ok(lc3.rate is None, "no lengths recorded gives no rate")

    # THE finding: discarding inconsistent verdicts is selection on outcome.
    # 10 clear pairs A wins; 10 close pairs the judge flips on.
    mixed = [_cmp(f"clear{i}", "a", "a", la=100, lb=100) for i in range(10)] + [
        _cmp(f"close{i}", "a", "b", la=100, lb=100) for i in range(10)
    ]
    rep = pairwise.assess(mixed)
    ok(close(rep.discard_rate, 0.5), "half the pairs were discarded")
    ok(
        rep.a_win_consistent_only is not None
        and close(rep.a_win_consistent_only, 1.0),
        "consistent-only says A wins every time",
    )
    ok(close(rep.a_win_ties_as_ties, 0.75), "ties-as-ties says 0.75")
    ok(
        rep.divergence is not None and rep.divergence > 0.2,
        "the two estimates diverge sharply",
    )
    ok(rep.quotable_single_number is None, "no single number is quotable here")
    ok("NO SINGLE NUMBER" in rep.note, "refusal is explicit")
    ok("discard rate" in rep.note, "discard rate named in the refusal")

    # a clean, low-discard, unbiased set does yield a quotable number
    good = (
        [_cmp(f"a{i}", "a", "a", la=100, lb=100) for i in range(10)]
        + [_cmp(f"b{i}", "b", "b", la=100, lb=100) for i in range(9)]
        + [_cmp("x", "a", "b", la=100, lb=100)]
    )
    rep2 = pairwise.assess(good)
    ok(rep2.discard_rate <= 0.25, "discard rate within tolerance")
    ok(rep2.quotable_single_number is not None, "clean data gives a quotable number")
    ok("quotable" in rep2.note, "note says quotable")

    # bias alone disqualifies the number even with zero discards
    rep3 = pairwise.assess([_cmp(str(i), "a", "b") for i in range(20)])
    ok(close(rep3.discard_rate, 1.0), "a pure position judge is 100% inconsistent")
    ok(rep3.quotable_single_number is None, "biased judge gives no quotable number")

    ok(raises(lambda: pairwise.assess([]), ValueError), "empty comparison set rejected")


# ==========================================================================
# trajectory
# ==========================================================================

def test_trajectory() -> None:
    ref = Trajectory.of([Call("search"), Call("fetch"), Call("summarise")])

    # exact match
    exact = Trajectory.of(
        [Call("search"), Call("fetch"), Call("summarise")], final_correct=True
    )
    s = trajectory.score(exact, ref)
    ok(s is not None and close(s.tool_precision, 1.0), "exact match precision 1")
    ok(s is not None and close(s.tool_recall, 1.0), "exact match recall 1")
    ok(s is not None and close(s.tool_f1, 1.0), "exact match f1 1")
    ok(s is not None and close(s.step_ratio, 1.0), "exact match step ratio 1")
    ok(s is not None and s.efficiency_flag is None, "no efficiency flag on exact match")

    # no reference means no score — not a zero
    ok(trajectory.score(exact, None) is None, "no reference returns None")
    ok(
        trajectory.score(exact, Trajectory.of([])) is None,
        "empty reference returns None",
    )

    # order matters when ordered=True
    reordered = Trajectory.of([Call("fetch"), Call("search"), Call("summarise")])
    s_ord = trajectory.score(reordered, ref, ordered=True)
    s_unord = trajectory.score(reordered, ref, ordered=False)
    ok(s_ord is not None and s_ord.tool_f1 < 1.0, "reordering costs f1 when ordered")
    ok(s_unord is not None and close(s_unord.tool_f1, 1.0), "unordered ignores order")

    # the efficiency bug: right answer, far too many calls
    bloated = Trajectory.of(
        [Call("search")] * 6 + [Call("fetch"), Call("summarise")], final_correct=True
    )
    s_b = trajectory.score(bloated, ref)
    ok(s_b is not None and s_b.step_ratio > 2.0, "bloated trajectory step ratio > 2")
    ok(s_b is not None and s_b.efficiency_flag is not None, "efficiency flag fires")
    ok(s_b is not None and s_b.tool_precision < 0.5, "bloat costs precision")
    ok(s_b is not None and close(s_b.tool_recall, 1.0), "bloat keeps recall")
    ok(s_b is not None and s_b.final_correct is True, "right answer still recorded")

    # argument correctness only over matched calls
    aref = Trajectory.of([Call("search", {"q": "x"}), Call("fetch", {"id": 1})])
    agot = Trajectory.of([Call("search", {"q": "y"}), Call("fetch", {"id": 1})])
    s_a = trajectory.score(agot, aref)
    ok(s_a is not None and close(s_a.arg_correctness or -1, 0.5), "1 of 2 args correct")
    ok(s_a is not None and s_a.n_matched == 2, "both tool names matched")

    # no matched calls -> argument correctness undefined, not zero
    s_none = trajectory.score(Trajectory.of([Call("other")]), aref)
    ok(s_none is not None and s_none.arg_correctness is None, "no matches, no arg score")
    ok(s_none is not None and close(s_none.tool_f1, 0.0), "no matches gives f1 0")

    # error recovery
    recov = Trajectory.of([Call("search", ok=False), Call("search"), Call("fetch")])
    s_r = trajectory.score(recov, ref)
    ok(s_r is not None and s_r.errors == 1, "one error counted")
    ok(s_r is not None and s_r.recovered == 1, "error was recovered")
    ok(s_r is not None and close(s_r.recovery_rate or -1, 1.0), "recovery rate 1")

    dead = Trajectory.of([Call("search"), Call("fetch", ok=False)])
    s_d = trajectory.score(dead, ref)
    ok(s_d is not None and s_d.unrecovered == 1, "trailing error is unrecovered")
    # 0.0 is falsy, so `x or -1` would silently test -1 here. Check None explicitly.
    ok(
        s_d is not None and s_d.recovery_rate is not None and close(s_d.recovery_rate, 0.0),
        "recovery rate 0",
    )

    s_clean = trajectory.score(exact, ref)
    ok(s_clean is not None and s_clean.recovery_rate is None, "no errors, no rate")

    # THE finding: reference dependence
    ref_a = Trajectory.of([Call("search"), Call("fetch"), Call("summarise")])
    ref_b = Trajectory.of([Call("lookup"), Call("render")])
    spread = trajectory.score_against_any(exact, [ref_a, ref_b])
    ok(spread is not None and close(spread.best.tool_f1, 1.0), "best reference matched")
    ok(spread is not None and spread.best_index == 0, "best index identified")
    ok(spread is not None and spread.spread > 0.15, "spread exceeds tolerance")
    ok(spread is not None and "REFERENCE-DEPENDENT" in spread.note, "dependence flagged")

    # one reference: dependence is unmeasured, and says so
    one = trajectory.score_against_any(exact, [ref_a])
    ok(one is not None and close(one.spread, 0.0), "single reference has zero spread")
    ok(one is not None and "unmeasured" in one.note, "single reference caveat")

    # agreeing references: with 3-call paths, f1 granularity is too coarse for a
    # 0.15 tolerance, so use longer paths that differ by one call.
    long_a = Trajectory.of([Call(f"t{i}") for i in range(10)])
    long_b = Trajectory.of([Call(f"t{i}") for i in range(9)] + [Call("other")])
    agree = trajectory.score_against_any(long_a, [long_a, long_b])
    ok(agree is not None and agree.spread < 0.15, "near-identical references agree")
    ok(agree is not None and "references agree" in agree.note, "agreeing references")

    ok(trajectory.score_against_any(exact, []) is None, "no references returns None")
    ok(
        trajectory.score_against_any(exact, [Trajectory.of([])]) is None,
        "only-empty references returns None",
    )

    # trajectory aggregates
    ok(bloated.n_steps == 8, "step count")
    ok(dead.n_errors == 1, "error count")
    ok(Trajectory.of([Call("a", tokens=5), Call("b", tokens=7)]).total_tokens == 12,
       "token total")


# ==========================================================================
# retrieval
# ==========================================================================

def test_retrieval() -> None:
    # THE ceiling: k=5 against 20 relevant docs caps recall at 0.25
    ok(close(retrieval.recall_ceiling(5, 20), 0.25), "recall@5 ceiling with 20 relevant")
    ok(close(retrieval.recall_ceiling(20, 5), 1.0), "ceiling caps at 1")
    ok(close(retrieval.recall_ceiling(5, 5), 1.0), "k == relevant gives ceiling 1")
    ok(close(retrieval.recall_ceiling(3, 0), 0.0), "no relevant docs gives ceiling 0")

    rel20 = [f"d{i}" for i in range(20)]
    perfect5 = [f"d{i}" for i in range(5)]
    ok(close(retrieval.recall_at_k(perfect5, rel20, 5), 0.25), "perfect recall@5 is 0.25")

    rep = retrieval.assess([perfect5], [rel20], k=5)
    ok(close(rep.recall, 0.25), "report recall")
    ok(close(rep.ceiling, 0.25), "report ceiling")
    ok(rep.attainment is not None and close(rep.attainment, 1.0),
       "attainment is 1.0 — the system was perfect")
    ok("CEILING BINDING" in rep.note, "binding ceiling flagged")
    ok("attainment" in rep.summary(), "summary directs the reader to attainment")

    # unconstrained case
    rep2 = retrieval.assess([["d0", "d1"]], [["d0", "d1"]], k=5)
    ok(close(rep2.ceiling, 1.0), "ceiling 1 when k >= relevant")
    ok("unconstrained" in rep2.note, "unconstrained note")

    # precision and MRR
    ok(close(retrieval.precision_at_k(["a", "x", "b"], ["a", "b"], 3), 2 / 3), "precision@3")
    ok(close(retrieval.precision_at_k(["x", "y"], ["a"], 2), 0.0), "precision 0")
    ok(close(retrieval.reciprocal_rank(["x", "a"], ["a"]), 0.5), "RR at rank 2")
    ok(close(retrieval.reciprocal_rank(["a"], ["a"]), 1.0), "RR at rank 1")
    ok(close(retrieval.reciprocal_rank(["x"], ["a"]), 0.0), "RR with no hit")
    ok(close(retrieval.mrr([["a"], ["x", "b"]], [["a"], ["b"]]), 0.75), "MRR over two queries")

    # nDCG: ideal computed over at most k, so a perfect top-k ranking scores 1
    relevance = {f"d{i}": 1.0 for i in range(20)}
    ok(
        close(retrieval.ndcg_at_k(perfect5, relevance, 5), 1.0),
        "nDCG@5 is 1.0 for a perfect top-5 against 20 relevant",
    )
    graded = {"a": 3.0, "b": 2.0, "c": 1.0}
    ok(close(retrieval.ndcg_at_k(["a", "b", "c"], graded, 3), 1.0), "graded ideal order")
    ok(retrieval.ndcg_at_k(["c", "b", "a"], graded, 3) < 1.0, "worse order scores lower")
    ok(close(retrieval.ndcg_at_k(["x"], {}, 3), 0.0), "no relevance gives nDCG 0")
    ok(close(retrieval.dcg([1.0]), 1.0), "dcg of a single gain at rank 1")
    ok(close(retrieval.dcg([0.0, 1.0]), 1 / math.log2(3)), "dcg discount at rank 2")

    # THE 2x2: correct-but-unsupported inflates end-to-end accuracy
    ev = [True] * 5 + [False] * 5
    corr = [True] * 5 + [True] * 3 + [False] * 2
    dec = retrieval.decompose(ev, corr)
    ok(dec.supported_correct == 5, "supported+correct cell")
    ok(dec.unsupported_correct == 3, "unsupported+correct cell")
    ok(dec.unsupported_incorrect == 2, "unsupported+incorrect cell")
    ok(dec.supported_incorrect == 0, "supported+incorrect cell")
    ok(close(dec.end_to_end_accuracy, 0.8), "end-to-end accuracy 0.8")
    ok(close(dec.retrieval_sufficiency, 0.5), "retrieval sufficiency 0.5")
    ok(dec.faithfulness is not None and close(dec.faithfulness, 1.0), "faithfulness 1.0")
    ok("WARNING" in dec.note, "parametric-knowledge warning fires at 30%")
    ok("parametric" in dec.note, "warning names the cause")

    # faithfulness undefined when nothing was retrieved
    dec2 = retrieval.decompose([False, False], [True, False])
    ok(dec2.faithfulness is None, "faithfulness undefined with no evidence present")
    ok("undefined" in dec2.summary(), "summary reports undefined faithfulness")

    # a clean system: everything supported and everything right, no warning
    dec3 = retrieval.decompose([True] * 10, [True] * 10)
    ok(dec3.unsupported_correct == 0, "no parametric component")
    ok("no large parametric" in dec3.note, "clean note")
    ok(dec3.faithfulness is not None and close(dec3.faithfulness, 1.0), "faithfulness 1.0")

    # one unfaithful case out of ten: faithfulness drops below sufficiency, and
    # that is correctly diagnosed as generation rather than retrieval
    dec3b = retrieval.decompose([True] * 10, [True] * 9 + [False])
    ok(dec3b.faithfulness is not None and close(dec3b.faithfulness, 0.9), "faithfulness 0.9")
    ok(close(dec3b.retrieval_sufficiency, 1.0), "retrieval found everything")
    ok("generation is the weaker half" in dec3b.note, "single unfaithful case diagnosed")

    # generation is the weaker half
    dec4 = retrieval.decompose([True] * 10, [True] * 4 + [False] * 6)
    ok(close(dec4.retrieval_sufficiency, 1.0), "retrieval found everything")
    ok(dec4.faithfulness is not None and dec4.faithfulness < 1.0, "generation lagged")
    ok("generation is the weaker half" in dec4.note, "weaker-half diagnosis")

    # validation
    ok(raises(lambda: retrieval.decompose([True], [True, False]), ValueError),
       "mismatched decompose lengths rejected")
    ok(raises(lambda: retrieval.decompose([], []), ValueError), "empty decompose rejected")
    ok(raises(lambda: retrieval.assess([], [], 5), ValueError), "empty assess rejected")
    ok(raises(lambda: retrieval.assess([["a"]], [["a"]], 0), ValueError), "k=0 rejected")
    ok(raises(lambda: retrieval.recall_ceiling(0, 5), ValueError), "ceiling k=0 rejected")
    ok(raises(lambda: retrieval.mrr([], []), ValueError), "empty MRR rejected")


# ==========================================================================
# assertions
# ==========================================================================

TAX = ("bad_json", "missing_field", "too_long", "hallucinated_citation")


def test_assertions() -> None:
    # a taxonomy is required
    ok(raises(lambda: assertions.AssertionSuite([]), ValueError), "empty taxonomy rejected")

    # covers is required and must be in the taxonomy
    ok(
        raises(lambda: assertions.Assertion("x", "", lambda p: True), ValueError),
        "assertion without covers rejected",
    )
    suite = assertions.AssertionSuite(TAX)
    ok(
        raises(
            lambda: suite.add(assertions.Assertion("x", "not_a_category", lambda p: True)),
            ValueError,
        ),
        "covers outside taxonomy rejected",
    )
    ok(raises(lambda: suite.run([("c", {})]), ValueError), "no assertions registered")

    # partial coverage: 2 of 4 categories
    suite.add(assertions.Assertion("json", "bad_json", lambda p: isinstance(p, dict)))
    suite.add(assertions.Assertion("field", "missing_field", lambda p: "id" in p))

    cases = [(f"c{i}", {"id": i}) for i in range(10)]
    res = suite.run(cases)
    ok(close(res.pass_rate, 1.0), "every case passes the registered assertions")
    ok(close(res.coverage, 0.5), "coverage is 2 of 4 categories")
    ok(len(res.uncovered) == 2, "two categories uncovered")
    ok("too_long" in res.uncovered, "too_long reported uncovered")

    # THE refusal: 100% pass rate yields NO quality estimate at 50% coverage
    ok(res.quality_estimate is None, "100% pass but incomplete coverage gives None")
    sf = res.shortfall()
    ok(sf is not None and "no assertion" in sf, "shortfall names the gap")
    ok("NO ASSERTION" in res.summary(), "summary marks uncovered categories")
    ok("not a statement about the system" in res.summary(), "summary refuses the reading")

    # complete coverage unlocks the estimate
    full = assertions.AssertionSuite(TAX)
    full.add(assertions.Assertion("a", "bad_json", lambda p: isinstance(p, dict)))
    full.add(assertions.Assertion("b", "missing_field", lambda p: "id" in p))
    full.add(assertions.Assertion("c", "too_long", lambda p: len(str(p)) < 100))
    full.add(assertions.Assertion("d", "hallucinated_citation", lambda p: "cite" not in p))
    res2 = full.run(cases)
    ok(close(res2.coverage, 1.0), "complete coverage")
    ok(res2.quality_estimate is not None, "complete coverage gives an estimate")
    ok(close(res2.quality_estimate or -1, 1.0), "estimate equals the pass rate")
    ok(res2.shortfall() is None, "no shortfall when complete")

    # a failing case moves the rate but not the coverage
    res3 = full.run(cases + [("bad", {"nope": 1})])
    ok(res3.quality_estimate is not None and res3.quality_estimate < 1.0, "failure lowers rate")
    ok(close(res3.per_category["missing_field"], 10 / 11), "per-category rate")

    # an assertion that raises is a broken measurement, not a failure
    boom = assertions.AssertionSuite(TAX)
    boom.add(assertions.Assertion("a", "bad_json", lambda p: isinstance(p, dict)))
    boom.add(assertions.Assertion("b", "missing_field", lambda p: p["id"] > 0))
    boom.add(assertions.Assertion("c", "too_long", lambda p: len(str(p)) < 100))
    boom.add(assertions.Assertion("d", "hallucinated_citation", lambda p: True))
    res4 = boom.run([("ok", {"id": 1}), ("missing", {})])
    ok(res4.errored == 1, "one assertion raised")
    ok(res4.quality_estimate is None, "an errored evaluation blocks the estimate")
    sf4 = res4.shortfall()
    ok(sf4 is not None and "broken measurements" in sf4, "errors named as broken measurement")
    outcomes = [o for c in res4.cases for o in c.outcomes if o.errored]
    ok(len(outcomes) == 1 and not outcomes[0].counted, "errored outcome is not counted")
    ok(outcomes[0].error is not None and "KeyError" in outcomes[0].error, "error recorded")

    ok(raises(lambda: full.run([]), ValueError), "empty case list rejected")


# ==========================================================================
# gate
# ==========================================================================

def _cat(name, n, passed, base, thr=0.70, maxreg=2.0):
    return gate.CategoryResult(name, n, passed, base, thr, maxreg)


def test_gate() -> None:
    # THE arithmetic: 12 categories at 5% is a 46% false-failure rate
    ok(close(gate.family_wise_false_failure_rate(1, 0.05), 0.05), "k=1 is alpha")
    fw12 = gate.family_wise_false_failure_rate(12, 0.05)
    ok(0.45 < fw12 < 0.47, "12 categories at 5% gives about 46%")
    ok(
        gate.family_wise_false_failure_rate(20, 0.05)
        > gate.family_wise_false_failure_rate(10, 0.05),
        "fwer increases in k",
    )
    ok(raises(lambda: gate.family_wise_false_failure_rate(0), ValueError), "k=0 rejected")

    # Holm
    ok(gate.holm([]) == [], "holm of empty")
    adj = gate.holm([0.01, 0.04, 0.03])
    ok(close(adj[0], 0.03), "smallest p times 3")
    ok(all(a >= b for a, b in zip(adj, [0.01, 0.04, 0.03])), "holm never decreases a p")
    ok(all(0.0 <= a <= 1.0 for a in gate.holm([0.5, 0.6, 0.9])), "holm stays in [0,1]")
    # monotone in the sorted order
    adj2 = gate.holm([0.001, 0.002, 0.003, 0.9])
    srt = sorted(range(4), key=lambda i: [0.001, 0.002, 0.003, 0.9][i])
    ok(
        all(adj2[srt[i]] <= adj2[srt[i + 1]] for i in range(3)),
        "holm is monotone in rank order",
    )

    # MDE refuses at tiny n rather than returning a meaningless number
    ok(gate.mde_proportion(8, 0.5) is None, "n<10 gives no MDE")
    ok(gate.mde_proportion(100, 0.0) is None, "degenerate base rate gives no MDE")
    ok(gate.mde_proportion(100, 1.0) is None, "base rate 1 gives no MDE")
    m100 = gate.mde_proportion(100, 0.5)
    m400 = gate.mde_proportion(400, 0.5)
    ok(m100 is not None and m400 is not None and m400 < m100, "MDE shrinks with n")
    ok(m100 is not None and 15 < m100 < 25, "MDE at n=100, p=0.5 is roughly 20pp")

    # THE sizing result: n=400 at base 0.90 has an MDE of ~5.9pp, so it CANNOT
    # gate on a 2pp allowance. The gate refuses rather than passing it.
    undersized = [_cat("summarise", 400, 360, 0.90, thr=0.70, maxreg=2.0)]
    d0 = gate.decide(undersized)
    ok(d0.verdict == "INCONCLUSIVE", "n=400 cannot gate a 2pp allowance at base 0.90")
    uv = [c for c in d0.categories if c.name == "summarise"][0]
    ok(close(uv.delta_pp, 0.0), "no regression observed at all")
    ok(uv.mde_pp is not None and 5.0 < uv.mde_pp < 7.0, "MDE at n=400 is about 6pp")
    ok("would not have been detected" in uv.reason, "undersizing explained, not hidden")

    # a clean build passes once each category is actually powered for its
    # allowance: a 2pp allowance at base 0.90 needs n of roughly 3,600.
    good = [
        _cat("summarise", 4000, 3600, 0.90),
        _cat("extract", 4000, 3560, 0.89),
        _cat("classify", 4000, 3800, 0.95),
    ]
    d = gate.decide(good)
    ok(d.verdict == "PASS", "clean build passes")
    ok(
        all(c.mde_pp is not None and c.mde_pp <= 2.0 for c in d.categories),
        "every passing category is powered for its allowance",
    )
    ok(d.failed == (), "nothing failed")
    ok(d.inconclusive == (), "nothing inconclusive")
    ok(all(c.verdict == "PASS" for c in d.categories), "every category passes")
    ok("Holm applied" in d.summary(), "summary reports multiplicity control")

    # a floor breach fails without needing significance
    floor = [
        _cat("summarise", 400, 360, 0.90),
        _cat("safety", 400, 200, 0.95, thr=0.90),
    ]
    d2 = gate.decide(floor)
    ok(d2.verdict == "FAIL", "floor breach fails the gate")
    ok("safety" in d2.failed, "the breaching category is named")
    sv = [c for c in d2.categories if c.name == "safety"][0]
    ok("absolute floor" in sv.reason, "floor breach reason")
    ok("does not need a significance test" in sv.reason, "floor needs no test")

    # THE refusal: a category too small to measure cannot gate
    small = [
        _cat("summarise", 400, 360, 0.90),
        _cat("rare_mode", 8, 6, 0.90),
    ]
    d3 = gate.decide(small)
    ok(d3.verdict == "INCONCLUSIVE", "a tiny category makes the gate inconclusive")
    ok("rare_mode" in d3.inconclusive, "the tiny category is named")
    rv = [c for c in d3.categories if c.name == "rare_mode"][0]
    ok(rv.mde_pp is None, "no MDE for n=8")
    ok("can only fire by noise" in rv.reason, "reason explains the refusal")
    ok("not a pass" in d3.summary(), "summary states inconclusive is not a pass")

    # a real regression that survives Holm fails
    reg = [
        _cat("summarise", 2000, 1600, 0.90, thr=0.70, maxreg=2.0),
        _cat("extract", 2000, 1790, 0.89),
    ]
    d4 = gate.decide(reg)
    ok(d4.verdict == "FAIL", "a large well-powered regression fails")
    sv4 = [c for c in d4.categories if c.name == "summarise"][0]
    ok(sv4.delta_pp < -2.0, "regression exceeds the allowance")
    ok("survives Holm" in sv4.reason, "Holm survival stated")

    # a regression at an n that cannot resolve it is inconclusive, not a pass
    weak = [_cat("summarise", 40, 33, 0.90, thr=0.70, maxreg=2.0)]
    d5 = gate.decide(weak, min_n=30)
    ok(d5.verdict in ("FAIL", "INCONCLUSIVE"), "underpowered regression is not a PASS")

    # no regression but inadequate power is still not a clean pass
    thin = [_cat("summarise", 50, 45, 0.90, thr=0.70, maxreg=2.0)]
    d6 = gate.decide(thin, min_n=30)
    ok(d6.verdict == "INCONCLUSIVE", "adequate rate but inadequate power is inconclusive")
    tv = [c for c in d6.categories if c.name == "summarise"][0]
    ok("would not have been detected" in tv.reason, "power shortfall explained")

    # fwer is reported even on a passing build
    ok(d.uncorrected_fwer > 0.05, "fwer reported for a 3-category gate")
    ok(d.n_categories == 3, "category count reported")

    ok(raises(lambda: gate.decide([]), ValueError), "empty gate rejected")


# ==========================================================================

def main() -> int:
    for fn in (
        test_passk,
        test_pairwise,
        test_trajectory,
        test_retrieval,
        test_assertions,
        test_gate,
    ):
        fn()
    if FAIL:
        print(f"{PASS} passed, {len(FAIL)} FAILED:")
        for f in FAIL:
            print(f"  FAIL: {f}")
        return 1
    print(f"all green ({PASS} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
