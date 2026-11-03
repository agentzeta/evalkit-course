#!/usr/bin/env python3
"""Lab 8 — eval infrastructure: six places the standard recipe is incomplete.

    python3 -I labs/lab8_infrastructure.py

Each section states the standard advice, then shows what it leaves out, using
numbers computed here rather than quoted.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalinfra import assertions, gate, pairwise, passk, retrieval, trajectory
from evalinfra.trajectory import Call, Trajectory


def rule(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


# --------------------------------------------------------------------------
rule("1. pass^k is easy to compute and hard to certify")

print("""The advice: production reliability lives in pass^k, so run each case
several times. Correct. What it omits is that pass^k amplifies the error in
your estimate of p, so the sample size that was fine for pass@1 is not fine
here.

A system measured at 80/100:""")
print()
rep = passk.assess(100, 80, 5, target=0.30)
print(rep.summary())

print(f"""
The point estimate clears the target and the lower bound does not. Reporting
{rep.pass_hat_k:.3f} here is reporting the middle of an interval running from
{rep.pass_hat_k_lo:.3f} to {rep.pass_hat_k_hi:.3f}, against a target of
{rep.target:.2f}. How much data would it take?""")
print()
for p in (0.85, 0.90, 0.95):
    n = passk.n_for_pass_hat_k(0.30, 5, assumed_p=p)
    print(f"  at a true per-run rate of {p:.2f}: {n:,} runs per case")
need = passk.required_per_run_rate(0.30, 5)
print(f"\n  and at any rate at or below {need:.5f}, no amount of data certifies it:")
print(f"  n_for_pass_hat_k(0.30, 5, assumed_p={need:.5f}) -> "
      f"{passk.n_for_pass_hat_k(0.30, 5, assumed_p=need)}")

print("""
Suite-level compounding, which is the figure nobody computes:""")
for n_cases, per in ((20, 0.99), (40, 0.99), (40, 0.999)):
    print(f"  {n_cases} cases each at pass^k {per}: "
          f"whole suite {passk.suite_all_pass([per] * n_cases):.3f}")


# --------------------------------------------------------------------------
rule("2. The standard fix for judge position bias is a selection on outcome")

print("""The advice: run every A/B comparison twice with the order swapped and
count only consistent verdicts. Right about the disease. The side effect is
that the pairs a judge flips on are the close ones, so discarding them leaves
the easy comparisons — and the surviving win rate is quoted as if it were the
whole sample.

Ten clear pairs where A wins, ten close pairs the judge flips on:""")
print()
mixed = [pairwise.Comparison(f"clear{i}", "a", "a", 100, 100) for i in range(10)] + [
    pairwise.Comparison(f"close{i}", "a", "b", 100, 100) for i in range(10)
]
print(pairwise.assess(mixed).summary())

print("""
A judge that simply prefers whatever it sees first:""")
print()
print(pairwise.assess(
    [pairwise.Comparison(str(i), "a", "b", 100, 100) for i in range(20)]
).summary())


# --------------------------------------------------------------------------
rule("3. Trajectory scoring measures your choice of reference")

print("""The advice: score tool-call precision and recall against a reference
path. It assumes the reference is the only correct route. Most tasks have
several.

One agent trajectory, scored against two references that are both valid:""")
print()
agent = Trajectory.of(
    [Call("search", {"q": "x"}), Call("fetch", {"id": 1}), Call("summarise")],
    final_correct=True,
)
spread = trajectory.score_against_any(
    agent,
    [
        # route A: search, fetch the document, summarise it
        Trajectory.of([Call("search", {"q": "x"}), Call("fetch", {"id": 1}),
                       Call("summarise")]),
        # route B: search returns enough to summarise from; no fetch needed.
        # Both are correct ways to do the task.
        Trajectory.of([Call("search", {"q": "x"}), Call("summarise")]),
    ],
)
assert spread is not None
print(spread.summary())

print("""
And the cost bug that an outcome-only eval cannot see — right answer, eight
calls where three would do:""")
print()
s = trajectory.score(
    Trajectory.of([Call("search")] * 6 + [Call("fetch"), Call("summarise")],
                  final_correct=True),
    Trajectory.of([Call("search"), Call("fetch"), Call("summarise")]),
)
assert s is not None
print(s.summary())


# --------------------------------------------------------------------------
rule("4. recall@k has a ceiling you did not choose")

print("""The advice: measure retrieval with recall@k. With R relevant documents
and cutoff k, recall@k can never exceed k/R. A perfect system looks like a
failing one.

A system that retrieved every document it had room for, k=5, 20 relevant:""")
print()
rel = [f"d{i}" for i in range(20)]
print(retrieval.assess([[f"d{i}" for i in range(5)]], [rel], k=5).summary())

print("""
And the 2x2 that separates a retrieval failure from a generation failure. The
cell to look at is correct-but-unsupported: right answer, evidence never
retrieved. Those cases pass an end-to-end eval and break on a model swap.""")
print()
print(retrieval.decompose(
    [True] * 5 + [False] * 5,
    [True] * 8 + [False] * 2,
).summary())


# --------------------------------------------------------------------------
rule("5. An assertion suite's pass rate is not a quality metric")

print("""The advice: deterministic assertions first — cheap, fast, zero
variance. All true. But the pass rate is over the failure modes the
assertions can see, and a suite covering three of eight categories renders
the same green as one covering all eight.

Four known failure categories, assertions for two of them, every case
passing:""")
print()
TAX = ("bad_json", "missing_field", "too_long", "hallucinated_citation")
partial = assertions.AssertionSuite(TAX)
partial.add(assertions.Assertion("json", "bad_json", lambda p: isinstance(p, dict)))
partial.add(assertions.Assertion("field", "missing_field", lambda p: "id" in p))
cases = [(f"c{i}", {"id": i}) for i in range(10)]
print(partial.run(cases).summary())

print("""
The same cases against a suite that covers the whole taxonomy:""")
print()
full = assertions.AssertionSuite(TAX)
full.add(assertions.Assertion("a", "bad_json", lambda p: isinstance(p, dict)))
full.add(assertions.Assertion("b", "missing_field", lambda p: "id" in p))
full.add(assertions.Assertion("c", "too_long", lambda p: len(str(p)) < 100))
full.add(assertions.Assertion("d", "hallucinated_citation", lambda p: "cite" not in p))
print(full.run(cases).summary())


# --------------------------------------------------------------------------
rule("6. Per-category gates multiply your false-failure rate")

print("""The advice: set per-category thresholds so a win in one area cannot
hide a regression in another. Right, and it multiplies the chance of a
spurious red build:""")
print()
for k in (1, 3, 6, 12, 20):
    print(f"  {k:>2} categories at alpha=0.05: "
          f"{gate.family_wise_false_failure_rate(k, 0.05):.1%} chance of at "
          f"least one false failure")

print("""
The second problem is sizing. A 2-point regression allowance at a 90% base
rate needs about 3,600 examples per category to be detectable. Here is what
400 buys:""")
print()
print(gate.decide([gate.CategoryResult("summarise", 400, 360, 0.90, 0.70, 2.0)]).summary())

print("""
A build where one category breaches its floor and one is too small to
measure. Note that INCONCLUSIVE blocks the merge: "we could not tell" is not
"fine".""")
print()
print(gate.decide([
    gate.CategoryResult("summarise", 4000, 3600, 0.90, 0.70, 2.0),
    gate.CategoryResult("safety", 4000, 2000, 0.95, 0.90, 2.0),
    gate.CategoryResult("rare_mode", 8, 6, 0.90, 0.70, 2.0),
]).summary())


# --------------------------------------------------------------------------
rule("What these six share")

print("""Every one of them is a case where the recipe produces a number and
the number is read as meaning more than it does. The fix is never a better
model or a cleverer metric; it is reporting the thing alongside the number
that tells you whether to believe it — an interval, a ceiling, a coverage
figure, a discard rate, a minimum detectable effect, a spread across
references.

A function that cannot honestly answer returns None and reports the
shortfall. It never relaxes a constraint to produce a number. That rule is
what makes an eval suite a control rather than a dashboard.""")
print()
