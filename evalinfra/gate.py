"""CI gating on per-category thresholds — and the family-wise cost.

Per-category thresholds are the right design. One global score lets a large
win in an easy category hide a regression in a critical one, and splitting
the gate by category fixes that.

It also multiplies your false-failure rate, and nobody budgets for it.

**With k categories each tested at alpha, the chance of at least one
spurious failure is 1 - (1 - alpha)^k.** Twelve categories at 5% gives 46%.
Build goes red on roughly every other clean PR, the team learns to re-run
until it passes, and the gate is now decorative — worse than no gate, because
it still produces the green badge.

The second problem is quieter. **A category with eight examples cannot gate
anything.** Its minimum detectable effect is larger than any regression you
would care about, so it can only fire by noise. This module refuses to gate
such a category rather than letting it contribute false reds.

`decide` therefore returns one of three verdicts — PASS, FAIL, or
INCONCLUSIVE — and INCONCLUSIVE is not a pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

__all__ = [
    "CategoryResult",
    "CategoryVerdict",
    "GateDecision",
    "family_wise_false_failure_rate",
    "holm",
    "mde_proportion",
    "decide",
]

Verdict = Literal["PASS", "FAIL", "INCONCLUSIVE"]


# --------------------------------------------------------------------------
# the arithmetic people skip
# --------------------------------------------------------------------------

def family_wise_false_failure_rate(k: int, alpha: float = 0.05) -> float:
    """P(at least one spurious failure) across k independent category tests."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return 1.0 - (1.0 - alpha) ** k


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order.

    Uniformly more powerful than Bonferroni and valid without independence
    assumptions, which is what you want when categories share examples.
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvalues[idx]
        running = max(running, val)
        adjusted[idx] = min(1.0, running)
    return adjusted


def _z(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def mde_proportion(
    n: int,
    base_rate: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float | None:
    """Minimum detectable change in a proportion, in percentage points.

    Returns None when n is too small for the normal approximation to mean
    anything (n < 10), rather than returning a number that implies the test
    is usable.
    """
    if n < 10:
        return None
    if not 0.0 < base_rate < 1.0:
        return None
    z_a = _z(1 - alpha / 2)
    z_b = _z(power)
    var = base_rate * (1 - base_rate)
    return (z_a + z_b) * math.sqrt(2 * var / n) * 100.0


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CategoryResult:
    """One category's measurement on this build and the last accepted one."""

    name: str
    n: int
    passed: int
    baseline_rate: float
    threshold: float
    """Absolute floor the category must hold, as a rate in [0, 1]."""
    max_regression_pp: float = 2.0
    """How many percentage points below baseline is tolerated."""

    @property
    def rate(self) -> float:
        return self.passed / self.n if self.n else 0.0

    @property
    def delta_pp(self) -> float:
        return (self.rate - self.baseline_rate) * 100.0


@dataclass(frozen=True)
class CategoryVerdict:
    name: str
    verdict: Verdict
    rate: float
    baseline_rate: float
    delta_pp: float
    mde_pp: float | None
    p_value: float | None
    p_adjusted: float | None
    reason: str


@dataclass(frozen=True)
class GateDecision:
    verdict: Verdict
    categories: tuple[CategoryVerdict, ...]
    n_categories: int
    alpha: float
    uncorrected_fwer: float
    failed: tuple[str, ...]
    inconclusive: tuple[str, ...]

    def summary(self) -> str:
        lines = [
            f"GATE: {self.verdict}",
            f"  {self.n_categories} categories at alpha={self.alpha}; "
            f"uncorrected family-wise false-failure rate would be "
            f"{self.uncorrected_fwer:.1%} — Holm applied.",
            "",
        ]
        for c in self.categories:
            mde = "n/a" if c.mde_pp is None else f"{c.mde_pp:>5.1f}pp"
            padj = "n/a" if c.p_adjusted is None else f"{c.p_adjusted:.4f}"
            lines.append(
                f"  {c.verdict:<13} {c.name:<26} "
                f"{c.rate:.3f} (base {c.baseline_rate:.3f}, "
                f"{c.delta_pp:+.1f}pp)  mde {mde}  p_adj {padj}"
            )
            lines.append(f"                {c.reason}")
        if self.failed:
            lines.append("")
            lines.append(f"  FAILED: {', '.join(self.failed)}")
        if self.inconclusive:
            lines.append(
                f"  INCONCLUSIVE (not a pass): {', '.join(self.inconclusive)}"
            )
        return "\n".join(lines)


def _prop_test_p(n: int, passed: int, baseline: float) -> float | None:
    """Two-sided one-sample test of the observed rate against the baseline."""
    if n < 10 or not 0.0 < baseline < 1.0:
        return None
    se = math.sqrt(baseline * (1 - baseline) / n)
    if se == 0:
        return None
    z = (passed / n - baseline) / se
    # two-sided normal tail
    return math.erfc(abs(z) / math.sqrt(2))


def decide(
    results: Sequence[CategoryResult],
    alpha: float = 0.05,
    min_n: int = 30,
) -> GateDecision:
    """Gate a build on per-category thresholds, with multiplicity controlled.

    A category fails only when it breaches its absolute floor, or when a
    regression beyond `max_regression_pp` survives Holm correction. A
    category too small to measure is INCONCLUSIVE, and any INCONCLUSIVE makes
    the whole gate INCONCLUSIVE — which must block the merge the same way a
    FAIL does. Treating "we could not tell" as "fine" is how an eval suite
    stops being a control.
    """
    if not results:
        raise ValueError("no category results supplied")

    k = len(results)
    raw_p: list[float | None] = [
        _prop_test_p(r.n, r.passed, r.baseline_rate) for r in results
    ]
    have = [i for i, p in enumerate(raw_p) if p is not None]
    adj: list[float | None] = [None] * k
    if have:
        adj_vals = holm([raw_p[i] for i in have])  # type: ignore[arg-type]
        for slot, i in enumerate(have):
            adj[i] = adj_vals[slot]

    verdicts: list[CategoryVerdict] = []
    for i, r in enumerate(results):
        mde = mde_proportion(r.n, r.baseline_rate)
        p, pa = raw_p[i], adj[i]

        if r.n < min_n or mde is None:
            v: Verdict = "INCONCLUSIVE"
            reason = (
                f"n={r.n} below the minimum of {min_n}"
                if r.n < min_n
                else f"n={r.n} too small to estimate an effect"
            )
            reason += "; this category cannot gate, it can only fire by noise."
        elif r.rate < r.threshold:
            v = "FAIL"
            reason = (
                f"below absolute floor {r.threshold:.3f} — "
                f"a floor breach does not need a significance test."
            )
        elif r.delta_pp < -r.max_regression_pp:
            if pa is not None and pa < alpha:
                v = "FAIL"
                reason = (
                    f"regression of {-r.delta_pp:.1f}pp exceeds the "
                    f"{r.max_regression_pp:.1f}pp allowance and survives Holm "
                    f"(p_adj={pa:.4f})."
                )
            else:
                v = "INCONCLUSIVE"
                reason = (
                    f"regression of {-r.delta_pp:.1f}pp exceeds the allowance "
                    f"but does not survive Holm"
                    + (f" (p_adj={pa:.4f})" if pa is not None else "")
                    + f"; mde here is {mde:.1f}pp, so this n cannot resolve it."
                )
        elif mde > r.max_regression_pp:
            v = "INCONCLUSIVE"
            reason = (
                f"no regression observed, but mde {mde:.1f}pp is larger than "
                f"the {r.max_regression_pp:.1f}pp allowance — a real regression "
                f"of that size would not have been detected."
            )
        else:
            v = "PASS"
            reason = (
                f"holds floor {r.threshold:.3f} and is within the "
                f"{r.max_regression_pp:.1f}pp allowance; "
                f"mde {mde:.1f}pp is adequate."
            )

        verdicts.append(
            CategoryVerdict(
                name=r.name,
                verdict=v,
                rate=r.rate,
                baseline_rate=r.baseline_rate,
                delta_pp=r.delta_pp,
                mde_pp=mde,
                p_value=p,
                p_adjusted=pa,
                reason=reason,
            )
        )

    failed = tuple(c.name for c in verdicts if c.verdict == "FAIL")
    inconc = tuple(c.name for c in verdicts if c.verdict == "INCONCLUSIVE")
    overall: Verdict = "FAIL" if failed else ("INCONCLUSIVE" if inconc else "PASS")

    return GateDecision(
        verdict=overall,
        categories=tuple(verdicts),
        n_categories=k,
        alpha=alpha,
        uncorrected_fwer=family_wise_false_failure_rate(k, alpha),
        failed=failed,
        inconclusive=inconc,
    )
