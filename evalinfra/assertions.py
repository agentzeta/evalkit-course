"""Deterministic assertions, and the coverage they do not have.

Deterministic checks are the right first layer. Schema validation, required
fields, length limits, code that runs, SQL that executes against a test
database: cheap, fast, zero variance. The advice to do them first is correct.

What the advice omits is the thing that makes them dangerous.

**An assertion suite's pass rate is not a quality metric.** It is a pass rate
over the failure modes the assertions can see. A suite at 100% that addresses
three of your eight known failure categories has told you about three
categories, and it renders in a dashboard identically to a suite that covers
all eight. The green is the same green.

So this module makes you declare which failure category each assertion
covers, and refuses to summarise quality until coverage is complete:

    `SuiteResult.quality_estimate` returns None whenever any declared failure
    category has no assertion addressing it, and names the uncovered
    categories in the shortfall.

The taxonomy comes from error analysis on real traces — lab1 in this course.
Assertions written without one are assertions about what was easy to check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

__all__ = [
    "Assertion",
    "AssertionOutcome",
    "CaseResult",
    "SuiteResult",
    "AssertionSuite",
]


@dataclass(frozen=True)
class Assertion:
    """One deterministic check.

    `covers` names the failure category from your taxonomy that this check
    addresses. It is required. An assertion that does not map to a known
    failure mode is either checking something you have never seen go wrong,
    or your taxonomy is incomplete — and both are worth knowing before the
    suite goes green.
    """

    name: str
    covers: str
    check: Callable[[Any], bool]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("assertion needs a name")
        if not self.covers:
            raise ValueError(
                f"assertion {self.name!r} must declare the failure category "
                f"it covers; pass covers='<category>'"
            )


@dataclass(frozen=True)
class AssertionOutcome:
    name: str
    covers: str
    passed: bool
    errored: bool
    error: str | None = None

    @property
    def counted(self) -> bool:
        """An assertion that raised is not a pass and not a fail.

        Counting an exception as a failure manufactures findings; counting it
        as a pass hides them. It is a broken measurement and is reported as
        its own category.
        """
        return not self.errored


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    outcomes: tuple[AssertionOutcome, ...]

    @property
    def passed(self) -> bool:
        return all(o.passed for o in self.outcomes if o.counted)

    @property
    def errored(self) -> int:
        return sum(1 for o in self.outcomes if o.errored)


@dataclass(frozen=True)
class SuiteResult:
    n_cases: int
    n_assertions: int
    cases: tuple[CaseResult, ...]
    pass_rate: float
    per_category: dict[str, float]
    errored: int
    taxonomy: tuple[str, ...]
    covered: tuple[str, ...]
    uncovered: tuple[str, ...]
    coverage: float

    @property
    def quality_estimate(self) -> float | None:
        """The suite pass rate as a quality figure — or None.

        None whenever coverage is incomplete or any assertion errored. There
        is no partial credit and no best-effort number, because a number here
        is read as "the system is fine".
        """
        if self.uncovered or self.errored:
            return None
        return self.pass_rate

    def shortfall(self) -> str | None:
        if self.quality_estimate is not None:
            return None
        parts = []
        if self.uncovered:
            parts.append(
                f"{len(self.uncovered)} of {len(self.taxonomy)} failure "
                f"categories have no assertion: {', '.join(self.uncovered)}"
            )
        if self.errored:
            parts.append(
                f"{self.errored} assertion evaluations raised; those are "
                f"broken measurements, not results"
            )
        return "; ".join(parts)

    def summary(self) -> str:
        lines = [
            f"{self.n_assertions} assertions over {self.n_cases} cases",
            f"  raw pass rate   {self.pass_rate:.3f}",
            f"  taxonomy coverage {self.coverage:.3f} "
            f"({len(self.covered)}/{len(self.taxonomy)} categories)",
        ]
        for cat in self.taxonomy:
            if cat in self.per_category:
                lines.append(f"    {cat:<28} {self.per_category[cat]:.3f}")
            else:
                lines.append(f"    {cat:<28} NO ASSERTION")
        if self.errored:
            lines.append(f"  errored evaluations: {self.errored}")
        q = self.quality_estimate
        if q is None:
            lines.append(f"  quality estimate: NONE — {self.shortfall()}")
            lines.append(
                "  The pass rate above describes the categories that have "
                "assertions. It is not a statement about the system."
            )
        else:
            lines.append(f"  quality estimate: {q:.3f} (coverage complete)")
        return "\n".join(lines)


class AssertionSuite:
    """A deterministic assertion layer with explicit taxonomy coverage."""

    def __init__(self, taxonomy: Sequence[str]) -> None:
        if not taxonomy:
            raise ValueError(
                "a suite needs a failure taxonomy. Run error analysis on real "
                "traces first; assertions written without one check what was "
                "easy to check."
            )
        self.taxonomy: tuple[str, ...] = tuple(taxonomy)
        self._assertions: list[Assertion] = []

    def add(self, assertion: Assertion) -> "AssertionSuite":
        if assertion.covers not in self.taxonomy:
            raise ValueError(
                f"assertion {assertion.name!r} covers {assertion.covers!r}, "
                f"which is not in the taxonomy {self.taxonomy}. Either add the "
                f"category or fix the assertion — do not let a check float "
                f"free of the taxonomy."
            )
        self._assertions.append(assertion)
        return self

    def run(self, cases: Sequence[tuple[str, Any]]) -> SuiteResult:
        if not self._assertions:
            raise ValueError("no assertions registered")
        if not cases:
            raise ValueError("no cases supplied")

        results: list[CaseResult] = []
        by_cat: dict[str, list[int]] = {}
        errored = 0

        for case_id, payload in cases:
            outcomes: list[AssertionOutcome] = []
            for a in self._assertions:
                try:
                    ok = bool(a.check(payload))
                    outcomes.append(
                        AssertionOutcome(a.name, a.covers, ok, False, None)
                    )
                    by_cat.setdefault(a.covers, []).append(1 if ok else 0)
                except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                    errored += 1
                    outcomes.append(
                        AssertionOutcome(
                            a.name, a.covers, False, True, f"{type(exc).__name__}: {exc}"
                        )
                    )
            results.append(CaseResult(case_id, tuple(outcomes)))

        n_pass = sum(1 for r in results if r.passed)
        covered = tuple(c for c in self.taxonomy if c in by_cat)
        uncovered = tuple(c for c in self.taxonomy if c not in by_cat)

        return SuiteResult(
            n_cases=len(cases),
            n_assertions=len(self._assertions),
            cases=tuple(results),
            pass_rate=n_pass / len(cases),
            per_category={k: sum(v) / len(v) for k, v in by_cat.items()},
            errored=errored,
            taxonomy=self.taxonomy,
            covered=covered,
            uncovered=uncovered,
            coverage=len(covered) / len(self.taxonomy),
        )
