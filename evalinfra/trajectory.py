"""Agent trajectory scoring — and the reference-path problem.

The standard recipe scores an agent's tool calls against a reference path:
precision, recall, argument correctness, step count, error recovery. The
recipe works and it carries an assumption that is almost always false.

**It assumes the reference path is the only correct one.** Most real tasks
have several valid orderings and several valid tool choices. Scored against a
single reference, an agent that does the job correctly by another route is
marked down, and the metric reports a capability failure where there was a
style difference.

So this module requires you to be explicit:

    ordered=True    the sequence matters (a migration, a checkout)
    ordered=False   the set matters (three independent lookups)

and lets you supply SEVERAL references, scoring against the best match while
reporting the spread across them. A wide spread means your metric is
reference-dependent and the number should not be quoted without saying which
reference produced it.

`score` returns None when there is no reference to score against. It will not
invent one from the agent's own behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

__all__ = [
    "Call",
    "Trajectory",
    "TrajectoryScore",
    "score",
    "ReferenceSpread",
    "score_against_any",
]


@dataclass(frozen=True)
class Call:
    """One tool call in a trajectory."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    tokens: int = 0

    def key(self) -> str:
        return self.tool


@dataclass(frozen=True)
class Trajectory:
    calls: tuple[Call, ...]
    final_correct: bool | None = None

    @classmethod
    def of(cls, calls: Sequence[Call], final_correct: bool | None = None) -> "Trajectory":
        return cls(tuple(calls), final_correct)

    @property
    def n_steps(self) -> int:
        return len(self.calls)

    @property
    def n_errors(self) -> int:
        return sum(1 for c in self.calls if not c.ok)

    @property
    def total_tokens(self) -> int:
        return sum(c.tokens for c in self.calls)


# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TrajectoryScore:
    ordered: bool
    tool_precision: float
    tool_recall: float
    tool_f1: float
    arg_correctness: float | None
    n_matched: int
    step_ratio: float
    errors: int
    recovered: int
    unrecovered: int
    recovery_rate: float | None
    final_correct: bool | None
    efficiency_flag: str | None

    def summary(self) -> str:
        lines = [
            f"{'ordered' if self.ordered else 'unordered'} scoring",
            f"  tool precision {self.tool_precision:.3f}  "
            f"recall {self.tool_recall:.3f}  f1 {self.tool_f1:.3f}",
        ]
        if self.arg_correctness is None:
            lines.append("  argument correctness: undefined (no matched calls)")
        else:
            lines.append(
                f"  argument correctness {self.arg_correctness:.3f} "
                f"over {self.n_matched} matched calls"
            )
        lines.append(
            f"  step ratio {self.step_ratio:.2f} "
            f"(agent steps / reference steps)"
        )
        if self.errors:
            rr = (
                "n/a"
                if self.recovery_rate is None
                else f"{self.recovery_rate:.3f}"
            )
            lines.append(
                f"  errors {self.errors}: recovered {self.recovered}, "
                f"unrecovered {self.unrecovered}, rate {rr}"
            )
        if self.final_correct is not None:
            lines.append(f"  final answer correct: {self.final_correct}")
        if self.efficiency_flag:
            lines.append(f"  {self.efficiency_flag}")
        return "\n".join(lines)


def _multiset_match(
    got: Sequence[Call], want: Sequence[Call]
) -> list[tuple[int, int]]:
    """Greedy multiset matching on tool name. Returns (got_idx, want_idx)."""
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for gi, g in enumerate(got):
        for wi, w in enumerate(want):
            if wi in used:
                continue
            if g.key() == w.key():
                used.add(wi)
                pairs.append((gi, wi))
                break
    return pairs


def _lcs_match(got: Sequence[Call], want: Sequence[Call]) -> list[tuple[int, int]]:
    """Longest common subsequence on tool name — order-respecting matching."""
    n, m = len(got), len(want)
    tbl = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if got[i].key() == want[j].key():
                tbl[i][j] = 1 + tbl[i + 1][j + 1]
            else:
                tbl[i][j] = max(tbl[i + 1][j], tbl[i][j + 1])
    pairs: list[tuple[int, int]] = []
    i = j = 0
    while i < n and j < m:
        if got[i].key() == want[j].key():
            pairs.append((i, j))
            i += 1
            j += 1
        elif tbl[i + 1][j] >= tbl[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def _args_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a == b


def score(
    agent: Trajectory,
    reference: Trajectory | None,
    ordered: bool = True,
    step_ratio_flag: float = 2.0,
) -> TrajectoryScore | None:
    """Score one trajectory against one reference.

    Returns None when `reference` is None. A trajectory cannot be scored
    against itself and this function will not pretend otherwise.
    """
    if reference is None:
        return None
    if not reference.calls:
        return None

    got, want = agent.calls, reference.calls
    pairs = _lcs_match(got, want) if ordered else _multiset_match(got, want)
    matched = len(pairs)

    precision = matched / len(got) if got else 0.0
    recall = matched / len(want)
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )

    if matched:
        n_args_ok = sum(
            1 for gi, wi in pairs if _args_equal(got[gi].args, want[wi].args)
        )
        arg_corr: float | None = n_args_ok / matched
    else:
        arg_corr = None

    step_ratio = len(got) / len(want)

    # error recovery: an error is "recovered" if any later call succeeds
    recovered = unrecovered = 0
    for idx, c in enumerate(got):
        if c.ok:
            continue
        if any(later.ok for later in got[idx + 1 :]):
            recovered += 1
        else:
            unrecovered += 1
    errors = recovered + unrecovered
    recovery_rate = None if errors == 0 else recovered / errors

    flag = None
    if step_ratio >= step_ratio_flag:
        flag = (
            f"EFFICIENCY: {len(got)} calls where {len(want)} suffice "
            f"({step_ratio:.1f}x). Cost and latency bug even if the answer is right."
        )

    return TrajectoryScore(
        ordered=ordered,
        tool_precision=precision,
        tool_recall=recall,
        tool_f1=f1,
        arg_correctness=arg_corr,
        n_matched=matched,
        step_ratio=step_ratio,
        errors=errors,
        recovered=recovered,
        unrecovered=unrecovered,
        recovery_rate=recovery_rate,
        final_correct=agent.final_correct,
        efficiency_flag=flag,
    )


# --------------------------------------------------------------------------
# the reference-dependence check
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ReferenceSpread:
    best: TrajectoryScore
    best_index: int
    all_f1: tuple[float, ...]
    spread: float
    note: str

    def summary(self) -> str:
        return "\n".join(
            [
                f"scored against {len(self.all_f1)} references; "
                f"best is #{self.best_index} at f1 {self.best.tool_f1:.3f}",
                f"  f1 across references: "
                f"{', '.join(f'{v:.3f}' for v in self.all_f1)}",
                f"  spread {self.spread:.3f}",
                f"  {self.note}",
            ]
        )


def score_against_any(
    agent: Trajectory,
    references: Sequence[Trajectory],
    ordered: bool = True,
    max_spread: float = 0.15,
) -> ReferenceSpread | None:
    """Score against several valid references, keeping the best.

    The `spread` is the point. If the same trajectory scores 0.40 against one
    valid reference and 1.00 against another, the metric is measuring your
    choice of reference as much as the agent, and a single-reference harness
    would have reported one of those two numbers with no indication that the
    other existed.
    """
    usable = [r for r in references if r.calls]
    if not usable:
        return None
    scores = [score(agent, r, ordered=ordered) for r in usable]
    scored = [(i, s) for i, s in enumerate(scores) if s is not None]
    if not scored:
        return None
    f1s = tuple(s.tool_f1 for _, s in scored)
    best_i, best_s = max(scored, key=lambda t: t[1].tool_f1)
    spread = max(f1s) - min(f1s)
    if len(f1s) == 1:
        note = "only one reference supplied; reference dependence is unmeasured."
    elif spread > max_spread:
        note = (
            f"REFERENCE-DEPENDENT: spread {spread:.3f} exceeds {max_spread:.3f}. "
            f"Report which reference was used, or score set-wise instead."
        )
    else:
        note = "references agree; the score is not an artefact of reference choice."
    return ReferenceSpread(
        best=best_s,
        best_index=best_i,
        all_f1=f1s,
        spread=spread,
        note=note,
    )
