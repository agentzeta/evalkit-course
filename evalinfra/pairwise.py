"""Pairwise judging — position bias, length confound, and the cost of the
standard fix.

The standard advice is: run every A/B comparison twice with the order
swapped, and count only the verdicts that agree. That advice is right about
the disease and quiet about the side effect.

**Discarding inconsistent verdicts is a selection on outcome.** The pairs a
judge flips on are the pairs where the two answers are close. Throwing them
away leaves the easy comparisons, so the surviving win rate is a win rate on
an easier population than the one you sampled — and it is reported as though
it were the whole thing.

So this module computes three estimates, not one:

    consistent-only   the standard figure; biased toward clear-cut pairs
    ties-as-ties      inconsistent pairs counted as ties; conservative
    discard rate      how much of your data the first estimate threw away

If the discard rate is high, the first two numbers diverge and the
consistent-only figure should not be quoted on its own. `assess` refuses to
pick one for you.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

__all__ = [
    "Comparison",
    "PositionBias",
    "position_bias",
    "LengthConfound",
    "length_confound",
    "PairwiseReport",
    "assess",
]

Winner = Literal["a", "b", "tie"]


@dataclass(frozen=True)
class Comparison:
    """One pair judged in both orders.

    `first_shown` records which item the judge saw first in each of the two
    presentations, so a judge that always prefers position one is detectable.

    forward:  a shown first, verdict `forward`
    reverse:  b shown first, verdict `reverse`

    Verdicts are in terms of the ITEMS (a/b/tie), not positions. Normalising
    to items at the recording site is deliberate: position arithmetic done
    later is where sign errors live.
    """

    pair_id: str
    forward: Winner
    reverse: Winner
    len_a: int | None = None
    len_b: int | None = None

    @property
    def consistent(self) -> bool:
        return self.forward == self.reverse

    @property
    def agreed_winner(self) -> Winner | None:
        return self.forward if self.consistent else None


# --------------------------------------------------------------------------
# position bias
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PositionBias:
    n_pairs: int
    first_position_wins: int
    decisive: int
    rate: float | None
    p_value: float | None
    note: str

    def summary(self) -> str:
        if self.rate is None:
            return f"position bias: {self.note}"
        return (
            f"position bias: first-shown won {self.first_position_wins}/"
            f"{self.decisive} decisive presentations "
            f"({self.rate:.3f}), two-sided p={self.p_value:.4f}. {self.note}"
        )


def _binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value by the method of small likelihoods."""
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    obs = probs[k]
    # include every outcome at least as unlikely as the observed one
    tol = obs * (1 + 1e-9)
    return min(1.0, sum(pr for pr in probs if pr <= tol))


def position_bias(comparisons: Sequence[Comparison]) -> PositionBias:
    """Does the judge prefer whatever it sees first?

    Each pair contributes two presentations. A presentation is decisive when
    its verdict is not a tie. We count how often the item shown first won.
    Under no position bias this is a fair coin.
    """
    first_wins = 0
    decisive = 0
    for c in comparisons:
        # forward presentation: a was first
        if c.forward != "tie":
            decisive += 1
            if c.forward == "a":
                first_wins += 1
        # reverse presentation: b was first
        if c.reverse != "tie":
            decisive += 1
            if c.reverse == "b":
                first_wins += 1
    if decisive == 0:
        return PositionBias(
            n_pairs=len(comparisons),
            first_position_wins=0,
            decisive=0,
            rate=None,
            p_value=None,
            note="no decisive presentations; every verdict was a tie.",
        )
    rate = first_wins / decisive
    p = _binom_two_sided(first_wins, decisive)
    if p < 0.05 and rate > 0.5:
        note = "FIRST-POSITION PREFERENCE. Order randomisation is not enough; the bias must be reported."
    elif p < 0.05:
        note = "LAST-POSITION PREFERENCE. Unusual, and worth checking the recording convention before believing it."
    else:
        note = "no detectable position preference at this sample size."
    return PositionBias(
        n_pairs=len(comparisons),
        first_position_wins=first_wins,
        decisive=decisive,
        rate=rate,
        p_value=p,
        note=note,
    )


# --------------------------------------------------------------------------
# length confound
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LengthConfound:
    n_usable: int
    longer_won: int
    rate: float | None
    p_value: float | None
    stratified: dict[str, float]
    note: str

    def summary(self) -> str:
        if self.rate is None:
            return f"length confound: {self.note}"
        lines = [
            f"length confound: the longer answer won {self.longer_won}/"
            f"{self.n_usable} decisive consistent pairs "
            f"({self.rate:.3f}), p={self.p_value:.4f}. {self.note}"
        ]
        for band, r in sorted(self.stratified.items()):
            lines.append(f"    {band:>18}  win rate for A: {r:.3f}")
        return "\n".join(lines)


def length_confound(
    comparisons: Sequence[Comparison],
    near_tie_frac: float = 0.10,
) -> LengthConfound:
    """Is the judge rewarding verbosity?

    Uses only consistent, decisive pairs with both lengths recorded. Pairs
    whose lengths are within `near_tie_frac` of each other are excluded from
    the headline rate — they carry no length signal — but they are kept for
    the stratified view, which is what shows whether A wins only when A is
    longer.
    """
    usable = [
        c
        for c in comparisons
        if c.consistent
        and c.agreed_winner in ("a", "b")
        and c.len_a is not None
        and c.len_b is not None
    ]
    if not usable:
        return LengthConfound(
            0, 0, None, None, {},
            "no consistent decisive pairs with lengths recorded.",
        )

    strat: dict[str, list[int]] = {
        "A much longer": [],
        "A longer": [],
        "similar length": [],
        "B longer": [],
        "B much longer": [],
    }
    longer_won = 0
    counted = 0
    for c in usable:
        la, lb = c.len_a, c.len_b
        denom = max(1, min(la, lb))
        ratio = (la - lb) / denom
        if ratio > 0.50:
            band = "A much longer"
        elif ratio > near_tie_frac:
            band = "A longer"
        elif ratio < -0.50:
            band = "B much longer"
        elif ratio < -near_tie_frac:
            band = "B longer"
        else:
            band = "similar length"
        strat[band].append(1 if c.agreed_winner == "a" else 0)

        if band != "similar length":
            counted += 1
            a_is_longer = la > lb
            if (a_is_longer and c.agreed_winner == "a") or (
                not a_is_longer and c.agreed_winner == "b"
            ):
                longer_won += 1

    if counted == 0:
        return LengthConfound(
            len(usable), 0, None, None,
            {k: sum(v) / len(v) for k, v in strat.items() if v},
            "every usable pair had near-identical lengths; no length signal.",
        )
    rate = longer_won / counted
    p = _binom_two_sided(longer_won, counted)
    if p < 0.05 and rate > 0.5:
        note = "VERBOSITY PREFERENCE. Cap or normalise length before reading the win rate as quality."
    elif p < 0.05:
        note = "BREVITY PREFERENCE."
    else:
        note = "no detectable length preference at this sample size."
    return LengthConfound(
        n_usable=counted,
        longer_won=longer_won,
        rate=rate,
        p_value=p,
        stratified={k: sum(v) / len(v) for k, v in strat.items() if v},
        note=note,
    )


# --------------------------------------------------------------------------
# the three estimates
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PairwiseReport:
    n_pairs: int
    n_consistent: int
    discard_rate: float
    a_win_consistent_only: float | None
    a_win_ties_as_ties: float
    divergence: float | None
    bias: PositionBias
    length: LengthConfound
    quotable_single_number: float | None
    note: str

    def summary(self) -> str:
        lines = [
            f"pairs judged in both orders: {self.n_pairs}",
            f"  consistent: {self.n_consistent}  "
            f"discarded: {self.discard_rate:.1%}",
        ]
        if self.a_win_consistent_only is None:
            lines.append("  consistent-only A win rate: undefined (no consistent decisive pairs)")
        else:
            lines.append(
                f"  A win rate, consistent only : "
                f"{self.a_win_consistent_only:.3f}   <- the standard figure"
            )
        lines.append(
            f"  A win rate, ties as ties    : "
            f"{self.a_win_ties_as_ties:.3f}   <- conservative"
        )
        if self.divergence is not None:
            lines.append(f"  divergence between them     : {self.divergence:.3f}")
        lines.append("  " + self.bias.summary())
        lines.append("  " + self.length.summary())
        lines.append(f"  VERDICT: {self.note}")
        return "\n".join(lines)


def assess(
    comparisons: Sequence[Comparison],
    max_discard_rate: float = 0.25,
    max_divergence: float = 0.05,
) -> PairwiseReport:
    """Score a set of order-swapped pairwise comparisons, honestly.

    `quotable_single_number` is None unless the discard rate is acceptable
    AND the two estimates agree AND neither bias check fired. There is no
    fallback to the convenient figure: if the judge is biased or the discard
    rate is high, the correct output is two numbers and a caveat, and this
    function will give you exactly that.
    """
    n = len(comparisons)
    if n == 0:
        raise ValueError("no comparisons supplied")

    consistent = [c for c in comparisons if c.consistent]
    n_cons = len(consistent)
    discard_rate = 1.0 - n_cons / n

    cons_decisive = [c for c in consistent if c.agreed_winner in ("a", "b")]
    if cons_decisive:
        a_cons = sum(1 for c in cons_decisive if c.agreed_winner == "a") / len(
            cons_decisive
        )
    else:
        a_cons = None

    # ties-as-ties: inconsistent pairs are genuine ties, scored 0.5 each
    score = 0.0
    for c in comparisons:
        if not c.consistent:
            score += 0.5
        elif c.agreed_winner == "a":
            score += 1.0
        elif c.agreed_winner == "tie":
            score += 0.5
    a_ties = score / n

    divergence = None if a_cons is None else abs(a_cons - a_ties)
    bias = position_bias(comparisons)
    length = length_confound(comparisons)

    problems: list[str] = []
    if discard_rate > max_discard_rate:
        problems.append(
            f"discard rate {discard_rate:.1%} exceeds {max_discard_rate:.0%} — "
            f"the consistent-only figure describes an easier subset than you sampled"
        )
    if divergence is not None and divergence > max_divergence:
        problems.append(
            f"the two estimates differ by {divergence:.3f}, more than "
            f"{max_divergence:.3f}"
        )
    if bias.p_value is not None and bias.p_value < 0.05:
        problems.append("position bias is detectable")
    if length.p_value is not None and length.p_value < 0.05:
        problems.append("a length preference is detectable")
    if a_cons is None:
        problems.append("no consistent decisive pairs")

    if problems:
        quotable = None
        note = "NO SINGLE NUMBER IS QUOTABLE. " + "; ".join(problems) + "."
    else:
        quotable = a_cons
        note = (
            f"both estimates agree and no bias fired; "
            f"A win rate {a_cons:.3f} is quotable."
        )

    return PairwiseReport(
        n_pairs=n,
        n_consistent=n_cons,
        discard_rate=discard_rate,
        a_win_consistent_only=a_cons,
        a_win_ties_as_ties=a_ties,
        divergence=divergence,
        bias=bias,
        length=length,
        quotable_single_number=quotable,
        note=note,
    )
