"""evalinfra — the eval-infrastructure layer.

Six modules covering the parts of a production eval stack that the standard
recipes name but do not supply, each built around the specific thing the
standard recipe leaves out.

    passk       pass@k vs pass^k, and why pass^k is far harder to CERTIFY
                than to compute. n=100 at 80% cannot certify pass^5 >= 0.30.

    pairwise    order-swapped judging — plus the cost of the standard fix:
                discarding inconsistent verdicts is a selection on outcome.

    trajectory  agent tool-path scoring, with the single-reference assumption
                made explicit and measurable.

    retrieval   recall@k's arithmetic ceiling, and the 2x2 that separates a
                retrieval failure from a generation failure. Includes the
                cell most dashboards cannot see: correct but unsupported.

    assertions  deterministic checks with taxonomy coverage accounting. A
                100%-passing suite covering three of eight failure modes is
                not a quality signal, and renders the same green.

    gate        per-category CI gating with the family-wise false-failure rate
                controlled, and a third verdict: INCONCLUSIVE is not a pass.

The design rule from the rest of the course holds throughout: **a function
that cannot honestly answer returns None and reports the shortfall. It never
relaxes a constraint to produce a number.**
"""

from __future__ import annotations

from . import assertions, gate, pairwise, passk, retrieval, trajectory

__all__ = [
    "assertions",
    "gate",
    "pairwise",
    "passk",
    "retrieval",
    "trajectory",
]

__version__ = "0.1.0"
