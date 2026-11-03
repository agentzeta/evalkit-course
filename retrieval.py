"""Retrieval metrics, their ceilings, and the joint accounting with generation.

Two things go wrong in practice and both are arithmetic rather than modelling.

**1. Recall@k has a ceiling you did not choose.** With R relevant documents
and a cutoff k, recall@k can never exceed k/R. A system reporting recall@5 =
0.25 against 20 relevant documents is *perfect* — it retrieved every
document it had room for — and it will be read as a 75% failure. This module
reports attainment against the ceiling alongside the raw figure. The same
applies to nDCG@k, whose ideal DCG is computed over at most k documents.

**2. Scoring the final answer cannot tell you which half broke.** Faithful
generation over bad retrieval produces a confidently wrong answer; unfaithful
generation over good retrieval produces a right answer for the wrong reason,
which will not survive a change of corpus. The 2x2 in `decompose` separates
them, and it has a cell most dashboards cannot see: **correct but
unsupported**, the answer the model knew without being told. Those cases pass
an end-to-end eval and are the ones that break when you swap the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "recall_ceiling",
    "recall_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "mrr",
    "dcg",
    "ndcg_at_k",
    "RetrievalReport",
    "assess",
    "Decomposition",
    "decompose",
]


# --------------------------------------------------------------------------
# ceilings
# --------------------------------------------------------------------------

def recall_ceiling(k: int, n_relevant: int) -> float:
    """The largest recall@k attainable. min(1, k / n_relevant)."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if n_relevant < 0:
        raise ValueError(f"n_relevant must be >= 0, got {n_relevant}")
    if n_relevant == 0:
        return 0.0
    return min(1.0, k / n_relevant)


# --------------------------------------------------------------------------
# point metrics
# --------------------------------------------------------------------------

def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    hit = sum(1 for d in retrieved[:k] if d in rel)
    return hit / len(rel)


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    rel = set(relevant)
    window = retrieved[:k]
    if not window:
        return 0.0
    return sum(1 for d in window if d in rel) / len(window)


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    rel = set(relevant)
    for i, d in enumerate(retrieved, start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def mrr(
    retrieved_per_query: Sequence[Sequence[str]],
    relevant_per_query: Sequence[Sequence[str]],
) -> float:
    if len(retrieved_per_query) != len(relevant_per_query):
        raise ValueError("retrieved and relevant must be the same length")
    if not retrieved_per_query:
        raise ValueError("no queries supplied")
    return sum(
        reciprocal_rank(r, g)
        for r, g in zip(retrieved_per_query, relevant_per_query)
    ) / len(retrieved_per_query)


def dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain with the standard log2(i+1) discount."""
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(
    retrieved: Sequence[str],
    relevance: dict[str, float],
    k: int,
) -> float:
    """nDCG@k with graded relevance.

    The ideal DCG is computed over at most k documents, which is what makes
    nDCG@k reach 1.0 for a system that ranked everything it could fit
    perfectly — the behaviour people expect and which a naive implementation
    (ideal over ALL relevant docs) silently breaks.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    gains = [relevance.get(d, 0.0) for d in retrieved[:k]]
    ideal = sorted((g for g in relevance.values() if g > 0), reverse=True)[:k]
    idcg = dcg(ideal)
    if idcg == 0.0:
        return 0.0
    return dcg(gains) / idcg


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalReport:
    k: int
    n_queries: int
    recall: float
    ceiling: float
    attainment: float | None
    precision: float
    mrr: float
    ndcg: float | None
    note: str

    def summary(self) -> str:
        lines = [
            f"retrieval @{self.k} over {self.n_queries} queries",
            f"  recall@{self.k}    {self.recall:.3f}   "
            f"ceiling {self.ceiling:.3f}",
        ]
        if self.attainment is None:
            lines.append("  attainment    undefined (ceiling is zero)")
        else:
            lines.append(
                f"  attainment    {self.attainment:.3f}   "
                f"<- read THIS, not recall, when k < relevant count"
            )
        lines.append(f"  precision@{self.k} {self.precision:.3f}")
        lines.append(f"  MRR           {self.mrr:.3f}")
        if self.ndcg is not None:
            lines.append(f"  nDCG@{self.k}      {self.ndcg:.3f}")
        lines.append(f"  {self.note}")
        return "\n".join(lines)


def assess(
    retrieved_per_query: Sequence[Sequence[str]],
    relevant_per_query: Sequence[Sequence[str]],
    k: int,
    relevance_per_query: Sequence[dict[str, float]] | None = None,
) -> RetrievalReport:
    """Retrieval metrics with the ceiling made explicit."""
    if len(retrieved_per_query) != len(relevant_per_query):
        raise ValueError("retrieved and relevant must be the same length")
    n = len(retrieved_per_query)
    if n == 0:
        raise ValueError("no queries supplied")

    recalls = [
        recall_at_k(r, g, k) for r, g in zip(retrieved_per_query, relevant_per_query)
    ]
    ceilings = [recall_ceiling(k, len(set(g))) for g in relevant_per_query]
    precisions = [
        precision_at_k(r, g, k)
        for r, g in zip(retrieved_per_query, relevant_per_query)
    ]

    mean_recall = sum(recalls) / n
    mean_ceiling = sum(ceilings) / n
    attainment = None if mean_ceiling == 0 else mean_recall / mean_ceiling

    ndcg_val = None
    if relevance_per_query is not None:
        if len(relevance_per_query) != n:
            raise ValueError("relevance_per_query must match the query count")
        ndcg_val = sum(
            ndcg_at_k(r, rel, k)
            for r, rel in zip(retrieved_per_query, relevance_per_query)
        ) / n

    capped = sum(1 for c in ceilings if c < 1.0)
    if capped == 0:
        note = "k is at least the relevant count for every query; recall is unconstrained."
    else:
        note = (
            f"CEILING BINDING on {capped}/{n} queries (k < relevant count). "
            f"Raw recall understates performance; quote attainment."
        )
    return RetrievalReport(
        k=k,
        n_queries=n,
        recall=mean_recall,
        ceiling=mean_ceiling,
        attainment=attainment,
        precision=sum(precisions) / n,
        mrr=mrr(retrieved_per_query, relevant_per_query),
        ndcg=ndcg_val,
        note=note,
    )


# --------------------------------------------------------------------------
# joint accounting
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Decomposition:
    n: int
    supported_correct: int
    supported_incorrect: int
    unsupported_correct: int
    unsupported_incorrect: int
    end_to_end_accuracy: float
    retrieval_sufficiency: float
    faithfulness: float | None
    note: str

    def summary(self) -> str:
        return "\n".join(
            [
                f"joint retrieval/generation accounting over {self.n} cases",
                "",
                "                   answer correct   answer wrong",
                f"  evidence present      {self.supported_correct:>6}"
                f"           {self.supported_incorrect:>6}",
                f"  evidence absent       {self.unsupported_correct:>6}"
                f"           {self.unsupported_incorrect:>6}",
                "",
                f"  end-to-end accuracy     {self.end_to_end_accuracy:.3f}"
                f"   <- the only number most dashboards show",
                f"  retrieval sufficiency   {self.retrieval_sufficiency:.3f}"
                f"   <- fraction where the evidence was there to find",
                (
                    "  faithfulness            undefined"
                    " (no cases with evidence present)"
                    if self.faithfulness is None
                    else f"  faithfulness            {self.faithfulness:.3f}"
                    "   <- correct GIVEN the evidence was retrieved"
                ),
                "",
                f"  {self.note}",
            ]
        )


def decompose(
    evidence_retrieved: Sequence[bool],
    answer_correct: Sequence[bool],
) -> Decomposition:
    """Split end-to-end accuracy into a retrieval half and a generation half.

    The interesting cell is **unsupported_correct**: the answer was right and
    the supporting evidence was never retrieved. Those cases are the model
    answering from parametric knowledge. They inflate end-to-end accuracy,
    they are invisible to any metric that only scores the final answer, and
    they are the first thing to break when the model is swapped — which is
    exactly when everyone trusts the eval suite most.
    """
    if len(evidence_retrieved) != len(answer_correct):
        raise ValueError("inputs must be the same length")
    n = len(evidence_retrieved)
    if n == 0:
        raise ValueError("no cases supplied")

    sc = si = uc = ui = 0
    for ev, ok in zip(evidence_retrieved, answer_correct):
        if ev and ok:
            sc += 1
        elif ev and not ok:
            si += 1
        elif not ev and ok:
            uc += 1
        else:
            ui += 1

    e2e = (sc + uc) / n
    suff = (sc + si) / n
    faith = None if (sc + si) == 0 else sc / (sc + si)

    frac_unsupported_correct = uc / n
    if frac_unsupported_correct >= 0.10:
        note = (
            f"WARNING: {frac_unsupported_correct:.1%} of cases are correct with "
            f"NO retrieved evidence. End-to-end accuracy {e2e:.3f} is carrying "
            f"parametric knowledge, not retrieval. Expect this to move on a "
            f"model change."
        )
    elif faith is not None and faith < suff:
        note = (
            "retrieval is finding evidence the generator is not using; "
            "generation is the weaker half."
        )
    else:
        note = "no large parametric-knowledge component detected."
    return Decomposition(
        n=n,
        supported_correct=sc,
        supported_incorrect=si,
        unsupported_correct=uc,
        unsupported_incorrect=ui,
        end_to_end_accuracy=e2e,
        retrieval_sufficiency=suff,
        faithfulness=faith,
        note=note,
    )
