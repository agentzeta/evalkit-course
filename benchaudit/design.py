"""Design effect and effective sample size for clustered data.

In a model-generated benchmark the clustering unit is the generating prompt. In
a biological dataset it is the donor, the cell line, the batch, the animal, or
the imaging session — and it is almost never the row. A single-cell dataset with
500,000 cells drawn from 8 donors supports a donor-level claim at n ≈ 8, not
n = 500,000, and the gap between those two numbers is several orders of
magnitude of false confidence.

Biostatistics has had the right machinery for this for decades under the name
**pseudoreplication**, and the quantity is the **design effect**:

    D     = 1 + (m̄ − 1) · ICC
    n_eff = n / D

where `m̄` is the mean cluster size and ICC is the intra-cluster correlation.
The two limits are the whole story:

  ICC = 0  →  D = 1          every row is independent; n_eff = n
  ICC = 1  →  D = m̄          each cluster is one observation; n_eff = clusters

The module computes ICC from a one-way random-effects decomposition, reports it
with the mean-squares it came from, and refuses to return a figure when the
design cannot support one — a single cluster, or clusters of size one, carry no
information about between-cluster variance and the honest answer is "unknown",
not 0.

Canonical references for the underlying problem are Lazic on pseudoreplication
in neuroscience and Leek et al. on batch effects in high-throughput data.
*Not re-verified in this session; cited as the standard literature.*
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class DesignEffect:
    n_rows: int
    n_clusters: int
    mean_cluster: float
    icc: float
    design_effect: float
    effective_n: float
    ms_between: float
    ms_within: float
    balanced: bool
    reason: str = ""

    @property
    def estimable(self) -> bool:
        return not math.isnan(self.icc)

    @property
    def inflation(self) -> float:
        """How much too narrow a naive confidence interval is."""
        return math.sqrt(self.design_effect) if self.design_effect >= 1 else 1.0

    def sentence(self) -> str:
        if not self.estimable:
            return (
                f"Effective n is not estimable: {self.reason} "
                f"({self.n_rows} rows in {self.n_clusters} clusters)."
            )
        return (
            f"{self.n_rows} rows in {self.n_clusters} clusters, mean size "
            f"{self.mean_cluster:.1f}. ICC = {self.icc:.3f}, design effect "
            f"{self.design_effect:.2f}, so the effective sample size is "
            f"{self.effective_n:.0f} and a confidence interval computed on "
            f"{self.n_rows} rows is too narrow by a factor of "
            f"{self.inflation:.2f}."
        )


def design_effect(values: Sequence[float], groups: Sequence[Any]) -> DesignEffect:
    """ICC, design effect and effective n from a one-way random-effects split.

    `values` is the measurement whose precision is in question — a per-row score,
    an expression level, a binary label. `groups` is the clustering unit: donor,
    batch, cell line, generating prompt.

    The ICC estimate is the standard ANOVA one and **can come out negative**,
    which happens when between-cluster variance is smaller than within. A
    negative ICC is not a real negative correlation; it is an estimate of zero
    with noise on it, so it is clamped to 0 and the design effect becomes 1.
    Clamping is reported rather than silent, because a clamped ICC means the data
    are consistent with no clustering at all and the effective-n figure is then
    an upper bound.
    """
    if len(values) != len(groups):
        raise ValueError(f"{len(values)} values but {len(groups)} groups")
    n = len(values)
    if n == 0:
        return _unestimable(0, 0, 0.0, "no rows")

    by: dict[Any, list[float]] = {}
    for v, g in zip(values, groups):
        if isinstance(v, float) and math.isnan(v):
            continue
        by.setdefault(g, []).append(float(v))
    if not by:
        return _unestimable(n, 0, 0.0, "every value is missing")

    sizes = [len(v) for v in by.values()]
    k = len(by)
    n_used = sum(sizes)
    mbar = n_used / k

    if k < 2:
        return _unestimable(
            n_used, k, mbar,
            "a single cluster carries no information about between-cluster "
            "variance",
        )
    if max(sizes) < 2:
        return _unestimable(
            n_used, k, mbar,
            "every cluster has one member, so within-cluster variance cannot be "
            "estimated",
        )

    grand = sum(sum(v) for v in by.values()) / n_used
    means = {g: sum(v) / len(v) for g, v in by.items()}

    ss_between = sum(len(v) * (means[g] - grand) ** 2 for g, v in by.items())
    ss_within = sum((x - means[g]) ** 2 for g, v in by.items() for x in v)
    df_between, df_within = k - 1, n_used - k
    if df_within <= 0:
        return _unestimable(n_used, k, mbar, "no within-cluster degrees of freedom")

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    balanced = min(sizes) == max(sizes)
    # For unbalanced designs the standard ANOVA estimator uses m0 rather than
    # the plain mean cluster size; using m̄ there biases the ICC.
    m0 = (
        mbar
        if balanced
        else (n_used - sum(s * s for s in sizes) / n_used) / (k - 1)
    )

    if ms_within == 0.0 and ms_between == 0.0:
        return _unestimable(
            n_used, k, mbar, "the measurement is constant, so it has no variance"
        )

    denom = ms_between + (m0 - 1) * ms_within
    if denom == 0.0:
        return _unestimable(n_used, k, mbar, "variance decomposition is degenerate")

    icc_raw = (ms_between - ms_within) / denom
    icc = max(0.0, min(1.0, icc_raw))
    d = 1.0 + (mbar - 1.0) * icc

    return DesignEffect(
        n_rows=n_used,
        n_clusters=k,
        mean_cluster=mbar,
        icc=icc,
        design_effect=d,
        effective_n=n_used / d if d > 0 else float(k),
        ms_between=ms_between,
        ms_within=ms_within,
        balanced=balanced,
        reason=(
            f"ICC estimate was {icc_raw:.3f} and has been clamped to 0; the data "
            f"are consistent with no clustering, so the effective-n figure is an "
            f"upper bound"
            if icc_raw < 0
            else ""
        ),
    )


def _unestimable(
    n: int, k: int, mbar: float, reason: str
) -> DesignEffect:
    return DesignEffect(
        n_rows=n,
        n_clusters=k,
        mean_cluster=mbar,
        icc=float("nan"),
        design_effect=float("nan"),
        effective_n=float("nan"),
        ms_between=float("nan"),
        ms_within=float("nan"),
        balanced=False,
        reason=reason,
    )


def n_for_power(effective_n_needed: float, mean_cluster: float, icc: float) -> int:
    """Rows needed to reach a target effective n at this clustering.

    The arithmetic that makes the problem concrete when planning a study: if ICC
    is 0.3 and you will collect 50 cells per donor, each additional *row* buys
    about 1/15th of an independent observation, and the only way to buy
    independent observations is more donors.
    """
    if mean_cluster <= 0 or effective_n_needed <= 0:
        return 0
    d = 1.0 + (mean_cluster - 1.0) * max(0.0, min(1.0, icc))
    return math.ceil(effective_n_needed * d)


def clusters_for_power(effective_n_needed: float, mean_cluster: float, icc: float) -> int:
    """Clusters (donors, batches, prompts) needed to reach a target effective n."""
    rows = n_for_power(effective_n_needed, mean_cluster, icc)
    return math.ceil(rows / mean_cluster) if mean_cluster > 0 else 0
