"""The benchmark integrity audit.

Runs the checks that a benchmark author should run before release, and that a
reader should run before believing a result. Everything here is computed from
the benchmark alone: no model is called, no network is touched.

The checks, in the order they are reported:

  1. **Multiplicity** — how many genuinely distinct items are there, exactly
     and under near-duplicate clustering, and what is the effective sample
     size that follows.
  2. **Degenerate baselines** — can a solver that does none of the task beat
     the trivial baseline? Four probes: flag-everything, out-of-fold item
     identity, length, and the single most label-correlated token.
  3. **Ceiling violation** — if the authors claim an irreducible bound, does
     any task-blind probe exceed it? A violation is a proof, not a hint.
  4. **Protocol gap** — the same model under row-wise and group-wise splitting.
     The difference is what row-wise splitting was concealing.

The audit **degrades rather than refuses**. Without labels it still reports
multiplicity and effective n, which is the finding that applies to every
generated benchmark regardless of task.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit.leakage import (  # noqa: E402
    degenerate_baselines,
    grouping_gap,
    naive_bayes_fit_predict,
    template_multiplicity,
)

from .dedup import DuplicateReport, cluster_labels, effective_n_curve, find_duplicates  # noqa: E402
from .design import DesignEffect, design_effect  # noqa: E402
from .loader import Benchmark  # noqa: E402

CRITICAL, WARNING, INFO, OK = "CRITICAL", "WARNING", "INFO", "OK"
_ORDER = {CRITICAL: 0, WARNING: 1, INFO: 2, OK: 3}


def _median_words(texts: Sequence[str]) -> int:
    if not texts:
        return 0
    lens = sorted(len(t.split()) for t in texts)
    return lens[len(lens) // 2]


@dataclass
class Finding:
    severity: str
    check: str
    headline: str
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.severity}] {self.check}: {self.headline}"


@dataclass
class AuditResult:
    benchmark: str
    n_rows: int
    n_audited: int
    findings: list[Finding] = field(default_factory=list)
    effective_n: int = 0
    effective_n_range: tuple[int, int] = (0, 0)
    effective_n_curve: list[tuple[float, int, float]] = field(default_factory=list)
    representation: str = ""
    duplicates: DuplicateReport | None = None
    design: DesignEffect | None = None
    effective_n_source: str = "near-duplicate clustering"
    notes: list[str] = field(default_factory=list)

    def by_severity(self, sev: str) -> list[Finding]:
        return [f for f in self.findings if f.severity == sev]

    @property
    def worst(self) -> str:
        return min((f.severity for f in self.findings), key=lambda s: _ORDER[s], default=OK)

    def sorted_findings(self) -> list[Finding]:
        return sorted(self.findings, key=lambda f: _ORDER[f.severity])


def audit(
    bm: Benchmark,
    claimed_ceiling: float | None = None,
    dup_threshold: float = 0.8,
    positive_label: Any = None,
    k_folds: int = 5,
    seed: int = 0,
) -> AuditResult:
    """Run the full audit, degrading gracefully by what the benchmark provides."""
    res = AuditResult(
        benchmark=bm.name,
        n_rows=bm.n_raw_rows,
        n_audited=len(bm),
    )

    if bm.skipped_rows:
        res.findings.append(
            Finding(
                WARNING,
                "loading",
                f"{len(bm.skipped_rows)} of {bm.n_raw_rows} rows were skipped",
                "Rows with an empty item or label are excluded from every number "
                "below. Check whether they are genuinely malformed or whether the "
                "column mapping is wrong.",
                {"skipped": bm.skipped_rows[:20], "total_skipped": len(bm.skipped_rows)},
            )
        )

    if len(bm) < 10:
        res.notes.append("Fewer than 10 usable rows; most checks are uninformative.")
        return res

    _check_multiplicity(bm, res, dup_threshold)
    _check_design(bm, res)

    if not bm.has_labels:
        res.notes.append(
            "No label column: degenerate baselines, ceiling violation and the "
            "protocol gap were skipped. Multiplicity and effective n above apply "
            "regardless of task."
        )
        return res

    if bm.is_binary:
        _check_binary(bm, res, claimed_ceiling, positive_label, k_folds, seed)
    else:
        res.notes.append(
            f"Label column has {len(bm.label_values)} distinct values. The "
            f"degenerate-baseline battery is defined for binary labels; the "
            f"protocol gap below is computed multi-class."
        )
        _check_protocol_gap(bm, res, k_folds, seed)

    return res


# --------------------------------------------------------------------------


def _check_multiplicity(bm: Benchmark, res: AuditResult, threshold: float) -> None:
    if bm.is_binary:
        lab = bm.binary_labels()
    else:
        lab = [0] * len(bm.items)
    exact = template_multiplicity(bm.items, lab)
    dups = find_duplicates(bm.items, threshold=threshold)
    res.duplicates = dups
    res.effective_n = dups.n_clusters
    res.effective_n_curve = effective_n_curve(bm.items, unit=dups.unit, k=dups.k)
    res.representation = f"{dups.unit}-{dups.k} shingles"

    n = len(bm)
    counts = [c for _, c, _ in res.effective_n_curve] or [dups.n_clusters]
    n_strict, n_loose = max(counts), min(counts)
    res.effective_n_range = (n_loose, n_strict)

    # Grade on the sweep, not on one threshold. A point estimate at a strict
    # threshold is exactly how a templated set passes for a diverse one.
    ratio = n_loose / n if n else 1.0
    collapse = n_loose / n_strict if n_strict else 1.0

    ev = {
        "rows": n,
        "exact_distinct": exact.n_distinct,
        "representation": res.representation,
        "clusters_at_threshold": dups.n_clusters,
        "threshold": threshold,
        "effective_n_strictest": n_strict,
        "effective_n_loosest": n_loose,
        "max_cluster": dups.max_cluster,
        "weighted_mean_cluster": round(dups.mean_cluster_weighted, 2),
        "duplication_rate": round(dups.duplication_rate, 4),
    }
    if dups.truncated:
        ev["candidate_cap_hit"] = True

    # A declared clustering column is authoritative; near-duplicate clustering
    # is a guess at the unit. When the benchmark states its own unit, the guess
    # must not outrank the declaration, so its severity is capped and the
    # design-effect finding carries the effective-n claim instead.
    declared = bm.has_groups

    if ratio < 0.25 and not declared:
        sev, head = CRITICAL, (
            f"{n} rows reduce to as few as {n_loose} independent items "
            f"({ratio:.0%} of the row count)"
        )
    elif ratio < 0.6 or collapse < 0.5 or (ratio < 0.25 and declared):
        sev, head = WARNING, (
            f"{n} rows reduce to between {n_loose} and {n_strict} independent "
            f"items depending on the duplicate threshold"
        )
    else:
        sev, head = OK, (
            f"{n_loose}–{n_strict} independent items in {n} rows — "
            f"little duplication detected at any threshold"
        )

    inflation = (dups.mean_cluster_weighted ** 0.5) if dups.mean_cluster_weighted > 1 else 1.0
    detail = (
        f"Shingling: {res.representation}, chosen from a median item length of "
        f"{_median_words(bm.items)} words. Exact string matching finds "
        f"{exact.n_distinct} distinct items; near-duplicate clustering at "
        f"Jaccard >= {threshold} finds {dups.n_clusters}, and the full sweep "
        f"ranges from {n_strict} down to {n_loose}. The largest cluster at the "
        f"stated threshold holds {dups.max_cluster} rows.\n\n"
        f"If labels are constant within a cluster, the intra-cluster correlation "
        f"is 1 and each cluster contributes one independent observation. "
        f"Confidence intervals computed on {n} rows are then too narrow by "
        f"roughly a factor of {inflation:.1f}."
    )
    if bm.is_binary:
        total = exact.constant_label_groups + exact.mixed_label_groups
        if total:
            detail += (
                f"\n\nAmong exact-duplicate groups, {exact.constant_label_groups} of "
                f"{total} carry a single label and {exact.mixed_label_groups} are "
                f"mixed. The inflation factor above is an upper bound; it is reached "
                f"only where the label is constant within the group."
            )
            ev["constant_label_groups"] = exact.constant_label_groups
            ev["mixed_label_groups"] = exact.mixed_label_groups
    if declared:
        detail += (
            f"\n\nThis benchmark declares its own clustering unit in the "
            f"`{bm.group_column or 'grouping'}` column, which is authoritative. "
            f"The figures above come from guessing the unit from item text and "
            f"are reported as a cross-check only — see the design-effect finding "
            f"for the effective n that should be used."
        )
        ev["declared_grouping_present"] = True
    res.findings.append(Finding(sev, "multiplicity", head, detail, ev))


def _check_design(bm: Benchmark, res: AuditResult) -> None:
    """Design effect from a declared clustering column.

    Only runs when the benchmark states its own clustering unit — a donor, a
    batch, a cell line, a generating prompt. Near-duplicate clusters are a
    *guess* at the unit; a declared column is the real thing, and the two answer
    different questions, so this is reported separately from multiplicity rather
    than folded into it.
    """
    if not bm.has_groups or not bm.has_labels:
        return
    if bm.is_binary:
        vals: list[float] = [float(x) for x in bm.binary_labels()]
    else:
        order = {v: i for i, v in enumerate(bm.label_values)}
        vals = [float(order[v]) for v in bm.labels]
        if len(set(vals)) > 2:
            res.notes.append(
                f"Design effect skipped: the label column has "
                f"{len(bm.label_values)} values and the one-way decomposition "
                f"needs a single numeric measurement. Pass a numeric score "
                f"column to compute it."
            )
            return

    de = design_effect(vals, bm.groups)
    res.design = de
    if de.estimable:
        # The declared unit is authoritative, so the headline effective n comes
        # from here rather than from near-duplicate clustering.
        res.effective_n = int(round(de.effective_n))
        res.effective_n_source = f"design effect on `{bm.group_column or 'declared grouping'}`"
    if not de.estimable:
        res.notes.append(f"Design effect not estimable: {de.reason}")
        return

    ev = {
        "grouping_column": bm.group_column,
        "n_clusters": de.n_clusters,
        "mean_cluster": round(de.mean_cluster, 2),
        "icc": round(de.icc, 4),
        "design_effect": round(de.design_effect, 3),
        "effective_n": round(de.effective_n, 1),
        "interval_inflation": round(de.inflation, 3),
        "balanced": de.balanced,
    }
    if de.reason:
        ev["note"] = de.reason

    if de.design_effect >= 10:
        sev = CRITICAL
    elif de.design_effect >= 2:
        sev = WARNING
    else:
        sev = OK

    col = bm.group_column or "declared grouping"
    ev["grouping_column"] = col
    head = (
        f"Design effect {de.design_effect:.1f} on `{col}`: "
        f"effective n is {de.effective_n:.0f}, not {de.n_rows}"
    )
    detail = (
        de.sentence()
        + "\n\nThis is pseudoreplication: the rows are not independent "
        "observations of the quantity being claimed. Any interval, p-value or "
        "power calculation computed on the row count overstates the evidence by "
        f"the inflation factor above. The remedy is not more rows — at this ICC "
        f"and cluster size, additional rows within existing clusters buy almost "
        f"nothing. It is more clusters."
    )
    res.findings.append(Finding(sev, "design-effect", head, detail, ev))


def _check_binary(
    bm: Benchmark,
    res: AuditResult,
    claimed_ceiling: float | None,
    positive_label: Any,
    k_folds: int,
    seed: int,
) -> None:
    pos = positive_label if positive_label is not None else bm.positive_value()
    y = bm.binary_labels(pos)
    base = sum(y) / len(y)

    res.findings.append(
        Finding(
            INFO,
            "labels",
            f"Binary labels; positive class {pos!r} at {base:.1%} prevalence",
            "The rarer value is treated as positive unless overridden. "
            "Precision figures depend on this choice.",
            {"positive_value": str(pos), "base_rate": round(base, 4), "n_positive": sum(y)},
        )
    )

    baselines = degenerate_baselines(bm.items, y)
    trivial = next((b.f1 for b in baselines if b.name == "flag-everything"), 0.0)
    rows = [{"name": b.name, "f1": round(b.f1, 4), "reads": b.describes} for b in baselines]

    beating = [b for b in baselines if b.name != "flag-everything" and b.f1 > trivial + 0.05]
    if beating:
        worst = max(beating, key=lambda b: b.f1)
        res.findings.append(
            Finding(
                WARNING,
                "degenerate-baselines",
                f"'{worst.name}' reaches F1 {worst.f1:.3f} against a trivial "
                f"baseline of {trivial:.3f}",
                "A solver that does none of the task beats the trivial baseline, "
                "so some signal is reachable without performing the task. That is "
                "a shortcut, not necessarily a fatal one — but any reported score "
                "should be compared against this number, not against zero.",
                {"baselines": rows, "trivial": round(trivial, 4)},
            )
        )
    else:
        res.findings.append(
            Finding(
                OK,
                "degenerate-baselines",
                "No task-blind probe beats the trivial baseline by more than 0.05",
                "This does not establish the benchmark is clean. The battery is "
                "sound, not complete: it tests four channels and there are others.",
                {"baselines": rows, "trivial": round(trivial, 4)},
            )
        )

    if claimed_ceiling is not None:
        violations = [b for b in baselines if b.f1 > claimed_ceiling]
        # "Informative" means scoring above the trivial baseline, not merely
        # being a different probe. A length or identity probe that lands on
        # exactly 2p/(1+p) has degenerated into flag-everything: its best
        # in-sample rule was to answer positive to everything, so it found no
        # signal and attributing a violation to it would misdiagnose the defect.
        informative = [v for v in violations if v.f1 > trivial + 1e-9]
        if violations:
            w = max(violations, key=lambda b: b.f1)
            if informative:
                detail = (
                    "A solver that provably does none of the task cannot "
                    "legitimately exceed a bound on solvers that do. Either the "
                    "claimed ceiling is wrong or the items encode the labels. "
                    "Both are construction defects, and this is a proof rather "
                    "than an indication."
                )
            else:
                names = ", ".join(sorted(v.name for v in violations))
                detail = (
                    f"No probe exceeded the ceiling by more than the trivial "
                    f"baseline. Flagging every item positive scores "
                    f"2p/(1+p) = {trivial:.3f} at a base rate of {base:.1%}, so a "
                    f"claimed ceiling of {claimed_ceiling:.3f} is below the score "
                    f"obtainable by refusing to discriminate at all. The probes "
                    f"above the ceiling ({names}) all landed at or beneath that "
                    f"figure, which means their best in-sample rule was to answer "
                    f"positive to everything — they found no signal.\n\n"
                    f"This is a weaker diagnosis than leakage: the claimed bound is "
                    f"incoherent, but the items are not shown to encode their "
                    f"labels. Check whether the ceiling was quoted for a different "
                    f"metric — accuracy, or F1 on the other class — before treating "
                    f"it as a defect in the items."
                )
            res.findings.append(
                Finding(
                    CRITICAL,
                    "ceiling-violation",
                    (
                        f"'{w.name}' reaches {w.f1:.3f} against a claimed ceiling of "
                        f"{claimed_ceiling:.3f}"
                        if informative
                        else f"The claimed ceiling of {claimed_ceiling:.3f} is below "
                        f"the trivial baseline of {trivial:.3f}"
                    ),
                    detail,
                    {
                        "diagnosis": "label-encoding shortcut"
                        if informative
                        else "ceiling below trivial baseline",
                        "claimed_ceiling": claimed_ceiling,
                        "violations": [
                            {"name": v.name, "f1": round(v.f1, 4)} for v in violations
                        ],
                        "margin": round(w.f1 - claimed_ceiling, 4),
                    },
                )
            )
        else:
            res.findings.append(
                Finding(
                    OK,
                    "ceiling-violation",
                    f"No task-blind probe exceeds the claimed ceiling of "
                    f"{claimed_ceiling:.3f}",
                    "Absence of a violation proves nothing; the test is sound, not "
                    "complete.",
                    {"claimed_ceiling": claimed_ceiling},
                )
            )

    _check_protocol_gap(bm, res, k_folds, seed, binary_labels=y)


def _check_protocol_gap(
    bm: Benchmark,
    res: AuditResult,
    k_folds: int,
    seed: int,
    binary_labels: Sequence[int] | None = None,
) -> None:
    """Compare row-wise and group-wise cross-validation of the same model."""
    if binary_labels is not None:
        y: list[int] = list(binary_labels)
    else:
        vals = {v: i for i, v in enumerate(bm.label_values)}
        y = [vals[v] for v in bm.labels]
        if len(set(y)) != 2:
            res.notes.append(
                "Protocol gap skipped: the reference classifier is binary and the "
                "label column is multi-class. Supply a binary view to enable it."
            )
            return

    if bm.has_groups:
        groups: list[Any] = list(bm.groups)
        source = f"the '{bm.group_column}' column"
    elif res.duplicates is not None:
        groups = cluster_labels(res.duplicates, len(bm))
        source = f"near-duplicate clusters at Jaccard >= {res.duplicates.threshold}"
    else:
        return

    n_groups = len(set(groups))
    if n_groups < k_folds:
        res.notes.append(
            f"Protocol gap skipped: only {n_groups} groups, fewer than {k_folds} "
            f"folds. The benchmark does not contain enough independent units to "
            f"cross-validate this way, which is itself the finding."
        )
        return

    gap = grouping_gap(
        bm.items, y, groups, naive_bayes_fit_predict, k=k_folds, seed=seed
    )
    ev = {
        "row_wise_f1": round(gap.row_wise_f1, 4),
        "grouped_f1": round(gap.grouped_f1, 4),
        "gap": round(gap.gap, 4),
        "overlap_share": round(gap.overlap_share, 4),
        "grouped_folds": [round(f, 4) for f in gap.grouped_folds],
        "grouping_source": source,
        "n_groups": n_groups,
    }

    if gap.gap > 0.25:
        sev = CRITICAL
        head = (
            f"Row-wise CV scores {gap.row_wise_f1:.3f}, group-wise {gap.grouped_f1:.3f} "
            f"— a protocol gap of {gap.gap:+.3f}"
        )
    elif gap.gap > 0.10:
        sev = WARNING
        head = (
            f"Protocol gap of {gap.gap:+.3f} between row-wise "
            f"({gap.row_wise_f1:.3f}) and group-wise ({gap.grouped_f1:.3f}) CV"
        )
    else:
        sev = OK
        head = f"Protocol gap is small ({gap.gap:+.3f}); splitting choice changes little"

    spread = ""
    if gap.grouped_folds:
        lo, hi = min(gap.grouped_folds), max(gap.grouped_folds)
        if hi - lo > 0.3:
            spread = (
                f"\n\nGroup-wise fold scores range from {lo:.3f} to {hi:.3f}. "
                f"With {n_groups} groups across {k_folds} folds the estimate is "
                f"unstable, which is an honest measurement revealing that the "
                f"benchmark cannot support the question being asked of it."
            )

    detail = (
        f"The same elementary classifier, the same data, two splitting protocols. "
        f"Grouping by {source}. Under row-wise splitting, {gap.overlap_share:.0%} of "
        f"held-out items also appear in training.{spread}"
    )
    res.findings.append(Finding(sev, "protocol-gap", head, detail, ev))
