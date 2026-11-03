"""Rendering an audit as something a person or a CI job can read.

Three formats, one source of truth:

  `as_markdown` — for a repository, a release note, or a paper appendix.
  `as_json`     — for a CI gate or a cross-benchmark table.
  `as_text`     — for a terminal.

The rendering rule throughout: **a number is reported with what it depends on.**
An effective-n figure without its threshold, or a precision figure without its
base rate, is not a finding; it is a claim waiting to be misread.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from .audit import CRITICAL, INFO, OK, WARNING, AuditResult

_BADGE = {
    CRITICAL: "🔴 CRITICAL",
    WARNING: "🟠 WARNING",
    INFO: "🔵 INFO",
    OK: "🟢 OK",
}

_VERDICT = {
    CRITICAL: (
        "At least one construction defect is established, not merely suspected. "
        "Scores on this benchmark should not be reported without the "
        "qualification below."
    ),
    WARNING: (
        "No defect is proven, but at least one number reported from this "
        "benchmark would overstate the evidence. Read the findings before "
        "quoting a score."
    ),
    INFO: "No shortcut or protocol defect was detected by this battery.",
    OK: "No shortcut or protocol defect was detected by this battery.",
}


def as_text(res: AuditResult) -> str:
    lines = [
        f"benchmark      {res.benchmark}",
        f"rows           {res.n_rows} raw, {res.n_audited} audited",
        f"effective n    {res.effective_n}",
        f"worst finding  {res.worst}",
        "",
    ]
    for f in res.sorted_findings():
        lines.append(f"[{f.severity:<8}] {f.check:<22} {f.headline}")
    if res.notes:
        lines.append("")
        for n in res.notes:
            lines.append(f"note: {n}")
    return "\n".join(lines)


def as_json(res: AuditResult) -> str:
    payload: dict[str, Any] = {
        "benchmark": res.benchmark,
        "rows_raw": res.n_rows,
        "rows_audited": res.n_audited,
        "effective_n": res.effective_n,
        "effective_n_source": res.effective_n_source,
        "design_effect": (
            {
                "grouping_clusters": res.design.n_clusters,
                "mean_cluster": round(res.design.mean_cluster, 3),
                "icc": round(res.design.icc, 4),
                "design_effect": round(res.design.design_effect, 4),
                "effective_n": round(res.design.effective_n, 2),
                "interval_inflation": round(res.design.inflation, 4),
                "balanced": res.design.balanced,
            }
            if res.design is not None and res.design.estimable
            else None
        ),
        "worst_severity": res.worst,
        "effective_n_curve": [
            {"threshold": t, "n_clusters": c, "duplication_rate": round(d, 4)}
            for t, c, d in res.effective_n_curve
        ],
        "findings": [
            {
                "severity": f.severity,
                "check": f.check,
                "headline": f.headline,
                "detail": f.detail,
                "evidence": f.evidence,
            }
            for f in res.sorted_findings()
        ],
        "notes": res.notes,
    }
    return json.dumps(payload, indent=2, sort_keys=False)


def as_markdown(res: AuditResult) -> str:
    out: list[str] = []
    a = out.append

    a(f"# Benchmark integrity audit — `{res.benchmark}`")
    a("")
    a(f"**{_BADGE[res.worst]}** — {_VERDICT[res.worst]}")
    a("")
    a("| | |")
    a("|---|---|")
    a(f"| Rows in file | {res.n_rows} |")
    a(f"| Rows audited | {res.n_audited} |")
    lo, hi = res.effective_n_range
    declared = res.design is not None and res.design.estimable
    if declared:
        a(f"| **Effective n** | **{res.effective_n}** (from the declared clustering unit) |")
        if lo and hi and lo != hi:
            a(f"| Effective n, guessed from item text | {lo}–{hi} (cross-check only) |")
    elif lo and hi and lo != hi:
        a(f"| **Effective n** | **{lo}–{hi}** (threshold-dependent) |")
    else:
        a(f"| **Effective n** | **{res.effective_n}** |")
    if res.duplicates is not None:
        a(f"| Exact distinct items | {res.duplicates.exact_distinct} |")
        a(f"| Near-duplicate threshold | Jaccard ≥ {res.duplicates.threshold} |")
    if res.representation:
        a(f"| Shingling | {res.representation} |")
    if res.design is not None and res.design.estimable:
        d = res.design
        a(f"| Clusters (declared) | {d.n_clusters}, mean size {d.mean_cluster:.1f} |")
        a(f"| ICC | {d.icc:.3f} |")
        a(f"| Design effect | **{d.design_effect:.2f}** |")
        a(f"| Interval inflation | {d.inflation:.2f}× |")
    a(f"| Effective n from | {res.effective_n_source} |")
    a("")
    a(
        "Effective n is the number of independent items, not the number of rows. "
        "It is the figure that belongs in the denominator of any confidence "
        "interval computed from this benchmark. It is reported as a range "
        "because it depends on where the duplicate line is drawn, and a single "
        "number hides that dependence — which is how a set of templates passes "
        "for a diverse benchmark."
    )
    a("")

    if res.effective_n_curve:
        a("## Effective n against the duplicate threshold")
        a("")
        a("| Jaccard threshold | Independent items | Share of rows duplicated |")
        a("|---|---|---|")
        for t, c, d in res.effective_n_curve:
            a(f"| {t:.2f} | {c} | {d:.1%} |")
        a("")
        first, last = res.effective_n_curve[0], res.effective_n_curve[-1]
        if first[1] and last[1] / first[1] < 0.5:
            a(
                f"The count falls from {first[1]} at threshold {first[0]:.2f} to "
                f"{last[1]} at {last[0]:.2f}. A benchmark whose independent-item "
                f"count more than halves across this range is a small set of "
                f"templates wearing a large row count, and no single effective-n "
                f"figure describes it honestly."
            )
        else:
            a(
                f"The count is stable across the range "
                f"({last[1]}–{first[1]} items), which is what genuine diversity "
                f"looks like: the figure does not depend on where the line is drawn."
            )
        a("")

    a("## Findings")
    a("")
    for f in res.sorted_findings():
        a(f"### {_BADGE[f.severity]} · {f.check}")
        a("")
        a(f"**{f.headline}**")
        a("")
        if f.detail:
            a(f.detail)
            a("")
        if f.evidence:
            a("<details><summary>Evidence</summary>")
            a("")
            a("```json")
            a(json.dumps(f.evidence, indent=2, sort_keys=False, default=str))
            a("```")
            a("")
            a("</details>")
            a("")

    if res.notes:
        a("## What was not tested")
        a("")
        for n in res.notes:
            a(f"- {n}")
        a("")

    a("## What this audit does and does not establish")
    a("")
    a(
        "Every check here is **sound and incomplete**. A ceiling violation is a "
        "proof of a construction defect: a solver that provably does none of the "
        "task cannot legitimately exceed a bound on solvers that do. The absence "
        "of a violation proves nothing, because the battery tests four shortcut "
        "channels and a benchmark can leak through others."
    )
    a("")
    a(
        "The audit reads the benchmark only. It calls no model and cannot detect "
        "contamination of a model's training data by this benchmark — a different "
        "failure with a different remedy."
    )
    a("")
    return "\n".join(out)


def cross_table(results: Sequence[AuditResult]) -> str:
    """One row per benchmark: the table a multi-benchmark audit reports."""
    out = [
        "| Benchmark | Rows | Effective n | Ratio | Worst finding |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        ratio = r.effective_n / r.n_audited if r.n_audited else 0.0
        out.append(
            f"| `{r.benchmark}` | {r.n_audited} | {r.effective_n} | "
            f"{ratio:.0%} | {r.worst} |"
        )
    return "\n".join(out)
