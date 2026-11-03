"""benchaudit — benchmark integrity auditing.

A benchmark makes a claim about what a score means. This package checks that
claim using nothing but the benchmark itself: how many genuinely independent
items it contains, whether a solver that does none of the task can score on it,
whether any task-blind probe exceeds a bound the authors called irreducible,
and how much the choice of splitting protocol flatters the result.

Pure standard library. No model is called. No network is touched.

    from benchaudit import load, audit, as_markdown

    bm = load("bench.csv", item_column="prompt", label_column="gold")
    print(as_markdown(audit(bm, claimed_ceiling=0.87)))

Or from a shell:

    python -m benchaudit bench.csv --item prompt --ceiling 0.87 --format markdown
"""

from .audit import CRITICAL, INFO, OK, WARNING, AuditResult, Finding, audit
from .dedup import (
    Cluster,
    DuplicateReport,
    cluster_labels,
    effective_n_curve,
    find_duplicates,
    jaccard,
    minhash,
    normalise,
    shingles,
)
from .design import DesignEffect, clusters_for_power, design_effect, n_for_power
from .loader import Benchmark, from_lists, load
from .report import as_json, as_markdown, as_text, cross_table

__all__ = [
    "AuditResult",
    "Benchmark",
    "CRITICAL",
    "Cluster",
    "DesignEffect",
    "DuplicateReport",
    "clusters_for_power",
    "design_effect",
    "n_for_power",
    "Finding",
    "INFO",
    "OK",
    "WARNING",
    "as_json",
    "as_markdown",
    "as_text",
    "audit",
    "cluster_labels",
    "cross_table",
    "effective_n_curve",
    "find_duplicates",
    "from_lists",
    "jaccard",
    "load",
    "minhash",
    "normalise",
    "shingles",
]

__version__ = "0.1.0"
