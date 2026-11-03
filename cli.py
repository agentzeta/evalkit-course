"""Command-line entry point.

    python -m benchaudit BENCH.csv --ceiling 0.87 --format markdown -o report.md
    python -m benchaudit a.csv b.jsonl --format table

Exit codes are meant for a CI gate:

    0  no finding worse than INFO
    1  at least one WARNING
    2  at least one CRITICAL
    3  the audit could not run (bad file, no item column)

A benchmark author can therefore make release conditional on `exit 0`, which is
the point: the checks are cheap enough to run every time, and the ones that
matter are deductive rather than advisory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .audit import CRITICAL, WARNING, audit
from .loader import load
from .report import as_json, as_markdown, as_text, cross_table

_EXIT = {CRITICAL: 2, WARNING: 1}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="benchaudit",
        description=(
            "Audit a benchmark for leakage, duplication and protocol defects. "
            "Reads the benchmark only: no model is called and no network is used."
        ),
        epilog=(
            "Pass --item explicitly for any audit you intend to publish. A wrong "
            "column guess audits the wrong field, which is the one error this "
            "tool must not make quietly."
        ),
    )
    p.add_argument("paths", nargs="+", help="benchmark files (.csv/.tsv/.jsonl/.json)")
    p.add_argument("--item", default="", help="column holding the item text")
    p.add_argument("--label", default="", help="column holding the label")
    p.add_argument(
        "--group",
        default="",
        help="column holding the generating prompt or template id, if the "
        "benchmark records one; otherwise near-duplicate clusters are used",
    )
    p.add_argument(
        "--ceiling",
        type=float,
        default=None,
        help="irreducible bound the authors claim (Bayes rate, annotator "
        "agreement, reference-solver score). Enables the violation test.",
    )
    p.add_argument(
        "--positive",
        default=None,
        help="label value to treat as the positive class (default: the rarer)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Jaccard threshold for near-duplicate clustering (default 0.8)",
    )
    p.add_argument("--folds", type=int, default=5, help="CV folds (default 5)")
    p.add_argument("--seed", type=int, default=0, help="fold seed (default 0)")
    p.add_argument(
        "--format",
        choices=("text", "markdown", "json", "table"),
        default="text",
        help="'table' gives one row per benchmark across all inputs",
    )
    p.add_argument("-o", "--out", default="", help="write to this file instead of stdout")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    results = []
    for path in args.paths:
        try:
            bm = load(
                path,
                item_column=args.item,
                label_column=args.label,
                group_column=args.group,
            )
        except (OSError, ValueError) as exc:
            print(f"{path}: {exc}", file=sys.stderr)
            return 3
        results.append(
            audit(
                bm,
                claimed_ceiling=args.ceiling,
                dup_threshold=args.threshold,
                positive_label=args.positive,
                k_folds=args.folds,
                seed=args.seed,
            )
        )

    if args.format == "table":
        body = cross_table(results)
    else:
        render = {"text": as_text, "markdown": as_markdown, "json": as_json}[args.format]
        sep = "\n\n" if args.format != "json" else "\n"
        body = sep.join(render(r) for r in results)

    if args.out:
        Path(args.out).write_text(body + "\n", encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(body)

    worst = [r.worst for r in results]
    for sev, code in _EXIT.items():
        if sev in worst:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
