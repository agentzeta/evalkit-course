"""Loading real benchmark files.

Benchmarks arrive as CSV, TSV or JSONL with no agreed schema. This module
loads them, guesses which column is the item and which is the label, and lets
the caller override the guess. It does not transform the data.

Design rule: **the loader never silently drops rows.** A row with a missing
item or label is recorded as a problem and reported, because quietly discarding
10% of a benchmark would change every number downstream.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

# Column names, in priority order, that usually hold the item and the label.
_ITEM_HINTS = (
    "text", "item", "prompt", "question", "input", "content", "sentence",
    "query", "instruction", "passage", "document", "body",
)
_LABEL_HINTS = (
    "label", "answer", "target", "class", "gold", "output", "y",
    "ground_truth", "correct", "expected",
)
_GROUP_HINTS = (
    "group", "source", "template", "prompt_id", "origin", "cluster",
    "generator", "seed_prompt", "family",
)


@dataclass
class Benchmark:
    """A loaded benchmark, plus whatever went wrong loading it."""

    name: str
    items: list[str]
    labels: list[Any] = field(default_factory=list)
    groups: list[Any] = field(default_factory=list)
    scores: dict[str, list[float]] = field(default_factory=dict)
    item_column: str = ""
    label_column: str = ""
    group_column: str = ""
    skipped_rows: list[tuple[int, str]] = field(default_factory=list)
    n_raw_rows: int = 0

    def __len__(self) -> int:
        return len(self.items)

    @property
    def has_labels(self) -> bool:
        return bool(self.labels) and len(self.labels) == len(self.items)

    @property
    def has_groups(self) -> bool:
        return bool(self.groups) and len(self.groups) == len(self.items)

    @property
    def label_values(self) -> list[Any]:
        return sorted(set(self.labels), key=str) if self.has_labels else []

    @property
    def is_binary(self) -> bool:
        return self.has_labels and len(self.label_values) == 2

    def binary_labels(self, positive: Any = None) -> list[int]:
        """Map a two-valued label column to 0/1.

        `positive` names the value treated as 1. When omitted, the *rarer*
        value is taken as positive, which matches the convention that the
        positive class is the thing being hunted. The choice is reported, not
        assumed silently.
        """
        if not self.is_binary:
            raise ValueError(
                f"label column has {len(self.label_values)} distinct values; "
                f"binary analysis needs exactly 2"
            )
        if positive is None:
            counts = {v: self.labels.count(v) for v in self.label_values}
            positive = min(counts, key=lambda v: counts[v])
        return [1 if v == positive else 0 for v in self.labels]

    def positive_value(self) -> Any:
        counts = {v: self.labels.count(v) for v in self.label_values}
        return min(counts, key=lambda v: counts[v])


def _pick(fieldnames: Sequence[str], hints: Sequence[str]) -> str:
    """Choose a column by hint, preferring exact matches over substrings."""
    lower = {f.lower().strip(): f for f in fieldnames}
    for h in hints:
        if h in lower:
            return lower[h]
    for h in hints:
        for low, orig in lower.items():
            if h in low:
                return orig
    return ""


def _coerce_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def load(
    path: str | Path,
    item_column: str = "",
    label_column: str = "",
    group_column: str = "",
    score_columns: Sequence[str] = (),
    name: str = "",
) -> Benchmark:
    """Load a CSV, TSV or JSONL benchmark.

    Column arguments override the guess. Pass `item_column` explicitly for any
    real audit — a wrong guess silently audits the wrong field, and that is the
    one error this tool must not make quietly.
    """
    path = Path(path)
    rows, fieldnames = _read_rows(path)
    if not rows:
        raise ValueError(f"{path} contains no rows")

    item_col = item_column or _pick(fieldnames, _ITEM_HINTS)
    if not item_col:
        raise ValueError(
            f"could not identify an item column in {fieldnames!r}; "
            f"pass item_column explicitly"
        )
    if item_col not in fieldnames:
        raise ValueError(f"item column {item_col!r} not in {fieldnames!r}")

    label_col = label_column or _pick(fieldnames, _LABEL_HINTS)
    group_col = group_column or _pick(fieldnames, _GROUP_HINTS)
    if label_col and label_col not in fieldnames:
        raise ValueError(f"label column {label_col!r} not in {fieldnames!r}")
    if group_col and group_col not in fieldnames:
        raise ValueError(f"group column {group_col!r} not in {fieldnames!r}")

    bm = Benchmark(
        name=name or path.name,
        items=[],
        item_column=item_col,
        label_column=label_col,
        group_column=group_col,
        n_raw_rows=len(rows),
    )
    for col in score_columns:
        if col not in fieldnames:
            raise ValueError(f"score column {col!r} not in {fieldnames!r}")
        bm.scores[col] = []

    for i, row in enumerate(rows):
        item = row.get(item_col)
        if item is None or not str(item).strip():
            bm.skipped_rows.append((i, "empty item"))
            continue
        if label_col:
            lab = row.get(label_col)
            if lab is None or not str(lab).strip():
                bm.skipped_rows.append((i, "empty label"))
                continue
            bm.labels.append(lab)
        if group_col:
            bm.groups.append(row.get(group_col))
        bm.items.append(str(item))
        for col in score_columns:
            bm.scores[col].append(_coerce_float(row.get(col)) or 0.0)

    return bm


def _read_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        fields: list[str] = []
        for r in rows:
            for key in r:
                if key not in fields:
                    fields.append(key)
        return rows, fields
    if suffix == ".json":
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            for key in ("data", "rows", "examples", "items"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ValueError(f"{path} is not a list of records")
        fields = []
        for r in data:
            for key in r:
                if key not in fields:
                    fields.append(key)
        return data, fields

    delim = "\t" if suffix in {".tsv", ".tab"} else ","
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delim)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    return rows, fields


def from_lists(
    items: Sequence[str],
    labels: Sequence[Any] = (),
    groups: Sequence[Any] = (),
    name: str = "inline",
) -> Benchmark:
    """Build a Benchmark from in-memory lists, for tests and notebooks."""
    if labels and len(labels) != len(items):
        raise ValueError(f"{len(items)} items but {len(labels)} labels")
    if groups and len(groups) != len(items):
        raise ValueError(f"{len(items)} items but {len(groups)} groups")
    return Benchmark(
        name=name,
        items=list(items),
        labels=list(labels),
        groups=list(groups),
        n_raw_rows=len(items),
    )
