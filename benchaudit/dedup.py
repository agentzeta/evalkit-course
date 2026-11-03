"""Near-duplicate detection for benchmark items.

The fixture work used exact string matching to find repeated items. Real
generated benchmarks do not repeat exactly -- a model prompted fifty times
produces fifty paraphrases, not fifty copies. Exact matching reports 5,000
distinct items and misses that there are really fifty.

This module finds the clusters anyway, with shingling plus locality-sensitive
hashing, in pure standard library.

The method:
  1. Normalise text (case, punctuation, whitespace).
  2. Cut it into overlapping word k-shingles.
  3. Build a MinHash signature of each item's shingle set.
  4. Band the signature and bucket on each band, so items sharing any band
     become candidate pairs -- this avoids the O(n^2) comparison.
  5. Compute exact Jaccard on candidates only, and cluster above a threshold.

Why it matters: the effective sample size of a benchmark is the number of
independent items, and two paraphrases of one prompt are not independent.
`effective_n_curve` reports how that count moves with the threshold, so a
reader can see the sensitivity rather than trusting one cut.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Deliberately crude. A normaliser that is too clever hides real differences;
    this one only removes variation nobody would call a distinct item.
    """
    return _WS.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


def shingles(text: str, k: int = 3, unit: str = "word") -> set[str]:
    """Overlapping k-grams, over words or characters.

    `unit="word"` with k=3 is the usual default for prose. `unit="char"` with
    k=5 is the right default for short items, and the difference is not
    cosmetic: in an eleven-word sentence, changing a single token destroys
    three of nine word-trigrams and drops Jaccard to 0.5, so two instances of
    one template look like two distinct items. The same edit touches about six
    of sixty character-5-grams and leaves Jaccard near 0.8.

    Short items (fewer than k units) fall back to the whole string, so they
    still cluster with their exact copies rather than producing an empty set
    that matches nothing.
    """
    norm = normalise(text)
    if unit == "char":
        s = norm.replace(" ", "")
        if len(s) < k:
            return {s} if s else set()
        return {s[i : i + k] for i in range(len(s) - k + 1)}
    if unit != "word":
        raise ValueError(f"unit must be 'word' or 'char', got {unit!r}")
    words = norm.split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def choose_representation(texts: Sequence[str], long_item_words: int = 25) -> tuple[str, int]:
    """Pick a shingling scheme from the items' length, and say which.

    Returns `(unit, k)`. Short items get character 5-grams because a one-token
    edit would otherwise look like a new item; longer prose gets word
    trigrams, where content words carry the similarity and character overlap
    is dominated by common morphology.

    The choice is returned rather than applied silently, because it changes the
    duplication estimate and therefore belongs in the report.
    """
    if not texts:
        return "char", 5
    lengths = sorted(len(normalise(t).split()) for t in texts)
    median = lengths[len(lengths) // 2]
    return ("word", 3) if median >= long_item_words else ("char", 5)


def _hash(s: str, salt: int) -> int:
    """Stable 64-bit hash. Python's built-in hash is randomised per process,
    which would make signatures irreproducible between runs."""
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8, salt=salt.to_bytes(8, "little"))
    return int.from_bytes(h.digest(), "little")


def minhash(sh: set[str], n_perm: int = 64) -> tuple[int, ...]:
    """MinHash signature: the minimum hash under each of n_perm permutations.

    The probability two signatures agree at a given position equals the
    Jaccard similarity of their shingle sets, which is what makes banding work.
    """
    if not sh:
        return tuple([0] * n_perm)
    return tuple(min(_hash(s, p) for s in sh) for p in range(n_perm))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


@dataclass
class Cluster:
    """A group of items judged to be near-duplicates of each other."""

    members: list[int]
    representative: int
    size: int = 0

    def __post_init__(self) -> None:
        self.size = len(self.members)


@dataclass
class DuplicateReport:
    threshold: float
    n_items: int
    clusters: list[Cluster] = field(default_factory=list)
    exact_distinct: int = 0
    unit: str = "word"
    k: int = 3
    truncated: bool = False
    exact: bool = False

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    @property
    def max_cluster(self) -> int:
        return max((c.size for c in self.clusters), default=0)

    @property
    def mean_cluster_weighted(self) -> float:
        """Size-weighted mean cluster size: sum(m^2)/sum(m).

        Weighted because large clusters dominate variance inflation, which is
        what this number feeds into.
        """
        sizes = [c.size for c in self.clusters]
        total = sum(sizes)
        return sum(s * s for s in sizes) / total if total else 0.0

    @property
    def duplication_rate(self) -> float:
        """Share of items that are not the sole member of their cluster."""
        if not self.n_items:
            return 0.0
        dup = sum(c.size for c in self.clusters if c.size > 1)
        return dup / self.n_items


def choose_bands(threshold: float, n_perm: int) -> tuple[int, int]:
    """Pick (bands, rows) so the LSH S-curve turns near `threshold`.

    A banded signature makes a pair a candidate with probability
    `1 - (1 - J**rows)**bands`, which is a step function whose inflection sits
    at roughly `(1/bands)**(1/rows)`. Putting the inflection at the threshold
    is the standard choice: pairs well above it are nearly always found, pairs
    well below it nearly never.

    Recall *at* the threshold is only about 60% whatever the parameters, which
    is why `find_duplicates` compares all pairs exactly on small inputs instead
    of relying on this.
    """
    best, best_err = (n_perm, 1), float("inf")
    for bands in range(1, n_perm + 1):
        if n_perm % bands:
            continue
        rows = n_perm // bands
        err = abs((1.0 / bands) ** (1.0 / rows) - threshold)
        if err < best_err:
            best, best_err = (bands, rows), err
    return best


def _union_find(n: int, pairs: Iterable[tuple[int, int]]) -> list[int]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return [find(i) for i in range(n)]


def find_duplicates(
    texts: Sequence[str],
    threshold: float = 0.8,
    k: int | None = None,
    unit: str = "",
    n_perm: int = 128,
    bands: int = 0,
    exact_below: int = 2_000,
    max_pairs: int = 5_000_000,
) -> DuplicateReport:
    """Cluster near-duplicate items.

    `threshold` is Jaccard similarity on k-shingles. 0.8 is strict (close
    paraphrases only); 0.5 catches looser reuse of a template. Report more than
    one, which is what `effective_n_curve` does.

    `unit` and `k` default to whatever `choose_representation` picks for these
    items, and the choice is recorded on the report. Pass them to override.

    At or below `exact_below` items every pair is compared, so the clustering is
    exact. Above it, MinHash banding generates candidates first, which is
    approximate: pairs near the threshold are missed with meaningful
    probability, so the cluster count is an **upper bound** on the truth and
    `exact` on the report is False. Set `exact_below=0` to force the
    approximate path.

    `max_pairs` caps the candidate comparisons on the approximate path. If the
    cap is hit the result may under-cluster further, and `truncated` says so.
    """
    n = len(texts)
    auto_unit, auto_k = choose_representation(texts)
    unit = unit or auto_unit
    k = auto_k if k is None else k
    if n == 0:
        return DuplicateReport(
            threshold=threshold, n_items=0, exact_distinct=0, unit=unit, k=k, exact=True
        )
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    if bands and n_perm % bands != 0:
        raise ValueError(f"n_perm ({n_perm}) must be divisible by bands ({bands})")

    sh = [shingles(t, k, unit) for t in texts]
    truncated = False
    is_exact = n <= exact_below

    if is_exact:
        # Small enough to compare every pair, so compare every pair. An
        # approximate method whose recall is unknown is worse than n^2/2 on a
        # few thousand items.
        candidates = {(a, b) for a in range(n) for b in range(a + 1, n)}
    else:
        bands_, rows = (bands, n_perm // bands) if bands else choose_bands(threshold, n_perm)
        sigs = [minhash(s, n_perm) for s in sh]
        buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
        for i, sig in enumerate(sigs):
            for b in range(bands_):
                key = (b, sig[b * rows : (b + 1) * rows])
                buckets.setdefault(key, []).append(i)

        candidates = set()
        for members in buckets.values():
            if len(members) < 2:
                continue
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    if len(candidates) >= max_pairs:
                        truncated = True
                        break
                    candidates.add((members[x], members[y]))
                if truncated:
                    break
            if truncated:
                break

    linked = [(a, b) for a, b in candidates if jaccard(sh[a], sh[b]) >= threshold]
    roots = _union_find(n, linked)

    groups: dict[int, list[int]] = {}
    for i, r in enumerate(roots):
        groups.setdefault(r, []).append(i)

    clusters = [Cluster(members=sorted(m), representative=min(m)) for m in groups.values()]
    clusters.sort(key=lambda c: (-c.size, c.representative))

    return DuplicateReport(
        threshold=threshold,
        n_items=n,
        clusters=clusters,
        exact_distinct=len(set(texts)),
        unit=unit,
        k=k,
        truncated=truncated,
        exact=is_exact and not truncated,
    )


def effective_n_curve(
    texts: Sequence[str],
    thresholds: Sequence[float] = (0.95, 0.9, 0.8, 0.7, 0.6, 0.5),
    k: int | None = None,
    unit: str = "",
) -> list[tuple[float, int, float]]:
    """How the independent-item count moves with the duplicate threshold.

    Returns (threshold, n_clusters, duplication_rate) per threshold.

    This is the honest way to report effective n on real data: a single number
    hides the fact that it depends entirely on where you draw the line. If the
    count is stable across the range, the benchmark is genuinely diverse. If it
    collapses as the threshold loosens, it is a small set of templates wearing
    a large row count.
    """
    out = []
    for t in thresholds:
        rep = find_duplicates(texts, threshold=t, k=k, unit=unit)
        out.append((t, rep.n_clusters, rep.duplication_rate))
    return out


def cluster_labels(rep: DuplicateReport, n_items: int) -> list[int]:
    """Map each item index to its cluster id, for use as a grouping key."""
    out = [-1] * n_items
    for cid, c in enumerate(rep.clusters):
        for m in c.members:
            out[m] = cid
    return out
