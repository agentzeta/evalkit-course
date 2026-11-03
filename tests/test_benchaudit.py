"""Tests for the benchmark integrity audit.

Run:  python3 -I tests/test_benchaudit.py

As in the leakage suite, the negative controls carry more weight than the
positive ones. An audit tool that grades every benchmark CRITICAL is worthless,
so a genuinely diverse benchmark must come back clean, and the suite also
contains the specific regression that caught this tool reporting "87
independent items, little duplication" for a set built from five templates.
"""

from __future__ import annotations

import io
import json
import random
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchaudit import (  # noqa: E402
    CRITICAL,
    OK,
    WARNING,
    as_json,
    as_markdown,
    as_text,
    audit,
    cluster_labels,
    cross_table,
    effective_n_curve,
    find_duplicates,
    from_lists,
    jaccard,
    load,
    minhash,
    normalise,
    shingles,
)
from benchaudit.cli import main as cli_main  # noqa: E402
from benchaudit.dedup import choose_representation  # noqa: E402

_PASS = 0
_FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global _PASS
    if cond:
        _PASS += 1
    else:
        _FAIL.append(f"{name}: {detail}" if detail else name)


def _try(fn) -> bool:
    """True if fn raises, which is what several of these tests require."""
    try:
        fn()
    except Exception:
        return True
    return False


# ===================================================================
# normalisation and shingling
# ===================================================================

check("normalise lowercases", normalise("ABC") == "abc")
check("normalise strips punctuation", "," not in normalise("a, b"))
check("normalise collapses whitespace", normalise("a   b\n c") == "a b c")
check("normalise is idempotent", normalise(normalise("A,  B!")) == normalise("A,  B!"))

sh = shingles("the quick brown fox jumps", 3)
check("word shingles count", len(sh) == 3, str(sorted(sh)))
check("word shingles content", "the quick brown" in sh)
check("short item falls back to whole string", shingles("hi", 3) == {"hi"})
check("empty item gives empty set", shingles("", 3) == set())
check("empty item gives empty set (char)", shingles("", 5, "char") == set())

csh = shingles("abcdefg", 5, "char")
check("char shingles count", len(csh) == 3, str(sorted(csh)))
check("char shingles ignore spaces", shingles("ab cde", 5, "char") == {"abcde"})
check("unknown unit rejected", _try(lambda: shingles("a b c", 3, "token")))

# The representation claim in the docstring, verified rather than asserted:
# a one-token edit in a short sentence destroys word-trigram similarity but
# leaves character similarity intact.
a = "the sample was heated to 37 degrees and the yield measured"
b = "the sample was heated to 82 degrees and the yield measured"
jw = jaccard(shingles(a, 3), shingles(b, 3))
jc = jaccard(shingles(a, 5, "char"), shingles(b, 5, "char"))
check("one-token edit breaks word trigrams", jw < 0.6, f"{jw:.3f}")
check("one-token edit survives char 5-grams", jc > 0.75, f"{jc:.3f}")
check("char similarity exceeds word similarity here", jc > jw)

check("jaccard of identical sets is 1", jaccard({"a"}, {"a"}) == 1.0)
check("jaccard of disjoint sets is 0", jaccard({"a"}, {"b"}) == 0.0)
check("jaccard of two empty sets is 1", jaccard(set(), set()) == 1.0)
check("jaccard with one empty set is 0", jaccard(set(), {"a"}) == 0.0)
check("jaccard is symmetric", jaccard({"a", "b"}, {"b"}) == jaccard({"b"}, {"a", "b"}))

# ===================================================================
# MinHash
# ===================================================================

s1 = shingles("the quick brown fox jumps over the lazy dog", 3)
check("minhash length matches n_perm", len(minhash(s1, 32)) == 32)
check("minhash is deterministic", minhash(s1, 32) == minhash(s1, 32))
check("empty shingle set gives zero signature", minhash(set(), 8) == tuple([0] * 8))
check(
    "identical sets give identical signatures",
    minhash(set(s1), 16) == minhash(set(s1), 16),
)
sig_agree = sum(x == y for x, y in zip(minhash(s1, 128), minhash(s1 | {"zz zz zz"}, 128)))
check("signature agreement tracks jaccard", sig_agree > 100, str(sig_agree))

# Stability across processes is the reason blake2b is used instead of hash().
_code = (
    "import sys; sys.path.insert(0, %r);"
    "from benchaudit.dedup import minhash, shingles;"
    "print(minhash(shingles('a b c d e f', 3), 8)[0])"
    % str(Path(__file__).resolve().parent.parent)
)
import subprocess  # noqa: E402

_r1 = subprocess.run([sys.executable, "-c", _code], capture_output=True, text=True)
_r2 = subprocess.run([sys.executable, "-c", _code], capture_output=True, text=True)
check(
    "minhash is stable across processes",
    _r1.stdout.strip() == _r2.stdout.strip() != "",
    f"{_r1.stdout.strip()!r} vs {_r2.stdout.strip()!r} / {_r1.stderr[-200:]}",
)

# ===================================================================
# duplicate detection
# ===================================================================

rep = find_duplicates([], threshold=0.8)
check("empty input gives empty report", rep.n_clusters == 0 and rep.n_items == 0)
check("empty input duplication rate is 0", rep.duplication_rate == 0.0)
check("empty input weighted mean is 0", rep.mean_cluster_weighted == 0.0)

exact = ["hello world this is a test"] * 5
rep = find_duplicates(exact, threshold=0.8)
check("exact copies form one cluster", rep.n_clusters == 1, str(rep.n_clusters))
check("cluster size equals row count", rep.max_cluster == 5)
check("exact_distinct counts unique strings", rep.exact_distinct == 1)
check("duplication rate is 1 when all duplicate", rep.duplication_rate == 1.0)
check("weighted mean cluster is 5", abs(rep.mean_cluster_weighted - 5.0) < 1e-9)
check("representation recorded", rep.unit in {"word", "char"} and rep.k > 0)

distinct = [
    "entirely unrelated sentence about mountains",
    "a completely different claim concerning finance",
    "zebras graze upon the open savannah plains",
]
rep = find_duplicates(distinct, threshold=0.8)
check("distinct items stay separate", rep.n_clusters == 3, str(rep.n_clusters))
check("duplication rate is 0 when none duplicate", rep.duplication_rate == 0.0)

check("threshold must be positive", _try(lambda: find_duplicates(["a"], threshold=0.0)))
check("threshold must be <= 1", _try(lambda: find_duplicates(["a"], threshold=1.5)))
check(
    "bands must divide n_perm",
    _try(lambda: find_duplicates(["a b c"], n_perm=64, bands=15)),
)

# Transitive clustering: A~B and B~C puts all three together even if A!~C.
# Built so the end pair genuinely falls below the threshold (0.25) while each
# adjacent pair clears it (0.538), which is what makes it a transitivity test.
_b = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu".split()
_B, _C = _b[:], _b[:]
_B[2] = "XX"
_C[2], _C[9] = "XX", "YY"
chain = [" ".join(x) for x in (_b, _B, _C)]
check("chain: adjacent pairs clear the threshold",
      jaccard(shingles(chain[0], 3), shingles(chain[1], 3)) >= 0.5
      and jaccard(shingles(chain[1], 3), shingles(chain[2], 3)) >= 0.5)
check("chain: the end pair does not",
      jaccard(shingles(chain[0], 3), shingles(chain[2], 3)) < 0.5)
rep = find_duplicates(chain, threshold=0.5, unit="word", k=3)
check("near-duplicates cluster transitively", rep.n_clusters == 1, str(rep.n_clusters))
check("small inputs are clustered exactly", rep.exact)
check("the approximate path is declared as such",
      not find_duplicates(chain, threshold=0.5, unit="word", k=3, exact_below=0).exact)

labels = cluster_labels(rep, 3)
check("cluster_labels covers every item", all(x >= 0 for x in labels))
check("cluster_labels is constant within a cluster", len(set(labels)) == 1)
check(
    "cluster_labels ids are contiguous from zero",
    sorted(set(cluster_labels(find_duplicates(distinct), 3))) == [0, 1, 2],
)

# ===================================================================
# representation selection
# ===================================================================

short = ["one two three four five"] * 10
long = [" ".join(f"w{i}" for i in range(60))] * 10
check("short items get char shingles", choose_representation(short)[0] == "char")
check("long items get word shingles", choose_representation(long)[0] == "word")
check("char default k is 5", choose_representation(short)[1] == 5)
check("word default k is 3", choose_representation(long)[1] == 3)
check("empty corpus has a default", choose_representation([])[0] == "char")

# ===================================================================
# the effective-n curve
# ===================================================================

curve = effective_n_curve(exact)
check("curve has one row per threshold", len(curve) == 6)
check("curve rows are (threshold, count, rate)", all(len(r) == 3 for r in curve))
check("curve is monotone non-increasing in count",
      all(curve[i][1] >= curve[i + 1][1] for i in range(len(curve) - 1)),
      str([c[1] for c in curve]))
check("curve on exact copies is flat at 1", {c[1] for c in curve} == {1})
check("curve rate is in [0, 1]", all(0.0 <= c[2] <= 1.0 for c in curve))

# ===================================================================
# the loader
# ===================================================================

bm = from_lists(["a b c", "d e f"], ["x", "y"], ["g1", "g1"])
check("from_lists sets length", len(bm) == 2)
check("from_lists has labels", bm.has_labels)
check("from_lists has groups", bm.has_groups)
check("from_lists is binary", bm.is_binary)
check("from_lists rejects ragged labels", _try(lambda: from_lists(["a"], ["x", "y"])))
check("from_lists rejects ragged groups", _try(lambda: from_lists(["a"], (), ["g", "h"])))
check("no labels means has_labels false", not from_lists(["a", "b"]).has_labels)
check("label_values is empty without labels", from_lists(["a"]).label_values == [])

tri = from_lists(["a", "b", "c"], ["x", "y", "z"])
check("three label values is not binary", not tri.is_binary)
check("binary_labels rejects multiclass", _try(tri.binary_labels))
check("label_values lists all values", tri.label_values == ["x", "y", "z"])

imbal = from_lists(["a"] * 7 + ["b"] * 3, ["no"] * 7 + ["yes"] * 3)
check("rarer value becomes positive", imbal.positive_value() == "yes")
check("binary_labels marks the rarer class", sum(imbal.binary_labels()) == 3)
check("explicit positive is honoured", sum(imbal.binary_labels("no")) == 7)

with tempfile.TemporaryDirectory() as td:
    d = Path(td)

    (d / "b.csv").write_text(
        "prompt,gold,template\n"
        "the first item here,A,t1\n"
        "the second item here,B,t1\n"
        ",B,t2\n"
        "the fourth item here,,t2\n"
        "the fifth item here,A,t3\n",
        encoding="utf-8",
    )
    b = load(d / "b.csv")
    check("csv infers item column", b.item_column == "prompt")
    check("csv infers label column", b.label_column == "gold")
    check("csv infers group column", b.group_column == "template")
    check("csv records raw row count", b.n_raw_rows == 5)
    check("csv skips the empty item", (2, "empty item") in b.skipped_rows)
    check("csv skips the empty label", (3, "empty label") in b.skipped_rows)
    check("csv keeps the good rows", len(b) == 3, str(len(b)))
    check("skipped rows are reported, not silent", len(b.skipped_rows) == 2)
    check("name defaults to the filename", b.name == "b.csv")
    check("items and labels stay aligned", len(b.items) == len(b.labels))
    check("items and groups stay aligned", len(b.items) == len(b.groups))

    (d / "t.tsv").write_text("text\tlabel\nfoo bar\tA\nbaz qux\tB\n", encoding="utf-8")
    check("tsv is read with tab delimiter", len(load(d / "t.tsv")) == 2)

    (d / "b.jsonl").write_text(
        '{"question": "first q", "answer": "yes"}\n'
        '{"question": "second q", "answer": "no"}\n',
        encoding="utf-8",
    )
    j = load(d / "b.jsonl")
    check("jsonl infers item column", j.item_column == "question")
    check("jsonl infers label column", j.label_column == "answer")
    check("jsonl reads both rows", len(j) == 2)

    (d / "w.json").write_text(
        '{"data": [{"input": "aa bb", "target": 1}, {"input": "cc dd", "target": 0}]}',
        encoding="utf-8",
    )
    w = load(d / "w.json")
    check("json unwraps a data key", len(w) == 2)
    check("json infers columns", (w.item_column, w.label_column) == ("input", "target"))

    (d / "o.csv").write_text("a,b\nxx,1\nyy,2\n", encoding="utf-8")
    check("unguessable item column raises", _try(lambda: load(d / "o.csv")))
    check(
        "explicit item column overrides the guess",
        load(d / "o.csv", item_column="a").item_column == "a",
    )
    check(
        "unknown explicit column raises",
        _try(lambda: load(d / "o.csv", item_column="nope")),
    )
    check(
        "unknown label column raises",
        _try(lambda: load(d / "o.csv", item_column="a", label_column="nope")),
    )
    check(
        "exact hint beats substring",
        load(d / "exact.csv" if False else d / "b.csv").item_column == "prompt",
    )

    (d / "e.csv").write_text("text,label\n", encoding="utf-8")
    check("empty file raises", _try(lambda: load(d / "e.csv")))

    # The CLI, end to end, including the exit codes a CI gate depends on.
    rows = ["text,label"]
    random.seed(11)
    for gi in range(4):
        for _ in range(15):
            tag = " anomalous" if gi % 2 == 0 else ""
            rows.append(
                f"template {gi} measurement at {random.randint(10, 99)} units{tag},"
                f"{'pos' if gi % 2 == 0 else 'neg'}"
            )
    (d / "leaky.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = cli_main([str(d / "leaky.csv")])
    check("cli exits nonzero on a defective benchmark", code in (1, 2), str(code))
    check("cli text output names the benchmark", "leaky.csv" in buf.getvalue())

    buf = io.StringIO()
    with redirect_stdout(buf):
        cli_main([str(d / "leaky.csv"), "--format", "json"])
    payload = json.loads(buf.getvalue())
    check("cli json is parseable", isinstance(payload, dict))
    check("cli json carries findings", len(payload["findings"]) >= 3)
    check("cli json reports effective n", "effective_n" in payload)

    buf = io.StringIO()
    with redirect_stdout(buf):
        cli_main([str(d / "leaky.csv"), str(d / "b.csv"), "--format", "table"])
    check("cli table has one row per benchmark",
          buf.getvalue().count("|\n") >= 4, buf.getvalue())

    out = d / "r.md"
    cli_main([str(d / "leaky.csv"), "--format", "markdown", "-o", str(out)])
    check("cli writes the output file", out.exists())
    check("cli markdown has a heading", out.read_text().startswith("# Benchmark"))

    check("cli exits 3 on a missing file", cli_main([str(d / "nope.csv")]) == 3)

# ===================================================================
# the audit, on a benchmark whose defects are known by construction
# ===================================================================

random.seed(7)
TMPL = [
    "the sample was heated to {t} degrees and the yield was measured",
    "we observed a shift in the absorption spectrum near {t} nanometres",
    "the catalyst loading was varied between {t} and twice that value",
    "cell viability dropped after {t} hours of continuous exposure",
    "the mesh was refined until the residuals fell below {t} percent",
]
items, labs, grps = [], [], []
for gi, t in enumerate(TMPL):
    for _ in range(20):
        txt = t.format(t=random.randint(10, 99))
        if gi % 2 == 0:
            txt += " anomalous"
        items.append(txt)
        labs.append("pos" if gi % 2 == 0 else "neg")
        grps.append(f"T{gi}")

leaky = from_lists(items, labs, grps, name="leaky")
res = audit(leaky, claimed_ceiling=0.60)

check("audit records the row count", res.n_audited == 100)
check("audit grades the leaky benchmark CRITICAL", res.worst == CRITICAL, res.worst)
checks_run = {f.check for f in res.findings}
check("multiplicity was checked", "multiplicity" in checks_run)
check("degenerate baselines were checked", "degenerate-baselines" in checks_run)
check("ceiling violation was checked", "ceiling-violation" in checks_run)
check("protocol gap was checked", "protocol-gap" in checks_run)
check("labels finding was emitted", "labels" in checks_run)

# REGRESSION. The first version of this tool graded exactly this benchmark
# "87 independent items in 100 rows -- little duplication detected", because
# it read one strict threshold on word trigrams. Five templates must not
# present as eighty-seven independent items.
mult = next(f for f in res.findings if f.check == "multiplicity")
check("REG-B1: templated set is not graded OK", mult.severity != OK, mult.headline)
check(
    "REG-B1: effective n recovers the template count",
    min(c for _, c, _ in res.effective_n_curve) <= 10,
    str(res.effective_n_curve),
)
check("REG-B1: effective n is reported as a range", res.effective_n_range[0] <= res.effective_n_range[1])
check("REG-B1: representation is disclosed", "shingle" in res.representation)

gap = next(f for f in res.findings if f.check == "protocol-gap")
check("protocol gap is detected", gap.severity in (CRITICAL, WARNING), gap.severity)
check("protocol gap is positive", gap.evidence["gap"] > 0.2, str(gap.evidence["gap"]))
check("row-wise beats grouped here", gap.evidence["row_wise_f1"] > gap.evidence["grouped_f1"])
check("grouping source is named", "grouping_source" in gap.evidence)
check("declared groups are preferred when present",
      leaky.group_column == "" and gap.evidence["n_groups"] == 5,
      str(gap.evidence["n_groups"]))

ceil = next(f for f in res.findings if f.check == "ceiling-violation")
check("ceiling violation fires below the shortcut score",
      ceil.severity == CRITICAL, ceil.severity)
check("violation reports the margin", ceil.evidence["margin"] > 0)

high = audit(leaky, claimed_ceiling=0.999)
hc = next(f for f in high.findings if f.check == "ceiling-violation")
check("no violation against an unreachable ceiling", hc.severity == OK, hc.severity)

no_ceiling = audit(leaky)
check("ceiling check is skipped when no ceiling is claimed",
      "ceiling-violation" not in {f.check for f in no_ceiling.findings})

# ===================================================================
# NEGATIVE CONTROLS. These matter more than the positives above.
# ===================================================================

words = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega koppa sampi digamma"
).split()
random.seed(3)
clean_items = [" ".join(random.sample(words, 12)) for _ in range(120)]
clean_labs = ["pos" if i % 2 else "neg" for i in range(120)]
clean = from_lists(clean_items, clean_labs, name="clean")
# 0.80 and not 0.60: on a balanced set, flag-everything scores 2p/(1+p) = 0.667
# arithmetically, so a ceiling below that is beaten without reading the items.
# The first version of this suite claimed 0.60 was a safe ceiling and the audit
# correctly flagged it; the audit was right and the test was wrong.
cres = audit(clean, claimed_ceiling=0.80)

check("clean benchmark is not graded CRITICAL", cres.worst != CRITICAL, cres.worst)
cm = next(f for f in cres.findings if f.check == "multiplicity")
check("clean: multiplicity is OK", cm.severity == OK, cm.headline)
check("clean: effective n equals the row count",
      cres.effective_n == 120, str(cres.effective_n))
check("clean: curve is flat", len({c for _, c, _ in cres.effective_n_curve}) == 1,
      str(cres.effective_n_curve))
cg = next(f for f in cres.findings if f.check == "protocol-gap")
check("clean: protocol gap is OK", cg.severity == OK, cg.headline)
check("clean: protocol gap is near zero", abs(cg.evidence["gap"]) < 0.15,
      str(cg.evidence["gap"]))
cc = next(f for f in cres.findings if f.check == "ceiling-violation")
check("clean: no ceiling violation", cc.severity == OK, cc.headline)
cb = next(f for f in cres.findings if f.check == "degenerate-baselines")
check("clean: no shortcut beats the trivial baseline", cb.severity == OK, cb.headline)

check("clean benchmark would pass a CI gate", cres.worst in (OK, "INFO"), cres.worst)

# ===================================================================
# graceful degradation
# ===================================================================

unlab = audit(from_lists(clean_items, name="unlabelled"))
ck = {f.check for f in unlab.findings}
check("unlabelled: multiplicity still reported", "multiplicity" in ck)
check("unlabelled: no baseline check", "degenerate-baselines" not in ck)
check("unlabelled: no ceiling check", "ceiling-violation" not in ck)
check("unlabelled: the omission is stated", any("label" in n for n in unlab.notes),
      str(unlab.notes))
check("unlabelled: effective n is still computed", unlab.effective_n > 0)

tiny = audit(from_lists(["a b c", "d e f"], ["x", "y"], name="tiny"))
check("tiny benchmark returns without crashing", tiny.n_audited == 2)
check("tiny benchmark says why nothing ran", any("10" in n for n in tiny.notes),
      str(tiny.notes))

multi = from_lists(clean_items[:60], [f"c{i % 4}" for i in range(60)], name="multiclass")
mres = audit(multi)
check("multiclass: audit completes", mres.n_audited == 60)
check("multiclass: baseline battery is skipped",
      "degenerate-baselines" not in {f.check for f in mres.findings})
check("multiclass: the reason is stated", any("distinct values" in n for n in mres.notes),
      str(mres.notes))

few_groups = from_lists(
    ["item number %d with some filler words here" % i for i in range(20)],
    ["pos" if i % 2 else "neg" for i in range(20)],
    ["g%d" % (i % 3) for i in range(20)],
    name="fewgroups",
)
fres = audit(few_groups, k_folds=5)
check("too few groups: gap is skipped, not faked",
      "protocol-gap" not in {f.check for f in fres.findings})
check("too few groups: the reason is stated",
      any("groups" in n for n in fres.notes), str(fres.notes))

skipped = from_lists(clean_items[:30], clean_labs[:30], name="skipped")
skipped.skipped_rows = [(0, "empty item")]
skipped.n_raw_rows = 31
sres = audit(skipped)
check("skipped rows produce a warning",
      any(f.check == "loading" and f.severity == WARNING for f in sres.findings))

# ===================================================================
# reporting
# ===================================================================

md = as_markdown(res)
check("markdown has a title", md.startswith("# Benchmark integrity audit"))
check("markdown states the verdict", "CRITICAL" in md)
check("markdown shows the effective-n curve", "Jaccard threshold" in md)
check("markdown names every finding", all(f.check in md for f in res.findings))
check("markdown discloses what is untested", "does and does not establish" in md)
check("markdown says the battery is incomplete", "incomplete" in md)
check("markdown mentions training contamination is out of scope",
      "contamination" in md)
check("markdown interpolates the row count", f"| {res.n_audited} |" in md)
check("markdown interpolates the representation", res.representation in md)

# A ceiling below the trivial baseline is a different diagnosis from leakage,
# and the report must say which. Both are CRITICAL; only one means the items
# encode their labels.
low = audit(clean, claimed_ceiling=0.40)
lc = next(f for f in low.findings if f.check == "ceiling-violation")
check("ceiling below trivial baseline is diagnosed as such",
      lc.evidence["diagnosis"] == "ceiling below trivial baseline", str(lc.evidence))
check("no violating probe beats the trivial baseline",
      max(v["f1"] for v in lc.evidence["violations"]) <= 0.667 + 1e-3,
      str(lc.evidence["violations"]))
check("leakage diagnosis is distinguished",
      ceil.evidence["diagnosis"] == "label-encoding shortcut", str(ceil.evidence))

txt = as_text(res)
check("text output names the benchmark", "leaky" in txt)
check("text output lists findings", "[CRITICAL" in txt)

js = json.loads(as_json(res))
check("json has the benchmark name", js["benchmark"] == "leaky")
check("json has the worst severity", js["worst_severity"] == CRITICAL)
check("json findings carry evidence", all("evidence" in f for f in js["findings"]))
check("json curve is a list of dicts",
      all(set(r) == {"threshold", "n_clusters", "duplication_rate"}
          for r in js["effective_n_curve"]))
check("json is sorted worst-first",
      js["findings"][0]["severity"] == CRITICAL)

tbl = cross_table([res, cres])
check("cross table has both rows", "leaky" in tbl and "clean" in tbl)
check("cross table is markdown", tbl.startswith("| Benchmark"))
check("cross table reports the ratio", "%" in tbl)

check("findings sort worst first",
      [f.severity for f in res.sorted_findings()]
      == sorted([f.severity for f in res.findings],
                key=lambda s: {"CRITICAL": 0, "WARNING": 1, "INFO": 2, "OK": 3}[s]))
check("by_severity filters", all(f.severity == CRITICAL
                                for f in res.by_severity(CRITICAL)))
check("Finding renders as a string", str(res.findings[0]).startswith("["))

def test_all_checks_passed() -> None:
    """Lets pytest run this file correctly.

    The checks above execute at import. Without this sentinel pytest would find
    no test function: a passing suite would report "no tests ran" (exit 5, a
    false red) and a failing one would exit during collection. This turns the
    recorded result into one pytest outcome.
    """
    assert not _FAIL, f"{len(_FAIL)} check(s) failed: " + "; ".join(_FAIL[:10])


if __name__ == "__main__":
    print(f"\n{_PASS} checks passed")
    if _FAIL:
        print(f"{len(_FAIL)} FAILED:")
        for f in _FAIL:
            print(f"  - {f}")
        sys.exit(1)
    print(f"all green ({_PASS} checks)")
