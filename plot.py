"""Minimal SVG plots, written by hand.

No matplotlib, no plotly, nothing to install. Three plots carry an evals
course: the score distributions with the threshold on them, the
precision-recall-versus-threshold curve, and the reliability diagram. Each
writes a standalone .svg you can open in any browser or drop into slides.

Both themes: every plot uses a transparent background and a stroke colour that
reads on white and on dark, because half your audience will have dark mode on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .calibration import reliability_bins
from .metrics import MetricRow

W, H = 720, 400
PAD_L, PAD_R, PAD_T, PAD_B = 64, 20, 28, 48

INK = "#6b7280"      # axes and text: mid grey, legible on both themes
FAIL = "#dc6803"     # orange
PASS = "#2e7fd4"     # blue
MARK = "#9333ea"     # threshold marker
GRID = "#9ca3af"


def _header(title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" font-family="system-ui,sans-serif">',
        f'<text x="{PAD_L}" y="18" font-size="13" fill="{INK}" '
        f'font-weight="600">{title}</text>',
    ]


def _axes(xlabel: str, ylabel: str) -> list[str]:
    x0, y0, x1, y1 = PAD_L, H - PAD_B, W - PAD_R, PAD_T
    return [
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{INK}" '
        f'stroke-width="1"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{INK}" '
        f'stroke-width="1"/>',
        f'<text x="{(x0 + x1) / 2}" y="{H - 12}" font-size="11" fill="{INK}" '
        f'text-anchor="middle">{xlabel}</text>',
        f'<text x="14" y="{(y0 + y1) / 2}" font-size="11" fill="{INK}" '
        f'text-anchor="middle" transform="rotate(-90 14 {(y0 + y1) / 2})">'
        f"{ylabel}</text>",
    ]


def _sx(v: float) -> float:
    return PAD_L + v * (W - PAD_L - PAD_R)


def _sy(v: float) -> float:
    return (H - PAD_B) - v * (H - PAD_B - PAD_T)


def _ticks() -> list[str]:
    out = []
    for i in range(6):
        v = i / 5
        out.append(
            f'<text x="{_sx(v)}" y="{H - PAD_B + 15}" font-size="10" '
            f'fill="{INK}" text-anchor="middle">{v:.1f}</text>'
        )
        out.append(
            f'<text x="{PAD_L - 8}" y="{_sy(v) + 3}" font-size="10" '
            f'fill="{INK}" text-anchor="end">{v:.1f}</text>'
        )
        if i:
            out.append(
                f'<line x1="{PAD_L}" y1="{_sy(v)}" x2="{W - PAD_R}" '
                f'y2="{_sy(v)}" stroke="{GRID}" stroke-width="0.5" '
                f'stroke-dasharray="2 3" opacity="0.4"/>'
            )
    return out


def _write(path: str | Path, parts: list[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n</svg>\n", encoding="utf-8")
    return path


def _histogram(values: Sequence[float], n_bins: int) -> list[int]:
    counts = [0] * n_bins
    for v in values:
        b = min(n_bins - 1, max(0, int(v * n_bins)))
        counts[b] += 1
    return counts


def score_distributions(
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float,
    path: str | Path,
    title: str = "Score distributions",
    n_bins: int = 20,
) -> Path:
    """The single most useful plot in evals.

    Two histograms, failures and passes, with the threshold drawn on top. The
    overlap IS the problem: no threshold can separate what the model does not
    separate, and one look tells you whether to tune the threshold or go back
    and fix the judge prompt.
    """
    fails = [s for s, y in zip(scores, labels) if y == 1]
    passes = [s for s, y in zip(scores, labels) if y == 0]
    hf, hp = _histogram(fails, n_bins), _histogram(passes, n_bins)
    peak = max([*hf, *hp, 1])

    parts = _header(title) + _axes("judge score", "traces")
    parts += _ticks()
    bw = (W - PAD_L - PAD_R) / n_bins

    for counts, colour, off in ((hp, PASS, 0.0), (hf, FAIL, 0.5)):
        for i, c in enumerate(counts):
            if not c:
                continue
            h = (c / peak) * (H - PAD_B - PAD_T)
            x = PAD_L + i * bw + off * bw
            parts.append(
                f'<rect x="{x:.1f}" y="{H - PAD_B - h:.1f}" '
                f'width="{bw * 0.5:.1f}" height="{h:.1f}" fill="{colour}" '
                f'opacity="0.75"/>'
            )

    tx = _sx(threshold)
    parts.append(
        f'<line x1="{tx:.1f}" y1="{PAD_T}" x2="{tx:.1f}" y2="{H - PAD_B}" '
        f'stroke="{MARK}" stroke-width="2" stroke-dasharray="5 3"/>'
    )
    parts.append(
        f'<text x="{tx + 5:.1f}" y="{PAD_T + 12}" font-size="10" fill="{MARK}">'
        f"threshold {threshold:.3f}</text>"
    )
    parts.append(
        f'<rect x="{W - PAD_R - 110}" y="{PAD_T + 2}" width="9" height="9" '
        f'fill="{FAIL}" opacity="0.75"/>'
        f'<text x="{W - PAD_R - 97}" y="{PAD_T + 10}" font-size="10" '
        f'fill="{INK}">failure</text>'
        f'<rect x="{W - PAD_R - 110}" y="{PAD_T + 16}" width="9" height="9" '
        f'fill="{PASS}" opacity="0.75"/>'
        f'<text x="{W - PAD_R - 97}" y="{PAD_T + 24}" font-size="10" '
        f'fill="{INK}">pass</text>'
    )
    return _write(path, parts)


def pr_vs_threshold(
    rows: Sequence[MetricRow],
    path: str | Path,
    min_precision: float | None = None,
    min_recall: float | None = None,
    title: str = "Precision and recall against threshold",
) -> Path:
    """Precision and recall as the threshold moves, with the feasible band.

    The shaded region is where both floors are met. If it is empty, the plot
    has told you the model is disqualified before you tuned anything.
    """
    rows = sorted(rows, key=lambda r: r.threshold)
    parts = _header(title) + _axes("threshold", "metric")
    parts += _ticks()

    if min_precision is not None and min_recall is not None:
        ok = [r for r in rows if r.precision >= min_precision and r.recall >= min_recall]
        if ok:
            lo, hi = _sx(ok[0].threshold), _sx(ok[-1].threshold)
            parts.append(
                f'<rect x="{lo:.1f}" y="{PAD_T}" width="{max(1.0, hi - lo):.1f}" '
                f'height="{H - PAD_B - PAD_T}" fill="#10b981" opacity="0.12"/>'
            )
            parts.append(
                f'<text x="{(lo + hi) / 2:.1f}" y="{PAD_T + 12}" font-size="10" '
                f'fill="#059669" text-anchor="middle">both floors met</text>'
            )

    for attr, colour, label in (
        ("precision", PASS, "precision"),
        ("recall", FAIL, "recall"),
    ):
        pts = " ".join(
            f"{_sx(r.threshold):.1f},{_sy(getattr(r, attr)):.1f}" for r in rows
        )
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{colour}" '
            f'stroke-width="2"/>'
        )
        last = rows[-1]
        parts.append(
            f'<text x="{_sx(last.threshold) - 4:.1f}" '
            f'y="{_sy(getattr(last, attr)) - 6:.1f}" font-size="10" '
            f'fill="{colour}" text-anchor="end">{label}</text>'
        )
    return _write(path, parts)


def reliability_diagram(
    scores: Sequence[float],
    labels: Sequence[int],
    path: str | Path,
    n_bins: int = 5,
    title: str = "Reliability diagram",
    calibrated: Sequence[float] | None = None,
) -> Path:
    """Predicted probability against observed failure rate.

    The diagonal is perfect calibration. Above it the model under-predicts
    failure; below it, it cries wolf. Point area scales with bin size, so a
    wildly off point holding four traces does not read as louder than a
    near-perfect one holding forty.
    """
    parts = _header(title) + _axes("mean predicted probability", "observed failure rate")
    parts += _ticks()
    parts.append(
        f'<line x1="{_sx(0)}" y1="{_sy(0)}" x2="{_sx(1)}" y2="{_sy(1)}" '
        f'stroke="{INK}" stroke-width="1" stroke-dasharray="4 4" opacity="0.6"/>'
    )

    series = [(reliability_bins(scores, labels, n_bins), FAIL, "raw")]
    if calibrated is not None:
        series.append((reliability_bins(calibrated, labels, n_bins), PASS, "calibrated"))

    for bins, colour, label in series:
        if not bins:
            continue
        pts = " ".join(f"{_sx(b.mean_pred):.1f},{_sy(b.observed_rate):.1f}" for b in bins)
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{colour}" '
            f'stroke-width="1.5" opacity="0.8"/>'
        )
        for b in bins:
            r = 3.0 + 7.0 * (b.n / max(1, max(x.n for x in bins))) ** 0.5
            parts.append(
                f'<circle cx="{_sx(b.mean_pred):.1f}" '
                f'cy="{_sy(b.observed_rate):.1f}" r="{r:.1f}" fill="{colour}" '
                f'opacity="0.8"/>'
            )
        parts.append(
            f'<text x="{_sx(bins[-1].mean_pred) + 8:.1f}" '
            f'y="{_sy(bins[-1].observed_rate) + 3:.1f}" font-size="10" '
            f'fill="{colour}">{label}</text>'
        )
    return _write(path, parts)
