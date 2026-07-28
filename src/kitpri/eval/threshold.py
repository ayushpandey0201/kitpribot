"""
Threshold sweep and selection. Thresholds are PER-MODEL config values —
sweep on VALIDATION probabilities only (the v4 test set was never used for
threshold selection; keep it that way).
"""

from __future__ import annotations

from kitpri.eval.metrics import binary_metrics


def sweep_thresholds(probs: list[float], labels: list[float],
                     start: float = 0.05, stop: float = 0.95, step: float = 0.01) -> dict:
    """Returns {'best': {...}, 'sweep': [per-threshold metric rows]}."""
    rows = []
    t = start
    while t <= stop + 1e-9:
        m = binary_metrics(probs, labels, threshold=round(t, 4))
        m["threshold"] = round(t, 4)
        rows.append(m)
        t += step
    best = max(rows, key=lambda r: r["f1"])
    return {"best": best, "sweep": rows}
