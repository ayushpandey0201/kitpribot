"""Binary classification metrics (no sklearn dependency). Label 1 = cooking."""

from __future__ import annotations

from kitpri.labels import label_from_probability


def binary_metrics(probs: list[float], labels: list[float], threshold: float) -> dict:
    preds = [label_from_probability(p, threshold) for p in probs]
    labels_i = [int(round(l)) for l in labels]

    tp = sum(1 for p, l in zip(preds, labels_i) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(preds, labels_i) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(preds, labels_i) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels_i) if p == 0 and l == 1)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / len(labels_i) if labels_i else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "auc": _auc(probs, labels_i),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def _auc(probs: list[float], labels: list[int]) -> float:
    """Rank-based AUC (Mann-Whitney U), ties handled by midrank."""
    pos = [p for p, l in zip(probs, labels) if l == 1]
    neg = [p for p, l in zip(probs, labels) if l == 0]
    if not pos or not neg:
        return 0.0
    ranked = sorted((p, i) for i, p in enumerate(probs))
    ranks = [0.0] * len(probs)
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        midrank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[ranked[k][1]] = midrank
        i = j + 1
    pos_rank_sum = sum(r for r, l in zip(ranks, labels) if l == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
