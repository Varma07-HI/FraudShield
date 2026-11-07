"""Metrics that matter for fraud: PR-AUC, recall at fixed precision, money saved."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def recall_at_precision(y_true, scores, min_precision=0.9):
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ok = precision >= min_precision
    return float(recall[ok].max()) if ok.any() else 0.0


def summarize(y_true, scores):
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "recall_at_precision_90": recall_at_precision(y_true, scores, 0.9),
    }


def threshold_table(y_true, scores, amounts, n_points=200):
    """For each threshold: how many alerts, frauds caught/missed, and the money involved."""
    y_true, scores, amounts = map(np.asarray, (y_true, scores, amounts))
    grid = np.unique(np.concatenate([
        np.linspace(0.0, 1.0, 101),
        np.quantile(scores, 1 - np.logspace(-5, -1, n_points // 2)),
    ]))
    rows = []
    for t in grid:
        flagged = scores >= t
        tp = flagged & (y_true == 1)
        rows.append({
            "threshold": float(t),
            "flagged": int(flagged.sum()),
            "caught": int(tp.sum()),
            "missed": int(((~flagged) & (y_true == 1)).sum()),
            "caught_amount": float(amounts[tp].sum()),
            "missed_amount": float(amounts[(~flagged) & (y_true == 1)].sum()),
        })
    return pd.DataFrame(rows)


def net_savings(table, review_cost, loss_multiplier=1.0):
    """Money saved = fraud value caught - cost of reviewing every alert."""
    return loss_multiplier * table["caught_amount"] - review_cost * table["flagged"]


def best_threshold(table, review_cost, loss_multiplier=1.0):
    savings = net_savings(table, review_cost, loss_multiplier)
    return float(table.loc[savings.idxmax(), "threshold"])