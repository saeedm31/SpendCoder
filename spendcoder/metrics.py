"""How good is a classifier, in numbers a data scientist and an accounting lead both understand."""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .taxonomy import CATEGORIES


def scores(y_true, probs):
    """Accuracy and F1 from predicted probabilities (rows = lines, columns = categories)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(probs).argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=range(len(CATEGORIES)), zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def automation(y_true, probs, thresholds=(0.5, 0.7, 0.8, 0.9, 0.95)):
    """The business view: if lines above a confidence threshold are coded automatically and the rest go to a
    person, what share is automated and how accurate are the automatic ones?"""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    rows = []
    for t in thresholds:
        auto = conf >= t
        rows.append({
            "threshold": t,
            "automated_pct": round(100 * float(auto.mean()), 1),
            "accuracy_of_automated_pct": round(100 * float((pred[auto] == y_true[auto]).mean()), 1) if auto.any() else None,
            "to_review_pct": round(100 * float((~auto).mean()), 1),
        })
    return rows


def threshold_for_accuracy(y_true, probs, target=0.95):
    """Lowest confidence threshold whose automatic lines reach the target accuracy (pick it on validation)."""
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    for t in np.round(np.arange(0.30, 0.995, 0.01), 2):
        auto = conf >= t
        if auto.sum() >= 20 and (pred[auto] == y_true[auto]).mean() >= target:
            return float(t)
    return 0.99


def report(y_true, probs):
    """Per-category precision, recall and F1 as a dict (for the model card)."""
    return classification_report(np.asarray(y_true), np.asarray(probs).argmax(axis=1),
                                 labels=range(len(CATEGORIES)), target_names=CATEGORIES,
                                 output_dict=True, zero_division=0)
