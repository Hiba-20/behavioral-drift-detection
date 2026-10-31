"""Détecteurs statistiques, sans réseau de neurones (étape 5).

- z-score glissant : écart de la moyenne des derniers jours à la baseline personnelle ;
- CUSUM : somme cumulée des petits écarts qui persistent.

Les deux travaillent sur les colonnes *_z (déjà exprimées par rapport à la baseline
personnelle) et gardent la plus grande valeur parmi les 4 variables, dans les deux
sens (hausse ou baisse). Ils ne savent pas quelle dérive chercher.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.config import BASELINE_DAYS, CUSUM_K, MONITOR_START, ROLLING_WINDOW
from src.dataset import Z_COLUMNS

Z_NAMES = list(Z_COLUMNS.values())


def to_array(dataset):
    """Retourne (person_id, tableau z de forme personnes x jours x variables)."""
    ordered = dataset.sort_values(["person_id", "day"])
    ids = ordered["person_id"].unique()
    return ids, ordered[Z_NAMES].to_numpy().reshape(len(ids), -1, len(Z_NAMES))


def to_scores_frame(ids, scores):
    """Remet un tableau de scores (personnes x jours) au format long attendu par metrics."""
    n_days = scores.shape[1]
    return pd.DataFrame(
        {
            "person_id": np.repeat(ids, n_days),
            "day": np.tile(np.arange(n_days), len(ids)),
            "score": scores.ravel(),
        }
    )


def rolling_mean(z, window):
    """Moyenne des `window` derniers jours (jour courant inclus) ; NaN les premiers jours."""
    out = np.full(z.shape, np.nan)
    out[:, window - 1 :] = sliding_window_view(z, window, axis=1).mean(axis=-1)
    return out


def fit_rolling_scale(train_dataset, window=ROLLING_WINDOW):
    """Fluctuation normale de la moyenne glissante, par variable.

    Mesurée sur les jours de baseline des personnes du TRAIN uniquement.
    """
    _, z = to_array(train_dataset)
    means = rolling_mean(z[:, :BASELINE_DAYS], window)[:, window - 1 :]
    return means.std(axis=(0, 1))


def rolling_zscore_scores(dataset, scale, window=ROLLING_WINDOW):
    """Score = plus grand écart (en valeur absolue) de la moyenne glissante, en fluctuations normales."""
    ids, z = to_array(dataset)
    standardized = np.abs(rolling_mean(z, window)) / scale
    return to_scores_frame(ids, standardized.max(axis=2))


def cusum_scores(dataset, k=CUSUM_K, start=MONITOR_START):
    """Score = contenu de la plus pleine des 8 « tirelires » (4 variables x hausse / baisse)."""
    ids, z = to_array(dataset)
    up = np.zeros_like(z[:, 0])
    down = np.zeros_like(z[:, 0])
    scores = np.zeros(z.shape[:2])
    for t in range(start, z.shape[1]):
        up = np.maximum(0, up + z[:, t] - k)
        down = np.maximum(0, down - z[:, t] - k)
        scores[:, t] = np.maximum(up, down).max(axis=1)
    return to_scores_frame(ids, scores)
