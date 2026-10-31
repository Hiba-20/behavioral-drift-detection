"""Détecteurs scikit-learn sur des agrégats de 7 jours (étape 6).

Chaque ligne = une personne, un jour, décrit par ses 7 derniers jours :
pour chaque variable normalisée, la moyenne, l'écart-type et la pente.
- Random Forest : supervisé, apprend avec les étiquettes `label` ;
- Isolation Forest : non supervisé, appris sur des journées normales seulement.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from src.baseline_stats import Z_NAMES, to_array, to_scores_frame
from src.config import MONITOR_START, ROLLING_WINDOW, SEED
from src.metrics import calibrate_threshold, evaluate

WINDOW = ROLLING_WINDOW
STATS = ("mean", "std", "slope")
FEATURE_NAMES = [f"{z}_{stat}" for z in Z_NAMES for stat in STATS]
FOREST_GRID = [
    {"max_depth": depth, "min_samples_leaf": leaf}
    for depth in (4, 8, None)
    for leaf in (20, 100)
]


def window_features(dataset, window=WINDOW):
    """Retourne (person_id, tableau personnes x jours x 12) ; NaN avant le 7e jour."""
    ids, z = to_array(dataset)
    n_persons, n_days, n_variables = z.shape
    windows = sliding_window_view(z, window, axis=1)  # personnes x fenêtres x variables x jours
    position = np.arange(window) - (window - 1) / 2  # -3 ... +3 : position dans la fenêtre
    slope = (windows * position).sum(axis=-1) / (position**2).sum()
    stats = np.stack([windows.mean(axis=-1), windows.std(axis=-1), slope], axis=-1)

    features = np.full((n_persons, n_days, n_variables * len(STATS)), np.nan)
    features[:, window - 1 :] = stats.reshape(n_persons, n_days - window + 1, -1)
    return ids, features


def training_table(dataset):
    """(X, y) : une ligne par personne et par jour de surveillance, étiquette = `label`."""
    ids, features = window_features(dataset)
    labels = dataset.sort_values(["person_id", "day"])["label"].to_numpy()
    labels = labels.reshape(len(ids), -1)
    kept = np.arange(features.shape[1]) >= MONITOR_START
    return features[:, kept].reshape(-1, features.shape[2]), labels[:, kept].ravel()


def window_scores(dataset, predict):
    """Applique `predict` (lignes de 12 colonnes -> score) à chaque jour ayant 7 jours d'historique."""
    ids, features = window_features(dataset)
    n_persons, n_days, n_columns = features.shape
    scores = np.full((n_persons, n_days), np.nan)
    valid = features[:, WINDOW - 1 :].reshape(-1, n_columns)
    scores[:, WINDOW - 1 :] = predict(valid).reshape(n_persons, -1)
    return to_scores_frame(ids, scores)


def fit_forest(train, shuffle_labels=False, **params):
    """Random Forest entraînée sur les personnes du train."""
    X, y = training_table(train)
    if shuffle_labels:  # contrôle : des étiquettes sans lien avec les données
        y = np.random.default_rng(SEED).permutation(y)
    settings = {"n_estimators": 200, "class_weight": "balanced_subsample", "n_jobs": -1}
    return RandomForestClassifier(random_state=SEED, **{**settings, **params}).fit(X, y)


def forest_scores(model, dataset):
    """Score = proportion d'arbres qui votent « dérive »."""
    return window_scores(dataset, lambda X: model.predict_proba(X)[:, 1])


def choose_forest(train, val, truth, grid=FOREST_GRID):
    """Essaie chaque réglage ; garde celui qui a le meilleur F1 sur la VALIDATION."""
    tried = []
    for params in grid:
        scores = forest_scores(fit_forest(train, **params), val)
        threshold = calibrate_threshold(scores, truth)
        tried.append({**params, "val_f1": evaluate(scores, truth, threshold)["f1"]})
    best = max(tried, key=lambda row: row["val_f1"])
    return {k: best[k] for k in ("max_depth", "min_samples_leaf")}, tried


def fit_isolation_forest(train):
    """Isolation Forest apprise uniquement sur les journées normales (label = 0) du train."""
    X, y = training_table(train)
    return IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1).fit(X[y == 0])


def isolation_scores(model, dataset):
    """Score = anomalie (plus haut = plus anormal)."""
    return window_scores(dataset, lambda X: -model.score_samples(X))
