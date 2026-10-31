"""Évaluation des détecteurs, au niveau des personnes (étape 4).

Un détecteur fournit un score par personne et par jour : plus il est élevé,
plus le jour est anormal. Il y a alarme quand score > seuil, à partir du jour
MONITOR_START (avant, la baseline sert de référence : aucune alarme ne compte).

Pour chaque personne :
- détection : une alarme à partir de d0, chez une personne qui dérive ;
- fausse alarme : une alarme avant d0, ou n'importe quand chez un témoin.
Une personne compte au plus une fois comme détection et une fois comme fausse alarme.

Attention : un détecteur qui reste en alarme sans interruption avant d0 compte à
la fois comme fausse alarme et, dès d0, comme détection de délai 0.
"""

import numpy as np
import pandas as pd

from src.config import (
    AMPLITUDES_SIGMA,
    DRIFT_TYPES,
    MONITOR_START,
    TARGET_FALSE_ALARM_RATE,
)


def person_outcomes(scores, truth, threshold, monitor_start=MONITOR_START):
    """Résultat par personne. `scores` : colonnes person_id, day, score."""
    matrix = scores.pivot(index="person_id", columns="day", values="score")
    days = matrix.columns.to_numpy()
    info = truth.set_index("person_id").loc[matrix.index]

    d0 = info["d0"].to_numpy(dtype=float, na_value=np.nan)
    d0 = np.where(np.isnan(d0), np.inf, d0)[:, None]  # témoin : d0 = infini

    alarm = (matrix.to_numpy() > threshold) & (days >= monitor_start)
    false_alarm = (alarm & (days < d0)).any(axis=1)
    after_d0 = alarm & (days >= d0)
    detected = after_d0.any(axis=1)
    first_alarm_day = days[after_d0.argmax(axis=1)]  # sans signification si non détecté
    delay = np.where(detected, first_alarm_day - d0[:, 0], np.nan)

    return pd.DataFrame(
        {
            "person_id": matrix.index.to_numpy(),
            "drift_type": info["drift_type"].to_numpy(),
            "level": info["level"].to_numpy(),
            "is_drifted": np.isfinite(d0[:, 0]),
            "false_alarm": false_alarm,
            "detected": detected,
            "delay": delay,
        }
    )


def summarize(outcomes):
    """Précision, rappel, F1, délai et taux de fausses alarmes."""
    drifted = outcomes[outcomes["is_drifted"]]
    controls = outcomes[~outcomes["is_drifted"]]
    tp = int(drifted["detected"].sum())
    fn = len(drifted) - tp
    fp = int(outcomes["false_alarm"].sum())
    f1_denominator = 2 * tp + fp + fn
    delays = drifted["delay"].dropna()  # seulement les dérives détectées
    return {
        "n_drifted": len(drifted),
        "n_controls": len(controls),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / len(drifted) if len(drifted) else np.nan,
        "f1": 2 * tp / f1_denominator if f1_denominator else np.nan,
        "false_alarm_rate": controls["false_alarm"].mean() if len(controls) else np.nan,
        "median_delay": delays.median() if len(delays) else np.nan,
        "mean_delay": delays.mean() if len(delays) else np.nan,
    }


def evaluate(scores, truth, threshold, monitor_start=MONITOR_START):
    """Raccourci : scores + seuil -> dictionnaire de métriques."""
    return summarize(person_outcomes(scores, truth, threshold, monitor_start))


def calibrate_and_evaluate(val_scores, test_scores, truth):
    """Règle le seuil sur la validation, puis mesure sur le test sans y retoucher."""
    threshold = calibrate_threshold(val_scores, truth)
    test_outcomes = person_outcomes(test_scores, truth, threshold)
    return {
        "threshold": threshold,
        "val": evaluate(val_scores, truth, threshold),
        "test": summarize(test_outcomes),
        "test_outcomes": test_outcomes,
    }


def summarize_by_case(outcomes):
    """Taux de détection et délai médian par type et par niveau de dérive."""
    drifted = outcomes[outcomes["is_drifted"]]
    table = drifted.groupby(["drift_type", "level"]).agg(
        n=("detected", "size"),
        detection_rate=("detected", "mean"),
        median_delay=("delay", "median"),
    )
    order = pd.MultiIndex.from_product(
        [DRIFT_TYPES, list(AMPLITUDES_SIGMA)], names=["drift_type", "level"]
    )
    return table.reindex(order)


def calibrate_threshold(
    scores, truth, target=TARGET_FALSE_ALARM_RATE, monitor_start=MONITOR_START
):
    """Plus petit seuil tel qu'au plus `target` des témoins aient une alarme.

    À calculer sur la VALIDATION uniquement. Seuls les témoins de `scores` servent.
    """
    control_ids = truth.loc[truth["drift_type"] == "none", "person_id"]
    monitored = scores[scores["person_id"].isin(control_ids) & (scores["day"] >= monitor_start)]
    peaks = monitored.groupby("person_id")["score"].max().to_numpy()  # score max de chaque témoin
    n_allowed = int(np.floor(target * len(peaks) + 1e-9))  # témoins autorisés à dépasser le seuil
    return np.sort(peaks)[::-1][n_allowed]
