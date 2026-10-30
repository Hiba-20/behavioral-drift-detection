"""Injection de dérives comportementales connues et vérité terrain (étape 2).

Trois dérives, toutes progressives à partir du jour d0 :
- sleep  : le coucher glisse plus tard, l'heure de lever ne change pas ;
- meals  : chaque repas a une probabilité croissante d'être sauté ;
- social : chaque interaction a une probabilité croissante de ne pas avoir lieu.

L'ampleur est exprimée en écarts-types de la personne (voir config.AMPLITUDES_SIGMA).
"""

import numpy as np
import pandas as pd

from src.config import (
    AMPLITUDES_SIGMA,
    CONTROL_FRACTION,
    D0_RANGE,
    DRIFT_TYPES,
    RAMP_DAYS,
    WEEKEND_SOCIAL_FACTOR,
)


def assign_drifts(rng, n_persons, amplitudes=AMPLITUDES_SIGMA):
    """Table de vérité terrain : une ligne par personne (témoin ou dérivante)."""
    n_drifted = n_persons - round(CONTROL_FRACTION * n_persons)
    cells = [(drift_type, level) for drift_type in DRIFT_TYPES for level in amplitudes]
    drifted_ids = rng.permutation(n_persons)[:n_drifted]

    drift_type = np.full(n_persons, "none", dtype=object)
    level = np.full(n_persons, "none", dtype=object)
    amplitude = np.zeros(n_persons)
    d0 = np.full(n_persons, np.nan)

    for i, person in enumerate(drifted_ids):
        drift_type[person], level[person] = cells[i % len(cells)]
        amplitude[person] = amplitudes[level[person]]
    d0[drifted_ids] = rng.integers(D0_RANGE[0], D0_RANGE[1] + 1, n_drifted)

    return pd.DataFrame(
        {
            "person_id": np.arange(n_persons),
            "drift_type": drift_type,
            "level": level,
            "amplitude_sigma": amplitude,
            "d0": pd.array(d0, dtype="Int64"),  # <NA> pour les témoins
        }
    )


def weekly_mean_social_rate(profiles):
    """Taux moyen d'interactions sur la semaine : 5 jours normaux + 2 jours de week-end."""
    weekend_share = 2 / 7
    return profiles["social_rate"].to_numpy() * (
        1 + (WEEKEND_SOCIAL_FACTOR - 1) * weekend_share
    )


def drift_ramp(d0, n_days, ramp_days=RAMP_DAYS):
    """Matrice (personnes x jours) : 0 avant d0, puis monte vers 1 en `ramp_days` jours.

    d0 est un vecteur de floats, NaN pour les témoins (leur ligne reste à 0).
    """
    days = np.arange(n_days)[None, :]
    ramp = np.clip((days - d0[:, None]) / ramp_days, 0, 1)
    return np.nan_to_num(ramp)


def inject_drifts(rng, journal, profiles, truth, ramp_days=RAMP_DAYS):
    """Retourne une copie du journal avec les dérives de `truth` appliquées."""
    n_persons = len(profiles)
    n_days = journal["day"].nunique()
    shape = (n_persons, n_days)

    d0 = truth["d0"].to_numpy(dtype=float, na_value=np.nan)
    strength = drift_ramp(d0, n_days, ramp_days) * truth["amplitude_sigma"].to_numpy()[:, None]

    def only(drift_type):
        """Force de la dérive de ce type seulement (0 pour les autres personnes)."""
        return strength * (truth["drift_type"] == drift_type).to_numpy()[:, None]

    # Sommeil : coucher décalé de (force x variabilité du coucher) minutes.
    bed_sd = profiles["bed_sd"].to_numpy()[:, None]
    bed = journal["bedtime_min"].to_numpy().reshape(shape)
    bed_new = bed + np.round(only("sleep") * bed_sd).astype(int)

    # Repas : le nombre moyen baisse de (force x écart-type des repas).
    p = profiles["meal_p"].to_numpy()[:, None]
    meal_sd = np.sqrt(3 * p * (1 - p))
    skip_prob = np.clip(only("meals") * meal_sd / (3 * p), 0, 1)
    meals_new = rng.binomial(journal["meals"].to_numpy().reshape(shape), 1 - skip_prob)

    # Interactions : le nombre moyen baisse de (force x écart-type de Poisson = racine du taux).
    rate = weekly_mean_social_rate(profiles)[:, None]
    drop_prob = np.clip(only("social") / np.sqrt(rate), 0, 1)
    social_new = rng.binomial(journal["social"].to_numpy().reshape(shape), 1 - drop_prob)

    drifted = journal.copy()
    drifted["bedtime_min"] = bed_new.ravel()
    drifted["meals"] = meals_new.ravel()
    drifted["social"] = social_new.ravel()
    return drifted
