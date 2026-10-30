"""Simulateur de journaux d'activités quotidiennes, SANS dérive (étape 1).

Chaque ligne du journal est un jour d'une personne. Les heures sont en minutes
après 18h : 300 = 23h00, 390 = 00h30 (pas de saut à minuit).
"""

import numpy as np
import pandas as pd

from src.config import (
    AR_RHO,
    N_DAYS,
    N_PERSONS,
    WEEKEND_BED_SHIFT_MIN,
    WEEKEND_SLEEP_EXTRA_MIN,
    WEEKEND_SOCIAL_FACTOR,
)


def draw_person_profiles(rng, n_persons):
    """Tire une seule fois les habitudes de chaque personne."""
    return pd.DataFrame(
        {
            "person_id": np.arange(n_persons),
            "bed_mean": rng.normal(300, 40, n_persons),  # coucher moyen (min après 18h)
            "bed_sd": rng.uniform(25, 50, n_persons),  # variabilité quotidienne du coucher
            "sleep_mean": rng.normal(450, 40, n_persons),  # durée de sommeil moyenne
            "sleep_sd": rng.uniform(25, 50, n_persons),  # variabilité quotidienne du sommeil
            "meal_p": rng.uniform(0.80, 0.95, n_persons),  # proba de prendre chaque repas
            "social_rate": rng.uniform(3, 8, n_persons),  # interactions moyennes par jour
        }
    )


def ar1_noise(rng, n_persons, n_days, rho):
    """Bruit centré réduit où chaque jour dépend un peu de la veille."""
    fresh = rng.standard_normal((n_persons, n_days))
    noise = np.empty_like(fresh)
    noise[:, 0] = fresh[:, 0]
    for t in range(1, n_days):
        noise[:, t] = rho * noise[:, t - 1] + np.sqrt(1 - rho**2) * fresh[:, t]
    return noise


def is_weekend(n_days):
    """Vrai pour les jours 5 et 6 de chaque semaine (le jour 0 est un lundi)."""
    return (np.arange(n_days) % 7) >= 5


def simulate_normal(rng, n_persons=N_PERSONS, n_days=N_DAYS):
    """Retourne (journal, profils) : le journal quotidien et les habitudes de chacun."""
    profiles = draw_person_profiles(rng, n_persons)
    weekend = is_weekend(n_days)

    # [:, None] transforme (n_persons,) en (n_persons, 1) pour combiner avec (n_days,)
    bed_mean = profiles["bed_mean"].to_numpy()[:, None]
    bed_sd = profiles["bed_sd"].to_numpy()[:, None]
    sleep_mean = profiles["sleep_mean"].to_numpy()[:, None]
    sleep_sd = profiles["sleep_sd"].to_numpy()[:, None]
    meal_p = profiles["meal_p"].to_numpy()[:, None]
    social_rate = profiles["social_rate"].to_numpy()[:, None]

    bed = (
        bed_mean
        + bed_sd * ar1_noise(rng, n_persons, n_days, AR_RHO)
        + WEEKEND_BED_SHIFT_MIN * weekend
    )
    sleep = (
        sleep_mean
        + sleep_sd * ar1_noise(rng, n_persons, n_days, AR_RHO)
        + WEEKEND_SLEEP_EXTRA_MIN * weekend
    )
    wake = bed + sleep

    meals = rng.binomial(3, meal_p, size=(n_persons, n_days))
    daily_social_rate = social_rate * (1 + (WEEKEND_SOCIAL_FACTOR - 1) * weekend)
    social = rng.poisson(daily_social_rate)

    journal = pd.DataFrame(
        {
            "person_id": np.repeat(np.arange(n_persons), n_days),
            "day": np.tile(np.arange(n_days), n_persons),
            "bedtime_min": np.round(bed).astype(int).ravel(),
            "waketime_min": np.round(wake).astype(int).ravel(),
            "meals": meals.ravel(),
            "social": social.ravel(),
        }
    )
    return journal, profiles
