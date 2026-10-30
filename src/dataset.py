"""Construction du jeu de données : journal dérivé, étiquettes, découpe par personne,
normalisation par la baseline personnelle et sauvegarde (étape 3).
"""

import math

import numpy as np
import pandas as pd

from src.config import (
    BASELINE_DAYS,
    BASELINE_STD_FLOOR,
    DATA_DIR,
    LABEL_MIN_RAMP,
    N_PERSONS,
    RAMP_DAYS,
    SEED,
    SPLIT_FRACTIONS,
    make_rng,
)
from src.drift import assign_drifts, inject_drifts
from src.simulator import simulate_normal

# Variable brute -> nom de la colonne normalisée.
Z_COLUMNS = {
    "bedtime_min": "bedtime_z",
    "waketime_min": "waketime_z",
    "meals": "meals_z",
    "social": "social_z",
}


def split_persons(rng, truth, fractions=SPLIT_FRACTIONS):
    """Ajoute à la vérité terrain la colonne `split` (train / val / test), par personne.

    La découpe est faite séparément dans chaque case (type x niveau, et témoins)
    pour que chaque groupe ait le même mélange de dérives.
    """
    split = np.full(len(truth), "test", dtype=object)  # le test reçoit le reste
    strata = truth["drift_type"] + "/" + truth["level"]
    for _, members in truth.groupby(strata):
        ids = rng.permutation(members["person_id"].to_numpy())
        n_train = round(fractions["train"] * len(ids))
        n_val = round(fractions["val"] * len(ids))
        split[ids[:n_train]] = "train"
        split[ids[n_train : n_train + n_val]] = "val"

    with_split = truth.copy()
    with_split["split"] = split
    return with_split


def add_labels(journal, truth, min_ramp=LABEL_MIN_RAMP):
    """Ajoute `label` : 1 à partir du jour où la dérive est « assez avancée », 0 sinon.

    Les témoins ont toujours label = 0. Cette colonne sert à entraîner et à
    évaluer, jamais d'entrée à un modèle.
    """
    start = truth["d0"].to_numpy(dtype=float, na_value=np.nan) + math.ceil(min_ramp * RAMP_DAYS)
    start_per_row = start[journal["person_id"].to_numpy()]  # NaN pour les témoins

    labeled = journal.copy()
    labeled["label"] = (journal["day"].to_numpy() >= start_per_row).astype(int)
    return labeled


def normalize_by_baseline(journal):
    """Ajoute les colonnes `*_z` : écart à la baseline personnelle, en écarts-types.

    La baseline d'une personne = ses BASELINE_DAYS premiers jours (sans dérive).
    """
    variables = list(Z_COLUMNS)
    baseline = journal[journal["day"] < BASELINE_DAYS].groupby("person_id")[variables]
    mean = baseline.mean()
    std = baseline.std().clip(lower=BASELINE_STD_FLOOR)

    normalized = journal.copy()
    for variable, z_column in Z_COLUMNS.items():
        person_mean = journal["person_id"].map(mean[variable])
        person_std = journal["person_id"].map(std[variable])
        normalized[z_column] = (journal[variable] - person_mean) / person_std
    return normalized


def build_dataset(seed=SEED):
    """Pipeline complet : retourne (journal, vérité terrain, profils)."""
    rng = make_rng(seed)
    journal, profiles = simulate_normal(rng)
    truth = assign_drifts(rng, N_PERSONS)
    drifted = inject_drifts(rng, journal, profiles, truth)
    truth = split_persons(rng, truth)
    dataset = normalize_by_baseline(add_labels(drifted, truth))
    return dataset, truth, profiles


def select_split(dataset, truth, split):
    """Les lignes du journal appartenant aux personnes d'un groupe (train, val ou test)."""
    ids = truth.loc[truth["split"] == split, "person_id"]
    return dataset[dataset["person_id"].isin(ids)]


def save_dataset(dataset, truth, profiles, directory=DATA_DIR):
    """Écrit les trois tables en CSV."""
    directory.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(directory / "journal.csv", index=False)
    truth.to_csv(directory / "ground_truth.csv", index=False)
    profiles.to_csv(directory / "profiles.csv", index=False)
