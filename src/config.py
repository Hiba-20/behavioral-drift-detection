"""Paramètres centraux du projet : un seul endroit à modifier."""

import random

import numpy as np

# Graine aléatoire fixe : mêmes données et mêmes résultats à chaque exécution.
SEED = 42

# Graines supplémentaires pour mesurer la robustesse (étape 8).
EXTRA_SEEDS = [1, 2, 3, 4]

# Taille du jeu de données (valeurs par défaut du plan).
N_PERSONS = 600
N_DAYS = 150
BASELINE_DAYS = 45  # jours 0 à 44 : jamais de dérive, référence personnelle

# Paramètres du simulateur.
AR_RHO = 0.3  # corrélation entre le bruit d'un jour et celui de la veille
WEEKEND_BED_SHIFT_MIN = 30  # coucher plus tard le week-end
WEEKEND_SLEEP_EXTRA_MIN = 30  # sommeil plus long le week-end
WEEKEND_SOCIAL_FACTOR = 1.3  # 30 % d'interactions en plus le week-end


def make_rng(seed=SEED):
    """Retourne un générateur NumPy reproductible."""
    return np.random.default_rng(seed)


def set_global_seed(seed=SEED):
    """Fixe la graine des modules qui ont un état aléatoire global."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
