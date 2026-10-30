"""Paramètres centraux du projet : un seul endroit à modifier."""

import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

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

# Paramètres des dérives (étape 2).
DRIFT_TYPES = ["sleep", "meals", "social"]
AMPLITUDES_SIGMA = {"weak": 0.5, "medium": 1.0, "strong": 1.5}  # en écarts-types de la personne
CONTROL_FRACTION = 0.5  # part de personnes sans dérive
D0_RANGE = (60, 100)  # jour de départ de la dérive, tiré dans cet intervalle (bornes incluses)
RAMP_DAYS = 30  # durée de la montée progressive avant le plateau

# Jeu de données (étape 3).
SPLIT_FRACTIONS = {"train": 0.6, "val": 0.2}  # le test reçoit le reste (20 %)
LABEL_MIN_RAMP = 0.0  # 0 : étiquette « dérive » dès d0 ; 0.5 : seulement quand la dérive a atteint 50 %
BASELINE_STD_FLOOR = 0.1  # évite de diviser par ~0 si une personne est presque constante


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
