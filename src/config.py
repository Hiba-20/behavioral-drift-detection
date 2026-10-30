"""Paramètres centraux du projet : un seul endroit à modifier."""

import random

import numpy as np

# Graine aléatoire fixe : mêmes données et mêmes résultats à chaque exécution.
SEED = 42

# Graines supplémentaires pour mesurer la robustesse (étape 8).
EXTRA_SEEDS = [1, 2, 3, 4]


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
