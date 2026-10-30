"""Graphiques du projet."""

import matplotlib.pyplot as plt

VARIABLES = [
    ("bedtime_min", "Coucher\n(min après 18h)"),
    ("waketime_min", "Lever\n(min après 18h)"),
    ("meals", "Repas"),
    ("social", "Interactions"),
]


def plot_person_series(journal, person_id, path=None):
    """Une courbe par variable pour une personne ; sauvegarde si `path` est donné."""
    one = journal[journal["person_id"] == person_id]
    fig, axes = plt.subplots(len(VARIABLES), 1, figsize=(10, 7), sharex=True)
    for ax, (column, label) in zip(axes, VARIABLES):
        ax.plot(one["day"], one[column], marker=".", linewidth=0.8)
        ax.set_ylabel(label)
    axes[-1].set_xlabel("Jour")
    axes[0].set_title(f"Personne {person_id} : journal simulé sans dérive")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120)
    return fig
