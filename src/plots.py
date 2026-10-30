"""Graphiques du projet."""

import matplotlib.pyplot as plt

from src.config import BASELINE_DAYS

VARIABLES = [
    ("bedtime_min", "Coucher\n(min après 18h)"),
    ("waketime_min", "Lever\n(min après 18h)"),
    ("meals", "Repas"),
    ("social", "Interactions"),
]

# Une variable par type de dérive : (colonne, libellé).
COLUMN_BY_TYPE = {
    "sleep": ("bedtime_min", "Coucher (min après 18h)"),
    "meals": ("meals", "Repas par jour"),
    "social": ("social", "Interactions par jour"),
}
LEVEL_ORDER = ["weak", "medium", "strong"]


def plot_person_series(journal, person_id, d0=None, path=None):
    """Une courbe par variable pour une personne ; trait vertical à d0 si donné."""
    one = journal[journal["person_id"] == person_id]
    fig, axes = plt.subplots(len(VARIABLES), 1, figsize=(10, 7), sharex=True)
    for ax, (column, label) in zip(axes, VARIABLES):
        ax.plot(one["day"], one[column], marker=".", linewidth=0.8)
        ax.set_ylabel(label)
        if d0 is not None:
            ax.axvline(d0, color="red", linestyle="--")
    axes[-1].set_xlabel("Jour")
    axes[0].set_title(f"Personne {person_id}")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120)
    return fig


def plot_drift_profiles(journal, truth, path=None):
    """Écart moyen à la baseline personnelle, en jours relatifs à d0, par type et niveau."""
    df = journal.merge(truth, on="person_id")
    df = df[df["drift_type"] != "none"].copy()
    df["rel_day"] = df["day"] - df["d0"].astype(int)
    df = df[df["rel_day"].between(-40, 49)]  # fenêtre disponible pour toutes les personnes

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    for ax, (drift_type, (column, label)) in zip(axes, COLUMN_BY_TYPE.items()):
        sub = df[df["drift_type"] == drift_type].copy()
        baseline = sub[sub["day"] < BASELINE_DAYS].groupby("person_id")[column].mean()
        sub["deviation"] = sub[column] - sub["person_id"].map(baseline)
        for level in LEVEL_ORDER:
            curve = sub[sub["level"] == level].groupby("rel_day")["deviation"].mean()
            ax.plot(curve.index, curve.rolling(7, center=True).mean(), label=level)
        ax.axvline(0, color="grey", linestyle="--")
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_title(f"Dérive : {drift_type}")
        ax.set_xlabel("Jours depuis d0")
        ax.set_ylabel(f"Écart à la baseline\n{label}")
    axes[0].legend(title="Ampleur")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120)
    return fig
