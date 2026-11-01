"""1D-CNN PyTorch sur des fenêtres de 14 jours (étape 7).

Entrée : pour chaque personne et chaque jour, les CNN_WINDOW derniers jours de
chacune des 4 variables normalisées (tenseur 4 x 14). Sortie : un score (logit) ;
plus il est élevé, plus le réseau juge que la dérive a commencé.
"""

import copy

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.baseline_stats import to_array, to_scores_frame
from src.config import CNN_WINDOW, MONITOR_START, SEED, set_global_seed


def make_windows(dataset, window=CNN_WINDOW):
    """Retourne (person_id, fenêtres de forme personnes x fenêtres x 4 variables x jours)."""
    ids, z = to_array(dataset)
    return ids, sliding_window_view(z, window, axis=1).astype(np.float32)


def training_windows(dataset, window=CNN_WINDOW):
    """(X, y) : une fenêtre par personne et par jour de surveillance ; étiquette du dernier jour."""
    ids, windows = make_windows(dataset, window)
    labels = dataset.sort_values(["person_id", "day"])["label"].to_numpy()
    labels = labels.reshape(len(ids), -1)
    end_day = np.arange(windows.shape[1]) + window - 1  # jour où se termine chaque fenêtre
    kept = end_day >= MONITOR_START
    X = windows[:, kept].reshape(-1, *windows.shape[2:])
    y = labels[:, window - 1 :][:, kept].ravel().astype(np.float32)
    return X, y


class DriftCNN(nn.Module):
    """Deux convolutions 1D, une moyenne sur le temps, puis un score."""

    def __init__(self, n_channels=4, hidden=16, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, 2 * hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # moyenne de chaque motif sur les 14 jours
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(2 * hidden, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def _mean_loss(model, X, y, loss_fn, chunk=8192):
    """Perte moyenne sur un ensemble, sans entraîner (mode évaluation)."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(X), chunk):
            logits = model(X[start : start + chunk])
            total += loss_fn(logits, y[start : start + chunk]).item() * len(logits)
    return total / len(X)


def train_cnn(
    train,
    val,
    epochs=30,
    patience=5,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=256,
    seed=SEED,
    shuffle_labels=False,
):
    """Entraîne sur le train, s'arrête quand la perte de validation stagne.

    Retourne (modèle aux meilleurs poids, historique des pertes par époque).
    """
    set_global_seed(seed)
    X, y = training_windows(train)
    X_val, y_val = training_windows(val)
    if shuffle_labels:  # contrôle : des étiquettes sans lien avec les données
        rng = np.random.default_rng(seed)
        y, y_val = rng.permutation(y), rng.permutation(y_val)
    X, y, X_val, y_val = (torch.from_numpy(a) for a in (X, y, X_val, y_val))

    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    positive_weight = torch.tensor((1 - y.mean().item()) / y.mean().item())  # compense le déséquilibre
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)

    model = DriftCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history, best_loss, best_weights, waited = [], np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in loader:
            optimizer.zero_grad()  # remet les corrections à zéro
            loss = loss_fn(model(batch_X), batch_y)  # erreur sur ce mini-lot
            loss.backward()  # calcule comment chaque poids influence l'erreur
            optimizer.step()  # corrige les poids
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": _mean_loss(model, X, y, loss_fn),
                "val_loss": _mean_loss(model, X_val, y_val, loss_fn),
            }
        )
        if history[-1]["val_loss"] < best_loss:
            best_loss, best_weights, waited = history[-1]["val_loss"], copy.deepcopy(model.state_dict()), 0
        else:
            waited += 1
            if waited >= patience:
                break

    model.load_state_dict(best_weights)
    return model.eval(), history


def cnn_scores(model, dataset, chunk=8192):
    """Score (logit) de chaque personne et chaque jour ayant 14 jours d'historique."""
    ids, windows = make_windows(dataset)
    n_persons, n_windows = windows.shape[:2]
    flat = torch.from_numpy(windows.reshape(-1, *windows.shape[2:]))
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(flat[i : i + chunk]) for i in range(0, len(flat), chunk)])

    scores = np.full((n_persons, dataset["day"].nunique()), np.nan)
    scores[:, CNN_WINDOW - 1 :] = logits.numpy().reshape(n_persons, n_windows)
    return to_scores_frame(ids, scores)
