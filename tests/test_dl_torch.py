import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.baseline_stats import Z_NAMES
from src.config import CNN_WINDOW, MONITOR_START, N_DAYS
from src.dataset import build_dataset, select_split
from src.dl_torch import DriftCNN, cnn_scores, make_windows, train_cnn, training_windows
from src.metrics import calibrate_and_evaluate


def ramp_dataset(n_days=20):
    """Une personne : la première variable monte de 1 par jour, les autres valent 0."""
    frame = pd.DataFrame({"person_id": 0, "day": np.arange(n_days), "label": 0})
    for name in Z_NAMES:
        frame[name] = 0.0
    frame[Z_NAMES[0]] = np.arange(n_days, dtype=float)
    return frame


def test_windows_have_channels_first_and_the_right_content():
    ids, windows = make_windows(ramp_dataset())
    assert windows.shape == (1, 20 - CNN_WINDOW + 1, 4, CNN_WINDOW)
    assert windows.dtype == np.float32
    assert np.array_equal(windows[0, 0, 0], np.arange(CNN_WINDOW))  # premiers 14 jours
    assert np.array_equal(windows[0, 1, 0], np.arange(1, CNN_WINDOW + 1))  # décalée d'un jour
    assert (windows[0, :, 1:] == 0).all()  # les autres variables


@pytest.fixture(scope="module")
def splits():
    dataset, truth, _ = build_dataset()
    train, val, test = (select_split(dataset, truth, s) for s in ["train", "val", "test"])
    return dataset, train, val, test, truth


def test_training_windows_cover_monitoring_days_and_labels_are_aligned(splits):
    _, train, _, _, _ = splits
    X, y = training_windows(train)
    n_persons = train["person_id"].nunique()
    assert X.shape == (n_persons * (N_DAYS - MONITOR_START), 4, CNN_WINDOW)
    assert y.sum() == train[train["day"] >= MONITOR_START]["label"].sum()


def test_model_outputs_one_score_per_window():
    assert DriftCNN()(torch.randn(10, 4, CNN_WINDOW)).shape == (10,)


def test_network_is_small():
    assert sum(p.numel() for p in DriftCNN().parameters()) < 5000


def test_network_can_memorize_a_tiny_batch():
    # vérifie que le code d'entraînement est correct : 32 exemples doivent être appris par cœur
    torch.manual_seed(0)
    X, y = torch.randn(32, 4, CNN_WINDOW), (torch.rand(32) > 0.5).float()
    net = DriftCNN(dropout=0.0)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(400):
        optimizer.zero_grad()
        loss = loss_fn(net(X), y)
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.01


def test_training_is_reproducible_with_the_same_seed(splits):
    _, train, val, _, _ = splits
    _, first = train_cnn(train, val, epochs=2)
    _, second = train_cnn(train, val, epochs=2)
    assert [h["val_loss"] for h in first] == pytest.approx([h["val_loss"] for h in second], abs=1e-6)


def test_scores_have_the_format_expected_by_metrics(splits):
    _, train, val, _, _ = splits
    model, history = train_cnn(train, val, epochs=1)
    scores = cnn_scores(model, val)
    assert list(scores.columns) == ["person_id", "day", "score"]
    assert scores["score"].isna().sum() == val["person_id"].nunique() * (CNN_WINDOW - 1)
    assert len(history) == 1


@pytest.fixture(scope="module")
def trained_result(splits):
    _, train, val, test, truth = splits
    model, history = train_cnn(train, val)
    return calibrate_and_evaluate(cnn_scores(model, val), cnn_scores(model, test), truth), history


def test_trained_cnn_respects_calibration_and_finds_many_drifts(trained_result):
    result, _ = trained_result
    assert result["val"]["false_alarm_rate"] <= 0.05
    # un réseau jamais entraîné trouve en moyenne ~20 % des dérives (max observé 39 %)
    assert result["test"]["recall"] > 0.5


def test_validation_loss_improved_during_training(trained_result):
    _, history = trained_result
    assert min(h["val_loss"] for h in history) < history[0]["val_loss"]
    assert len(history) <= 30
