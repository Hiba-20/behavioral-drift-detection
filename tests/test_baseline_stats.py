import numpy as np
import pandas as pd
import pytest

from src.baseline_stats import (
    Z_NAMES,
    cusum_scores,
    fit_rolling_scale,
    rolling_mean,
    rolling_zscore_scores,
)
from src.dataset import build_dataset, select_split
from src.metrics import calibrate_and_evaluate, summarize_by_case


def tiny_dataset(first_variable):
    """Une personne, jours 0..n-1 ; `first_variable` est le z du coucher, les autres valent 0."""
    n = len(first_variable)
    frame = pd.DataFrame({"person_id": 0, "day": np.arange(n)})
    for name in Z_NAMES:
        frame[name] = 0.0
    frame[Z_NAMES[0]] = first_variable
    return frame


def test_rolling_mean_is_the_mean_of_the_last_days():
    z = np.arange(10, dtype=float).reshape(1, 10, 1)
    out = rolling_mean(z, 3)
    assert np.isnan(out[0, :2, 0]).all()
    assert out[0, 2, 0] == 1.0  # (0 + 1 + 2) / 3
    assert out[0, 9, 0] == 8.0  # (7 + 8 + 9) / 3


def test_cusum_matches_hand_computation():
    # k = 0,5 : la tirelire « hausse » vaut 0, .5, 1, 1.5, 1 ; puis la tirelire « baisse » prend le relais
    z = [0, 1, 1, 1, 0, -1, -1, -1, -1]
    scores = cusum_scores(tiny_dataset(z), k=0.5, start=0)["score"].to_numpy()
    assert scores == pytest.approx([0, 0.5, 1.0, 1.5, 1.0, 0.5, 1.0, 1.5, 2.0])


def test_cusum_never_goes_below_zero_on_normal_days():
    scores = cusum_scores(tiny_dataset([0.0] * 20), k=0.5, start=0)["score"]
    assert (scores == 0).all()


def test_cusum_reacts_to_a_lasting_shift_but_ignores_small_noise():
    z = [0.3] * 10 + [2.0] * 10  # 0,3 < k : rien ne s'accumule ; puis +2σ : +1,5 par jour
    scores = cusum_scores(tiny_dataset(z), k=0.5, start=0)["score"].to_numpy()
    assert scores[:10].max() == 0
    assert scores[13] == pytest.approx(6.0)  # 4 jours x 1,5


def test_rolling_zscore_of_a_lasting_shift():
    z = [0.0] * 10 + [2.0] * 10
    scale = np.ones(len(Z_NAMES))
    scores = rolling_zscore_scores(tiny_dataset(z), scale, window=7)["score"].to_numpy()
    assert np.isnan(scores[:6]).all()  # pas assez de jours pour la première fenêtre
    assert scores[16] == pytest.approx(2.0)  # la fenêtre 10..16 ne contient que des jours à +2
    assert scores[12] == pytest.approx(2.0 * 3 / 7)  # 3 jours à +2 sur 7


def test_scores_frame_has_the_columns_expected_by_metrics():
    frame = cusum_scores(tiny_dataset([0.0] * 5), start=0)
    assert list(frame.columns) == ["person_id", "day", "score"]


@pytest.fixture(scope="module")
def results():
    dataset, truth, _ = build_dataset()
    train, val, test = (select_split(dataset, truth, s) for s in ["train", "val", "test"])
    scale = fit_rolling_scale(train)
    methods = {
        "zscore": lambda d: rolling_zscore_scores(d, scale),
        "cusum": cusum_scores,
    }
    return {
        name: calibrate_and_evaluate(score(val), score(test), truth)
        for name, score in methods.items()
    }


@pytest.mark.parametrize("name", ["zscore", "cusum"])
def test_false_alarm_rate_on_validation_respects_the_target(results, name):
    assert results[name]["val"]["false_alarm_rate"] <= 0.05


@pytest.mark.parametrize("name", ["zscore", "cusum"])
def test_detector_beats_a_random_one(results, name):
    # un détecteur au hasard, calibré pareil, trouve ~4 % des dérives (étape 4)
    assert results[name]["test"]["recall"] > 0.25


@pytest.mark.parametrize("name", ["zscore", "cusum"])
def test_strong_drifts_are_found_more_often_than_weak_ones(results, name):
    by_case = summarize_by_case(results[name]["test_outcomes"])
    rate = by_case["detection_rate"].groupby("level").mean()
    assert rate["strong"] > rate["weak"] + 0.3
