import numpy as np
import pandas as pd
import pytest

from src.baseline_stats import Z_NAMES
from src.config import MONITOR_START, N_DAYS
from src.dataset import build_dataset, select_split
from src.metrics import calibrate_and_evaluate
from src.ml_sklearn import (
    FEATURE_NAMES,
    fit_forest,
    fit_isolation_forest,
    forest_scores,
    isolation_scores,
    training_table,
    window_features,
)

FAST_FOREST = {"n_estimators": 50, "max_depth": 8, "min_samples_leaf": 100}


def ramp_dataset(n_days=12):
    """Une personne : la première variable monte de 1 par jour, les autres valent 0."""
    frame = pd.DataFrame({"person_id": 0, "day": np.arange(n_days), "label": 0})
    for name in Z_NAMES:
        frame[name] = 0.0
    frame[Z_NAMES[0]] = np.arange(n_days, dtype=float)
    return frame


def test_features_have_the_expected_names_and_shape():
    ids, features = window_features(ramp_dataset())
    assert features.shape == (1, 12, len(FEATURE_NAMES))
    assert len(FEATURE_NAMES) == 12


def test_window_features_match_hand_computation():
    _, features = window_features(ramp_dataset())
    first = FEATURE_NAMES.index("bedtime_z_mean")
    assert np.isnan(features[0, :6]).all()  # pas encore 7 jours d'historique
    assert features[0, 6, first] == 3.0  # moyenne de 0..6
    assert features[0, 6, first + 1] == pytest.approx(np.std(np.arange(7)))  # écart-type
    assert features[0, 6, first + 2] == pytest.approx(1.0)  # pente d'une rampe de pente 1
    assert features[0, 11, first + 2] == pytest.approx(1.0)
    other = FEATURE_NAMES.index("social_z_mean")
    assert (features[0, 6:, other : other + 3] == 0).all()


@pytest.fixture(scope="module")
def splits():
    dataset, truth, _ = build_dataset()
    train, val, test = (select_split(dataset, truth, s) for s in ["train", "val", "test"])
    return train, val, test, truth


def test_training_table_uses_only_monitoring_days_of_the_given_persons(splits):
    train, _, _, _ = splits
    X, y = training_table(train)
    n_persons = train["person_id"].nunique()
    assert X.shape == (n_persons * (N_DAYS - MONITOR_START), 12)
    assert not np.isnan(X).any()
    assert y.sum() == train[train["day"] >= MONITOR_START]["label"].sum()  # étiquettes alignées


def test_random_forest_beats_random_and_label_shuffle_control_fails(splits):
    train, val, test, truth = splits
    real = fit_forest(train, **FAST_FOREST)
    shuffled = fit_forest(train, shuffle_labels=True, **FAST_FOREST)
    real_result = calibrate_and_evaluate(forest_scores(real, val), forest_scores(real, test), truth)
    shuffled_result = calibrate_and_evaluate(
        forest_scores(shuffled, val), forest_scores(shuffled, test), truth
    )
    assert real_result["val"]["false_alarm_rate"] <= 0.05
    assert real_result["test"]["recall"] > 0.2
    # étiquettes sans lien avec les données : le modèle ne doit rien trouver (hasard ~ 4 %)
    assert shuffled_result["test"]["recall"] < 0.15


def test_isolation_forest_scores_have_the_format_expected_by_metrics(splits):
    train, val, _, truth = splits
    scores = isolation_scores(fit_isolation_forest(train), val)
    assert list(scores.columns) == ["person_id", "day", "score"]
    assert scores["score"].isna().sum() == val["person_id"].nunique() * 6
