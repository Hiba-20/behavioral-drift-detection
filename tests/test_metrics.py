import numpy as np
import pandas as pd
import pytest

from src.config import BASELINE_DAYS, N_DAYS, N_PERSONS, SEED, make_rng
from src.dataset import split_persons
from src.drift import assign_drifts
from src.metrics import (
    calibrate_threshold,
    evaluate,
    person_outcomes,
    summarize_by_case,
)

THRESHOLD = 0.5  # pour les scores 0 / 1 des détecteurs factices


@pytest.fixture(scope="module")
def truth():
    rng = make_rng(SEED)
    return split_persons(rng, assign_drifts(rng, N_PERSONS))


def group_truth(truth, split):
    return truth[truth["split"] == split]


def fake_scores(group, rule):
    """Scores d'un détecteur factice : rule(jours, d0) -> scores (d0 = infini pour un témoin)."""
    ids = group["person_id"].to_numpy()
    day = np.tile(np.arange(N_DAYS), len(ids))
    d0 = group["d0"].to_numpy(dtype=float, na_value=np.nan)
    d0 = np.repeat(np.where(np.isnan(d0), np.inf, d0), N_DAYS)
    return pd.DataFrame(
        {"person_id": np.repeat(ids, N_DAYS), "day": day, "score": rule(day, d0).astype(float)}
    )


def test_oracle_is_perfect(truth):
    group = group_truth(truth, "test")
    result = evaluate(fake_scores(group, lambda day, d0: day >= d0), group, THRESHOLD)
    assert result["recall"] == 1.0
    assert result["precision"] == 1.0
    assert result["f1"] == 1.0
    assert result["fp"] == 0
    assert result["false_alarm_rate"] == 0.0
    assert result["median_delay"] == 0


def test_always_alarm_finds_everything_but_alarms_everyone(truth):
    group = group_truth(truth, "test")
    result = evaluate(fake_scores(group, lambda day, d0: np.ones_like(day)), group, THRESHOLD)
    assert result["recall"] == 1.0
    assert result["false_alarm_rate"] == 1.0
    assert result["fp"] == len(group)  # témoins, et alarmes avant d0 des autres
    assert result["precision"] == pytest.approx(result["n_drifted"] / (result["n_drifted"] + len(group)))


def test_never_alarm_finds_nothing(truth):
    group = group_truth(truth, "test")
    result = evaluate(fake_scores(group, lambda day, d0: np.zeros_like(day)), group, THRESHOLD)
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0
    assert result["fp"] == 0
    assert np.isnan(result["precision"])  # aucune alarme : précision indéfinie
    assert np.isnan(result["median_delay"])


def test_delay_equals_lateness(truth):
    group = group_truth(truth, "test")
    result = evaluate(fake_scores(group, lambda day, d0: day >= d0 + 7), group, THRESHOLD)
    assert result["recall"] == 1.0
    assert result["median_delay"] == 7
    assert result["mean_delay"] == 7


def test_alarm_before_d0_is_a_false_alarm(truth):
    group = group_truth(truth, "test")
    result = evaluate(fake_scores(group, lambda day, d0: day >= d0 - 10), group, THRESHOLD)
    assert result["fp"] == result["n_drifted"]


def test_alarms_during_baseline_are_ignored(truth):
    group = group_truth(truth, "test")
    result = evaluate(
        fake_scores(group, lambda day, d0: day < BASELINE_DAYS), group, THRESHOLD
    )
    assert result["fp"] == 0
    assert result["tp"] == 0


def test_calibration_gives_exactly_the_target_on_known_scores(truth):
    group = group_truth(truth, "val")
    controls = group.loc[group["drift_type"] == "none", "person_id"].to_numpy()
    rank = {person: i + 1 for i, person in enumerate(controls)}  # score maximal 1, 2, ..., 100

    def rule(day, d0):
        return np.zeros_like(day, dtype=float)

    scores = fake_scores(group, rule)
    mask = (scores["day"] == 50) & scores["person_id"].isin(controls)
    scores.loc[mask, "score"] = scores.loc[mask, "person_id"].map(rank)

    threshold = calibrate_threshold(scores, group)
    assert threshold == len(controls) - 5  # le 6e plus grand score : 5 témoins le dépassent
    assert evaluate(scores, group, threshold)["false_alarm_rate"] == 0.05


def test_random_detector_respects_calibration_and_finds_little(truth):
    # Le seuil est estimé sur ~100 témoins : le taux de fausses alarmes sur le test varie
    # beaucoup d'une graine à l'autre (écart-type ~3 points), donc on teste la moyenne.
    val, test = group_truth(truth, "val"), group_truth(truth, "test")
    test_false_alarm_rates, test_recalls = [], []
    for seed in range(30):
        rng = make_rng(seed)
        val_scores = fake_scores(val, lambda day, d0: rng.uniform(size=len(day)))
        test_scores = fake_scores(test, lambda day, d0: rng.uniform(size=len(day)))
        threshold = calibrate_threshold(val_scores, val)
        assert evaluate(val_scores, val, threshold)["false_alarm_rate"] <= 0.05  # garanti
        on_test = evaluate(test_scores, test, threshold)
        test_false_alarm_rates.append(on_test["false_alarm_rate"])
        test_recalls.append(on_test["recall"])
    assert 0.03 < np.mean(test_false_alarm_rates) < 0.09  # ~5 % en moyenne
    assert np.mean(test_recalls) < 0.10  # un détecteur au hasard ne trouve presque rien


def test_summary_by_case_has_all_nine_cells_and_perfect_oracle(truth):
    group = group_truth(truth, "test")
    scores = fake_scores(group, lambda day, d0: day >= d0)
    table = summarize_by_case(person_outcomes(scores, group, THRESHOLD))
    assert len(table) == 9
    assert (table["detection_rate"] == 1.0).all()
    assert (table["median_delay"] == 0).all()
