import numpy as np
import pytest

from src.config import (
    AMPLITUDES_SIGMA,
    D0_RANGE,
    DRIFT_TYPES,
    N_DAYS,
    N_PERSONS,
    RAMP_DAYS,
    SEED,
    make_rng,
)
from src.drift import assign_drifts, inject_drifts, weekly_mean_social_rate
from src.simulator import simulate_normal

SHAPE = (N_PERSONS, N_DAYS)


def build(amplitudes=AMPLITUDES_SIGMA, seed=SEED):
    rng = make_rng(seed)
    journal, profiles = simulate_normal(rng)
    truth = assign_drifts(rng, N_PERSONS, amplitudes)
    drifted = inject_drifts(rng, journal, profiles, truth)
    return journal, profiles, truth, drifted


@pytest.fixture(scope="module")
def data():
    return build()


def column(df, name):
    return df[name].to_numpy().reshape(SHAPE)


def after_plateau_mask(truth):
    """(personnes x jours) : vrai pour les jours où la dérive a atteint son plateau."""
    d0 = truth["d0"].to_numpy(dtype=float, na_value=np.nan)[:, None]
    return np.arange(N_DAYS)[None, :] >= d0 + RAMP_DAYS  # faux si d0 est NaN


def test_half_controls_and_balanced_cells(data):
    _, _, truth, _ = data
    assert (truth["drift_type"] == "none").sum() == N_PERSONS // 2
    cells = truth[truth["drift_type"] != "none"].groupby(["drift_type", "level"]).size()
    assert len(cells) == len(DRIFT_TYPES) * len(AMPLITUDES_SIGMA)
    assert cells.between(33, 34).all()


def test_d0_in_range_for_drifted_and_missing_for_controls(data):
    _, _, truth, _ = data
    drifted = truth[truth["drift_type"] != "none"]
    assert drifted["d0"].between(*D0_RANGE).all()
    assert truth.loc[truth["drift_type"] == "none", "d0"].isna().all()


def test_controls_and_days_before_d0_are_unchanged(data):
    journal, _, truth, drifted = data
    d0 = truth["d0"].to_numpy(dtype=float, na_value=np.nan)[:, None]
    before = np.arange(N_DAYS)[None, :] <= d0  # faux pour les témoins : traités à part
    controls = (truth["drift_type"] == "none").to_numpy()[:, None]
    unchanged = before | controls
    for name in ["bedtime_min", "waketime_min", "meals", "social"]:
        assert np.array_equal(
            column(journal, name)[unchanged], column(drifted, name)[unchanged]
        )


def test_wake_time_never_changes(data):
    journal, _, _, drifted = data
    assert np.array_equal(journal["waketime_min"], drifted["waketime_min"])


def test_zero_amplitude_changes_nothing():
    journal, _, _, drifted = build({"weak": 0.0, "medium": 0.0, "strong": 0.0})
    assert journal.equals(drifted)


def test_same_seed_gives_identical_drifted_data():
    assert build()[3].equals(build()[3])


def test_each_drift_only_changes_its_own_variable(data):
    journal, _, truth, drifted = data
    columns = {"sleep": "bedtime_min", "meals": "meals", "social": "social"}
    for drift_type, own in columns.items():
        rows = (truth["drift_type"] == drift_type).to_numpy()
        for name in columns.values():
            changed = column(journal, name)[rows] != column(drifted, name)[rows]
            assert changed.any() == (name == own)


def test_sleep_shift_matches_amplitude_in_sigmas(data):
    journal, profiles, truth, drifted = data
    diff = column(drifted, "bedtime_min") - column(journal, "bedtime_min")
    in_sigma = diff / profiles["bed_sd"].to_numpy()[:, None]
    plateau = after_plateau_mask(truth)
    for level, amplitude in AMPLITUDES_SIGMA.items():
        rows = ((truth["drift_type"] == "sleep") & (truth["level"] == level)).to_numpy()
        mask = plateau[rows]
        assert abs(in_sigma[rows][mask].mean() - amplitude) < 0.05


def test_meal_and_social_drops_match_amplitude_in_sigmas(data):
    journal, profiles, truth, drifted = data
    p = profiles["meal_p"].to_numpy()[:, None]
    sigma = {
        "meals": np.sqrt(3 * p * (1 - p)),
        "social": np.sqrt(weekly_mean_social_rate(profiles)[:, None]),
    }
    plateau = after_plateau_mask(truth)
    for drift_type in ["meals", "social"]:
        drop = column(journal, drift_type) - column(drifted, drift_type)
        in_sigma = drop / sigma[drift_type]
        for level, amplitude in AMPLITUDES_SIGMA.items():
            rows = ((truth["drift_type"] == drift_type) & (truth["level"] == level)).to_numpy()
            mask = plateau[rows]
            assert abs(in_sigma[rows][mask].mean() - amplitude) < 0.1
