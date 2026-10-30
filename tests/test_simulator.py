import numpy as np
import pandas as pd

from src.config import AR_RHO, N_DAYS, N_PERSONS, SEED, make_rng
from src.simulator import is_weekend, simulate_normal


def test_same_seed_gives_identical_data():
    a, _ = simulate_normal(make_rng(SEED))
    b, _ = simulate_normal(make_rng(SEED))
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_gives_different_data():
    a, _ = simulate_normal(make_rng(SEED))
    b, _ = simulate_normal(make_rng(SEED + 1))
    assert not a.equals(b)


def test_shape_columns_and_no_missing_values():
    journal, profiles = simulate_normal(make_rng(SEED))
    assert len(journal) == N_PERSONS * N_DAYS
    assert len(profiles) == N_PERSONS
    assert list(journal.columns) == [
        "person_id",
        "day",
        "bedtime_min",
        "waketime_min",
        "meals",
        "social",
    ]
    assert not journal.isna().any().any()


def test_values_are_plausible():
    journal, _ = simulate_normal(make_rng(SEED))
    assert journal["meals"].between(0, 3).all()
    assert (journal["social"] >= 0).all()
    assert (journal["waketime_min"] > journal["bedtime_min"]).all()
    assert 270 < journal["bedtime_min"].mean() < 330  # autour de 23h00
    assert 720 < journal["waketime_min"].mean() < 800  # autour de 06h30


def test_weekend_has_more_social_interactions():
    journal, _ = simulate_normal(make_rng(SEED))
    weekend = is_weekend(N_DAYS)[journal["day"].to_numpy()]
    ratio = journal.loc[weekend, "social"].mean() / journal.loc[~weekend, "social"].mean()
    assert 1.25 < ratio < 1.35  # facteur théorique : 1,3


def test_bedtime_noise_is_correlated_from_one_day_to_the_next():
    journal, profiles = simulate_normal(make_rng(SEED))
    bed = journal["bedtime_min"].to_numpy().reshape(N_PERSONS, N_DAYS)
    weekend = is_weekend(N_DAYS)
    residual = (
        bed - profiles["bed_mean"].to_numpy()[:, None] - 30 * weekend
    ) / profiles["bed_sd"].to_numpy()[:, None]
    lag1 = np.mean(residual[:, 1:] * residual[:, :-1])
    assert abs(lag1 - AR_RHO) < 0.03
