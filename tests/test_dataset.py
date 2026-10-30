import numpy as np
import pandas as pd
import pytest

from src.config import (
    BASELINE_DAYS,
    D0_RANGE,
    N_DAYS,
    N_PERSONS,
    RAMP_DAYS,
    SPLIT_FRACTIONS,
)
from src.dataset import (
    Z_COLUMNS,
    add_labels,
    build_dataset,
    save_dataset,
    select_split,
)


@pytest.fixture(scope="module")
def built():
    return build_dataset()


def test_baseline_is_always_free_of_drift():
    assert BASELINE_DAYS <= D0_RANGE[0]


def test_build_is_reproducible(built):
    again = build_dataset()
    assert built[0].equals(again[0])
    assert built[1].equals(again[1])


def test_every_person_is_in_exactly_one_split(built):
    _, truth, _ = built
    assert truth["split"].isin(["train", "val", "test"]).all()
    ids = {name: set(truth.loc[truth["split"] == name, "person_id"]) for name in ["train", "val", "test"]}
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])
    assert sum(len(s) for s in ids.values()) == N_PERSONS


def test_rows_of_one_person_never_span_two_splits(built):
    dataset, truth, _ = built
    train = select_split(dataset, truth, "train")
    test = select_split(dataset, truth, "test")
    assert set(train["person_id"]).isdisjoint(set(test["person_id"]))
    assert len(train) + len(test) + len(select_split(dataset, truth, "val")) == len(dataset)


def test_split_is_stratified(built):
    _, truth, _ = built
    strata = truth["drift_type"] + "/" + truth["level"]
    for _, members in truth.groupby(strata):
        n = len(members)
        counts = members["split"].value_counts()
        assert abs(counts.get("train", 0) - SPLIT_FRACTIONS["train"] * n) <= 1
        assert abs(counts.get("val", 0) - SPLIT_FRACTIONS["val"] * n) <= 1
        assert counts.get("test", 0) >= 1  # chaque case est présente dans le test


def test_controls_are_never_labeled(built):
    dataset, truth, _ = built
    controls = truth.loc[truth["drift_type"] == "none", "person_id"]
    assert (dataset.loc[dataset["person_id"].isin(controls), "label"] == 0).all()


@pytest.mark.parametrize("min_ramp, offset", [(0.0, 0), (0.5, RAMP_DAYS // 2)])
def test_label_starts_at_d0_plus_offset(built, min_ramp, offset):
    dataset, truth, _ = built
    labeled = add_labels(dataset.drop(columns="label"), truth, min_ramp)
    drifted = truth[truth["drift_type"] != "none"]
    for person_id, d0 in zip(drifted["person_id"], drifted["d0"]):
        rows = labeled[labeled["person_id"] == person_id].sort_values("day")
        assert (rows["label"].to_numpy() == (rows["day"].to_numpy() >= d0 + offset)).all()


def test_baseline_days_are_centered_and_scaled(built):
    dataset, _, _ = built
    baseline = dataset[dataset["day"] < BASELINE_DAYS].groupby("person_id")
    for z_column in Z_COLUMNS.values():
        assert baseline[z_column].mean().abs().max() < 1e-9
        assert baseline[z_column].std().between(0.99, 1.01).mean() > 0.99  # sauf plancher d'écart-type


def test_controls_stay_centered_after_baseline(built):
    dataset, truth, _ = built
    controls = truth.loc[truth["drift_type"] == "none", "person_id"]
    after = dataset[dataset["person_id"].isin(controls) & (dataset["day"] >= BASELINE_DAYS)]
    for z_column in Z_COLUMNS.values():
        assert abs(after[z_column].mean()) < 0.1


def test_sleep_drift_shows_in_bedtime_z_but_not_in_wake_z(built):
    dataset, truth, _ = built
    strong = truth[(truth["drift_type"] == "sleep") & (truth["level"] == "strong")]
    merged = dataset.merge(strong[["person_id", "d0"]], on="person_id")
    plateau = merged[merged["day"] >= merged["d0"].astype(int) + RAMP_DAYS]
    assert 1.0 < plateau["bedtime_z"].mean() < 1.8  # amplitude 1,5σ, un peu réduite par le week-end
    assert abs(plateau["waketime_z"].mean()) < 0.15


def test_save_dataset_writes_three_csv_files(built, tmp_path):
    dataset, truth, profiles = built
    save_dataset(dataset, truth, profiles, tmp_path)
    assert len(pd.read_csv(tmp_path / "journal.csv")) == N_PERSONS * N_DAYS
    assert len(pd.read_csv(tmp_path / "ground_truth.csv")) == N_PERSONS
    assert len(pd.read_csv(tmp_path / "profiles.csv")) == N_PERSONS
