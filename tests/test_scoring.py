"""The scoring contract, with the rejection paths exercised one by one."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import needs_data
from uev.features import WARMUP
from uev.scoring import ContractError, InputContract, Scorer


@pytest.fixture
def contract():
    return InputContract(known_zones=[1, 2], capacity={1: 10, 2: 20}, min_history_hours=8)


@pytest.fixture
def good_history():
    index = pd.date_range("2022-09-01", periods=8, freq="h")
    rows = []
    for zone in (1, 2):
        rows.append(pd.DataFrame({
            "zone_id": zone, "timestamp": index,
            "occupancy": np.full(8, 3.0), "volume": np.full(8, 21.0),
            "volume_11kw": np.full(8, 21.0), "duration": np.full(8, 2.4),
            "e_price": np.full(8, 1.0), "s_price": np.full(8, 0.5),
        }))
    return pd.concat(rows, ignore_index=True)


def test_a_valid_history_is_accepted(contract, good_history):
    contract.validate(good_history)


def test_missing_columns_are_reported(contract, good_history):
    with pytest.raises(ContractError, match="missing required columns"):
        contract.validate(good_history.drop(columns=["volume"]))


def test_an_empty_frame_is_rejected(contract, good_history):
    with pytest.raises(ContractError, match="empty"):
        contract.validate(good_history.iloc[0:0])


def test_a_non_datetime_timestamp_is_rejected(contract, good_history):
    bad = good_history.assign(timestamp=good_history["timestamp"].astype(str))
    with pytest.raises(ContractError, match="not a datetime"):
        contract.validate(bad)


def test_an_unknown_zone_is_rejected(contract, good_history):
    bad = good_history.copy()
    bad.loc[0, "zone_id"] = 99
    with pytest.raises(ContractError, match="unknown zone ids"):
        contract.validate(bad)


def test_duplicate_keys_are_rejected(contract, good_history):
    bad = pd.concat([good_history, good_history.iloc[[0]]], ignore_index=True)
    with pytest.raises(ContractError, match="duplicate"):
        contract.validate(bad)


def test_negative_values_are_rejected(contract, good_history):
    bad = good_history.copy()
    bad.loc[0, "volume"] = -1.0
    with pytest.raises(ContractError, match="negative"):
        contract.validate(bad)


def test_missing_values_are_rejected(contract, good_history):
    bad = good_history.copy()
    bad.loc[0, "duration"] = np.nan
    with pytest.raises(ContractError, match="missing values"):
        contract.validate(bad)


def test_occupancy_above_capacity_is_rejected(contract, good_history):
    bad = good_history.copy()
    bad.loc[0, "occupancy"] = 999.0
    with pytest.raises(ContractError, match="exceeds installed"):
        contract.validate(bad)


def test_too_little_history_is_rejected(contract, good_history):
    with pytest.raises(ContractError, match="hours of history"):
        contract.validate(good_history[good_history["timestamp"] < "2022-09-01 04:00"])


def test_a_gap_in_the_history_is_rejected(contract):
    """The history is long enough that removing one hour still clears the minimum,
    so the failure reported is the gap and not the length."""
    index = pd.date_range("2022-09-01", periods=12, freq="h")
    rows = []
    for zone in (1, 2):
        rows.append(pd.DataFrame({
            "zone_id": zone, "timestamp": index,
            "occupancy": np.full(12, 3.0), "volume": np.full(12, 21.0),
            "volume_11kw": np.full(12, 21.0), "duration": np.full(12, 2.4),
            "e_price": np.full(12, 1.0), "s_price": np.full(12, 0.5),
        }))
    history = pd.concat(rows, ignore_index=True)
    bad = history[history["timestamp"] != pd.Timestamp("2022-09-01 05:00")]
    with pytest.raises(ContractError, match="gap or a non hourly"):
        contract.validate(bad)


def test_ragged_zones_are_rejected(contract, good_history):
    bad = good_history.drop(good_history.index[-1])
    with pytest.raises(ContractError):
        contract.validate(bad)


def test_every_violation_is_collected_not_only_the_first(contract, good_history):
    bad = good_history.copy()
    bad.loc[0, "volume"] = -1.0
    bad.loc[1, "duration"] = -2.0
    with pytest.raises(ContractError) as raised:
        contract.validate(bad)
    assert len(raised.value.violations) >= 2


def test_violations_are_available_as_a_list(contract, good_history):
    bad = good_history.copy()
    bad.loc[0, "occupancy"] = 999.0
    try:
        contract.validate(bad)
    except ContractError as error:
        assert isinstance(error.violations, list) and error.violations


@needs_data
def test_the_scorer_returns_one_row_per_zone(bundle):
    capacity = dict(zip(bundle.zones["zone_id"], bundle.zones["capacity_points"]))
    contract = InputContract(known_zones=list(bundle.zones["zone_id"]), capacity=capacity)
    history = bundle.long[bundle.long["timestamp"].isin(bundle.index[:WARMUP + 2])].copy()
    scorer = Scorer(predict=lambda x: np.full(x.shape[0], 4.0), contract=contract,
                    reference=bundle, target="occupancy")
    out = scorer.score(history, horizon=1)
    assert len(out) == len(bundle.zones)
    assert set(out.columns) == {"zone_id", "timestamp", "horizon", "target", "prediction"}


@needs_data
def test_the_scorer_forecasts_the_hour_after_the_history(bundle):
    capacity = dict(zip(bundle.zones["zone_id"], bundle.zones["capacity_points"]))
    contract = InputContract(known_zones=list(bundle.zones["zone_id"]), capacity=capacity)
    history = bundle.long[bundle.long["timestamp"].isin(bundle.index[:WARMUP + 2])].copy()
    scorer = Scorer(predict=lambda x: np.full(x.shape[0], 4.0), contract=contract,
                    reference=bundle, target="occupancy")
    for horizon in (1, 6, 24):
        out = scorer.score(history, horizon=horizon)
        assert out["timestamp"].iloc[0] == bundle.index[WARMUP + 1] + pd.Timedelta(hours=horizon)


@needs_data
def test_the_scorer_clips_to_installed_capacity(bundle):
    capacity = dict(zip(bundle.zones["zone_id"], bundle.zones["capacity_points"]))
    contract = InputContract(known_zones=list(bundle.zones["zone_id"]), capacity=capacity)
    history = bundle.long[bundle.long["timestamp"].isin(bundle.index[:WARMUP + 2])].copy()
    scorer = Scorer(predict=lambda x: np.full(x.shape[0], 1e6), contract=contract,
                    reference=bundle, target="occupancy")
    out = scorer.score(history, horizon=1)
    ceilings = out["zone_id"].map(capacity)
    assert (out["prediction"] <= ceilings).all()


@needs_data
def test_the_scorer_never_returns_a_negative_forecast(bundle):
    capacity = dict(zip(bundle.zones["zone_id"], bundle.zones["capacity_points"]))
    contract = InputContract(known_zones=list(bundle.zones["zone_id"]), capacity=capacity)
    history = bundle.long[bundle.long["timestamp"].isin(bundle.index[:WARMUP + 2])].copy()
    scorer = Scorer(predict=lambda x: np.full(x.shape[0], -50.0), contract=contract,
                    reference=bundle, target="occupancy")
    assert (scorer.score(history, horizon=1)["prediction"] >= 0).all()


@needs_data
def test_a_non_positive_horizon_is_rejected(bundle):
    capacity = dict(zip(bundle.zones["zone_id"], bundle.zones["capacity_points"]))
    contract = InputContract(known_zones=list(bundle.zones["zone_id"]), capacity=capacity)
    history = bundle.long[bundle.long["timestamp"].isin(bundle.index[:WARMUP + 2])].copy()
    scorer = Scorer(predict=lambda x: np.zeros(x.shape[0]), contract=contract,
                    reference=bundle, target="occupancy")
    with pytest.raises(ContractError, match="horizon must be at least 1"):
        scorer.score(history, horizon=0)


@needs_data
def test_the_scorer_refuses_history_that_is_too_short_to_build_features(bundle):
    capacity = dict(zip(bundle.zones["zone_id"], bundle.zones["capacity_points"]))
    contract = InputContract(known_zones=list(bundle.zones["zone_id"]), capacity=capacity,
                             min_history_hours=24)
    history = bundle.long[bundle.long["timestamp"].isin(bundle.index[:30])].copy()
    scorer = Scorer(predict=lambda x: np.zeros(x.shape[0]), contract=contract,
                    reference=bundle, target="occupancy")
    with pytest.raises(ContractError, match="more history is required"):
        scorer.score(history, horizon=1)
