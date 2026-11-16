import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from quant_forecast_lab.session_contract import forecast_contract, label_contract, utc_timestamp


def history(last):
    calendar = xcals.get_calendar("XNYS", start="2026-01-01", end="2027-01-10")
    dates = calendar.sessions_in_range("2026-01-02", last)[-80:].tz_localize("UTC")
    return pd.DataFrame({"Close": 100 + np.arange(len(dates)) / 10}, index=dates)


def contract(last, as_of, received_at=None, **kwargs):
    return forecast_contract(
        history(last), calendar_name="XNYS", as_of=as_of, received_at=received_at or as_of,
        mode="live", **kwargs,
    )


@pytest.mark.parametrize("origin,now,target,opened,closed", [
    ("2026-11-25", "2026-11-25T21:30:00Z", "2026-11-27", "2026-11-27T14:30:00+00:00", "2026-11-27T18:00:00+00:00"),
    ("2026-11-27", "2026-11-27T18:30:00Z", "2026-11-30", "2026-11-30T14:30:00+00:00", "2026-11-30T21:00:00+00:00"),
    ("2026-03-06", "2026-03-06T21:30:00Z", "2026-03-09", "2026-03-09T13:30:00+00:00", "2026-03-09T20:00:00+00:00"),
])
def test_next_session_observes_holiday_early_close_and_dst(origin, now, target, opened, closed):
    result = contract(origin, now)
    assert result["target_session"] == target
    assert result["target_open_at"] == opened and result["target_close_at"] == closed
    assert result["vendor_point_in_time_verified"] is False


@pytest.mark.parametrize("now,received,match", [
    ("2026-11-16T20:59:00Z", "2026-11-16T20:58:00Z", "future"),
    ("2026-11-16T21:19:00Z", "2026-11-16T21:18:00Z", "future"),
    ("2026-11-17T14:30:00Z", "2026-11-16T21:30:00Z", "stale or late"),
    ("2026-11-18T21:30:00Z", "2026-11-16T21:30:00Z", "stale or late"),
    ("2026-11-16T21:30:00Z", "2026-11-16T21:10:00Z", "receipt"),
    ("2026-11-16T21:30:00Z", "2026-11-16T21:31:00Z", "receipt"),
])
def test_temporal_failures_are_rejected(now, received, match):
    with pytest.raises(ValueError, match=match):
        contract("2026-11-16", now, received)


@pytest.mark.parametrize("defect", ["gap", "holiday", "intraday"])
def test_calendar_does_not_silently_fill_or_drop(defect):
    close = history("2026-11-27")
    if defect == "gap":
        close = close.drop(close.index[5])
    elif defect == "holiday":
        close.loc[pd.Timestamp("2026-11-26", tz="UTC")] = 100.0
        close = close.sort_index()
    else:
        close.index += pd.Timedelta(hours=12)
    with pytest.raises(ValueError, match="session"):
        forecast_contract(close, calendar_name="XNYS", as_of="2026-11-27T18:30:00Z",
                          received_at="2026-11-27T18:25:00Z", mode="live")


def test_replay_cannot_be_mislabelled_as_live_receipt_evidence():
    result = forecast_contract(history("2026-11-16"), calendar_name="XNYS", as_of="2026-11-16T21:30:00Z",
                               received_at="2027-01-01T00:00:00Z", mode="replay")
    assert result["availability_evidence"] == "replay_assumption"
    with pytest.raises(ValueError, match="timezone-aware"):
        utc_timestamp("2026-11-16 21:30")


def test_labels_wait_for_target_close_and_reject_revised_origin():
    forecast = {**contract("2026-11-16", "2026-11-16T21:30:00Z"),
                "origin_close": float(history("2026-11-16").Close.iloc[-1])}
    labels = history("2026-11-17")
    labels.loc[pd.Timestamp("2026-11-16", tz="UTC"), "Close"] = forecast["origin_close"]
    with pytest.raises(ValueError, match="not yet"):
        label_contract(labels, forecast, received_at="2026-11-17T21:10:00Z", as_of="2026-11-17T21:15:00Z")
    assert label_contract(labels, forecast, received_at="2026-11-17T21:25:00Z",
                          as_of="2026-11-17T21:30:00Z") == labels.Close.iloc[-1]
    labels.loc[pd.Timestamp("2026-11-16", tz="UTC"), "Close"] += 1.0
    with pytest.raises(ValueError, match="revised"):
        label_contract(labels, forecast, received_at="2026-11-17T21:25:00Z", as_of="2026-11-17T21:30:00Z")


def test_labels_require_exact_target_and_valid_receipt():
    forecast = {**contract("2026-11-16", "2026-11-16T21:30:00Z"),
                "origin_close": float(history("2026-11-16").Close.iloc[-1])}
    with pytest.raises(ValueError, match="exact origin"):
        label_contract(history("2026-11-16"), forecast,
                       received_at="2026-11-17T21:25:00Z", as_of="2026-11-17T21:30:00Z")
    with pytest.raises(ValueError, match="receipt"):
        label_contract(history("2026-11-17"), forecast,
                       received_at="2026-11-17T21:35:00Z", as_of="2026-11-17T21:30:00Z")
