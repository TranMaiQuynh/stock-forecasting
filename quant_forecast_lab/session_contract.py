"""Daily close-to-next-close contracts; exchange labels are not publication times."""

from __future__ import annotations

from importlib.metadata import version

import pandas as pd


def utc_timestamp(value, name="timestamp") -> pd.Timestamp:
    """Reject naive/invalid instants rather than silently assuming a timezone."""
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"{name} must be a timezone-aware instant.")
    return stamp.tz_convert("UTC")


def _calendar(name, first, last):
    import exchange_calendars as xcals

    if name not in {"XNYS", "XNAS"}:
        raise ValueError("This daily US-equity lane supports explicit XNYS/XNAS calendars only.")
    return xcals.get_calendar(name, start=first, end=last)


def validate_sessions(close, calendar_name):
    """Require a continuous sequence of exchange-session dates, without filling."""
    dates = close.index
    if not isinstance(dates, pd.DatetimeIndex) or dates.tz is None or dates.empty:
        raise ValueError("Expected a validated close frame indexed by UTC session dates.")
    dates = dates.tz_convert("UTC")
    if not dates.equals(dates.normalize()) or not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("Daily labels must be unique, ordered midnight UTC session dates.")
    calendar = _calendar(calendar_name, dates[0].date(), (dates[-1] + pd.Timedelta(days=15)).date())
    expected = calendar.sessions_in_range(dates[0].date(), dates[-1].date())
    if not dates.tz_localize(None).equals(expected):
        raise ValueError("Input contains a non-session or missing exchange session; no repair is performed.")
    return calendar


def forecast_contract(close, *, calendar_name, as_of, received_at, mode, publication_delay_minutes=20):
    """Allow forecasts only after a completed close and before the next session opens.

    The buffer is a conservative configured policy, not a vendor latency claim.
    In replay mode data availability is assumed, never prospective evidence.
    """
    if mode not in {"live", "replay"}:
        raise ValueError("mode must be live or replay.")
    if (
        isinstance(publication_delay_minutes, bool)
        or not isinstance(publication_delay_minutes, int)
        or not 0 <= publication_delay_minutes <= 240
    ):
        raise ValueError("publication_delay_minutes must be an integer between 0 and 240.")
    now = utc_timestamp(as_of, "as_of")
    receipt = utc_timestamp(received_at, "received_at")
    calendar = validate_sessions(close, calendar_name)
    origin = close.index[-1].tz_convert("UTC").tz_localize(None)
    target = calendar.next_session(origin)
    origin_close = calendar.session_close(origin)
    target_open = calendar.session_open(target)
    target_close = calendar.session_close(target)
    ready = origin_close + pd.Timedelta(minutes=publication_delay_minutes)
    if now < ready:
        raise ValueError("Origin close/publication buffer is still in the future.")
    if now >= target_open:
        raise ValueError("Forecast is stale or late: the target session has already opened.")
    if mode == "live" and not ready <= receipt <= now:
        raise ValueError("Live snapshot receipt must follow the origin close/buffer and precede the forecast.")
    return {
        "schema_version": 1,
        "mode": mode,
        "calendar": calendar_name,
        "calendar_version": version("exchange-calendars"),
        "origin_session": origin.date().isoformat(),
        "target_session": target.date().isoformat(),
        "origin_close_at": origin_close.isoformat(),
        "target_open_at": target_open.isoformat(),
        "target_close_at": target_close.isoformat(),
        "as_of": now.isoformat(),
        "received_at": receipt.isoformat(),
        "publication_delay_minutes": publication_delay_minutes,
        "horizon": 1,
        "target": "next_session_log_return",
        "availability_evidence": "local_receipt" if mode == "live" else "replay_assumption",
        "vendor_point_in_time_verified": False,
    }


def label_contract(close, forecast, *, received_at, as_of):
    """Use the first compatible label vintage, never mix adjusted-price bases."""
    calendar = validate_sessions(close, forecast["calendar"])
    now = utc_timestamp(as_of, "as_of")
    receipt = utc_timestamp(received_at, "received_at")
    origin = pd.Timestamp(forecast["origin_session"], tz="UTC")
    target = pd.Timestamp(forecast["target_session"], tz="UTC")
    if origin not in close.index or target not in close.index:
        raise ValueError("Label snapshot must include the exact origin and target sessions.")
    if calendar.next_session(origin.tz_localize(None)) != target.tz_localize(None):
        raise ValueError("Stored target does not match the calendar's next session.")
    ready = calendar.session_close(target.tz_localize(None)) + pd.Timedelta(
        minutes=forecast["publication_delay_minutes"]
    )
    if now < ready:
        raise ValueError("Target label is not yet available after its close/publication buffer.")
    if forecast["mode"] == "live" and not ready <= receipt <= now:
        raise ValueError("Live label receipt must follow the target close/buffer and precede scoring.")
    if float(close.loc[origin, "Close"]) != forecast["origin_close"]:
        raise ValueError("Origin close was revised; adjusted-price bases differ. Do not silently score.")
    if close.index[-1] > now.normalize():
        raise ValueError("Label snapshot contains future sessions.")
    return float(close.loc[target, "Close"])
