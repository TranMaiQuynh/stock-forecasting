"""Explicit online acquisition, separate from the offline experiment runner."""

from __future__ import annotations

from datetime import date
from importlib.metadata import version

import pandas as pd

from .experiment import sha256_file, tracked_run, write_json
from .research import validate_close_frame


def acquire_market_snapshot(asset, start, end, *, output_root="log", downloader=None):
    """Save a daily adjusted-close snapshot and vendor provenance without repair.

    ``end`` is exclusive. Downloaded historical adjustments may be revised by
    the vendor; this is not a point-in-time dataset or executable trade price.
    """
    if not isinstance(asset, str) or not asset.strip() or any(c.isspace() for c in asset) or "," in asset:
        raise ValueError("asset must identify exactly one ticker.")
    if date.fromisoformat(start) >= date.fromisoformat(end):
        raise ValueError("start must precede the exclusive end date.")
    options = {
        "tickers": asset,
        "start": start,
        "end": end,
        "interval": "1d",
        "auto_adjust": True,
        "back_adjust": False,
        "repair": False,
        "keepna": True,
        "actions": True,
        "threads": False,
        "progress": False,
        "ignore_tz": True,
        "multi_level_index": False,
        "timeout": 30,
    }
    settings = {"data_kind": "market", "provider": "Yahoo Finance", "download_options": options}
    with tracked_run(output_root, "market_snapshot", settings) as (run, logger):
        client_version = None
        if downloader is None:
            import yfinance as yf

            downloader = yf.download
            client_version = version("yfinance")
        raw = downloader(**options)
        if not isinstance(raw, pd.DataFrame) or raw.empty or isinstance(raw.columns, pd.MultiIndex):
            raise ValueError("Provider must return a nonempty single-asset frame.")
        if not isinstance(raw.index, pd.DatetimeIndex):
            raise ValueError("Provider must return a DatetimeIndex of session dates.")
        raw = raw.copy()
        raw.index.name = "Date"
        raw.to_csv(run / "vendor_data.csv")
        close = validate_close_frame(raw.reset_index())
        # Reject an unexpected provider range instead of silently trimming it.
        first, last = close.index[0].date(), close.index[-1].date()
        if first < date.fromisoformat(start) or last >= date.fromisoformat(end):
            raise ValueError("Provider returned observations outside the requested interval.")
        close.to_csv(run / "close.csv")
        write_json(
            run / "data_manifest.json",
            {
                "provider": "Yahoo Finance via yfinance",
                "provider_url": f"https://finance.yahoo.com/quote/{asset}/history/",
                "yfinance_version": client_version,
                "asset": asset,
                "download_options": options,
                "observations": len(close),
                "first_session": str(close.index[0]),
                "last_session": str(close.index[-1]),
                "snapshot_sha256": sha256_file(run / "close.csv"),
                "vendor_sha256": sha256_file(run / "vendor_data.csv"),
                "retrieved_at": "See enclosing run manifest started_at/ended_at (UTC)",
                "adjustment_policy": "yfinance auto_adjust=True; historical adjusted close; vendor may revise values",
                "timestamp_semantics": "Exchange session labels normalized to UTC, not actual publication timestamps",
                "missing_policy": "Reject invalid closes or duplicate/unsorted dates; no fill, drop or repair",
                "calendar_verified": False,
                "point_in_time_verified": False,
                "usage": "Personal research; not redistribution or a trade-execution dataset",
            },
        )
        logger.info("Snapshot validated: asset=%s observations=%d", asset, len(close))
    return run
