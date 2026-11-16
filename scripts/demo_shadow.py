"""Offline synthetic replay: refit once, record twice, score delayed labels, stop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="log")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from quant_forecast_lab.artifacts import verify_run
    from quant_forecast_lab.candidate import prepare_candidate
    from quant_forecast_lab.experiment import tracked_run, write_json
    from quant_forecast_lab.research import ResearchConfig, run_offline_experiment
    from quant_forecast_lab.shadow import (
        capture_snapshot,
        create_protocol,
        record_forecast,
        score_snapshot,
        shadow_report,
    )

    settings = {"data_kind": "synthetic", "mode": "replay", "seed": 42, "market_evidence": False}
    with tracked_run(args.output_root, "shadow_demo", settings) as (run, logger):
        calendar = xcals.get_calendar("XNYS", start="2020-01-01", end="2022-01-01")
        dates = calendar.sessions_in_range("2020-01-02", "2021-06-01")[:321]
        returns = np.random.default_rng(42).normal(0, 0.01, len(dates))
        for i in range(1, len(returns)):
            returns[i] += 0.2 * returns[i - 1]
        frame = pd.DataFrame({"Date": dates, "Close": 100 * np.exp(np.cumsum(returns))})
        training = run / "development.csv"
        frame.iloc[:160].to_csv(training, index=False)
        source = run_offline_experiment(
            training, output_root=run, data_kind="synthetic",
            config=ResearchConfig(asset="SYNTHETIC_DEMO", lookbacks=(1,), ridge_alphas=(0.1,),
                                  penalty_mode="per_sample", max_analysis_lag=3),
        )
        snapshots = []
        for n in (len(frame) - 2, len(frame) - 1, len(frame)):
            csv = run / f"closes_{n}.csv"
            frame.iloc[:n].to_csv(csv, index=False)
            snapshots.append(capture_snapshot(csv, asset="SYNTHETIC_DEMO", data_kind="synthetic", output_root=run))
        first_origin = frame.Date.iloc[-3]
        first_time = (calendar.session_close(first_origin) + pd.Timedelta(minutes=30)).isoformat()
        candidate = prepare_candidate(source, snapshots[0], mode="replay", as_of=first_time, output_root=run)
        protocol = run / "protocol.json"
        create_protocol(candidate, protocol, protocol_id="synthetic-shadow-demo", earliest_origin=first_origin.date().isoformat(),
                        max_forecasts=2)
        database = run / "ledger.sqlite"
        for snapshot, origin in zip(snapshots[:2], frame.Date.iloc[-3:-1], strict=True):
            instant = (calendar.session_close(origin) + pd.Timedelta(minutes=30)).isoformat()
            record_forecast(candidate, snapshot, protocol, database=database, mode="replay", as_of=instant, output_root=run)
        score_time = (calendar.session_close(frame.Date.iloc[-1]) + pd.Timedelta(minutes=30)).isoformat()
        scored = score_snapshot(snapshots[-1], asset="SYNTHETIC_DEMO", database=database, mode="replay",
                                as_of=score_time, output_root=run)
        report = shadow_report(database, mode="replay", as_of=score_time)
        group = report["groups"][0]
        if group["forecasts"] != 2 or group["scored"] != 2 or group["pending"] or group["missing_origins"]:
            raise RuntimeError("Demo failed its record/score/completeness acceptance checks.")
        # Exact retries return the original forecast even after its deadline.
        _, created = record_forecast(candidate, snapshots[0], protocol, database=database, mode="replay",
                                     as_of=score_time, output_root=run)
        if created:
            raise RuntimeError("Demo created a duplicate forecast.")
        checks = []
        for manifest in sorted(run.rglob("manifest.json")):
            if manifest.parent != run:
                checks.append(verify_run(manifest.parent))
        write_json(run / "report.json", report)
        write_json(run / "verification.json", checks)
        write_json(run / "receipt.json", {
            **settings, "model_run": str(candidate), "score_run": str(scored),
            "forecasts": 2, "scored": 2, "duplicate_retry_created": False,
            "verified_child_runs": len(checks), "promotion": "disabled",
        })
        logger.info("SYNTHETIC REPLAY ONLY: one pinned refit; two forecasts/labels; no market improvement claim")
    verified = verify_run(run)
    print(json.dumps({"demo_run": str(run), "verified_artifacts": verified["verified_artifacts"],
                      "data_kind": "synthetic", "mode": "replay", "market_evidence": False, "groups": report["groups"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
