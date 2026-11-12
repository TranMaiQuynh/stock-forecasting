"""Offline next-observation forecasts from an existing numerical research model."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import verify_run
from .experiment import sha256_file, tracked_run, write_json
from .research import predict_saved_model, validate_close_frame


def run_saved_forecast(run_directory, input_csv, *, asset, output_root="log"):
    """Validate a saved run and CSV history, then predict once without fitting.

    Forecast origins must follow configuration selection. Both the research
    candidate and zero-return baseline are reported; no model is promoted.
    """
    source = Path(run_directory).resolve()
    if (Path(output_root).resolve() / "saved_ridge_forecast").is_relative_to(source):
        raise ValueError("Forecast output must be outside the source model run.")
    settings = {
        "source_run": str(source),
        "input_csv": str(input_csv),
        "asset": asset,
        "data_kind": "user_supplied",
        "operation": "inference_only",
    }
    with tracked_run(output_root, "saved_ridge_forecast", settings) as (run, logger):
        verified = verify_run(source)
        if verified["model_name"] != "ridge_log_return":
            raise ValueError("Expected a completed numerical Ridge research run.")
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        required = {"artifacts/model.json", "data_manifest.json", "split_manifest.json"}
        if not required <= set(manifest["artifact_sha256"]):
            raise ValueError("Source model run is missing its data/split/model contract.")
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        data = json.loads((source / "data_manifest.json").read_text(encoding="utf-8"))
        split = json.loads((source / "split_manifest.json").read_text(encoding="utf-8"))
        if (
            not isinstance(asset, str)
            or not asset.strip()
            or asset != config.get("asset")
            or asset != data.get("asset")
        ):
            raise ValueError("Input asset must match the source model asset exactly.")
        if split.get("horizon") != 1 or split.get("target") != "next_observation_log_return":
            raise ValueError("Unsupported forecast target/horizon contract.")
        close = validate_close_frame(pd.read_csv(input_csv))
        origin = close.index[-1]
        available_after = pd.to_datetime(split["validation"]["last_target"], utc=True, errors="raise")
        if pd.isna(available_after) or origin < available_after:
            raise ValueError("Forecast origin precedes the source model's configuration-selection cutoff.")
        close.to_csv(run / "input_snapshot.csv")
        artifact = run / "artifacts/model.json"
        artifact.parent.mkdir(parents=True)
        shutil.copyfile(source / "artifacts/model.json", artifact)
        if sha256_file(artifact) != manifest["artifact_sha256"]["artifacts/model.json"]:
            raise ValueError("Copied model artifact failed checksum validation.")
        predicted = predict_saved_model(artifact, close.Close)
        origin_close = float(close.Close.iloc[-1])
        with np.errstate(over="ignore", under="ignore"):
            predicted_close = float(origin_close * np.exp(predicted))
        if not np.isfinite(predicted_close) or predicted_close <= 0:
            raise ValueError("Forecast price must be finite and positive.")
        forecast = {
            "asset": asset,
            "data_kind": "user_supplied",
            "model_data_kind": data["data_kind"],
            "forecast_origin": str(origin),
            "target_timestamp": None,
            "target_policy": "next observed close; calendar date not guessed",
            "horizon": 1,
            "actual": None,
            "origin_close": origin_close,
            "predicted_log_return": predicted,
            "predicted_close": predicted_close,
            "baseline_log_return": 0.0,
            "baseline_close": origin_close,
            "model_status": "research_candidate_not_promoted",
            "point_in_time_verified": False,
        }
        write_json(run / "forecast.json", forecast)
        write_json(
            run / "data_manifest.json",
            {
                "asset": asset,
                "observations": len(close),
                "input_sha256": sha256_file(input_csv),
                "snapshot_sha256": sha256_file(run / "input_snapshot.csv"),
                "last_observation": str(origin),
                "input_contract": "Caller supplies the same close definition/frequency as training; adjustment/calendar validity unverified",
                "missing_policy": "reject; no sorting, filling or dropping invalid observations",
            },
        )
        write_json(
            run / "model_source.json",
            {
                **verified,
                "source_run": str(source),
                "source_model_sha256": sha256_file(artifact),
                "available_after": str(available_after),
                "fit_performed": False,
            },
        )
        write_json(
            run / "capabilities.json",
            {
                "training": "not_applicable_inference_only",
                "diagnostics": "refer_to_source_training_run",
                "uncertainty": "not_implemented",
                "promotion": "not_performed",
                "execution": "not_implemented",
            },
        )
        (run / "forecast_values.txt").write_text(json.dumps(forecast, indent=2) + "\n", encoding="utf-8")
        (run / "summary.md").write_text(
            f"# Saved-model offline forecast\n\nAsset: {asset}; origin: {origin}.\n\n"
            f"Candidate next log return: {predicted:.8f}; price: {predicted_close:.8f}.\n"
            f"Zero-return baseline price: {origin_close:.8f}.\n\n"
            "No actual or future session date is invented. Source artifacts were verified; "
            "no preprocessing/model fitting, tuning or promotion occurred. "
            "A prediction is not evidence of forecast skill; refer to the source evaluation.\n",
            encoding="utf-8",
        )
        logger.info("Source artifacts verified; inference completed without fitting")
    return run
