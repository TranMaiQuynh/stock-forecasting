"""Nested chronological evaluation orchestrating the existing offline runner."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .experiment import sha256_file, tracked_run, write_json
from .research import ResearchConfig, _return_metrics, run_offline_experiment, synthetic_close, validate_close_frame
from .validation import expanding_window_splits


@dataclass(frozen=True)
class WalkForwardConfig:
    min_train_size: int = 756
    validation_size: int = 126
    test_size: int = 126
    final_holdout_start: str | None = None
    final_holdout_fraction: float = 0.2
    feature_groups: tuple[str, ...] = ("returns", "returns_abs")
    development_only: bool = False

    def validate(self):
        for name in ("min_train_size", "validation_size", "test_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 5:
                raise ValueError(f"{name} must be an integer of at least 5 observations.")
        if not 0 < self.final_holdout_fraction < 1:
            raise ValueError("final_holdout_fraction must be between zero and one.")
        if self.final_holdout_start is not None:
            if pd.isna(pd.to_datetime(self.final_holdout_start, utc=True, errors="raise")):
                raise ValueError("final_holdout_start must be a valid date.")
        if not isinstance(self.development_only, bool) or (self.development_only and self.final_holdout_start is None):
            raise ValueError("development_only requires a boolean flag and an explicit final_holdout_start cutoff.")
        if (
            not self.feature_groups
            or len(set(self.feature_groups)) != len(self.feature_groups)
            or not set(self.feature_groups) <= {"returns", "returns_abs"}
        ):
            raise ValueError("feature_groups must be distinct supported groups.")


def run_walk_forward_experiment(
    input_csv=None, *, output_root="log", config=None, walk_config=None, data_kind="user_supplied", protocol_path=None
) -> Path:
    """Select a feature group on development folds, then score one final holdout.

    Each child calls run_offline_experiment: causal diagnostics/scaling,
    validation-only selection, explicit refits and a reload check at every origin.
    Development outer tests can select the group; the final holdout cannot.
    """
    config = config or ResearchConfig()
    walk = walk_config or WalkForwardConfig()
    config.validate()
    walk.validate()
    metric_name = "selected_log_return" if config.include_baselines else "ridge_log_return"
    if data_kind not in {"market", "synthetic", "user_supplied"}:
        raise ValueError("Unknown data_kind.")
    kind = "synthetic" if input_csv is None else data_kind
    settings = {
        "research": asdict(config),
        "walk_forward": asdict(walk),
        "data_kind": kind,
        "input_csv": str(input_csv) if input_csv else None,
        "protocol": "outer development folds compare groups; inner validation selects candidates; explicit causal refit policy",
        "protocol_sha256": sha256_file(protocol_path) if protocol_path else None,
    }
    model_name = "walk_forward_return" if config.include_baselines else "walk_forward_ridge"
    with tracked_run(output_root, model_name, settings) as (run, logger):
        frame = synthetic_close(config.seed) if input_csv is None else pd.read_csv(input_csv)
        if walk.development_only:
            if "Date" not in frame:
                raise ValueError("CSV must have Date and Close columns for a single asset.")
            dates = pd.to_datetime(frame.Date, utc=True, errors="raise")
            frame = frame.loc[dates < pd.to_datetime(walk.final_holdout_start, utc=True)].copy()
        close = validate_close_frame(frame)
        snapshot = run / "input_snapshot.csv"
        close.to_csv(snapshot)
        if protocol_path:
            (run / "protocol.json").write_bytes(Path(protocol_path).read_bytes())
        if walk.final_holdout_start is None:
            holdout_start = int(len(close) * (1 - walk.final_holdout_fraction))
        else:
            holdout_start = int(close.index.searchsorted(pd.to_datetime(walk.final_holdout_start, utc=True)))
        if (not walk.development_only and len(close) - holdout_start < 5) or holdout_start - walk.validation_size < walk.min_train_size:
            raise ValueError("Insufficient history before/within the final holdout.")
        if walk.min_train_size - max(config.lookbacks) - 1 < 12:
            raise ValueError("Insufficient common training targets for candidate lookbacks.")
        folds = list(
            expanding_window_splits(
                holdout_start,
                walk.min_train_size,
                walk.test_size,
                validation_size=walk.validation_size,
                step_size=walk.test_size,
            )
        )
        if not folds:
            raise ValueError("No complete development folds before the final holdout.")
        child_count = len(folds) * len(walk.feature_groups) + (0 if walk.development_only else 1)
        ridge_budget = child_count * len(config.lookbacks) * len(config.ridge_alphas)
        trial_budget = ridge_budget + child_count * (2 if config.include_baselines else 0)
        if len(folds) > 32 or trial_budget > 1024:
            raise ValueError("Reference evaluation permits at most 32 folds and 1024 total Ridge trials.")
        split_record = {
            "horizon": 1,
            "step_size": walk.test_size,
            "overlapping_scored_targets": False,
            "final_holdout_start_index": holdout_start,
            "final_holdout_first_target": None if walk.development_only else str(close.index[holdout_start]),
            "development_only": walk.development_only,
            "development_before": walk.final_holdout_start if walk.development_only else None,
            "development_unscored_tail": [folds[-1].test.stop, holdout_start],
            "final_train_stop": holdout_start - walk.validation_size,
            "final_validation_stop": holdout_start,
            "trial_budget": trial_budget,
            "folds": [
                {
                    "fold": i,
                    "train": [0, f.train.stop],
                    "validation": [f.validation.start, f.validation.stop],
                    "test": [f.test.start, f.test.stop],
                }
                for i, f in enumerate(folds)
            ],
        }
        write_json(run / "split_manifest.json", split_record)
        write_json(
            run / "data_manifest.json",
            {
                "asset": config.asset,
                "data_kind": kind,
                "observations": len(close),
                "snapshot_sha256": sha256_file(snapshot),
                "input_sha256": sha256_file(input_csv) if input_csv else None,
                "point_in_time_verified": False,
                "frequency": "input observation order; no exchange calendar assertion",
            },
        )
        records, prediction_frames = [], []
        # Fixed simplicity order also makes ties deterministic.
        groups = [g for g in ("returns", "returns_abs") if g in walk.feature_groups]
        for i, fold in enumerate(folds):
            reference_targets = None
            for group in groups:
                logger.info("Development fold=%d feature_group=%s", i, group)
                child = run_offline_experiment(
                    snapshot,
                    output_root=run / "folds" / f"fold_{i:03d}" / group,
                    config=replace(config, feature_group=group),
                    data_kind=kind,
                    split_boundaries=(fold.train.stop, fold.validation.stop, fold.test.stop),
                )
                predicted = pd.read_csv(child / "predictions.csv")
                predicted = predicted[predicted.split == "test"].copy()
                targets = predicted[["forecast_origin", "target_timestamp", "actual_log_return"]].reset_index(drop=True)
                if reference_targets is None:
                    reference_targets = targets
                else:
                    pd.testing.assert_frame_equal(targets, reference_targets)
                predicted["fold"] = i
                predicted["feature_group"] = group
                predicted["child_run"] = child.relative_to(run).as_posix()
                prediction_frames.append(predicted)
                metrics = json.loads((child / "metrics.json").read_text(encoding="utf-8"))["test"]
                records.append(
                    {
                        "fold": i,
                        "feature_group": group,
                        "observations": len(predicted),
                        "rmse": metrics[metric_name]["rmse"],
                        "baseline_rmse": metrics["zero_log_return"]["rmse"],
                        "rmse_skill_vs_zero": metrics[metric_name]["rmse_skill_vs_zero"],
                        "selected_estimator": json.loads((child / "artifacts/model.json").read_text(encoding="utf-8"))["estimator_kind"],
                        "child_run": child.relative_to(run).as_posix(),
                    }
                )
                pd.DataFrame(records).to_csv(run / "fold_metrics.csv", index=False)
        development = pd.concat(prediction_frames, ignore_index=True)
        if development.duplicated(["feature_group", "target_timestamp"]).any():
            raise RuntimeError("Development targets overlap across folds.")
        development.to_csv(run / "development_predictions.csv", index=False)
        aggregate = {}
        for group in groups:
            subset = development[development.feature_group == group]
            aggregate[group] = {
                "observations": len(subset),
                metric_name: _return_metrics(subset.actual_log_return, subset.predicted_log_return),
                "zero_log_return": _return_metrics(subset.actual_log_return, np.zeros(len(subset))),
            }
            aggregate[group]["always_up_directional_accuracy"] = float(np.mean(subset.actual_log_return > 0))
            if "training_mean_log_return" in subset:
                aggregate[group]["training_mean"] = _return_metrics(subset.actual_log_return, subset.training_mean_log_return)
                baseline_error = aggregate[group]["training_mean"]["rmse"]
                aggregate[group][metric_name]["rmse_skill_vs_training_mean"] = (
                    1 - aggregate[group][metric_name]["rmse"] / baseline_error if baseline_error else None
                )
        selected = groups[0]
        for group in groups[1:]:
            if aggregate[group][metric_name]["rmse"] < aggregate[selected][metric_name]["rmse"] - 1e-12:
                selected = group
        selection = {
            "selected_feature_group": selected,
            "development_metrics": aggregate,
            "selection_scope": "development outer predictions only; final holdout not scored",
            "tie_policy": "Prefer simpler returns within absolute RMSE tolerance 1e-12",
        }
        # Persist the choice before opening the final evaluation stage.
        write_json(run / "selection.json", selection)
        if walk.development_only:
            write_json(run / "metrics.json", {
                "development": aggregate, "selected_feature_group": selected, "folds": len(folds),
                "actual_ridge_trials": ridge_budget, "actual_validation_candidates": trial_budget,
                "final": {"status": "not_scored_development_only"},
                "interpretation": "Exploratory development comparison; policy selection needs a separate future confirmation; no promotion",
            })
            (run / "summary.md").write_text(
                f"# Development-only walk-forward\n\nAsset: {config.asset}; data kind: **{kind}**.\n\n"
                f"Data strictly before: {walk.final_holdout_start}; complete folds: {len(folds)}; candidates: {trial_budget}.\n\n"
                f"Refit policy: {config.refit_policy}; training window: {config.training_window}; selected group: {selected}.\n\n"
                "All candidate trials, checkpoint scopes and fold baselines are retained. Final holdout was excluded. "
                "Development results can guide selection but cannot confirm it. No profitability or promotion claim.\n",
                encoding="utf-8",
            )
            logger.info("Development folds completed; final holdout excluded")
            return run
        final = run_offline_experiment(
            snapshot,
            output_root=run / "final",
            config=replace(config, feature_group=selected),
            data_kind=kind,
            split_boundaries=(holdout_start - walk.validation_size, holdout_start, len(close)),
        )
        final_predictions = pd.read_csv(final / "predictions.csv")
        final_predictions = final_predictions[final_predictions.split == "test"].copy()
        final_predictions["feature_group"] = selected
        final_predictions.to_csv(run / "final_predictions.csv", index=False)
        final_metrics = json.loads((final / "metrics.json").read_text(encoding="utf-8"))["test"]
        report = {
            "development": aggregate,
            "selected_feature_group": selected,
            "final": final_metrics,
            "final_child_run": final.relative_to(run).as_posix(),
            "folds": len(folds),
            "actual_ridge_trials": ridge_budget,
            "actual_validation_candidates": trial_budget,
            "interpretation": "Negative results retained. Final holdout is reporting only; no retuning or promotion.",
        }
        write_json(run / "metrics.json", report)
        (run / "summary.md").write_text(
            f"# Walk-forward return research\n\nAsset: {config.asset}; data kind: **{kind}**.\n\n"
            f"Development folds: {len(folds)}; trials: {trial_budget}; selected feature group: {selected}.\n"
            f"Unscored development tail (exclusive indices): {split_record['development_unscored_tail']}.\n\n"
            f"Final return RMSE: {final_metrics[metric_name]['rmse']:.8f}; "
            f"zero-return RMSE: {final_metrics['zero_log_return']['rmse']:.8f}; "
            f"skill: {final_metrics[metric_name]['rmse_skill_vs_zero']}.\n\n"
            "Development outer folds select the feature group; inner validation selects lookback/alpha. "
            f"Refit policy: {config.refit_policy}; training window: {config.training_window}. "
            "Each forecast uses only observations available at its origin.\n\n"
            "Every child includes training ACF/PACF/ADF/KPSS, all trials, predictions, artifacts and a reload check. "
            "No trading backtest, point-in-time data guarantee, statistical significance or market-wide edge is claimed. "
            "Do not retune from the final metrics. See selection.json, fold_metrics.csv and child manifests.\n",
            encoding="utf-8",
        )
        if not (final / "artifacts/model.json").is_file() or final_predictions.empty:
            raise RuntimeError("Final model or predictions missing.")
        logger.info("All folds complete; final holdout scored after feature selection")
    return run
