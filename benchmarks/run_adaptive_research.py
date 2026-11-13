"""Replay registered development-only policy comparisons from verified v1 data."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd


def validate_protocol(protocol):
    """Reject inconsistent budgets, unsupported selection and unsafe path names."""
    from quant_forecast_lab.research import ResearchConfig

    if protocol["protocol_id"] != "baseline-guard-development-v2" or protocol["status"] != "exploratory_development_only":
        raise ValueError("Expected the registered development-only v2 protocol.")
    if protocol["baselines"] != ["zero", "training_mean"] or protocol["feature_group"] != "returns":
        raise ValueError("This experiment requires the registered baselines and signed-return features.")
    cutoff = pd.to_datetime(protocol["development_before"], utc=True, errors="raise")
    if pd.isna(cutoff) or cutoff > pd.Timestamp("2023-01-01", tz="UTC"):
        raise ValueError("A valid development cutoff no later than 2023-01-01 is required.")
    if not protocol["assets"] or len(set(protocol["assets"])) != len(protocol["assets"]):
        raise ValueError("Assets must be distinct and nonempty.")
    if any(not isinstance(a, str) or not a or a in {".", ".."} or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in a) for a in protocol["assets"]):
        raise ValueError("Assets must be safe uppercase ticker identifiers.")
    names = [p["name"] for p in protocol["policies"]]
    if not names or len(set(names)) != len(names) or any(
        not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name) for name in names
    ):
        raise ValueError("Policy names must be unique simple path components.")
    candidates = len(protocol["lookbacks"]) * len(protocol["ridge_lambdas"]) + 2
    budget = protocol["budget"]
    if (
        budget["candidates_per_policy_per_fold"] != candidates or budget["policies"] != len(names)
        or budget["assets"] != len(protocol["assets"])
        or budget["total_validation_candidates"] != candidates * len(names) * len(protocol["assets"]) * budget["folds_per_asset"]
    ):
        raise ValueError("Protocol candidate budget is inconsistent.")
    development = protocol["development"]
    if development["step_size"] != development["test_size"]:
        raise ValueError("Overlapping development targets are not supported.")
    configs = []
    for policy in protocol["policies"]:
        config = ResearchConfig(
            lookbacks=tuple(protocol["lookbacks"]), ridge_alphas=tuple(protocol["ridge_lambdas"]),
            include_baselines=True, penalty_mode="per_sample", refit_policy=policy["refit_policy"],
            training_window=policy["training_window"], feature_group=protocol["feature_group"],
        )
        config.validate()
        configs.append(config)
    return configs


def _figure(frame, intervals, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assets = frame.asset.unique()
    fig, axes = plt.subplots(len(assets), 1, figsize=(10, 4 * len(assets)), squeeze=False)
    for ax, asset in zip(axes[:, 0], assets, strict=True):
        part = frame[frame.asset == asset]
        x = np.arange(len(part))
        skills = part.rmse_skill_vs_zero.to_numpy() * 100
        ax.bar(x, skills, color=["#287d5f" if value >= 0 else "#ac4949" for value in skills])
        for position, (_, row) in enumerate(part.iterrows()):
            interval = intervals[asset][row.policy]["versus_zero"]
            ax.plot([position, position], [interval["interval_lower"] * 100, interval["interval_upper"] * 100], color="black")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, part.policy, rotation=12)
        ax.set_ylabel("RMSE skill vs zero (%)")
        ax.set_title(f"{asset} | development only | marginal exploratory intervals")
    fig.suptitle("Baseline-eligible policy comparison: no final test, promotion or PnL claim")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main(argv=None):
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from dataclasses import replace

    from quant_forecast_lab.artifacts import verify_run
    from quant_forecast_lab.experiment import sha256_file, tracked_run, write_json
    from quant_forecast_lab.uncertainty import paired_rmse_skill_interval
    from quant_forecast_lab.walk_forward import WalkForwardConfig, run_walk_forward_experiment

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, help="Completed original market_protocol run; no downloads.")
    parser.add_argument("--protocol", default="benchmarks/protocols/baseline_guard_v2.json")
    parser.add_argument("--output-root", default="log")
    args = parser.parse_args(argv)
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    configs = validate_protocol(protocol)
    source = Path(args.source_run).resolve()
    verified = verify_run(source)
    if verified["model_name"] != "market_protocol":
        raise ValueError("Expected an original verified market protocol run.")
    references = json.loads((source / "child_runs.json").read_text(encoding="utf-8"))
    snapshots = {}
    for asset in protocol["assets"]:
        parent = (source / references[asset]).resolve()
        if not parent.is_relative_to(source):
            raise ValueError("Source reference escapes its verified run.")
        data = json.loads((parent / "data_manifest.json").read_text(encoding="utf-8"))
        if data["asset"] != asset or data["data_kind"] != "market":
            raise ValueError("Source asset or data kind does not match.")
        snapshots[asset] = parent / "input_snapshot.csv"
    settings = {
        "data_kind": "market", "evaluation_scope": "exploratory_development_only",
        "protocol_sha256": sha256_file(args.protocol), "runner_sha256": sha256_file(__file__),
        "source_verification": verified,
        "snapshots": {a: {"path": str(p), "sha256": sha256_file(p)} for a, p in snapshots.items()},
    }
    development = protocol["development"]
    walk = WalkForwardConfig(
        min_train_size=development["min_train_size"], validation_size=development["validation_size"],
        test_size=development["test_size"], final_holdout_start=protocol["development_before"],
        development_only=True, feature_groups=(protocol["feature_group"],),
    )
    with tracked_run(args.output_root, "baseline_guard_development", settings) as (run, logger):
        (run / "protocol.json").write_bytes(Path(args.protocol).read_bytes())
        write_json(run / "source_verified.json", settings)
        rows, intervals, choices = [], {}, {}
        total_candidates = 0
        for asset, snapshot in snapshots.items():
            intervals[asset] = {}
            best_policy, best_score = None, float("inf")
            reference_targets = None
            for policy, config in zip(protocol["policies"], configs, strict=True):
                config = replace(config, asset=asset)
                logger.info("Starting asset=%s policy=%s", asset, policy["name"])
                print(f"Starting {asset}: {policy['name']}", flush=True)
                child = run_walk_forward_experiment(
                    snapshot, output_root=run / asset / policy["name"], config=config,
                    walk_config=walk, data_kind="market", protocol_path=args.protocol,
                )
                metrics = json.loads((child / "metrics.json").read_text(encoding="utf-8"))
                if metrics["folds"] != protocol["budget"]["folds_per_asset"]:
                    raise ValueError("Actual development folds differ from the registered budget.")
                total_candidates += metrics["actual_validation_candidates"]
                predicted = pd.read_csv(child / "development_predictions.csv")
                target_view = predicted[["forecast_origin", "target_timestamp", "actual_log_return"]].reset_index(drop=True)
                if reference_targets is None:
                    reference_targets = target_view
                else:
                    pd.testing.assert_frame_equal(target_view, reference_targets)
                if not (pd.to_datetime(predicted.target_timestamp, utc=True) < pd.to_datetime(protocol["development_before"], utc=True)).all():
                    raise RuntimeError("Excluded holdout targets entered development results.")
                stats = metrics["development"][protocol["feature_group"]]
                selected = stats["selected_log_return"]
                folds = pd.read_csv(child / "fold_metrics.csv")
                row = {
                    "asset": asset, "policy": policy["name"], "observations": len(predicted),
                    "folds": metrics["folds"], "validation_candidates": metrics["actual_validation_candidates"],
                    "rmse": selected["rmse"], "mae": selected["mae"],
                    "rmse_skill_vs_zero": selected["rmse_skill_vs_zero"],
                    "rmse_skill_vs_training_mean": selected["rmse_skill_vs_training_mean"],
                    "directional_accuracy": selected["directional_accuracy"],
                    "always_up_directional_accuracy": stats["always_up_directional_accuracy"],
                    "baseline_selected_folds": int((folds.selected_estimator != "ridge").sum()),
                    "positive_skill_folds": int((folds.rmse_skill_vs_zero > 0).sum()),
                    "child_run": child.relative_to(run).as_posix(),
                }
                rows.append(row)
                uncertainty = protocol["uncertainty"]
                options = {k: uncertainty[k] for k in ("block_length", "replicates", "coverage", "seed")}
                intervals[asset][policy["name"]] = {
                    "versus_zero": paired_rmse_skill_interval(predicted.actual_log_return, predicted.predicted_log_return, **options),
                    "versus_training_mean": paired_rmse_skill_interval(
                        predicted.actual_log_return, predicted.predicted_log_return, predicted.training_mean_log_return, **options,
                    ),
                }
                if selected["rmse"] < best_score - 1e-12:
                    best_policy, best_score = policy["name"], selected["rmse"]
                pd.DataFrame(rows).to_csv(run / "development_comparison.csv", index=False)
                write_json(run / "uncertainty.json", intervals)
                print(f"Completed {asset}: {policy['name']}; skill vs zero={row['rmse_skill_vs_zero']:.6%}", flush=True)
            choices[asset] = {"policy": best_policy, "development_rmse": best_score, "confirmation_status": "pending_future_evaluation"}
        if total_candidates != protocol["budget"]["total_validation_candidates"]:
            raise RuntimeError("Completed candidate count differs from the registered budget.")
        write_json(run / "selected_policies.json", choices)
        write_json(run / "metrics.json", {
            "evaluation_scope": "exploratory_development_only", "actual_validation_candidates": total_candidates,
            "comparisons": rows, "selected_policies": choices,
            "confirmation": protocol["confirmation"], "model_promotion": "disabled",
            "point_in_time_verified": False,
        })
        _figure(pd.DataFrame(rows), intervals, run / "comparison.png")
        (run / "summary.md").write_text(
            "# Baseline-eligible adaptive forecasting\n\nScope: exploratory development only.\n\n"
            + pd.DataFrame(rows).drop(columns="child_run").to_string(index=False)
            + "\n\nSelected policies need independent future confirmation. Intervals are marginal, exploratory "
            "and not adjusted for policy selection. Known 2023-2025 holdout was excluded. No promotion, "
            "profitability, point-in-time data or production-readiness claim.\n", encoding="utf-8",
        )
        write_json(run / "effective_policies.json", {p["name"]: asdict(c) for p, c in zip(protocol["policies"], configs, strict=True)})
        logger.info("Completed every registered policy; candidate count=%d; confirmation remains pending", total_candidates)
    print(f"Run: {run}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
