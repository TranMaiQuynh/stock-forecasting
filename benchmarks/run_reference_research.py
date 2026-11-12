"""Replay the frozen market protocol from existing, verified local snapshots.

Run from the Git root: python benchmarks/run_reference_research.py --snapshots mapping.json
The mapping contains {"AAPL": "path/to/snapshot_run", "MSFT": "path/to/snapshot_run"}.
No downloads, trading execution or result-driven protocol edits occur here.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd


def validate_snapshot(asset, folder, source):
    from quant_forecast_lab.experiment import sha256_file

    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    provenance = json.loads((folder / "data_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed" or provenance["asset"] != asset:
        raise ValueError("Snapshot status/asset does not match the frozen protocol.")
    expected = {
        "start": source["start"],
        "end": source["end_exclusive"],
        "interval": source["interval"],
        "auto_adjust": source["auto_adjust"],
        "back_adjust": source["back_adjust"],
        "repair": source["repair"],
        "keepna": source["keepna"],
    }
    if any(provenance["download_options"].get(key) != value for key, value in expected.items()):
        raise ValueError("Snapshot acquisition options do not match the frozen protocol.")
    if provenance["yfinance_version"] != "1.0" or provenance["provider"] != "Yahoo Finance via yfinance":
        raise ValueError("The frozen protocol requires the identified Yahoo/yfinance 1.0 snapshot.")
    for name, digest in manifest["artifact_sha256"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha256_file(path) != digest:
            raise ValueError("Snapshot artifacts failed checksum validation.")
    if sha256_file(folder / "close.csv") != provenance["snapshot_sha256"]:
        raise ValueError("Snapshot close checksum mismatch.")
    return folder / "close.csv"


def main(argv=None):
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from quant_forecast_lab.experiment import sha256_file, tracked_run, write_json
    from quant_forecast_lab.research import ResearchConfig
    from quant_forecast_lab.walk_forward import WalkForwardConfig, run_walk_forward_experiment

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", required=True, help="JSON mapping asset to an existing snapshot run directory.")
    parser.add_argument("--protocol", default="benchmarks/protocols/return_history_ablation.json")
    parser.add_argument("--output-root", default="log")
    args = parser.parse_args(argv)
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    mapping = json.loads(Path(args.snapshots).read_text(encoding="utf-8-sig"))
    if protocol["protocol_id"] != "return-history-ablation-v1" or protocol["horizon"] != 1:
        raise ValueError("Unsupported frozen research protocol.")
    if protocol["development"]["step_size"] != protocol["development"]["test_size"]:
        raise ValueError("This protocol requires nonoverlapping development targets.")
    snapshots = {asset: validate_snapshot(asset, mapping[asset], protocol["source"]) for asset in protocol["assets"]}
    settings = {
        "protocol_sha256": sha256_file(args.protocol),
        "runner_sha256": sha256_file(__file__),
        "data_kind": "market",
        "snapshots": {asset: {"path": str(path), "sha256": sha256_file(path)} for asset, path in snapshots.items()},
    }
    with tracked_run(args.output_root, "market_protocol", settings) as (run, logger):
        (run / "protocol.json").write_bytes(Path(args.protocol).read_bytes())
        rows, references = [], {}
        for asset, source in snapshots.items():
            config = ResearchConfig(
                asset=asset,
                lookbacks=tuple(protocol["lookbacks"]),
                ridge_alphas=tuple(protocol["ridge_alphas"]),
                max_analysis_lag=protocol["max_analysis_lag"],
                diagnostic_alpha=protocol["diagnostic_alpha"],
                seed=42,
            )
            walk = WalkForwardConfig(
                min_train_size=protocol["development"]["min_train_size"],
                validation_size=protocol["development"]["validation_size"],
                test_size=protocol["development"]["test_size"],
                final_holdout_start=protocol["final_holdout_start"],
                feature_groups=tuple(protocol["feature_groups"]),
            )
            write_json(run / f"{asset}_settings.json", {"research": asdict(config), "walk_forward": asdict(walk)})
            result = run_walk_forward_experiment(
                source,
                output_root=run / asset,
                config=config,
                walk_config=walk,
                data_kind="market",
                protocol_path=args.protocol,
            )
            references[asset] = result.relative_to(run).as_posix()
            metrics = json.loads((result / "metrics.json").read_text(encoding="utf-8"))
            final_model = json.loads(
                (result / metrics["final_child_run"] / "artifacts/model.json").read_text(encoding="utf-8")
            )
            rows.append(
                {
                    "asset": asset,
                    "development_folds": metrics["folds"],
                    "ridge_trials": metrics["actual_ridge_trials"],
                    "development_returns_rmse": metrics["development"]["returns"]["ridge_log_return"]["rmse"],
                    "development_returns_abs_rmse": metrics["development"]["returns_abs"]["ridge_log_return"]["rmse"],
                    "selected_feature_group": metrics["selected_feature_group"],
                    "final_lookback": final_model["lookback"],
                    "final_ridge_alpha": final_model["ridge_alpha"],
                    "final_observations": metrics["final"]["observations"],
                    "final_return_rmse": metrics["final"]["ridge_log_return"]["rmse"],
                    "final_zero_return_rmse": metrics["final"]["zero_log_return"]["rmse"],
                    "final_rmse_skill_vs_zero": metrics["final"]["ridge_log_return"]["rmse_skill_vs_zero"],
                }
            )
            pd.DataFrame(rows).to_csv(run / "market_results.csv", index=False)
            write_json(run / "child_runs.json", references)
            logger.info(
                "Completed asset=%s folds=%d trials=%d", asset, metrics["folds"], metrics["actual_ridge_trials"]
            )
        (run / "summary.md").write_text(
            "# Frozen market research protocol\n\n"
            + pd.DataFrame(rows).to_string(index=False)
            + "\n\nPositive skill means lower return RMSE than the zero-return baseline; negative skill means worse. "
            "This is a descriptive experiment on two preselected surviving companies, not statistical significance "
            "or a trading profitability claim. Final results must not be used to retune this protocol. "
            "Data were downloaded retrospectively, with vendor-adjusted closes; point-in-time correctness is unverified. "
            "All child artifacts, fold comparisons and negative results are retained.\n",
            encoding="utf-8",
        )
    print(f"Market protocol run: {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
