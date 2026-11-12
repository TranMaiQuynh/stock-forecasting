"""Command-line research utilities that preserve legacy training entry points."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from .arena import arena_frame, render_html_report
from .benchmark import benchmark_frame


def _add_research_options(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", help="Single-asset Date,Close CSV; no network download.")
    source.add_argument("--synthetic", action="store_true", help="Labelled smoke data, not a market benchmark.")
    parser.add_argument("--asset", default="UNSPECIFIED")
    parser.add_argument("--data-kind", choices=["user_supplied", "market", "synthetic"], default="user_supplied")
    parser.add_argument("--lookbacks", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--ridge-alphas", nargs="+", type=float, default=[1.0, 10.0])
    parser.add_argument("--max-analysis-lag", type=int, default=20)
    parser.add_argument("--diagnostic-alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="log")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant-forecast")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Compare aligned price predictions with a last-price naive baseline.",
    )
    benchmark.add_argument("--csv", required=True)
    benchmark.add_argument("--actual-col", required=True)
    benchmark.add_argument(
        "--prediction-col",
        action="append",
        dest="prediction_cols",
        required=True,
    )
    benchmark.add_argument("--previous-actual-col")
    benchmark.add_argument("--json", action="store_true", dest="as_json")

    arena = subparsers.add_parser(
        "arena",
        help="Compare multiple out-of-sample forecasts across assets/horizons.",
    )
    arena.add_argument("--csv", required=True)
    arena.add_argument("--actual-col", required=True)
    arena.add_argument(
        "--prediction-col",
        action="append",
        dest="prediction_cols",
        required=True,
    )
    arena.add_argument(
        "--group-col",
        action="append",
        dest="group_cols",
        default=[],
        help="Optional grouping column such as Ticker or Horizon; repeat as needed.",
    )
    arena.add_argument("--time-col", help="Optional chronological sort column such as Date.")
    arena.add_argument("--previous-actual-col")
    arena.add_argument("--cost-bps", type=float, default=0.0)
    arena.add_argument(
        "--periods-per-year", type=int, default=252, help="Use 365 for daily continuous markets such as crypto."
    )
    arena.add_argument("--output", default="reports/generated/arena.html")
    arena.add_argument("--title", default="Financial Forecasting Arena")

    experiment = subparsers.add_parser("experiment", help="Run an offline one-step log-return reference experiment.")
    _add_research_options(experiment)
    experiment.add_argument("--train-fraction", type=float, default=0.6)
    experiment.add_argument("--validation-fraction", type=float, default=0.2)
    experiment.add_argument("--feature-group", choices=["returns", "returns_abs"], default="returns")

    walk = subparsers.add_parser(
        "walk-forward", help="Compare feature groups on development folds, then score a final holdout."
    )
    _add_research_options(walk)
    walk.add_argument("--min-train-size", type=int, default=756)
    walk.add_argument("--validation-size", type=int, default=126)
    walk.add_argument("--test-size", type=int, default=126)
    walk.add_argument("--final-holdout-start", help="First final-holdout session date, e.g. 2023-01-01.")
    walk.add_argument(
        "--final-holdout-fraction", type=float, default=0.2, help="Used only when no holdout date is supplied."
    )
    walk.add_argument(
        "--feature-groups", nargs="+", choices=["returns", "returns_abs"], default=["returns", "returns_abs"]
    )

    acquire = subparsers.add_parser(
        "acquire", help="Download and validate a daily adjusted Yahoo snapshot (requires network)."
    )
    acquire.add_argument("--asset", required=True)
    acquire.add_argument("--start", required=True)
    acquire.add_argument("--end", required=True, help="Exclusive end date.")
    acquire.add_argument("--output-root", default="log")

    verify = subparsers.add_parser("verify-run", help="Verify the inventory and checksums of a completed run.")
    verify.add_argument("--run-dir", required=True)

    predict = subparsers.add_parser(
        "predict", help="Predict the next observation from a saved numerical model and CSV history."
    )
    predict.add_argument("--run-dir", required=True, help="Completed ridge_log_return child run containing the model.")
    predict.add_argument("--csv", required=True)
    predict.add_argument("--asset", required=True, help="Must match the source model asset.")
    predict.add_argument("--output-root", default="log")
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "verify-run":
        from .artifacts import verify_run

        print(json.dumps(verify_run(args.run_dir), indent=2))
        return 0

    if args.command == "predict":
        from .forecast import run_saved_forecast

        output = run_saved_forecast(args.run_dir, args.csv, asset=args.asset, output_root=args.output_root)
        result = json.loads((output / "forecast.json").read_text(encoding="utf-8"))
        print(json.dumps({"run_directory": str(output), **result}, indent=2))
        return 0

    if args.command == "acquire":
        from .market_data import acquire_market_snapshot

        output = acquire_market_snapshot(args.asset, args.start, args.end, output_root=args.output_root)
        print(f"Snapshot: {output}")
        return 0

    if args.command in {"experiment", "walk-forward"}:
        from .research import ResearchConfig, run_offline_experiment

        config = ResearchConfig(
            asset=args.asset,
            lookbacks=tuple(args.lookbacks),
            ridge_alphas=tuple(args.ridge_alphas),
            train_fraction=getattr(args, "train_fraction", 0.6),
            validation_fraction=getattr(args, "validation_fraction", 0.2),
            max_analysis_lag=args.max_analysis_lag,
            diagnostic_alpha=args.diagnostic_alpha,
            seed=args.seed,
            feature_group=getattr(args, "feature_group", "returns"),
        )
        if args.command == "walk-forward":
            from .walk_forward import WalkForwardConfig, run_walk_forward_experiment

            walk_config = WalkForwardConfig(
                min_train_size=args.min_train_size,
                validation_size=args.validation_size,
                test_size=args.test_size,
                final_holdout_start=args.final_holdout_start,
                final_holdout_fraction=args.final_holdout_fraction,
                feature_groups=tuple(args.feature_groups),
            )
            output = run_walk_forward_experiment(
                args.csv,
                output_root=args.output_root,
                config=config,
                walk_config=walk_config,
                data_kind=args.data_kind,
            )
        else:
            output = run_offline_experiment(
                args.csv, output_root=args.output_root, config=config, data_kind=args.data_kind
            )
        print(f"Run: {output}")
        return 0

    if args.command == "benchmark":
        frame = pd.read_csv(args.csv)
        result = benchmark_frame(
            frame,
            actual_col=args.actual_col,
            prediction_cols=args.prediction_cols,
            previous_actual_col=args.previous_actual_col,
        )
        if args.as_json:
            print(json.dumps(result.reset_index().to_dict(orient="records"), indent=2))
        else:
            print(result.to_string(float_format=lambda value: f"{value:.6f}"))
        return 0

    if args.command == "arena":
        frame = pd.read_csv(args.csv)
        result = arena_frame(
            frame,
            actual_col=args.actual_col,
            prediction_cols=args.prediction_cols,
            group_cols=args.group_cols,
            time_col=args.time_col,
            previous_actual_col=args.previous_actual_col,
            transaction_cost_bps=args.cost_bps,
            periods_per_year=args.periods_per_year,
        )
        output = render_html_report(result, args.output, title=args.title)
        print(result.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
        print(f"\nReport: {output}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
