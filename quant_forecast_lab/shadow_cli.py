"""CLI wiring for the local shadow pilot; heavy work is imported on execution."""

from __future__ import annotations

import json


def add_shadow_commands(subparsers):
    shadow = subparsers.add_parser("shadow", help="Record and score a bounded prospective/replay batch pilot.")
    commands = shadow.add_subparsers(dest="shadow_command", required=True)
    capture = commands.add_parser("capture", help="Capture a local CSV with a real local receipt time.")
    capture.add_argument("--csv", required=True)
    capture.add_argument("--asset", required=True)
    capture.add_argument("--data-kind", choices=["market", "user_supplied", "synthetic"], default="user_supplied")
    capture.add_argument("--calendar", choices=["XNYS", "XNAS"], default="XNYS")
    capture.add_argument("--output-root", default="log")

    prepare = commands.add_parser("prepare", help="Refit one pinned recipe once; no search or promotion.")
    prepare.add_argument("--run-dir", required=True, help="Original selected numerical research run.")
    prepare.add_argument("--snapshot-run", required=True)
    prepare.add_argument("--calendar", choices=["XNYS", "XNAS"], default="XNYS")

    init = commands.add_parser("init", help="Create a new protocol pinning one model checkpoint; refuses overwrite.")
    init.add_argument("--run-dir", required=True)
    init.add_argument("--protocol-id", required=True)
    init.add_argument("--output", required=True)
    init.add_argument("--calendar", choices=["XNYS", "XNAS"], default="XNYS")
    init.add_argument("--earliest-origin", default="2026-11-16")
    init.add_argument("--max-forecasts", type=int, default=252)
    init.add_argument("--minimum-skill", type=float, default=0.01, help="Fractional descriptive RMSE threshold, not investment return.")
    init.add_argument("--publication-delay-minutes", type=int, default=20)

    record = commands.add_parser("record", help="Predict without fitting and append one forecast transactionally.")
    record.add_argument("--run-dir", required=True)
    record.add_argument("--snapshot-run", required=True)
    record.add_argument("--protocol", required=True)

    score = commands.add_parser("score", help="Score pending forecasts using an observed later snapshot.")
    score.add_argument("--snapshot-run", required=True)
    score.add_argument("--asset", required=True)
    score.add_argument("--protocol-id")
    for parser in (prepare, record, score):
        parser.add_argument("--mode", choices=["live", "replay"], required=True)
        parser.add_argument("--as-of", help="Timezone-aware simulated instant; required only in replay, forbidden in live.")
        parser.add_argument("--output-root", default="log")
    report = commands.add_parser("report", help="Show current paired errors and collection/stop status.")
    report.add_argument("--mode", choices=["live", "replay"], required=True)
    report.add_argument("--protocol-id")
    for parser in (record, score, report):
        parser.add_argument("--ledger", default="log/shadow/ledger.sqlite")


def run_shadow_command(args):
    from .shadow import capture_snapshot, create_protocol, record_forecast, score_snapshot, shadow_report

    command = args.shadow_command
    if command == "capture":
        run = capture_snapshot(args.csv, asset=args.asset, data_kind=args.data_kind,
                               calendar=args.calendar, output_root=args.output_root)
        result = {"snapshot_run": str(run)}
    elif command == "prepare":
        from .candidate import prepare_candidate

        run = prepare_candidate(args.run_dir, args.snapshot_run, mode=args.mode, as_of=args.as_of,
                                calendar=args.calendar, output_root=args.output_root)
        result = {"model_run": str(run), "performance_evaluation": "not_performed"}
    elif command == "init":
        result = create_protocol(args.run_dir, args.output, protocol_id=args.protocol_id, calendar=args.calendar,
                                 earliest_origin=args.earliest_origin, max_forecasts=args.max_forecasts,
                                 minimum_skill=args.minimum_skill, publication_delay_minutes=args.publication_delay_minutes)
    elif command == "record":
        record, created = record_forecast(args.run_dir, args.snapshot_run, args.protocol, database=args.ledger,
                                          mode=args.mode, as_of=args.as_of, output_root=args.output_root)
        result = {"created": created, "forecast": record}
    elif command == "score":
        run = score_snapshot(args.snapshot_run, asset=args.asset, database=args.ledger, mode=args.mode,
                             as_of=args.as_of, output_root=args.output_root, protocol_id=args.protocol_id)
        result = {"score_run": str(run)}
    else:
        result = shadow_report(args.ledger, mode=args.mode, protocol_id=args.protocol_id)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0
