"""Verify a committed local checkout in an isolated environment with exact fixtures.

No deletion, history rewrite, network download or fixture upload is performed.
Dependency wheels must already be available in the specified uv cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def main(argv=None):
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(default_root))
    parser.add_argument("--reference-source", required=True)
    parser.add_argument("--uv", default=shutil.which("uv") or str(default_root / "log/tooling/bin/uv.exe"))
    parser.add_argument("--cache-dir", default=str(default_root / "log/uv-cache"))
    parser.add_argument("--output-root", default=str(default_root / "log/clean_replay"))
    args = parser.parse_args(argv)
    source, output_root = Path(args.source).resolve(), Path(args.output_root).resolve()
    git_root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=source, text=True).strip()
    ).resolve()
    if source != git_root or output_root.is_relative_to(source / ".git"):
        raise ValueError("Use the repository root and an output directory outside Git metadata.")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    run = output_root / run_id
    run.mkdir(parents=True, exist_ok=False)
    checkout = run / "source"
    environment = source / ".venv" / "replay" / run_id
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    temporary = source / "log" / "replay_tmp" / run_id[-8:]
    temporary.mkdir(parents=True, exist_ok=False)
    uv, cache = str(Path(args.uv).resolve()), str(Path(args.cache_dir).resolve())
    env = {
        **os.environ,
        "PYTHONPATH": "",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "MPLBACKEND": "Agg",
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
        "TMP": str(temporary),
        "TEMP": str(temporary),
    }
    receipt = {
        "status": "running",
        "source_commit": head,
        "checkout": str(checkout),
        "environment": str(environment),
        "temporary_directory": str(temporary),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "network": "offline",
        "started_at": datetime.now(UTC).isoformat(),
        "stages": [],
    }

    def save():
        (run / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    def stage(name, command, cwd=source):
        print(f"Stage: {name}", flush=True)
        result = subprocess.run(
            command, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600
        )
        (run / f"{name}.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
        receipt["stages"].append({"name": name, "command": list(map(str, command)), "exit_code": result.returncode})
        save()
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {run / (name + '.txt')}")
        return result.stdout

    save()
    try:
        stage("clone", ["git", "clone", "--no-hardlinks", "--no-checkout", "--", str(source), str(checkout)])
        stage("checkout", ["git", "checkout", "--detach", head], cwd=checkout)
        stage(
            "environment", [uv, "venv", str(environment), "--python", sys.executable, "--offline", "--cache-dir", cache]
        )
        stage(
            "install",
            [
                uv,
                "pip",
                "install",
                "--offline",
                "--python",
                str(python),
                "-r",
                "requirements.txt",
                "-c",
                "constraints-win-py312.txt",
                "--cache-dir",
                cache,
            ],
            cwd=checkout,
        )
        stage(
            "fixtures",
            [
                str(python),
                "scripts/provision_reference.py",
                "--source",
                str(Path(args.reference_source).resolve()),
                "--destination",
                str(checkout / "examples/runs/reference-v7-ftse"),
            ],
            cwd=checkout,
        )
        probe = (
            "import sys;from pathlib import Path;import quant_forecast_lab,stock_prediction_deep_learning;"
            f"root=Path({str(checkout)!r}).resolve();"
            "assert sys.prefix!=sys.base_prefix;"
            "assert Path(quant_forecast_lab.__file__).resolve().is_relative_to(root);"
            "assert Path(stock_prediction_deep_learning.__file__).resolve().is_relative_to(root);"
            "print(sys.version);print(quant_forecast_lab.__file__)"
        )
        stage("imports", [str(python), "-c", probe], cwd=checkout)
        stage("dependencies", [uv, "pip", "check", "--python", str(python), "--cache-dir", cache])
        stage(
            "lint",
            [str(python), "-m", "ruff", "check", "quant_forecast_lab", "tests", "tests_runtime", "scripts"],
            cwd=checkout,
        )
        result = stage(
            "tests",
            [str(python), "-m", "pytest", "-q", "-o", "addopts=", "tests", "tests_runtime", "--tb=short"],
            cwd=checkout,
        )
        counts = re.findall(r"(\d+) passed", result)
        if not counts or re.search(r"\b\d+ skipped\b", result):
            raise RuntimeError("Expected a complete passing suite without skipped tests.")
        receipt["tests_passed"] = int(counts[-1])
        smoke = stage(
            "smoke",
            [
                str(python),
                "-m",
                "quant_forecast_lab.cli",
                "experiment",
                "--synthetic",
                "--asset",
                "CLEAN_REPLAY",
                "--lookbacks",
                "1",
                "--ridge-alphas",
                "1",
                "--max-analysis-lag",
                "5",
            ],
            cwd=checkout,
        )
        smoke_runs = re.findall(r"^Run: (.+)$", smoke, flags=re.MULTILINE)
        if len(smoke_runs) != 1:
            raise RuntimeError("Synthetic smoke did not report exactly one saved model run.")
        model_run = Path(smoke_runs[0].strip()).resolve()
        if not model_run.is_relative_to(checkout / "log"):
            raise RuntimeError("Synthetic model run is outside the clean checkout log directory.")
        stage(
            "verify_model",
            [str(python), "-m", "quant_forecast_lab.cli", "verify-run", "--run-dir", str(model_run)],
            cwd=checkout,
        )
        predicted = stage(
            "predict",
            [
                str(python),
                "-m",
                "quant_forecast_lab.cli",
                "predict",
                "--run-dir",
                str(model_run),
                "--csv",
                str(model_run / "input_snapshot.csv"),
                "--asset",
                "CLEAN_REPLAY",
            ],
            cwd=checkout,
        )
        forecast = json.loads(predicted)
        if forecast["actual"] is not None or forecast["target_timestamp"] is not None:
            raise RuntimeError("Inference invented a future actual or session timestamp.")
        stage(
            "verify_forecast",
            [str(python), "-m", "quant_forecast_lab.cli", "verify-run", "--run-dir", forecast["run_directory"]],
            cwd=checkout,
        )
        status = stage("git_status", ["git", "status", "--porcelain"], cwd=checkout)
        if status.strip():
            raise RuntimeError("Clean replay changed tracked/untracked source files.")
        receipt["status"] = "completed"
    except BaseException as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt["ended_at"] = datetime.now(UTC).isoformat()
        save()
    print(f"Replay verified: {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
