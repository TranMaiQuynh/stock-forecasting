import copy
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = runpy.run_path(str(ROOT / "benchmarks/run_adaptive_research.py"))


def protocol():
    return json.loads((ROOT / "benchmarks/protocols/baseline_guard_v2.json").read_text())


def test_registered_protocol_has_explicit_704_candidate_budget_and_causal_policies():
    data = protocol()
    configs = RUNNER["validate_protocol"](data)
    assert len(configs) == 4
    assert all(c.include_baselines and c.penalty_mode == "per_sample" for c in configs)
    assert [(c.refit_policy, c.training_window) for c in configs] == [
        ("frozen", None), ("frozen", 756), ("monthly", None), ("monthly", 756),
    ]
    assert data["budget"]["total_validation_candidates"] == 704
    assert data["confirmation"]["status"] == "pending_future_evaluation"


@pytest.mark.parametrize("section,key,value", [
    (None, "development_before", "2024-01-01"), (None, "development_before", "NaT"),
    ("budget", "total_validation_candidates", 703), ("development", "step_size", 1),
    (None, "baselines", ["zero"]), (None, "assets", ["../AAPL", "MSFT"]),
])
def test_invalid_or_holdout_reusing_protocols_are_rejected(section, key, value):
    data = copy.deepcopy(protocol())
    target = data if section is None else data[section]
    target[key] = value
    with pytest.raises(ValueError):
        RUNNER["validate_protocol"](data)


def test_source_run_is_verified_before_any_market_experiment(tmp_path, monkeypatch):
    import quant_forecast_lab.artifacts as artifacts

    def corrupt_source(*args, **kwargs):
        raise ValueError("source integrity failure")

    monkeypatch.setattr(artifacts, "verify_run", corrupt_source)
    with pytest.raises(ValueError, match="source integrity failure"):
        RUNNER["main"](["--source-run", str(tmp_path), "--protocol", str(ROOT / "benchmarks/protocols/baseline_guard_v2.json")])
    assert not (tmp_path / "baseline_guard_development").exists()
