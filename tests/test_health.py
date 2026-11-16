"""Concurrent publisher checks for runner/watchdog shared reports."""

import json
from concurrent.futures import ThreadPoolExecutor

from quant_forecast_lab.health import write_health


def test_concurrent_health_publishers_leave_complete_documents(tmp_path):
    report = {"status": "unhealthy", "mode": "replay", "checked_as_of": "2026-11-16T23:00:00Z",
              "assets": [], "alerts": [{"code": "missing_forecasts"}]}
    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(lambda _: write_health(tmp_path, report), range(12)))
    assert json.loads((tmp_path / "health.json").read_text()) == report
    assert (tmp_path / "health.html").read_text().endswith("</html>")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["health.html", "health.json"]
