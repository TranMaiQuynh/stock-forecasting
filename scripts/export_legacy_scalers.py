"""Export trusted local pickle scalers under their original sklearn version.

Pickle inputs can execute code. Use only the repository's already trusted
reference artifacts, in the matching isolated environment. Inputs are never
rewritten; numerical JSON and original-version probe outputs go to a new run.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning


def main(argv=None):
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from quant_forecast_lab.experiment import sha256_file, tracked_run, write_json
    from quant_forecast_lab.scalers import save_minmax_scaler

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Trusted reference run containing two fitted pickle scalers.")
    parser.add_argument("--output-root", default="log")
    args = parser.parse_args(argv)
    source = Path(args.source).resolve()
    files = [source / name for name in ("min_max_scaler.pkl", "input_scaler.pkl")]
    original_hashes = {p.name: sha256_file(p) for p in files}
    settings = {"source": str(source), "source_sha256": original_hashes, "sklearn_version": sklearn.__version__}
    with tracked_run(args.output_root, "scaler_migration", settings) as (run, logger):
        probes = {}
        for path in files:
            with warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                with path.open("rb") as handle:
                    scaler = pickle.load(handle)
            save_minmax_scaler(scaler, run / path.with_suffix(".json").name)
            span = np.where(scaler.data_range_ > 0, scaler.data_range_, 1)
            raw = scaler.data_min_ + np.array([-0.5, 0, 0.25, 0.5, 1, 1.5])[:, None] * span
            scaled = np.linspace(-0.25, 1.25, 9)[:, None] * np.ones((1, scaler.n_features_in_))
            frame = pd.DataFrame(raw, columns=scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else raw
            probes[path.stem] = {
                "source_sha256": original_hashes[path.name],
                "sklearn_version": sklearn.__version__,
                "raw_input": raw.tolist(),
                "transformed": scaler.transform(frame).tolist(),
                "scaled_input": scaled.tolist(),
                "inverse_transformed": scaler.inverse_transform(scaled).tolist(),
            }
            logger.info("Exported %s using matching sklearn %s", path.name, sklearn.__version__)
        write_json(run / "original_version_probes.json", probes)
        if any(sha256_file(path) != original_hashes[path.name] for path in files):
            raise RuntimeError("A source scaler changed during migration.")
    print(f"Scaler migration: {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
