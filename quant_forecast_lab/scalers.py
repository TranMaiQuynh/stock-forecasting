"""Versioned numerical persistence for fitted MinMaxScaler state, without pickle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted

from .experiment import write_json

_ARRAY_FIELDS = ("scale_", "min_", "data_min_", "data_max_", "data_range_")


def minmax_from_record(record):
    """Restore only the supported fitted numerical contract; never import a type from JSON."""
    if record.get("schema_version") != 1 or record.get("estimator") != "MinMaxScaler":
        raise ValueError("Unsupported MinMaxScaler schema.")
    count, samples = record["n_features"], record["n_samples_seen"]
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (count, samples)):
        raise ValueError("Scaler feature/sample counts must be positive integers.")
    dtype = record.get("dtype")
    if dtype not in {"float32", "float64"}:
        raise ValueError("Scaler dtype must be float32 or float64.")
    interval = np.asarray(record["feature_range"], dtype=float)
    if interval.shape != (2,) or not np.isfinite(interval).all() or interval[0] >= interval[1]:
        raise ValueError("Scaler feature_range must be finite and increasing.")
    if not isinstance(record["clip"], bool):
        raise ValueError("Scaler clip must be boolean.")
    arrays = {name: np.asarray(record[name], dtype=dtype) for name in _ARRAY_FIELDS}
    if any(value.shape != (count,) or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Scaler arrays must match n_features and contain finite values.")
    if (
        (arrays["scale_"] <= 0).any()
        or (arrays["data_range_"] < 0).any()
        or (arrays["data_max_"] < arrays["data_min_"]).any()
    ):
        raise ValueError("Invalid scaler scale or data bounds.")
    names = record.get("feature_names")
    if names is not None and (
        not isinstance(names, list)
        or len(names) != count
        or any(not isinstance(n, str) or not n for n in names)
        or len(set(names)) != count
    ):
        raise ValueError("Scaler feature names must be ordered, nonempty and unique.")
    scaler = MinMaxScaler(feature_range=tuple(interval), clip=record["clip"])
    for name, value in arrays.items():
        setattr(scaler, name, value)
    scaler.n_features_in_ = count
    scaler.n_samples_seen_ = samples
    if names is not None:
        scaler.feature_names_in_ = np.asarray(names, dtype=object)
    return scaler


def minmax_to_record(scaler):
    """Serialize learned parameters, their dtype and feature order, not executable code."""
    if type(scaler) is not MinMaxScaler:
        raise ValueError("Only a fitted sklearn MinMaxScaler is supported.")
    check_is_fitted(scaler, _ARRAY_FIELDS)
    record = {
        "schema_version": 1,
        "estimator": "MinMaxScaler",
        "source_sklearn_version": sklearn.__version__,
        "n_features": int(scaler.n_features_in_),
        "n_samples_seen": int(scaler.n_samples_seen_),
        "dtype": scaler.scale_.dtype.name,
        "feature_range": list(scaler.feature_range),
        "clip": bool(scaler.clip),
        "feature_names": scaler.feature_names_in_.tolist() if hasattr(scaler, "feature_names_in_") else None,
        **{name: getattr(scaler, name).tolist() for name in _ARRAY_FIELDS},
    }
    minmax_from_record(record)
    return record


def save_minmax_scaler(scaler, path):
    write_json(path, minmax_to_record(scaler))


def load_minmax_scaler(path):
    return minmax_from_record(json.loads(Path(path).read_text(encoding="utf-8")))
