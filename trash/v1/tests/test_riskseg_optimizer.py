from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from riskseg import RiskSegOptimizer


def _make_synthetic_data(n=240, seed=42):
    rng = np.random.default_rng(seed)

    segment = rng.choice(["A", "B", "C"], size=n, p=[0.4, 0.4, 0.2])
    regime = rng.choice(["low", "mid", "high"], size=n, p=[0.3, 0.4, 0.3])
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)

    score = (
        1.3 * x1
        - 0.8 * x2
        + 0.6 * (segment == "A").astype(float)
        - 0.5 * (segment == "C").astype(float)
        + 0.4 * (regime == "high").astype(float)
        + 0.25 * x3
        + rng.normal(scale=0.35, size=n)
    )

    y = (score > np.median(score)).astype(int)

    X = pd.DataFrame(
        {
            "segment": segment,
            "regime": regime,
            "x1": x1,
            "x2": x2,
            "x3": x3,
        }
    )
    return X, y


def test_riskseg_fit_predict_and_evaluate():
    X, y = _make_synthetic_data()

    model = RiskSegOptimizer(
        max_depth=2,
        min_samples_leaf=0.15,
        validation_fraction=0.25,
        random_state=42,
        verbose=0,
        screening_mode="factorial_target_interactions",
        combiner_method="stacking",
        prediction_mode="global_stacking",
        auto_bin_numeric=True,
        top_k_variables=3,
    )

    fitted = model.fit(X, y)

    assert fitted is model
    assert getattr(model, "is_fitted_", False) is True

    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all((proba[:, 1] >= 0.0) & (proba[:, 1] <= 1.0))

    summary = model.get_summary()
    assert summary["n_nodes"] >= 1
    assert summary["n_leaves"] >= 1
    assert "riskseg_global_stacking_auc" in summary

    node_table = model.get_node_table()
    assert not node_table.empty

    eval_summary = model.evaluate_dataset(X, y, dataset_name="synthetic")
    assert eval_summary["dataset_name"] == "synthetic"
    assert eval_summary["n_observations"] == len(X)
    assert 0.0 <= eval_summary["auc"] <= 1.0


def test_riskseg_can_treat_integer_encoded_categories_as_categorical_features():
    X_raw, y = _make_synthetic_data(n=280, seed=123)

    mapping = {"A": 0, "B": 1, "C": 2}
    X_encoded = X_raw.copy()
    X_encoded["segment_code"] = X_encoded.pop("segment").map(mapping).astype(int)

    model_raw = RiskSegOptimizer(
        max_depth=2,
        min_samples_leaf=0.12,
        validation_fraction=0.25,
        random_state=42,
        verbose=0,
        screening_mode="factorial_target_interactions",
        combiner_method="marginal_odds",
        prediction_mode="leaf",
        top_k_variables=2,
    ).fit(X_raw, y)

    model_encoded = RiskSegOptimizer(
        max_depth=2,
        min_samples_leaf=0.12,
        validation_fraction=0.25,
        random_state=42,
        verbose=0,
        screening_mode="factorial_target_interactions",
        combiner_method="marginal_odds",
        prediction_mode="leaf",
        top_k_variables=2,
        categorical_features=["segment_code"],
    ).fit(X_encoded, y)

    raw_auc = model_raw.score(X_raw, y)
    encoded_auc = model_encoded.score(X_encoded, y)

    assert raw_auc >= 0.8
    assert encoded_auc >= 0.8
    assert abs(raw_auc - encoded_auc) <= 0.05


def test_riskseg_can_scale_numeric_model_view_without_breaking_predictions():
    X, y = _make_synthetic_data(n=220, seed=99)
    X = X.copy()
    X["large_scale"] = (X["x1"] * 1000.0) + 5000.0

    model = RiskSegOptimizer(
        max_depth=2,
        min_samples_leaf=0.12,
        validation_fraction=0.25,
        random_state=42,
        verbose=0,
        screening_mode="factorial_target_interactions",
        combiner_method="stacking",
        prediction_mode="global_stacking",
        scale_model_numeric=True,
    ).fit(X, y)

    proba = model.predict_proba(X)

    assert proba.shape == (len(X), 2)
    assert "large_scale" in model._model_numeric_scalers_


def test_binned_numeric_columns_can_generate_contiguous_groups():
    X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 16)})

    model = RiskSegOptimizer(
        auto_bin_numeric=True,
        n_numeric_bins=4,
        use_grouping=False,
        group_binned_numeric=True,
        verbose=0,
    )

    _, seg_view = model._prepare_views_fit(X)
    groups = model._generate_split_groups(seg_view["x"])

    assert any(len(group) == 2 for group in groups)
