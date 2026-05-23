import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import DataConversionWarning
from sklearn.base import clone, is_classifier
from sklearn.pipeline import Pipeline

from riskseg import RiskSegOptimizer, RiskSegRaizClassifier


def _make_synthetic_data(n=180, seed=7):
    rng = np.random.default_rng(seed)
    segment = rng.choice(["A", "B", "C"], size=n, p=[0.35, 0.45, 0.20])
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    score = 1.4 * x1 - 0.7 * x2 + 0.8 * (segment == "A") + rng.normal(scale=0.3, size=n)
    y = (score > np.median(score)).astype(int)
    X = pd.DataFrame({"segment": segment, "x1": x1, "x2": x2})
    return X, y


def test_riskseg_is_importable_classifier_and_cloneable():
    model = RiskSegOptimizer(
        max_depth=1,
        min_samples_leaf=0.15,
        validation_fraction=0.25,
        screening_mode="factorial_segment_only",
        verbose=0,
    )

    cloned = clone(model)

    assert isinstance(cloned, RiskSegOptimizer)
    assert cloned.get_params()["max_depth"] == 1
    assert is_classifier(model)


def test_riskseg_raiz_is_importable_classifier_cloneable_and_thesis_aligned():
    model = RiskSegRaizClassifier(random_state=123, verbose=0)

    cloned = clone(model)
    params = cloned.get_params()

    assert isinstance(cloned, RiskSegRaizClassifier)
    assert is_classifier(model)
    assert params["metric"] == "error"
    assert params["prediction_mode"] == "local_combiner"
    assert params["combiner_method"] == "stacking"
    assert params["max_depth"] == 2
    assert params["min_samples_leaf"] == 0.05
    assert params["validation_fraction"] == 0.35
    assert params["n_numeric_bins"] == 4
    assert params["top_k_variables"] == 3
    assert params["logistic_C"] == 100.0
    assert params["random_state"] == 123


def test_riskseg_can_be_pipeline_final_estimator():
    X, y = _make_synthetic_data()
    pipe = Pipeline(
        [
            (
                "riskseg",
                RiskSegOptimizer(
                    max_depth=1,
                    min_samples_leaf=0.15,
                    validation_fraction=0.25,
                    screening_mode="factorial_segment_only",
                    prediction_mode="global_stacking",
                    verbose=0,
                ),
            )
        ]
    )

    pipe.fit(X, y)
    proba = pipe.predict_proba(X.iloc[:12])

    assert proba.shape == (12, 2)
    assert np.all((proba[:, 1] >= 0.0) & (proba[:, 1] <= 1.0))


def test_riskseg_supports_local_combiner_prediction_mode():
    X, y = _make_synthetic_data()
    model = RiskSegOptimizer(
        max_depth=1,
        min_samples_leaf=0.15,
        validation_fraction=0.25,
        screening_mode="factorial_segment_only",
        prediction_mode="local_combiner",
        verbose=0,
    ).fit(X, y)

    proba = model.predict_proba(X.iloc[:10])

    assert proba.shape == (10, 2)
    assert np.all((proba[:, 1] >= 0.0) & (proba[:, 1] <= 1.0))


def test_predict_rejects_inputs_with_unexpected_feature_count():
    X, y = _make_synthetic_data()
    model = RiskSegOptimizer(
        max_depth=1,
        min_samples_leaf=0.15,
        validation_fraction=0.25,
        screening_mode="factorial_segment_only",
        verbose=0,
    ).fit(X, y)

    with pytest.raises(ValueError, match="RiskSegOptimizer is expecting 3 features"):
        model.predict(np.zeros((10, 2)))


def test_predict_rejects_1d_input_with_sklearn_message():
    X, y = _make_synthetic_data()
    model = RiskSegOptimizer(
        max_depth=1,
        min_samples_leaf=0.15,
        validation_fraction=0.25,
        screening_mode="factorial_segment_only",
        verbose=0,
    ).fit(X, y)

    with pytest.raises(ValueError, match="Reshape your data"):
        model.predict(X.iloc[0].to_numpy())


def test_fit_rejects_complex_features():
    X, y = _make_synthetic_data()
    X_complex = X.copy()
    X_complex["x1"] = X_complex["x1"].astype(complex) + 1j

    with pytest.raises(ValueError, match="Complex data not supported"):
        RiskSegOptimizer(
            max_depth=1,
            min_samples_leaf=0.15,
            validation_fraction=0.25,
            screening_mode="factorial_segment_only",
            verbose=0,
        ).fit(X_complex, y)


def test_fit_rejects_unhashable_object_features():
    X, y = _make_synthetic_data()
    X_bad = X.copy()
    X_bad["segment"] = [{"bad": value} for value in range(len(X_bad))]

    with pytest.raises(TypeError, match="argument must be a string or number"):
        RiskSegOptimizer(
            max_depth=1,
            min_samples_leaf=0.15,
            validation_fraction=0.25,
            screening_mode="factorial_segment_only",
            verbose=0,
        ).fit(X_bad, y)


def test_fit_rejects_zero_feature_input():
    y = np.array([0, 1] * 6)

    with pytest.raises(ValueError, match=r"0 feature\(s\).*minimum of 1"):
        RiskSegOptimizer(verbose=0).fit(np.empty((12, 0)), y)


def test_fit_rejects_one_sample_input_with_clear_message():
    X, y = _make_synthetic_data()

    with pytest.raises(ValueError, match="1 sample"):
        RiskSegOptimizer(verbose=0).fit(X.iloc[:1], y[:1])


def test_fit_rejects_nan_and_infinite_features():
    X, y = _make_synthetic_data()
    X_nan = X.copy()
    X_nan.loc[0, "x1"] = np.nan
    X_inf = X.copy()
    X_inf.loc[0, "x1"] = np.inf

    with pytest.raises(ValueError, match="NaN"):
        RiskSegOptimizer(verbose=0).fit(X_nan, y)

    with pytest.raises(ValueError, match="inf"):
        RiskSegOptimizer(verbose=0).fit(X_inf, y)


def test_fit_rejects_sparse_input_with_clear_message():
    sparse = pytest.importorskip("scipy.sparse")
    y = np.array([0, 1] * 6)
    X_sparse = sparse.csr_matrix(np.ones((12, 3)))

    with pytest.raises(TypeError, match="Sparse input is not supported"):
        RiskSegOptimizer(verbose=0).fit(X_sparse, y)


def test_fit_rejects_targets_without_two_classes():
    X, _ = _make_synthetic_data()
    y = np.zeros(len(X), dtype=int)

    with pytest.raises(ValueError, match="two classes"):
        RiskSegOptimizer(verbose=0).fit(X, y)


def test_fit_rejects_missing_y_with_sklearn_message():
    X, _ = _make_synthetic_data()

    with pytest.raises(ValueError, match="requires y to be passed"):
        RiskSegOptimizer(verbose=0).fit(X, None)


def test_fit_rejects_multiclass_target_with_sklearn_message():
    X, y = _make_synthetic_data()
    y_multiclass = y.copy()
    y_multiclass[:10] = 2

    with pytest.raises(ValueError, match="Only binary classification is supported"):
        RiskSegOptimizer(verbose=0).fit(X, y_multiclass)


def test_fit_predict_preserves_non_numeric_binary_class_labels():
    X, y_numeric = _make_synthetic_data()
    y = np.where(y_numeric == 1, "yes", "no")

    model = RiskSegOptimizer(
        max_depth=1,
        min_samples_leaf=0.15,
        validation_fraction=0.25,
        screening_mode="factorial_segment_only",
        verbose=0,
    ).fit(X, y)

    prediction = model.predict(X.iloc[:20])

    assert model.classes_.tolist() == ["no", "yes"]
    assert set(prediction).issubset({"no", "yes"})
    assert model.predict_proba(X.iloc[:20]).shape == (20, 2)


def test_fit_rejects_continuous_target_with_sklearn_message():
    X, _ = _make_synthetic_data()
    y = np.linspace(0.0, 1.0, len(X))

    with pytest.raises(ValueError, match="continuous"):
        RiskSegOptimizer(verbose=0).fit(X, y)


def test_fit_rejects_unknown_target_type_with_sklearn_message():
    X, _ = _make_synthetic_data()
    y = np.array([{"label": i % 2} for i in range(len(X))], dtype=object)

    with pytest.raises(ValueError, match="Unknown label type"):
        RiskSegOptimizer(verbose=0).fit(X, y)


def test_fit_warns_when_target_is_column_vector():
    X, y = _make_synthetic_data()

    with pytest.warns(DataConversionWarning):
        RiskSegOptimizer(
            max_depth=1,
            min_samples_leaf=0.15,
            validation_fraction=0.25,
            screening_mode="factorial_segment_only",
            verbose=0,
        ).fit(X, y.reshape(-1, 1))
