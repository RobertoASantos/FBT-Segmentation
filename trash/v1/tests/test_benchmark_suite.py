import pandas as pd

from benchmarks.suite import (
    BenchmarkDatasetSpec,
    apply_dataset_schema,
    decode_arff_frame,
    get_dataset_specs,
    get_model_builders,
    recommend_riskseg_benchmark_params,
)


def test_dataset_specs_cover_ten_large_binary_benchmarks():
    specs = get_dataset_specs()

    assert len(specs) >= 10
    assert all(spec.expected_rows >= 5000 for spec in specs)
    assert all(spec.expected_classes == 2 for spec in specs)


def test_model_builders_include_riskseg_and_ten_baselines():
    builders = get_model_builders(random_state=42)

    assert "RiskSeg" in builders
    assert len(builders) >= 11
    assert "LogisticRegression" in builders
    assert "MLPClassifier" in builders
    assert "BaggingLogisticRegression" in builders
    assert "XGBoost" in builders


def test_decode_arff_frame_turns_bytes_and_question_marks_into_strings_and_nan():
    frame = pd.DataFrame(
        {
            "city": [b"Sao Paulo", b"Campinas"],
            "segment": [b"retail", b"?"],
            "age": [30.0, 41.0],
        }
    )

    decoded = decode_arff_frame(frame)

    assert decoded.loc[0, "city"] == "Sao Paulo"
    assert decoded.loc[0, "segment"] == "retail"
    assert pd.isna(decoded.loc[1, "segment"])


def test_apply_dataset_schema_casts_explicit_categorical_columns_to_strings():
    frame = pd.DataFrame(
        {
            "workclass": [0, 1, 0],
            "age": [25.0, 41.0, 33.0],
        }
    )
    spec = BenchmarkDatasetSpec(
        name="adult",
        expected_rows=3,
        expected_features=2,
        categorical_columns=("workclass",),
    )

    transformed = apply_dataset_schema(frame, spec)

    assert transformed["workclass"].dtype == object
    assert transformed.loc[0, "workclass"] == "0"
    assert transformed["age"].dtype == float


def test_adult_dataset_spec_declares_encoded_categorical_columns():
    adult_spec = next(spec for spec in get_dataset_specs() if spec.name == "adult")

    assert "workclass" in adult_spec.categorical_columns
    assert "native-country" in adult_spec.categorical_columns


def test_numeric_low_dimensional_benchmark_preset_uses_deeper_tree():
    X = pd.DataFrame(
        [[0.0] * 5, [1.0] * 5, [2.0] * 5, [3.0] * 5],
        columns=[f"x{i}" for i in range(5)],
    )
    y = pd.Series([0, 0, 1, 1])
    spec = BenchmarkDatasetSpec(name="toy", expected_rows=4, expected_features=5)

    params = recommend_riskseg_benchmark_params(X, y, spec)

    assert params["max_depth"] == 3
    assert params["top_k_variables"] == 3
    assert params["group_binned_numeric"] is True


def test_numeric_imbalanced_high_dimensional_preset_uses_more_bins():
    X = pd.DataFrame(
        [[float(i)] * 80 for i in range(10)],
        columns=[f"x{i}" for i in range(80)],
    )
    y = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    spec = BenchmarkDatasetSpec(name="toy", expected_rows=10, expected_features=80)

    params = recommend_riskseg_benchmark_params(X, y, spec)

    assert params["n_numeric_bins"] == 8
    assert params["max_depth"] == 2
    assert params["top_k_variables"] == 3
