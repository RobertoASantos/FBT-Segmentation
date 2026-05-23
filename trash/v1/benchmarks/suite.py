from __future__ import annotations

import argparse
import gzip
import io
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import urllib3
from xgboost import XGBClassifier

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from riskseg import RiskSegOptimizer

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


PMLB_DATA_ROOT = "https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets"


@dataclass(frozen=True)
class BenchmarkDatasetSpec:
    name: str
    expected_rows: int
    expected_features: int
    expected_classes: int = 2
    positive_class: int | float | str = 1
    positive_label_name: str | None = None
    categorical_columns: tuple[str, ...] = ()
    drop_columns: tuple[str, ...] = ()


@dataclass
class BenchmarkResult:
    dataset: str
    model: str
    source_rows: int
    rows: int
    features: int
    train_rows: int
    test_rows: int
    positive_rate_train: float
    positive_rate_test: float
    roc_auc: float
    average_precision: float
    accuracy: float
    fit_seconds: float
    score_seconds: float


class DataFrameImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        frame = pd.DataFrame(X).copy()
        self.columns_ = list(frame.columns)
        self.numeric_columns_ = frame.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns_ = [c for c in self.columns_ if c not in self.numeric_columns_]

        self.numeric_fill_values_ = {
            column: float(frame[column].median()) if frame[column].notna().any() else 0.0
            for column in self.numeric_columns_
        }
        self.categorical_fill_values_ = {}
        for column in self.categorical_columns_:
            mode = frame[column].mode(dropna=True)
            self.categorical_fill_values_[column] = mode.iloc[0] if not mode.empty else "missing"
        return self

    def transform(self, X):
        frame = pd.DataFrame(X, columns=self.columns_).copy()
        for column, fill_value in self.numeric_fill_values_.items():
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(fill_value)
        for column, fill_value in self.categorical_fill_values_.items():
            frame[column] = frame[column].astype(object).where(frame[column].notna(), fill_value)
        return frame


def decode_arff_frame(frame: pd.DataFrame) -> pd.DataFrame:
    decoded = frame.copy()
    for column in decoded.columns:
        if decoded[column].dtype == object:
            decoded[column] = decoded[column].map(
                lambda value: value.decode("utf-8", errors="replace")
                if isinstance(value, (bytes, bytearray))
                else value
            )
            decoded[column] = decoded[column].replace("?", np.nan)
    return decoded


def get_dataset_specs() -> list[BenchmarkDatasetSpec]:
    return [
        BenchmarkDatasetSpec(
            "adult",
            48842,
            14,
            categorical_columns=(
                "workclass",
                "education",
                "marital-status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "native-country",
            ),
        ),
        BenchmarkDatasetSpec(
            "churn",
            5000,
            20,
            categorical_columns=("international plan", "voice mail plan"),
        ),
        BenchmarkDatasetSpec("phoneme", 5404, 5),
        BenchmarkDatasetSpec("clean2", 6598, 168),
        BenchmarkDatasetSpec("ring", 7400, 20),
        BenchmarkDatasetSpec("twonorm", 7400, 20),
        BenchmarkDatasetSpec(
            "mushroom",
            8124,
            22,
            categorical_columns=(
                "cap-shape",
                "cap-surface",
                "cap-color",
                "bruises?",
                "odor",
                "gill-attachment",
                "gill-spacing",
                "gill-size",
                "gill-color",
                "stalk-shape",
                "stalk-root",
                "stalk-surface-above-ring",
                "stalk-surface-below-ring",
                "stalk-color-above-ring",
                "stalk-color-below-ring",
                "veil-type",
                "veil-color",
                "ring-number",
                "ring-type",
                "spore-print-color",
                "population",
                "habitat",
            ),
        ),
        BenchmarkDatasetSpec("coil2000", 9822, 85),
        BenchmarkDatasetSpec("magic", 19020, 10),
        BenchmarkDatasetSpec("shuttle", 58000, 9, positive_class=1, positive_label_name="class_1_vs_rest"),
    ]


def apply_dataset_schema(frame: pd.DataFrame, spec: BenchmarkDatasetSpec) -> pd.DataFrame:
    transformed = frame.copy()

    if spec.drop_columns:
        transformed = transformed.drop(columns=list(spec.drop_columns), errors="ignore")

    for column in spec.categorical_columns:
        if column not in transformed.columns:
            continue
        transformed[column] = transformed[column].map(
            lambda value: np.nan if pd.isna(value) else str(value)
        ).astype(object)

    return transformed


def build_one_hot_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            )
        )

    return ColumnTransformer(transformers=transformers, sparse_threshold=0.0)


def build_mixed_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )

    return ColumnTransformer(transformers=transformers, sparse_threshold=0.0)


def _xgboost_estimator(random_state: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )


def get_model_builders(random_state: int = 42):
    def logistic_builder(
        X: pd.DataFrame,
        y: pd.Series | None = None,
        spec: BenchmarkDatasetSpec | None = None,
    ):
        return Pipeline(
            steps=[
                ("preprocess", build_one_hot_preprocessor(X)),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        solver="lbfgs",
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def bagging_lr_builder(
        X: pd.DataFrame,
        y: pd.Series | None = None,
        spec: BenchmarkDatasetSpec | None = None,
    ):
        return Pipeline(
            steps=[
                ("preprocess", build_one_hot_preprocessor(X)),
                (
                    "model",
                    BaggingClassifier(
                        estimator=LogisticRegression(
                            C=1.0,
                            solver="lbfgs",
                            max_iter=2000,
                            random_state=random_state,
                        ),
                        n_estimators=25,
                        max_samples=0.8,
                        bootstrap=True,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def scaled_builder(estimator):
        def builder(
            X: pd.DataFrame,
            y: pd.Series | None = None,
            spec: BenchmarkDatasetSpec | None = None,
        ):
            return Pipeline(
                steps=[
                    ("preprocess", build_one_hot_preprocessor(X)),
                    ("model", estimator),
                ]
            )

        return builder

    def raw_builder(estimator):
        def builder(
            X: pd.DataFrame,
            y: pd.Series | None = None,
            spec: BenchmarkDatasetSpec | None = None,
        ):
            return Pipeline(
                steps=[
                    ("preprocess", build_mixed_preprocessor(X)),
                    ("model", estimator),
                ]
            )

        return builder

    def riskseg_builder(
        X: pd.DataFrame,
        y: pd.Series | None = None,
        spec: BenchmarkDatasetSpec | None = None,
    ):
        params = recommend_riskseg_benchmark_params(X, y, spec, random_state=random_state)
        return Pipeline(
            steps=[
                ("imputer", DataFrameImputer()),
                (
                    "model",
                    RiskSegOptimizer(**params),
                ),
            ]
        )

    return {
        "RiskSeg": riskseg_builder,
        "LogisticRegression": logistic_builder,
        "MLPClassifier": scaled_builder(
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                alpha=1e-4,
                learning_rate_init=1e-3,
                batch_size=128,
                max_iter=150,
                early_stopping=True,
                random_state=random_state,
            )
        ),
        "BaggingLogisticRegression": bagging_lr_builder,
        "XGBoost": raw_builder(_xgboost_estimator(random_state)),
        "RandomForest": raw_builder(
            RandomForestClassifier(
                n_estimators=300,
                max_features="sqrt",
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            )
        ),
        "GaussianNB": scaled_builder(GaussianNB(var_smoothing=1e-9)),
        "DecisionTree": raw_builder(
            DecisionTreeClassifier(
                min_samples_leaf=10,
                random_state=random_state,
            )
        ),
        "ExtraTrees": raw_builder(
            ExtraTreesClassifier(
                n_estimators=300,
                max_features="sqrt",
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            )
        ),
        "HistGradientBoosting": raw_builder(
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=6,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                random_state=random_state,
            )
        ),
        "AdaBoost": raw_builder(
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=2, random_state=random_state),
                n_estimators=200,
                learning_rate=0.05,
                random_state=random_state,
            )
        ),
        "KNeighbors": scaled_builder(
            KNeighborsClassifier(
                n_neighbors=21,
                weights="distance",
                metric="minkowski",
            )
        ),
    }


def recommend_riskseg_benchmark_params(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray | None,
    spec: BenchmarkDatasetSpec | None,
    random_state: int = 42,
) -> dict:
    categorical_features = tuple(spec.categorical_columns) if spec is not None else ()
    params = {
        "metric": "auc",
        "max_depth": 2,
        "min_samples_leaf": 0.05,
        "validation_fraction": 0.35,
        "combiner_method": "stacking",
        "prediction_mode": "global_stacking",
        "use_validation_to_accept_split": True,
        "use_grouping": False,
        "top_k_variables": 1,
        "factorial_max_interaction_features": 8,
        "n_numeric_bins": 4,
        "categorical_features": categorical_features,
        "group_binned_numeric": True,
        "scale_model_numeric": True,
        "random_state": random_state,
        "verbose": 0,
    }

    numeric_only = (
        len(categorical_features) == 0
        and all(pd.api.types.is_numeric_dtype(X[column]) for column in X.columns)
    )

    if not numeric_only:
        return params

    positive_rate = None
    if y is not None:
        positive_rate = float(pd.Series(y).mean())

    params["top_k_variables"] = 3

    if X.shape[1] >= 40 and positive_rate is not None and positive_rate <= 0.10:
        params["n_numeric_bins"] = 8
        params["max_depth"] = 2
        return params

    if X.shape[1] <= 10:
        params["n_numeric_bins"] = 6
        params["max_depth"] = 3
        return params

    params["n_numeric_bins"] = 4
    params["max_depth"] = 3
    return params


def load_pmlb_dataset(
    spec: BenchmarkDatasetSpec,
    cache_dir: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = cache_dir / f"{spec.name}.tsv.gz"

    if not dataset_path.exists():
        response = requests.get(
            f"{PMLB_DATA_ROOT}/{spec.name}/{spec.name}.tsv.gz",
            verify=False,
            timeout=120,
        )
        response.raise_for_status()
        dataset_path.write_bytes(response.content)

    frame = pd.read_csv(io.BytesIO(gzip.decompress(dataset_path.read_bytes())), sep="\t")
    target = frame.pop("target")

    if spec.expected_classes == 2 and target.nunique(dropna=True) > 2:
        target = (target == spec.positive_class).astype(int)
    else:
        target = target.astype(int)

    if len(frame) != spec.expected_rows:
        raise ValueError(f"{spec.name} returned {len(frame)} rows, expected {spec.expected_rows}.")
    if frame.shape[1] != spec.expected_features:
        raise ValueError(
            f"{spec.name} returned {frame.shape[1]} features, expected {spec.expected_features}."
        )
    if target.nunique(dropna=True) != spec.expected_classes:
        raise ValueError(f"{spec.name} returned {target.nunique(dropna=True)} classes.")

    return frame, target.astype(int)


def evaluate_model(
    dataset_name: str,
    model_name: str,
    estimator,
    source_rows: int,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> BenchmarkResult:
    fit_start = time.perf_counter()
    estimator.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    score_start = time.perf_counter()
    probabilities = estimator.predict_proba(X_test)[:, 1]
    predictions = estimator.predict(X_test)
    score_seconds = time.perf_counter() - score_start

    positive_label = sorted(y_train.unique())[-1]
    y_test_binary = (y_test == positive_label).astype(int)

    return BenchmarkResult(
        dataset=dataset_name,
        model=model_name,
        source_rows=source_rows,
        rows=len(X_train) + len(X_test),
        features=X_train.shape[1],
        train_rows=len(X_train),
        test_rows=len(X_test),
        positive_rate_train=float((y_train == positive_label).mean()),
        positive_rate_test=float((y_test == positive_label).mean()),
        roc_auc=float(roc_auc_score(y_test_binary, probabilities)),
        average_precision=float(average_precision_score(y_test_binary, probabilities)),
        accuracy=float(accuracy_score(y_test, predictions)),
        fit_seconds=float(fit_seconds),
        score_seconds=float(score_seconds),
    )


def run_benchmark(
    output_dir: str | Path = "artifacts/benchmarks",
    random_state: int = 42,
    test_size: float = 0.2,
    max_rows: int | None = None,
    dataset_names: list[str] | None = None,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_dir = output_path / "openml_cache"

    dataset_specs = get_dataset_specs()
    if dataset_names:
        requested = set(dataset_names)
        dataset_specs = [spec for spec in dataset_specs if spec.name in requested]

    model_builders = get_model_builders(random_state=random_state)
    if model_names:
        requested = set(model_names)
        model_builders = {
            name: builder for name, builder in model_builders.items() if name in requested
        }

    results: list[BenchmarkResult] = []

    for spec in dataset_specs:
        X, y = load_pmlb_dataset(spec, cache_dir=cache_dir)
        X = apply_dataset_schema(X, spec)
        source_rows = len(X)
        if max_rows is not None and len(X) > max_rows:
            sampled = train_test_split(
                X,
                y,
                train_size=max_rows,
                stratify=y,
                random_state=random_state,
            )
            X, _, y, _ = sampled
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )

        for model_name, builder in model_builders.items():
            estimator = builder(X_train, y_train, spec)
            result = evaluate_model(
                dataset_name=spec.name,
                model_name=model_name,
                estimator=estimator,
                source_rows=source_rows,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
            )
            results.append(result)
            print(
                f"[{spec.name}] {model_name}: "
                f"AUC={result.roc_auc:.4f} AP={result.average_precision:.4f} "
                f"ACC={result.accuracy:.4f} fit={result.fit_seconds:.2f}s"
            )

    frame = pd.DataFrame([asdict(result) for result in results]).sort_values(
        ["dataset", "roc_auc", "average_precision"],
        ascending=[True, False, False],
    )
    frame.to_csv(output_path / "benchmark_results.csv", index=False)

    summary = (
        frame.groupby("model")[["roc_auc", "average_precision", "accuracy", "fit_seconds"]]
        .mean()
        .sort_values("roc_auc", ascending=False)
        .reset_index()
    )
    summary.to_csv(output_path / "benchmark_summary.csv", index=False)

    rank_frame = frame.copy()
    rank_frame["auc_rank"] = rank_frame.groupby("dataset")["roc_auc"].rank(
        ascending=False, method="average"
    )
    rank_summary = (
        rank_frame.groupby("model")[["auc_rank", "roc_auc", "average_precision", "accuracy"]]
        .mean()
        .sort_values(["auc_rank", "roc_auc"], ascending=[True, False])
        .reset_index()
    )
    rank_summary.to_csv(output_path / "benchmark_rank_summary.csv", index=False)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RiskSeg benchmark suite.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/benchmarks",
        help="Directory where CSV outputs and cached datasets will be stored.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Holdout fraction used for each dataset.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split and stochastic estimators.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional stratified cap applied before the holdout split.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional subset of dataset names to run.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of model names to run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_benchmark(
        output_dir=args.output_dir,
        random_state=args.random_state,
        test_size=args.test_size,
        max_rows=args.max_rows,
        dataset_names=args.datasets,
        model_names=args.models,
    )


if __name__ == "__main__":
    main()
