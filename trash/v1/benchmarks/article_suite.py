from __future__ import annotations

import argparse
import io
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import urllib3

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from riskseg import RiskSegOptimizer, RiskSegRaizClassifier

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "target",
]

GERMAN_COLUMNS = [
    "status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings",
    "employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_people",
    "telephone",
    "foreign_worker",
    "target",
]

MAGIC_COLUMNS = [
    "fLength",
    "fWidth",
    "fSize",
    "fConc",
    "fConc1",
    "fAsym",
    "fM3Long",
    "fM3Trans",
    "fAlpha",
    "fDist",
    "target",
]

SPAMBASE_COLUMNS = [f"x{i}" for i in range(57)] + ["target"]
CHESS_COLUMNS = [f"A{i:02d}" for i in range(36)] + ["target"]


@dataclass(frozen=True)
class ArticleDatasetSpec:
    name: str
    categorical_columns: tuple[str, ...]


@dataclass
class ArticleBenchmarkResult:
    dataset: str
    model: str
    fold: int
    train_rows: int
    test_rows: int
    error_rate: float
    roc_auc: float
    average_precision: float
    fit_seconds: float
    score_seconds: float


class LinearProbabilityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p1 = np.clip(self.model.predict(X), 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def get_article_dataset_specs() -> list[ArticleDatasetSpec]:
    return [
        ArticleDatasetSpec(
            "adult",
            (
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
        ArticleDatasetSpec(
            "german",
            (
                "status",
                "credit_history",
                "purpose",
                "savings",
                "employment_since",
                "personal_status_sex",
                "other_debtors",
                "property",
                "other_installment_plans",
                "housing",
                "job",
                "telephone",
                "foreign_worker",
            ),
        ),
        ArticleDatasetSpec("magic", ()),
        ArticleDatasetSpec("spambase", ()),
        ArticleDatasetSpec("chess", tuple(CHESS_COLUMNS[:-1])),
    ]


def _request_text(url: str, cache_path: Path) -> str:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        response = requests.get(url, verify=False, timeout=120)
        response.raise_for_status()
        cache_path.write_text(response.text, encoding="utf-8")
    return cache_path.read_text(encoding="utf-8")


def _load_adult(cache_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    train_text = _request_text(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        cache_dir / "adult.data",
    )
    test_text = _request_text(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
        cache_dir / "adult.test",
    )
    train = pd.read_csv(
        io.StringIO(train_text),
        header=None,
        names=ADULT_COLUMNS,
        skipinitialspace=True,
    )
    test = pd.read_csv(
        io.StringIO(test_text),
        header=None,
        names=ADULT_COLUMNS,
        skipinitialspace=True,
        comment="|",
    )
    frame = pd.concat([train, test], ignore_index=True)
    frame["target"] = (
        frame["target"].astype(str).str.strip().str.rstrip(".").map({">50K": 1, "<=50K": 0})
    )
    for column in ADULT_COLUMNS[:-1]:
        if column not in {
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        }:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.drop(columns=["target"]), frame["target"].astype(int)


def _load_german(cache_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    text = _request_text(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
        cache_dir / "german.data",
    )
    frame = pd.read_csv(io.StringIO(text), sep=r"\s+", header=None, names=GERMAN_COLUMNS)
    categorical = set(get_article_dataset_specs()[1].categorical_columns)
    for column in GERMAN_COLUMNS[:-1]:
        if column not in categorical:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    target = (frame["target"].astype(int) == 2).astype(int)
    return frame.drop(columns=["target"]), target


def _load_magic(cache_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    text = _request_text(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data",
        cache_dir / "magic04.data",
    )
    frame = pd.read_csv(io.StringIO(text), header=None, names=MAGIC_COLUMNS)
    for column in MAGIC_COLUMNS[:-1]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    target = frame["target"].astype(str).str.strip().map({"g": 0, "h": 1}).astype(int)
    return frame.drop(columns=["target"]), target


def _load_spambase(cache_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    text = _request_text(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",
        cache_dir / "spambase.data",
    )
    frame = pd.read_csv(io.StringIO(text), header=None, names=SPAMBASE_COLUMNS)
    for column in SPAMBASE_COLUMNS[:-1]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.drop(columns=["target"]), frame["target"].astype(int)


def _load_chess(cache_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    text = _request_text(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king-pawn/kr-vs-kp.data",
        cache_dir / "kr-vs-kp.data",
    )
    frame = pd.read_csv(io.StringIO(text), header=None, names=CHESS_COLUMNS)
    raw_target = frame["target"].astype(str).str.strip()
    classes = sorted(raw_target.unique().tolist())
    mapping = {label: index for index, label in enumerate(classes)}
    target = raw_target.map(mapping).astype(int)
    return frame.drop(columns=["target"]), target


def load_article_dataset(
    spec: ArticleDatasetSpec,
    cache_dir: str | Path = "artifacts/article_datasets",
) -> tuple[pd.DataFrame, pd.Series]:
    cache_path = Path(cache_dir) / spec.name
    loaders = {
        "adult": _load_adult,
        "german": _load_german,
        "magic": _load_magic,
        "spambase": _load_spambase,
        "chess": _load_chess,
    }
    X, y = loaders[spec.name](cache_path)
    for column in spec.categorical_columns:
        if column in X.columns:
            X[column] = X[column].astype(str)
    return X.reset_index(drop=True), y.reset_index(drop=True)


def build_article_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
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
                        ("scaler", MinMaxScaler()),
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


def get_article_model_builders(random_state: int = 42):
    def logistic_builder(X: pd.DataFrame, spec: ArticleDatasetSpec):
        return Pipeline(
            steps=[
                ("preprocess", build_article_preprocessor(X)),
                (
                    "model",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=3000,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def linear_builder(X: pd.DataFrame, spec: ArticleDatasetSpec):
        return Pipeline(
            steps=[
                ("preprocess", build_article_preprocessor(X)),
                ("model", LinearProbabilityClassifier()),
            ]
        )

    def mlp_builder(X: pd.DataFrame, spec: ArticleDatasetSpec):
        return Pipeline(
            steps=[
                ("preprocess", build_article_preprocessor(X)),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(10,),
                        activation="logistic",
                        max_iter=300,
                        early_stopping=True,
                        validation_fraction=0.35,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def riskseg_builder(X: pd.DataFrame, spec: ArticleDatasetSpec):
        return RiskSegOptimizer.paper_preset(
            categorical_features=spec.categorical_columns,
            random_state=random_state,
        )

    def riskseg_raiz_builder(X: pd.DataFrame | None, spec: ArticleDatasetSpec):
        return RiskSegRaizClassifier.article_uci_preset(
            categorical_features=spec.categorical_columns,
            random_state=random_state,
        )

    return {
        "PaperRiskSegLogistic": riskseg_builder,
        "PaperRiskSegRaizLogistic": riskseg_raiz_builder,
        "PaperLogisticRegression": logistic_builder,
        "PaperLinearProbability": linear_builder,
        "PaperMLPClassifier": mlp_builder,
    }


def evaluate_article_model(
    dataset_name: str,
    model_name: str,
    estimator,
    fold: int,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> ArticleBenchmarkResult:
    fit_start = time.perf_counter()
    estimator.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    score_start = time.perf_counter()
    probabilities = estimator.predict_proba(X_test)[:, 1]
    predictions = estimator.predict(X_test)
    score_seconds = time.perf_counter() - score_start

    return ArticleBenchmarkResult(
        dataset=dataset_name,
        model=model_name,
        fold=fold,
        train_rows=len(X_train),
        test_rows=len(X_test),
        error_rate=float(1.0 - accuracy_score(y_test, predictions)),
        roc_auc=float(roc_auc_score(y_test, probabilities)),
        average_precision=float(average_precision_score(y_test, probabilities)),
        fit_seconds=float(fit_seconds),
        score_seconds=float(score_seconds),
    )


def run_article_benchmark(
    output_dir: str | Path = "artifacts/article_benchmark",
    random_state: int = 42,
    n_splits: int = 10,
    dataset_names: list[str] | None = None,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_specs = get_article_dataset_specs()
    if dataset_names:
        requested = set(dataset_names)
        dataset_specs = [spec for spec in dataset_specs if spec.name in requested]

    model_builders = get_article_model_builders(random_state=random_state)
    if model_names:
        requested = set(model_names)
        model_builders = {
            name: builder for name, builder in model_builders.items() if name in requested
        }

    results: list[ArticleBenchmarkResult] = []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for spec in dataset_specs:
        X, y = load_article_dataset(spec)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
            X_train = X.iloc[train_idx].reset_index(drop=True)
            X_test = X.iloc[test_idx].reset_index(drop=True)
            y_train = y.iloc[train_idx].reset_index(drop=True)
            y_test = y.iloc[test_idx].reset_index(drop=True)

            for model_name, builder in model_builders.items():
                estimator = builder(X_train, spec)
                result = evaluate_article_model(
                    dataset_name=spec.name,
                    model_name=model_name,
                    estimator=estimator,
                    fold=fold,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                )
                results.append(result)
                print(
                    f"[{spec.name}][fold {fold:02d}] {model_name}: "
                    f"ERR={result.error_rate:.4f} AUC={result.roc_auc:.4f} "
                    f"fit={result.fit_seconds:.2f}s"
                )

    frame = pd.DataFrame([asdict(result) for result in results])
    frame.to_csv(output_path / "article_benchmark_results.csv", index=False)

    summary = (
        frame.groupby(["dataset", "model"])[["error_rate", "roc_auc", "average_precision", "fit_seconds"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(output_path / "article_benchmark_summary.csv", index=False)

    ranking = (
        frame.assign(error_rank=frame.groupby(["dataset", "fold"])["error_rate"].rank(method="average"))
        .groupby("model")[["error_rate", "roc_auc", "fit_seconds", "error_rank"]]
        .mean()
        .sort_values(["error_rank", "error_rate"])
        .reset_index()
    )
    ranking.to_csv(output_path / "article_benchmark_rank_summary.csv", index=False)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the article-aligned RiskSeg benchmark.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/article_benchmark",
        help="Directory where CSV outputs will be stored.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for cross-validation and stochastic estimators.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=10,
        help="Number of outer stratified folds.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional subset of article dataset names.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of article benchmark models.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_article_benchmark(
        output_dir=args.output_dir,
        random_state=args.random_state,
        n_splits=args.n_splits,
        dataset_names=args.datasets,
        model_names=args.models,
    )


if __name__ == "__main__":
    main()
