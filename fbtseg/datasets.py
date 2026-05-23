"""Loaders das 5 bases UCI usadas no paper ICAI 2012.

Mantém o pacote auto-contido para reprodução da Tabela 1 do paper
(Santos & Barros, 2012a, Seção 3):

- **Adult**:    48.842 linhas, 14 vars (6 num + 8 cat), alvo `>50K`.
- **Chess**:    3.196 linhas, 36 vars categóricas, alvo "branco/preto vence".
- **German**:   1.000 linhas, 20 vars (7 num + 13 cat), risco de crédito.
- **Magic**:    19.020 linhas, 10 vars numéricas, detecção gamma vs hadron.
- **Spambase**: 4.601 linhas, 57 vars numéricas, detecção de spam.

Os dados são baixados de https://archive.ics.uci.edu/ml/ na primeira
chamada e cacheados localmente em `cache_dir` (default
`artifacts/article_datasets/<base>/`). A partir da segunda chamada,
não há rede.

Referências (`docs/references.md`):
- SANTOS & BARROS, 2012 (ICAI) — Seção 3 (datasets).
- BLAKE & MERZ — UCI Repository of Machine Learning Databases.
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import urllib3

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# --------------------------------------------------------------------------- #
# Especificacoes                                                              #
# --------------------------------------------------------------------------- #

ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "target",
]

GERMAN_COLUMNS = [
    "status", "duration", "credit_history", "purpose", "credit_amount",
    "savings", "employment_since", "installment_rate", "personal_status_sex",
    "other_debtors", "residence_since", "property", "age",
    "other_installment_plans", "housing", "existing_credits", "job",
    "num_people", "telephone", "foreign_worker", "target",
]

MAGIC_COLUMNS = [
    "fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym",
    "fM3Long", "fM3Trans", "fAlpha", "fDist", "target",
]

SPAMBASE_COLUMNS = [f"x{i}" for i in range(57)] + ["target"]
CHESS_COLUMNS = [f"A{i:02d}" for i in range(36)] + ["target"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    categorical_columns: tuple


def article_specs() -> list[DatasetSpec]:
    """As 5 bases UCI do paper ICAI 2012, com as categoricas marcadas."""
    return [
        DatasetSpec(
            "adult",
            (
                "workclass", "education", "marital-status", "occupation",
                "relationship", "race", "sex", "native-country",
            ),
        ),
        DatasetSpec(
            "german",
            (
                "status", "credit_history", "purpose", "savings",
                "employment_since", "personal_status_sex", "other_debtors",
                "property", "other_installment_plans", "housing", "job",
                "telephone", "foreign_worker",
            ),
        ),
        DatasetSpec("magic", ()),
        DatasetSpec("spambase", ()),
        DatasetSpec("chess", tuple(CHESS_COLUMNS[:-1])),
    ]


def get_spec(name: str) -> DatasetSpec:
    for spec in article_specs():
        if spec.name == name:
            return spec
    raise KeyError(f"Dataset '{name}' nao reconhecido. Disponiveis: "
                   f"{[s.name for s in article_specs()]}")


# --------------------------------------------------------------------------- #
# Helpers de cache HTTP                                                       #
# --------------------------------------------------------------------------- #


def _request_text(url: str, cache_path: Path) -> str:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        response = requests.get(url, verify=False, timeout=120)
        response.raise_for_status()
        cache_path.write_text(response.text, encoding="utf-8")
    return cache_path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Loaders individuais                                                         #
# --------------------------------------------------------------------------- #


def _load_adult(cache_dir: Path):
    train_text = _request_text(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        cache_dir / "adult.data",
    )
    test_text = _request_text(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
        cache_dir / "adult.test",
    )
    train = pd.read_csv(
        io.StringIO(train_text), header=None, names=ADULT_COLUMNS, skipinitialspace=True,
    )
    test = pd.read_csv(
        io.StringIO(test_text), header=None, names=ADULT_COLUMNS,
        skipinitialspace=True, comment="|",
    )
    frame = pd.concat([train, test], ignore_index=True)
    frame["target"] = (
        frame["target"].astype(str).str.strip().str.rstrip(".").map({">50K": 1, "<=50K": 0})
    )
    spec = get_spec("adult")
    for column in ADULT_COLUMNS[:-1]:
        if column not in spec.categorical_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.drop(columns=["target"]), frame["target"].astype(int)


def _load_german(cache_dir: Path):
    text = _request_text(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
        cache_dir / "german.data",
    )
    frame = pd.read_csv(io.StringIO(text), sep=r"\s+", header=None, names=GERMAN_COLUMNS)
    spec = get_spec("german")
    cats = set(spec.categorical_columns)
    for column in GERMAN_COLUMNS[:-1]:
        if column not in cats:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    target = (frame["target"].astype(int) == 2).astype(int)
    return frame.drop(columns=["target"]), target


def _load_magic(cache_dir: Path):
    text = _request_text(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data",
        cache_dir / "magic04.data",
    )
    frame = pd.read_csv(io.StringIO(text), header=None, names=MAGIC_COLUMNS)
    for column in MAGIC_COLUMNS[:-1]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    target = frame["target"].astype(str).str.strip().map({"g": 0, "h": 1}).astype(int)
    return frame.drop(columns=["target"]), target


def _load_spambase(cache_dir: Path):
    text = _request_text(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",
        cache_dir / "spambase.data",
    )
    frame = pd.read_csv(io.StringIO(text), header=None, names=SPAMBASE_COLUMNS)
    for column in SPAMBASE_COLUMNS[:-1]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.drop(columns=["target"]), frame["target"].astype(int)


def _load_chess(cache_dir: Path):
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


_LOADERS = {
    "adult": _load_adult,
    "german": _load_german,
    "magic": _load_magic,
    "spambase": _load_spambase,
    "chess": _load_chess,
}


def load_article_dataset(
    name_or_spec: str | DatasetSpec,
    cache_dir: str | Path = "artifacts/article_datasets",
) -> tuple[pd.DataFrame, pd.Series]:
    """Carrega uma das 5 bases UCI do paper ICAI 2012.

    Devolve `(X, y)` com `y` binario {0, 1}. As colunas categoricas
    chegam ja como `str` (consistente com `categorical_features` da V2).
    """
    spec = get_spec(name_or_spec) if isinstance(name_or_spec, str) else name_or_spec
    base_dir = Path(cache_dir) / spec.name
    X, y = _LOADERS[spec.name](base_dir)
    for column in spec.categorical_columns:
        if column in X.columns:
            X[column] = X[column].astype(str)
    return X.reset_index(drop=True), y.reset_index(drop=True)
