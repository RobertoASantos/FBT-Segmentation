"""CLI mínimo do fbtseg.

Implementa dois subcomandos:

- ``fit``: treina um modelo a partir de uma das bases UCI do paper
  (`--dataset {adult,german,magic,spambase,chess}`) ou de um CSV
  qualquer (`--csv data.csv --target y`). Salva `model.pkl` e
  `results.json` se `--output-dir` for fornecido.

- ``predict``: aplica um `model.pkl` previamente salvo sobre um CSV
  novo, escrevendo as probabilidades em CSV de saída.

Uso típico:

    python -m fbtseg fit --dataset adult --preset article_uci
    python -m fbtseg fit --csv data.csv --target y --preset thesis
    python -m fbtseg fit --dataset magic --preset article_uci --output-dir runs/magic
    python -m fbtseg predict --model runs/magic/model.pkl --csv novos.csv --output preds.csv

Depois de `pip install -e .`, o entry point `fbtseg` também fica
disponível:

    fbtseg fit --dataset chess --preset article_uci

Referências (`docs/references.md`):
- SANTOS, 2010 — Tese, Cap. 4.
- SANTOS & BARROS, 2012 (ICAI) — `article_uci_preset`.
- SANTOS & BARROS, 2012 (ICTAI) — `article_synthetic_preset`.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

from .datasets import article_specs, load_article_dataset
from .estimator import (
    RiskSegV2,
    article_synthetic_preset,
    article_uci_preset,
    thesis_preset,
)

PRESETS = {
    "thesis": thesis_preset,
    "article_uci": article_uci_preset,
    "article_synthetic": article_synthetic_preset,
}


def _load_article_dataset(name: str):
    spec = next((s for s in article_specs() if s.name == name), None)
    if spec is None:
        raise SystemExit(f"Dataset '{name}' nao reconhecido. Disponiveis: "
                         f"{[s.name for s in article_specs()]}")
    X, y = load_article_dataset(spec)
    return X, y, spec.categorical_columns


def _load_csv(path: str, target: str, categorical_columns: tuple):
    df = pd.read_csv(path)
    if target not in df.columns:
        raise SystemExit(f"Coluna alvo '{target}' nao existe no CSV.")
    y = df[target]
    X = df.drop(columns=[target])
    return X, y, tuple(categorical_columns or ())


def cmd_fit(args):
    if args.dataset:
        X, y, cat_cols = _load_article_dataset(args.dataset)
    elif args.csv:
        X, y, cat_cols = _load_csv(args.csv, args.target, args.categorical or ())
    else:
        raise SystemExit("forneca --dataset ou --csv.")

    print(f"[fit] linhas={len(X)} cols={X.shape[1]} categoricas={len(cat_cols)} preset={args.preset}")

    if args.test_size > 0:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )
    else:
        Xtr, ytr = X, y
        Xte, yte = None, None

    factory = PRESETS[args.preset]
    overrides = {}
    if args.max_depth is not None:
        overrides["max_depth"] = args.max_depth
    if args.top_k is not None:
        overrides["top_k_variables"] = args.top_k
    if args.metric is not None:
        overrides["metric"] = args.metric

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = factory(categorical_features=cat_cols, **overrides)
        t0 = time.perf_counter()
        model.fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0

    print(f"[fit] treinado em {fit_s:.2f}s")
    print(model.plot_model_tree())

    summary = model.get_summary()
    print("\n[summary]")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    results = {
        "summary": summary,
        "fit_seconds": fit_s,
        "preset": args.preset,
        "overrides": overrides,
        "n_train": int(len(ytr)),
    }

    if Xte is not None:
        metrics = model.evaluate(Xte, yte)
        results["test_metrics"] = metrics
        print("\n[test metrics]")
        for k, v in metrics.items():
            print(f"  {k:12s} {v:.4f}")

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
        with (out / "model.pkl").open("wb") as f:
            pickle.dump(model, f)
        print(f"\n[saved] {out}/results.json e model.pkl")


def cmd_predict(args):
    with open(args.model, "rb") as f:
        model: RiskSegV2 = pickle.load(f)
    X = pd.read_csv(args.csv)
    proba = model.predict_proba(X)[:, 1]
    out = pd.DataFrame({"proba_1": proba})
    out["pred"] = (proba >= model.classification_threshold).astype(int)
    if args.output:
        out.to_csv(args.output, index=False)
        print(f"[predict] {args.output} ({len(out)} linhas)")
    else:
        print(out.head())


def main():
    parser = argparse.ArgumentParser(prog="fbtseg")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fit = sub.add_parser("fit", help="treina um modelo")
    src = p_fit.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", choices=["adult", "german", "magic", "spambase", "chess"])
    src.add_argument("--csv", type=str, help="caminho de um CSV com a coluna alvo")
    p_fit.add_argument("--target", default="target")
    p_fit.add_argument("--categorical", nargs="*", default=None)
    p_fit.add_argument(
        "--preset", choices=list(PRESETS.keys()), default="article_uci"
    )
    p_fit.add_argument("--max-depth", type=int, default=None)
    p_fit.add_argument("--top-k", type=int, default=None)
    p_fit.add_argument(
        "--metric",
        choices=["error", "auc", "ks", "lift", "precision", "odds_ratio"],
        default=None,
    )
    p_fit.add_argument("--test-size", type=float, default=0.25)
    p_fit.add_argument("--seed", type=int, default=42)
    p_fit.add_argument("--output-dir", type=str, default=None)
    p_fit.set_defaults(func=cmd_fit)

    p_pred = sub.add_parser("predict", help="faz predicao com modelo salvo")
    p_pred.add_argument("--model", required=True, help="caminho do model.pkl")
    p_pred.add_argument("--csv", required=True, help="CSV de entrada")
    p_pred.add_argument("--output", default=None, help="CSV de saida")
    p_pred.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
