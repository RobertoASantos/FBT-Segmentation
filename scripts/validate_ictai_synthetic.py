"""Validacao do RiskSegV2 contra os datasets sinteticos do paper ICTAI 2012.

Recria os quatro datasets descritos na Secao III do paper, com a
mesma divisao 50/25/25 treino/validacao/teste, faz `n_replications`
replicas (default = 30, como o paper) e mede o erro medio.

Esperado para o Dataset 2 + Logistic Regression com 5k registros:
- Base Logistic Simple ~ 11.11%
- FBTSeg ~ 7.03%

Valor de referencia: tabela do paper (Section IV.B).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from fbtseg import article_synthetic_preset


# --------------------------------------------------------------------------- #
# Geracao dos datasets (equacoes IV.A do paper)                               #
# --------------------------------------------------------------------------- #


def _to_target(h: np.ndarray) -> np.ndarray:
    """y = 1 se 1 / (1 + exp(-h)) > 0.5 senao 0 — Eq. 1 do paper."""
    p = 1.0 / (1.0 + np.exp(-h))
    return (p > 0.5).astype(int)


# Constantes mu_i fixadas por dataset (sorteadas uma vez via seed=1234).
# O paper usa "an arbitrarily chosen constant" — fixamos os valores aqui
# para que cada replicacao varie apenas em x e em epsilon.
_FIXED_RNG = np.random.default_rng(1234)
_BETA_D1 = _FIXED_RNG.normal(0, 1, size=10)
_GAMMA_D1 = _FIXED_RNG.normal(0, 1, size=5)
_ALPHA_D1 = _FIXED_RNG.normal(0, 1, size=3)
_PHI_D3 = _FIXED_RNG.normal(0, 1, size=4)
_DELTA_D4 = _FIXED_RNG.normal(0, 1, size=4)


def generate_dataset_1(n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.Series]:
    """Strong interaction (dataset discreto)."""
    x = np.stack([rng.choice(np.arange(1, 6), size=n) for _ in range(10)], axis=1).astype(float)
    beta = _BETA_D1
    gamma = _GAMMA_D1
    alpha = _ALPHA_D1
    eps = rng.normal(0, 1, size=n)
    h = (
        beta[0] * x[:, 0]
        + beta[1] * x[:, 1]
        + beta[2] * x[:, 2]
        + beta[3] * x[:, 3]
        + beta[4] * x[:, 4]
        + beta[5] * x[:, 5]
        + beta[6] * x[:, 6]
        + beta[7] * x[:, 7]
        + beta[8] * x[:, 8]
        + beta[9] * x[:, 9]
        + gamma[0] * x[:, 0] * x[:, 5]
        + gamma[1] * x[:, 0] * x[:, 7]
        + gamma[2] * x[:, 1] * x[:, 7]
        + gamma[3] * x[:, 2] * x[:, 8]
        + gamma[4] * x[:, 4] * x[:, 6]
        + alpha[0] * x[:, 3] * x[:, 4] * x[:, 5]
        + alpha[1] * x[:, 7] * x[:, 8] * x[:, 9]
        + alpha[2] * x[:, 1] * x[:, 3] * x[:, 7]
        + eps
    )
    h = h - np.median(h)
    cols = [f"x{i}" for i in range(10)]
    X = pd.DataFrame(x, columns=cols)
    y = pd.Series(_to_target(h), name="y")
    return X, y


def generate_dataset_2(n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.Series]:
    """Non-linear structure — h = 10 sin(pi x1 x2) + 20 (x3 - 0.5)^2 + 10 x4 + 5 x5 + eps."""
    x = rng.uniform(0, 1, size=(n, 10))
    eps = rng.normal(0, 1, size=n)
    h = (
        10.0 * np.sin(np.pi * x[:, 0] * x[:, 1])
        + 20.0 * (x[:, 2] - 0.5) ** 2
        + 10.0 * x[:, 3]
        + 5.0 * x[:, 4]
        + eps
    )
    # centraliza h para metas balanceadas
    h = h - np.median(h)
    cols = [f"x{i}" for i in range(10)]
    X = pd.DataFrame(x, columns=cols)
    y = pd.Series(_to_target(h), name="y")
    return X, y


def generate_dataset_3(n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.Series]:
    """Additive linear — h = sum phi_i x_i + eps."""
    x = rng.uniform(0, 1, size=(n, 10))
    phi = _PHI_D3
    eps = rng.normal(0, 1, size=n)
    h = (
        phi[0] * x[:, 0]
        + phi[1] * x[:, 1]
        + phi[2] * x[:, 2]
        + phi[3] * x[:, 3]
        + eps
    )
    h = h - np.median(h)
    cols = [f"x{i}" for i in range(10)]
    X = pd.DataFrame(x, columns=cols)
    y = pd.Series(_to_target(h), name="y")
    return X, y


def generate_dataset_4(n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.Series]:
    """Quadratic — h = sum delta_i x_i^2 + eps."""
    x = rng.uniform(0, 1, size=(n, 10))
    delta = _DELTA_D4
    eps = rng.normal(0, 1, size=n)
    h = (
        delta[0] * x[:, 0] ** 2
        + delta[1] * x[:, 1] ** 2
        + delta[2] * x[:, 2] ** 2
        + delta[3] * x[:, 3] ** 2
        + eps
    )
    h = h - np.median(h)
    cols = [f"x{i}" for i in range(10)]
    X = pd.DataFrame(x, columns=cols)
    y = pd.Series(_to_target(h), name="y")
    return X, y


GENERATORS = {
    1: generate_dataset_1,
    2: generate_dataset_2,
    3: generate_dataset_3,
    4: generate_dataset_4,
}


# --------------------------------------------------------------------------- #
# Avaliacao                                                                   #
# --------------------------------------------------------------------------- #


def _split_50_25_25(n: int, rng: np.random.Generator):
    idx = rng.permutation(n)
    n_tr = n // 2
    n_va = n // 4
    tr = idx[:n_tr]
    va = idx[n_tr : n_tr + n_va]
    te = idx[n_tr + n_va :]
    return tr, va, te


def evaluate_baseline(X_tr, y_tr, X_te, y_te):
    scaler = MinMaxScaler().fit(X_tr.values)
    X_tr_s = scaler.transform(X_tr.values)
    X_te_s = scaler.transform(X_te.values)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000, random_state=42)
    lr.fit(X_tr_s, y_tr)
    p = lr.predict_proba(X_te_s)[:, 1]
    pred = (p >= 0.5).astype(int)
    return float(np.mean(pred != y_te.to_numpy()))


def evaluate_v2(X_tr, y_tr, X_te, y_te):
    rs = article_synthetic_preset()
    rs.fit(X_tr, y_tr)
    p = rs.predict_proba(X_te)[:, 1]
    pred = (p >= 0.5).astype(int)
    return float(np.mean(pred != y_te.to_numpy()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--sizes", nargs="*", type=int, default=[1000, 3000, 5000])
    parser.add_argument("--n-replications", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="artifacts/v2_ictai_synthetic")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    rng_root = np.random.default_rng(args.seed)
    for ds_id in args.datasets:
        gen = GENERATORS[ds_id]
        for n in args.sizes:
            for rep in range(args.n_replications):
                rng = np.random.default_rng(rng_root.integers(0, 2**32 - 1))
                X, y = gen(n, rng)
                tr, va, te = _split_50_25_25(n, rng)
                # juntamos train + val no fit, deixando o V2 fazer seu split interno
                X_tr_full = pd.concat([X.iloc[tr], X.iloc[va]], ignore_index=True)
                y_tr_full = pd.concat([y.iloc[tr], y.iloc[va]], ignore_index=True)
                X_te = X.iloc[te].reset_index(drop=True)
                y_te = y.iloc[te].reset_index(drop=True)

                t0 = time.perf_counter()
                err_base = evaluate_baseline(X_tr_full, y_tr_full, X_te, y_te)
                t_base = time.perf_counter() - t0

                t1 = time.perf_counter()
                err_v2 = evaluate_v2(X_tr_full, y_tr_full, X_te, y_te)
                t_v2 = time.perf_counter() - t1

                rows.append(
                    {
                        "dataset": f"D{ds_id}",
                        "n": n,
                        "rep": rep,
                        "err_LR": err_base,
                        "err_V2": err_v2,
                        "fit_s_LR": t_base,
                        "fit_s_V2": t_v2,
                    }
                )
                print(
                    f"[D{ds_id}|n={n}|rep={rep+1}/{args.n_replications}] "
                    f"LR={err_base:.4f}  V2={err_v2:.4f}  fit_V2={t_v2:.1f}s"
                )

    df = pd.DataFrame(rows)
    df.to_csv(out / "ictai_synthetic_results.csv", index=False)

    summary = (
        df.groupby(["dataset", "n"]).agg(
            err_LR_mean=("err_LR", "mean"),
            err_LR_std=("err_LR", "std"),
            err_V2_mean=("err_V2", "mean"),
            err_V2_std=("err_V2", "std"),
            delta_mean=("err_V2", lambda s: s.mean() - df.loc[s.index, "err_LR"].mean()),
        )
        .reset_index()
        .sort_values(["dataset", "n"])
    )
    summary.to_csv(out / "ictai_synthetic_summary.csv", index=False)
    print("\n=== Summary (mean +/- std) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    (out / "args.json").write_text(json.dumps(vars(args), indent=2))


if __name__ == "__main__":
    main()
