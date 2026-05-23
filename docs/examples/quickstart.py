"""
RiskSeg V2 - Quickstart

Este script funciona como um "notebook" linear: voce pode executa-lo
todo de uma vez ou cole-lo trecho a trecho num REPL/Jupyter. Mostra
o ciclo basico: carregar a base, treinar, predizer, inspecionar a
arvore e comparar com a regressao logistica.

Execucao direta:
    PYTHONPATH=. python docs/examples/v2_quickstart.py
"""

# %% [markdown]
# # RiskSegV2 — Quickstart
#
# Aqui usamos a base **Adult** (UCI), com 48k linhas e mistura de
# variaveis categoricas e numericas. O alvo e dicotomico (`>50K` vs
# `<=50K`), exatamente o cenario para o qual o RiskSeg foi pensado.

# %%
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from fbtseg import (
    LinearProbabilityClassifier,
    article_uci_preset,
    get_spec,
    load_article_dataset,
    thesis_preset,
)

warnings.filterwarnings("ignore")


def make_preprocessor(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    transformers = []
    if num_cols:
        transformers.append(
            ("num",
             Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", MinMaxScaler())]),
             num_cols))
    if cat_cols:
        transformers.append(
            ("cat",
             Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                       ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
             cat_cols))
    return ColumnTransformer(transformers, sparse_threshold=0.0)


# %% [markdown]
# ## 1. Carregar Adult e separar treino/teste

# %%
spec = get_spec("adult")
X, y = load_article_dataset(spec)
print(f"Adult: {len(X)} linhas, {X.shape[1]} variaveis")
print(f"Categoricas: {spec.categorical_columns}")
print(f"Base rate (positivos): {y.mean():.4f}")

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# %% [markdown]
# ## 2. Baseline Logistic Regression
#
# Usamos `penalty=None` (sem regularizacao) para aproximar o
# `PROC LOGISTIC` do SAS usado no paper.

# %%
pre = make_preprocessor(Xtr)
Ztr = pre.fit_transform(Xtr, ytr)
Zte = pre.transform(Xte)

lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000, random_state=42)
lr.fit(Ztr, ytr)
p_lr = lr.predict_proba(Zte)[:, 1]
print(f"LR  error={1 - accuracy_score(yte, (p_lr >= 0.5).astype(int)):.4f}")
print(f"LR  AUC={roc_auc_score(yte, p_lr):.4f}")

# %% [markdown]
# ## 3. RiskSegV2 com preset article_uci
#
# Esse preset replica o protocolo do paper ICAI 2012 (max_depth=2,
# min_leaf=5%, val=35%, top_k=1).

# %%
model = article_uci_preset(categorical_features=spec.categorical_columns)
model.fit(Xtr, ytr)
p_v2 = model.predict_proba(Xte)[:, 1]
print(f"V2  error={1 - accuracy_score(yte, (p_v2 >= 0.5).astype(int)):.4f}")
print(f"V2  AUC={roc_auc_score(yte, p_v2):.4f}")

# %% [markdown]
# ## 4. Inspecionar a arvore montada

# %%
print(model.plot_model_tree())

# %% [markdown]
# Resumo agregado (terminologia da tese):

# %%
import pprint
pprint.pprint(model.get_summary())

# %% [markdown]
# Tabela por no:

# %%
tree_df = model.get_tree_summary()
print(tree_df.to_string(index=False))

# %% [markdown]
# ## 5. Bloco completo de metricas

# %%
metrics = model.evaluate(Xte, yte)
for k, v in metrics.items():
    print(f"  {k:12s} {v:.4f}")

# %% [markdown]
# ## 6. Trocar base learner: Linear Probability Classifier
#
# O paper compara FBTSeg cruzado com Linear/Logistic/MLP. Vamos rodar
# com Linear:

# %%
# (LinearProbabilityClassifier ja importado no topo)

model_linear = article_uci_preset(
    categorical_features=spec.categorical_columns,
    base_estimator=LinearProbabilityClassifier(),
)
model_linear.fit(Xtr, ytr)
p_lin = model_linear.predict_proba(Xte)[:, 1]
print(f"fbtseg+Linear  error={1 - accuracy_score(yte, (p_lin >= 0.5).astype(int)):.4f}")
print(f"fbtseg+Linear  AUC={roc_auc_score(yte, p_lin):.4f}")

# %% [markdown]
# ## 7. Mudar a metrica D para AUC e re-treinar

# %%
model_auc = article_uci_preset(
    categorical_features=spec.categorical_columns,
    metric="auc",
    max_loss_pct=0.02,
)
model_auc.fit(Xtr, ytr)
p_auc = model_auc.predict_proba(Xte)[:, 1]
print(f"V2(auc)  error={1 - accuracy_score(yte, (p_auc >= 0.5).astype(int)):.4f}")
print(f"V2(auc)  AUC={roc_auc_score(yte, p_auc):.4f}")

# %% [markdown]
# ## 8. Limitar variaveis candidatas a split (parametro P da tese)

# %%
model_p = article_uci_preset(
    categorical_features=spec.categorical_columns,
    screening_variables=("education", "occupation", "marital-status", "age", "capital-gain"),
)
model_p.fit(Xtr, ytr)
print("Variaveis usadas:", model_p.get_summary()["used_variables_unique"])
