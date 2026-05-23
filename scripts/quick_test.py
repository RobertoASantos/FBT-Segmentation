#!/usr/bin/env python
"""Quick test: fbtseg em 30 segundos.

Treina fbtseg + LogisticRegression no dataset Chess (pequeno)
e compara error_rate.

    python scripts/quick_test.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from fbtseg import article_uci_preset, load_article_dataset
from fbtseg.datasets import get_spec

# Carrega dataset Chess (3.196 linhas, 36 categorias)
spec = get_spec("chess")
X, y = load_article_dataset(spec)
print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")

# Split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Train: {len(Xtr)}, Test: {len(Xte)}")

# --- fbtseg ---
print("\n[fbtseg]", end=" ", flush=True)
model_fbtseg = article_uci_preset(categorical_features=spec.categorical_columns)
model_fbtseg.fit(Xtr, ytr)
proba_fbtseg = model_fbtseg.predict_proba(Xte)[:, 1]
pred_fbtseg = (proba_fbtseg >= 0.5).astype(int)
error_fbtseg = (pred_fbtseg != yte).mean()
print(f"Error Rate: {error_fbtseg*100:.2f}%")
print(model_fbtseg.plot_model_tree())

# --- LogisticRegression (baseline) ---
print("\n[LogisticRegression]", end=" ", flush=True)
X_enc = Xtr.copy()
Xte_enc = Xte.copy()
encoders = {}
for col in spec.categorical_columns:
    le = LabelEncoder()
    X_enc[col] = le.fit_transform(X_enc[col].astype(str))
    Xte_enc[col] = le.transform(Xte_enc[col].astype(str))
    encoders[col] = le

model_lr = LogisticRegression(penalty=None, max_iter=1000, random_state=42)
model_lr.fit(X_enc, ytr)
proba_lr = model_lr.predict_proba(Xte_enc)[:, 1]
pred_lr = (proba_lr >= 0.5).astype(int)
error_lr = (pred_lr != yte).mean()
print(f"Error Rate: {error_lr*100:.2f}%")

# Comparação
print("\n" + "="*50)
print(f"fbtseg:                {error_fbtseg*100:.2f}%")
print(f"LogisticRegression:    {error_lr*100:.2f}%")
print(f"Ganho:                 {(error_lr - error_fbtseg)*100:+.2f}%")
print("="*50)
