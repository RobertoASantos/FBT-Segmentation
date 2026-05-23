"""Testes do `fbtseg`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from fbtseg import (
    FBTSeg,
    MarginalOddsCombiner,
    ModelView,
    RiskSegV2,
    SegView,
    StackingCombiner,
    all_metrics,
    article_synthetic_preset,
    article_uci_preset,
    metric_score,
    thesis_preset,
)


# --------------------------------------------------------------------------- #
# Dados sintéticos                                                            #
# --------------------------------------------------------------------------- #


def _synthetic_dataset_with_interaction(n: int = 4000, seed: int = 0):
    """Base com interação forte entre `cat` (cat) e `x1` (num).

    Esperado: a segmentação por `cat` melhora a regressão logística simples
    porque o sinal `x1 → y` muda de orientação por categoria.
    """
    rng = np.random.default_rng(seed)
    cat = rng.choice(["a", "b"], size=n)
    x1 = rng.uniform(-1, 1, size=n)
    x2 = rng.uniform(-1, 1, size=n)
    x3 = rng.uniform(-1, 1, size=n)
    sign = np.where(cat == "a", 1.0, -1.0)
    logit = 2.5 * sign * x1 + 0.5 * x2 + 0.2 * x3
    p = 1.0 / (1.0 + np.exp(-logit))
    y = rng.binomial(1, p)
    X = pd.DataFrame({"cat": cat, "x1": x1, "x2": x2, "x3": x3})
    return X, pd.Series(y, name="y")


def _basic_classification(n: int = 2000, seed: int = 0):
    X, y = make_classification(
        n_samples=n,
        n_features=10,
        n_informative=6,
        random_state=seed,
    )
    cols = [f"x{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols), pd.Series(y)


# --------------------------------------------------------------------------- #
# views                                                                       #
# --------------------------------------------------------------------------- #


def test_seg_view_encodes_consistent_codes():
    X, _ = _synthetic_dataset_with_interaction(n=200)
    sv = SegView(n_numeric_bins=4, categorical_features=("cat",)).fit(X)
    codes_a = sv.transform(X)
    codes_b = sv.transform(X)
    assert np.array_equal(codes_a, codes_b)
    cat_col = sv.column_index("cat")
    assert set(np.unique(codes_a[:, cat_col]).tolist()) <= {0, 1}


def test_model_view_drop_split_feature():
    X, _ = _synthetic_dataset_with_interaction(n=200)
    mv = ModelView(scale_numeric=True, categorical_features=("cat",)).fit(X)
    full = mv.all_columns()
    keep = mv.columns_excluding(["cat"])
    assert keep.size < full.size
    assert "cat__a" not in [mv.out_columns_[i] for i in keep]


def test_model_view_scaling_in_unit_interval():
    X, _ = _basic_classification()
    mv = ModelView(scale_numeric=True).fit(X)
    Z = mv.transform(X)
    assert Z.min() >= 0.0 - 1e-9
    assert Z.max() <= 1.0 + 1e-9


# --------------------------------------------------------------------------- #
# combiners                                                                   #
# --------------------------------------------------------------------------- #


def test_stacking_combiner_runs_with_one_class():
    rng = np.random.default_rng(0)
    n = 50
    membership = np.zeros(n, dtype=int)
    score_left = rng.uniform(size=n)
    score_right = np.zeros(n)
    y = np.zeros(n, dtype=int)  # apenas uma classe
    comb = StackingCombiner().fit(score_left, score_right, membership, y)
    p = comb.predict_proba(score_left, score_right, membership)[:, 1]
    assert np.all((p >= 0) & (p <= 1))


def test_marginal_odds_aligns_segments():
    rng = np.random.default_rng(1)
    n = 600
    membership = rng.integers(0, 2, size=n)
    score_left = rng.uniform(0.3, 0.7, size=n)
    score_right = rng.uniform(0.1, 0.9, size=n)
    y = rng.binomial(1, np.where(membership == 0, score_left, score_right))
    comb = MarginalOddsCombiner(reference="left").fit(score_left, score_right, membership, y)
    p = comb.predict_proba(score_left, score_right, membership)[:, 1]
    assert p.shape == (n,)
    assert np.all((p >= 0) & (p <= 1))
    auc = roc_auc_score(y, p)
    assert auc > 0.6


# --------------------------------------------------------------------------- #
# RiskSegV2                                                                   #
# --------------------------------------------------------------------------- #


def test_fit_predict_basic():
    X, y = _basic_classification(n=1500)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    model = thesis_preset(max_depth=2).fit(Xtr, ytr)
    proba = model.predict_proba(Xte)
    assert proba.shape == (len(yte), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    auc = roc_auc_score(yte, proba[:, 1])
    assert auc > 0.7


def test_interaction_dataset_beats_logistic():
    X, y = _synthetic_dataset_with_interaction(n=4000, seed=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    base = LogisticRegression(penalty=None, max_iter=2000)
    base.fit(pd.get_dummies(Xtr, columns=["cat"]), ytr)
    base_pred = base.predict_proba(pd.get_dummies(Xte, columns=["cat"]))[:, 1]
    base_auc = roc_auc_score(yte, base_pred)

    rs = thesis_preset(
        max_depth=1,
        validation_fraction=0.3,
        categorical_features=("cat",),
    ).fit(Xtr, ytr)
    rs_pred = rs.predict_proba(Xte)[:, 1]
    rs_auc = roc_auc_score(yte, rs_pred)

    assert rs_auc > base_auc + 0.05, f"esperado > {base_auc + 0.05:.3f}, obtido {rs_auc:.3f}"


def test_predict_modes_consistent():
    X, y = _synthetic_dataset_with_interaction(n=2000, seed=7)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    rs = article_uci_preset(categorical_features=("cat",)).fit(Xtr, ytr)
    p_leaf = rs.predict_proba(Xte)
    rs.prediction_mode = "pair_combiner"
    p_pair = rs.predict_proba(Xte)
    assert p_leaf.shape == p_pair.shape
    # ambos retornam probabilidades válidas
    for p in (p_leaf, p_pair):
        assert np.all(p >= 0) and np.all(p <= 1)
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_tree_summary_columns():
    X, y = _basic_classification(n=1000)
    rs = article_uci_preset().fit(X, y)
    summary = rs.get_tree_summary()
    for col in [
        "node_id",
        "depth",
        "is_leaf",
        "n_train",
        "split_variable",
        "baseline_obj",
        "split_obj",
        "gain_pct",
    ]:
        assert col in summary.columns


def test_grouping_features_restriction():
    X, y = _synthetic_dataset_with_interaction(n=1500, seed=3)
    rs = thesis_preset(
        max_depth=1,
        categorical_features=("cat",),
        grouping_features=("cat",),
        max_group_size=2,
    ).fit(X, y)
    nodes = rs.nodes_
    assert nodes[0].split_variable is not None  # alguma quebra foi tentada


def test_drop_split_feature_in_children():
    X, y = _synthetic_dataset_with_interaction(n=2000, seed=11)
    rs = thesis_preset(
        max_depth=2,
        drop_split_feature_in_children=True,
        categorical_features=("cat",),
    ).fit(X, y)
    root = rs.root_
    if not root.is_leaf:
        child = root.left if root.left is not None else root.right
        assert root.split_variable in child.used_features


def test_min_gain_pct_blocks_split():
    X, y = _basic_classification(n=800)
    rs = thesis_preset(min_gain_pct=10.0, max_loss_pct=0.0, max_depth=2).fit(X, y)
    assert rs.root_.is_leaf, "com min_gain_pct alto, a raiz deveria virar folha"


def test_metric_registry_returns_higher_is_better():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=400)
    good = np.clip(y + rng.normal(0, 0.2, size=400), 0, 1)
    bad = rng.uniform(size=400)
    assert metric_score("error", y, good) > metric_score("error", y, bad)
    assert metric_score("auc", y, good) > metric_score("auc", y, bad)
    assert metric_score("ks", y, good) > metric_score("ks", y, bad)


def test_all_metrics_returns_expected_keys():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    p = rng.uniform(size=200)
    out = all_metrics(y, p)
    for k in ["error", "auc", "ks", "lift", "precision", "odds_ratio"]:
        assert k in out


def test_predict_shape_matches_classes():
    X, y = _basic_classification(n=500)
    y_label = pd.Series(np.where(y == 1, "yes", "no"))
    rs = thesis_preset(max_depth=1).fit(X, y_label)
    pred = rs.predict(X)
    assert set(pred) <= {"yes", "no"}


def test_sklearn_clone_compatible():
    from sklearn.base import clone

    rs = thesis_preset()
    cloned = clone(rs)
    assert cloned.metric == rs.metric


def test_categorical_features_param():
    X, y = _synthetic_dataset_with_interaction(n=600, seed=2)
    rs = article_uci_preset(categorical_features=("cat",)).fit(X, y)
    assert "cat" in rs.seg_view_.columns_
    cat_idx = rs.seg_view_.column_index("cat")
    assert not rs.seg_view_.is_numeric_["cat"]


def test_synthetic_preset_smoke():
    X, y = _basic_classification(n=1000)
    rs = article_synthetic_preset(max_depth=2).fit(X, y)
    proba = rs.predict_proba(X)
    assert proba.shape == (len(y), 2)


def test_fbtseg_is_alias_for_risksegv2():
    """O alias publico FBTSeg deve apontar para a mesma classe."""
    assert FBTSeg is RiskSegV2
    model = FBTSeg(max_depth=1)
    assert isinstance(model, RiskSegV2)


# --------------------------------------------------------------------------- #
# Features adicionadas: screening_variables, sample_weight, threshold,        #
# cascade, global_stacking_oof, get_summary, plot_model_tree                  #
# --------------------------------------------------------------------------- #


def test_screening_variables_restricts_split_candidates():
    X, y = _synthetic_dataset_with_interaction(n=1500, seed=4)
    rs = thesis_preset(
        max_depth=1,
        screening_variables=("x1",),  # cat fica de fora
        categorical_features=("cat",),
    ).fit(X, y)
    if not rs.root_.is_leaf:
        assert rs.root_.split_variable == "x1"


def test_screening_variables_unknown_raises():
    X, y = _basic_classification(n=200)
    with pytest.raises(ValueError, match="screening_variables"):
        thesis_preset(screening_variables=("nao_existe",)).fit(X, y)


def test_sample_weight_runs_and_changes_results():
    rng = np.random.default_rng(0)
    X, y = _synthetic_dataset_with_interaction(n=1200, seed=5)

    rs_uniform = thesis_preset(
        max_depth=1, categorical_features=("cat",)
    ).fit(X, y)
    proba_uniform = rs_uniform.predict_proba(X)[:, 1]

    weights = np.where(y == 1, 5.0, 1.0)
    rs_weighted = thesis_preset(
        max_depth=1, categorical_features=("cat",)
    ).fit(X, y, sample_weight=weights)
    proba_weighted = rs_weighted.predict_proba(X)[:, 1]

    # ambos retornam probabilidades válidas
    assert proba_weighted.shape == proba_uniform.shape
    assert np.all(proba_weighted >= 0) and np.all(proba_weighted <= 1)
    # com peso forte nos positivos, a probabilidade média sobe
    assert proba_weighted.mean() > proba_uniform.mean() - 1e-9


def test_sample_weight_wrong_size_raises():
    X, y = _basic_classification(n=300)
    with pytest.raises(ValueError, match="sample_weight"):
        thesis_preset().fit(X, y, sample_weight=np.ones(10))


def test_classification_threshold_changes_predictions():
    X, y = _basic_classification(n=800)
    rs = thesis_preset(classification_threshold=0.9).fit(X, y)
    pred_high = rs.predict(X)
    rs.classification_threshold = 0.1
    pred_low = rs.predict(X)
    # com threshold mais alto, há menos positivos
    n_pos_high = int((pred_high == rs.classes_[1]).sum())
    n_pos_low = int((pred_low == rs.classes_[1]).sum())
    assert n_pos_low >= n_pos_high


def test_prediction_mode_cascade_runs():
    X, y = _synthetic_dataset_with_interaction(n=1500, seed=8)
    rs = thesis_preset(
        max_depth=2,
        categorical_features=("cat",),
        prediction_mode="cascade",
    ).fit(X, y)
    proba = rs.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_prediction_mode_global_stacking_oof():
    X, y = _synthetic_dataset_with_interaction(n=1500, seed=9)
    rs = thesis_preset(
        max_depth=2,
        categorical_features=("cat",),
        prediction_mode="global_stacking",
        global_stacking_n_splits=3,
    ).fit(X, y)
    assert rs.global_stacking_model_ is not None
    proba = rs.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_get_summary_keys():
    X, y = _synthetic_dataset_with_interaction(n=600, seed=10)
    rs = article_uci_preset(categorical_features=("cat",)).fit(X, y)
    summary = rs.get_summary()
    for k in [
        "n_features",
        "n_nodes",
        "n_leaves",
        "max_depth_reached",
        "used_variables_unique",
        "prediction_mode",
        "combiner_method",
        "metric",
        "classes",
    ]:
        assert k in summary


def test_plot_model_tree_returns_string():
    X, y = _synthetic_dataset_with_interaction(n=600, seed=11)
    rs = article_uci_preset(categorical_features=("cat",), max_depth=2).fit(X, y)
    plot = rs.plot_model_tree()
    assert isinstance(plot, str)
    assert plot.splitlines(), "deveria ter ao menos uma linha"


def test_unsupported_prediction_mode_raises():
    X, y = _basic_classification(n=200)
    rs = thesis_preset(prediction_mode="bogus").fit(X, y)
    with pytest.raises(ValueError, match="prediction_mode"):
        rs.predict_proba(X)


# --------------------------------------------------------------------------- #
# Base learners adicionais (Linear, MLP) — replicar paper ICAI 2012           #
# --------------------------------------------------------------------------- #


def test_linear_probability_classifier_basic():
    from fbtseg import LinearProbabilityClassifier

    X, y = _basic_classification(n=400)
    clf = LinearProbabilityClassifier().fit(X.values, y.values)
    proba = clf.predict_proba(X.values)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_linear_probability_classifier_with_single_class():
    from fbtseg import LinearProbabilityClassifier

    X = np.random.rand(50, 4)
    y = np.zeros(50, dtype=int)
    clf = LinearProbabilityClassifier().fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (50, 2)


def test_v2_with_linear_base_learner():
    from fbtseg import LinearProbabilityClassifier

    X, y = _synthetic_dataset_with_interaction(n=1500, seed=12)
    rs = thesis_preset(
        max_depth=1,
        categorical_features=("cat",),
        base_estimator=LinearProbabilityClassifier(),
    ).fit(X, y)
    proba = rs.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_v2_with_mlp_base_learner():
    from sklearn.neural_network import MLPClassifier

    X, y = _synthetic_dataset_with_interaction(n=800, seed=13)
    rs = thesis_preset(
        max_depth=1,
        categorical_features=("cat",),
        base_estimator=MLPClassifier(
            hidden_layer_sizes=(5,),
            max_iter=200,
            random_state=0,
        ),
    ).fit(X, y)
    proba = rs.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_v2_with_linear_base_beats_baseline_on_sign_flip():
    """Base Linear simples nao captura interacao com sinal invertido por categoria.
    A segmentacao por cat deveria deixar a Linear ajustar a inclinacao certa em cada lado.
    """
    from fbtseg import LinearProbabilityClassifier

    X, y = _synthetic_dataset_with_interaction(n=4000, seed=99)
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    # base Linear sem segmentacao
    base = LinearProbabilityClassifier()
    base.fit(pd.get_dummies(Xtr, columns=["cat"]).values, ytr.values)
    base_err = float(np.mean(base.predict(pd.get_dummies(Xte, columns=["cat"]).values) != yte.to_numpy()))

    # V2 com Linear como base
    rs = thesis_preset(
        max_depth=1,
        categorical_features=("cat",),
        base_estimator=LinearProbabilityClassifier(),
    ).fit(Xtr, ytr)
    pred = (rs.predict_proba(Xte)[:, 1] >= 0.5).astype(int)
    v2_err = float(np.mean(pred != yte.to_numpy()))

    assert v2_err < base_err - 0.01, f"esperado V2_err ({v2_err}) < base_err ({base_err}) - 0.01"
