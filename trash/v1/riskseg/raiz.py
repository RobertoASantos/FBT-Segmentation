"""Implementacao raiz do RiskSeg/FBTSeg descrito na tese."""

from sklearn.linear_model import LogisticRegression

from .optimizer import RiskSegOptimizer


class RiskSegRaizClassifier(RiskSegOptimizer):
    """Estimador RiskSeg alinhado ao protocolo original da tese/artigo.

    Esta classe preserva o `RiskSegOptimizer` atual e fixa defaults mais
    literais: erro de classificacao, validacao interna, numericas em quatro
    faixas para segmentacao, modelos treinados nas folhas e sem usar o
    `global_stacking` como predicao final.
    """

    def __init__(
        self,
        base_estimator=None,
        screening_estimator=None,
        screening_variables=None,
        categorical_features=None,
        node_estimator=None,
        segment_estimator=None,
        local_combiner_estimator=None,
        global_combiner_estimator=None,
        screening_mode="factorial_target_interactions",
        combiner_method="stacking",
        prediction_mode="local_combiner",
        metric="error",
        top_rate=0.05,
        classification_threshold=0.5,
        max_depth=2,
        min_samples_leaf=0.05,
        min_gain=0.0,
        validation_fraction=0.35,
        validation_mode="random_holdout",
        random_state=42,
        stratify=True,
        auto_bin_numeric=True,
        n_numeric_bins=4,
        numeric_binning="quantile",
        group_binned_numeric=True,
        scale_model_numeric=True,
        model_numeric_scaling="minmax",
        factorial_max_interaction_features=8,
        factorial_feature_selector="variance",
        factorial_include_main_effects=True,
        factorial_drop_first=False,
        use_grouping=False,
        max_group_size=2,
        top_k_variables=3,
        use_top_k_variables=True,
        use_validation_to_accept_split=True,
        global_only_if_no_gain=True,
        global_stacking_C=1.0,
        logistic_C=100.0,
        logistic_max_iter=5000,
        verbose=0,
    ):
        self.logistic_C = logistic_C
        self.logistic_max_iter = logistic_max_iter
        super().__init__(
            base_estimator=base_estimator,
            screening_estimator=screening_estimator,
            screening_variables=screening_variables,
            categorical_features=categorical_features,
            node_estimator=node_estimator,
            segment_estimator=segment_estimator,
            local_combiner_estimator=local_combiner_estimator,
            global_combiner_estimator=global_combiner_estimator,
            screening_mode=screening_mode,
            combiner_method=combiner_method,
            prediction_mode=prediction_mode,
            metric=metric,
            top_rate=top_rate,
            classification_threshold=classification_threshold,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_gain=min_gain,
            validation_fraction=validation_fraction,
            validation_mode=validation_mode,
            random_state=random_state,
            stratify=stratify,
            auto_bin_numeric=auto_bin_numeric,
            n_numeric_bins=n_numeric_bins,
            numeric_binning=numeric_binning,
            group_binned_numeric=group_binned_numeric,
            scale_model_numeric=scale_model_numeric,
            model_numeric_scaling=model_numeric_scaling,
            factorial_max_interaction_features=factorial_max_interaction_features,
            factorial_feature_selector=factorial_feature_selector,
            factorial_include_main_effects=factorial_include_main_effects,
            factorial_drop_first=factorial_drop_first,
            use_grouping=use_grouping,
            max_group_size=max_group_size,
            top_k_variables=top_k_variables,
            use_top_k_variables=use_top_k_variables,
            use_validation_to_accept_split=use_validation_to_accept_split,
            global_only_if_no_gain=global_only_if_no_gain,
            global_stacking_C=global_stacking_C,
            verbose=verbose,
        )

    def _default_logistic_estimator(self):
        return LogisticRegression(
            solver="lbfgs",
            C=self.logistic_C,
            max_iter=self.logistic_max_iter,
            random_state=self.random_state,
        )

    def _build_local_combiner_estimator(self):
        if self.local_combiner_estimator is not None:
            return super()._build_local_combiner_estimator()
        return self._default_logistic_estimator()

    def _build_global_combiner_estimator(self):
        if self.global_combiner_estimator is not None:
            return super()._build_global_combiner_estimator()
        return self._default_logistic_estimator()

    def fit(self, X, y):
        """Ajusta o modelo raiz sem suporte publico a `sample_weight`.

        O protocolo original usa treino/validacao por amostragem, sem pesos por
        observacao. Manter a assinatura sem `sample_weight` evita prometer uma
        equivalencia que nao faz parte da tecnica descrita.
        """
        return super().fit(X, y, sample_weight=None)

    @classmethod
    def article_uci_params(cls, **overrides):
        """Parametros do experimento UCI do artigo ICAI 2012."""
        params = {
            "metric": "error",
            "max_depth": 2,
            "min_samples_leaf": 0.05,
            "validation_fraction": 0.35,
            "auto_bin_numeric": True,
            "n_numeric_bins": 4,
            "numeric_binning": "quantile",
            "group_binned_numeric": True,
            "scale_model_numeric": True,
            "model_numeric_scaling": "minmax",
            "use_grouping": False,
            "top_k_variables": 3,
            "use_top_k_variables": True,
            "factorial_max_interaction_features": 8,
            "prediction_mode": "local_combiner",
            "combiner_method": "stacking",
            "use_validation_to_accept_split": True,
            "global_only_if_no_gain": True,
            "logistic_C": 100.0,
            "logistic_max_iter": 5000,
            "verbose": 0,
        }
        params.update(overrides)
        return params

    @classmethod
    def article_uci_preset(cls, **overrides):
        """Cria o `riskseg_raiz` com parametros do artigo UCI."""
        return cls(**cls.article_uci_params(**overrides))


RiskSegRaiz = RiskSegRaizClassifier
