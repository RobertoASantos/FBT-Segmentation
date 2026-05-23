from riskseg import RiskSegOptimizer, RiskSegRaizClassifier

from benchmarks.article_suite import get_article_dataset_specs, get_article_model_builders


def test_paper_preset_exposes_article_aligned_defaults():
    model = RiskSegOptimizer.paper_preset()

    params = model.get_params()

    assert params["metric"] == "error"
    assert params["top_k_variables"] == 1
    assert params["validation_fraction"] == 0.35
    assert params["n_numeric_bins"] == 4
    assert params["group_binned_numeric"] is True
    assert params["scale_model_numeric"] is True
    assert params["prediction_mode"] == "global_stacking"


def test_article_suite_exposes_raiz_without_replacing_current_riskseg():
    builders = get_article_model_builders(random_state=99)

    assert "PaperRiskSegLogistic" in builders
    assert "PaperRiskSegRaizLogistic" in builders

    spec = get_article_dataset_specs()[2]
    model = builders["PaperRiskSegRaizLogistic"](None, spec)

    assert isinstance(model, RiskSegRaizClassifier)
    assert model.get_params()["random_state"] == 99
    assert model.get_params()["prediction_mode"] == "local_combiner"
    assert model.get_params()["top_k_variables"] == 3
    assert model.get_params()["logistic_C"] == 100.0


def test_article_suite_declares_the_five_datasets_from_the_paper():
    specs = get_article_dataset_specs()
    names = [spec.name for spec in specs]

    assert names == ["adult", "german", "magic", "spambase", "chess"]
