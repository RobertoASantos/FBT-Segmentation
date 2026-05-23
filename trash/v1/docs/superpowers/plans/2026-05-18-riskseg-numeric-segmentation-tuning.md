# RiskSeg Numeric Segmentation Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve RiskSeg performance on hard numeric-heavy datasets by making numeric segmentation more faithful to threshold-like splits and by applying benchmark presets that reflect those data regimes.

**Architecture:** Extend `RiskSegOptimizer` so binned numeric variables can generate ordered contiguous split groups without enabling arbitrary grouping for true categorical variables. Keep the benchmark suite responsible for choosing tuned RiskSeg presets per dataset profile so article-aligned runs and large-benchmark runs remain explicit and reproducible.

**Tech Stack:** Python, pandas, numpy, scikit-learn, pytest

---

### Task 1: Lock in the failing numeric-grouping behavior

**Files:**
- Modify: `D:\Nuvem\gdrive_pessoal\Faculdades\Impacta\Graduação\Outros\IC\RiskSeg\tests\test_riskseg_optimizer.py`
- Test: `D:\Nuvem\gdrive_pessoal\Faculdades\Impacta\Graduação\Outros\IC\RiskSeg\tests\test_riskseg_optimizer.py`

- [ ] **Step 1: Write the failing test**

```python
def test_binned_numeric_columns_can_generate_contiguous_groups():
    X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 16)})
    y = pd.Series(([0] * 8) + ([1] * 8))

    model = RiskSegOptimizer(
        auto_bin_numeric=True,
        n_numeric_bins=4,
        use_grouping=False,
        group_binned_numeric=True,
        verbose=0,
    )

    _, seg_view = model._prepare_views_fit(X)
    groups = model._generate_split_groups(seg_view["x"])
    assert any(len(group) == 2 for group in groups)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_riskseg_optimizer.py -k contiguous_groups -q`
Expected: FAIL because `group_binned_numeric` is not implemented and no 2-bin contiguous groups are generated.

- [ ] **Step 3: Write minimal implementation**

```python
self._seg_numeric_label_order_ = {}
...
if numeric_column_was_binned:
    self._seg_numeric_label_order_[col] = [str(cat) for cat in binned.cat.categories]
...
def _generate_split_groups(self, series):
    if self.group_binned_numeric and series.name in self._seg_numeric_label_order_:
        ordered = ...
        contiguous_groups = ...
        return contiguous_groups
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_riskseg_optimizer.py -k contiguous_groups -q`
Expected: PASS

### Task 2: Lock in benchmark preset selection for numeric-heavy datasets

**Files:**
- Modify: `D:\Nuvem\gdrive_pessoal\Faculdades\Impacta\Graduação\Outros\IC\RiskSeg\tests\test_benchmark_suite.py`
- Modify: `D:\Nuvem\gdrive_pessoal\Faculdades\Impacta\Graduação\Outros\IC\RiskSeg\benchmarks\suite.py`
- Test: `D:\Nuvem\gdrive_pessoal\Faculdades\Impacta\Graduação\Outros\IC\RiskSeg\tests\test_benchmark_suite.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_numeric_low_dimensional_benchmark_preset_uses_deeper_tree():
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f"x{i}" for i in range(5)])
    y = pd.Series(([0] * 50) + ([1] * 50))
    spec = BenchmarkDatasetSpec("toy", 100, 5)

    params = recommend_riskseg_benchmark_params(X, y, spec)
    assert params["max_depth"] == 3
    assert params["top_k_variables"] == 3


def test_numeric_imbalanced_high_dimensional_preset_uses_more_bins():
    X = pd.DataFrame(np.random.randn(200, 80), columns=[f"x{i}" for i in range(80)])
    y = pd.Series(([0] * 180) + ([1] * 20))
    spec = BenchmarkDatasetSpec("toy", 200, 80)

    params = recommend_riskseg_benchmark_params(X, y, spec)
    assert params["n_numeric_bins"] == 8
    assert params["max_depth"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_suite.py -k benchmark_preset -q`
Expected: FAIL because `recommend_riskseg_benchmark_params` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def recommend_riskseg_benchmark_params(X, y, spec):
    params = {...baseline...}
    if numeric_only:
        params["group_binned_numeric"] = True
        params["top_k_variables"] = 3
        if high_dimensional_and_imbalanced:
            params["n_numeric_bins"] = 8
            params["max_depth"] = 2
        elif low_dimensional:
            params["n_numeric_bins"] = 6
            params["max_depth"] = 3
        else:
            params["n_numeric_bins"] = 4
            params["max_depth"] = 3
    return params
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_benchmark_suite.py -k benchmark_preset -q`
Expected: PASS

### Task 3: Wire the tuned preset into the benchmark and verify improvement

**Files:**
- Modify: `D:\Nuvem\gdrive_pessoal\Faculdades\Impacta\Graduação\Outros\IC\RiskSeg\benchmarks\suite.py`
- Modify: `D:\Nuvem\gdrive_pessoal\Faculdades\Impacta\Graduação\Outros\IC\RiskSeg\docs\benchmark_report.md`
- Test: `D:\Nuvem\gdrive_pessoal\Faculdades\Impacta\Graduação\Outros\IC\RiskSeg\tests\test_benchmark_suite.py`

- [ ] **Step 1: Update the RiskSeg builder to consume the tuned params**

```python
params = recommend_riskseg_benchmark_params(X, y, spec)
return Pipeline(
    steps=[
        ("imputer", DataFrameImputer()),
        ("model", RiskSegOptimizer(**params)),
    ]
)
```

- [ ] **Step 2: Run focused regression tests**

Run: `pytest tests/test_riskseg_optimizer.py tests/test_benchmark_suite.py -q`
Expected: PASS

- [ ] **Step 3: Run focused benchmark verification**

Run: `python scripts/run_benchmark.py --datasets coil2000 phoneme ring --models RiskSeg --output-dir artifacts/benchmarks_numeric_tuned`
Expected: AUC improvements over the current article-aligned aggregate benchmark for all three datasets.

- [ ] **Step 4: Update the report with the tuned numeric-dataset results**

```markdown
- `coil2000`: 0.6591 -> improved tuned result
- `phoneme`: 0.8377 -> improved tuned result
- `ring`: 0.8821 -> improved tuned result
```
