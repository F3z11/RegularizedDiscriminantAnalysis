# type: ignore

import os
import pickle
import sys
from typing import Any

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator, parametrize_with_checks

# Add the src directory to sys.path to allow importing the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

try:
    from regularizeddiscriminantanalysis import RegularizedDiscriminantAnalysis
except ImportError:
    raise ImportError(
        "Could not import 'RegularizedDiscriminantAnalysis'. Ensure it is installed or discoverable."
    ) from None


@pytest.mark.parametrize(
    "estimator_cls",
    [RegularizedDiscriminantAnalysis],
)
def test_clone_rda(estimator_cls: type[Any]) -> None:
    estimator = estimator_cls()
    auto = clone(estimator)
    assert isinstance(auto, RegularizedDiscriminantAnalysis)


@pytest.mark.parametrize(
    "estimator_cls",
    [RegularizedDiscriminantAnalysis],
)
def test_check_estimator_basic(estimator_cls: type[Any]) -> None:
    estimator = estimator_cls()
    check_estimator(estimator)


@parametrize_with_checks([RegularizedDiscriminantAnalysis()])
def test_sklearn_compatibility(estimator: Any, check: Any) -> None:
    check(estimator)


@pytest.mark.parametrize(
    "estimator_cls",
    [RegularizedDiscriminantAnalysis],
)
def test_pipeline_usage(estimator_cls: type[Any]) -> None:
    rng = np.random.RandomState(0)
    X = rng.randn(100, 10)
    # Using StandardScaler then the estimator
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("cov", estimator_cls()),
        ]
    )
    pipe.fit(X)
    # If estimator has a .score method, call it
    score_val = pipe.named_steps["cov"].score(X)
    assert np.isfinite(score_val)


@pytest.mark.parametrize(
    "estimator_cls",
    [RegularizedDiscriminantAnalysis],
)
def test_grid_search_cv(estimator_cls: type[Any]) -> None:
    rng = np.random.RandomState(1)
    X = rng.randn(50, 5)
    y = rng.randn(50)
    param_grid: dict[str, Any] = {
        "lambda_": [0.0, 0.5, 1.0],
        "gamma": [0.0, 0.5, 1.0],
    }
    gs = GridSearchCV(estimator_cls(), param_grid=param_grid, cv=3, scoring="r2")
    gs.fit(X, y)
    best = gs.best_estimator_
    assert isinstance(best, estimator_cls)
    assert best is not gs.estimator


@pytest.mark.parametrize(
    "estimator_cls",
    [RegularizedDiscriminantAnalysis],
)
def test_clone_and_pickle(estimator_cls: type[Any]) -> None:
    rng = np.random.RandomState(2)
    X = rng.randn(80, 8)
    y = np.ones(80)
    est = estimator_cls()
    est.fit(X, y)
    s = pickle.dumps(est)
    est2 = pickle.loads(s)

    assert isinstance(est2, estimator_cls)
