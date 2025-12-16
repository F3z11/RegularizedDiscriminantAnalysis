# type: ignore

import os
import pickle
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator, check_is_fitted, parametrize_with_checks

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
def test_pipeline(estimator_cls: type[Any]) -> None:
    est = estimator_cls()

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", est)])
    pipe2 = clone(pipe)

    params = pipe2.get_params(deep=True)
    assert "clf__lambda_" in params
    assert "clf__gamma" in params
    assert "clf__reg_param" in params
    assert "scaler__with_mean" in params
    assert "scaler__with_std" in params


@dataclass(frozen=True)
class GridSearchCheckResult:
    best_params: dict[str, Any]
    best_score: float


def assert_gridsearch_compatible(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    param_grid: dict[str, list[Any]],
    scoring: str = "accuracy",
    cv_splits: int = 5,
    refit: bool = True,
) -> GridSearchCheckResult:
    try:
        est2 = clone(estimator)
    except Exception as e:
        raise AssertionError(
            "Estimator is not cloneable. Ensure __init__ only assigns parameters "
            "verbatim and you don't override __getstate__/__setstate__ oddly."
        ) from e

    params = est2.get_params(deep=True)
    missing = [k for k in param_grid.keys() if k not in params]
    if missing:
        raise AssertionError(
            f"param_grid contains keys not in estimator.get_params(): {missing}.\n"
            f"Available params: {sorted(params.keys())}"
        )

    first_setting = {k: v[0] for k, v in param_grid.items()}
    try:
        est2.set_params(**first_setting)
    except Exception as e:
        raise AssertionError("Estimator.set_params failed for provided grid keys.") from e

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=0)
    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        refit=refit,
        error_score="raise",
        n_jobs=1,
        return_train_score=True,
    )
    gs.fit(X, y)

    if not hasattr(gs, "best_estimator_"):
        raise AssertionError("GridSearchCV did not produce best_estimator_.")

    try:
        check_is_fitted(gs.best_estimator_)
    except Exception as e:
        raise AssertionError(
            "best_estimator_ is not recognized as fitted. "
            "Ensure fit() sets fitted attributes and/or __sklearn_is_fitted__."
        ) from e

    if not hasattr(gs, "best_params_") or not hasattr(gs, "best_score_"):
        raise AssertionError("GridSearchCV did not populate best_params_ / best_score_.")

    return GridSearchCheckResult(
        best_params=dict(gs.best_params_), best_score=float(gs.best_score_)
    )


@pytest.mark.parametrize("estimator_cls", [RegularizedDiscriminantAnalysis])
def test_gridsearch(estimator_cls: type[Any]) -> None:
    X, y = make_classification(
        n_samples=400, n_features=10, n_informative=6, n_redundant=0, n_classes=3, random_state=0
    )

    est = estimator_cls()
    grid = {
        "lambda_": [0.0, 0.5, 1.0],
        "gamma": [0.0, 0.2, 0.8],
        "reg_param": [1e-8, 1e-6],
    }

    res = assert_gridsearch_compatible(est, X, y, param_grid=grid)
    assert set(res.best_params.keys()) == set(grid.keys())


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
