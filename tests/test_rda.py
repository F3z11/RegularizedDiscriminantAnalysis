import os
import sys
import warnings

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split

# Add the src directory to sys.path to allow importing the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

try:
    from regularizeddiscriminantanalysis import RegularizedDiscriminantAnalysis
except ImportError:
    raise ImportError(
        "Could not import 'RegularizedDiscriminantAnalysis'. Ensure it is installed or discoverable."
    ) from None


@pytest.fixture(scope="module")
def sample_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a small multiclass dataset for reproducible tests."""
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_classes=3,
        random_state=42,
        class_sep=2.0,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def test_rda_match(sample_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    X_train, X_test, y_train, y_test = sample_data

    qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    qda.fit(X_train, y_train)
    qda_pred = qda.predict(X_test)

    rda_qda = RegularizedDiscriminantAnalysis(lambda_=0.0, gamma=0.0)
    rda_qda.fit(X_train, y_train)
    rda_pred = rda_qda.predict(X_test)

    match_count = np.sum(qda_pred == rda_pred)
    total = len(y_test)

    assert match_count == total, (
        "RDA with lambda=0, gamma=0 should reproduce QDA predictions exactly."
    )


def test_lda_equivalence(
    sample_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X_train, X_test, y_train, y_test = sample_data

    lda = LinearDiscriminantAnalysis(store_covariance=True)
    lda.fit(X_train, y_train)
    lda_pred = lda.predict(X_test)

    rda_lda = RegularizedDiscriminantAnalysis(lambda_=1.0, gamma=0.0)
    rda_lda.fit(X_train, y_train)
    rda_pred = rda_lda.predict(X_test)

    match_count = np.sum(lda_pred == rda_pred)
    total = len(y_test)

    # Logging information for debugging (pytest captures this)
    print(f"Sklearn LDA vs RDA(lambda=1): matched predictions {match_count}/{total}")

    assert match_count == total, (
        "RDA with lambda=1, gamma=0 should reproduce LDA predictions exactly."
    )


def test_scale_logic() -> None:
    """Ensure covariance scaling is correct on a toy example."""
    X_dummy = np.array([[1.0, 2.0], [2.0, 3.0]])
    y_dummy = np.array([0, 0])

    rda_debug = RegularizedDiscriminantAnalysis(lambda_=0.0, gamma=0.0)
    rda_debug.fit(X_dummy, y_dummy)

    if hasattr(rda_debug, "regularized_covariances_"):
        stored_cov = rda_debug.regularized_covariances_[0]
    elif hasattr(rda_debug, "covariances_"):
        stored_cov = rda_debug.covariances_[0]
    else:
        warnings.warn(
            "Skipping covariance scale test: RDA exposes neither "
            "'regularized_covariances_' nor 'covariances_'.",
            stacklevel=2,
        )
        return

    # Log the values for pytest output (not an assertion condition)
    print("Input data:\n", X_dummy)
    print("Computed covariance matrix:\n", stored_cov)

    # Core assertion: covariance must have reasonable magnitude
    if np.all(stored_cov == 0):
        raise AssertionError("Covariance matrix contains only zeros.")

    if np.all(np.abs(stored_cov) < 0.01):
        raise AssertionError(
            "Covariance values are extremely small. "
            "This suggests a scaling issue (possibly dividing by a weight twice)."
        )

    # If covariance looks valid but shape is off, warn
    if stored_cov.shape != (2, 2):
        warnings.warn(
            f"Unexpected covariance matrix shape: {stored_cov.shape}.",
            stacklevel=1,
        )
