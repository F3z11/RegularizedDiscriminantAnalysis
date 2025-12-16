# Regularized Discriminant Analysis (RDA)

[![PyPI version](https://badge.fury.io/py/RegularizedDiscriminantAnalysis.svg)](https://badge.fury.io/py/RegularizedDiscriminantAnalysis)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![mypy](https://github.com/F3z11/RegularizedDiscriminantAnalysis/actions/workflows/mypy.yml/badge.svg)](https://github.com/F3z11/RegularizedDiscriminantAnalysis/actions/workflows/mypy.yml)
[![ruff](https://github.com/F3z11/RegularizedDiscriminantAnalysis/actions/workflows/mypy.yml/badge.svg)](https://github.com/F3z11/RegularizedDiscriminantAnalysis/actions/workflows/ruff.yml)
[![pytest](https://github.com/F3z11/RegularizedDiscriminantAnalysis/actions/workflows/mypy.yml/badge.svg)](https://github.com/F3z11/RegularizedDiscriminantAnalysis/actions/workflows/pytest.yml)


A scikit-learn compatible implementation of Regularized Discriminant Analysis (RDA) as proposed by Friedman (1989).

RDA is a classifier that varies continuously between Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA) using two regularization parameters: $\lambda$ and $\gamma$.

This implementation uses SVD (Singular Value Decomposition) via `scipy.linalg.pinv` for numerically stable computation of covariance matrix inverses.

## Installation

Install from PyPI:

```bash
pip install RegularizedDiscriminantAnalysis
```

Or install locally:

```bash
pip install .
```

## Usage

```python
from regularizeddiscriminantanalysis import RegularizedDiscriminantAnalysis
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_classification(n_samples=100, n_features=4, n_informative=3, n_redundant=0, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize RDA
# lambda_=0.0, gamma=0.0 -> QDA
# lambda_=1.0, gamma=0.0 -> LDA
clf = RegularizedDiscriminantAnalysis(lambda_=0.5, gamma=0.2)

# Fit
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
print(f"Accuracy: {clf.score(X_test, y_test):.2f}")
```

## Parameters

- `lambda_` (float, default=0.0): Covariance mixing parameter (0 <= lambda <= 1).
    - 0.0: Class-specific covariances (QDA-like)
    - 1.0: Pooled covariance (LDA-like)
- `gamma` (float, default=0.0): Eigenvalue shrinkage parameter (0 <= gamma <= 1).
    - 0.0: No shrinkage
    - 1.0: Shrinks covariance towards a scalar multiple of identity matrix (Spherical)


## Authors

- Federico Clerici — fez.cle@gmail.com 
- Raffaele D'Agostino — raffaeledagostino11@gmail.com 


## License

MIT

## References

Friedman, J. H. (1989). [Regularized discriminant analysis](https://doi.org/10.1080/01621459.1989.10478752). *Journal of the American statistical association*, 84(405), 165-175.
