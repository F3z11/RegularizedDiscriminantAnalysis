from __future__ import annotations

import numpy as np
from scipy.linalg import pinv
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import (  # type: ignore
    check_is_fitted,
    check_X_y,
    validate_data,
)


class RegularizedDiscriminantAnalysis(ClassifierMixin, BaseEstimator):
    """
    Regularized Discriminant Analysis (RDA)

    A classifier that varies continuously between Linear Discriminant Analysis (LDA)
    and Quadratic Discriminant Analysis (QDA) using two regularization parameters.

    Implementation based on Friedman (1989): "Regularized Discriminant Analysis".

    Parameters
    ----------
    lambda_ : float, default=0.0
        Covariance mixing parameter (0 <= lambda_ <= 1).
        - 0.0: Class-specific covariances (QDA-like)
        - 1.0: Pooled covariance (LDA-like)

    gamma : float, default=0.0
        Eigenvalue shrinkage parameter (0 <= gamma <= 1).
        - 0.0: No shrinkage
        - 1.0: Shrinks covariance towards a scalar multiple of identity matrix (Spherical)

    priors : array-like of shape (n_classes,), default=None
        Class prior probabilities. If None, estimated from training data.

    reg_param : float, default=1e-6
        Small constant added to diagonal for numerical stability during inversion.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.

    n_classes_ : int
        Number of classes.

    means_ : ndarray of shape (n_classes, n_features)
        Class means.

    covariances_ : ndarray of shape (n_classes, n_features, n_features)
        Class covariances (biased estimates).

    pooled_cov_ : ndarray of shape (n_features, n_features)
        Weighted average of class covariances.

    priors_ : ndarray of shape (n_classes,)
        Class priors.

    regularized_covariances_ : ndarray of shape (n_classes, n_features, n_features)
        Covariances after applying lambda and gamma regularization.

    precisions_ : ndarray of shape (n_classes, n_features, n_features)
        Inverse of regularized covariances.

    log_dets_ : ndarray of shape (n_classes,)
        Log-determinants of regularized covariances.
    """

    def __init__(
        self,
        lambda_: float = 0.0,
        gamma: float = 0.0,
        priors: np.ndarray | None = None,
        reg_param: float = 1e-6,
    ):
        # self.lambda_ = lambda_ this actually causes a problem cause sklearn.check_is_fitted thinks the estimator has already been fitted
        self._is_fitted = False

        self.lambda_ = lambda_
        self.gamma = gamma
        self.priors = priors
        self.reg_param = reg_param

    def __sklearn_is_fitted__(self) -> bool:
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted

    def fit(self, X: np.ndarray, y: np.ndarray) -> RegularizedDiscriminantAnalysis:
        """
        Fit the RDA model according to the given training data and parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate parameters
        if not (0 <= self.lambda_ <= 1):
            raise ValueError(f"lambda_ must be between 0 and 1, got {self.lambda_}")
        if not (0 <= self.gamma <= 1):
            raise ValueError(f"gamma must be between 0 and 1, got {self.gamma}")

        # Check inputs
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        n_classes = len(self.classes_)
        n_samples, self.n_features_in_ = X.shape

        # 1. Compute Basic Statistics (Means, Covariances, Priors)
        self.means_ = np.empty((n_classes, self.n_features_in_), dtype=float)
        self.covariances_ = np.empty(
            (n_classes, self.n_features_in_, self.n_features_in_), dtype=float
        )
        self._class_counts = np.empty(n_classes, dtype=int)

        # We need the total scatter/covariance for pooling
        # Friedman uses biased (MLE) estimates in derivation, so we use ddof=0 (divide by N)
        # to keep math consistent with the mixing weights (N_k vs N). (coherent also with sklearn)

        for idx, k in enumerate(self.classes_):
            X_k = X[y == k]
            N_k = X_k.shape[0]

            self._class_counts[idx] = N_k
            self.means_[idx] = np.mean(X_k, axis=0)

            cov_k = np.cov(X_k, rowvar=False, bias=True)

            if cov_k.ndim == 0:
                cov_k = np.zeros((self.n_features_in_, self.n_features_in_))

            self.covariances_[idx] = cov_k

        # Compute Priors
        if self.priors is None:
            self.priors_ = self._class_counts / n_samples
        else:
            self.priors_ = np.array(self.priors)

        # 2. Compute Pooled Covariance (Weighted Average)
        # Friedman Eq (15): S = Sum(S_k) and Eq (14): Sigma_pool = S / N.
        # This is equivalent to weighted average of biased class covariances.
        self.pooled_cov_ = np.average(self.covariances_, axis=0, weights=self._class_counts)

        # 3. Apply Regularization (Lambda & Gamma) - Pre-compute inverses
        self._apply_regularization(n_samples, self.n_features_in_)

        self._is_fitted = True

        return self

    def _apply_regularization(self, n_samples: int, n_features: int) -> None:
        """
        Computes the regularized covariance matrices, their inverses (precisions),
        and log-determinants. Stores them for use in prediction.
        """
        self.regularized_covariances_ = np.empty(
            (len(self.classes_), n_features, n_features),
            dtype=float,
        )
        self.precisions_ = np.empty(
            (len(self.classes_), n_features, n_features),
            dtype=float,
        )
        self.log_dets_ = np.empty(len(self.classes_), dtype=float)

        Identity = np.eye(n_features)

        for k in range(len(self.classes_)):
            N_k = self._class_counts[k]
            Sigma_k = self.covariances_[k]

            # --- Step A: Lambda Mixing (Covariance vs Pool) ---
            # Friedman Eq (16) logic adapted for Covariances:
            # S_k(lambda) = (1-lambda)S_k + lambda*S
            # Sigma_k(lambda) = S_k(lambda) / [(1-lambda)N_k + lambda*N]

            # Let's compute the mixing weight alpha_k
            denom = (1 - self.lambda_) * N_k + self.lambda_ * n_samples
            alpha_k = ((1 - self.lambda_) * N_k) / denom
            beta_k = (self.lambda_ * n_samples) / denom

            Sigma_lambda_k = alpha_k * Sigma_k + beta_k * self.pooled_cov_

            # --- Step B: Gamma Mixing (Shrinkage towards Identity) ---
            # Friedman Eq (18): (1-gamma)Sigma_k(lambda) + gamma*(tr(Sigma_k(lambda))/p)*I

            avg_eig = np.trace(Sigma_lambda_k) / n_features

            Sigma_final_k = (1 - self.gamma) * Sigma_lambda_k + self.gamma * avg_eig * Identity

            # Add small constant for numerical stability
            Sigma_final_k += self.reg_param * Identity

            self.regularized_covariances_[k] = Sigma_final_k

            # Compute Precision (Inverse) and Log-Determinant
            # Using pinv (pseudo-inverse) is safer for potentially singular matrices (with SVD)
            self.precisions_[k] = pinv(Sigma_final_k)

            # Compute Log Determinant (using slogdet for stability)
            sign, logdet = np.linalg.slogdet(Sigma_final_k)
            if sign <= 0:
                # If determinant is 0 or negative (shouldn't happen with reg_param), fallback
                logdet = np.log(max(np.linalg.det(Sigma_final_k), 1e-10))
            self.log_dets_[k] = logdet

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        # X = check_array(X)
        X = validate_data(self, X, reset=False)

        scores = self._decision_function(X)
        try:
            res: np.ndarray = self.classes_[np.argmax(scores, axis=1)]
        except IndexError as err:
            raise IndexError(
                "The number of classes in the training data does not match "
                "the number of classes during prediction."
            ) from err

        return res

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the discriminant score (log-posterior) for each class.
        Score_k = - (log|Sigma_k| + (x-mu_k)^T Sigma_k^-1 (x-mu_k)) + 2 * log(prior_k)
        """
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, len(self.classes_)))

        for k in range(len(self.classes_)):
            # Get pre-computed parameters
            precision = self.precisions_[k]
            log_det = self.log_dets_[k]
            mean = self.means_[k]
            prior = self.priors_[k]

            # Centered data
            X_centered = X - mean

            # Mahalanobis Distance squared: (x-mu)^T * Sigma^-1 * (x-mu)
            # Efficient calculation: sum( (X_centered @ precision) * X_centered, axis=1)
            mahalanobis = np.sum(np.dot(X_centered, precision) * X_centered, axis=1)

            # Score (Quadratic term + Log Det term + Prior term)
            # We omit constant terms (like n_features * log(2pi)) as they don't affect argmax
            scores[:, k] = -(log_det + mahalanobis) + 2 * np.log(prior)

        return scores

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        probs : ndarray of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in self.classes_.
        """
        check_is_fitted(self)
        # X = check_array(X)
        X = validate_data(self, X, reset=False)

        # Get log-posterior scores (unnormalized)
        decision_scores = self._decision_function(X)

        # Apply Softmax (Numerically Stable)
        # Subtract max per row to avoid overflow in exp()
        max_scores = np.max(decision_scores, axis=1, keepdims=True)
        exp_scores = np.exp(decision_scores - max_scores)
        probs: np.ndarray = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return probs

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return log-probability estimates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        log_probs : ndarray of shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the model.
        """
        log_proba: np.ndarray = np.log(self.predict_proba(X))
        return log_proba
