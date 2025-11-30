import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

import sys
import os

# Add the src directory to sys.path to allow importing the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import your class
try:
    from regularizeddiscriminantanalysis import RegularizedDiscriminantAnalysis
except ImportError:
    raise ImportError("Could not import 'RegularizedDiscriminantAnalysis' from 'regularizeddiscriminantanalysis'. Make sure the package is installed or in the python path.")

def test_rda_implementation():
    print("="*60)
    print("RDA IMPLEMENTATION TEST SUITE")
    print("="*60)

    # 1. Generate Synthetic Data
    # --------------------------
    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=3, n_redundant=0, 
        n_classes=3, random_state=42, class_sep=2.0
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Data generated: {len(y_train)} training samples, 3 classes, 5 features.\n")

    # 2. Test QDA Equivalence (lambda=0, gamma=0)
    # -------------------------------------------
    print(f"TEST 1: QDA Equivalence (lambda=0, gamma=0)")
    
    # Sklearn QDA
    qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    qda.fit(X_train, y_train)
    qda_pred = qda.predict(X_test)
    
    # Your RDA
    rda_qda = RegularizedDiscriminantAnalysis(lambda_=0.0, gamma=0.0)
    rda_qda.fit(X_train, y_train)
    rda_pred = rda_qda.predict(X_test)
    
    match_count = np.sum(qda_pred == rda_pred)
    total = len(y_test)
    print(f"  - Sklearn QDA Accuracy: {accuracy_score(y_test, qda_pred):.4f}")
    print(f"  - Your RDA Accuracy:    {accuracy_score(y_test, rda_pred):.4f}")
    print(f"  - Prediction Match:     {match_count}/{total} ({match_count/total:.1%})")
    
    if match_count == total:
        print("  ✅ SUCCESS: Matches Sklearn QDA exactly.")
    else:
        print("  ⚠️ WARNING: Does not match exactly. (Small differences are expected due to biased/unbiased estimator choices, but accuracy should be similar).")

    print("-" * 60)

    # 3. Test LDA Equivalence (lambda=1, gamma=0)
    # -------------------------------------------
    print(f"TEST 2: LDA Equivalence (lambda=1, gamma=0)")
    
    # Sklearn LDA
    lda = LinearDiscriminantAnalysis(store_covariance=True)
    lda.fit(X_train, y_train)
    lda_pred = lda.predict(X_test)
    
    # Your RDA
    rda_lda = RegularizedDiscriminantAnalysis(lambda_=1.0, gamma=0.0)
    rda_lda.fit(X_train, y_train)
    rda_pred_lda = rda_lda.predict(X_test)
    
    match_count = np.sum(lda_pred == rda_pred_lda)
    print(f"  - Sklearn LDA Accuracy: {accuracy_score(y_test, lda_pred):.4f}")
    print(f"  - Your RDA Accuracy:    {accuracy_score(y_test, rda_pred_lda):.4f}")
    print(f"  - Prediction Match:     {match_count}/{total} ({match_count/total:.1%})")
    
    if match_count == total:
        print("  ✅ SUCCESS: Matches Sklearn LDA exactly.")
    else:
        print("  ⚠️ WARNING: Mismatch found. Check your 'pooled covariance' calculation. Sklearn uses weighted average.")

    print("-" * 60)

    # 4. Math Logic & Scale Bug Check
    # -------------------------------
    print(f"TEST 3: Scale/Math Logic Check")
    
    # We create a tiny dummy dataset where we know the answer
    X_dummy = np.array([[1., 2.], [2., 3.]]) # Class 0
    y_dummy = np.array([0, 0])
    
    # Fit RDA
    rda_debug = RegularizedDiscriminantAnalysis(lambda_=0.0, gamma=0.0)
    rda_debug.fit(X_dummy, y_dummy)
    
    # Check the stored covariance for class 0
    # Standard Covariance of [[1,2], [2,3]] is [[0.5, 0.5], [0.5, 0.5]] (Using N=2, biased? or N-1=1 unbiased?)
    # Sklearn uses Unbiased by default usually, or Biased depending on solver.
    # Numpy cov uses (N-1) by default.
    
    try:
        # Accessing internal attribute - adjust name if you changed it (e.g. regularized_covariances_ or covariances_)
        if hasattr(rda_debug, 'regularized_covariances_'):
            stored_cov = rda_debug.regularized_covariances_[0]
        elif hasattr(rda_debug, 'covariances_'):
            stored_cov = rda_debug.covariances_[0]
        else:
            print("  ❓ SKIPPING: Could not find 'covariances_' attribute to inspect.")
            return

        print(f"  - Input Data (Class 0):\n{X_dummy}")
        print(f"  - Your Computed Covariance:\n{stored_cov}")
        
        # Heuristic check: Elements should be around 0.5, not 0.005
        if np.all(np.abs(stored_cov) < 0.01):
            print("  ❌ CRITICAL FAIL: Covariance values are tiny. You likely have the 'Scale Bug' (dividing by weight twice).")
        elif np.all(stored_cov == 0):
             print("  ❌ FAIL: Covariance is zero.")
        else:
            print("  ✅ PASS: Covariance scale looks reasonable.")
            
    except Exception as e:
        print(f"  ❌ ERROR during inspection: {e}")

if __name__ == "__main__":
    test_rda_implementation()