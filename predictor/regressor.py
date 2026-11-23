"""
Heart Disease Prediction Model - Machine Learning Pipeline.

Root Cause Analysis:
In the Cleveland Heart Disease dataset (heartdata.csv), the raw 'target' column
encodes angiographic disease status where:
  - target = 0 : Presence of heart disease (> 50% diameter narrowing, high risk)
  - target = 1 : Absence of heart disease (< 50% diameter narrowing, healthy)

When models are trained directly on Y = df['target'], class 1 represents 'healthy'.
Consequently, model.predict_proba()[:, 1] outputs the probability of being healthy,
causing high-risk cardiovascular patients to be incorrectly classified with very
low probabilities (e.g. 9.12% instead of ~92.14%).

Fix:
We map the target to standard medical and machine learning convention:
  Y = 1 - df['target']
Where:
  1 = Heart Disease Present (High Risk)
  0 = No Heart Disease / Healthy (Low Risk)

Both custom gradient-descent Logistic Regression (LogitRegression) and
scikit-learn Logistic Regression in a StandardScaler pipeline are supported.
"""

import os
import pickle
import numpy as np
import pandas as pd

from django.conf import settings

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# 13 Clinical features in standard order
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]


class LogitRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression with Gradient Descent and Sigmoid activation.
    Implements standard scikit-learn estimator interface.
    """

    def __init__(self, learning_rate=0.05, iterations=3000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.W = None
        self.b = 0.0
        self.classes_ = np.array([0, 1])

    @staticmethod
    def sigmoid(z):
        """Numerically stable sigmoid function."""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, Y):
        """Fit weights using batch gradient descent."""
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float).ravel()
        m, n = X.shape

        self.W = np.zeros(n)
        self.b = 0.0

        for _ in range(self.iterations):
            linear_model = np.dot(X, self.W) + self.b
            A = self.sigmoid(linear_model)
            error = A - Y

            dW = np.dot(X.T, error) / m
            db = np.sum(error) / m

            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

        return self

    def predict_proba(self, X):
        """Return probability estimates for [Class 0 (Healthy), Class 1 (Disease)]."""
        X = np.asarray(X, dtype=float)
        linear_model = np.dot(X, self.W) + self.b
        prob_disease = self.sigmoid(linear_model)
        prob_healthy = 1.0 - prob_disease
        return np.column_stack((prob_healthy, prob_disease))

    def predict(self, X):
        """Predict binary class (1 = Heart Disease, 0 = Healthy)."""
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= 0.5).astype(int)


def find():
    """
    Load the heart disease dataset and apply verified clinical target encoding.
    """
    csv_path = os.path.join(
        settings.BASE_DIR,
        'predictor',
        'heartdata.csv'
    )

    df = pd.read_csv(csv_path)

    # Validate feature presence
    X = df[FEATURE_NAMES].copy()

    # Correct target mapping: 1 = Disease (previously 0), 0 = Healthy (previously 1)
    Y = (1 - df['target']).astype(int)

    return X, Y


def train_model(use_custom=False):
    """
    Train and evaluate the Logistic Regression model using an integrated
    StandardScaler pipeline so training and inference use identical scaling.
    """
    X, Y = find()

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.20,
        random_state=42,
        stratify=Y
    )

    classifier = (
        LogitRegression(learning_rate=0.05, iterations=3000)
        if use_custom
        else LogisticRegression(max_iter=2000, solver='lbfgs', random_state=42)
    )

    # Scaling + Classifier in a single pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])

    # Train model
    model.fit(X_train, Y_train)

    # Evaluate model
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(Y_test, predictions)
    prec = precision_score(Y_test, predictions)
    rec = recall_score(Y_test, predictions)
    f1 = f1_score(Y_test, predictions)
    auc = roc_auc_score(Y_test, probabilities)

    print("\n========== MODEL EVALUATION ==========")
    print(f"Accuracy  : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Precision : {prec:.4f} ({prec * 100:.2f}%)")
    print(f"Recall    : {rec:.4f} ({rec * 100:.2f}%)")
    print(f"F1-Score  : {f1:.4f} ({f1 * 100:.2f}%)")
    print(f"ROC AUC   : {auc:.4f} ({auc * 100:.2f}%)")
    print("\nConfusion Matrix:")
    print(confusion_matrix(Y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(Y_test, predictions, target_names=['Healthy', 'Heart Disease']))

    return model


def get_model(force_retrain=False):
    """
    Load the cached model pipeline or train a new one if missing or requested.
    """
    model_path = os.path.join(
        settings.BASE_DIR,
        'predictor',
        'saved_model.pkl'
    )

    if not force_retrain and os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[Warning] Could not load saved_model.pkl: {e}. Retraining...")

    model = train_model()

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    return model


def predict_heart_disease(input_data):
    """
    Predicts heart disease presence and probability for a given feature vector.

    Parameters:
        input_data (list or array-like): 13 clinical parameters matching FEATURE_NAMES.

    Returns:
        prediction (int): 1 if probability >= 0.5 (High Risk), else 0 (Low Risk).
        probability (float): Probability of heart disease presence [0.0, 1.0].
    """
    model = get_model()

    # Pass as DataFrame with exact feature names to match StandardScaler pipeline
    if isinstance(input_data, (list, tuple, np.ndarray)):
        arr = np.asarray(input_data, dtype=float).reshape(1, -1)
        input_df = pd.DataFrame(arr, columns=FEATURE_NAMES)
    else:
        input_df = input_data

    # Probability of Class 1 (Heart Disease)
    probability = float(model.predict_proba(input_df)[0][1])

    prediction = int(probability >= 0.5)

    return prediction, probability
