import os
import numpy as np
import pandas as pd
from django.test import TestCase
from django.conf import settings
from .regressor import (
    find,
    get_model,
    predict_heart_disease,
    LogitRegression,
    FEATURE_NAMES
)


class MLPipelineTestCase(TestCase):
    """
    Unit and integration tests for the Heart Disease ML prediction pipeline.
    """

    def test_feature_order_and_dataset(self):
        """Verify all 13 feature orders, presence, and target values."""
        X, Y = find()
        self.assertEqual(X.shape[1], 13)
        self.assertEqual(list(X.columns), FEATURE_NAMES)
        self.assertEqual(set(Y.unique()), {0, 1})
        # 1 represents heart disease (138 records), 0 represents healthy (165 records)
        self.assertEqual((Y == 1).sum(), 138)
        self.assertEqual((Y == 0).sum(), 165)

    def test_high_risk_patient_prediction(self):
        """
        Verify high-risk patient with ST depression (oldpeak=2.5), angina (exang=1),
        blocked vessels (ca=2), and defect (thal=3) yields high probability (> 80%).
        """
        example_input = [63, 1, 3, 145, 260, 1, 0, 120, 1, 2.5, 2, 2, 3]
        prediction, probability = predict_heart_disease(example_input)
        self.assertEqual(prediction, 1)
        self.assertGreater(probability, 0.80)
        self.assertAlmostEqual(probability, 0.9214, places=2)

    def test_low_risk_patient_prediction(self):
        """Verify asymptomatic patient with normal markers yields low risk (< 0.50)."""
        healthy_input = [40, 0, 0, 110, 170, 0, 0, 172, 0, 0.0, 2, 0, 2]
        prediction, probability = predict_heart_disease(healthy_input)
        self.assertEqual(prediction, 0)
        self.assertLess(probability, 0.50)

    def test_custom_logit_regression(self):
        """Verify Custom Gradient Descent LogitRegression functions accurately."""
        X, Y = find()
        model = LogitRegression(learning_rate=0.05, iterations=500)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, Y)

        example_input = np.array([[63, 1, 3, 145, 260, 1, 0, 120, 1, 2.5, 2, 2, 3]])
        ex_scaled = scaler.transform(example_input)
        prob = model.predict_proba(ex_scaled)[0, 1]
        self.assertGreater(prob, 0.80)
