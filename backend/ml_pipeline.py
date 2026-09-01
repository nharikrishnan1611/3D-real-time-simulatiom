import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import shap

# This simulates a pre-trained ML pipeline for the hackathon demo.
# In a real environment, you would load models from disk (e.g. .pkl files)

class FloodRiskPredictor:
    def __init__(self):
        # Create a dummy dataset to train the models so they are ready for inference
        np.random.seed(42)
        # Features: [elevation (m), building_height (m), distance_to_water (m), building_type_encoded]
        X_train = np.random.rand(100, 4) * [20, 100, 500, 3] 
        # Target: Vulnerability Score (0-100)
        y_train = (20 - X_train[:, 0]) * 3 + (500 - X_train[:, 2]) * 0.1
        y_train = np.clip(y_train, 0, 100)

        # Train XGBoost
        self.xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50)
        self.xgb_model.fit(X_train, y_train)

        # Train Logistic Regression for a binary 'collapse' risk (dummy)
        y_class = (y_train > 60).astype(int)
        self.lr_model = LogisticRegression()
        self.lr_model.fit(X_train, y_class)

        # Initialize SHAP explainer for XGBoost
        self.explainer = shap.TreeExplainer(self.xgb_model)

    def predict(self, elevation, building_height, distance_to_water, building_type, flood_level):
        """
        Predicts the vulnerability score and generates SHAP explanations.
        """
        # Encode building type simply
        type_map = {'residential': 0, 'modern': 1, 'skyscraper': 2}
        b_type = type_map.get(building_type, 0)

        # Effective elevation after flood
        eff_elevation = max(0, elevation - flood_level)
        
        # Prepare feature vector
        features = np.array([[eff_elevation, building_height, distance_to_water, b_type]])
        
        # Get predictions
        vuln_score = self.xgb_model.predict(features)[0]
        collapse_prob = self.lr_model.predict_proba(features)[0][1]

        # Get SHAP values
        shap_values = self.explainer.shap_values(features)
        
        # Format SHAP explanation
        feature_names = ['Effective Elevation', 'Building Height', 'Distance to Water', 'Building Type']
        explanations = {}
        for i, name in enumerate(feature_names):
            explanations[name] = float(shap_values[0][i])

        # Normalize vulnerability
        vuln_score = min(max(float(vuln_score), 0.0), 100.0)

        return {
            "vulnerability_score": vuln_score,
            "collapse_probability": float(collapse_prob),
            "shap_explanation": explanations
        }

# Global instance
predictor = FloodRiskPredictor()
