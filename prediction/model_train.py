import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from pathlib import Path
import joblib

from config import ML_CONFIG, PROCESSED_DATA_DIR, FILE_PATTERNS

class CrimePredictionModel:
    def __init__(self):
        self.models = {}
        self.model_results = {}
        self.feature_importance = {}
        self.best_model = None
        self.best_score = 0
        
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for model training"""
        # Select features for ML
        feature_columns = [
            "risk_score", "hour", "weekday", "reputation",
            "nearby_agent_count", "nearby_offenders", "nearby_guardians",
            "distance_from_center", "episode_progress",
            "is_night", "is_weekend", "is_high_risk", "is_low_risk"
        ]
        
        # Add rolling features if available
        rolling_features = [col for col in df.columns if col.startswith("rolling_")]
        feature_columns.extend(rolling_features)
        
        # Add grid-level features if available
        grid_features = [col for col in df.columns if col.startswith("is_crime_") or col.startswith("risk_score_")]
        feature_columns.extend(grid_features)
        
        # Add hourly features if available
        hourly_features = [col for col in df.columns if col.startswith("hourly_")]
        feature_columns.extend(hourly_features)
        
        # Filter available features
        available_features = [col for col in feature_columns if col in df.columns]
        
        # Prepare feature matrix
        X = df[available_features].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Create binary labels (crime occurred or not)
        y = df["is_crime"].astype(int)
        
        return X, y
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train Random Forest model"""
        print("Training Random Forest...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=ML_CONFIG["test_size"], 
            random_state=ML_CONFIG["random_state"], stratify=y
        )
        
        # Define parameter grid for hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Grid search with cross-validation
        rf = RandomForestClassifier(random_state=ML_CONFIG["random_state"])
        grid_search = GridSearchCV(
            rf, param_grid, cv=ML_CONFIG["cv_folds"], 
            scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_rf = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_rf.predict(X_test)
        y_pred_proba = best_rf.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Feature importance
        feature_importance = dict(zip(X.columns, best_rf.feature_importances_))
        
        results = {
            "model": "RandomForest",
            "best_params": grid_search.best_params_,
            "metrics": metrics,
            "feature_importance": feature_importance,
            "model_object": best_rf
        }
        
        self.models["RandomForest"] = best_rf
        self.model_results["RandomForest"] = results
        self.feature_importance["RandomForest"] = feature_importance
        
        return results
    
    def train_xgboost(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train XGBoost model"""
        print("Training XGBoost...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=ML_CONFIG["test_size"], 
            random_state=ML_CONFIG["random_state"], stratify=y
        )
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        # Grid search
        xgb_model = xgb.XGBClassifier(random_state=ML_CONFIG["random_state"])
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=ML_CONFIG["cv_folds"], 
            scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_xgb = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_xgb.predict(X_test)
        y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Feature importance
        feature_importance = dict(zip(X.columns, best_xgb.feature_importances_))
        
        results = {
            "model": "XGBoost",
            "best_params": grid_search.best_params_,
            "metrics": metrics,
            "feature_importance": feature_importance,
            "model_object": best_xgb
        }
        
        self.models["XGBoost"] = best_xgb
        self.model_results["XGBoost"] = results
        self.feature_importance["XGBoost"] = feature_importance
        
        return results
    
    def train_logistic_regression(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train Logistic Regression model"""
        print("Training Logistic Regression...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=ML_CONFIG["test_size"], 
            random_state=ML_CONFIG["random_state"], stratify=y
        )
        
        # Define parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        
        # Grid search
        lr = LogisticRegression(random_state=ML_CONFIG["random_state"])
        grid_search = GridSearchCV(
            lr, param_grid, cv=ML_CONFIG["cv_folds"], 
            scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_lr = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_lr.predict(X_test)
        y_pred_proba = best_lr.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Feature importance (coefficients)
        feature_importance = dict(zip(X.columns, np.abs(best_lr.coef_[0])))
        
        results = {
            "model": "LogisticRegression",
            "best_params": grid_search.best_params_,
            "metrics": metrics,
            "feature_importance": feature_importance,
            "model_object": best_lr
        }
        
        self.models["LogisticRegression"] = best_lr
        self.model_results["LogisticRegression"] = results
        self.feature_importance["LogisticRegression"] = feature_importance
        
        return results
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive model metrics"""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }
        
        return metrics
    
    def compare_models(self) -> pd.DataFrame:
        """Compare performance of all trained models"""
        comparison_data = []
        
        for model_name, results in self.model_results.items():
            metrics = results["metrics"]
            comparison_data.append({
                "Model": model_name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-Score": metrics["f1_score"],
                "ROC-AUC": metrics["roc_auc"]
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Find best model
        best_idx = comparison_df["F1-Score"].idxmax()
        self.best_model = comparison_df.loc[best_idx, "Model"]
        self.best_score = comparison_df.loc[best_idx, "F1-Score"]
        
        return comparison_df
    
    def plot_model_comparison(self, save_path: str = None) -> None:
        """Create visualization comparing model performance"""
        comparison_df = self.compare_models()
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # Bar plot of F1 scores
        axes[0, 0].bar(comparison_df["Model"], comparison_df["F1-Score"])
        axes[0, 0].set_title('F1-Score Comparison')
        axes[0, 0].set_ylabel('F1-Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Bar plot of ROC-AUC scores
        axes[0, 1].bar(comparison_df["Model"], comparison_df["ROC-AUC"])
        axes[0, 1].set_title('ROC-AUC Comparison')
        axes[0, 1].set_ylabel('ROC-AUC')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Precision vs Recall
        axes[1, 0].scatter(comparison_df["Precision"], comparison_df["Recall"], 
                          s=100, alpha=0.7)
        for i, model in enumerate(comparison_df["Model"]):
            axes[1, 0].annotate(model, (comparison_df["Precision"].iloc[i], 
                                       comparison_df["Recall"].iloc[i]))
        axes[1, 0].set_xlabel('Precision')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].set_title('Precision vs Recall')
        
        # Confusion matrices heatmap
        best_model_name = self.best_model
        best_model_results = self.model_results[best_model_name]
        cm = np.array(best_model_results["metrics"]["confusion_matrix"])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title(f'Confusion Matrix - {best_model_name}')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, model_name: str = None, top_n: int = 15, 
                              save_path: str = None) -> None:
        """Plot feature importance for a specific model"""
        if model_name is None:
            model_name = self.best_model
        
        if model_name not in self.feature_importance:
            print(f"Model {model_name} not found")
            return
        
        # Get feature importance
        importance_dict = self.feature_importance[model_name]
        
        # Sort by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        features, importance = zip(*top_features)
        y_pos = np.arange(len(features))
        
        ax.barh(y_pos, importance)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Feature Importance - {model_name}')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def save_models(self, model_dir: str = None) -> None:
        """Save trained models to disk"""
        if model_dir is None:
            model_dir = PROCESSED_DATA_DIR / "ml_models"
        
        model_dir = Path(model_dir)
        model_dir.mkdir(exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = model_dir / f"{model_name.lower()}_model.pkl"
            joblib.dump(model, model_path)
            print(f"Saved {model_name} to {model_path}")
        
        # Save model results
        results_path = model_dir / "model_results.json"
        export_results = {}
        
        for model_name, results in self.model_results.items():
            export_results[model_name] = {
                "model": results["model"],
                "best_params": results["best_params"],
                "metrics": results["metrics"],
                "feature_importance": results["feature_importance"]
            }
        
        with open(results_path, 'w') as f:
            json.dump(export_results, f, indent=2)
        
        print(f"Model results saved to {results_path}")
    
    def load_models(self, model_dir: str = None) -> None:
        """Load trained models from disk"""
        if model_dir is None:
            model_dir = PROCESSED_DATA_DIR / "ml_models"
        
        model_dir = Path(model_dir)
        
        # Load model objects
        for model_file in model_dir.glob("*_model.pkl"):
            model_name = model_file.stem.replace("_model", "").title()
            self.models[model_name] = joblib.load(model_file)
            print(f"Loaded {model_name} from {model_file}")
        
        # Load model results
        results_file = model_dir / "model_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                self.model_results = json.load(f)
            print(f"Loaded model results from {results_file}")
    
    def predict_crime_probability(self, features: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """Predict crime probability using trained model"""
        if model_name is None:
            model_name = self.best_model
        
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        predictions = model.predict_proba(features)[:, 1]
        
        return predictions
    
    def export_training_results(self, filename: str = None) -> str:
        """Export training results to JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_training_results_{timestamp}.json"
        
        export_data = {
            "training_timestamp": datetime.now().isoformat(),
            "model_comparison": self.compare_models().to_dict("records"),
            "best_model": self.best_model,
            "best_score": self.best_score,
            "model_results": {}
        }
        
        for model_name, results in self.model_results.items():
            export_data["model_results"][model_name] = {
                "best_params": results["best_params"],
                "metrics": results["metrics"],
                "top_features": dict(sorted(
                    results["feature_importance"].items(), 
                    key=lambda x: x[1], reverse=True
                )[:10])
            }
        
        filepath = PROCESSED_DATA_DIR / filename
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Training results exported to {filepath}")
        return str(filepath)
    
    def run_complete_training(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run complete model training pipeline"""
        print("Preparing training data...")
        X, y = self.prepare_training_data(df)
        
        print(f"Training data shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Train all models
        rf_results = self.train_random_forest(X, y)
        xgb_results = self.train_xgboost(X, y)
        lr_results = self.train_logistic_regression(X, y)
        
        # Compare models
        comparison_df = self.compare_models()
        print("\nModel Comparison:")
        print(comparison_df.to_string(index=False))
        
        print(f"\nBest model: {self.best_model} (F1-Score: {self.best_score:.4f})")
        
        # Save models and results
        self.save_models()
        self.export_training_results()
        
        return {
            "comparison": comparison_df,
            "best_model": self.best_model,
            "best_score": self.best_score,
            "model_results": self.model_results
        }
    
    def get_model_parameters(self):
        """Get model parameters for tuning"""
        rf_params = {
            'randomforestclassifier__n_estimators': [50, 100, 200],
            'randomforestclassifier__max_depth': [10, 20, None],
            'randomforestclassifier__min_samples_split': [2, 5, 10],
            'randomforestclassifier__min_samples_leaf': [1, 2, 4]
        }
        
        xgb_params = {
            'xgbclassifier__n_estimators': [50, 100, 200],
            'xgbclassifier__max_depth': [3, 6, 9],
            'xgbclassifier__learning_rate': [0.01, 0.1, 0.2],
            'xgbclassifier__subsample': [0.8, 0.9, 1.0]
        }
        
        logistic_params = {
            'logisticregression__max_iter': [1000],  # Increased from default
            'logisticregression__C': [0.1, 1.0, 10.0],
            'logisticregression__class_weight': ['balanced'],
            'logisticregression__solver': ['saga']  # Better for imbalanced data
        }
        
        return {
            "RandomForest": rf_params,
            "XGBoost": xgb_params,
            "LogisticRegression": logistic_params
        }

if __name__ == "__main__":
    # Example usage
    predictor = CrimePredictionModel()
    
    # Load processed data (assuming it exists)
    from src.data.preprocessing import CrimeDataPreprocessor
    
    preprocessor = CrimeDataPreprocessor()
    processed_df = preprocessor.process_simulation_data()
    
    if not processed_df.empty:
        results = predictor.run_complete_training(processed_df)
        print("Model training completed!")
        
        # Create visualizations
        predictor.plot_model_comparison()
        predictor.plot_feature_importance()