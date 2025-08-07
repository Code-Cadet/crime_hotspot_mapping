import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulation.agents import OffenderAgent, TargetAgent, GuardianAgent, BaseAgent
from simulation.simulator import CrimeHotspotSimulator
from risk.risk_model import RiskTerrainModel
from clustering.cluster_analysis import CrimeClusterAnalyzer
from prediction.model_train import CrimePredictionModel
from src.data.preprocessing import CrimeDataPreprocessor

class TestAgents(unittest.TestCase):
    def setUp(self):
        self.offender = OffenderAgent("offender_1", (10, 10))
        self.target = TargetAgent("target_1", (20, 20))
        self.guardian = GuardianAgent("guardian_1", (15, 15))
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        self.assertEqual(self.offender.agent_id, "offender_1")
        self.assertEqual(self.offender.position, (10, 10))
        self.assertEqual(self.offender.agent_type, "offender")
        self.assertIsInstance(self.offender.reputation, float)
        self.assertIsInstance(self.offender.q_table, dict)
    
    def test_agent_movement(self):
        """Test agent movement within grid bounds"""
        original_pos = self.offender.position
        new_pos = self.offender.move((100, 100))
        
        self.assertIsInstance(new_pos, tuple)
        self.assertEqual(len(new_pos), 2)
        self.assertTrue(0 <= new_pos[0] < 100)
        self.assertTrue(0 <= new_pos[1] < 100)
    
    def test_offender_decision_making(self):
        """Test offender decision making"""
        environment_state = {
            "risk_score": 0.5,
            "nearby_agents": ["target_1", "guardian_1"],
            "position": (10, 10),
            "grid_size": (100, 100)
        }
        
        result = self.offender.decide_action(environment_state)
        
        self.assertIn("action", result)
        self.assertIn("action_result", result)
        self.assertIn("reward", result)
        self.assertIn("new_position", result)
        self.assertIsInstance(result["reward"], float)
    
    def test_target_decision_making(self):
        """Test target decision making"""
        environment_state = {
            "risk_score": 0.7,
            "nearby_agents": ["offender_1"],
            "position": (20, 20),
            "grid_size": (100, 100)
        }
        
        result = self.target.decide_action(environment_state)
        
        self.assertIn("action", result)
        self.assertIn("action_result", result)
        self.assertIn("reward", result)
        self.assertIn("new_position", result)
    
    def test_guardian_decision_making(self):
        """Test guardian decision making"""
        environment_state = {
            "risk_score": 0.6,
            "nearby_agents": ["offender_1"],
            "position": (15, 15),
            "grid_size": (100, 100)
        }
        
        result = self.guardian.decide_action(environment_state)
        
        self.assertIn("action", result)
        self.assertIn("action_result", result)
        self.assertIn("reward", result)
        self.assertIn("new_position", result)
    
    def test_reputation_update(self):
        """Test reputation update mechanism"""
        original_reputation = self.offender.reputation
        self.offender.update_reputation(0.1)
        
        self.assertNotEqual(self.offender.reputation, original_reputation)
        self.assertTrue(0.0 <= self.offender.reputation <= 1.0)

class TestRiskModel(unittest.TestCase):
    def setUp(self):
        self.risk_model = RiskTerrainModel((50, 50))
    
    def test_risk_model_initialization(self):
        """Test risk model initialization"""
        self.assertEqual(self.risk_model.grid_size, (50, 50))
        self.assertIsInstance(self.risk_model.risk_grid, np.ndarray)
        self.assertEqual(self.risk_model.risk_grid.shape, (50, 50))
    
    def test_risk_score_retrieval(self):
        """Test risk score retrieval for specific positions"""
        score = self.risk_model.get_risk_score((25, 25))
        self.assertIsInstance(score, float)
        self.assertTrue(0.0 <= score <= 1.0)
    
    def test_risk_grid_export(self):
        """Test risk grid export functionality"""
        stats = self.risk_model.get_risk_statistics()
        
        self.assertIn("mean_risk", stats)
        self.assertIn("std_risk", stats)
        self.assertIn("max_risk", stats)
        self.assertIn("min_risk", stats)
        self.assertIn("high_risk_areas", stats)
        self.assertIn("low_risk_areas", stats)
        
        for value in stats.values():
            self.assertIsInstance(value, float)
    
    def test_risk_factors_generation(self):
        """Test risk factors generation"""
        self.assertIsInstance(self.risk_model.poi_density, np.ndarray)
        self.assertIsInstance(self.risk_model.road_proximity, np.ndarray)
        self.assertIsInstance(self.risk_model.lighting_score, np.ndarray)
        self.assertIsInstance(self.risk_model.landuse_score, np.ndarray)
        
        # Check that all factors are within [0, 1] range
        for factor in [self.risk_model.poi_density, self.risk_model.road_proximity,
                      self.risk_model.lighting_score, self.risk_model.landuse_score]:
            self.assertTrue(np.all(factor >= 0))
            self.assertTrue(np.all(factor <= 1))

class TestSimulator(unittest.TestCase):
    def setUp(self):
        self.simulator = CrimeHotspotSimulator((20, 20))
    
    def test_simulator_initialization(self):
        """Test simulator initialization"""
        self.assertEqual(self.simulator.grid_size, (20, 20))
        self.assertIsInstance(self.simulator.agents, dict)
        self.assertIsInstance(self.simulator.risk_model, RiskTerrainModel)
    
    def test_agent_initialization(self):
        """Test agent initialization in simulator"""
        self.simulator.initialize_agents()
        
        # Check that agents were created
        self.assertGreater(len(self.simulator.agents), 0)
        
        # Check agent types
        agent_types = [agent.agent_type for agent in self.simulator.agents.values()]
        self.assertIn("offender", agent_types)
        self.assertIn("target", agent_types)
        self.assertIn("guardian", agent_types)
    
    def test_environment_state_generation(self):
        """Test environment state generation"""
        self.simulator.initialize_agents()
        agent_id = list(self.simulator.agents.keys())[0]
        agent = self.simulator.agents[agent_id]
        
        environment_state = self.simulator._get_environment_state(agent.position, agent_id)
        
        self.assertIn("risk_score", environment_state)
        self.assertIn("nearby_agents", environment_state)
        self.assertIn("position", environment_state)
        self.assertIn("grid_size", environment_state)
        
        self.assertIsInstance(environment_state["risk_score"], float)
        self.assertIsInstance(environment_state["nearby_agents"], list)
    
    def test_episode_statistics_calculation(self):
        """Test episode statistics calculation"""
        # Create mock events
        mock_events = [
            {"action_result": "successful_assault", "position": (10, 10), "risk_score": 0.7},
            {"action_result": "failed_assault", "position": (15, 15), "risk_score": 0.5},
            {"action_result": "successful_arrest", "position": (12, 12), "risk_score": 0.6},
            {"action_result": "moved", "position": (8, 8), "risk_score": 0.3}
        ]
        
        total_rewards = {"agent_1": 10.0, "agent_2": 5.0}
        
        stats = self.simulator._calculate_episode_stats(mock_events, total_rewards)
        
        self.assertIn("total_events", stats)
        self.assertIn("crime_events", stats)
        self.assertIn("arrest_events", stats)
        self.assertIn("successful_crimes", stats)
        self.assertIn("successful_arrests", stats)
        self.assertIn("crime_positions", stats)
        self.assertIn("arrest_positions", stats)
        self.assertIn("hourly_crime_distribution", stats)
        self.assertIn("total_rewards", stats)
        self.assertIn("average_risk_score", stats)
        self.assertIn("agent_type_counts", stats)

class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        self.preprocessor = CrimeDataPreprocessor()
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization"""
        self.assertIsInstance(self.preprocessor.scaler, object)
        self.assertIsNone(self.preprocessor.feature_selector)
        self.assertIsNone(self.preprocessor.processed_data)
    
    def test_feature_engineering(self):
        """Test feature engineering functionality"""
        # Create mock data
        mock_data = pd.DataFrame({
            "step": [60, 120, 180],
            "position": [(10, 10), (20, 20), (30, 30)],
            "agent_type": ["offender", "target", "guardian"],
            "action_result": ["successful_assault", "evaded", "patrolled"],
            "risk_score": [0.7, 0.5, 0.3],
            "reputation": [0.6, 0.8, 0.9],
            "nearby_agents": [["target_1"], ["offender_1"], []]
        })
        
        engineered_data = self.preprocessor.engineer_features(mock_data)
        
        # Check that new features were created
        expected_features = ["hour", "weekday", "time_period", "is_night", "is_weekend",
                           "x_coord", "y_coord", "distance_from_center", "nearby_agent_count",
                           "is_crime", "is_arrest", "is_successful", "is_movement"]
        
        for feature in expected_features:
            self.assertIn(feature, engineered_data.columns)
    
    def test_spatial_features_creation(self):
        """Test spatial features creation"""
        # Create mock data with coordinates
        mock_data = pd.DataFrame({
            "x_coord": [10, 20, 30, 40, 50],
            "y_coord": [10, 20, 30, 40, 50],
            "is_crime": [True, False, True, False, True],
            "is_arrest": [False, True, False, True, False],
            "risk_score": [0.7, 0.5, 0.8, 0.3, 0.9],
            "reputation": [0.6, 0.8, 0.4, 0.9, 0.5],
            "nearby_agent_count": [2, 1, 3, 0, 2]
        })
        
        spatial_data = self.preprocessor.create_spatial_features(mock_data)
        
        # Check that grid features were created
        self.assertIn("grid_x", spatial_data.columns)
        self.assertIn("grid_y", spatial_data.columns)
        self.assertIn("grid_id", spatial_data.columns)
    
    def test_ml_features_preparation(self):
        """Test ML features preparation"""
        # Create mock data with engineered features
        mock_data = pd.DataFrame({
            "risk_score": [0.7, 0.5, 0.8],
            "hour": [12, 18, 22],
            "weekday": [1, 3, 5],
            "reputation": [0.6, 0.8, 0.4],
            "nearby_agent_count": [2, 1, 3],
            "distance_from_center": [15.0, 25.0, 35.0],
            "is_crime": [True, False, True]
        })
        
        X, y = self.preprocessor.prepare_ml_features(mock_data)
        
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), len(y))
        self.assertGreater(len(X.columns), 0)

class TestClusteringAnalysis(unittest.TestCase):
    def setUp(self):
        self.analyzer = CrimeClusterAnalyzer()
    
    def test_analyzer_initialization(self):
        """Test cluster analyzer initialization"""
        self.assertIsInstance(self.analyzer.scaler, object)
        self.assertIsNone(self.analyzer.kmeans_model)
        self.assertIsNone(self.analyzer.dbscan_model)
        self.assertEqual(self.analyzer.cluster_results, {})
    
    def test_clustering_data_preparation(self):
        """Test clustering data preparation"""
        # Create mock data
        mock_data = pd.DataFrame({
            "x_coord": [10, 20, 30, 40, 50],
            "y_coord": [10, 20, 30, 40, 50],
            "risk_score": [0.7, 0.5, 0.8, 0.3, 0.9],
            "reputation": [0.6, 0.8, 0.4, 0.9, 0.5],
            "nearby_agent_count": [2, 1, 3, 0, 2],
            "hour": [12, 18, 22, 6, 14],
            "weekday": [1, 3, 5, 0, 2]
        })
        
        X_scaled, X_original = self.analyzer.prepare_clustering_data(mock_data)
        
        self.assertIsInstance(X_scaled, np.ndarray)
        self.assertIsInstance(X_original, pd.DataFrame)
        self.assertEqual(X_scaled.shape[0], X_original.shape[0])
    
    def test_kmeans_clustering(self):
        """Test K-means clustering"""
        # Create mock scaled data
        X_scaled = np.random.random((20, 7))
        
        results = self.analyzer.apply_kmeans_clustering(X_scaled, n_clusters=3)
        
        self.assertIn("method", results)
        self.assertIn("n_clusters", results)
        self.assertIn("cluster_labels", results)
        self.assertIn("cluster_centers", results)
        self.assertIn("silhouette_score", results)
        self.assertEqual(results["method"], "kmeans")
        self.assertEqual(results["n_clusters"], 3)
    
    def test_dbscan_clustering(self):
        """Test DBSCAN clustering"""
        # Create mock scaled data
        X_scaled = np.random.random((20, 7))
        
        results = self.analyzer.apply_dbscan_clustering(X_scaled, eps=0.5, min_samples=3)
        
        self.assertIn("method", results)
        self.assertIn("eps", results)
        self.assertIn("min_samples", results)
        self.assertIn("n_clusters", results)
        self.assertIn("n_noise", results)
        self.assertIn("cluster_labels", results)
        self.assertEqual(results["method"], "dbscan")
    
    def test_optimal_clusters_finding(self):
        """Test optimal number of clusters finding"""
        # Create mock scaled data
        X_scaled = np.random.random((30, 7))
        
        optimal_k = self.analyzer.find_optimal_kmeans_clusters(X_scaled, max_clusters=5)
        
        self.assertIn("k_range", optimal_k)
        self.assertIn("inertias", optimal_k)
        self.assertIn("silhouette_scores", optimal_k)
        self.assertIn("elbow_k", optimal_k)
        self.assertIn("best_silhouette_k", optimal_k)
        self.assertIn("recommended_k", optimal_k)

class TestModelTraining(unittest.TestCase):
    def setUp(self):
        self.predictor = CrimePredictionModel()
    
    def test_predictor_initialization(self):
        """Test model predictor initialization"""
        self.assertEqual(self.predictor.models, {})
        self.assertEqual(self.predictor.model_results, {})
        self.assertEqual(self.predictor.feature_importance, {})
        self.assertIsNone(self.predictor.best_model)
        self.assertEqual(self.predictor.best_score, 0)
    
    def test_training_data_preparation(self):
        """Test training data preparation"""
        # Create mock data
        mock_data = pd.DataFrame({
            "risk_score": [0.7, 0.5, 0.8, 0.3, 0.9],
            "hour": [12, 18, 22, 6, 14],
            "weekday": [1, 3, 5, 0, 2],
            "reputation": [0.6, 0.8, 0.4, 0.9, 0.5],
            "nearby_agent_count": [2, 1, 3, 0, 2],
            "distance_from_center": [15.0, 25.0, 35.0, 45.0, 55.0],
            "is_crime": [True, False, True, False, True]
        })
        
        X, y = self.predictor.prepare_training_data(mock_data)
        
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), len(y))
        self.assertGreater(len(X.columns), 0)
        self.assertTrue(all(y.isin([0, 1])))  # Binary labels
    
    def test_metrics_calculation(self):
        """Test metrics calculation"""
        y_true = pd.Series([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 1, 0])
        y_pred_proba = np.array([0.9, 0.1, 0.8, 0.7, 0.3])
        
        metrics = self.predictor._calculate_metrics(y_true, y_pred, y_pred_proba)
        
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertIn("confusion_matrix", metrics)
        
        for metric_name, value in metrics.items():
            if metric_name != "confusion_matrix":
                self.assertIsInstance(value, float)
                self.assertTrue(0.0 <= value <= 1.0)
    
    def test_model_comparison(self):
        """Test model comparison functionality"""
        # Create mock model results
        self.predictor.model_results = {
            "RandomForest": {
                "metrics": {
                    "accuracy": 0.85,
                    "precision": 0.82,
                    "recall": 0.78,
                    "f1_score": 0.80,
                    "roc_auc": 0.88
                }
            },
            "XGBoost": {
                "metrics": {
                    "accuracy": 0.87,
                    "precision": 0.84,
                    "recall": 0.80,
                    "f1_score": 0.82,
                    "roc_auc": 0.90
                }
            }
        }
        
        comparison_df = self.predictor.compare_models()
        
        self.assertIsInstance(comparison_df, pd.DataFrame)
        self.assertIn("Model", comparison_df.columns)
        self.assertIn("Accuracy", comparison_df.columns)
        self.assertIn("Precision", comparison_df.columns)
        self.assertIn("Recall", comparison_df.columns)
        self.assertIn("F1-Score", comparison_df.columns)
        self.assertIn("ROC-AUC", comparison_df.columns)
        self.assertEqual(len(comparison_df), 2)

if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2) 