import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RISK_LAYERS_DIR = DATA_DIR / "risk_layers"

# Simulation parameters
SIMULATION_CONFIG = {
    "grid_size": (100, 100),  # 100x100 grid for Roysambu ward
    "episodes": 50,
    "steps_per_episode": 1000,
    "learning_rate": 0.1,
    "discount_factor": 0.95,
    "exploration_rate": 0.1,
    "reputation_decay": 0.99,
    "risk_threshold": 0.7
}

# Agent parameters
AGENT_CONFIG = {
    "offender_count": 20,
    "target_count": 50,
    "guardian_count": 10,
    "movement_range": 3,
    "vision_range": 5,
    "base_offense_probability": 0.3,
    "base_arrest_probability": 0.4
}

# Risk terrain modeling parameters
RTM_CONFIG = {
    "grid_resolution": 100,
    "poi_weight": 0.3,
    "road_weight": 0.25,
    "lighting_weight": 0.2,
    "landuse_weight": 0.25,
    "spatial_decay": 0.1
}

# Machine learning parameters
ML_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "feature_columns": [
        "risk_score", "hour", "weekday", "poi_density",
        "road_proximity", "lighting_score", "landuse_score",
        "offender_reputation", "guardian_reputation"
    ]
}

# Clustering parameters
CLUSTERING_CONFIG = {
    "kmeans_n_clusters": 5,
    "dbscan_eps": 0.1,
    "dbscan_min_samples": 5,
    "silhouette_threshold": 0.3
}

# Roysambu ward coordinates (approximate)
ROYSAMBU_BOUNDS = {
    "min_lat": -1.25,
    "max_lat": -1.20,
    "min_lon": 36.85,
    "max_lon": 36.90
}

# File naming conventions
FILE_PATTERNS = {
    "simulation_log": "simulation_episode_{episode}.json",
    "risk_grid": "risk_grid_{timestamp}.geojson",
    "cluster_results": "clusters_{method}_{timestamp}.json",
    "model_results": "model_training_results_{model}_{timestamp}.json",
    "processed_data": "processed_simulation_data_{timestamp}.csv"
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    "page_title": "Crime Hotspot Simulation - Roysambu Ward",
    "page_icon": "🚨",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "map_center": [-1.225, 36.875],
    "map_zoom": 14
}

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RISK_LAYERS_DIR]:
    directory.mkdir(parents=True, exist_ok=True) 