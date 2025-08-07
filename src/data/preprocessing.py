import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
import geopandas as gpd
from shapely.geometry import Point

from config import PROCESSED_DATA_DIR, ML_CONFIG, ROYSAMBU_BOUNDS

class CrimeDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.processed_data = None
        
    def load_simulation_data(self, episode_files: List[str] = None) -> pd.DataFrame:
        """Load and combine simulation data from multiple episodes"""
        if episode_files is None:
            # Load all episode files
            episode_files = list(PROCESSED_DATA_DIR.glob("simulation_episode_*.json"))
        
        all_events = []
        
        for episode_file in episode_files:
            try:
                with open(episode_file, 'r') as f:
                    episode_data = json.load(f)
                
                events = episode_data.get("events", [])
                episode_num = episode_data.get("episode", 0)
                
                for event in events:
                    event["episode"] = episode_num
                    all_events.append(event)
                    
            except Exception as e:
                print(f"Error loading {episode_file}: {e}")
        
        df = pd.DataFrame(all_events)
        print(f"Loaded {len(df)} events from {len(episode_files)} episodes")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw simulation data"""
        if df.empty:
            return df
        
        # Time-based features
        df["hour"] = (df["step"] // 60) % 24
        df["weekday"] = (df["step"] // (60 * 24)) % 7
        df["time_period"] = df["hour"].apply(self._categorize_time_period)
        df["is_night"] = (df["hour"] >= 22) | (df["hour"] <= 6)
        df["is_weekend"] = df["weekday"].isin([5, 6])
        
        # Spatial features
        df["x_coord"] = df["position"].apply(lambda x: x[0] if isinstance(x, (list, tuple)) else 0)
        df["y_coord"] = df["position"].apply(lambda x: x[1] if isinstance(x, (list, tuple)) else 0)
        df["distance_from_center"] = np.sqrt((df["x_coord"] - 50)**2 + (df["y_coord"] - 50)**2)
        
        # Agent interaction features
        df["nearby_agent_count"] = df["nearby_agents"].apply(len)
        df["nearby_offenders"] = df["nearby_agents"].apply(
            lambda x: len([a for a in x if a.startswith("offender")])
        )
        df["nearby_guardians"] = df["nearby_agents"].apply(
            lambda x: len([a for a in x if a.startswith("guardian")])
        )
        df["nearby_targets"] = df["nearby_agents"].apply(
            lambda x: len([a for a in x if a.startswith("target")])
        )
        
        # Action-based features
        df["is_crime"] = df["action_result"].isin(["successful_assault", "failed_assault"])
        df["is_arrest"] = df["action_result"].isin(["successful_arrest", "failed_arrest"])
        df["is_successful"] = df["action_result"].isin(["successful_assault", "successful_arrest"])
        df["is_movement"] = df["action"].isin(["move", "evade", "patrol"])
        
        # Reputation features
        df["reputation_level"] = pd.cut(df["reputation"], bins=3, labels=["low", "medium", "high"])
        
        # Risk-based features
        df["risk_level"] = pd.cut(df["risk_score"], bins=5, labels=["very_low", "low", "medium", "high", "very_high"])
        df["is_high_risk"] = df["risk_score"] > 0.7
        df["is_low_risk"] = df["risk_score"] < 0.3
        
        # Episode-based features
        df["episode_progress"] = df["step"] / 1000  # Assuming 1000 steps per episode
        
        return df
    
    def _categorize_time_period(self, hour: int) -> str:
        """Categorize hours into time periods"""
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 22:
            return "evening"
        else:
            return "night"
    
    def create_spatial_features(self, df: pd.DataFrame, grid_size: Tuple[int, int] = (100, 100)) -> pd.DataFrame:
        """Create spatial aggregation features"""
        # Create grid-based features
        df["grid_x"] = (df["x_coord"] // 10) * 10  # 10x10 grid cells
        df["grid_y"] = (df["y_coord"] // 10) * 10
        df["grid_id"] = df["grid_x"].astype(str) + "_" + df["grid_y"].astype(str)
        
        # Calculate grid-level statistics
        grid_stats = df.groupby("grid_id").agg({
            "is_crime": ["count", "sum", "mean"],
            "is_arrest": ["count", "sum", "mean"],
            "risk_score": ["mean", "std", "max"],
            "reputation": ["mean", "std"],
            "nearby_agent_count": ["mean", "max"]
        }).round(3)
        
        grid_stats.columns = ["_".join(col).strip() for col in grid_stats.columns]
        grid_stats = grid_stats.reset_index()
        
        # Merge back to original dataframe
        df = df.merge(grid_stats, on="grid_id", how="left")
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal aggregation features"""
        # Hourly statistics
        hourly_stats = df.groupby("hour").agg({
            "is_crime": ["count", "sum", "mean"],
            "is_arrest": ["count", "sum", "mean"],
            "risk_score": ["mean", "std"],
            "reputation": ["mean", "std"]
        }).round(3)
        
        hourly_stats.columns = ["hourly_" + "_".join(col).strip() for col in hourly_stats.columns]
        hourly_stats = hourly_stats.reset_index()
        
        # Merge back to original dataframe
        df = df.merge(hourly_stats, on="hour", how="left")
        
        # Rolling statistics (last 10 events per agent)
        df = df.sort_values(["agent_id", "step"])
        df["rolling_crime_rate"] = df.groupby("agent_id")["is_crime"].rolling(10, min_periods=1).mean().reset_index(0, drop=True)
        df["rolling_risk_avg"] = df.groupby("agent_id")["risk_score"].rolling(10, min_periods=1).mean().reset_index(0, drop=True)
        
        return df
    
    def prepare_ml_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and labels for machine learning"""
        # Select features for ML
        feature_columns = [
            "risk_score", "hour", "weekday", "reputation",
            "nearby_agent_count", "nearby_offenders", "nearby_guardians",
            "distance_from_center", "episode_progress",
            "is_night", "is_weekend", "is_high_risk", "is_low_risk",
            "rolling_crime_rate", "rolling_risk_avg"
        ]
        
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
    
    def scale_features(self, X: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """Scale features using StandardScaler"""
        if fit_scaler:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        return X_scaled
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = 20) -> pd.DataFrame:
        """Select top k features using statistical tests"""
        self.feature_selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        selected_features = X.columns[self.feature_selector.get_support()].tolist()
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        return X_selected_df
    
    def create_geospatial_data(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Convert to GeoDataFrame for spatial analysis"""
        # Convert grid coordinates to lat/lon (approximate for Roysambu)
        lat_range = ROYSAMBU_BOUNDS["max_lat"] - ROYSAMBU_BOUNDS["min_lat"]
        lon_range = ROYSAMBU_BOUNDS["max_lon"] - ROYSAMBU_BOUNDS["min_lon"]
        
        df["latitude"] = ROYSAMBU_BOUNDS["min_lat"] + (df["y_coord"] / 100) * lat_range
        df["longitude"] = ROYSAMBU_BOUNDS["min_lon"] + (df["x_coord"] / 100) * lon_range
        
        # Create Point geometries
        geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        
        return gdf
    
    def export_processed_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """Export processed data to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processed_simulation_data_{timestamp}.csv"
        
        filepath = PROCESSED_DATA_DIR / filename
        df.to_csv(filepath, index=False)
        
        print(f"Processed data exported to {filepath}")
        return str(filepath)
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for the dataset"""
        summary = {
            "total_events": len(df),
            "unique_agents": df["agent_id"].nunique(),
            "episodes": df["episode"].nunique(),
            "date_range": {
                "start": df["step"].min(),
                "end": df["step"].max()
            },
            "crime_events": df["is_crime"].sum(),
            "arrest_events": df["is_arrest"].sum(),
            "agent_type_distribution": df["agent_type"].value_counts().to_dict(),
            "time_period_distribution": df["time_period"].value_counts().to_dict(),
            "risk_score_stats": {
                "mean": df["risk_score"].mean(),
                "std": df["risk_score"].std(),
                "min": df["risk_score"].min(),
                "max": df["risk_score"].max()
            },
            "reputation_stats": {
                "mean": df["reputation"].mean(),
                "std": df["reputation"].std(),
                "min": df["reputation"].min(),
                "max": df["reputation"].max()
            }
        }
        
        return summary
    
    def process_simulation_data(self, episode_files: List[str] = None) -> pd.DataFrame:
        """Complete data processing pipeline"""
        print("Loading simulation data...")
        df = self.load_simulation_data(episode_files)
        
        if df.empty:
            print("No data to process")
            return df
        
        print("Engineering features...")
        df = self.engineer_features(df)
        
        print("Creating spatial features...")
        df = self.create_spatial_features(df)
        
        print("Creating temporal features...")
        df = self.create_temporal_features(df)
        
        print("Generating data summary...")
        summary = self.get_data_summary(df)
        print(f"Data summary: {summary}")
        
        self.processed_data = df
        return df

if __name__ == "__main__":
    preprocessor = CrimeDataPreprocessor()
    processed_df = preprocessor.process_simulation_data()
    
    if not processed_df.empty:
        # Export processed data
        preprocessor.export_processed_data(processed_df)
        
        # Prepare ML features
        X, y = preprocessor.prepare_ml_features(processed_df)
        print(f"ML features shape: {X.shape}")
        print(f"Target distribution: {y.value_counts()}") 