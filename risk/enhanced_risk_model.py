import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

# Try to import TensorFlow, but don't fail if it's not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow is available - enhanced ML features enabled")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - using base risk model only")

from .risk_model import RiskTerrainModel
from config import ML_CONFIG, PROCESSED_DATA_DIR

class EnhancedRiskModel:
    def __init__(self, grid_size: Tuple[int, int], use_ml: bool = True):
        self.grid_size = grid_size
        self.use_ml = use_ml and TENSORFLOW_AVAILABLE
        self.base_risk_model = RiskTerrainModel(grid_size)
        self.scaler = StandardScaler()
        self.lstm_model = None
        self.cnn_model = None
        self.temporal_data = []
        self.spatial_data = []
        
        if not TENSORFLOW_AVAILABLE and use_ml:
            print("Warning: TensorFlow not available. Enhanced ML features will be disabled.")
            print("The model will use only the base risk terrain model.")
        
    def prepare_temporal_data(self, simulation_events: List[Dict[str, Any]], 
                            sequence_length: int = 24) -> np.ndarray:
        """Prepare temporal data for LSTM model"""
        if not simulation_events:
            return np.array([])
        
        # Group events by hour
        hourly_data = {}
        for event in simulation_events:
            hour = (event.get("step", 0) // 60) % 24
            if hour not in hourly_data:
                hourly_data[hour] = []
            hourly_data[hour].append(event)
        
        # Create temporal features
        temporal_features = []
        for hour in range(24):
            events = hourly_data.get(hour, [])
            
            # Calculate features for this hour
            crime_count = len([e for e in events if e.get("action_result") in ["successful_assault", "failed_assault"]])
            arrest_count = len([e for e in events if e.get("action_result") in ["successful_arrest", "failed_arrest"]])
            avg_risk = np.mean([e.get("risk_score", 0) for e in events]) if events else 0
            avg_reputation = np.mean([e.get("reputation", 0.5) for e in events]) if events else 0.5
            
            temporal_features.append([
                crime_count,
                arrest_count,
                avg_risk,
                avg_reputation,
                hour,  # Time feature
                (hour // 6) % 4  # Time period (morning, afternoon, evening, night)
            ])
        
        # Create sequences for LSTM
        sequences = []
        for i in range(len(temporal_features) - sequence_length + 1):
            sequence = temporal_features[i:i + sequence_length]
            sequences.append(sequence)
        
        return np.array(sequences)
    
    def prepare_spatial_data(self, simulation_events: List[Dict[str, Any]]) -> np.ndarray:
        """Prepare spatial data for CNN model"""
        if not simulation_events:
            return np.array([])
        
        # Create spatial event density maps
        spatial_maps = []
        
        for event in simulation_events:
            if "position" in event:
                x, y = event["position"]
                if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                    # Create event map
                    event_map = np.zeros(self.grid_size)
                    event_map[x, y] = 1.0
                    
                    # Add risk context
                    risk_context = self.base_risk_model.get_risk_grid()
                    
                    # Combine event and risk data
                    combined_map = np.stack([
                        event_map,
                        risk_context,
                        self.base_risk_model.poi_density,
                        self.base_risk_model.road_proximity,
                        1 - self.base_risk_model.lighting_score  # Inverted lighting (darkness = risk)
                    ], axis=-1)
                    
                    spatial_maps.append(combined_map)
        
        return np.array(spatial_maps)
    
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> Optional[Any]:
        """Build LSTM model for temporal prediction"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available - LSTM model cannot be built")
            return None
        
        try:
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
        except Exception as e:
            print(f"Error building LSTM model: {e}")
            return None
    
    def build_cnn_model(self, input_shape: Tuple[int, int, int]) -> Optional[Any]:
        """Build CNN model for spatial prediction"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available - CNN model cannot be built")
            return None
        
        try:
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                Flatten(),
                Dense(64, activation='relu'),
                Dropout(0.5),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
        except Exception as e:
            print(f"Error building CNN model: {e}")
            return None
    
    def train_models(self, simulation_data: List[Dict[str, Any]]):
        """Train LSTM and CNN models on simulation data"""
        if not self.use_ml:
            print("ML features disabled - skipping model training")
            return
        
        print("Preparing training data...")
        
        # Prepare temporal data
        temporal_data = self.prepare_temporal_data(simulation_data)
        if len(temporal_data) > 0:
            print(f"Temporal data shape: {temporal_data.shape}")
            
            # Prepare labels (next hour crime prediction)
            labels = []
            for i in range(len(temporal_data)):
                if i + 1 < len(temporal_data):
                    next_hour_crimes = temporal_data[i + 1, -1, 0]  # Crime count from next hour
                    labels.append(1 if next_hour_crimes > 0 else 0)
                else:
                    labels.append(0)
            
            labels = np.array(labels)
            
            # Split data
            if len(temporal_data) > 10:
                X_train, X_test, y_train, y_test = train_test_split(
                    temporal_data[:-1], labels, test_size=0.2, random_state=ML_CONFIG["random_state"]
                )
                
                # Train LSTM model
                self.lstm_model = self.build_lstm_model((temporal_data.shape[1], temporal_data.shape[2]))
                if self.lstm_model:
                    print("Training LSTM model...")
                    try:
                        self.lstm_model.fit(
                            X_train, y_train,
                            epochs=50,
                            batch_size=32,
                            validation_data=(X_test, y_test),
                            verbose=1
                        )
                    except Exception as e:
                        print(f"Error training LSTM model: {e}")
                        self.lstm_model = None
        
        # Prepare spatial data
        spatial_data = self.prepare_spatial_data(simulation_data)
        if len(spatial_data) > 0:
            print(f"Spatial data shape: {spatial_data.shape}")
            
            # Create spatial labels (high risk areas)
            spatial_labels = []
            for map_data in spatial_data:
                # Label as high risk if event occurred in high-risk area
                event_map = map_data[:, :, 0]
                risk_map = map_data[:, :, 1]
                high_risk_event = np.any((event_map > 0) & (risk_map > 0.7))
                spatial_labels.append(1 if high_risk_event else 0)
            
            spatial_labels = np.array(spatial_labels)
            
            # Split spatial data
            if len(spatial_data) > 10:
                X_train, X_test, y_train, y_test = train_test_split(
                    spatial_data, spatial_labels, test_size=0.2, random_state=ML_CONFIG["random_state"]
                )
                
                # Train CNN model
                self.cnn_model = self.build_cnn_model((spatial_data.shape[1], spatial_data.shape[2], spatial_data.shape[3]))
                if self.cnn_model:
                    print("Training CNN model...")
                    try:
                        self.cnn_model.fit(
                            X_train, y_train,
                            epochs=30,
                            batch_size=16,
                            validation_data=(X_test, y_test),
                            verbose=1
                        )
                    except Exception as e:
                        print(f"Error training CNN model: {e}")
                        self.cnn_model = None
    
    def predict_risk(self, current_state: Dict[str, Any]) -> Dict[str, float]:
        """Predict risk using both base model and ML models"""
        base_risk = self.base_risk_model.get_risk_score(current_state.get("position", (0, 0)))
        
        prediction_result = {
            "base_risk": base_risk,
            "enhanced_risk": base_risk,
            "temporal_factor": 1.0,
            "spatial_factor": 1.0
        }
        
        if not self.use_ml or (self.lstm_model is None and self.cnn_model is None):
            return prediction_result
        
        # Temporal prediction
        if self.lstm_model and "temporal_context" in current_state:
            try:
                temporal_input = np.array([current_state["temporal_context"]])
                temporal_prediction = self.lstm_model.predict(temporal_input, verbose=0)[0][0]
                prediction_result["temporal_factor"] = float(temporal_prediction)
            except Exception as e:
                print(f"Error in temporal prediction: {e}")
        
        # Spatial prediction
        if self.cnn_model and "spatial_context" in current_state:
            try:
                spatial_input = np.array([current_state["spatial_context"]])
                spatial_prediction = self.cnn_model.predict(spatial_input, verbose=0)[0][0]
                prediction_result["spatial_factor"] = float(spatial_prediction)
            except Exception as e:
                print(f"Error in spatial prediction: {e}")
        
        # Combine predictions
        prediction_result["enhanced_risk"] = (
            base_risk * 0.4 +
            prediction_result["temporal_factor"] * 0.3 +
            prediction_result["spatial_factor"] * 0.3
        )
        
        return prediction_result
    
    def update_with_events(self, recent_events: List[Dict[str, Any]]):
        """Update models with recent events"""
        # Update base risk model
        self.base_risk_model.update_risk_factors(recent_events)
        
        # Retrain ML models periodically
        if len(self.temporal_data) > 1000:  # Retrain every 1000 events
            self.train_models(self.temporal_data)
            self.temporal_data = []
    
    def save_models(self, model_dir: str = None):
        """Save trained models"""
        if model_dir is None:
            model_dir = PROCESSED_DATA_DIR / "ml_models"
        
        model_dir = Path(model_dir)
        model_dir.mkdir(exist_ok=True)
        
        if self.lstm_model and TENSORFLOW_AVAILABLE:
            try:
                self.lstm_model.save(model_dir / "lstm_model.h5")
                print("LSTM model saved")
            except Exception as e:
                print(f"Error saving LSTM model: {e}")
        
        if self.cnn_model and TENSORFLOW_AVAILABLE:
            try:
                self.cnn_model.save(model_dir / "cnn_model.h5")
                print("CNN model saved")
            except Exception as e:
                print(f"Error saving CNN model: {e}")
        
        # Save scaler
        try:
            joblib.dump(self.scaler, model_dir / "scaler.pkl")
            print("Scaler saved")
        except Exception as e:
            print(f"Error saving scaler: {e}")
        
        print(f"Models saved to {model_dir}")
    
    def load_models(self, model_dir: str = None):
        """Load trained models"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available - cannot load ML models")
            return
        
        if model_dir is None:
            model_dir = PROCESSED_DATA_DIR / "ml_models"
        
        model_dir = Path(model_dir)
        
        if (model_dir / "lstm_model.h5").exists():
            try:
                self.lstm_model = tf.keras.models.load_model(model_dir / "lstm_model.h5")
                print("LSTM model loaded")
            except Exception as e:
                print(f"Error loading LSTM model: {e}")
        
        if (model_dir / "cnn_model.h5").exists():
            try:
                self.cnn_model = tf.keras.models.load_model(model_dir / "cnn_model.h5")
                print("CNN model loaded")
            except Exception as e:
                print(f"Error loading CNN model: {e}")
        
        if (model_dir / "scaler.pkl").exists():
            try:
                self.scaler = joblib.load(model_dir / "scaler.pkl")
                print("Scaler loaded")
            except Exception as e:
                print(f"Error loading scaler: {e}")
        
        print(f"Models loaded from {model_dir}")
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for trained models"""
        performance = {
            "lstm_available": self.lstm_model is not None,
            "cnn_available": self.cnn_model is not None,
            "tensorflow_available": TENSORFLOW_AVAILABLE,
            "base_risk_stats": self.base_risk_model.get_risk_statistics()
        }
        
        return performance 