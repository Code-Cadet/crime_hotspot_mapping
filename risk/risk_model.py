import numpy as np
import random
from typing import Tuple, List, Dict, Any
from scipy.spatial.distance import cdist
from config import RTM_CONFIG, ROYSAMBU_BOUNDS

class RiskTerrainModel:
    def __init__(self, grid_size: Tuple[int, int]):
        self.grid_size = grid_size
        self.risk_grid = np.zeros(grid_size)
        self.poi_density = np.zeros(grid_size)
        self.road_proximity = np.zeros(grid_size)
        self.lighting_score = np.zeros(grid_size)
        self.landuse_score = np.zeros(grid_size)
        
        # Generate synthetic environmental data
        self._generate_poi_data()
        self._generate_road_network()
        self._generate_lighting_data()
        self._generate_landuse_data()
        
        # Calculate initial risk scores
        self._calculate_risk_scores()
    
    def _generate_poi_data(self):
        """Generate synthetic Points of Interest data"""
        # Create high-density areas (commercial zones)
        commercial_centers = [
            (25, 25), (75, 25), (25, 75), (75, 75), (50, 50)
        ]
        
        for center_x, center_y in commercial_centers:
            radius = random.randint(8, 15)
            for x in range(max(0, center_x - radius), min(self.grid_size[0], center_x + radius)):
                for y in range(max(0, center_y - radius), min(self.grid_size[1], center_y + radius)):
                    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if distance <= radius:
                        self.poi_density[x, y] += random.uniform(0.3, 0.8) * (1 - distance/radius)
        
        # Add some random POIs
        for _ in range(20):
            x, y = random.randint(0, self.grid_size[0]-1), random.randint(0, self.grid_size[1]-1)
            self.poi_density[x, y] += random.uniform(0.1, 0.5)
        
        # Normalize POI density
        self.poi_density = np.clip(self.poi_density, 0, 1)
    
    def _generate_road_network(self):
        """Generate synthetic road network"""
        # Main roads (horizontal and vertical)
        main_roads = [
            (0, 25, self.grid_size[0], 25),  # Horizontal road
            (0, 75, self.grid_size[0], 75),  # Horizontal road
            (25, 0, 25, self.grid_size[1]),  # Vertical road
            (75, 0, 75, self.grid_size[1]),  # Vertical road
        ]
        
        # Secondary roads
        secondary_roads = [
            (0, 50, self.grid_size[0], 50),
            (50, 0, 50, self.grid_size[1]),
        ]
        
        all_roads = main_roads + secondary_roads
        
        # Calculate proximity to roads
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                min_distance = float('inf')
                for road in all_roads:
                    # Calculate distance to road segment
                    road_x1, road_y1, road_x2, road_y2 = road
                    distance = self._point_to_line_distance(x, y, road_x1, road_y1, road_x2, road_y2)
                    min_distance = min(min_distance, distance)
                
                # Convert distance to proximity score (closer = higher score)
                self.road_proximity[x, y] = max(0, 1 - min_distance / 20)
    
    def _point_to_line_distance(self, px, py, x1, y1, x2, y2):
        """Calculate distance from point to line segment"""
        A = px - x1
        B = py - y1
        C = x2 - x1
        D = y2 - y1
        
        dot = A * C + B * D
        len_sq = C * C + D * D
        
        if len_sq == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        
        param = dot / len_sq
        
        if param < 0:
            xx, yy = x1, y1
        elif param > 1:
            xx, yy = x2, y2
        else:
            xx = x1 + param * C
            yy = y1 + param * D
        
        return np.sqrt((px - xx)**2 + (py - yy)**2)
    
    def _generate_lighting_data(self):
        """Generate synthetic lighting data"""
        # Areas with good lighting (near main roads and commercial areas)
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                # Higher lighting near roads
                road_factor = self.road_proximity[x, y] * 0.6
                
                # Higher lighting in commercial areas
                poi_factor = self.poi_density[x, y] * 0.4
                
                # Add some randomness
                random_factor = random.uniform(0.1, 0.3)
                
                self.lighting_score[x, y] = min(1.0, road_factor + poi_factor + random_factor)
    
    def _generate_landuse_data(self):
        """Generate synthetic land use data"""
        # Create different land use zones
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                # Commercial areas (high risk)
                if self.poi_density[x, y] > 0.5:
                    self.landuse_score[x, y] = random.uniform(0.7, 1.0)
                # Residential areas (medium risk)
                elif self.road_proximity[x, y] > 0.3:
                    self.landuse_score[x, y] = random.uniform(0.4, 0.7)
                # Industrial/abandoned areas (high risk)
                elif random.random() < 0.1:
                    self.landuse_score[x, y] = random.uniform(0.8, 1.0)
                # Open spaces (low risk)
                else:
                    self.landuse_score[x, y] = random.uniform(0.1, 0.4)
    
    def _calculate_risk_scores(self):
        """Calculate composite risk scores"""
        self.risk_grid = (
            RTM_CONFIG["poi_weight"] * self.poi_density +
            RTM_CONFIG["road_weight"] * self.road_proximity +
            RTM_CONFIG["lighting_weight"] * (1 - self.lighting_score) +  # Lower lighting = higher risk
            RTM_CONFIG["landuse_weight"] * self.landuse_score
        )
        
        # Apply spatial decay
        self._apply_spatial_decay()
        
        # Normalize to [0, 1]
        self.risk_grid = np.clip(self.risk_grid, 0, 1)
    
    def _apply_spatial_decay(self):
        """Apply spatial decay to risk scores"""
        decay_kernel = np.array([
            [0.1, 0.2, 0.1],
            [0.2, 1.0, 0.2],
            [0.1, 0.2, 0.1]
        ])
        
        from scipy.ndimage import convolve
        self.risk_grid = convolve(self.risk_grid, decay_kernel, mode='constant', cval=0)
    
    def get_risk_score(self, position: Tuple[int, int]) -> float:
        """Get risk score for a specific position"""
        x, y = position
        if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
            return float(self.risk_grid[x, y])
        return 0.0
    
    def get_risk_grid(self) -> np.ndarray:
        """Get the complete risk grid"""
        return self.risk_grid.copy()
    
    def update_risk_factors(self, recent_events: List[Dict[str, Any]]):
        """Update risk factors based on recent events"""
        if not recent_events:
            return
        
        # Create event density map
        event_density = np.zeros(self.grid_size)
        
        for event in recent_events:
            if "position" in event:
                x, y = event["position"]
                if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                    # Add event impact with spatial decay
                    for dx in range(-3, 4):
                        for dy in range(-3, 4):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                                distance = np.sqrt(dx**2 + dy**2)
                                decay = np.exp(-distance * RTM_CONFIG["spatial_decay"])
                                event_density[nx, ny] += decay
        
        # Normalize event density
        if event_density.max() > 0:
            event_density = event_density / event_density.max()
        
        # Update risk grid with recent events
        event_weight = 0.3
        self.risk_grid = (1 - event_weight) * self.risk_grid + event_weight * event_density
        
        # Re-normalize
        self.risk_grid = np.clip(self.risk_grid, 0, 1)
    
    def get_risk_statistics(self) -> Dict[str, float]:
        """Get statistics about the risk grid"""
        return {
            "mean_risk": float(np.mean(self.risk_grid)),
            "std_risk": float(np.std(self.risk_grid)),
            "max_risk": float(np.max(self.risk_grid)),
            "min_risk": float(np.min(self.risk_grid)),
            "high_risk_areas": float(np.sum(self.risk_grid > 0.7) / self.risk_grid.size),
            "low_risk_areas": float(np.sum(self.risk_grid < 0.3) / self.risk_grid.size)
        }
    
    def export_risk_data(self, filename: str):
        """Export risk data to file"""
        import json
        
        risk_data = {
            "grid_size": self.grid_size,
            "risk_grid": self.risk_grid.tolist(),
            "poi_density": self.poi_density.tolist(),
            "road_proximity": self.road_proximity.tolist(),
            "lighting_score": self.lighting_score.tolist(),
            "landuse_score": self.landuse_score.tolist(),
            "statistics": self.get_risk_statistics(),
            "config": RTM_CONFIG
        }
        
        with open(filename, 'w') as f:
            json.dump(risk_data, f, indent=2) 