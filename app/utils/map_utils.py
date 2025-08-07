import folium
from folium import plugins
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import geopandas as gpd
from shapely.geometry import Point
import json

from config import ROYSAMBU_BOUNDS, DASHBOARD_CONFIG

class MapVisualizer:
    def __init__(self):
        self.center = DASHBOARD_CONFIG["map_center"]
        self.zoom = DASHBOARD_CONFIG["map_zoom"]
        
    def create_base_map(self) -> folium.Map:
        """Create a base Folium map centered on Roysambu"""
        return folium.Map(
            location=self.center,
            zoom_start=self.zoom,
            tiles='OpenStreetMap'
        )
    
    def _convert_grid_to_latlon(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert grid coordinates to lat/lon coordinates"""
        if df.empty:
            return df
            
        # Check if we have grid coordinates
        if 'x_coord' in df.columns and 'y_coord' in df.columns:
            # Convert grid coordinates to lat/lon (approximate for Roysambu)
            lat_range = ROYSAMBU_BOUNDS["max_lat"] - ROYSAMBU_BOUNDS["min_lat"]
            lon_range = ROYSAMBU_BOUNDS["max_lon"] - ROYSAMBU_BOUNDS["min_lon"]
            
            df = df.copy()
            df["latitude"] = ROYSAMBU_BOUNDS["min_lat"] + (df["y_coord"] / 100) * lat_range
            df["longitude"] = ROYSAMBU_BOUNDS["min_lon"] + (df["x_coord"] / 100) * lon_range
            
        return df
    
    def add_heatmap(self, map_obj: folium.Map, data: pd.DataFrame, 
                   lat_col: str = 'latitude', lon_col: str = 'longitude',
                   weight_col: str = None, radius: int = 15) -> folium.Map:
        """Add a heatmap layer to the map"""
        if data.empty:
            return map_obj
        
        # Convert grid coordinates if needed
        data = self._convert_grid_to_latlon(data)
        
        # Prepare heatmap data
        heatmap_data = []
        for _, row in data.iterrows():
            if pd.notna(row[lat_col]) and pd.notna(row[lon_col]):
                if weight_col and weight_col in row:
                    weight = float(row[weight_col])
                else:
                    weight = 1.0
                heatmap_data.append([row[lat_col], row[lon_col], weight])
        
        if heatmap_data:
            plugins.HeatMap(
                heatmap_data,
                radius=radius,
                blur=15,
                max_zoom=13
            ).add_to(map_obj)
        
        return map_obj
    
    def add_cluster_markers(self, map_obj: folium.Map, cluster_data: List[Dict[str, Any]],
                           color_map: Dict[int, str] = None) -> folium.Map:
        """Add cluster markers to the map"""
        if not cluster_data:
            return map_obj
        
        if color_map is None:
            color_map = {
                0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 
                4: 'orange', 5: 'darkred', 6: 'lightred', 7: 'beige'
            }
        
        for cluster_info in cluster_data:
            if "spatial_center" in cluster_info:
                center = cluster_info["spatial_center"]
                
                # Convert grid coordinates to lat/lon
                lat_range = ROYSAMBU_BOUNDS["max_lat"] - ROYSAMBU_BOUNDS["min_lat"]
                lon_range = ROYSAMBU_BOUNDS["max_lon"] - ROYSAMBU_BOUNDS["min_lon"]
                
                lat = ROYSAMBU_BOUNDS["min_lat"] + (center["y"] / 100) * lat_range
                lon = ROYSAMBU_BOUNDS["min_lon"] + (center["x"] / 100) * lon_range
                
                cluster_id = cluster_info.get("cluster_id", 0)
                color = color_map.get(cluster_id % len(color_map), 'gray')
                
                # Create popup content
                popup_content = f"""
                <b>Cluster {cluster_id}</b><br>
                Crime Rate: {cluster_info.get('crime_rate', 0):.2f}<br>
                Size: {cluster_info.get('size', 0)}<br>
                Avg Risk: {cluster_info.get('avg_risk_score', 0):.2f}
                """
                
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=10,
                    popup=folium.Popup(popup_content, max_width=300),
                    color=color,
                    fill=True,
                    fillOpacity=0.7
                ).add_to(map_obj)
        
        return map_obj
    
    def add_risk_surface(self, map_obj: folium.Map, risk_grid: np.ndarray,
                        grid_size: Tuple[int, int] = (100, 100)) -> folium.Map:
        """Add risk surface overlay to the map"""
        if risk_grid is None:
            return map_obj
        
        # Convert risk grid to lat/lon coordinates
        lat_range = ROYSAMBU_BOUNDS["max_lat"] - ROYSAMBU_BOUNDS["min_lat"]
        lon_range = ROYSAMBU_BOUNDS["max_lon"] - ROYSAMBU_BOUNDS["min_lon"]
        
        # Sample points from the risk grid
        sample_step = 5  # Sample every 5th point to avoid overcrowding
        risk_points = []
        
        for i in range(0, grid_size[0], sample_step):
            for j in range(0, grid_size[1], sample_step):
                if i < risk_grid.shape[0] and j < risk_grid.shape[1]:
                    risk_value = risk_grid[i, j]
                    if risk_value > 0.3:  # Only show high-risk areas
                        lat = ROYSAMBU_BOUNDS["min_lat"] + (j / 100) * lat_range
                        lon = ROYSAMBU_BOUNDS["min_lon"] + (i / 100) * lon_range
                        risk_points.append([lat, lon, risk_value])
        
        if risk_points:
            # Create a heatmap for risk surface
            plugins.HeatMap(
                risk_points,
                radius=8,
                blur=10,
                max_zoom=13,
                gradient={0.3: 'yellow', 0.5: 'orange', 0.7: 'red', 1.0: 'darkred'}
            ).add_to(map_obj)
        
        return map_obj
    
    def add_agent_trajectories(self, map_obj: folium.Map, trajectory_data: List[Dict[str, Any]],
                             agent_type: str = None) -> folium.Map:
        """Add agent trajectory lines to the map"""
        if not trajectory_data:
            return map_obj
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(trajectory_data)
        df = self._convert_grid_to_latlon(df)
        
        # Filter by agent type if specified
        if agent_type and 'agent_type' in df.columns:
            df = df[df['agent_type'] == agent_type]
        
        # Group by agent and create trajectories
        if 'agent_id' in df.columns:
            for agent_id, agent_data in df.groupby('agent_id'):
                if len(agent_data) > 1:
                    # Sort by time step
                    agent_data = agent_data.sort_values('step')
                    
                    # Create trajectory line
                    trajectory_points = []
                    for _, row in agent_data.iterrows():
                        if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                            trajectory_points.append([row['latitude'], row['longitude']])
                    
                    if len(trajectory_points) > 1:
                        # Choose color based on agent type
                        color_map = {
                            'offender': 'red',
                            'target': 'blue', 
                            'guardian': 'green'
                        }
                        color = color_map.get(agent_data.iloc[0].get('agent_type', 'offender'), 'gray')
                        
                        folium.PolyLine(
                            trajectory_points,
                            color=color,
                            weight=2,
                            opacity=0.7,
                            popup=f"Agent {agent_id}"
                        ).add_to(map_obj)
        
        return map_obj
    
    def add_time_slider(self, map_obj: folium.Map, time_series_data: pd.DataFrame,
                       time_col: str = 'hour') -> folium.Map:
        """Add time-based filtering to the map"""
        if time_series_data.empty or time_col not in time_series_data.columns:
            return map_obj
        
        # This would require additional JavaScript for time slider functionality
        # For now, just add a note about time filtering
        folium.Element(
            """
            <div style="position: fixed; top: 10px; right: 10px; z-index: 1000; background: white; padding: 10px; border: 1px solid black;">
                <h4>Time Filter</h4>
                <p>Use sidebar filters to adjust time range</p>
            </div>
            """
        ).add_to(map_obj)
        
        return map_obj
    
    def create_crime_hotspot_map(self, crime_data: pd.DataFrame,
                               cluster_data: List[Dict[str, Any]] = None,
                               risk_grid: np.ndarray = None) -> folium.Map:
        """Create comprehensive crime hotspot map"""
        map_obj = self.create_base_map()
        
        if not crime_data.empty:
            crime_points = crime_data[crime_data['is_crime'] == True]
            self.add_heatmap(map_obj, crime_points, radius=20, weight_col='risk_score')
        
        if cluster_data:
            self.add_cluster_markers(map_obj, cluster_data)
        
        if risk_grid is not None:
            self.add_risk_surface(map_obj, risk_grid)
        
        folium.LayerControl().add_to(map_obj)
        return map_obj
    
    def create_agent_activity_map(self, agent_data: pd.DataFrame,
                                agent_type: str = None) -> folium.Map:
        """Create map showing agent activity"""
        map_obj = self.create_base_map()
        
        if not agent_data.empty:
            # Filter by agent type if specified
            if agent_type and 'agent_type' in agent_data.columns:
                agent_data = agent_data[agent_data['agent_type'] == agent_type]
            
            # Add agent positions as markers
            agent_data = self._convert_grid_to_latlon(agent_data)
            
            color_map = {
                'offender': 'red',
                'target': 'blue',
                'guardian': 'green'
            }
            
            for _, row in agent_data.iterrows():
                if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                    agent_type = row.get('agent_type', 'unknown')
                    color = color_map.get(agent_type, 'gray')
                    
                    popup_content = f"""
                    <b>Agent {row.get('agent_id', 'unknown')}</b><br>
                    Type: {agent_type}<br>
                    Action: {row.get('action', 'unknown')}<br>
                    Risk Score: {row.get('risk_score', 0):.2f}
                    """
                    
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=5,
                        popup=folium.Popup(popup_content, max_width=200),
                        color=color,
                        fill=True,
                        fillOpacity=0.6
                    ).add_to(map_obj)
        
        folium.LayerControl().add_to(map_obj)
        return map_obj
    
    def export_map_to_html(self, map_obj: folium.Map, filename: str) -> str:
        """Export map to HTML file"""
        map_obj.save(filename)
        return filename
    
    def get_map_as_html(self, map_obj: folium.Map) -> str:
        """Get map as HTML string"""
        return map_obj._repr_html_()

def create_sample_map():
    """Create a sample map for testing"""
    visualizer = MapVisualizer()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'x_coord': [25, 50, 75],
        'y_coord': [25, 50, 75],
        'is_crime': [True, False, True],
        'risk_score': [0.8, 0.3, 0.9]
    })
    
    map_obj = visualizer.create_crime_hotspot_map(sample_data)
    return map_obj 