import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class ChartVisualizer:
    def __init__(self):
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8'
        }
    
    def create_model_comparison_chart(self, model_results: Dict[str, Any]) -> go.Figure:
        """Create bar chart comparing model performance"""
        models = []
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        for model_name, results in model_results.items():
            if 'metrics' in results:
                models.append(model_name)
        
        fig = go.Figure()
        
        for metric in metrics:
            values = []
            for model_name in models:
                if 'metrics' in model_results[model_name]:
                    values.append(model_results[model_name]['metrics'].get(metric, 0))
            
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=models,
                y=values,
                text=[f'{v:.3f}' for v in values],
                textposition='auto',
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Models',
            yaxis_title='Score',
            barmode='group',
            height=500
        )
        
        return fig
    
    def create_feature_importance_chart(self, feature_importance: Dict[str, float], 
                                      top_n: int = 15) -> go.Figure:
        """Create horizontal bar chart for feature importance"""
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        features, importance = zip(*top_features)
        
        fig = go.Figure(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker_color=self.color_scheme['primary']
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=max(400, len(features) * 25)
        )
        
        return fig
    
    def create_time_series_chart(self, data: pd.DataFrame, 
                               time_col: str = 'hour',
                               value_col: str = 'is_crime',
                               title: str = 'Crime Incidents Over Time') -> go.Figure:
        """Create time series chart"""
        if data.empty:
            return go.Figure()
        
        # Group by time
        time_data = data.groupby(time_col)[value_col].sum().reset_index()
        
        fig = go.Figure(go.Scatter(
            x=time_data[time_col],
            y=time_data[value_col],
            mode='lines+markers',
            line=dict(color=self.color_scheme['danger'], width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Hour of Day',
            yaxis_title='Number of Incidents',
            height=400
        )
        
        return fig
    
    def create_risk_distribution_chart(self, data: pd.DataFrame, 
                                     risk_col: str = 'risk_score') -> go.Figure:
        """Create histogram of risk score distribution"""
        if data.empty:
            return go.Figure()
        
        fig = go.Figure(go.Histogram(
            x=data[risk_col],
            nbinsx=30,
            marker_color=self.color_scheme['warning'],
            opacity=0.7
        ))
        
        fig.update_layout(
            title='Risk Score Distribution',
            xaxis_title='Risk Score',
            yaxis_title='Frequency',
            height=400
        )
        
        return fig
    
    def create_cluster_analysis_chart(self, cluster_data: List[Dict[str, Any]]) -> go.Figure:
        """Create scatter plot of cluster analysis"""
        if not cluster_data:
            return go.Figure()
        
        # Extract cluster information
        x_coords = []
        y_coords = []
        sizes = []
        colors = []
        labels = []
        
        for cluster in cluster_data:
            if 'spatial_center' in cluster:
                center = cluster['spatial_center']
                x_coords.append(center['x'])
                y_coords.append(center['y'])
                sizes.append(cluster.get('size', 10) * 2)
                colors.append(cluster.get('crime_rate', 0))
                labels.append(f"Cluster {cluster.get('cluster_id', 'N/A')}")
        
        fig = go.Figure(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Crime Rate")
            ),
            text=labels,
            hovertemplate='<b>%{text}</b><br>' +
                         'Crime Rate: %{marker.color:.3f}<br>' +
                         'Size: %{marker.size}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title='Crime Cluster Analysis',
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            height=500
        )
        
        return fig
    
    def create_agent_activity_chart(self, data: pd.DataFrame,
                                  agent_type_col: str = 'agent_type',
                                  action_col: str = 'action') -> go.Figure:
        """Create bar chart of agent activity"""
        if data.empty:
            return go.Figure()
        
        # Count actions by agent type
        activity_data = data.groupby([agent_type_col, action_col]).size().reset_index(name='count')
        
        fig = px.bar(
            activity_data,
            x=agent_type_col,
            y='count',
            color=action_col,
            title='Agent Activity by Type and Action',
            color_discrete_map={
                'move': self.color_scheme['primary'],
                'assault': self.color_scheme['danger'],
                'arrest': self.color_scheme['success'],
                'patrol': self.color_scheme['info'],
                'evade': self.color_scheme['warning']
            }
        )
        
        fig.update_layout(height=400)
        
        return fig
    
    def create_confusion_matrix_chart(self, confusion_matrix: List[List[int]],
                                    labels: List[str] = None) -> go.Figure:
        """Create heatmap of confusion matrix"""
        if labels is None:
            labels = ['No Crime', 'Crime']
        
        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            x=labels,
            y=labels,
            colorscale='Blues',
            text=confusion_matrix,
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=400
        )
        
        return fig
    
    def create_geographic_distribution_chart(self, data: pd.DataFrame,
                                           lat_col: str = 'latitude',
                                           lon_col: str = 'longitude',
                                           color_col: str = 'risk_score') -> go.Figure:
        """Create scatter plot of geographic distribution"""
        if data.empty:
            return go.Figure()
        
        fig = go.Figure(go.Scattermapbox(
            lat=data[lat_col],
            lon=data[lon_col],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=8,
                color=data[color_col],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Risk Score")
            ),
            text=data.get('agent_id', ''),
            hovertemplate='<b>%{text}</b><br>' +
                         'Risk: %{marker.color:.3f}<br>' +
                         'Lat: %{lat:.4f}<br>' +
                         'Lon: %{lon:.4f}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title='Geographic Distribution of Events',
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=data[lat_col].mean(), lon=data[lon_col].mean()),
                zoom=12
            ),
            height=500
        )
        
        return fig
    
    def create_summary_statistics_chart(self, summary_stats: Dict[str, Any]) -> go.Figure:
        """Create summary statistics display"""
        # Create a table-like visualization
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Metric', 'Value'],
                fill_color='lightblue',
                align='left',
                font=dict(size=14)
            ),
            cells=dict(
                values=[
                    list(summary_stats.keys()),
                    [f"{v:.3f}" if isinstance(v, float) else str(v) for v in summary_stats.values()]
                ],
                fill_color='lavender',
                align='left',
                font=dict(size=12)
            )
        )])
        
        fig.update_layout(
            title='Summary Statistics',
            height=300
        )
        
        return fig
    
    def create_episode_progress_chart(self, episode_data: List[Dict[str, Any]]) -> go.Figure:
        """Create chart showing simulation episode progress"""
        if not episode_data:
            return go.Figure()
        
        episodes = [ep['episode'] for ep in episode_data]
        crime_counts = [ep.get('crime_events', 0) for ep in episode_data]
        arrest_counts = [ep.get('arrest_events', 0) for ep in episode_data]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=episodes,
            y=crime_counts,
            mode='lines+markers',
            name='Crime Events',
            line=dict(color=self.color_scheme['danger'])
        ))
        
        fig.add_trace(go.Scatter(
            x=episodes,
            y=arrest_counts,
            mode='lines+markers',
            name='Arrest Events',
            line=dict(color=self.color_scheme['success'])
        ))
        
        fig.update_layout(
            title='Simulation Episode Progress',
            xaxis_title='Episode',
            yaxis_title='Number of Events',
            height=400
        )
        
        return fig
    
    def create_risk_heatmap(self, risk_grid: np.ndarray) -> go.Figure:
        """Create heatmap of risk grid"""
        fig = go.Figure(data=go.Heatmap(
            z=risk_grid,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Risk Score")
        ))
        
        fig.update_layout(
            title='Risk Terrain Model',
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            height=500
        )
        
        return fig
    
    def create_multi_metric_dashboard(self, model_results: Dict[str, Any],
                                    feature_importance: Dict[str, float],
                                    time_data: pd.DataFrame) -> List[go.Figure]:
        """Create multiple charts for dashboard"""
        charts = []
        
        # Model comparison
        if model_results:
            charts.append(self.create_model_comparison_chart(model_results))
        
        # Feature importance
        if feature_importance:
            charts.append(self.create_feature_importance_chart(feature_importance))
        
        # Time series
        if not time_data.empty:
            charts.append(self.create_time_series_chart(time_data))
        
        # Risk distribution
        if not time_data.empty and 'risk_score' in time_data.columns:
            charts.append(self.create_risk_distribution_chart(time_data))
        
        return charts

def create_sample_charts():
    """Create sample charts for testing"""
    visualizer = ChartVisualizer()
    
    # Sample model results
    sample_model_results = {
        'RandomForest': {
            'metrics': {
                'accuracy': 0.85,
                'precision': 0.82,
                'recall': 0.78,
                'f1_score': 0.80,
                'roc_auc': 0.88
            }
        },
        'XGBoost': {
            'metrics': {
                'accuracy': 0.87,
                'precision': 0.84,
                'recall': 0.80,
                'f1_score': 0.82,
                'roc_auc': 0.90
            }
        }
    }
    
    # Sample feature importance
    sample_feature_importance = {
        'risk_score': 0.25,
        'hour': 0.18,
        'reputation': 0.15,
        'nearby_agents': 0.12,
        'distance_from_center': 0.10
    }
    
    # Sample time data
    sample_time_data = pd.DataFrame({
        'hour': range(24),
        'is_crime': np.random.poisson(5, 24),
        'risk_score': np.random.uniform(0.3, 0.8, 24)
    })
    
    # Create charts
    model_chart = visualizer.create_model_comparison_chart(sample_model_results)
    feature_chart = visualizer.create_feature_importance_chart(sample_feature_importance)
    time_chart = visualizer.create_time_series_chart(sample_time_data)
    
    return model_chart, feature_chart, time_chart, visualizer 