import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DASHBOARD_CONFIG, PROCESSED_DATA_DIR, ROYSAMBU_BOUNDS
from utils.map_utils import MapVisualizer
from utils.chart_utils import ChartVisualizer
from src.data.preprocessing import CrimeDataPreprocessor
from clustering.cluster_analysis import CrimeClusterAnalyzer
from prediction.model_train import CrimePredictionModel
from risk.risk_model import RiskTerrainModel

class CrimeHotspotDashboard:
    def __init__(self):
        st.set_page_config(
            page_title=DASHBOARD_CONFIG["page_title"],
            page_icon=DASHBOARD_CONFIG["page_icon"],
            layout=DASHBOARD_CONFIG["layout"],
            initial_sidebar_state=DASHBOARD_CONFIG["initial_sidebar_state"]
        )
        
        self.map_visualizer = MapVisualizer()
        self.chart_visualizer = ChartVisualizer()
        self.data_loaded = False
        self.processed_data = None
        self.cluster_results = None
        self.model_results = None
        self.risk_model = None
        
    def load_data(self):
        """Load and process simulation data"""
        if not self.data_loaded:
            try:
                # Check if processed data exists
                processed_files = list(PROCESSED_DATA_DIR.glob("processed_simulation_data_*.csv"))
                
                if processed_files:
                    # Load most recent processed data
                    latest_file = max(processed_files, key=lambda x: x.stat().st_mtime)
                    self.processed_data = pd.read_csv(latest_file)
                    st.success(f"Loaded processed data: {latest_file.name}")
                else:
                    # Try to process raw simulation data
                    preprocessor = CrimeDataPreprocessor()
                    self.processed_data = preprocessor.process_simulation_data()
                    
                    if not self.processed_data.empty:
                        st.success("Processed simulation data successfully")
                    else:
                        st.warning("No simulation data found. Please run simulation first.")
                        return
                
                # Load cluster results
                cluster_files = list(PROCESSED_DATA_DIR.glob("clusters_*.json"))
                if cluster_files:
                    latest_cluster_file = max(cluster_files, key=lambda x: x.stat().st_mtime)
                    with open(latest_cluster_file, 'r') as f:
                        self.cluster_results = json.load(f)
                    st.success(f"Loaded cluster results: {latest_cluster_file.name}")
                
                # Load model results
                model_files = list(PROCESSED_DATA_DIR.glob("model_training_results_*.json"))
                if model_files:
                    latest_model_file = max(model_files, key=lambda x: x.stat().st_mtime)
                    with open(latest_model_file, 'r') as f:
                        self.model_results = json.load(f)
                    st.success(f"Loaded model results: {latest_model_file.name}")
                
                # Initialize risk model
                self.risk_model = RiskTerrainModel((100, 100))
                
                self.data_loaded = True
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    def sidebar_filters(self):
        """Create sidebar filters"""
        st.sidebar.header("🔧 Dashboard Controls")
        
        # Data selection
        st.sidebar.subheader("📊 Data Selection")
        
        # Time filters
        st.sidebar.subheader("⏰ Time Filters")
        if self.processed_data is not None and not self.processed_data.empty:
            hour_range = st.sidebar.slider(
                "Hour of Day",
                min_value=0,
                max_value=23,
                value=(0, 23)
            )
            
            weekday_filter = st.sidebar.multiselect(
                "Day of Week",
                options=range(7),
                default=range(7),
                format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x]
            )
        else:
            hour_range = (0, 23)
            weekday_filter = range(7)
        
        # Risk filters
        st.sidebar.subheader("⚠️ Risk Filters")
        risk_threshold = st.sidebar.slider(
            "Minimum Risk Score",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1
        )
        
        # Agent type filters
        st.sidebar.subheader("👥 Agent Filters")
        agent_types = st.sidebar.multiselect(
            "Agent Types",
            options=['offender', 'target', 'guardian'],
            default=['offender', 'target', 'guardian']
        )
        
        # Cluster filters
        st.sidebar.subheader("📍 Cluster Filters")
        cluster_method = st.sidebar.selectbox(
            "Clustering Method",
            options=['kmeans', 'dbscan'],
            index=0
        )
        
        return {
            'hour_range': hour_range,
            'weekday_filter': weekday_filter,
            'risk_threshold': risk_threshold,
            'agent_types': agent_types,
            'cluster_method': cluster_method
        }
    
    def filter_data(self, filters):
        """Apply filters to data"""
        if self.processed_data is None or self.processed_data.empty:
            return pd.DataFrame()
        
        filtered_data = self.processed_data.copy()
        
        # Time filters
        filtered_data = filtered_data[
            (filtered_data['hour'] >= filters['hour_range'][0]) &
            (filtered_data['hour'] <= filters['hour_range'][1])
        ]
        
        filtered_data = filtered_data[filtered_data['weekday'].isin(filters['weekday_filter'])]
        
        # Risk filter
        filtered_data = filtered_data[filtered_data['risk_score'] >= filters['risk_threshold']]
        
        # Agent type filter
        filtered_data = filtered_data[filtered_data['agent_type'].isin(filters['agent_types'])]
        
        return filtered_data
    
    def hotspot_map_tab(self, filters):
        """Hotspot KDE Map Tab"""
        st.header("🗺️ Crime Hotspot Analysis")
        
        if self.processed_data is None or self.processed_data.empty:
            st.warning("No data available. Please run simulation first.")
            return
        
        filtered_data = self.filter_data(filters)
        
        if filtered_data.empty:
            st.warning("No data matches the selected filters.")
            return
        
        # Create crime hotspot map
        crime_data = filtered_data[filtered_data['is_crime'] == True]
        
        # Get cluster data
        cluster_data = []
        if self.cluster_results and filters['cluster_method'] in self.cluster_results:
            cluster_data = self.cluster_results[filters['cluster_method']].get('hotspots', [])
        
        # Create map
        map_obj = self.map_visualizer.create_crime_hotspot_map(
            crime_data=crime_data,
            cluster_data=cluster_data,
            risk_grid=self.risk_model.get_risk_grid() if self.risk_model else None
        )
        
        # Display map
        st.components.v1.html(
            self.map_visualizer.get_map_as_html(map_obj),
            height=600
        )
        
        # Map controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export Map"):
                map_path = self.map_visualizer.export_map_to_html(map_obj, "crime_hotspot_map")
                st.success(f"Map exported to {map_path}")
        
        with col2:
            show_risk_surface = st.checkbox("Show Risk Surface", value=True)
        
        with col3:
            show_clusters = st.checkbox("Show Clusters", value=True)
        
        # Statistics
        st.subheader("📈 Hotspot Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Crime Events", len(crime_data))
        
        with col2:
            st.metric("High Risk Areas", len(filtered_data[filtered_data['risk_score'] > 0.7]))
        
        with col3:
            st.metric("Average Risk Score", f"{filtered_data['risk_score'].mean():.3f}")
        
        with col4:
            st.metric("Identified Clusters", len(cluster_data))
    
    def cluster_view_tab(self, filters):
        """Cluster Analysis Tab"""
        st.header("📍 Cluster Analysis")
        
        if self.cluster_results is None:
            st.warning("No cluster results available. Please run clustering analysis first.")
            return
        
        # Select clustering method
        method = filters['cluster_method']
        if method not in self.cluster_results:
            st.error(f"No results for {method} clustering method.")
            return
        
        cluster_data = self.cluster_results[method]
        
        # Display cluster information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Cluster Results")
            results = cluster_data.get('results', {})
            
            if results:
                st.write(f"**Method:** {results.get('method', 'N/A')}")
                st.write(f"**Number of Clusters:** {results.get('n_clusters', 'N/A')}")
                st.write(f"**Silhouette Score:** {results.get('silhouette_score', 0):.3f}")
                
                if method == 'dbscan':
                    st.write(f"**Noise Points:** {results.get('n_noise', 'N/A')}")
        
        with col2:
            st.subheader("🎯 Hotspot Summary")
            hotspots = cluster_data.get('hotspots', [])
            
            if hotspots:
                st.write(f"**Identified Hotspots:** {len(hotspots)}")
                
                for i, hotspot in enumerate(hotspots[:5]):  # Show top 5
                    st.write(f"**Hotspot {i+1}:** Crime Rate {hotspot.get('crime_rate', 0):.3f}")
            else:
                st.write("No hotspots identified.")
        
        # Cluster visualization
        st.subheader("📊 Cluster Visualization")
        
        if self.processed_data is not None and not self.processed_data.empty:
            filtered_data = self.filter_data(filters)
            
            if not filtered_data.empty:
                # Create cluster chart
                cluster_chart = self.chart_visualizer.create_cluster_analysis_chart(
                    cluster_data.get('hotspots', [])
                )
                st.plotly_chart(cluster_chart, use_container_width=True)
        
        # Cluster details table
        st.subheader("📋 Cluster Details")
        analysis = cluster_data.get('analysis', {})
        
        if analysis:
            cluster_df = pd.DataFrame.from_dict(analysis, orient='index')
            st.dataframe(cluster_df)
    
    def risk_surface_tab(self, filters):
        """Risk Surface Tab"""
        st.header("⚠️ Risk Terrain Model")
        
        if self.risk_model is None:
            st.warning("Risk model not available.")
            return
        
        # Display risk grid
        risk_grid = self.risk_model.get_risk_grid()
        
        # Create risk heatmap
        risk_chart = self.chart_visualizer.create_risk_heatmap(risk_grid)
        st.plotly_chart(risk_chart, use_container_width=True)
        
        # Risk statistics
        st.subheader("📊 Risk Statistics")
        risk_stats = self.risk_model.get_risk_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Risk", f"{risk_stats['mean_risk']:.3f}")
        
        with col2:
            st.metric("Max Risk", f"{risk_stats['max_risk']:.3f}")
        
        with col3:
            st.metric("High Risk Areas", f"{risk_stats['high_risk_areas']:.1%}")
        
        with col4:
            st.metric("Low Risk Areas", f"{risk_stats['low_risk_areas']:.1%}")
        
        # Risk factors
        st.subheader("🔍 Risk Factors")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**POI Density**")
            st.progress(self.risk_model.poi_density.mean())
            
            st.write("**Road Proximity**")
            st.progress(self.risk_model.road_proximity.mean())
        
        with col2:
            st.write("**Lighting Score**")
            st.progress(self.risk_model.lighting_score.mean())
            
            st.write("**Land Use Score**")
            st.progress(self.risk_model.landuse_score.mean())
    
    def model_metrics_tab(self, filters):
        """Model Metrics Tab"""
        st.header("🤖 Machine Learning Models")
        
        if self.model_results is None:
            st.warning("No model results available. Please train models first.")
            return
        
        # Model comparison
        st.subheader("📊 Model Performance Comparison")
        
        if 'model_comparison' in self.model_results:
            comparison_df = pd.DataFrame(self.model_results['model_comparison'])
            st.dataframe(comparison_df, use_container_width=True)
            
            # Create comparison chart
            comparison_chart = self.chart_visualizer.create_model_comparison_chart(
                self.model_results.get('model_results', {})
            )
            st.plotly_chart(comparison_chart, use_container_width=True)
        
        # Feature importance
        st.subheader("🎯 Feature Importance")
        
        if 'model_results' in self.model_results:
            best_model = self.model_results.get('best_model', 'RandomForest')
            model_data = self.model_results['model_results'].get(best_model, {})
            
            if 'top_features' in model_data:
                feature_importance = model_data['top_features']
                feature_chart = self.chart_visualizer.create_feature_importance_chart(
                    feature_importance
                )
                st.plotly_chart(feature_chart, use_container_width=True)
        
        # Model details
        st.subheader("📋 Model Details")
        
        if 'model_results' in self.model_results:
            for model_name, model_data in self.model_results['model_results'].items():
                with st.expander(f"📈 {model_name}"):
                    st.write(f"**Best Parameters:** {model_data.get('best_params', 'N/A')}")
                    
                    metrics = model_data.get('metrics', {})
                    if metrics:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
                            st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
                        
                        with col2:
                            st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
                            st.metric("F1-Score", f"{metrics.get('f1_score', 0):.3f}")
                        
                        with col3:
                            st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
    
    def time_series_tab(self, filters):
        """Time Series Analysis Tab"""
        st.header("⏰ Temporal Analysis")
        
        if self.processed_data is None or self.processed_data.empty:
            st.warning("No data available.")
            return
        
        filtered_data = self.filter_data(filters)
        
        if filtered_data.empty:
            st.warning("No data matches the selected filters.")
            return
        
        # Time series charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🕐 Crime Incidents by Hour")
            time_chart = self.chart_visualizer.create_time_series_chart(
                filtered_data,
                value_col='is_crime',
                title='Crime Incidents by Hour'
            )
            st.plotly_chart(time_chart, use_container_width=True)
        
        with col2:
            st.subheader("📅 Crime Incidents by Day")
            day_chart = self.chart_visualizer.create_time_series_chart(
                filtered_data,
                time_col='weekday',
                value_col='is_crime',
                title='Crime Incidents by Day of Week'
            )
            st.plotly_chart(day_chart, use_container_width=True)
        
        # Risk distribution over time
        st.subheader("⚠️ Risk Score Distribution")
        risk_chart = self.chart_visualizer.create_risk_distribution_chart(filtered_data)
        st.plotly_chart(risk_chart, use_container_width=True)
        
        # Agent activity over time
        st.subheader("👥 Agent Activity Patterns")
        activity_chart = self.chart_visualizer.create_agent_activity_chart(filtered_data)
        st.plotly_chart(activity_chart, use_container_width=True)
    
    def simulation_summary_tab(self, filters):
        """Simulation Summary Tab"""
        st.header("🎮 Simulation Summary")
        
        if self.processed_data is None or self.processed_data.empty:
            st.warning("No simulation data available.")
            return
        
        # Simulation statistics
        st.subheader("📊 Simulation Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Events", len(self.processed_data))
        
        with col2:
            st.metric("Unique Agents", self.processed_data['agent_id'].nunique())
        
        with col3:
            st.metric("Episodes", self.processed_data['episode'].nunique())
        
        with col4:
            st.metric("Crime Events", self.processed_data['is_crime'].sum())
        
        # Agent type distribution
        st.subheader("👥 Agent Distribution")
        
        agent_counts = self.processed_data['agent_type'].value_counts()
        st.bar_chart(agent_counts)
        
        # Episode progress
        st.subheader("📈 Episode Progress")
        
        episode_stats = self.processed_data.groupby('episode').agg({
            'is_crime': 'sum',
            'is_arrest': 'sum',
            'risk_score': 'mean'
        }).reset_index()
        
        episode_chart = self.chart_visualizer.create_episode_progress_chart(
            episode_stats.to_dict('records')
        )
        st.plotly_chart(episode_chart, use_container_width=True)
        
        # Configuration
        st.subheader("⚙️ Simulation Configuration")
        
        config_data = {
            "Grid Size": "100x100",
            "Episodes": "50",
            "Steps per Episode": "1000",
            "Offender Agents": "20",
            "Target Agents": "50",
            "Guardian Agents": "10"
        }
        
        config_df = pd.DataFrame(list(config_data.items()), columns=['Parameter', 'Value'])
        st.dataframe(config_df, use_container_width=True)
    
    def run_dashboard(self):
        """Run the main dashboard"""
        st.title("🚨 Crime Hotspot Simulation & Mapping")
        st.markdown("**Roysambu Ward, Nairobi**")
        
        # Load data
        self.load_data()
        
        # Sidebar filters
        filters = self.sidebar_filters()
        
        # Main content
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🗺️ Hotspot Map", 
            "📍 Cluster View", 
            "⚠️ Risk Surface", 
            "🤖 Model Metrics", 
            "⏰ Time Series", 
            "🎮 Simulation Summary"
        ])
        
        with tab1:
            self.hotspot_map_tab(filters)
        
        with tab2:
            self.cluster_view_tab(filters)
        
        with tab3:
            self.risk_surface_tab(filters)
        
        with tab4:
            self.model_metrics_tab(filters)
        
        with tab5:
            self.time_series_tab(filters)
        
        with tab6:
            self.simulation_summary_tab(filters)
        
        # Footer
        st.markdown("---")
        st.markdown(
            "**Crime Hotspot Simulation & Mapping** | "
            "Agent-Based Modeling | Risk Terrain Modeling | Machine Learning"
        )

if __name__ == "__main__":
    dashboard = CrimeHotspotDashboard()
    dashboard.run_dashboard() 