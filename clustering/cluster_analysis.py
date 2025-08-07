import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

from config import CLUSTERING_CONFIG, PROCESSED_DATA_DIR, ROYSAMBU_BOUNDS

class CrimeClusterAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.dbscan_model = None
        self.cluster_results = {}
        
    def prepare_clustering_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Prepare data for clustering analysis"""
        # Select features for clustering
        clustering_features = [
            "x_coord", "y_coord", "risk_score", "reputation",
            "nearby_agent_count", "nearby_offenders", "nearby_guardians",
            "distance_from_center", "hour", "weekday"
        ]
        
        # Filter available features
        available_features = [col for col in clustering_features if col in df.columns]
        
        # Prepare feature matrix
        X = df[available_features].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, X
    
    def apply_kmeans_clustering(self, X: np.ndarray, n_clusters: int = None) -> Dict[str, Any]:
        """Apply K-means clustering"""
        if n_clusters is None:
            n_clusters = CLUSTERING_CONFIG["kmeans_n_clusters"]
        
        # Apply K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Calculate metrics
        silhouette_avg = silhouette_score(X, cluster_labels)
        calinski_score = calinski_harabasz_score(X, cluster_labels)
        davies_score = davies_bouldin_score(X, cluster_labels)
        
        # Get cluster centers
        cluster_centers = kmeans.cluster_centers_
        
        # Store results
        results = {
            "method": "kmeans",
            "n_clusters": n_clusters,
            "cluster_labels": cluster_labels.tolist(),
            "cluster_centers": cluster_centers.tolist(),
            "silhouette_score": silhouette_avg,
            "calinski_harabasz_score": calinski_score,
            "davies_bouldin_score": davies_score,
            "inertia": kmeans.inertia_,
            "model": kmeans
        }
        
        self.kmeans_model = kmeans
        return results
    
    def apply_dbscan_clustering(self, X: np.ndarray, eps: float = None, min_samples: int = None) -> Dict[str, Any]:
        """Apply DBSCAN clustering"""
        if eps is None:
            eps = CLUSTERING_CONFIG["dbscan_eps"]
        if min_samples is None:
            min_samples = CLUSTERING_CONFIG["dbscan_min_samples"]
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(X)
        
        # Calculate metrics (only for non-noise points)
        non_noise_mask = cluster_labels != -1
        if np.sum(non_noise_mask) > 1 and len(np.unique(cluster_labels[non_noise_mask])) > 1:
            silhouette_avg = silhouette_score(X[non_noise_mask], cluster_labels[non_noise_mask])
            calinski_score = calinski_harabasz_score(X[non_noise_mask], cluster_labels[non_noise_mask])
            davies_score = davies_bouldin_score(X[non_noise_mask], cluster_labels[non_noise_mask])
        else:
            silhouette_avg = calinski_score = davies_score = 0.0
        
        # Count clusters and noise points
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(cluster_labels == -1)
        
        # Store results
        results = {
            "method": "dbscan",
            "eps": eps,
            "min_samples": min_samples,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "cluster_labels": cluster_labels.tolist(),
            "silhouette_score": silhouette_avg,
            "calinski_harabasz_score": calinski_score,
            "davies_bouldin_score": davies_score,
            "model": dbscan
        }
        
        self.dbscan_model = dbscan
        return results
    
    def find_optimal_kmeans_clusters(self, X: np.ndarray, max_clusters: int = 10) -> Dict[str, Any]:
        """Find optimal number of clusters for K-means using elbow method and silhouette analysis"""
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, cluster_labels))
        
        # Find elbow point (simplified method)
        elbow_k = self._find_elbow_point(k_range, inertias)
        
        # Find best silhouette score
        best_silhouette_k = k_range[np.argmax(silhouette_scores)]
        
        return {
            "k_range": list(k_range),
            "inertias": inertias,
            "silhouette_scores": silhouette_scores,
            "elbow_k": elbow_k,
            "best_silhouette_k": best_silhouette_k,
            "recommended_k": best_silhouette_k  # Prefer silhouette score
        }
    
    def _find_elbow_point(self, k_range: range, inertias: List[float]) -> int:
        """Find elbow point using simplified method"""
        if len(inertias) < 3:
            return k_range[0]
        
        # Calculate second derivative
        second_derivative = []
        for i in range(1, len(inertias) - 1):
            second_deriv = inertias[i+1] - 2*inertias[i] + inertias[i-1]
            second_derivative.append(second_deriv)
        
        # Find point with maximum second derivative
        elbow_idx = np.argmax(second_derivative) + 1
        return k_range[elbow_idx]
    
    def analyze_cluster_characteristics(self, df: pd.DataFrame, cluster_labels: np.ndarray, 
                                      method: str) -> Dict[str, Any]:
        """Analyze characteristics of each cluster"""
        df_clustered = df.copy()
        df_clustered[f"{method}_cluster"] = cluster_labels
        
        cluster_analysis = {}
        
        for cluster_id in np.unique(cluster_labels):
            if cluster_id == -1:  # DBSCAN noise
                cluster_name = "noise"
            else:
                cluster_name = f"cluster_{cluster_id}"
            
            cluster_data = df_clustered[df_clustered[f"{method}_cluster"] == cluster_id]
            
            if len(cluster_data) == 0:
                continue
            
            # Calculate cluster statistics
            cluster_stats = {
                "size": len(cluster_data),
                "percentage": len(cluster_data) / len(df) * 100,
                "crime_rate": cluster_data["is_crime"].mean() if "is_crime" in cluster_data.columns else 0,
                "arrest_rate": cluster_data["is_arrest"].mean() if "is_arrest" in cluster_data.columns else 0,
                "avg_risk_score": cluster_data["risk_score"].mean(),
                "avg_reputation": cluster_data["reputation"].mean(),
                "avg_nearby_agents": cluster_data["nearby_agent_count"].mean(),
                "spatial_center": {
                    "x": cluster_data["x_coord"].mean(),
                    "y": cluster_data["y_coord"].mean()
                },
                "time_distribution": cluster_data["hour"].value_counts().to_dict(),
                "agent_type_distribution": cluster_data["agent_type"].value_counts().to_dict()
            }
            
            cluster_analysis[cluster_name] = cluster_stats
        
        return cluster_analysis
    
    def identify_hotspots(self, df: pd.DataFrame, cluster_labels: np.ndarray, 
                         method: str, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Identify crime hotspots based on clustering results"""
        df_clustered = df.copy()
        df_clustered[f"{method}_cluster"] = cluster_labels
        
        hotspots = []
        
        for cluster_id in np.unique(cluster_labels):
            if cluster_id == -1:  # Skip noise in DBSCAN
                continue
            
            cluster_data = df_clustered[df_clustered[f"{method}_cluster"] == cluster_id]
            
            # Calculate crime rate for this cluster
            crime_rate = cluster_data["is_crime"].mean() if "is_crime" in cluster_data.columns else 0
            
            if crime_rate >= threshold:
                # This is a hotspot
                hotspot_info = {
                    "cluster_id": int(cluster_id),
                    "method": method,
                    "crime_rate": crime_rate,
                    "size": len(cluster_data),
                    "spatial_center": {
                        "x": cluster_data["x_coord"].mean(),
                        "y": cluster_data["y_coord"].mean()
                    },
                    "avg_risk_score": cluster_data["risk_score"].mean(),
                    "time_pattern": cluster_data["hour"].value_counts().head(3).to_dict(),
                    "agent_pattern": cluster_data["agent_type"].value_counts().to_dict()
                }
                
                hotspots.append(hotspot_info)
        
        return hotspots
    
    def create_cluster_visualization(self, df: pd.DataFrame, cluster_results: Dict[str, Any], 
                                   save_path: str = None) -> None:
        """Create visualization of clustering results"""
        method = cluster_results["method"]
        cluster_labels = np.array(cluster_results["cluster_labels"])
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{method.upper()} Clustering Results', fontsize=16)
        
        # Scatter plot of clusters
        scatter = axes[0, 0].scatter(df["x_coord"], df["y_coord"], 
                                   c=cluster_labels, cmap='viridis', alpha=0.6)
        axes[0, 0].set_title('Spatial Distribution of Clusters')
        axes[0, 0].set_xlabel('X Coordinate')
        axes[0, 0].set_ylabel('Y Coordinate')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # Risk score vs cluster
        axes[0, 1].boxplot([df[cluster_labels == i]["risk_score"] 
                           for i in np.unique(cluster_labels) if i != -1])
        axes[0, 1].set_title('Risk Score Distribution by Cluster')
        axes[0, 1].set_xlabel('Cluster')
        axes[0, 1].set_ylabel('Risk Score')
        
        # Crime rate by cluster
        crime_rates = []
        cluster_ids = []
        for cluster_id in np.unique(cluster_labels):
            if cluster_id != -1:
                cluster_data = df[cluster_labels == cluster_id]
                crime_rate = cluster_data["is_crime"].mean() if "is_crime" in cluster_data.columns else 0
                crime_rates.append(crime_rate)
                cluster_ids.append(cluster_id)
        
        axes[1, 0].bar(cluster_ids, crime_rates)
        axes[1, 0].set_title('Crime Rate by Cluster')
        axes[1, 0].set_xlabel('Cluster')
        axes[1, 0].set_ylabel('Crime Rate')
        
        # Time distribution heatmap
        time_cluster_data = pd.DataFrame({
            'hour': df['hour'],
            'cluster': cluster_labels
        })
        time_cluster_pivot = time_cluster_data.groupby(['hour', 'cluster']).size().unstack(fill_value=0)
        
        sns.heatmap(time_cluster_pivot, ax=axes[1, 1], cmap='YlOrRd')
        axes[1, 1].set_title('Time-Cluster Distribution')
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].set_ylabel('Hour of Day')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Clustering visualization saved to {save_path}")
        
        plt.show()
    
    def export_cluster_results(self, cluster_results: Dict[str, Any], 
                             cluster_analysis: Dict[str, Any],
                             hotspots: List[Dict[str, Any]],
                             filename: str = None) -> str:
        """Export clustering results to JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"clusters_{cluster_results['method']}_{timestamp}.json"
        
        export_data = {
            "clustering_results": cluster_results,
            "cluster_analysis": cluster_analysis,
            "hotspots": hotspots,
            "timestamp": datetime.now().isoformat(),
            "config": CLUSTERING_CONFIG
        }
        
        # Remove non-serializable objects
        if "model" in export_data["clustering_results"]:
            del export_data["clustering_results"]["model"]
        
        filepath = PROCESSED_DATA_DIR / filename
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Cluster results exported to {filepath}")
        return str(filepath)
    
    def run_complete_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run complete clustering analysis pipeline"""
        print("Preparing clustering data...")
        X_scaled, X_original = self.prepare_clustering_data(df)
        
        print("Finding optimal K-means clusters...")
        optimal_k = self.find_optimal_kmeans_clusters(X_scaled)
        recommended_k = optimal_k["recommended_k"]
        
        print(f"Applying K-means with {recommended_k} clusters...")
        kmeans_results = self.apply_kmeans_clustering(X_scaled, recommended_k)
        
        print("Applying DBSCAN clustering...")
        dbscan_results = self.apply_dbscan_clustering(X_scaled)
        
        # Analyze cluster characteristics
        print("Analyzing cluster characteristics...")
        kmeans_analysis = self.analyze_cluster_characteristics(
            df, np.array(kmeans_results["cluster_labels"]), "kmeans"
        )
        dbscan_analysis = self.analyze_cluster_characteristics(
            df, np.array(dbscan_results["cluster_labels"]), "dbscan"
        )
        
        # Identify hotspots
        print("Identifying crime hotspots...")
        kmeans_hotspots = self.identify_hotspots(
            df, np.array(kmeans_results["cluster_labels"]), "kmeans"
        )
        dbscan_hotspots = self.identify_hotspots(
            df, np.array(dbscan_results["cluster_labels"]), "dbscan"
        )
        
        # Store results
        self.cluster_results = {
            "kmeans": {
                "results": kmeans_results,
                "analysis": kmeans_analysis,
                "hotspots": kmeans_hotspots
            },
            "dbscan": {
                "results": dbscan_results,
                "analysis": dbscan_analysis,
                "hotspots": dbscan_hotspots
            },
            "optimal_k_analysis": optimal_k
        }
        
        # Export results
        self.export_cluster_results(kmeans_results, kmeans_analysis, kmeans_hotspots)
        self.export_cluster_results(dbscan_results, dbscan_analysis, dbscan_hotspots)
        
        return self.cluster_results

if __name__ == "__main__":
    # Example usage
    analyzer = CrimeClusterAnalyzer()
    
    # Load processed data (assuming it exists)
    from src.data.preprocessing import CrimeDataPreprocessor
    
    preprocessor = CrimeDataPreprocessor()
    processed_df = preprocessor.process_simulation_data()
    
    if not processed_df.empty:
        results = analyzer.run_complete_analysis(processed_df)
        print("Clustering analysis completed!")
        print(f"K-means hotspots: {len(results['kmeans']['hotspots'])}")
        print(f"DBSCAN hotspots: {len(results['dbscan']['hotspots'])}") 