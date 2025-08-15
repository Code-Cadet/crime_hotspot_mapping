"""
Spatial Clustering Module for Crime Hotspot Mapping

This module provides comprehensive clustering analysis capabilities for crime hotspot data.
It includes K-means and DBSCAN clustering methods with automatic optimal parameter selection.

Main Components:
- clustering.py: Main clustering analysis script
- utils.py: Helper functions for plotting, scaling, and evaluation
- cluster_analysis.py: Legacy clustering analysis (deprecated)

Usage:
    # As standalone script
    python clustering/clustering.py --method kmeans --plot
    
    # As imported module
    from clustering.clustering import run_clustering_analysis
    results = run_clustering_analysis(data_path, method="kmeans")
"""

from .clustering import (
    run_clustering_analysis,
    run_kmeans_clustering,
    run_dbscan_clustering,
    setup_logging
)

from .utils import (
    load_and_preprocess_data,
    find_optimal_k,
    evaluate_clustering,
    plot_spatial_clusters,
    plot_cluster_characteristics
)

__version__ = "1.0.0"
__author__ = "Crime Hotspot Mapping Team"

__all__ = [
    # Main clustering functions
    "run_clustering_analysis",
    "run_kmeans_clustering", 
    "run_dbscan_clustering",
    "setup_logging",
    
    # Utility functions
    "load_and_preprocess_data",
    "find_optimal_k",
    "evaluate_clustering",
    "plot_spatial_clusters",
    "plot_cluster_characteristics"
] 