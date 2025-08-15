#!/usr/bin/env python3
"""
Automated Model Training Script for Crime Hotspot Simulation
Trains machine learning models on processed simulation data
"""

import sys
import os
from pathlib import Path
import argparse
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.preprocessing import CrimeDataPreprocessor
from clustering.clustering import run_clustering_analysis
from prediction.model_train import CrimePredictionModel
from config import PROCESSED_DATA_DIR

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def check_data_availability():
    """Check if required data files are available"""
    logger = logging.getLogger(__name__)
    
    # Check for processed data - prioritize engineered_features.csv
    engineered_features_file = PROCESSED_DATA_DIR / "engineered_features.csv"
    if engineered_features_file.exists():
        logger.info("Found engineered_features.csv")
        return True
    
    # Fallback to old pattern
    processed_files = list(PROCESSED_DATA_DIR.glob("processed_simulation_data_*.csv"))
    if not processed_files:
        logger.error("No processed simulation data found!")
        logger.info("Please run simulation and data preprocessing first.")
        return False
    
    logger.info(f"Found {len(processed_files)} processed data files")
    return True

def run_data_preprocessing():
    """Run data preprocessing if needed"""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting data preprocessing...")
    
    try:
        preprocessor = CrimeDataPreprocessor()
        processed_data = preprocessor.process_simulation_data()
        
        if processed_data.empty:
            logger.error("No data was processed!")
            return False
        
        logger.info(f"Successfully processed {len(processed_data)} events")
        
        # Export processed data
        output_file = preprocessor.export_processed_data(processed_data)
        logger.info(f"Processed data exported to: {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during data preprocessing: {str(e)}")
        return False

def run_clustering_analysis_pipeline():
    """Run clustering analysis"""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting clustering analysis...")
    
    try:
        # Check for engineered_features.csv
        engineered_features_file = PROCESSED_DATA_DIR / "engineered_features.csv"
        if not engineered_features_file.exists():
            logger.error("No processed data available for clustering")
            return False
        
        # Run clustering analysis using the new module
        results = run_clustering_analysis(
            data_path=engineered_features_file,
            method="kmeans",
            sample_size=10000,  # Use sample for faster processing
            plot=False  # Disable plotting for automated pipeline
        )
        
        logger.info("Clustering analysis completed successfully")
        logger.info(f"Method: {results['method']}")
        logger.info(f"Number of clusters: {results['evaluation']['n_clusters']}")
        logger.info(f"Silhouette score: {results['evaluation']['silhouette_score']:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during clustering analysis: {str(e)}")
        return False

def run_model_training():
    """Run machine learning model training"""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting model training...")
    
    try:
        # Load processed data - check for engineered_features.csv first
        engineered_features_file = PROCESSED_DATA_DIR / "engineered_features.csv"
        if engineered_features_file.exists():
            logger.info("Found engineered_features.csv, using for model training")
            import pandas as pd
            processed_data = pd.read_csv(engineered_features_file)
        else:
            # Fallback to old pattern
            processed_files = list(PROCESSED_DATA_DIR.glob("processed_simulation_data_*.csv"))
            if not processed_files:
                logger.error("No processed data available for model training")
                return False
            
            latest_file = max(processed_files, key=lambda x: x.stat().st_mtime)
            import pandas as pd
            processed_data = pd.read_csv(latest_file)
        
        if processed_data.empty:
            logger.error("Processed data is empty")
            return False
        
        # Run model training
        predictor = CrimePredictionModel()
        training_results = predictor.run_complete_training(processed_data)
        
        logger.info("Model training completed successfully")
        logger.info(f"Best model: {training_results['best_model']}")
        logger.info(f"Best F1-Score: {training_results['best_score']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return False

def run_full_pipeline():
    """Run the complete training pipeline"""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Starting Crime Hotspot Model Training Pipeline")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    # Step 1: Check data availability
    logger.info("\nStep 1: Checking data availability...")
    if not check_data_availability():
        logger.error("Data availability check failed. Exiting.")
        return False
    
    # Step 2: Data preprocessing (if needed)
    logger.info("\nStep 2: Data preprocessing...")
    if not run_data_preprocessing():
        logger.error("Data preprocessing failed. Exiting.")
        return False
    
    # Step 3: Clustering analysis
    logger.info("\nStep 3: Clustering analysis...")
    if not run_clustering_analysis_pipeline():
        logger.error("Clustering analysis failed. Exiting.")
        return False
    
    # Step 4: Model training
    logger.info("\nStep 4: Model training...")
    if not run_model_training():
        logger.error("Model training failed. Exiting.")
        return False
    
    # Pipeline completed
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info(f"Total duration: {duration}")
    logger.info("=" * 60)
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Train machine learning models for crime hotspot simulation"
    )
    
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Only run data preprocessing"
    )
    
    parser.add_argument(
        "--cluster-only",
        action="store_true",
        help="Only run clustering analysis"
    )
    
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only run model training"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(log_level)
    
    try:
        if args.preprocess_only:
            logger.info("Running data preprocessing only...")
            success = run_data_preprocessing()
        elif args.cluster_only:
            logger.info("Running clustering analysis only...")
            success = run_clustering_analysis_pipeline()
        elif args.train_only:
            logger.info("Running model training only...")
            success = run_model_training()
        else:
            logger.info("Running full pipeline...")
            success = run_full_pipeline()
        
        if success:
            logger.info("Operation completed successfully!")
            sys.exit(0)
        else:
            logger.error("Operation failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 