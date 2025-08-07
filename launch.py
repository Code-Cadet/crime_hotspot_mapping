#!/usr/bin/env python3
"""
Launch Script for Crime Hotspot Simulation & Dashboard
Runs simulation and launches Streamlit dashboard
"""

import sys
import os
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulation.simulator import CrimeHotspotSimulator
from config import SIMULATION_CONFIG

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('launch.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed"""
    logger = logging.getLogger(__name__)
    
    required_packages = [
        ('streamlit', 'streamlit'),
        ('pandas', 'pandas'), 
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('folium', 'folium'),
        ('plotly', 'plotly'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn')
    ]
    
    missing_packages = []
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Please install missing packages using: pip install -r requirements.txt")
        return False
    
    logger.info("All required dependencies are installed")
    return True

def run_simulation(episodes=None, quick_mode=False):
    """Run the crime hotspot simulation"""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting crime hotspot simulation...")
    
    try:
        # Create simulator
        simulator = CrimeHotspotSimulator()
        
        # Adjust episodes for quick mode
        if quick_mode:
            episodes = min(episodes or 5, 5)
            logger.info(f"Quick mode: Running {episodes} episodes")
        else:
            episodes = episodes or SIMULATION_CONFIG["episodes"]
            logger.info(f"Running {episodes} episodes")
        
        # Run simulation
        start_time = datetime.now()
        episode_stats = simulator.run_simulation(episodes)
        end_time = datetime.now()
        
        duration = end_time - start_time
        
        logger.info(f"Simulation completed in {duration}")
        logger.info(f"Processed {len(episode_stats)} episodes")
        
        # Print summary statistics
        total_crimes = sum(ep.get('crime_events', 0) for ep in episode_stats)
        total_arrests = sum(ep.get('arrest_events', 0) for ep in episode_stats)
        
        logger.info(f"Total crime events: {total_crimes}")
        logger.info(f"Total arrest events: {total_arrests}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during simulation: {str(e)}")
        return False

def launch_dashboard(port=8501, headless=False):
    """Launch the Streamlit dashboard"""
    logger = logging.getLogger(__name__)
    
    logger.info("Launching Streamlit dashboard...")
    
    try:
        # Check if dashboard file exists
        dashboard_path = Path("app/dashboard.py")
        if not dashboard_path.exists():
            logger.error(f"Dashboard file not found: {dashboard_path}")
            return False
        
        # Build command
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.port", str(port),
            "--server.headless", str(headless).lower()
        ]
        
        logger.info(f"Starting dashboard on port {port}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Launch dashboard
        process = subprocess.Popen(cmd)
        
        logger.info("Dashboard launched successfully!")
        logger.info(f"Access the dashboard at: http://localhost:{port}")
        
        return process
        
    except Exception as e:
        logger.error(f"Error launching dashboard: {str(e)}")
        return None

def run_full_pipeline(episodes=None, quick_mode=False, port=8501, headless=False):
    """Run the complete pipeline: simulation + dashboard"""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Crime Hotspot Simulation & Dashboard Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Check dependencies
    logger.info("\nStep 1: Checking dependencies...")
    if not check_dependencies():
        logger.error("Dependency check failed. Exiting.")
        return False
    
    # Step 2: Run simulation
    logger.info("\nStep 2: Running simulation...")
    if not run_simulation(episodes, quick_mode):
        logger.error("Simulation failed. Exiting.")
        return False
    
    # Step 3: Launch dashboard
    logger.info("\nStep 3: Launching dashboard...")
    dashboard_process = launch_dashboard(port, headless)
    
    if dashboard_process is None:
        logger.error("Failed to launch dashboard. Exiting.")
        return False
    
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("Dashboard is running. Press Ctrl+C to stop.")
    logger.info("=" * 60)
    
    try:
        # Wait for dashboard process
        dashboard_process.wait()
    except KeyboardInterrupt:
        logger.info("\nShutting down dashboard...")
        dashboard_process.terminate()
        dashboard_process.wait()
        logger.info("Dashboard stopped.")
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Launch Crime Hotspot Simulation & Dashboard"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of simulation episodes (default: from config)"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: run fewer episodes for faster testing"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for Streamlit dashboard (default: 8501)"
    )
    
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run dashboard in headless mode"
    )
    
    parser.add_argument(
        "--simulation-only",
        action="store_true",
        help="Only run simulation, don't launch dashboard"
    )
    
    parser.add_argument(
        "--dashboard-only",
        action="store_true",
        help="Only launch dashboard, don't run simulation"
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
        if args.simulation_only:
            logger.info("Running simulation only...")
            success = run_simulation(args.episodes, args.quick)
        elif args.dashboard_only:
            logger.info("Launching dashboard only...")
            dashboard_process = launch_dashboard(args.port, args.headless)
            success = dashboard_process is not None
            if success:
                try:
                    dashboard_process.wait()
                except KeyboardInterrupt:
                    logger.info("Shutting down dashboard...")
                    dashboard_process.terminate()
                    dashboard_process.wait()
        else:
            logger.info("Running full pipeline...")
            success = run_full_pipeline(args.episodes, args.quick, args.port, args.headless)
        
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