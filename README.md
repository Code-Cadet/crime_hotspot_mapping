# 🚨 Crime Hotspot Simulation & Mapping

**Roysambu Ward, Nairobi** - Agent-Based Modeling for Urban Crime Analysis

A comprehensive research and simulation project that uses **agent-based modeling (ABM)**, **risk terrain modeling (RTM)**, **spatial clustering**, and **machine learning** to simulate, analyze, and visualize urban crime dynamics in Roysambu ward, Nairobi.

## 🎯 Project Overview

This project demonstrates how synthetic data generated from agent-based simulations can be used to develop intelligent crime prediction models when actual crime data is unavailable. The system combines multiple analytical approaches:

- **Agent-Based Simulation**: Q-learning agents (offenders, targets, guardians) with reputation systems
- **Risk Terrain Modeling**: Spatial risk assessment based on environmental factors
- **Machine Learning**: Multiple algorithms for crime prediction
- **Spatial Clustering**: Hotspot identification using K-means and DBSCAN
- **Interactive Dashboard**: Real-time visualization and analysis

## 📁 Project Structure

```
crime-hotspot-simulator/
├── 📊 data/                          # Data storage
│   ├── raw/                          # Raw simulation data
│   ├── processed/                    # Processed datasets
│   └── risk_layers/                  # Risk terrain data
├── 🤖 simulation/                    # Agent-based simulation
│   ├── agents.py                     # Agent classes with Q-learning
│   └── simulator.py                  # Main simulation orchestrator
├── ⚠️ risk/                          # Risk terrain modeling
│   ├── risk_model.py                 # Base RTM implementation
│   └── enhanced_risk_model.py        # ML-enhanced RTM (LSTM/CNN)
├── 📍 clustering/                    # Spatial clustering
│   └── cluster_analysis.py           # K-means and DBSCAN analysis
├── 🧠 prediction/                    # Machine learning
│   └── model_train.py                # Model training and evaluation
├── 🔧 src/data/                      # Data preprocessing
│   └── preprocessing.py              # Feature engineering
├── 📈 app/                           # Streamlit dashboard
│   ├── dashboard.py                  # Main dashboard application
│   └── utils/                        # Dashboard utilities
│       ├── map_utils.py              # Folium map utilities
│       └── chart_utils.py            # Plotly chart utilities
├── 📚 notebooks/                     # Jupyter notebooks
│   ├── exploratory/                  # EDA notebooks
│   └── visualization/                # Visualization notebooks
├── 🧪 test_simulation.py             # Unit tests
├── 🚀 launch.py                      # Launch script
├── 🎯 train_models.py                # Model training script
├── ⚙️ config.py                      # Configuration settings
├── 📋 requirements.txt               # Python dependencies
└── 📖 README.md                      # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd crime_hotspot_mapping

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Simulation & Dashboard

```bash
# Quick start (5 episodes, launches dashboard)
python launch.py --quick

# Full simulation (50 episodes, launches dashboard)
python launch.py

# Simulation only
python launch.py --simulation-only

# Dashboard only (if simulation data exists)
python launch.py --dashboard-only
```

### 3. Train Models

```bash
# Train all models
python train_models.py

# Preprocess data only
python train_models.py --preprocess-only

# Clustering analysis only
python train_models.py --cluster-only

# Model training only
python train_models.py --train-only
```

### 4. Run Tests

```bash
# Run all tests
python test_simulation.py

# Run with pytest
pytest test_simulation.py -v
```

## 🧠 Agent-Based Simulation

### Agent Types

1. **Offender Agents** (`OffenderAgent`)
   - Q-learning decision making
   - Reputation-based assault probability
   - Risk-aware behavior
   - Actions: move, assault, hide

2. **Target Agents** (`TargetAgent`)
   - Vulnerability and awareness attributes
   - Evasive behavior patterns
   - Actions: move, seek_help, evade

3. **Guardian Agents** (`GuardianAgent`)
   - Patrol efficiency and arrest reputation
   - Proactive crime prevention
   - Actions: patrol, arrest, investigate

### Q-Learning Implementation

```python
# Agent state representation
state = f"{risk_level}_{reputation_level}_{nearby_agents_count}"

# Q-value update
Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

## ⚠️ Risk Terrain Modeling

### Risk Factors

- **POI Density**: Points of interest concentration
- **Road Proximity**: Distance to road networks
- **Lighting Score**: Street lighting coverage
- **Land Use**: Commercial, residential, industrial areas

### Risk Calculation

```python
risk_score = (
    poi_weight * poi_density +
    road_weight * road_proximity +
    lighting_weight * (1 - lighting_score) +
    landuse_weight * landuse_score
)
```

## 📍 Spatial Clustering

### Methods

1. **K-means Clustering**
   - Optimal cluster number detection
   - Silhouette score evaluation
   - Hotspot identification

2. **DBSCAN Clustering**
   - Density-based clustering
   - Noise point identification
   - Adaptive cluster detection

### Hotspot Analysis

```python
# Identify crime hotspots
hotspots = analyzer.identify_hotspots(
    data, cluster_labels, method="kmeans", threshold=0.5
)
```

## 🤖 Machine Learning Models

### Algorithms

1. **Random Forest**
   - Hyperparameter tuning with GridSearchCV
   - Feature importance analysis
   - Robust performance

2. **XGBoost**
   - Gradient boosting implementation
   - Advanced regularization
   - High prediction accuracy

3. **Logistic Regression**
   - Interpretable coefficients
   - Feature selection
   - Baseline comparison

### Evaluation Metrics

- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve

## 📊 Dashboard Features

### Interactive Tabs

1. **🗺️ Hotspot Map**
   - Folium-based interactive maps
   - Crime heatmaps
   - Cluster visualization
   - Risk surface overlay

2. **📍 Cluster View**
   - Cluster analysis results
   - Hotspot identification
   - Spatial distribution

3. **⚠️ Risk Surface**
   - Risk terrain visualization
   - Factor analysis
   - Statistical summaries

4. **🤖 Model Metrics**
   - Model performance comparison
   - Feature importance
   - Confusion matrices

5. **⏰ Time Series**
   - Temporal crime patterns
   - Hourly distributions
   - Agent activity analysis

6. **🎮 Simulation Summary**
   - Simulation statistics
   - Agent distributions
   - Configuration overview

### Filters

- **Time Filters**: Hour range, day of week
- **Risk Filters**: Minimum risk threshold
- **Agent Filters**: Agent type selection
- **Cluster Filters**: Clustering method selection

## ⚙️ Configuration

### Simulation Parameters

```python
SIMULATION_CONFIG = {
    "grid_size": (100, 100),          # Simulation grid
    "episodes": 50,                   # Number of episodes
    "steps_per_episode": 1000,        # Steps per episode
    "learning_rate": 0.1,             # Q-learning rate
    "discount_factor": 0.95,          # Future reward discount
    "exploration_rate": 0.1,          # Exploration probability
    "reputation_decay": 0.99,         # Reputation decay rate
    "risk_threshold": 0.7             # Risk threshold for actions
}
```

### Agent Configuration

```python
AGENT_CONFIG = {
    "offender_count": 20,             # Number of offenders
    "target_count": 50,               # Number of targets
    "guardian_count": 10,             # Number of guardians
    "movement_range": 3,              # Movement range
    "vision_range": 5,                # Vision range
    "base_offense_probability": 0.3,  # Base assault probability
    "base_arrest_probability": 0.4    # Base arrest probability
}
```

## 📈 Data Flow

```
1. Simulation → Raw JSON logs
2. Preprocessing → Feature engineering
3. Clustering → Hotspot identification
4. ML Training → Model evaluation
5. Dashboard → Interactive visualization
```

## 🧪 Testing

### Unit Tests

- Agent behavior testing
- Risk model validation
- Clustering algorithm verification
- ML model evaluation
- Data preprocessing validation

### Test Coverage

```bash
# Run tests with coverage
pytest test_simulation.py --cov=. --cov-report=html
```

## 📊 Performance Metrics

### Simulation Performance

- **Episode Duration**: ~30-60 seconds per episode
- **Memory Usage**: ~500MB for 50 episodes
- **Data Generation**: ~50,000 events per episode

### Model Performance

- **Random Forest**: F1-Score ~0.80-0.85
- **XGBoost**: F1-Score ~0.82-0.87
- **Logistic Regression**: F1-Score ~0.75-0.80

## 🔧 Customization

### Adding New Agent Types

```python
class CustomAgent(BaseAgent):
    def __init__(self, agent_id, position):
        super().__init__(agent_id, position, "custom")
        # Add custom attributes
    
    def decide_action(self, environment_state):
        # Implement custom decision logic
        return action_result
```

### Modifying Risk Factors

```python
# In risk_model.py
def _calculate_risk_scores(self):
    self.risk_grid = (
        custom_weight * custom_factor +
        # ... other factors
    )
```

### Adding New ML Models

```python
# In model_train.py
def train_custom_model(self, X, y):
    # Implement custom model training
    return results
```

## 🚀 Deployment

### Local Development

```bash
# Development mode
python launch.py --quick --verbose

# Production mode
python launch.py --episodes 100 --headless
```

### Streamlit Cloud Deployment

1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Set deployment configuration
4. Deploy dashboard

### Docker Deployment

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/dashboard.py"]
```

## 📚 Research Applications

### Academic Use Cases

1. **Urban Planning**: Crime prevention strategies
2. **Law Enforcement**: Resource allocation optimization
3. **Social Science**: Behavioral pattern analysis
4. **Computer Science**: Multi-agent system research

### Publications

This project can support research in:
- Agent-based modeling
- Spatial analysis
- Machine learning applications
- Urban crime prediction
- Risk assessment methodologies

## 🤝 Contributing

### Development Workflow

1. Fork the repository
2. Create feature branch
3. Implement changes
4. Add tests
5. Submit pull request

### Code Standards

- Follow PEP 8 style guide
- Add docstrings to functions
- Include type hints
- Write unit tests
- Update documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Roysambu Ward**: Study area and context
- **Agent-Based Modeling**: Simulation methodology
- **Risk Terrain Modeling**: Spatial analysis approach
- **Streamlit**: Dashboard framework
- **Open Source Community**: Libraries and tools

## 📞 Contact

For questions, suggestions, or collaboration opportunities:

- **Project Repository**: [GitHub Link]
- **Documentation**: [Documentation Link]
- **Issues**: [GitHub Issues]

---

**Note**: This project uses synthetic data for research purposes. The simulation results are not based on actual crime data and should not be used for real-world crime prediction without proper validation and real data sources. 