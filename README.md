# Diabetes 30-Day Readmission Risk Stratification

## Project Overview

This project implements a comprehensive machine learning pipeline for predicting 30-day diabetes readmission risk using the UCI "Diabetes 130-US hospitals" dataset (101,766 encounters). The analysis combines traditional ML approaches with advanced Random Matrix Theory (RMT) diagnostics to understand feature space geometry, implements CUDA GPU acceleration for efficient training, and includes rigorous fairness evaluation across demographic subgroups. Key findings include XGBoost achieving 0.686 AUC with concerning age-based fairness disparities (12.8% accuracy gap), RMT revealing severe feature redundancy (effective rank 24.7/2,232), and a novel mini-experiment demonstrating that RMT-guided dimensionality reduction plus subgroup-aware thresholding can reduce fairness gaps by 83.7%. For detailed clinical findings and analysis results, see [`final_report.md`](final_report.md); for technical architecture and code explanations, see [`report.md`](report.md).

---

## Quick Start

### Prerequisites
- **Python 3.8+**
- **NVIDIA GPU with CUDA support** (tested on RTX 4070 Laptop GPU)
- **CUDA 11.8** (for GPU acceleration)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd diabetes-readmission-analysis
```

2. **Create and activate virtual environment:**
```bash
python -m venv resultenv
# Windows:
resultenv\Scripts\activate
# Linux/Mac:
source resultenv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify CUDA installation (optional but recommended):**
```bash
python cudaCheck.py
```

### Running the Analysis

**Full Analysis Pipeline (main experiment):**
```bash
python main.py
```
*Runtime: ~30-35 minutes with GPU acceleration*

**RMT-Guided Mini-Experiment:**
```bash
python miniModel.py
```
*Runtime: ~10-15 minutes with GPU acceleration*

**Load Pre-trained Models:**
```bash
python load_models_example.py
```

---

## Repository Structure

```
📂 diabetes-readmission-analysis/
├── 📋 README.md                     # This file - setup and usage guide
├── 📊 final_report.md              # Clinical findings and analysis results
├── 📋 report.md                    # Technical architecture and code documentation
├── 📄 diabetic_data.csv            # UCI diabetes dataset (101,766 encounters)
├── 📄 IDS_mapping.csv              # Optional ID mapping file
├── 📦 requirements.txt             # Python dependencies with CUDA packages
├── 🔧 cudaCheck.py                 # CUDA compatibility verification script
│
├── 🐍 main.py                      # Main 8-step analysis pipeline
├── 🧪 miniModel.py                 # RMT-guided reduction + fairness experiment
├── 📜 load_models_example.py       # Example for loading trained models
│
├── 📂 functions/                   # Modular analysis components
│   ├── 🔧 data_preprocessing.py         # Data pipeline & feature engineering
│   ├── 📈 exploratory_analysis.py       # Demographic analysis with Wilson CIs
│   ├── 🔬 rmt_analysis.py               # Random Matrix Theory diagnostics
│   ├── 🤖 model_training.py             # GPU-accelerated model training
│   ├── 🔍 feature_importance.py        # Bootstrap coefficient analysis
│   ├── ⚖️  subgroup_evaluation.py      # Fairness assessment
│   └── 📊 scaling_analysis.py           # Data efficiency curves
│
├── 📂 visualizations/              # Analysis plots (auto-generated, 300 DPI PNG)
│   ├── 01_demographic_readmission_rates_*.png    # Wilson CI demographic analysis
│   ├── 02_overall_covariance_spectrum.png        # RMT eigenspectrum analysis
│   ├── 03_subgroup_spectrum_*.png                # Demographic spectral differences
│   ├── 04_model_comparison_forest_plot.png       # Bootstrap CI model comparison
│   ├── 05_model_roc_curves.png                   # ROC curve analysis
│   ├── 06_feature_importance_coefficients.png    # Bootstrap coefficient analysis
│   ├── 07_subgroup_fairness_analysis.png         # Demographic fairness assessment
│   ├── 08_scaling_curves_analysis.png            # Data efficiency analysis
│   ├── mini_eigenspectrum_analysis.png           # RMT-guided reduction analysis
│   └── mini_fairness_comparison.png              # Subgroup-aware thresholding results
│
└── 📂 weights/                     # Trained model artifacts (auto-generated)
    ├── logistic_weights.pkl             # Logistic regression model
    ├── xgboost_weights.pkl              # XGBoost model
    ├── mlp_weights.pkl                  # Neural network model
    ├── model_results.pkl                # Performance metrics
    └── metadata.pkl                     # Training metadata
```

---

## Key Features

### 🔬 **Advanced Analytics**
- **Random Matrix Theory (RMT)**: Feature space geometry diagnostics using Marchenko-Pastur law
- **Bootstrap Confidence Intervals**: Statistical uncertainty quantification for all metrics
- **Wilson Confidence Intervals**: Exact statistical bounds for demographic proportions
- **Power Law Scaling Analysis**: Data efficiency assessment with theoretical modeling

### ⚡ **GPU Acceleration**
- **CUDA-Enabled XGBoost**: `gpu_hist` tree method for faster training
- **PyTorch Neural Networks**: GPU tensor operations with deterministic results
- **Automatic Fallback**: Graceful degradation to CPU if CUDA unavailable
- **Performance Monitoring**: Real-time GPU utilization and memory tracking

### ⚖️ **Fairness & Bias Analysis**
- **Demographic Parity Assessment**: Accuracy, sensitivity, specificity across age/gender/race
- **Subgroup-Aware Thresholding**: Per-group threshold optimization for fairness
- **Statistical Significance Testing**: Confidence interval overlap detection
- **Bias Mitigation Strategies**: RMT-guided approaches to address disparities

### 🏗️ **Production Engineering**
- **Modular Architecture**: Independent, testable function modules
- **Model Persistence**: Comprehensive serialization with metadata
- **Reproducible Results**: Multi-level seeding (Python/NumPy/PyTorch/CUDA)
- **Error Handling**: Robust index alignment and missing data strategies

---

## Analysis Pipeline (main.py)

The main analysis executes 8 sequential steps:

1. **📂 Data Loading & Preprocessing**: UCI dataset → 101,766 encounters → 2,232 features
2. **📊 Exploratory Analysis**: Demographic readmission patterns with Wilson CIs
3. **🔬 RMT Diagnostics**: Eigenspectrum analysis revealing effective rank 24.7/2,232
4. **🤖 Model Training**: GPU-accelerated XGBoost, Logistic Regression, Neural Network
5. **🔍 Feature Importance**: Bootstrap coefficient analysis with clinical interpretation
6. **⚖️ Fairness Evaluation**: Subgroup performance assessment across demographics
7. **📈 Scaling Analysis**: Data efficiency curves and power law fitting
8. **📋 Report Generation**: Automated summary and publication-quality visualizations

**Expected Output**: 12 high-resolution visualizations, trained model weights, comprehensive analysis report

---

## Mini-Experiment (miniModel.py)

Advanced experiment testing RMT-guided optimization:

- **Dimensionality Reduction**: 2,232 → 25 features using Marchenko-Pastur bounds
- **Subgroup-Aware Thresholding**: Per-demographic threshold optimization
- **Fairness Improvement**: 83.7% reduction in age accuracy gaps
- **Performance Trade-offs**: AUC vs fairness analysis

**Key Innovation**: Demonstrates that feature space geometry insights can guide both compression and fairness improvements.

---

## Dataset Requirements

### Primary Dataset
- **File**: `diabetic_data.csv`
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
- **Size**: 101,766 encounters, 50 features
- **Target**: 30-day readmission (<30 days = positive class, 11.2% prevalence)

### Optional Files
- **File**: `IDS_mapping.csv` (ID mappings, can be empty/minimal)
- **Format**: CSV with any ID mapping information

---

## Hardware Requirements

### Minimum Requirements
- **RAM**: 8GB+ (16GB recommended for full dataset)
- **Storage**: 2GB free space
- **CPU**: Multi-core processor (4+ cores recommended)

### Recommended for GPU Acceleration
- **GPU**: NVIDIA GPU with CUDA Compute Capability 6.0+
- **VRAM**: 4GB+ (tested on RTX 4070 with 12GB)
- **CUDA**: Version 11.8 (other versions may work but not tested)
- **Drivers**: Latest NVIDIA drivers with CUDA support

### Performance Benchmarks
- **CPU-only (16-core)**: ~45-60 minutes (main analysis)
- **GPU-accelerated (RTX 4070)**: ~30-35 minutes (main analysis)
- **Mini-experiment**: ~10-15 minutes (GPU) vs ~20-25 minutes (CPU)

---

## Troubleshooting

### Common Issues

**CUDA Not Found:**
```bash
# Verify CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**Memory Issues:**
```bash
# Reduce batch size in model_training.py
# Or run with fewer bootstrap samples in main.py
```

**Missing Dataset:**
```bash
# Download from UCI repository
# Place diabetic_data.csv in project root
```

**Module Import Errors:**
```bash
# Ensure virtual environment is activated
# Reinstall requirements: pip install -r requirements.txt
```

### Performance Optimization

**For Faster Execution:**
- Reduce bootstrap samples in `main.py` (change `N_BOOTSTRAP = 10` to `N_BOOTSTRAP = 5`)
- Use CPU-only mode if GPU memory limited
- Run mini-experiment only for quick insights

**For Maximum Accuracy:**
- Increase bootstrap samples to 100+
- Use full feature space (avoid dimensionality reduction)
- Run multiple random seeds and average results

---

**Made by**: Atharva Patel (atharvajpatel@berkeley.edu)  
**Generated**: 2025-09-22  
**GPU Tested**: NVIDIA RTX 4070 Laptop GPU  
**CUDA Version**: 11.8  
**Python Version**: 3.8+
