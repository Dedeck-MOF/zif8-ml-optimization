# zif8-ml-optimization
Experimental dataset and Python code for independently validating the machine learning analyses
# Machine Learning-Driven Optimization of ZnO to ZIF-8 Membrane Conversion

[![DOI](https://img.shields.io/badge/DOI-10.xxxx/xxxxx-blue)](https://doi.org/10.xxxx/xxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains the code and data for the machine learning analysis presented in:

> **Machine Learning-Driven Optimization of ZnO to ZIF-8 Membrane Conversion: A Data-Driven Approach for High-Quality Metal-Organic Framework Synthesis**
> 
> [Author names]
> 
> *ACS Applied Materials & Interfaces* (2025)

## Independent Validation

The original machine learning analysis was performed using WEKA 3.8. This Python implementation using scikit-learn serves as an **independent validation**, confirming that the main findings are robust across different ML frameworks.

### Validation Results

| Classifier | WEKA (Original) | Python (This repo) | Status |
|------------|-----------------|-------------------|--------|
| k-NN (k=1) | 92.65% | 92.65% | ✓ Exact match |
| k-NN (k=3) | 91.18% | 91.18% | ✓ Exact match |
| k-NN (k=5) | 92.65% | 91.18% | ✓ Within CV variance |
| Random Forest | 92.65% | 92.65% | ✓ Exact match |
| Decision Tree | 85.29% | 86.76% | ✓ Within CV variance |
| ZeroR (baseline) | 76.47% | 76.47% | ✓ Exact match |

**Note:** MLP and Naive Bayes classifiers are not included in this validation due to fundamental algorithmic differences between WEKA and scikit-learn implementations that affect reproducibility. The key findings regarding k-NN and Random Forest performance are fully validated.

## Repository Structure

```
├── zif8_ml_analysis.py      # Main analysis script
├── zif8_synthesis_data.csv  # Experimental dataset (68 instances)
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── LICENSE                  # MIT License
```

## Dataset

The dataset (`zif8_synthesis_data.csv`) contains 68 ZnO-to-ZIF-8 conversion experiments with the following features:

| Feature | Description | Range/Values |
|---------|-------------|--------------|
| `Solvent_1` | Primary solvent | MeOH, EtOH, H₂O, DMF |
| `Solvent_2` | Secondary solvent (co-solvent) | MeOH, EtOH, H₂O |
| `Ratio` | Volumetric ratio (Solvent_1:Solvent_2) | 0.5 - 4.0 |
| `Temperature` | Reaction temperature (°C) | 25 - 200 |
| `Duration` | Reaction duration (hours) | 0.017 - 24 |
| `Quality` | Membrane quality classification | High (≥85% coverage), Low (<85%) |

### Class Distribution

- High quality: 52 instances (76.5%)
- Low quality: 16 instances (23.5%)
- Class ratio: 3.25:1

### Fixed Experimental Parameters

- 2-Methylimidazole concentration: 1% w/v
- ZnO film thickness: ~50 nm (250 ALD cycles)
- ALD deposition temperature: 100°C
- Substrate: Monocrystalline silicon wafer

## Installation

1. Clone the repository:
```bash
git clone https://github.com/[username]/zif8-ml-optimization.git
cd zif8-ml-optimization
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the complete analysis:
```bash
python zif8_ml_analysis.py
```

This will:
1. Load and preprocess the dataset
2. Evaluate classifier configurations using 10-fold stratified cross-validation
3. Perform SMOTE augmentation analysis
4. Compute feature importance using multiple methods
5. Generate decision tree visualization
6. Save results to CSV files

## Output Files

The script generates the following output files:

| File | Description |
|------|-------------|
| `classifier_comparison_results.csv` | Performance metrics for all classifiers |
| `feature_importance.csv` | Feature importance scores from multiple methods |
| `decision_tree.png` | Decision tree visualization (PNG format) |
| `decision_tree.pdf` | Decision tree visualization (PDF format) |

## Key Results

### Classifier Performance (n=68, 10-fold CV)

| Classifier | Accuracy | Kappa | Recall (Low) | ROC-AUC |
|------------|----------|-------|--------------|---------|
| k-NN (k=1) | 92.65% | 0.781 | 75.0% | 0.865 |
| k-NN (k=5) | 91.18% | 0.744 | 75.0% | 0.900 |
| Random Forest | 92.65% | 0.781 | 75.0% | 0.889 |
| Decision Tree | 86.76% | 0.588 | 56.2% | 0.762 |
| ZeroR (baseline) | 76.47% | 0.000 | 0.0% | 0.500 |

### Main Conclusions Validated

1. **k-NN and Random Forest achieve equivalent high performance (~92.6% accuracy)**
2. **Both methods significantly outperform the baseline (76.5%)**
3. **SMOTE augmentation improves minority class detection**
4. **Primary solvent selection is a critical predictor of membrane quality**

## Reproducibility

This script ensures reproducibility through:
- Fixed random seed (seed=1, matching WEKA default)
- Stratified cross-validation preserving class distribution
- Version-controlled dependencies
- One-hot encoding for categorical variables (matching WEKA nominal handling)
- MinMax normalization for distance-based methods

## Citation

If you use this code or data, please cite:

```bibtex
@article{author2025zif8ml,
  title={Machine Learning-Driven Optimization of ZnO to ZIF-8 Membrane Conversion},
  author={[Authors]},
  journal={ACS Applied Materials \& Interfaces},
  year={2025},
  doi={10.xxxx/xxxxx}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact [corresponding author email].

## Acknowledgments

This work was performed at Institut Européen des Membranes (IEM), Université de Montpellier, CNRS, ENSCM, Montpellier, France.
