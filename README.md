# Data-based Modeling of Slug Flow Crystallization with Uncertainty Quantification

##  Introduction

This project implements a data-driven modeling workflow for the **Slug Flow Crystallization (SFC)** process using **Artificial Neural Networks (ANNs)** and **Uncertainty Quantification** techniques. By replacing complex first-principles models with cluster-specific **NARX (Nonlinear AutoRegressive with eXogenous inputs)** architectures and applying **Conformalized Quantile Regression (CQR)**, the project aims to deliver fast, accurate, and interpretable process predictions with reliable prediction intervals.

Additional investigations test the robustness of open-loop predictions and alternative uncertainty estimation approaches.

---

##  Table of Contents

- [Introduction](#-introduction)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Features](#-features)
- [Data](#-data)
- [Results](#-results)
- [Configuration](#-configuration)
- [Examples](#-examples)
- [Troubleshooting](#-troubleshooting)
- [Contributors](#-contributors)
- [License](#-license)

---

## 📁 Repository Structure

```plaintext
├── Beat-the-Felix/              # Beat-the-Felix competition setup
├── CleanedTxtFiles/            # Cleaned text files from raw experiments
├── Data/                       # Raw and preprocessed datasets
├── Results/                    # Trained models, plots, and result outputs
├── MLME_Final1.ipynb           # Main workflow notebook
├── Additional_Task_1.ipynb     # Open-loop ANN prediction analysis
├── Additional_Task_3.ipynb     # Alternative uncertainty quantification
├── BeatTheFelix.ipynb          # Competition evaluation
├── Group 11_Report_MLME.pdf    # Final report
├── AI_Usage_Report.pdf         # Declaration of AI tool usage
├── README.md                   # This README file
```

---

## ⚙️ Installation

1. **Clone this repository:**

```bash
git clone https://github.com/yourusername/mlme25-sfc-modeling.git
cd mlme25-sfc-modeling
```

2. **Create a virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is not available, make sure the following are installed:
> - `numpy`, `pandas`, `matplotlib`, `scikit-learn`
> - `torch`, `seaborn`, `statsmodels`, `notebook`

---

##  Usage

### 🔄 Main Pipeline Execution

Run the full pipeline:

```bash
jupyter notebook MLME_Final1.ipynb
```

This will:
- Load and preprocess the data
- Perform PCA and clustering
- Train cluster-specific NARX models
- Apply conformal prediction
- Save results to `/Results/`

###  Additional Analyses

- **Open-loop prediction stability:**
  ```bash
  jupyter notebook Additional_Task_1.ipynb
  ```

- **Alternative uncertainty estimation methods:**
  ```bash
  jupyter notebook Additional_Task_3.ipynb
  ```

###  Beat-the-Felix Competition

- Place test files in `Beat-the-Felix/`
- Run:
  ```bash
  jupyter notebook BeatTheFelix.ipynb
  ```

---

##  Features

- Lightweight modeling of nonlinear crystallization dynamics
- Cluster-specific NARX neural networks
- Conformal Quantile Regression (CQR) for reliable uncertainty bounds
- PCA and KMeans for dimensionality reduction and clustering
- Extendable to other dynamic chemical systems

---

##  Data

The `/Data/` folder contains raw and cleaned input variables including:
- **Temperature**: `T_PM`, `T_TM`
- **Concentration**: `c_PM`
- **Crystal Size Distribution**: `d10`, `d50`, `d90`
- Additional features used for clustering and prediction.

Preprocessed data is stored in `cleaned_data.csv`.

---

## 📈 Results

The `/Results/` directory includes:
- Trained PyTorch `.pt` models
- Quantile regressors
- Cluster assignments
- Model calibration/test/validation sets
- Performance visualizations (`.png`)
- MSE metrics and CSV summaries

---

## 🛠 Configuration

Model and experiment configurations are defined directly in the Jupyter notebooks. Key parameters:
- NARX architecture: hidden layers, lag order
- Clustering: number of clusters
- CQR parameters: quantile levels (e.g., 0.05, 0.95)

You can customize these directly in `MLME_Final1.ipynb`.

---

##  Examples

See the output cells in `MLME_Final1.ipynb` for:
- Data preprocessing
- PCA and cluster visualizations
- Prediction vs. ground truth plots
- Prediction intervals using CQR

---

##  Troubleshooting

| Issue                            | Solution |
|----------------------------------|----------|
| FileNotFoundError for CSV files | Ensure paths are correct and files are in `/Data/` |
| ModuleNotFoundError              | Check `pip install` for required libraries |
| Long execution time              | Consider using smaller data subsets for quick tests |
| Output folders not found         | Manually create `Results/` if not auto-generated |

---

## 👥 Contributors

This project was developed by :

- Tej Deepak Amin 
- Shamita Nalamutt 
- Prabhav Patel 
- Anvita Koyande 

---

## 📄 License

This project is shared for academic use. Please contact the authors for reuse or redistribution rights.
