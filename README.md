<h1 align="center">🧪 ToxiGuard AI</h1>

<p align="center">
  <strong>AI-Powered Drug Toxicity Prediction Platform</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Flask-Web%20App-lightgrey?style=for-the-badge&logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/XGBoost-ML%20Model-orange?style=for-the-badge&logo=xgboost" alt="XGBoost">
  <img src="https://img.shields.io/badge/RDKit-Cheminformatics-green?style=for-the-badge" alt="RDKit">
</p>

---

## 📖 Overview
**ToxiGuard AI** is an advanced, web-based cheminformatics tool designed to predict the toxicity of chemical compounds instantaneously. By leveraging robust Machine Learning (XGBoost) and state-of-the-art chemical feature extraction (RDKit), ToxiGuard AI provides pharmaceutical researchers with rapid safety profiles—saving critical time and resources in the early stages of drug discovery.

## ⚠️ Problem Statement
In traditional drug discovery, identifying the toxicity of a chemical compound requires extensive, expensive, and time-consuming *in vitro* and *in vivo* testing. High failure rates due to unforeseen toxicity are a leading cause of attrition in clinical trials. A fast, reliable, computational pre-screening step is essential to filter out high-risk candidates before physical synthesis begins.

## 💡 Solution Approach
ToxiGuard AI tackles this by framing toxicity prediction as a machine learning classification problem:
1. **Feature Engineering**: SMILES strings are parsed into molecular objects using `RDKit`. Important descriptors (Molecular Weight, LogP, H-Donors/Acceptors, TPSA) and high-dimensional Morgan Fingerprints are extracted.
2. **Modeling**: An `XGBoost` classifier, trained on the comprehensive **Tox21 dataset**, correlates these molecular and topological features with known toxicological outcomes.
3. **Serving**: A lightweight `Flask` backend serves the model, providing both an intuitive web interface for single interactive predictions and a batch API for high-throughput screening via CSV uploads.
4. **Explainability**: Using `SHAP` (SHapley Additive exPlanations), the model provides transparent, interpretable visual insights into *why* a compound was flagged as toxic.

## ⚙️ Tech Stack
*   **Backend / API**: Flask, Python
*   **Machine Learning**: XGBoost, Scikit-Learn, SHAP
*   **Cheminformatics**: RDKit
*   **Data Processing**: Pandas, NumPy
*   **Frontend UI**: HTML5, Jinja2 templating, Bootstrap 5

## ✨ Features
*   **🔍 Single SMILES Prediction**: Instantly assess individual compounds with real-time risk scores and color-coded flags.
*   **📂 Batch CSV Processing**: Upload datasets of multiple SMILES representations to evaluate hundreds of molecules simultaneously.
*   **📊 SHAP Explainability**: View global and sample-level feature importance to interpret model decisions scientifically.
*   **🛡️ Robust Error Handling**: Clean, validated backend logic seamlessly handles invalid or malformed SMILES strings safely.

## 📸 Screenshots

*(Replace placeholders with actual images of your application)*

| Single Prediction Interface | Batch Results Dashboard | Model Explainability (SHAP) |
|:---:|:---:|:---:|
| ![Single Prediction](https://via.placeholder.com/400x250?text=Single+Prediction) | ![Batch CSV Upload](https://via.placeholder.com/400x250?text=Batch+CSV+Upload) | ![SHAP Summary Plot](https://via.placeholder.com/400x250?text=SHAP+Summary+Plot) |

## 🚀 Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ToxiGuard-AI.git
   cd ToxiGuard-AI
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure the trained model exists:**
   Make sure the `toxiguard_model.pkl` file is present in the `models/` directory. If it isn't, you can run the Jupyter notebooks in `notebooks/` to regenerate it.

## 💻 How to Run

1. **Start the Flask Server:**
   Navigate into the application directory and run the main app.
   ```bash
   cd app
   python app.py
   ```

2. **Access the Web Interface:**
   Open your browser and navigate to the localhost address:
   ```
   http://127.0.0.1:5000/
   ```

---
*Built with ❤️ for AI & Healthcare Hackathons.*
