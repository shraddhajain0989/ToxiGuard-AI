import os
import logging
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
# Initialize basic logger for module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the trained machine learning model once at the module level
try:
    logging.info("Loading ToxiGuard ML model...")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'toxiguard_model.pkl')
    model = joblib.load(MODEL_PATH)
    try:
        explainer = shap.TreeExplainer(model)
        logging.info("SHAP Explainer loaded successfully.")
    except Exception as shap_e:
        logging.error(f"Failed to load SHAP explainer: {shap_e}")
        explainer = None
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    model = None
    explainer = None

def validate_smiles(smiles):
    """
    Validates a SMILES string and returns an RDKit Mol object.

    Args:
        smiles (str): The SMILES string of the molecule.

    Returns:
        rdkit.Chem.rdchem.Mol or None: Valid RDKit molecule object if successful, safely returns None otherwise.
    """
    try:
        if not isinstance(smiles, str) or not smiles.strip():
            return None
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception as e:
        logging.error(f"Error validating SMILES '{smiles}': {e}")
        return None

def featurize(smiles):
    """
    Generates chemical features and Morgan fingerprints for a validated SMILES string.

    Args:
        smiles (str): The SMILES string of the molecule.

    Returns:
        np.ndarray or None: Extracted feature array, or None if SMILES is invalid.
    """
    mol = validate_smiles(smiles)
    if mol is None:
        return None

    try:
        # Extract basic RDKit features
        features = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol)
        ]

        # Extract Morgan Fingerprints
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fp_array = np.array(fp)

        # Concatenate array and reshape for the ML model input
        return np.concatenate([features, fp_array]).reshape(1, -1)
    except Exception as e:
        logging.error(f"Error featurizing SMILES '{smiles}': {e}")
        return None

def predict_toxicity(smiles):
    """
    Predicts the toxicity of a chemical compound given its SMILES string.

    Args:
        smiles (str): The SMILES string of the molecule.

    Returns:
        tuple (int/float, float) or (None, None): Tuple of predicted class and probability.
        Returns (None, None) if SMILES is invalid or prediction fails.
    """
    if model is None:
        logging.error("Toxicity model is not loaded. Cannot predict.")
        return None, None

    features = featurize(smiles)

    if features is None:
        return None, None

    try:
        # Infer toxicity prediction and probability
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

        return pred, prob
    except Exception as e:
        logging.error(f"Prediction failed for SMILES '{smiles}': {e}")
        return None, None

def generate_shap_plot(features):
    """
    Generates a SHAP force plot image as a base64 string for the given features.
    
    Args:
        features (np.ndarray): Extracted feature array for a single molecule.
        
    Returns:
        str: Base64 string of the static PNG image, or None if failed.
    """
    if explainer is None or features is None:
        return None
        
    try:
        shap_values = explainer.shap_values(features)
        
        # If model outputs 2D shap values (e.g. some RF/XGB setups), take the positive class (toxic)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
            base_val = explainer.expected_value[1]
        else:
            sv = shap_values[0]
            base_val = explainer.expected_value
            
        feature_names = ['MolWt', 'LogP', 'H-Donors', 'H-Acceptors', 'TPSA'] + [f'FP_{i}' for i in range(1024)]
        
        plt.figure(figsize=(10, 3))
        # Use shap.plots.waterfall or a custom bar plot if waterfall is not perfectly compatible
        # shap.plots.force is standard, but matplotlib version for force plot
        # For simplicity and style, we quickly do a bar plot of top 10 features
        
        # Top 10 absolute impact
        abs_sv = np.abs(sv)
        top_indices = np.argsort(abs_sv)[-10:]
        top_sv = sv[top_indices]
        top_names = [feature_names[i] for i in top_indices]
        
        colors = ['#ff4d4f' if val > 0 else '#52c41a' for val in top_sv] # Red for +toxicity, Green for -toxicity
        
        plt.barh(top_names, top_sv, color=colors)
        plt.xlabel("SHAP Value (Impact on Model Output)", color='white')
        plt.title("Top Feature Contributions for this Molecule", color='white', pad=20)
        
        # Styling for dark mode dashboard
        ax = plt.gca()
        ax.set_facecolor('none')
        plt.gcf().patch.set_facecolor('none')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_visible(False) 
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True, dpi=120)
        plt.close()
        buf.seek(0)
        
        # Encode to base64
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return plot_base64
    except Exception as e:
        logging.error(f"Failed to generate SHAP plot: {e}")
        plt.close()
        return None

def format_prediction_result(pred, prob, smiles):
    """
    Helper function to format the prediction result into a structured JSON dict.
    
    Args:
        pred: prediction label (int)
        prob: prediction probability (float)
        smiles: input sequence
        
    Returns:
        dict: Standardized UI keys.
    """
    if pred is None or prob is None:
        return {
            "smiles": smiles,
            "score": "N/A",
            "explanation": "Invalid SMILES",
            "result": "Error",
            "color": "secondary"
        }
        
    # Scale float to integer percentage
    score = int(prob * 100)
    
    if score > 70:
        explanation = "High risk"
    elif score >= 40:
        explanation = "Moderate risk"
    else:
        explanation = "Low risk"
        
    return {
        "smiles": smiles,
        "score": score,
        "explanation": explanation,
        "result": "Toxic ⚠️" if pred == 1 else "Non-Toxic ✅",
        "color": "danger" if pred == 1 else "success"
    }

def process_batch_predictions(file_stream):
    """
    Reads a CSV file-like object using Pandas, extracts the proper SMILES column,
    and returns a structured list of prediction dictionaries.

    Args:
        file_stream (FileStorage): The incoming file from user POST Request.

    Returns:
        tuple (list, str or None): A list of dictionaries representing valid outputs; string if an error occurs.
    """
    try:
        df = pd.read_csv(file_stream)
        
        # Check for 'smiles' column case-insensitively
        smiles_col = None
        for col in df.columns:
            if col.lower().strip() == 'smiles':
                smiles_col = col
                break
        
        if not smiles_col:
            return None, "CSV must contain a 'SMILES' column."

        results = []
        for _, row in df.iterrows():
            smiles_val = str(row[smiles_col])
            
            # Retrieve probability arrays
            pred, prob = predict_toxicity(smiles_val)
            
            # Assemble properly formatted output template
            res_dict = format_prediction_result(pred, prob, smiles_val)
            results.append(res_dict)
            
        return results, None
    except Exception as e:
        logging.error(f"Failed processing batch CSV payload: {e}")
        return None, f"Error processing CSV: {str(e)}"