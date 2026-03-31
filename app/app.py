import os
from flask import Flask, render_template, request
from utils import predict_toxicity, format_prediction_result, process_batch_predictions, featurize, generate_shap_plot

app = Flask(__name__)

# Compute accurate absolute paths based on app.py's native location
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SHAP_IMG_PATH = os.path.join(APP_DIR, 'static', 'shap_summary.png')

def get_shap_plot_path():
    """Helper function to check if SHAP plot exists and return its path."""
    return True if os.path.exists(SHAP_IMG_PATH) else False

@app.route('/')
def home():
    """Renders the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles single SMILES string toxicity prediction."""
    smiles = request.form.get('smiles', '').strip()

    if not smiles:
        return render_template('index.html', error="No SMILES provided.")

    # Get raw model outputs
    pred, prob = predict_toxicity(smiles)

    if pred is None or prob is None:
        return render_template('index.html', error="Invalid SMILES ❌. Please enter a valid chemical structure.")

    # Generate dynamic SHAP plot
    features = featurize(smiles)
    dynamic_shap = generate_shap_plot(features) if features is not None else None

    # Format output for the UI
    res = format_prediction_result(pred, prob, smiles)

    return render_template(
        'index.html',
        result=res['result'],
        score=res['score'],
        smiles=res['smiles'],
        color=res['color'],
        explanation=res['explanation'],
        shap_plot=get_shap_plot_path(),
        dynamic_shap_base64=dynamic_shap
    )

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Handles batch prediction via a CSV file upload."""
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded.")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected.")

    if not file.filename.endswith('.csv'):
        return render_template('index.html', error="Please upload a valid CSV file.")

    # Process the batch using utils logic
    results, error_msg = process_batch_predictions(file)
    
    if error_msg:
        return render_template('index.html', error=error_msg)
        
    return render_template(
        'index.html',
        batch_results=results,
        shap_plot=get_shap_plot_path()
    )

if __name__ == '__main__':
    app.run(debug=True)