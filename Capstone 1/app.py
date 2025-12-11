from flask import Flask, request, jsonify
import joblib
import numpy as np
import onnxruntime as ort
import os

app = Flask(__name__)

# Load Model
MODEL_PATH = 'best_model.onnx'
SCALER_PATH = 'scaler.bin'

print("Loading scaler and ONNX model...")
try:
    scaler = joblib.load(SCALER_PATH)
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("Loaded successfully.")
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    scaler = None
    session = None

@app.route('/health', methods=['GET'])
def health():
    if session is None:
        return jsonify({'status': 'unhealthy', 'reason': 'model not loaded'}), 500
    return jsonify({'status': 'healthy'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if session is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        if 'features' not in data:
            return jsonify({'error': 'Input must contain "features" list'}), 400
        
        features = data['features']
        features = np.array(features).reshape(1, -1)
        
        # Scale
        features_scaled = scaler.transform(features).astype(np.float32)
        
        # Predict
        prediction = session.run([output_name], {input_name: features_scaled})[0][0][0]
        score = float(prediction)
        
        label = "Attractive" if score > 0.5 else "Unattractive"
        
        return jsonify({
            'prediction_score': score,
            'label': label
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9696)
