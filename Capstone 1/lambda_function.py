import json
import numpy as np
import joblib
import onnxruntime as ort
import os

# Load model and scaler
scaler_path = 'scaler.bin'
model_path = 'best_model.onnx'

print("Loading scaler and ONNX model...")
try:
    scaler = joblib.load(scaler_path)
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("Loaded successfully.")
except Exception as e:
    print(f"Error loading: {e}")
    scaler = None
    session = None

def handler(event, context):
    if session is None or scaler is None:
         return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Model/Scaler not loaded'})
        }

    try:
        body = json.loads(event['body'])
        if 'features' not in body:
             return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Input must contain "features" list'})
            }
        
        features = body['features']
        features = np.array(features).reshape(1, -1)
        
        # Scale
        features_scaled = scaler.transform(features).astype(np.float32)
        
        # Predict
        # ONNX run returns list of outputs
        prediction = session.run([output_name], {input_name: features_scaled})[0][0][0]
        
        score = float(prediction)
        label = "Attractive" if score > 0.5 else "Unattractive"
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'prediction_score': score,
                'label': label
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
