import numpy as np
import onnxruntime as ort
from PIL import Image
from io import BytesIO
from urllib import request
import os

# Load model outside handler
MODEL_NAME = "hair_classifier_empty.onnx"
model = ort.InferenceSession(MODEL_NAME)
input_name = model.get_inputs()[0].name
output_name = model.get_outputs()[0].name

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_input(x):
    x = np.array(x, dtype='float32')
    x /= 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    std = np.array([0.229, 0.224, 0.225], dtype='float32')
    x = (x - mean) / std
    return x

def predict(url):
    img = download_image(url)
    img_resized = prepare_image(img, (200, 200)) # HW8 target size
    X = preprocess_input(img_resized)
    
    # N, C, H, W
    X = X.transpose(2, 0, 1)
    X = np.expand_dims(X, axis=0)
    
    outputs = model.run([output_name], {input_name: X})
    return float(outputs[0][0][0])

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result
