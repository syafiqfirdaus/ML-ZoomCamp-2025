from io import BytesIO
from pathlib import Path
from urllib import request

import numpy as np
import onnxruntime as ort
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hair_classifier_v1.onnx"

if not MODEL_PATH.exists():
    raise FileNotFoundError("Model not found. Run `python download_model.py` first.")

model = ort.InferenceSession(str(MODEL_PATH))
input_name = model.get_inputs()[0].name
output_name = model.get_outputs()[0].name


def download_image(url):
    with request.urlopen(url, timeout=10) as response:
        image = Image.open(BytesIO(response.read()))
        return image.convert("RGB")


def preprocess_image(image):
    image = image.resize((200, 200), Image.Resampling.NEAREST)
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std
    return np.expand_dims(array.transpose(2, 0, 1), axis=0)


def predict(url):
    features = preprocess_image(download_image(url))
    output = model.run([output_name], {input_name: features})
    return float(output[0][0][0])
