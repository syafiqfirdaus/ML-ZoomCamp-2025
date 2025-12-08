import numpy as np
import onnxruntime as ort
from PIL import Image
from io import BytesIO
from urllib import request
import os

# Install dependencies if needed (assuming pip is available in shell, but here we just import)
# !pip install onnxruntime pillow numpy

PREFIX = "https://github.com/alexeygrigorev/large-datasets/releases/download/hairstyle"
DATA_URL = f"{PREFIX}/hair_classifier_v1.onnx.data"
MODEL_URL = f"{PREFIX}/hair_classifier_v1.onnx"

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        request.urlretrieve(url, filename)
        print("Done.")
    else:
        print(f"{filename} already exists.")

download_file(DATA_URL, "hair_classifier_v1.onnx.data")
download_file(MODEL_URL, "hair_classifier_v1.onnx")

print("\n--- Question 1 ---")
model = ort.InferenceSession("hair_classifier_v1.onnx")
input_name = model.get_inputs()[0].name
output_name = model.get_outputs()[0].name
print(f"Input name: {input_name}")
print(f"Output name: {output_name}")

print("\n--- Question 2 ---")
print("Target size: 200x200 (from HW8)")

print("\n--- Question 3 ---")
image_url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"

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

img = download_image(image_url)
img_resized = prepare_image(img, (200, 200))

def preprocess_input(x):
    x = np.array(x, dtype='float32')
    x /= 255.0
    # Normalize mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    std = np.array([0.229, 0.224, 0.225], dtype='float32')
    
    x = (x - mean) / std
    return x

X = preprocess_input(img_resized)
print(f"First pixel R channel: {X[0, 0, 0]}")

print("\n--- Question 4 ---")
# Transpose from H, W, C to N, C, H, W (1, 3, 200, 200)
X_transposed = X.transpose(2, 0, 1) # C, H, W
X_batch = np.expand_dims(X_transposed, axis=0) # N, C, H, W

outputs = model.run([output_name], {input_name: X_batch})
print(f"Model output: {outputs[0][0][0]}")
