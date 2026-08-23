from pathlib import Path
from urllib.request import urlretrieve


BASE_DIR = Path(__file__).resolve().parent
PREFIX = "https://github.com/alexeygrigorev/large-datasets/releases/download/hairstyle"
FILES = {
    "hair_classifier_v1.onnx": f"{PREFIX}/hair_classifier_v1.onnx",
    "hair_classifier_v1.onnx.data": f"{PREFIX}/hair_classifier_v1.onnx.data",
}


def download_models():
    for filename, url in FILES.items():
        destination = BASE_DIR / filename
        if destination.exists():
            print(f"Using existing {filename}")
            continue
        print(f"Downloading {filename}...")
        urlretrieve(url, destination)


if __name__ == "__main__":
    download_models()
