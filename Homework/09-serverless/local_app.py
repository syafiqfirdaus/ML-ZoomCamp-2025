from flask import Flask, jsonify, request

from inference import predict


app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


@app.post("/predict")
def predict_endpoint():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("url"), str):
        return jsonify({"error": 'Input must contain an image "url"'}), 400

    try:
        return jsonify({"prediction": predict(data["url"])})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696)
