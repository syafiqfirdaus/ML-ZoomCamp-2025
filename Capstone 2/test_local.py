import requests


def test_local_api():
    url = "http://localhost:9696/predict"
    features = [0.01, 0.02, 0.05, 0.01, 55.0, 0.5, 0.4]

    response = requests.post(url, json={"features": features}, timeout=10)
    print("Status Code:", response.status_code)
    print("Response:", response.text)
    response.raise_for_status()


if __name__ == "__main__":
    test_local_api()
