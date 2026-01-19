import requests
import json
import numpy as np

def test_lambda():
    url = "http://localhost:8080/2015-03-31/functions/function/invocations"
    
    # 7 features: Ret_1d, Ret_5d, Ret_20d, Vol_20d, RSI, MACD, MACD_Signal
    # Dummy values
    features = [0.01, 0.02, 0.05, 0.01, 55.0, 0.5, 0.4]
    
    payload = {
        "body": json.dumps({"features": features})
    }
    
    try:
        response = requests.post(url, json=payload)
        print("Status Code:", response.status_code)
        print("Response:", response.text)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_lambda()
