import requests
import time

# Wait a bit for the service to be ready
time.sleep(1)

url = "http://localhost:8000/predict"
client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

response = requests.post(url, json=client)
result = response.json()

print(f"Response: {result}")
print(f"\nQuestion 6:")
print(f"Probability: {result['probability']:.2f}")
