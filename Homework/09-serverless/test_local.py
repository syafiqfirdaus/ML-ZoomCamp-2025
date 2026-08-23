import requests


IMAGE_URL = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
response = requests.post(
    "http://localhost:9696/predict",
    json={"url": IMAGE_URL},
    timeout=15,
)
print(response.status_code, response.text)
response.raise_for_status()
