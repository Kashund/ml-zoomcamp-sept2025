import requests

url = "http://127.0.0.1:9696/predict"
payload = {
  "type": "TV",
  "season": "Spring",
  "year": 2025,
  "episodes": 12,
  "source": "Manga",
  "rating": "PG-13",
  "status": "Upcoming",
  "genres": ["Action", "Sci-Fi"],
  "themes": [],
  "demographics": ["Shounen"],
  "studios": []
}

print("POST", url)
r = requests.post(url, json=payload, timeout=10)
print("status:", r.status_code)
print(r.json())