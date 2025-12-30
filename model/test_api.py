import requests
import json

url = "http://127.0.0.1:5000/predict"


models = ['kmeans', 'dbscan', 'gmm']
for model in models:
    payload = {
        "model": model,
        "features": {
            "pc1": 0.12,
            "pc2": -0.45,
            "pc3": 0.78,
            "pc4": -0.23,
            "pc5": 0.56,
            "pc6": -0.89,
            "pc7": 0.34,
            "pc8": -0.67,
            "pc9": 0.9,
            "pc10": -0.12
        }
    }
    try:
        print(f"\nTesting {model.upper()}:")
        print(f"Sending payload: {json.dumps(payload, indent=2)}")
        response = requests.post(url, json=payload)
        print("Status Code:", response.status_code)
        print("Response:", response.json())
    except requests.exceptions.ConnectionError as e:
        print(f"Error: Connection refused. Make sure the server (api.py) is running.")
    except Exception as e:
        print(f"An error occurred: {e}")