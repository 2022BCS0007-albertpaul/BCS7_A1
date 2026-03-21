import json

def load_data():
    with open("data/processed_data.json") as f:
        return json.load(f)