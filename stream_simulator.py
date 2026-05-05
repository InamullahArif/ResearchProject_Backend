import os
import time
import random
import requests
from datetime import datetime, timedelta

# Use /score (legacy) or /predict-single for JSON { fraud_probability, label, ... }
API = "http://127.0.0.1:8000/predict-single"

users = ["U1", "U2", "U3"]
merchants = ["M1", "M2", "M3"]
cats = ["grocery", "electronics", "transfer", "fuel"]

t = datetime.now()

for i in range(200):
    t = t + timedelta(seconds=random.randint(5, 60))
    amount = round(random.uniform(5, 500), 2)

    # inject some suspicious spikes
    if random.random() < 0.05:
        amount = round(random.uniform(1500, 5000), 2)

    payload = {
        "timestamp": t.isoformat(),
        "user_id": random.choice(users),
        "amount": amount,
        "merchant": random.choice(merchants),
        "category": random.choice(cats),
        "location": "PK",
        "device_type": "mobile",
        "channel": "online"
    }

    headers = {}
    key = os.environ.get("FRAUD_API_KEY", "").strip()
    if key:
        headers["X-API-Key"] = key
    r = requests.post(API, json=payload, headers=headers, timeout=10)
    print(i, payload["user_id"], payload["amount"], r.json())
    time.sleep(0.2)