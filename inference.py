import os
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")

def run_inference():
    print("Starting inference...")

    res = requests.get(f"{BASE_URL}/reset")
    state = res.json()

    providers = state["providers"]
    sla = state["sla_max_latency"]

    valid = {k: v for k, v in providers.items() if v["latency"] <= sla}

    if valid:
        action = min(valid, key=lambda x: valid[x]["cost"])
    else:
        action = min(providers, key=lambda x: providers[x]["latency"])

    res = requests.post(f"{BASE_URL}/step", json={"action": action})
    result = res.json()

    print("Result:", result)
    return result


if __name__ == "__main__":
    run_inference()