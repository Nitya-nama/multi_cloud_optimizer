"""
inference.py — LLM-based inference script for the SLA-Aware Multi-Cloud Cost Optimizer.

Uses the OpenAI client to query a language model to select the optimal cloud provider
for each benchmark task. Reads credentials from environment variables:
  API_BASE_URL  — base URL for the LLM API endpoint
  MODEL_NAME    — model identifier to use for inference
  HF_TOKEN      — Hugging Face / API key

Usage:
    python inference.py
"""

import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Credentials & config ──────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")

# The OpenAI client points at whatever API_BASE_URL is configured.
# For Hugging Face-hosted models the base_url is the HF Inference Endpoints
# URL; authentication uses HF_TOKEN as the API key.
client = OpenAI(
    api_key=HF_TOKEN or "dummy",          # OpenAI client requires a non-empty key
    base_url=API_BASE_URL if API_BASE_URL != "http://localhost:7860" else None,
)

ENV_BASE_URL = os.getenv(
    "ENV_BASE_URL",
    "https://nityanama-multi-cloud-optimizer.hf.space"
)    # Always call the local env server

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_tasks() -> list:
    """Fetch the list of all benchmark tasks from the environment server."""
    resp = requests.get(f"{ENV_BASE_URL}/tasks", timeout=10)
    resp.raise_for_status()
    return resp.json()["tasks"]


def get_task_detail(task_id: str) -> dict:
    """Fetch full provider metrics for a single task."""
    resp = requests.get(f"{ENV_BASE_URL}/tasks/{task_id}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def grade_selection(task_id: str, selected_cloud: str) -> dict:
    """Submit a cloud selection to the grader and return the scored result."""
    resp = requests.post(
        f"{ENV_BASE_URL}/grader",
        json={"task_id": task_id, "selected_cloud": selected_cloud},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def ask_llm(task: dict) -> str:
    providers = task["providers"]
    sla = task["sla_max_latency"]

    # Sort all providers cheapest first
    sorted_providers = sorted(providers.items(), key=lambda x: x[1]["cost"])
    providers_text = "\n".join(
        f"  {name}: cost=${metrics['cost']}, latency={metrics['latency']}ms"
        for name, metrics in sorted_providers
    )

    # Pre-filter compliant providers and sort cheapest first
    compliant = {k: v for k, v in providers.items() if v["latency"] <= sla}
    sorted_compliant = sorted(compliant.items(), key=lambda x: x[1]["cost"])
    compliant_text = "\n".join(
        f"  {name}: cost=${metrics['cost']}, latency={metrics['latency']}ms {'<-- CHEAPEST' if i == 0 else ''}"
        for i, (name, metrics) in enumerate(sorted_compliant)
    ) if compliant else "  (none meet SLA - pick lowest latency)"

    prompt = f"""You are a cloud cost optimizer. Follow these exact steps:

STEP 1: The SLA latency limit is {sla} ms. Any provider with latency > {sla} ms is disqualified.
STEP 2: From the providers that meet the SLA, pick the one with the LOWEST cost.
STEP 3: Reply with ONLY that provider name — one word: aws, azure, or gcp.

All providers (sorted cheapest first):
{providers_text}

Providers that meet the SLA (latency <= {sla} ms), sorted cheapest first:
{compliant_text}

Your answer (one word only):"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0,
    )

    raw = response.choices[0].message.content.strip().lower()

    for provider in ["aws", "azure", "gcp"]:
        if provider in raw:
            return provider

    print(f"  [warn] LLM returned unexpected output: '{raw}' - falling back to greedy")
    return _greedy_fallback(task)


def _greedy_fallback(task: dict) -> str:
    """Cheapest SLA-compliant provider; lowest latency if none comply."""
    providers = task["providers"]
    sla = task["sla_max_latency"]
    valid = {k: v for k, v in providers.items() if v["latency"] <= sla}
    if valid:
        return min(valid, key=lambda p: valid[p]["cost"])
    return min(providers, key=lambda p: providers[p]["latency"])


# ── Main ──────────────────────────────────────────────────────────────────────

def run_inference() -> dict:
    """
    Run the LLM agent on all benchmark tasks and return a summary.

    Returns
    -------
    {
        "model"          : str,
        "tasks_evaluated": int,
        "average_reward" : float,
        "sla_violations" : int,
        "results"        : [ { task_id, selected_cloud, reward, sla_met, ... } ]
    }
    """
    print("=" * 60)
    print(f"  Multi-Cloud Optimizer - LLM inference")
    print(f"  Model : {MODEL_NAME}")
    print(f"  API   : {API_BASE_URL}")
    print("=" * 60)

    tasks   = get_tasks()
    results = []

    for task_summary in tasks:
        task_id = task_summary["task_id"]
        task    = get_task_detail(task_id)

        print(f"\n[{task_id:<18}] Asking LLM...", end=" ", flush=True)

        try:
            selected_cloud = ask_llm(task)
        except Exception as exc:
            print(f"LLM error ({exc}) - using greedy fallback")
            selected_cloud = _greedy_fallback(task)

        graded = grade_selection(task_id, selected_cloud)

        reward  = graded["reward"]
        sla_met = graded["sla_met"]
        cost    = graded["cost"]
        latency = graded["latency"]

        status = "OK" if sla_met else "SLA VIOLATION"
        print(f"-> {selected_cloud:<5}  cost={cost:>6.1f}  lat={latency:>6.1f}ms  reward={reward:.4f}  {status}")

        results.append({
            "task_id"        : task_id,
            "difficulty"     : task_summary["difficulty"],
            "selected_cloud" : selected_cloud,
            "cost"           : cost,
            "latency"        : latency,
            "sla_max_latency": task["sla_max_latency"],
            "sla_met"        : sla_met,
            "reward"         : reward,
        })

    total_reward   = sum(r["reward"] for r in results)
    average_reward = round(total_reward / len(results), 4) if results else 0.0
    sla_violations = sum(1 for r in results if not r["sla_met"])

    summary = {
        "model"          : MODEL_NAME,
        "tasks_evaluated": len(results),
        "average_reward" : average_reward,
        "sla_violations" : sla_violations,
        "results"        : results,
    }

    print("\n" + "=" * 60)
    print(f"  Tasks evaluated : {summary['tasks_evaluated']}")
    print(f"  Average reward  : {summary['average_reward']}")
    print(f"  SLA violations  : {summary['sla_violations']}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    result = run_inference()
    print("\nJSON output:")
    print(json.dumps(result, indent=2))