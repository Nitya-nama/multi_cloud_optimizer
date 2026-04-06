"""
inference.py — LLM-based inference script for the SLA-Aware Multi-Cloud Cost Optimizer.

Uses the OpenAI client to query a language model to select the optimal cloud provider
for each benchmark task. Reads credentials from environment variables:

    API_BASE_URL      — base URL for the LLM API endpoint
    MODEL_NAME        — model identifier to use for inference
    HF_TOKEN          — Hugging Face / API key (no default — must be set explicitly)
    LOCAL_IMAGE_NAME  — optional: only needed when using from_docker_image()

Usage:
    python inference.py
"""

import os
import json
import requests
from openai import OpenAI

# ── Credentials & config ──────────────────────────────────────────────────────
# FIX 1: API_BASE_URL and MODEL_NAME have defaults; HF_TOKEN does NOT
API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint-url>")
MODEL_NAME   = os.getenv("MODEL_NAME",   "<your-active-model-name>")
HF_TOKEN     = os.getenv("HF_TOKEN")                  # no default — required at runtime

# FIX 2: LOCAL_IMAGE_NAME added (optional — only used with from_docker_image())
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# FIX 3: OpenAI client uses the env vars directly, no fallback strings
client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)

ENV_BASE_URL = os.getenv(
    "ENV_BASE_URL", "https://nityanama-multi-cloud-optimizer.hf.space"
)

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

    sorted_providers = sorted(providers.items(), key=lambda x: x[1]["cost"])
    providers_text = "\n".join(
        f"  {name}: cost=${metrics['cost']}, latency={metrics['latency']}ms"
        for name, metrics in sorted_providers
    )

    compliant = {k: v for k, v in providers.items() if v["latency"] <= sla}
    sorted_compliant = sorted(compliant.items(), key=lambda x: x[1]["cost"])
    compliant_text = (
        "\n".join(
            f"  {name}: cost=${metrics['cost']}, latency={metrics['latency']}ms"
            + (" <-- CHEAPEST" if i == 0 else "")
            for i, (name, metrics) in enumerate(sorted_compliant)
        )
        if compliant
        else "  (none meet SLA - pick lowest latency)"
    )

    prompt = f"""You are an intelligent cloud optimizer.

Rules:
1. The SLA latency limit is {sla} ms. Any provider with latency > {sla} ms is disqualified.
2. From remaining providers:
   - Prefer LOW cost
   - ALSO prefer LOW latency (better performance)
3. Avoid providers that are very close to the SLA limit (they are risky).
4. If two providers have similar cost, choose the one with LOWER latency.

Choose the BEST trade-off between cost and latency.

All providers:
{providers_text}

Valid providers (meeting SLA):
{compliant_text}

Reply with ONLY ONE WORD: aws or azure or gcp
"""

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

    print(f"STEP warn LLM returned unexpected output: '{raw}' - falling back to greedy")
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
    # FIX 4: Required START/STEP/END structured log format
    print("START")

    tasks = get_tasks()
    results = []

    for task_summary in tasks:
        task_id = task_summary["task_id"]
        task = get_task_detail(task_id)

        try:
            selected_cloud = ask_llm(task)
        except Exception as exc:
            print(f"STEP {task_id} error={exc} using greedy fallback")
            selected_cloud = _greedy_fallback(task)

        graded        = grade_selection(task_id, selected_cloud)
        reward        = graded["reward"]
        sla_met       = graded["sla_met"]
        cost          = graded["cost"]
        latency       = graded["latency"]

        # Structured STEP log
        print(f"STEP {task_id} {selected_cloud} {reward}")

        results.append(
            {
                "task_id":         task_id,
                "difficulty":      task_summary["difficulty"],
                "selected_cloud":  selected_cloud,
                "cost":            cost,
                "latency":         latency,
                "sla_max_latency": task["sla_max_latency"],
                "sla_met":         sla_met,
                "reward":          reward,
            }
        )

    total_reward   = sum(r["reward"] for r in results)
    average_reward = round(total_reward / len(results), 4) if results else 0.0
    sla_violations = sum(1 for r in results if not r["sla_met"])

    summary = {
        "model":           MODEL_NAME,
        "tasks_evaluated": len(results),
        "average_reward":  average_reward,
        "sla_violations":  sla_violations,
        "results":         results,
    }

    print("END")
    return summary


if __name__ == "__main__":
    result = run_inference()
    print(json.dumps(result, indent=2))
