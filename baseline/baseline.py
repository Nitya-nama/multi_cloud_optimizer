"""
baseline.py — Simple rule-based baseline agent for the Multi-Cloud Cost Optimizer.

Strategy
--------
1. Filter providers that satisfy the SLA constraint (latency ≤ sla_max_latency).
2. Among valid providers, pick the one with the lowest cost.
3. If no provider satisfies the SLA, pick the one with the lowest latency
   (best-effort — still returns a suboptimal reward).

This greedy heuristic is the natural human intuition and serves as a lower-bound
performance bar for any learned agent.
"""

import sys
import os
from typing import Dict, Any, List, Optional

# Allow running standalone as well as via the API
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.cloud_env import CloudEnvironment
from tasks.tasks import TASKS, list_tasks


# ---------------------------------------------------------------------------
# Core strategy
# ---------------------------------------------------------------------------

def select_cloud(state: Dict[str, Any]) -> str:
    """
    Greedy baseline: cheapest provider that meets the SLA.

    Parameters
    ----------
    state : environment state dict (output of env.get_state())

    Returns
    -------
    provider name: "aws" | "azure" | "gcp"
    """
    providers = state["providers"]
    sla       = state["sla_max_latency"]

    # Filter SLA-compliant providers
    valid = {p: m for p, m in providers.items() if m["latency"] <= sla}

    if valid:
        # Choose cheapest compliant provider
        return min(valid, key=lambda p: valid[p]["cost"])
    else:
        # No valid option — choose minimum latency (best-effort)
        return min(providers, key=lambda p: providers[p]["latency"])


# ---------------------------------------------------------------------------
# Full evaluation over all tasks
# ---------------------------------------------------------------------------

def run_baseline(verbose: bool = True) -> Dict[str, Any]:
    """
    Run the baseline agent on every predefined task and return results.

    Returns
    -------
    {
        "results": [ {task_id, selected_cloud, cost, latency, sla_met, reward}, ... ],
        "average_reward": float,
        "total_tasks":    int,
        "sla_violations": int,
        "optimal_picks":  int,
    }
    """
    results:        List[Dict] = []
    total_reward    = 0.0
    sla_violations  = 0
    optimal_picks   = 0

    for task_id, task in TASKS.items():
        env    = CloudEnvironment(task=task, noise=0.0)   # no noise for reproducibility
        state  = env.reset()
        action = select_cloud(state)

        _, reward, _, info = env.step(action)

        is_optimal = (action == task.get("optimal_cloud"))

        row = {
            "task_id":        task_id,
            "difficulty":     task["difficulty"],
            "selected_cloud": action,
            "cost":           info["cost"],
            "latency":        info["latency"],
            "sla_max_latency": info["sla_max_latency"],
            "sla_met":        info["sla_met"],
            "reward":         info["reward"],
            "is_optimal":     is_optimal,
        }
        results.append(row)

        total_reward  += reward
        if not info["sla_met"]:
            sla_violations += 1
        if is_optimal:
            optimal_picks += 1

        if verbose:
            _print_row(row)

    avg_reward = round(total_reward / len(results), 4) if results else 0.0

    summary = {
        "results":        results,
        "average_reward": avg_reward,
        "total_tasks":    len(results),
        "sla_violations": sla_violations,
        "optimal_picks":  optimal_picks,
    }

    if verbose:
        _print_summary(summary)

    return summary


# ---------------------------------------------------------------------------
# Single-task evaluation (used by Flask /baseline endpoint)
# ---------------------------------------------------------------------------

def run_baseline_on_task(task):
    providers = task["providers"]
    sla = task["sla_max_latency"]

    valid = {
        k: v for k, v in providers.items() if v["latency"] <= sla
    }

    if valid:
        best = min(valid.items(), key=lambda x: x[1]["cost"])
    else:
        best = min(providers.items(), key=lambda x: x[1]["latency"])

    cloud, metrics = best

    # simple reward estimate
    reward = 0.8 if metrics["latency"] <= sla else 0.0

    return {
        "selected_cloud": cloud,
        "cost": metrics["cost"],
        "latency": metrics["latency"],
        "reward": reward
    }



# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _print_row(row: Dict) -> None:
    sla_tag = "✓" if row["sla_met"]   else "✗ SLA!"
    opt_tag = "★ optimal" if row["is_optimal"] else ""
    print(
        f"[{row['task_id']:<18}] "
        f"→ {row['selected_cloud']:<6} "
        f"cost={row['cost']:>6.1f}  lat={row['latency']:>6.1f}ms  "
        f"reward={row['reward']:.4f}  {sla_tag}  {opt_tag}"
    )


def _print_summary(summary: Dict) -> None:
    n = summary["total_tasks"]
    print("\n" + "=" * 60)
    print(f"  Baseline Performance Summary")
    print("=" * 60)
    print(f"  Tasks evaluated : {n}")
    print(f"  Average reward  : {summary['average_reward']:.4f}")
    print(f"  SLA violations  : {summary['sla_violations']} / {n}")
    print(f"  Optimal picks   : {summary['optimal_picks']} / {n}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_baseline(verbose=True)
