"""
tasks.py — Predefined benchmark tasks for the SLA-Aware Multi-Cloud Cost Optimizer.

Difficulty tiers
----------------
easy   → One provider is clearly cheapest AND fastest → obvious best pick.
medium → Cheapest provider violates SLA; agent must trade cost for compliance.
hard   → Tight SLA window + tricky pricing; wrong heuristics fail here.
"""

from typing import Dict, Any


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASKS: Dict[str, Dict[str, Any]] = {

    # -----------------------------------------------------------------------
    # EASY — GCP dominates on both cost and latency; trivial best choice.
    # -----------------------------------------------------------------------
    "easy": {
        "task_id":         "easy",
        "difficulty":      "easy",
        "description":     "API request with relaxed SLA. GCP is cheapest and fastest.",
        "job_type":        "api_request",
        "sla_max_latency": 90,   # ms — generous budget
        "providers": {
            "aws":   {"cost": 55,  "latency": 75},
            "azure": {"cost": 65,  "latency": 80},
            "gcp":   {"cost": 40,  "latency": 58},
        },
        "optimal_cloud":   "gcp",
        "hint":            "All providers meet SLA. Pick the cheapest.",
    },

    # -----------------------------------------------------------------------
    # MEDIUM — AWS is cheapest but violates SLA; GCP is the smart pick.
    # -----------------------------------------------------------------------
    "medium": {
        "task_id":         "medium",
        "difficulty":      "medium",
        "description":     "Batch job where the cheapest option exceeds latency SLA.",
        "job_type":        "batch_job",
        "sla_max_latency": 125,   # ms — moderate constraint
        "providers": {
            "aws":   {"cost": 55,  "latency": 145},   # ← cheapest but SLA violation!
            "azure": {"cost": 85,  "latency": 118},
            "gcp":   {"cost": 72,  "latency": 108},
        },
        "optimal_cloud":   "gcp",
        "hint":            "AWS violates SLA. Among compliant options, GCP is cheaper.",
    },

    # -----------------------------------------------------------------------
    # HARD — Tight SLA, misleading pricing; only one provider qualifies.
    # -----------------------------------------------------------------------
    "hard": {
        "task_id":         "hard",
        "difficulty":      "hard",
        "description":     (
            "ML training job with strict 210 ms SLA. Azure appears cheap but "
            "misses SLA by a whisker. AWS is expensive. Only GCP qualifies "
            "and is not obviously the best on sticker price."
        ),
        "job_type":        "ml_training",
        "sla_max_latency": 210,   # ms — strict
        "providers": {
            "aws":   {"cost": 140, "latency": 195},   # meets SLA but expensive
            "azure": {"cost": 105, "latency": 215},   # looks cheap — SLA violation!
            "gcp":   {"cost": 125, "latency": 205},   # meets SLA, 2nd cheapest
        },
        "optimal_cloud":   "gcp",
        "hint":            (
            "Azure violates SLA (215 > 210). AWS meets SLA but costs $140. "
            "GCP meets SLA at $125 — best valid option."
        ),
    },

    # -----------------------------------------------------------------------
    # BONUS — All three providers meet SLA; pure cost minimization.
    # -----------------------------------------------------------------------
    "bonus_cost": {
        "task_id":         "bonus_cost",
        "difficulty":      "easy",
        "description":     "All providers meet SLA. Straightforward cost minimization.",
        "job_type":        "api_request",
        "sla_max_latency": 100,
        "providers": {
            "aws":   {"cost": 48,  "latency": 62},
            "azure": {"cost": 60,  "latency": 55},
            "gcp":   {"cost": 38,  "latency": 70},
        },
        "optimal_cloud":   "gcp",
        "hint":            "All compliant. GCP is cheapest.",
    },

    # -----------------------------------------------------------------------
    # BONUS — Only one provider meets the extremely tight SLA.
    # -----------------------------------------------------------------------
    "bonus_strict_sla": {
        "task_id":         "bonus_strict_sla",
        "difficulty":      "hard",
        "description":     "Ultra-tight 60 ms SLA; only one provider qualifies.",
        "job_type":        "api_request",
        "sla_max_latency": 60,    # very tight
        "providers": {
            "aws":   {"cost": 40,  "latency": 65},   # SLA violation
            "azure": {"cost": 50,  "latency": 72},   # SLA violation
            "gcp":   {"cost": 45,  "latency": 55},   # only valid option
        },
        "optimal_cloud":   "gcp",
        "hint":            "AWS and Azure both violate the 60 ms SLA. GCP is the only valid choice.",
    },
}


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def get_task(task_id: str) -> Dict[str, Any]:
    """Return a task by ID, raising KeyError if not found."""
    if task_id not in TASKS:
        raise KeyError(f"Unknown task_id '{task_id}'. Available: {list(TASKS.keys())}")
    return TASKS[task_id]


def list_tasks() -> list:
    """Return a lightweight summary list of all tasks."""
    return [
        {
            "task_id":    tid,
            "difficulty": t["difficulty"],
            "job_type":   t["job_type"],
            "description": t["description"],
            "sla_max_latency": t["sla_max_latency"],
        }
        for tid, t in TASKS.items()
    ]
