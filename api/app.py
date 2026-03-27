"""
app.py — Flask REST API for the SLA-Aware Multi-Cloud Cost Optimizer.

Endpoints
---------
GET  /tasks            → list all available benchmark tasks
GET  /tasks/<task_id>  → full detail for one task
POST /grader           → score a cloud selection against a task
GET  /baseline         → run baseline agent on all tasks
GET  /baseline/<task_id> → run baseline agent on a single task
GET  /health           → liveness check
"""

import sys
import os

# Ensure sibling packages are importable regardless of working directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flask import Flask, request, jsonify, Response

from env.cloud_env  import CloudEnvironment, PROVIDERS
from tasks.tasks    import TASKS, list_tasks, get_task
from baseline.baseline import run_baseline, run_baseline_on_task


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error(message: str, code: int = 400) -> Response:
    return jsonify({"error": message}), code


def _validate_task_id(task_id: str):
    """Return (task_dict, None) or (None, error_response)."""
    try:
        return get_task(task_id), None
    except KeyError:
        return None, _error(
            f"Unknown task_id '{task_id}'. "
            f"Available task IDs: {list(TASKS.keys())}",
            404,
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness check."""
    return jsonify({"status": "ok", "service": "SLA-Aware Multi-Cloud Cost Optimizer"})

@app.route("/", methods=["GET"])
def home():
    return {
        "message": "SLA-Aware Multi-Cloud Cost Optimizer API",
        "available_endpoints": [
            "/tasks",
            "/grader (POST)",
            "/baseline"
        ]
    }

# ── /tasks ──────────────────────────────────────────────────────────────────

@app.get("/tasks")
def get_tasks():
    """
    Return a summary list of all benchmark tasks.

    Response
    --------
    {
        "count": 5,
        "tasks": [
            {
                "task_id": "easy",
                "difficulty": "easy",
                "job_type": "api_request",
                "description": "...",
                "sla_max_latency": 90
            },
            ...
        ]
    }
    """
    tasks = list_tasks()
    return jsonify({"count": len(tasks), "tasks": tasks})


@app.get("/tasks/<task_id>")
def get_task_detail(task_id: str):
    """
    Return full detail for a specific task (including provider metrics).

    Response
    --------
    {
        "task_id": "medium",
        "difficulty": "medium",
        "description": "...",
        "job_type": "batch_job",
        "sla_max_latency": 125,
        "providers": {
            "aws":   {"cost": 55, "latency": 145},
            "azure": {"cost": 85, "latency": 118},
            "gcp":   {"cost": 72, "latency": 108}
        },
        "hint": "..."
    }
    """
    task, err = _validate_task_id(task_id)
    if err:
        return err

    # Expose everything except internal 'optimal_cloud' key to avoid spoilers
    public = {k: v for k, v in task.items() if k != "optimal_cloud"}
    return jsonify(public)


# ── /grader ─────────────────────────────────────────────────────────────────

@app.post("/grader")
def grader():
    """
    Score a cloud provider selection against a task.

    Request body (JSON)
    -------------------
    {
        "task_id":       "easy",
        "selected_cloud": "gcp"
    }

    Response
    --------
    {
        "task_id":        "easy",
        "selected_cloud": "gcp",
        "cost":           40,
        "latency":        58,
        "sla_max_latency": 90,
        "sla_met":        true,
        "reward":         0.92,
        "grade":          "excellent"
    }
    """
    body = request.get_json(silent=True)

    if not body:
        return _error("Request body must be valid JSON.")

    task_id        = body.get("task_id")
    selected_cloud = body.get("selected_cloud")

    if not task_id:
        return _error("Missing required field: 'task_id'.")
    if not selected_cloud:
        return _error("Missing required field: 'selected_cloud'.")
    if selected_cloud not in PROVIDERS:
        return _error(f"Invalid 'selected_cloud'. Must be one of: {PROVIDERS}.")

    task, err = _validate_task_id(task_id)
    if err:
        return err

    # Run the environment with the fixed task (no noise for reproducibility)
    env   = CloudEnvironment(task=task, noise=0.0)
    state = env.reset()
    _, reward, _, info = env.step(selected_cloud)

    # Human-readable grade
    grade = _grade(reward, info["sla_met"])

    return jsonify({
        "task_id":         task_id,
        "selected_cloud":  info["selected_cloud"],
        "cost":            info["cost"],
        "latency":         info["latency"],
        "sla_max_latency": info["sla_max_latency"],
        "sla_met":         info["sla_met"],
        "reward":          info["reward"],
        "grade":           grade,
    })


# ── /baseline ────────────────────────────────────────────────────────────────

@app.get("/baseline")
def baseline_all():
    """
    Run the greedy baseline agent on all benchmark tasks.

    Response
    --------
    {
        "strategy":      "cheapest_sla_compliant",
        "average_reward": 0.87,
        "total_tasks":    5,
        "sla_violations": 0,
        "optimal_picks":  4,
        "results": [ ... ]
    }
    """
    summary = run_baseline(verbose=False)
    summary["strategy"] = "cheapest_sla_compliant"
    return jsonify(summary)


@app.get("/baseline/<task_id>")
def baseline_single(task_id: str):
    """
    Run the greedy baseline agent on a single task.

    Response
    --------
    {
        "task_id":        "hard",
        "selected_cloud": "gcp",
        "cost":           125,
        "latency":        205,
        "sla_max_latency": 210,
        "sla_met":        true,
        "reward":         0.78,
        "strategy":       "cheapest_sla_compliant"
    }
    """
    task, err = _validate_task_id(task_id)
    if err:
        return err

    info = run_baseline_on_task(task)
    info["task_id"] = task_id
    return jsonify(info)


# ── /compare ─────────────────────────────────────────────────────────────────

@app.get("/compare/<task_id>")
def compare_all_providers(task_id: str):
    """
    Score all three providers against a task — useful for exploration.

    Response
    --------
    {
        "task_id": "medium",
        "sla_max_latency": 125,
        "scores": {
            "aws":   {"cost": 55, "latency": 145, "sla_met": false, "reward": 0.0},
            "azure": {"cost": 85, "latency": 118, "sla_met": true,  "reward": 0.52},
            "gcp":   {"cost": 72, "latency": 108, "sla_met": true,  "reward": 0.85}
        },
        "best_provider": "gcp"
    }
    """
    task, err = _validate_task_id(task_id)
    if err:
        return err

    scores = {}
    for provider in PROVIDERS:
        env   = CloudEnvironment(task=task, noise=0.0)
        env.reset()
        _, _, _, info = env.step(provider)
        scores[provider] = {
            "cost":    info["cost"],
            "latency": info["latency"],
            "sla_met": info["sla_met"],
            "reward":  info["reward"],
        }

    best = max(scores, key=lambda p: scores[p]["reward"])

    return jsonify({
        "task_id":         task_id,
        "sla_max_latency": task["sla_max_latency"],
        "scores":          scores,
        "best_provider":   best,
    })


# ---------------------------------------------------------------------------
# Grade helper
# ---------------------------------------------------------------------------

def _grade(reward: float, sla_met: bool) -> str:
    if not sla_met:
        return "failed (SLA violation)"
    if reward >= 0.90:
        return "excellent"
    if reward >= 0.75:
        return "good"
    if reward >= 0.55:
        return "fair"
    return "poor"


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(_):
    return _error("Endpoint not found.", 404)


@app.errorhandler(405)
def method_not_allowed(_):
    return _error("HTTP method not allowed for this endpoint.", 405)


@app.errorhandler(500)
def internal_error(exc):
    return _error(f"Internal server error: {exc}", 500)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
