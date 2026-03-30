"""
app.py — Single-file Flask app for the SLA-Aware Multi-Cloud Cost Optimizer.
All environment, tasks, grader, baseline, and API logic in one file.
"""

import os
import sys
import random
import datetime
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Literal
from pydantic import BaseModel, Field
from flask import Flask, request, jsonify

# ─────────────────────────────────────────────────────────────
# Pydantic Models (OpenEnv spec)
# ─────────────────────────────────────────────────────────────

class ProviderMetrics(BaseModel):
    cost: float
    latency: float

class Observation(BaseModel):
    job_type: str
    sla_max_latency: float
    providers: Dict[Literal["aws", "azure", "gcp"], ProviderMetrics]

class Action(BaseModel):
    action: Literal["aws", "azure", "gcp"]

class Reward(BaseModel):
    reward: float = Field(..., ge=0.0, le=1.0)
    done: bool
    info: Dict

# ─────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────

PROVIDERS = ["aws", "azure", "gcp"]

PROVIDER_PROFILES = {
    "aws":   {"base_cost": {"ml_training": 120, "api_request": 40,  "batch_job": 70},
              "base_latency": {"ml_training": 200, "api_request": 60,  "batch_job": 130}},
    "azure": {"base_cost": {"ml_training": 110, "api_request": 50,  "batch_job": 80},
              "base_latency": {"ml_training": 220, "api_request": 70,  "batch_job": 120}},
    "gcp":   {"base_cost": {"ml_training": 100, "api_request": 35,  "batch_job": 65},
              "base_latency": {"ml_training": 180, "api_request": 55,  "batch_job": 110}},
}

JOB_TYPES  = ["ml_training", "api_request", "batch_job"]
SLA_LIMITS = {"ml_training": 250, "api_request": 80, "batch_job": 160}


class CloudEnvironment:
    def __init__(self, task: Optional[Dict] = None, noise: float = 0.10):
        self._fixed_task = task
        self._noise = noise
        self._state: Dict[str, Any] = {}
        self._done = False
        self.reset()

    def reset(self) -> Dict[str, Any]:
        self._done = False
        if self._fixed_task:
            job_type        = self._fixed_task["job_type"]
            sla_max_latency = self._fixed_task["sla_max_latency"]
            provider_data   = self._fixed_task["providers"]
        else:
            job_type        = random.choice(JOB_TYPES)
            sla_max_latency = SLA_LIMITS[job_type]
            provider_data   = self._generate_provider_data(job_type)
        self._state = {"job_type": job_type, "sla_max_latency": sla_max_latency, "providers": provider_data}
        return self._state

    def step(self, action: str) -> Tuple[Dict, float, bool, Dict]:
        if action not in PROVIDERS:
            raise ValueError(f"Invalid action '{action}'. Must be one of {PROVIDERS}.")
        if self._done:
            raise RuntimeError("Episode done. Call reset() first.")
        metrics = self._state["providers"][action]
        cost    = metrics["cost"]
        latency = metrics["latency"]
        sla     = self._state["sla_max_latency"]
        reward  = self._compute_reward(action, cost, latency, sla)
        info = {
            "selected_cloud" : action,
            "cost"           : round(cost, 2),
            "latency"        : round(latency, 2),
            "sla_max_latency": sla,
            "sla_met"        : latency <= sla,
            "reward"         : round(reward, 4),
        }
        self._done = True
        return self._state, reward, self._done, info

    def get_state(self) -> Dict[str, Any]:
        return dict(self._state)

    def _compute_reward(self, action: str, cost: float, latency: float, sla: float) -> float:
        if latency > sla:
            return 0.0
        providers = self._state["providers"]
        valid     = {p: m for p, m in providers.items() if m["latency"] <= sla}
        if not valid:
            return 0.1
        costs    = [m["cost"] for m in valid.values()]
        min_cost = min(costs)
        max_cost = max(costs)
        cost_score       = 1.0 if max_cost == min_cost else 1.0 - (cost - min_cost) / (max_cost - min_cost)
        headroom_bonus   = 0.15 * ((sla - latency) / sla)
        efficiency_bonus = 0.10 if cost == min_cost else 0.0
        return float(np.clip(0.75 * cost_score + headroom_bonus + efficiency_bonus, 0.30, 1.00))

    def _generate_provider_data(self, job_type: str) -> Dict:
        data = {}
        for provider, profile in PROVIDER_PROFILES.items():
            data[provider] = {
                "cost"   : round(profile["base_cost"][job_type]    * random.uniform(0.9, 1.1), 2),
                "latency": round(profile["base_latency"][job_type] * random.uniform(0.9, 1.1), 2),
            }
        return data
    
    


# ─────────────────────────────────────────────────────────────
# Tasks
# ─────────────────────────────────────────────────────────────

TASKS = {
    "easy": {
        "task_id": "easy", "difficulty": "easy",
        "description": "API request with relaxed SLA. GCP is cheapest and fastest.",
        "job_type": "api_request", "sla_max_latency": 90,
        "providers": {"aws": {"cost": 55, "latency": 75}, "azure": {"cost": 65, "latency": 80}, "gcp": {"cost": 40, "latency": 58}},
        "optimal_cloud": "gcp", "hint": "All providers meet SLA. Pick the cheapest.",
    },
    "medium": {
        "task_id": "medium", "difficulty": "medium",
        "description": "Batch job where the cheapest option exceeds latency SLA.",
        "job_type": "batch_job", "sla_max_latency": 125,
        "providers": {"aws": {"cost": 55, "latency": 145}, "azure": {"cost": 85, "latency": 118}, "gcp": {"cost": 72, "latency": 108}},
        "optimal_cloud": "gcp", "hint": "AWS violates SLA. Among compliant options, GCP is cheaper.",
    },
    "hard": {
        "task_id": "hard", "difficulty": "hard",
        "description": "ML training job with strict 210ms SLA. Azure appears cheap but misses SLA.",
        "job_type": "ml_training", "sla_max_latency": 210,
        "providers": {"aws": {"cost": 140, "latency": 195}, "azure": {"cost": 105, "latency": 215}, "gcp": {"cost": 125, "latency": 205}},
        "optimal_cloud": "gcp", "hint": "Azure violates SLA. GCP meets SLA at lowest cost.",
    },
    "bonus_cost": {
        "task_id": "bonus_cost", "difficulty": "easy",
        "description": "All providers meet SLA. Straightforward cost minimization.",
        "job_type": "api_request", "sla_max_latency": 100,
        "providers": {"aws": {"cost": 48, "latency": 62}, "azure": {"cost": 60, "latency": 55}, "gcp": {"cost": 38, "latency": 70}},
        "optimal_cloud": "gcp", "hint": "All compliant. GCP is cheapest.",
    },
    "bonus_strict_sla": {
        "task_id": "bonus_strict_sla", "difficulty": "hard",
        "description": "Ultra-tight 60ms SLA; only one provider qualifies.",
        "job_type": "api_request", "sla_max_latency": 60,
        "providers": {"aws": {"cost": 40, "latency": 65}, "azure": {"cost": 50, "latency": 72}, "gcp": {"cost": 45, "latency": 55}},
        "optimal_cloud": "gcp", "hint": "AWS and Azure violate the 60ms SLA. GCP is the only valid choice.",
    },
}


# ─────────────────────────────────────────────────────────────
# Baseline agent
# ─────────────────────────────────────────────────────────────

def greedy_select(task: Dict) -> str:
    providers = task["providers"]
    sla       = task["sla_max_latency"]
    valid     = {k: v for k, v in providers.items() if v["latency"] <= sla}
    if valid:
        return min(valid, key=lambda p: valid[p]["cost"])
    return min(providers, key=lambda p: providers[p]["latency"])

def run_baseline_on_task(task: Dict) -> Dict:
    selected = greedy_select(task)
    env = CloudEnvironment(task=task, noise=0.0)
    env.reset()
    _, reward, _, info = env.step(selected)
    return {"selected_cloud": selected, "reward": round(reward, 4),
            "sla_met": info["sla_met"], "cost": info["cost"], "latency": info["latency"]}

def run_baseline() -> Dict:
    results = []
    for task_id, task in TASKS.items():
        r = run_baseline_on_task(task)
        r["task_id"]    = task_id
        r["difficulty"] = task["difficulty"]
        results.append(r)
    avg = round(sum(r["reward"] for r in results) / len(results), 4)
    return {
        "strategy"       : "cheapest_sla_compliant",
        "average_reward" : avg,
        "total_tasks"    : len(results),
        "sla_violations" : sum(1 for r in results if not r["sla_met"]),
        "results"        : results,
    }


# ─────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

global_env = None

def _error(msg: str, code: int = 400):
    return jsonify({"error": msg}), code

def _grade(reward: float, sla_met: bool) -> str:
    if not sla_met:    return "failed (SLA violation)"
    if reward >= 0.90: return "excellent"
    if reward >= 0.75: return "good"
    if reward >= 0.55: return "fair"
    return "poor"


@app.get("/reset")
def reset_env():
    global global_env
    global_env = CloudEnvironment()
    obs = Observation(**global_env.reset())
    return jsonify(obs.model_dump())

@app.post("/step")
def step_env():
    global global_env
    data = request.json
    if not data or "action" not in data:
        return _error("Request body must include 'action' field.")
    try:
        action_model = Action(action=data["action"])
    except Exception as e:
        return _error(f"Invalid action. Must be one of: aws, azure, gcp.")
    if not global_env:
        return _error("Environment not initialized. Call /reset first.")
    state, reward, done, info = global_env.step(action_model.action)
    return jsonify({"state": Observation(**state).model_dump(),
                    "reward": round(reward, 4), "done": done, "info": info})

@app.get("/state")
def get_state():
    global global_env
    if not global_env:
        return _error("Environment not initialized. Call /reset first.")
    return jsonify(Observation(**global_env.get_state()).model_dump())

@app.get("/tasks")
def get_tasks():
    tasks = [{"task_id": t["task_id"], "difficulty": t["difficulty"],
               "job_type": t["job_type"], "description": t["description"],
               "sla_max_latency": t["sla_max_latency"]} for t in TASKS.values()]
    return jsonify({"count": len(tasks), "tasks": tasks})

@app.get("/tasks/<task_id>")
def get_task_detail(task_id: str):
    if task_id not in TASKS:
        return _error(f"Unknown task_id '{task_id}'.", 404)
    return jsonify({k: v for k, v in TASKS[task_id].items() if k != "optimal_cloud"})

@app.post("/grader")
def grader():
    data = request.json
    if not data or "task_id" not in data or "selected_cloud" not in data:
        return _error("Request body must include 'task_id' and 'selected_cloud'.")
    task_id, selected_cloud = data["task_id"], data["selected_cloud"]
    if task_id not in TASKS:
        return _error(f"Unknown task_id '{task_id}'.", 404)
    task = TASKS[task_id]
    env  = CloudEnvironment(task=task, noise=0.0)
    env.reset()
    _, reward, _, info = env.step(selected_cloud)
    baseline_reward = run_baseline_on_task(task)["reward"]
    return jsonify({
        "task_id"             : task_id,
        "selected_cloud"      : selected_cloud,
        "cost"                : info["cost"],
        "latency"             : info["latency"],
        "sla_max_latency"     : task["sla_max_latency"],
        "sla_met"             : info["sla_met"],
        "reward"              : round(reward, 4),
        "baseline_reward"     : round(baseline_reward, 4),
        "better_than_baseline": reward > baseline_reward,
        "is_optimal"          : selected_cloud == task.get("optimal_cloud"),
        "grade"               : _grade(reward, info["sla_met"]),
        "timestamp"           : datetime.datetime.utcnow().isoformat(),
    })

@app.get("/baseline")
def baseline_all():
    return jsonify(run_baseline())

@app.get("/baseline/<task_id>")
def baseline_single(task_id: str):
    if task_id not in TASKS:
        return _error(f"Unknown task_id '{task_id}'.", 404)
    info = run_baseline_on_task(TASKS[task_id])
    info["task_id"] = task_id
    return jsonify(info)

@app.get("/compare/<task_id>")
def compare_all_providers(task_id: str):
    if task_id not in TASKS:
        return _error(f"Unknown task_id '{task_id}'.", 404)
    task   = TASKS[task_id]
    scores = {}
    for provider in PROVIDERS:
        env = CloudEnvironment(task=task, noise=0.0)
        env.reset()
        _, _, _, info = env.step(provider)
        scores[provider] = {"cost": info["cost"], "latency": info["latency"],
                            "sla_met": info["sla_met"], "reward": info["reward"]}
    return jsonify({"task_id": task_id, "sla_max_latency": task["sla_max_latency"],
                    "scores": scores, "best_provider": max(scores, key=lambda p: scores[p]["reward"])})

@app.get("/leaderboard")
def leaderboard():
    results = run_baseline()["results"]
    return jsonify({"top_performers": sorted(results, key=lambda x: x["reward"], reverse=True),
                    "message": "Leaderboard based on reward score"})

@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "SLA-Aware Multi-Cloud Cost Optimizer"})

@app.get("/docs")
def docs():
    return jsonify({
        "description": "OpenEnv-compatible AI evaluation environment",
        "usage": {
            "GET /reset": "Start new episode",
            "POST /step": "Take action {action: aws|azure|gcp}",
            "GET /state": "Get current state",
            "GET /tasks": "List tasks",
            "POST /grader": "Evaluate decision"
        }
    })
    
@app.get("/explain/<task_id>")
def explain(task_id):
    if task_id not in TASKS:
        return _error("Invalid task_id", 404)

    task = TASKS[task_id]
    providers = task["providers"]
    sla = task["sla_max_latency"]

    explanation = []

    for p, m in providers.items():
        if m["latency"] > sla:
            explanation.append(f"{p} ❌ rejected (latency {m['latency']} > SLA {sla})")
        else:
            explanation.append(f"{p} ✅ valid (cost={m['cost']}, latency={m['latency']})")

    valid = {p: m for p, m in providers.items() if m["latency"] <= sla}

    best = min(valid, key=lambda p: valid[p]["cost"]) if valid else None

    return jsonify({
        "task_id": task_id,
        "decision_process": explanation,
        "best_choice": best,
        "reason": "Lowest cost among SLA-compliant providers"
    })
    
@app.get("/insights/<task_id>")
def insights(task_id):
    if task_id not in TASKS:
        return _error("Invalid task_id", 404)

    task = TASKS[task_id]
    providers = task["providers"]

    cheapest = min(providers, key=lambda p: providers[p]["cost"])
    fastest = min(providers, key=lambda p: providers[p]["latency"])

    return jsonify({
        "cheapest_provider": cheapest,
        "fastest_provider": fastest,
        "tradeoff_exists": cheapest != fastest,
        "insight": "Optimal decision depends on SLA, not just cost"
    })
    
@app.get("/agent_vs_baseline")
def agent_vs_baseline():
    baseline = run_baseline()

    try:
        from inference import run_inference
        agent = run_inference()

        return jsonify({
            "baseline_avg": baseline["average_reward"],
            "agent_avg": agent["average_reward"],
            "improvement": round(agent["average_reward"] - baseline["average_reward"], 4)
        })
    except:
        return jsonify({
            "baseline_avg": baseline["average_reward"],
            "agent_avg": "not available",
            "note": "Inference not configured"
        })    

@app.route("/")
def home():
    return jsonify({
        "message": "SLA-Aware Multi-Cloud Cost Optimizer API",
        "type": "AI Decision-Making Benchmark (OpenEnv)",
        "status": "running",
        "core_endpoints": ["/reset", "/step", "/state"],
        "evaluation": ["/tasks", "/grader", "/baseline"],
        "advanced": ["/explain/<task_id>", "/insights/<task_id>", "/agent_vs_baseline" , "/what_if/<task_id>"],
        "docs": "/docs"
    })
    
@app.get("/what_if/<task_id>")
def what_if(task_id):
    if task_id not in TASKS:
        return _error("Invalid task_id", 404)
    action = request.args.get("action")
    if action not in PROVIDERS:
        return _error("Query param 'action' must be one of aws, azure, gcp")
    task = TASKS[task_id]
    # Evaluate chosen action
    env = CloudEnvironment(task=task, noise=0.0)
    env.reset()
    _, reward, _, info = env.step(action)
    # Evaluate optimal action
    optimal = task.get("optimal_cloud")
    env_opt = CloudEnvironment(task=task, noise=0.0)
    env_opt.reset()
    _, opt_reward, _, opt_info = env_opt.step(optimal)
    loss = round(opt_reward - reward, 4)
    return jsonify({
        "task_id": task_id,
        "chosen_action": action,
        "optimal_action": optimal,
        "chosen_result": {
            "reward": round(reward, 4),
            "cost": info["cost"],
            "latency": info["latency"],
            "sla_met": info["sla_met"]
        },
        "optimal_result": {
            "reward": round(opt_reward, 4),
            "cost": opt_info["cost"],
            "latency": opt_info["latency"]
        },
        "performance_gap": loss,
        "insight": (
            "This shows how suboptimal decisions impact performance. "
            "AI agents must reason about SLA constraints, not just cost."
        )
    })    

@app.errorhandler(404)
def not_found(_):      return _error("Endpoint not found.", 404)
@app.errorhandler(405)
def method_not_allowed(_): return _error("HTTP method not allowed.", 405)
@app.errorhandler(500)
def internal_error(e): return _error(f"Internal server error: {e}", 500)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
