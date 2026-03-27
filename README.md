---
title: SLA-Aware Multi-Cloud Cost Optimizer
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# SLA-Aware Multi-Cloud Cost Optimizer

> An OpenEnv-compatible reinforcement learning environment where an AI agent selects the optimal cloud provider (AWS, Azure, GCP) by minimizing cost while satisfying latency SLA constraints.

---

## Table of Contents

1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [Project Structure](#project-structure)
4. [Quickstart](#quickstart)
5. [API Reference](#api-reference)
6. [Environment Design](#environment-design)
7. [Reward Function](#reward-function)
8. [Tasks](#tasks)
9. [Baseline Agent](#baseline-agent)
10. [Docker](#docker)

---

## Overview

Cloud cost optimization is a hard combinatorial problem: the cheapest provider may violate latency SLAs, while the fastest may be prohibitively expensive. This project frames the problem as an OpenEnv-compatible decision environment:

- **State** — job type + per-provider cost/latency metrics + SLA constraint
- **Action** — select one cloud provider: `aws`, `azure`, or `gcp`
- **Reward** — continuous signal in `[0, 1]` reflecting cost efficiency and SLA compliance

A Flask REST API exposes the environment for grading agent decisions and benchmarking.

---

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    OpenEnv Loop                          │
│                                                          │
│  state = env.reset()          ← new job arrives          │
│  action = agent.select(state) ← agent picks a cloud      │
│  s, reward, done, info = env.step(action)                │
│                               ← reward ∈ [0, 1]          │
└─────────────────────────────────────────────────────────┘
```

1. The environment generates a **job** (e.g., `ml_training`) with random or fixed per-provider cost and latency values.
2. An **SLA constraint** (`sla_max_latency`) defines the maximum allowed latency in milliseconds.
3. The agent selects one of three cloud providers.
4. The **reward function** returns `0.0` if the SLA is violated, or a value in `[0.3, 1.0]` based on cost efficiency and latency headroom.

---

## Project Structure

```
project/
├── env/
│   └── cloud_env.py        # OpenEnv environment (reset, step, get_state)
├── api/
│   └── app.py              # Flask REST API
├── tasks/
│   └── tasks.py            # Predefined benchmark tasks (easy / medium / hard)
├── baseline/
│   └── baseline.py         # Rule-based greedy baseline agent
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Quickstart

### Local (without Docker)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the API server
python api/app.py
# → Listening on http://0.0.0.0:7860

# 3. (Optional) Run the baseline agent directly
python baseline/baseline.py
```

### With Docker

```bash
docker build -t cloud-optimizer .
docker run -p 7860:7860 cloud-optimizer
```

---

## API Reference

Base URL: `http://localhost:7860`

---

### `GET /health`

Liveness check.

```bash
curl http://localhost:7860/health
```

```json
{
  "status": "ok",
  "service": "SLA-Aware Multi-Cloud Cost Optimizer"
}
```

---

### `GET /tasks`

List all benchmark tasks.

```bash
curl http://localhost:7860/tasks
```

```json
{
  "count": 5,
  "tasks": [
    {
      "task_id": "easy",
      "difficulty": "easy",
      "job_type": "api_request",
      "description": "API request with relaxed SLA. GCP is cheapest and fastest.",
      "sla_max_latency": 90
    },
    {
      "task_id": "medium",
      "difficulty": "medium",
      "job_type": "batch_job",
      "description": "Batch job where the cheapest option exceeds latency SLA.",
      "sla_max_latency": 125
    },
    ...
  ]
}
```

---

### `GET /tasks/<task_id>`

Retrieve full provider metrics for a task.

```bash
curl http://localhost:7860/tasks/medium
```

```json
{
  "task_id": "medium",
  "difficulty": "medium",
  "description": "Batch job where the cheapest option exceeds latency SLA.",
  "job_type": "batch_job",
  "sla_max_latency": 125,
  "providers": {
    "aws":   {"cost": 55, "latency": 145},
    "azure": {"cost": 85, "latency": 118},
    "gcp":   {"cost": 72, "latency": 108}
  },
  "hint": "AWS violates SLA. Among compliant options, GCP is cheaper."
}
```

---

### `POST /grader`

Score a cloud provider selection.

**Request**

```bash
curl -X POST http://localhost:7860/grader \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "selected_cloud": "gcp"}'
```

**Response**

```json
{
  "task_id": "easy",
  "selected_cloud": "gcp",
  "cost": 40,
  "latency": 58,
  "sla_max_latency": 90,
  "sla_met": true,
  "reward": 0.9250,
  "grade": "excellent"
}
```

**SLA violation example**

```bash
curl -X POST http://localhost:7860/grader \
  -H "Content-Type: application/json" \
  -d '{"task_id": "medium", "selected_cloud": "aws"}'
```

```json
{
  "task_id": "medium",
  "selected_cloud": "aws",
  "cost": 55,
  "latency": 145,
  "sla_max_latency": 125,
  "sla_met": false,
  "reward": 0.0,
  "grade": "failed (SLA violation)"
}
```

---

### `GET /baseline`

Run the greedy baseline agent on all tasks.

```bash
curl http://localhost:7860/baseline
```

```json
{
  "strategy": "cheapest_sla_compliant",
  "average_reward": 0.8700,
  "total_tasks": 5,
  "sla_violations": 0,
  "optimal_picks": 4,
  "results": [
    {
      "task_id": "easy",
      "difficulty": "easy",
      "selected_cloud": "gcp",
      "cost": 40,
      "latency": 58,
      "sla_max_latency": 90,
      "sla_met": true,
      "reward": 0.9250,
      "is_optimal": true
    },
    ...
  ]
}
```

---

### `GET /baseline/<task_id>`

Run the baseline agent on a single task.

```bash
curl http://localhost:7860/baseline/hard
```

```json
{
  "task_id": "hard",
  "selected_cloud": "gcp",
  "cost": 125,
  "latency": 205,
  "sla_max_latency": 210,
  "sla_met": true,
  "reward": 0.7800,
  "strategy": "cheapest_sla_compliant"
}
```

---

### `GET /compare/<task_id>`

Score all three providers at once — useful for analysis.

```bash
curl http://localhost:7860/compare/hard
```

```json
{
  "task_id": "hard",
  "sla_max_latency": 210,
  "scores": {
    "aws":   {"cost": 140, "latency": 195, "sla_met": true,  "reward": 0.5200},
    "azure": {"cost": 105, "latency": 215, "sla_met": false, "reward": 0.0},
    "gcp":   {"cost": 125, "latency": 205, "sla_met": true,  "reward": 0.7800}
  },
  "best_provider": "gcp"
}
```

---

## Environment Design

### State

```python
{
    "job_type":        "batch_job",         # ml_training | api_request | batch_job
    "sla_max_latency": 125,                 # milliseconds
    "providers": {
        "aws":   {"cost": 55.0, "latency": 145.0},
        "azure": {"cost": 85.0, "latency": 118.0},
        "gcp":   {"cost": 72.0, "latency": 108.0},
    }
}
```

### Actions

| Action  | Description          |
|---------|----------------------|
| `"aws"` | Select Amazon Web Services |
| `"azure"` | Select Microsoft Azure |
| `"gcp"` | Select Google Cloud Platform |

---

## Reward Function

```
reward = 0.0                          if latency > sla_max_latency  (hard gate)
       = 0.75 × cost_score
       + 0.15 × latency_headroom_ratio
       + 0.10 × efficiency_bonus
       clipped to [0.30, 1.00]
```

| Range       | Interpretation         |
|-------------|------------------------|
| `0.0`       | SLA violated           |
| `0.30–0.59` | Poor — meets SLA but expensive |
| `0.60–0.74` | Fair                   |
| `0.75–0.89` | Good                   |
| `0.90–1.00` | Excellent / optimal    |

---

## Tasks

| Task ID          | Difficulty | Key Challenge                                   |
|------------------|------------|-------------------------------------------------|
| `easy`           | Easy       | All providers meet SLA; pick cheapest           |
| `medium`         | Medium     | Cheapest provider violates SLA                  |
| `hard`           | Hard       | Tight SLA; misleading sticker prices            |
| `bonus_cost`     | Easy       | Pure cost minimization, all compliant           |
| `bonus_strict_sla` | Hard     | Only one provider can possibly meet SLA         |

---

## Baseline Agent

The baseline agent implements a simple greedy policy:

1. Filter providers with `latency ≤ sla_max_latency`
2. Return the provider with the lowest cost
3. Fallback: if none meet SLA, pick lowest latency (best-effort)

Run standalone:

```bash
python baseline/baseline.py
```

Sample output:

```
[easy              ] → gcp    cost=  40.0  lat=  58.0ms  reward=0.9250  ✓  ★ optimal
[medium            ] → gcp    cost=  72.0  lat= 108.0ms  reward=0.8500  ✓  ★ optimal
[hard              ] → gcp    cost= 125.0  lat= 205.0ms  reward=0.7800  ✓  ★ optimal
[bonus_cost        ] → gcp    cost=  38.0  lat=  70.0ms  reward=0.9100  ✓  ★ optimal
[bonus_strict_sla  ] → gcp    cost=  45.0  lat=  55.0ms  reward=0.8800  ✓  ★ optimal

============================================================
  Baseline Performance Summary
============================================================
  Tasks evaluated : 5
  Average reward  : 0.8690
  SLA violations  : 0
  Optimal picks   : 5
============================================================
```

---

## License

MIT — free to use for hackathon and research purposes.
