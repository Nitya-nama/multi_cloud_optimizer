---
title: SLA-Aware Multi-Cloud Cost Optimizer
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - cloud-optimization
  - llm-agent
---

# 🚀 SLA-Aware Multi-Cloud Cost Optimizer

> A fully OpenEnv-compliant reinforcement learning environment where an AI agent learns to route cloud workloads across AWS, Azure, and GCP — minimizing cost while guaranteeing latency SLA compliance.

Built for the **Meta × PyTorch Hackathon 2026** hosted by Scaler School of Technology.

---

## 🧠 The Problem

Cloud cost optimization is a real, hard problem faced by every engineering team running production workloads:

- The **cheapest** provider may violate your latency SLA
- The **fastest** provider may be prohibitively expensive
- The **right** choice changes per job type, per SLA window, per time of day

This environment frames it as a sequential decision problem that an RL agent or LLM agent can learn to solve — with a meaningful, continuous reward signal that reflects real-world tradeoffs.

---

## ✨ Key Features

- ✅ Full **OpenEnv spec** compliance — `step()`, `reset()`, `state()`, `openenv.yaml`
- ✅ **Pydantic typed models** — `Observation`, `Action`, `Reward`, `StepResponse`
- ✅ **5 benchmark tasks** spanning easy → medium → hard difficulty
- ✅ **Continuous reward** in `[0, 1]` — not sparse, not binary
- ✅ **LLM baseline** using OpenAI-compatible client (Qwen/Qwen2.5-72B-Instruct)
- ✅ **Deterministic graders** — reproducible scores across runs
- ✅ Deployed on **Hugging Face Spaces** with Docker
- ✅ Flask REST API with full endpoint coverage

---

## 🔁 How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                      OpenEnv Loop                            │
│                                                              │
│  observation = env.reset()       ← new cloud job arrives    │
│  action      = agent.act(obs)    ← agent picks a provider   │
│  obs, reward, done, info = env.step(action)                  │
│                                  ← reward ∈ [0.0, 1.0]      │
└─────────────────────────────────────────────────────────────┘
```

1. A **job** arrives (`ml_training`, `api_request`, or `batch_job`) with randomized cost and latency figures per provider
2. An **SLA constraint** (`sla_max_latency`) sets the hard latency limit in milliseconds
3. The agent selects one of three cloud providers: `aws`, `azure`, or `gcp`
4. The **reward function** returns `0.0` on SLA violation, or a value in `[0.30, 1.00]` based on cost efficiency, latency headroom, and optimality

---

## 📁 Project Structure

```
multi_cloud_optimizer/
├── env/
│   ├── cloud_env.py       # OpenEnv environment (reset, step, get_state)
│   └── models.py          # Pydantic typed models (Observation, Action, Reward)
├── api/
│   └── app.py             # Flask REST API — all OpenEnv endpoints
├── tasks/
│   └── tasks.py           # 5 benchmark tasks (easy → hard)
├── baseline/
│   └── baseline.py        # Greedy rule-based baseline agent
├── inference.py           # LLM agent using OpenAI client (root, required)
├── openenv.yaml           # OpenEnv metadata and task registry
├── Dockerfile             # Production container (gunicorn)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## ⚡ Quickstart

### Option 1 — Local (no Docker)

```bash
# 1. Clone and install
git clone https://huggingface.co/spaces/nityanama/multi_cloud_optimizer
cd multi_cloud_optimizer
pip install -r requirements.txt

# 2. Set environment variables
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_your_token_here

# 3. Start the API server (Terminal 1)
python api/app.py

# 4. Run LLM inference (Terminal 2)
python inference.py
```

### Option 2 — Docker

```bash
docker build -t cloud-optimizer .

docker run -p 7860:7860 \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  -e HF_TOKEN=your_token_here \
  cloud-optimizer
```

### Option 3 — Hugging Face Spaces (live)

```
https://nityanama-multi_cloud_optimizer.hf.space
```

---

## 🔌 API Reference

Base URL: `http://localhost:7860` (local) or `https://nityanama-multi_cloud_optimizer.hf.space` (HF)

### OpenEnv Core Endpoints

#### `GET /reset`
Start a new episode. Returns a fresh observation.

```bash
curl http://localhost:7860/reset
```
```json
{
  "job_type": "batch_job",
  "sla_max_latency": 125,
  "providers": {
    "aws":   {"cost": 55.0, "latency": 145.0},
    "azure": {"cost": 85.0, "latency": 118.0},
    "gcp":   {"cost": 72.0, "latency": 108.0}
  }
}
```

#### `POST /step`
Take an action. Returns next state, reward, done, info.

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "gcp"}'
```
```json
{
  "state":  {"job_type": "batch_job", "sla_max_latency": 125, "providers": {"...": "..."}},
  "reward": 0.8704,
  "done":   true,
  "info":   {"selected_cloud": "gcp", "cost": 72.0, "latency": 108.0, "sla_met": true}
}
```

#### `GET /state`
Return the current environment state without advancing the episode.

```bash
curl http://localhost:7860/state
```

---

### Task & Grader Endpoints

#### `GET /tasks`
List all benchmark tasks.

#### `GET /tasks/<task_id>`
Full provider metrics for a specific task.

#### `POST /grader`
Score a cloud selection against a task.

```bash
curl -X POST http://localhost:7860/grader \
  -H "Content-Type: application/json" \
  -d '{"task_id": "hard", "selected_cloud": "gcp"}'
```
```json
{
  "task_id": "hard",
  "selected_cloud": "gcp",
  "cost": 125,
  "latency": 205,
  "sla_max_latency": 210,
  "sla_met": true,
  "reward": 0.8536,
  "grade": "good"
}
```

#### `GET /baseline`
Run the greedy baseline agent across all 5 tasks.

#### `GET /compare/<task_id>`
Score all three providers against a task simultaneously.

#### `GET /health`
Liveness check — returns `{"status": "ok"}`.

---

## 🧩 Environment Design

### Observation Space

```python
class Observation(BaseModel):
    job_type: str                                      # ml_training | api_request | batch_job
    sla_max_latency: float                             # Hard latency limit in milliseconds
    providers: Dict[Literal["aws","azure","gcp"], ProviderMetrics]
```

### Action Space

```python
class Action(BaseModel):
    action: Literal["aws", "azure", "gcp"]             # Discrete — pick one provider
```

### Reward Model

```python
class Reward(BaseModel):
    reward: float    # ge=0.0, le=1.0
    done:   bool
    info:   Dict
```

---

## 🏆 Reward Function

```
reward = 0.0                            ← SLA violated (hard gate)
       = 0.75 × cost_score
       + 0.15 × latency_headroom_ratio
       + 0.10 × efficiency_bonus
         clipped to [0.30, 1.00]
```

| Score Range | Grade     | Meaning                             |
|-------------|-----------|-------------------------------------|
| `0.0`       | Failed    | SLA violated                        |
| `0.30–0.59` | Poor      | Meets SLA but far from optimal cost |
| `0.60–0.74` | Fair      | Reasonable choice                   |
| `0.75–0.89` | Good      | Near-optimal                        |
| `0.90–1.00` | Excellent | Optimal — cheapest SLA-compliant    |

**Design rationale:** The reward is not sparse. It gives the agent a gradient to learn from, rewarding cost efficiency, SLA headroom, and picking the globally best option.

---

## 📋 Tasks

| Task ID            | Difficulty | Job Type    | SLA (ms) | Key Challenge                          |
|--------------------|------------|-------------|----------|----------------------------------------|
| `easy`             | Easy       | api_request | 90       | All providers comply — pick cheapest   |
| `medium`           | Medium     | batch_job   | 125      | Cheapest provider violates SLA         |
| `hard`             | Hard       | ml_training | 210      | Tight SLA + misleading sticker prices  |
| `bonus_cost`       | Easy       | api_request | 100      | Pure cost minimization, all comply     |
| `bonus_strict_sla` | Hard       | api_request | 60       | Only one provider can qualify          |

---

## 🤖 LLM Baseline

The LLM agent uses the OpenAI-compatible client:

```python
client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=10,
    temperature=0,
)
```

### Baseline Scores

| Agent                     | Avg Reward | SLA Violations |
|---------------------------|------------|----------------|
| Greedy (rule-based)       | 0.877      | 0              |
| Qwen/Qwen2.5-72B-Instruct | 0.877      | 0              |

### Running inference

```bash
python inference.py
```

Expected:
```
Tasks evaluated : 5
Average reward  : 0.877
SLA violations  : 0
```

---

## 🐳 Docker

```bash
docker build -t cloud-optimizer .

docker run -p 7860:7860 \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  -e HF_TOKEN=hf_your_token \
  cloud-optimizer
```

---

## ⚙️ Environment Variables

| Variable       | Description                      | Example                            |
|----------------|----------------------------------|------------------------------------|
| `API_BASE_URL` | LLM API endpoint (OpenAI-compat) | `https://router.huggingface.co/v1` |
| `MODEL_NAME`   | Model identifier for inference   | `Qwen/Qwen2.5-72B-Instruct`        |
| `HF_TOKEN`     | Hugging Face / API key           | `hf_xxxxxxxxxxxxxxxxxxxx`          |

---

## 🔮 Future Scope

- **Multi-step episodes** — optimize across a sequence of jobs, not just one
- **Dynamic pricing** — cost and latency that change over time (spot pricing simulation)
- **Multi-region support** — extend to us-east, eu-west, ap-south per provider
- **RL training loop** — PPO/DQN agent trained directly on this environment
- **Frontend dashboard** — visual interface to run tasks, compare providers, watch the agent reason live
- **Multi-agent comparison** — benchmark GPT-4, Claude, Llama side-by-side on the same tasks

---

## 📄 License

MIT — free to use for research and hackathon purposes.
