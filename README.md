# 🚀 AI-Powered SLA-Aware Multi-Cloud Optimization Environment

> 🚀 Simulates real-world cloud routing decisions under practical SLA constraints using an AI-driven environment.

An OpenEnv-compatible system where agents optimize:

👉 **Cost vs Latency vs SLA trade-offs** across AWS, Azure, and GCP

Built for the **Meta × PyTorch Hackathon 2026**

---

## 🧠 Problem

Modern cloud systems must make complex decisions.

Choosing the cheapest provider is not enough — systems must balance:

- SLA (latency constraints)  
- Cost efficiency  
- Dynamic cloud conditions  

Traditional rule-based approaches fail to handle these trade-offs effectively.

---

## 💡 Solution

We model cloud routing as a **decision-making environment**.

Agents interact with the system by:

- Observing cloud conditions  
- Selecting a provider  
- Receiving a reward based on performance  

This enables intelligent optimization of real-world cloud decisions.

---

## 🤖 AI Component

This project is designed as an **AI-driven environment**:

- Reinforcement learning structure:
  - **State → Action → Reward**
- Continuous reward function (not binary)
- Deterministic evaluation (grading system)
- Supports integration with intelligent agents

👉 Designed to evaluate reasoning capabilities of AI agents under real-world constraints



---

## 🌐 Live Demo

👉 https://nityanama-multi_cloud_optimizer.hf.space  

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
  "baseline_reward": 0.9033,
  "better_than_baseline": true,
  "cost": 40,
  "grade": "excellent",
  "is_optimal": true,
  "latency": 58,
  "reward": 0.9033,
  "selected_cloud": "gcp",
  "sla_max_latency": 90
}
```

---

## 🔁 How It Works

```
observation = env.reset()
action = agent.act(observation)
obs, reward, done, info = env.step(action)
```

- Agent observes cloud conditions  
- Selects a cloud provider  
- Receives reward based on SLA + cost efficiency  

## 🏆 Reward Function
```python
reward = 0.0  # if SLA violated
reward = 0.75 * cost_score \
       + 0.15 * latency_headroom_ratio \
       + 0.10 * efficiency_bonus
```

- Penalizes SLA violations
- Rewards cost efficiency
- Encourages optimal decisions

## ✨ Key Features
- ✅ OpenEnv-compatible environment (reset, step, state)
- ✅ Multi-cloud simulation (AWS, Azure, GCP)
- ✅ Continuous reward system (not sparse)
- ✅ Benchmark tasks (easy → medium → hard)
- ✅ Deterministic grading system
- ✅ Flask-based REST API
- ✅ Deployable via Docker / Hugging Face Spaces

## 🔍 Explainability

/explain/{task}
→ Shows reasoning behind decisions and rejected options

## 📊 Insights

/insights/{task}
→ Compares cheapest vs fastest vs optimal

## 🔁 Counterfactual Analysis

/what_if/{task}?action=aws
→ Evaluates impact of suboptimal decisions

👉 Includes counterfactual reasoning to analyze decision quality

## 📸 Example Output

```json
{
  "selected_cloud": "gcp",
  "latency": 58,
  "cost": 40,
  "sla_max_latency": 90,
  "reward": 0.9033,
  "grade": "excellent"
}
```

---




## 📊 Impact

- AI-based cloud optimization systems
- Reinforcement learning experimentation
- DevOps / FinOps intelligent decision systems
- Benchmarking AI reasoning beyond text generation



## 🔮 Future Scope
- Multi-step decision environments
- Dynamic pricing simulation
- Multi-region cloud modeling
- RL training integration (PPO/DQN)
- Visualization dashboard

- **Multi-step episodes** — optimize across a sequence of jobs, not just one
- **Dynamic pricing** — cost and latency that change over time (spot pricing simulation)
- **Multi-region support** — extend to us-east, eu-west, ap-south per provider
- **RL training loop** — PPO/DQN agent trained directly on this environment
- **Frontend dashboard** — visual interface to run tasks, compare providers, watch the agent reason live
- **Multi-agent comparison** — benchmark GPT-4, Claude, Llama side-by-side on the same tasks

## 📄 License

MIT — free to use for research and hackathons