# 🚀 AI-Powered SLA-Aware Multi-Cloud Optimization Environment

> 🚀 Trains AI agents to make real-world cloud routing decisions under practical SLA constraints — similar to production systems used in large-scale cloud infrastructure.

> An OpenEnv-compatible reinforcement learning environment where AI agents learn to optimize **cost vs latency vs SLA trade-offs** across AWS, Azure, and GCP.

Built for the **Meta × PyTorch Hackathon 2026** hosted by Scaler School of Technology.

---

## 🧠 Problem

Modern cloud systems face complex decision-making challenges.

Choosing the cheapest provider is not enough — real-world systems must balance:

- SLA compliance  
- Cost vs performance trade-offs  
- Dynamic and context-aware decision-making  

Traditional rule-based systems fail to handle these multi-objective constraints effectively.

---

## 💡 Solution

We built an OpenEnv-compatible environment where AI agents:

- Select optimal cloud providers (AWS, Azure, GCP)  
- Respect strict latency constraints (SLA)  
- Optimize cost under real-world trade-offs  

This transforms cloud routing into a **reinforcement learning problem**, enabling intelligent agents to learn optimal strategies over time.

---

## 🧠 AI Component

This project is designed as an **AI-first system**:

- Frames cloud routing as a **reinforcement learning problem**  
- Uses a **reward function** to guide optimal decision-making  
- Supports integration with **PyTorch-based RL agents (PPO, DQN)**  
- Includes an **LLM-based agent baseline** for intelligent reasoning  

👉 Enables training, evaluation, and benchmarking of AI agents in real-world optimization scenarios.

---

## ✨ Key Features

- ✅ Full **OpenEnv spec** compliance — `step()`, `reset()`, `state()`, `openenv.yaml`
- ✅ **Pydantic typed models** — `Observation`, `Action`, `Reward`, `StepResponse`
- ✅ **5 benchmark tasks** spanning easy → medium → hard difficulty
- ✅ **Continuous reward** in `[0, 1]` — not sparse, not binary
- ✅ **LLM baseline agent** using OpenAI-compatible client
- ✅ **Deterministic graders** — reproducible evaluation
- ✅ Deployed on **Hugging Face Spaces** with Docker
- ✅ Flask REST API with full endpoint coverage

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
- Selects provider  
- Receives reward based on SLA + cost efficiency  

---

## 🏆 Reward Function

```
reward = 0.0 ← SLA violated
       = 0.75 × cost_score
       + 0.15 × latency_headroom_ratio
       + 0.10 × efficiency_bonus
```

- Penalizes SLA violations  
- Rewards cost efficiency  
- Encourages optimal decision-making  



## 📊 Impact

This project can be used for:

- Training AI agents for real-world infrastructure decisions  
- Cloud cost optimization systems  
- Reinforcement learning research  
- Intelligent DevOps and FinOps tools  

---

## 📁 Project Structure

```
multi_cloud_optimizer/
├── env/
├── api/
├── tasks/
├── baseline/
├── inference.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🤖 AI Baseline

- Greedy rule-based agent  
- LLM-based agent (Qwen)  
- Comparable performance benchmarking  

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