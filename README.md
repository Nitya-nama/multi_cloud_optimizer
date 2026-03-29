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

## ⚙️ How It Works


```python
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
✅ OpenEnv-compatible environment (reset, step, state)
✅ Multi-cloud simulation (AWS, Azure, GCP)
✅ Continuous reward system (not sparse)
✅ Benchmark tasks (easy → medium → hard)
✅ Deterministic grading system
✅ Flask-based REST API
✅ Deployable via Docker / Hugging Face Spaces
⭐ Advanced Features (WOW Factor)
🔍 Explainability

/explain/{task}
→ Shows reasoning behind decisions and rejected options

## 📊 Insights

/insights/{task}
→ Compares cheapest vs fastest vs optimal

🔁 Counterfactual Analysis

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

## 🌐 Live Demo

👉 https://nityanama-multi_cloud_optimizer.hf.space

---
## 📁 Project Structure
```
SCALER_FLAT/
├── env/
│   ├── cloud_env.py
│   └── models.py
├── api/
│   └── app.py
├── tasks/
│   └── tasks.py
├── baseline/
│   └── baseline.py
├── inference.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

---
## 🔌 API Example

```bash
GET /reset
curl http://localhost:7860/reset
```

**Response:**

```json
{
  "baseline_reward": 0.9033,
  "better_than_baseline": true,
  "cost": 40,
  "grade": "excellent",
  "latency": 58,
  "reward": 0.9033,
  "selected_cloud": "gcp",
  "sla_max_latency": 90
}
```


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


## 👥 Team

- Nitya Phaneesh Chandra Nama  
- Vanditha Hamsa S B  
- Chandan N  

## 📄 License

MIT — free to use for research and hackathons