# 🚀 CLOUD OPTIM : AI-Powered SLA-Aware Multi-Cloud Optimization Environment

> 🏆 AI system that learns optimal cloud decisions under SLA constraints using reward-based optimization.

> 🚀 Simulates real-world cloud routing decisions under practical SLA constraints using an AI-driven environment.

An OpenEnv-compatible system where agents optimize:

👉 **Cost vs Latency vs SLA trade-offs** across AWS, Azure, and GCP  

Built for the **Meta × PyTorch Hackathon 2026**

---

## 📸 UI Preview (Live Dashboard)

### 🧠 Recommendation + Task Insights
![Dashboard UI](./assets/dashboard1.png)

### 📊 Provider Comparison + Visualization
![Chart UI](./assets/dashboard2.png)

---

## 💡 Problem

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

- Reinforcement learning structure:
  - **State → Action → Reward**
- Continuous reward system (not binary)
- Deterministic evaluation (grading system)
- Supports intelligent agents (LLM / RL) using a reward-driven decision framework for SLA-aware optimization.

---

## 🎯 Key Features

- 🌐 Interactive Dashboard UI
- 📊 Scatter Plot Visualization (Cost vs Latency)
- 🧠 AI-generated insights & reasoning
- ⚡ Real-time provider comparison
- 🎯 SLA-aware optimization
- ⭐ Automatic best provider recommendation
- 📈 Clear cost vs latency trade-off visualization

---

## 🛠️ Tech Stack

- Backend: Flask
- AI Logic: Reinforcement Learning (reward-based system)
- Visualization: Charts (Cost vs Latency)
- Deployment: Hugging Face Spaces / Docker
- Language: Python

---

## 📊 Example Output

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

## 📈 Visualization Insight

- X-axis → Cost 💰  
- Y-axis → Latency ⚡  
- Each dot → Cloud provider  
- Highlighted point → Optimal choice  

👉 Helps clearly identify:
- Trade-offs  
- Best provider  
- SLA-safe region  

---

## 🌐 Live Demo

👉 https://nityanama-multi-cloud-optimizer.hf.space/

---

## 📁 Project Structure

```
multi_cloud_optimizer/
│── assets/
│   │── dashboard1.png
│   │── dashboard2.png
│
│── server/
│   │── app.py
│
│── .gitignore
│── Dockerfile
│── README.md
│── index.html
│── inference.py
│── openenv.yaml
│── pyproject.toml
│── requirements.txt
│── uv.lock
```

---

## ⚡ Quickstart

### Run Locally

```bash
git clone https://huggingface.co/spaces/nityanama/multi_cloud_optimizer
cd multi_cloud_optimizer
pip install -r requirements.txt
python server/app.py
```

---
## 📡 API Endpoints

- GET /reset → Start new environment
- POST /step → Take action (aws/azure/gcp)
- GET /tasks → View available tasks
- POST /grader → Evaluate your decision

---


## 🏗️ System Flow

User Request → Flask API → Cloud Environment → Reward Engine → Optimal Cloud Selection

- The API receives a task request
- The environment simulates cloud providers (AWS, Azure, GCP)
- The decision engine evaluates cost, latency, and SLA constraints
- A reward is computed based on performance
- The system outputs the optimal cloud provider

## 🔁 How It Works

```python
observation = env.reset()
action = agent.act(observation)
obs, reward, done, info = env.step(action)
```

---

## 🏆 Reward Function

```python
reward = 0.0  # if SLA violated
reward = 0.75 * cost_score \
       + 0.15 * latency_headroom_ratio \
       + 0.10 * efficiency_bonus
```

---

## 🔍 Explainability

- `/insights/{task}` → AI reasoning  
- `/compare/{task}` → provider comparison  
- `/what_if/{task}` → counterfactual analysis  

---

## 📊 Impact

- AI-based cloud optimization systems  
- Reinforcement learning experimentation  
- DevOps / FinOps intelligent systems  
- Benchmark for AI reasoning  

---

## 🔮 Future Scope

- Multi-step decision environments  
- Dynamic pricing simulation  
- Multi-region cloud modeling  
- RL training (PPO/DQN)  
- Multi-agent benchmarking  

---

## 📄 License

MIT — free to use for research and hackathons

