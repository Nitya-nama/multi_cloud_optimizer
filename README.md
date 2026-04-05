# 🚀 CLOUD OPTIM : AI-Powered SLA-Aware Multi-Cloud Optimization Environment

🏆 Built for Meta × PyTorch OpenEnv Hackathon 2026

> 🧠 AI system for optimizing cloud decisions under SLA constraints using reward-based evaluation

> 🚀 Simulates real-world cloud routing decisions under practical SLA constraints using an AI-driven environment

An OpenEnv-compatible system where agents optimize:

👉 **Cost vs Latency vs SLA trade-offs** across AWS, Azure, and GCP

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

* SLA (latency constraints)
* Cost efficiency
* Dynamic cloud conditions

Traditional rule-based approaches fail to handle these trade-offs effectively.

---

## 💡 Solution

We model cloud routing as a **decision-making environment**.

Agents interact with the system by:

* Observing cloud conditions
* Selecting a provider
* Receiving a reward based on performance

This enables intelligent optimization of real-world cloud decisions.

👉 This environment is compatible with both RL agents and LLM-based decision agents.

---

## 🧠 Environment Design (OpenEnv Compatible)

This project implements a Reinforcement Learning (RL)-style environment:

* `reset()` → initializes environment
* `step(action)` → agent takes decision
* reward → computed based on SLA, cost, and latency
* done → episode termination

### 🔁 Interaction Loop

```python
observation = env.reset()
action = agent.act(observation)
next_obs, reward, done, info = env.step(action)
```

---

## 🤖 AI Component

* Reinforcement learning structure:

  * **State → Action → Reward**
* Continuous reward system (not binary)
* Deterministic evaluation (grading system)
* Supports intelligent agents (LLM / RL) using a reward-driven decision framework for SLA-aware optimization.

---

## 🤖 Agent Implementations

### 1. LLM-Based Agent (Implemented)

* Uses a language model to:

  * Filter SLA-compliant providers
  * Optimize cost vs latency trade-offs
* Acts as an intelligent decision-making agent

---

### 2. Baseline Agent

* Greedy strategy:

  * Selects cheapest SLA-compliant provider
* Used for benchmarking performance

---

### 3. RL Agent (Supported)

The environment supports integration with RL algorithms:

* PPO
* DQN
* Policy Gradient

👉 Agents can be trained using the provided environment API.

👉 Enables comparison between rule-based, LLM-based, and learning-based approaches.

---

## 🎯 Key Features

* 🌐 Interactive Dashboard UI
* 📊 Scatter Plot Visualization (Cost vs Latency)
* 🧠 AI-generated insights & reasoning
* ⚡ Real-time provider comparison
* 🎯 SLA-aware optimization
* ⭐ Automatic best provider recommendation
* 📈 Clear cost vs latency trade-off visualization

---

## 🛠️ Tech Stack

* Backend: Flask
* AI Logic: Reward-based decision system (RL-compatible)
* Visualization: Charts (Cost vs Latency)
* Deployment: Hugging Face Spaces / Docker
* Language: Python

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

* X-axis → Cost 💰
* Y-axis → Latency ⚡
* Each dot → Cloud provider
* Highlighted point → Optimal choice

👉 Helps clearly identify:

* Trade-offs
* Best provider
* SLA-safe region

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
│── inference.py
│── openenv.yaml
│── index.html
│── Dockerfile
│── README.md
│── requirements.txt
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

* GET /reset → Start new environment
* POST /step → Take action (aws/azure/gcp)
* GET /tasks → View available tasks
* POST /grader → Evaluate your decision

---

## 🏗️ System Flow

User Request → Flask API → Cloud Environment → Reward Engine → Optimal Cloud Selection

* The API receives a task request
* The environment simulates cloud providers (AWS, Azure, GCP)
* The decision engine evaluates cost, latency, and SLA constraints
* A reward is computed based on performance
* The system outputs the optimal cloud provider

---

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

## 🧪 Evaluation & Grader

The system includes a built-in evaluation framework:

* Reward score (0–1 scale)
* SLA compliance check
* Baseline comparison
* Optimal decision validation

### 📊 Evaluation Metrics

* Average reward
* SLA violation rate
* Cost efficiency
* Latency performance

👉 Enables benchmarking of different agent strategies.

---

## 🔍 Explainability

* `/insights/{task}` → AI reasoning
* `/compare/{task}` → provider comparison
* `/what_if/{task}` → counterfactual analysis

---

## 📊 Impact

* AI-based cloud optimization systems
* Reinforcement learning experimentation
* DevOps / FinOps intelligent systems
* Benchmark for AI reasoning

---

## ⚖️ Baseline vs Agent Comparison

We compare different decision strategies:

* Greedy baseline (rule-based)
* LLM-based agent (reasoning-driven)

This helps measure:

* Improvement in reward
* Reduction in SLA violations
* Decision quality under constraints

---

## 💡 Why This Project Stands Out

* Combines RL environment + LLM reasoning
* Models real-world cloud decision trade-offs
* Includes full evaluation pipeline (tasks + grader + baseline)
* OpenEnv-compatible for research and experimentation

---

## 🔮 Future Scope

* Multi-step decision environments
* Dynamic pricing simulation
* Multi-region cloud modeling
* RL training (PPO/DQN)
* Multi-agent benchmarking

---

## 📄 License

MIT — free to use for research and hackathons
