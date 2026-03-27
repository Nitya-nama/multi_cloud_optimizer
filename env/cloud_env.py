"""
cloud_env.py — OpenEnv-compatible environment for SLA-Aware Multi-Cloud Cost Optimizer.

Implements reset(), step(action), and get_state() following the OpenEnv interface
pattern, where the agent selects a cloud provider to minimize cost while satisfying
latency SLA constraints.
"""

import random
import numpy as np
from typing import Dict, Any, Tuple, Optional


# ---------------------------------------------------------------------------
# Provider catalogue — realistic baseline figures (before per-task noise)
# ---------------------------------------------------------------------------
PROVIDERS = ["aws", "azure", "gcp"]

PROVIDER_PROFILES = {
    "aws": {
        "base_cost":    {"ml_training": 120, "api_request": 40,  "batch_job": 70},
        "base_latency": {"ml_training": 200, "api_request": 60,  "batch_job": 130},
    },
    "azure": {
        "base_cost":    {"ml_training": 110, "api_request": 50,  "batch_job": 80},
        "base_latency": {"ml_training": 220, "api_request": 70,  "batch_job": 120},
    },
    "gcp": {
        "base_cost":    {"ml_training": 100, "api_request": 35,  "batch_job": 65},
        "base_latency": {"ml_training": 180, "api_request": 55,  "batch_job": 110},
    },
}

JOB_TYPES  = ["ml_training", "api_request", "batch_job"]
SLA_LIMITS = {"ml_training": 250, "api_request": 80, "batch_job": 160}  # ms


class CloudEnvironment:
    """
    OpenEnv-compatible environment for multi-cloud job scheduling.

    State
    -----
    {
        "job_type": str,
        "sla_max_latency": int,          # ms — must not be exceeded
        "providers": {
            "aws":   {"cost": float, "latency": float},
            "azure": {"cost": float, "latency": float},
            "gcp":   {"cost": float, "latency": float},
        }
    }

    Actions
    -------
    Any of: "aws", "azure", "gcp"

    Reward
    ------
    0.0            → SLA violation
    0.3 – 0.6     → poor choice (meets SLA but far from optimal)
    0.7 – 0.9     → good choice
    ~1.0           → best choice (lowest cost that meets SLA)
    """

    def __init__(self, task: Optional[Dict] = None, noise: float = 0.10):
        """
        Parameters
        ----------
        task  : optional fixed task dict (used by the grader / baseline endpoints).
        noise : fractional random noise applied to base cost/latency values.
        """
        self._fixed_task = task
        self._noise      = noise
        self._state: Dict[str, Any] = {}
        self._done  = False
        self.reset()

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, Any]:
        """Reset to a new episode (random task or fixed task)."""
        self._done = False

        if self._fixed_task:
            job_type        = self._fixed_task["job_type"]
            sla_max_latency = self._fixed_task["sla_max_latency"]
            provider_data   = self._fixed_task["providers"]
        else:
            job_type        = random.choice(JOB_TYPES)
            sla_max_latency = SLA_LIMITS[job_type]
            provider_data   = self._generate_provider_data(job_type)

        self._state = {
            "job_type":        job_type,
            "sla_max_latency": sla_max_latency,
            "providers":       provider_data,
        }
        return self._state

    def step(self, action: str) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one action.

        Parameters
        ----------
        action : cloud provider name — one of "aws", "azure", "gcp"

        Returns
        -------
        (next_state, reward, done, info)
        """
        if action not in PROVIDERS:
            raise ValueError(f"Invalid action '{action}'. Must be one of {PROVIDERS}.")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        provider_metrics = self._state["providers"][action]
        cost    = provider_metrics["cost"]
        latency = provider_metrics["latency"]
        sla     = self._state["sla_max_latency"]

        reward = self._compute_reward(action, cost, latency, sla)

        info = {
            "selected_cloud": action,
            "cost":           round(cost, 2),
            "latency":        round(latency, 2),
            "sla_max_latency": sla,
            "sla_met":        latency <= sla,
            "reward":         round(reward, 4),
        }

        self._done = True
        return self._state, reward, self._done, info

    def get_state(self) -> Dict[str, Any]:
        """Return a copy of the current environment state."""
        return dict(self._state)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, action: str, cost: float, latency: float, sla: float) -> float:
        """
        Reward function in [0, 1].

        Design
        ------
        1. SLA gate   — if latency > sla, return 0.0 immediately.
        2. Cost score — normalised inverse cost across providers that meet SLA.
        3. Latency headroom bonus — reward having comfortable SLA margin.
        4. Efficiency bonus — small boost for the globally best choice.
        """
        # 1. Hard SLA gate
        if latency > sla:
            return 0.0

        providers  = self._state["providers"]
        valid      = {p: m for p, m in providers.items() if m["latency"] <= sla}

        if not valid:
            # Degenerate: no provider meets SLA — shouldn't happen with sane tasks
            return 0.1

        costs     = [m["cost"]    for m in valid.values()]
        latencies = [m["latency"] for m in valid.values()]

        min_cost  = min(costs)
        max_cost  = max(costs)

        # 2. Cost score ∈ [0, 1] — lower cost → higher score
        if max_cost == min_cost:
            cost_score = 1.0
        else:
            cost_score = 1.0 - (cost - min_cost) / (max_cost - min_cost)

        # 3. Latency headroom bonus ∈ [0, 0.15]
        headroom_ratio  = (sla - latency) / sla          # 0 → tight, 1 → very slack
        headroom_bonus  = 0.15 * headroom_ratio

        # 4. Efficiency bonus — 0.1 if this is the cheapest valid provider
        efficiency_bonus = 0.10 if cost == min_cost else 0.0

        # Weighted combination, clipped to [0.3, 1.0]
        raw_reward = 0.75 * cost_score + headroom_bonus + efficiency_bonus
        reward     = float(np.clip(raw_reward, 0.30, 1.00))

        # Snap to 0.0 just in case (should be caught by gate above)
        return reward

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _generate_provider_data(self, job_type: str) -> Dict[str, Dict]:
        """Add realistic noise to baseline provider figures."""
        data = {}
        for provider, profile in PROVIDER_PROFILES.items():
            base_cost    = profile["base_cost"][job_type]
            base_latency = profile["base_latency"][job_type]

            noisy_cost    = base_cost    * random.uniform(1 - self._noise, 1 + self._noise)
            noisy_latency = base_latency * random.uniform(1 - self._noise, 1 + self._noise)

            data[provider] = {
                "cost":    round(noisy_cost,    2),
                "latency": round(noisy_latency, 2),
            }
        return data

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def render(self) -> str:
        """Human-readable state summary."""
        s = self._state
        lines = [
            f"Job: {s['job_type']}  |  SLA ≤ {s['sla_max_latency']} ms",
            f"{'Provider':<8} {'Cost ($)':<12} {'Latency (ms)':<14} SLA OK?",
            "-" * 46,
        ]
        for p, m in s["providers"].items():
            ok = "✓" if m["latency"] <= s["sla_max_latency"] else "✗"
            lines.append(f"{p:<8} {m['cost']:<12.2f} {m['latency']:<14.2f} {ok}")
        return "\n".join(lines)

    def action_space(self):
        return list(PROVIDERS)

    def observation_space(self):
        return {
            "job_type":        "categorical",
            "sla_max_latency": "continuous",
            "providers":       {p: {"cost": "continuous", "latency": "continuous"} for p in PROVIDERS},
        }
