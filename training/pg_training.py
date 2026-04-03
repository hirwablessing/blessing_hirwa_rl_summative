"""
Policy Gradient Training Script for African Literacy AI Tutor.

Implements three algorithms:
1. REINFORCE (manual PyTorch implementation)
2. PPO (Proximal Policy Optimization via Stable Baselines 3)
3. A2C (Advantage Actor-Critic via Stable Baselines 3)

Each algorithm is trained with 10 hyperparameter configurations.
"""

import os
import sys
import json
import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from environment.custom_env import AfricanLiteracyTutorEnv


# ============================================================
# REINFORCE Implementation (Manual PyTorch)
# ============================================================

class REINFORCEPolicy(nn.Module):
    """Policy network for REINFORCE algorithm."""

    def __init__(self, obs_dim=23, act_dim=15, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
        )

    def forward(self, x):
        logits = self.network(x)
        return Categorical(logits=logits)

    def get_action(self, obs):
        dist = self.forward(obs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()


class REINFORCEAgent:
    """REINFORCE agent with baseline subtraction."""

    def __init__(self, lr=1e-3, gamma=0.99, hidden_size=128, ent_coef=0.01,
                 grad_clip=0.5, baseline_alpha=0.01):
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.grad_clip = grad_clip
        self.policy = REINFORCEPolicy(hidden_size=hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.baseline = 0.0
        self.baseline_alpha = baseline_alpha

    def train_episode(self, env):
        """Collect one episode and update policy."""
        obs, _ = env.reset()
        log_probs = []
        rewards = []
        entropies = []
        done = False

        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            action, log_prob, entropy = self.policy.get_action(obs_t)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)

        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        # Baseline subtraction (running mean)
        self.baseline = self.baseline_alpha * returns.mean().item() + (1 - self.baseline_alpha) * self.baseline
        advantages = returns - self.baseline

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy gradient loss
        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)
        policy_loss = -(log_probs_t * advantages.detach()).mean()
        entropy_loss = -entropies_t.mean()

        # Entropy bonus for exploration (uses configurable coefficient)
        loss = policy_loss + self.ent_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()

        return {
            "episode_reward": sum(rewards),
            "episode_length": len(rewards),
            "policy_loss": policy_loss.item(),
            "entropy": entropies_t.mean().item(),
        }

    def predict(self, obs, deterministic=False):
        """Predict action for evaluation."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            dist = self.policy.forward(obs_t)
            if deterministic:
                action = dist.probs.argmax().item()
            else:
                action = dist.sample().item()
        return action, None

    def save(self, path):
        torch.save(self.policy.state_dict(), path + ".pt")

    def load(self, path):
        self.policy.load_state_dict(torch.load(path + ".pt", weights_only=True))


# ============================================================
# SB3 Training Callback
# ============================================================

class PGMetricsCallback(BaseCallback):
    """Callback to log PG training metrics."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.entropies = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])

        # Track entropy
        if hasattr(self.model, "logger") and self.model.logger is not None:
            try:
                ent = self.model.logger.name_to_value.get("train/entropy_loss", None)
                if ent is not None:
                    self.entropies.append(abs(float(ent)))
            except Exception:
                pass

        return True


# ============================================================
# Hyperparameter Configurations
# ============================================================

REINFORCE_PARAMS = [
    {"run": 1, "learning_rate": 1e-3, "gamma": 0.99, "hidden_size": 128, "ent_coef": 0.01, "grad_clip": 0.5, "baseline_alpha": 0.01},
    {"run": 2, "learning_rate": 5e-4, "gamma": 0.99, "hidden_size": 256, "ent_coef": 0.005, "grad_clip": 1.0, "baseline_alpha": 0.02},
    {"run": 3, "learning_rate": 1e-4, "gamma": 0.95, "hidden_size": 128, "ent_coef": 0.02, "grad_clip": 0.5, "baseline_alpha": 0.05},
    {"run": 4, "learning_rate": 3e-3, "gamma": 0.98, "hidden_size": 64, "ent_coef": 0.0, "grad_clip": 0.3, "baseline_alpha": 0.01},
    {"run": 5, "learning_rate": 7e-4, "gamma": 0.97, "hidden_size": 128, "ent_coef": 0.015, "grad_clip": 0.8, "baseline_alpha": 0.03},
    {"run": 6, "learning_rate": 1e-3, "gamma": 0.99, "hidden_size": 256, "ent_coef": 0.03, "grad_clip": 0.5, "baseline_alpha": 0.005},
    {"run": 7, "learning_rate": 2e-4, "gamma": 0.96, "hidden_size": 128, "ent_coef": 0.01, "grad_clip": 2.0, "baseline_alpha": 0.1},
    {"run": 8, "learning_rate": 5e-4, "gamma": 0.995, "hidden_size": 64, "ent_coef": 0.005, "grad_clip": 0.5, "baseline_alpha": 0.02},
    {"run": 9, "learning_rate": 1e-3, "gamma": 0.93, "hidden_size": 256, "ent_coef": 0.02, "grad_clip": 0.3, "baseline_alpha": 0.01},
    {"run": 10, "learning_rate": 3e-4, "gamma": 0.99, "hidden_size": 128, "ent_coef": 0.01, "grad_clip": 1.0, "baseline_alpha": 0.05},
]

PPO_PARAMS = [
    {"run": 1, "learning_rate": 3e-4, "gamma": 0.99, "n_steps": 2048, "clip_range": 0.2, "ent_coef": 0.01, "batch_size": 64, "n_epochs": 10, "gae_lambda": 0.95},
    {"run": 2, "learning_rate": 1e-4, "gamma": 0.99, "n_steps": 1024, "clip_range": 0.1, "ent_coef": 0.005, "batch_size": 128, "n_epochs": 5, "gae_lambda": 0.98},
    {"run": 3, "learning_rate": 5e-4, "gamma": 0.95, "n_steps": 2048, "clip_range": 0.3, "ent_coef": 0.02, "batch_size": 64, "n_epochs": 10, "gae_lambda": 0.9},
    {"run": 4, "learning_rate": 3e-4, "gamma": 0.98, "n_steps": 4096, "clip_range": 0.2, "ent_coef": 0.01, "batch_size": 256, "n_epochs": 15, "gae_lambda": 0.95},
    {"run": 5, "learning_rate": 7e-4, "gamma": 0.97, "n_steps": 512, "clip_range": 0.15, "ent_coef": 0.0, "batch_size": 32, "n_epochs": 10, "gae_lambda": 0.95},
    {"run": 6, "learning_rate": 2e-4, "gamma": 0.99, "n_steps": 2048, "clip_range": 0.25, "ent_coef": 0.01, "batch_size": 64, "n_epochs": 10, "gae_lambda": 0.92},
    {"run": 7, "learning_rate": 1e-3, "gamma": 0.96, "n_steps": 1024, "clip_range": 0.2, "ent_coef": 0.03, "batch_size": 64, "n_epochs": 5, "gae_lambda": 0.95},
    {"run": 8, "learning_rate": 3e-4, "gamma": 0.995, "n_steps": 2048, "clip_range": 0.2, "ent_coef": 0.005, "batch_size": 128, "n_epochs": 20, "gae_lambda": 0.99},
    {"run": 9, "learning_rate": 5e-4, "gamma": 0.93, "n_steps": 1024, "clip_range": 0.1, "ent_coef": 0.02, "batch_size": 64, "n_epochs": 10, "gae_lambda": 0.95},
    {"run": 10, "learning_rate": 3e-4, "gamma": 0.99, "n_steps": 2048, "clip_range": 0.2, "ent_coef": 0.01, "batch_size": 64, "n_epochs": 10, "gae_lambda": 0.95},
]

A2C_PARAMS = [
    {"run": 1, "learning_rate": 7e-4, "gamma": 0.99, "n_steps": 5, "ent_coef": 0.01, "vf_coef": 0.5, "gae_lambda": 1.0},
    {"run": 2, "learning_rate": 3e-4, "gamma": 0.99, "n_steps": 16, "ent_coef": 0.005, "vf_coef": 0.25, "gae_lambda": 0.95},
    {"run": 3, "learning_rate": 1e-3, "gamma": 0.95, "n_steps": 5, "ent_coef": 0.02, "vf_coef": 0.5, "gae_lambda": 1.0},
    {"run": 4, "learning_rate": 5e-4, "gamma": 0.98, "n_steps": 32, "ent_coef": 0.01, "vf_coef": 0.75, "gae_lambda": 0.9},
    {"run": 5, "learning_rate": 7e-4, "gamma": 0.97, "n_steps": 10, "ent_coef": 0.0, "vf_coef": 0.5, "gae_lambda": 0.95},
    {"run": 6, "learning_rate": 2e-4, "gamma": 0.99, "n_steps": 5, "ent_coef": 0.01, "vf_coef": 0.5, "gae_lambda": 0.98},
    {"run": 7, "learning_rate": 1e-3, "gamma": 0.96, "n_steps": 8, "ent_coef": 0.03, "vf_coef": 0.25, "gae_lambda": 1.0},
    {"run": 8, "learning_rate": 5e-4, "gamma": 0.995, "n_steps": 20, "ent_coef": 0.005, "vf_coef": 0.5, "gae_lambda": 0.95},
    {"run": 9, "learning_rate": 7e-4, "gamma": 0.93, "n_steps": 5, "ent_coef": 0.02, "vf_coef": 0.5, "gae_lambda": 1.0},
    {"run": 10, "learning_rate": 3e-4, "gamma": 0.99, "n_steps": 10, "ent_coef": 0.01, "vf_coef": 0.5, "gae_lambda": 0.95},
]

TOTAL_TIMESTEPS = 50000
REINFORCE_EPISODES = 300


# ============================================================
# Training Functions
# ============================================================

def train_reinforce(params, save_dir):
    """Train REINFORCE with given hyperparameters."""
    run_id = params["run"]
    print(f"\n{'='*60}")
    print(f"REINFORCE Run {run_id}/10")
    print(f"{'='*60}")

    env = AfricanLiteracyTutorEnv()
    agent = REINFORCEAgent(
        lr=params["learning_rate"],
        gamma=params["gamma"],
        hidden_size=params["hidden_size"],
        ent_coef=params["ent_coef"],
        grad_clip=params["grad_clip"],
        baseline_alpha=params["baseline_alpha"],
    )

    episode_rewards = []
    entropies = []
    losses = []

    start_time = time.time()
    for ep in range(REINFORCE_EPISODES):
        metrics = agent.train_episode(env)
        episode_rewards.append(metrics["episode_reward"])
        entropies.append(metrics["entropy"])
        losses.append(metrics["policy_loss"])

        if (ep + 1) % 50 == 0:
            recent = np.mean(episode_rewards[-50:])
            print(f"  Episode {ep+1}/{REINFORCE_EPISODES} | Recent Mean Reward: {recent:.2f} | Entropy: {metrics['entropy']:.4f}")

    train_time = time.time() - start_time

    # Save model
    model_path = os.path.join(save_dir, f"reinforce_run_{run_id}")
    agent.save(model_path)

    # Evaluate
    eval_rewards = evaluate_reinforce(agent, n_episodes=20)
    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)

    # Convergence step
    convergence_step = REINFORCE_EPISODES
    if len(episode_rewards) > 10:
        rolling = np.convolve(episode_rewards, np.ones(10) / 10, mode="valid")
        threshold = mean_reward * 0.5
        converged = np.where(rolling > threshold)[0]
        if len(converged) > 0:
            convergence_step = int(converged[0])

    print(f"  Run {run_id}: Mean={mean_reward:.2f} +/- {std_reward:.2f}")
    env.close()

    return {
        "run": run_id,
        "mean_reward": round(float(mean_reward), 2),
        "std_reward": round(float(std_reward), 2),
        "convergence_step": convergence_step,
        "train_time_s": round(train_time, 1),
        "episode_rewards": episode_rewards,
        "entropies": entropies,
        "losses": losses,
        **{k: v for k, v in params.items() if k != "run"},
    }


def evaluate_reinforce(agent, n_episodes=20):
    env = AfricanLiteracyTutorEnv()
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
    env.close()
    return rewards


def train_sb3_algorithm(algo_class, algo_name, params, save_dir, total_timesteps):
    """Train an SB3 algorithm (PPO or A2C)."""
    run_id = params["run"]
    print(f"\n{'='*60}")
    print(f"{algo_name} Run {run_id}/10")
    print(f"{'='*60}")

    env = Monitor(AfricanLiteracyTutorEnv())
    callback = PGMetricsCallback()

    model_params = {k: v for k, v in params.items() if k != "run"}
    model = algo_class(
        "MlpPolicy",
        env,
        **model_params,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        seed=run_id * 42,
    )

    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    train_time = time.time() - start_time

    # Save model
    model_path = os.path.join(save_dir, f"{algo_name.lower()}_run_{run_id}")
    model.save(model_path)

    # Evaluate
    eval_rewards = []
    eval_env = AfricanLiteracyTutorEnv()
    for _ in range(20):
        obs, _ = eval_env.reset()
        ep_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            ep_reward += reward
            done = terminated or truncated
        eval_rewards.append(ep_reward)

    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)

    convergence_step = total_timesteps
    if len(callback.episode_rewards) > 10:
        rolling = np.convolve(callback.episode_rewards, np.ones(10) / 10, mode="valid")
        threshold = mean_reward * 0.5
        converged = np.where(rolling > threshold)[0]
        if len(converged) > 0:
            convergence_step = int(converged[0] * (total_timesteps / max(len(callback.episode_rewards), 1)))

    print(f"  Run {run_id}: Mean={mean_reward:.2f} +/- {std_reward:.2f}")
    env.close()
    eval_env.close()

    return {
        "run": run_id,
        "mean_reward": round(float(mean_reward), 2),
        "std_reward": round(float(std_reward), 2),
        "convergence_step": convergence_step,
        "train_time_s": round(train_time, 1),
        "episode_rewards": callback.episode_rewards,
        "entropies": callback.entropies,
        **{k: v for k, v in params.items() if k != "run"},
    }


# ============================================================
# Plotting
# ============================================================

def plot_all_results(reinforce_results, ppo_results, a2c_results, save_dir):
    """Generate comprehensive plots for all PG methods."""

    # --- Cumulative reward curves (all methods) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Policy Gradient Methods - Training Analysis", fontsize=14)

    # Subplot 1: REINFORCE reward curves
    ax = axes[0, 0]
    for res in reinforce_results:
        if res["episode_rewards"]:
            rolling = np.convolve(res["episode_rewards"], np.ones(10) / 10, mode="valid") if len(res["episode_rewards"]) > 10 else res["episode_rewards"]
            ax.plot(rolling, label=f"R{res['run']} LR={res['learning_rate']}", alpha=0.7)
    ax.set_title("REINFORCE - Episode Rewards")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (10-ep avg)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Subplot 2: PPO reward curves
    ax = axes[0, 1]
    for res in ppo_results:
        if res["episode_rewards"]:
            rolling = np.convolve(res["episode_rewards"], np.ones(10) / 10, mode="valid") if len(res["episode_rewards"]) > 10 else res["episode_rewards"]
            ax.plot(rolling, label=f"R{res['run']} LR={res['learning_rate']}", alpha=0.7)
    ax.set_title("PPO - Episode Rewards")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (10-ep avg)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Subplot 3: A2C reward curves
    ax = axes[1, 0]
    for res in a2c_results:
        if res["episode_rewards"]:
            rolling = np.convolve(res["episode_rewards"], np.ones(10) / 10, mode="valid") if len(res["episode_rewards"]) > 10 else res["episode_rewards"]
            ax.plot(rolling, label=f"R{res['run']} LR={res['learning_rate']}", alpha=0.7)
    ax.set_title("A2C - Episode Rewards")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (10-ep avg)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Subplot 4: Cross-method comparison (best runs)
    ax = axes[1, 1]
    best_reinforce = max(reinforce_results, key=lambda x: x["mean_reward"])
    best_ppo = max(ppo_results, key=lambda x: x["mean_reward"])
    best_a2c = max(a2c_results, key=lambda x: x["mean_reward"])
    methods = ["REINFORCE", "PPO", "A2C"]
    means = [best_reinforce["mean_reward"], best_ppo["mean_reward"], best_a2c["mean_reward"]]
    stds = [best_reinforce["std_reward"], best_ppo["std_reward"], best_a2c["std_reward"]]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    ax.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.set_title("Best Run Comparison (PG Methods)")
    ax.set_ylabel("Mean Eval Reward")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pg_reward_analysis.png"), dpi=150)
    plt.close()

    # --- Entropy curves ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Policy Gradient - Entropy Curves", fontsize=14)

    for idx, (results, name) in enumerate([
        (reinforce_results, "REINFORCE"),
        (ppo_results, "PPO"),
        (a2c_results, "A2C"),
    ]):
        ax = axes[idx]
        for res in results:
            entropies = res.get("entropies", [])
            if entropies:
                smoothed = np.convolve(entropies, np.ones(20) / 20, mode="valid") if len(entropies) > 20 else entropies
                ax.plot(smoothed, label=f"Run {res['run']}", alpha=0.7)
        ax.set_title(f"{name} Entropy")
        ax.set_xlabel("Update Step")
        ax.set_ylabel("Entropy")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pg_entropy_curves.png"), dpi=150)
    plt.close()

    print(f"\nPG plots saved to {save_dir}")


def save_results_csv(results, filename, save_dir, extra_fields):
    """Save results as CSV."""
    csv_path = os.path.join(save_dir, filename)
    fieldnames = ["run"] + extra_fields + ["mean_reward", "std_reward", "convergence_step", "train_time_s"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow({k: res.get(k, "") for k in fieldnames})
    print(f"Saved {csv_path}")


# ============================================================
# Main
# ============================================================

def main():
    save_dir = os.path.join(os.path.dirname(__file__), "..", "models", "pg")
    os.makedirs(save_dir, exist_ok=True)

    # --- REINFORCE ---
    print("\n" + "=" * 70)
    print("TRAINING REINFORCE (Manual PyTorch Implementation)")
    print("=" * 70)
    reinforce_results = []
    for params in REINFORCE_PARAMS:
        result = train_reinforce(params, save_dir)
        reinforce_results.append(result)

    # --- PPO ---
    print("\n" + "=" * 70)
    print("TRAINING PPO (Stable Baselines 3)")
    print("=" * 70)
    ppo_results = []
    for params in PPO_PARAMS:
        result = train_sb3_algorithm(PPO, "PPO", params, save_dir, TOTAL_TIMESTEPS)
        ppo_results.append(result)

    # --- A2C ---
    print("\n" + "=" * 70)
    print("TRAINING A2C (Stable Baselines 3)")
    print("=" * 70)
    a2c_results = []
    for params in A2C_PARAMS:
        result = train_sb3_algorithm(A2C, "A2C", params, save_dir, TOTAL_TIMESTEPS)
        a2c_results.append(result)

    # --- Results ---
    best_reinforce = max(reinforce_results, key=lambda x: x["mean_reward"])
    best_ppo = max(ppo_results, key=lambda x: x["mean_reward"])
    best_a2c = max(a2c_results, key=lambda x: x["mean_reward"])

    print(f"\n{'='*60}")
    print("BEST RUNS:")
    print(f"  REINFORCE: Run {best_reinforce['run']} -> {best_reinforce['mean_reward']:.2f}")
    print(f"  PPO:       Run {best_ppo['run']} -> {best_ppo['mean_reward']:.2f}")
    print(f"  A2C:       Run {best_a2c['run']} -> {best_a2c['mean_reward']:.2f}")
    print(f"{'='*60}")

    # Save best run info
    with open(os.path.join(save_dir, "best_runs.json"), "w") as f:
        json.dump({
            "reinforce": {"run": best_reinforce["run"], "mean_reward": best_reinforce["mean_reward"]},
            "ppo": {"run": best_ppo["run"], "mean_reward": best_ppo["mean_reward"]},
            "a2c": {"run": best_a2c["run"], "mean_reward": best_a2c["mean_reward"]},
        }, f, indent=2)

    # Save CSV tables
    save_results_csv(reinforce_results, "reinforce_results.csv", save_dir,
                     ["learning_rate", "gamma", "hidden_size", "ent_coef", "grad_clip", "baseline_alpha"])
    save_results_csv(ppo_results, "ppo_results.csv", save_dir,
                     ["learning_rate", "gamma", "n_steps", "clip_range", "ent_coef", "batch_size", "n_epochs", "gae_lambda"])
    save_results_csv(a2c_results, "a2c_results.csv", save_dir,
                     ["learning_rate", "gamma", "n_steps", "ent_coef", "vf_coef", "gae_lambda"])

    # Generate plots
    plot_all_results(reinforce_results, ppo_results, a2c_results, save_dir)


if __name__ == "__main__":
    main()
