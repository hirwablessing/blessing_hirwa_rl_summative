"""
DQN (Deep Q-Network) Training Script for African Literacy AI Tutor.

Trains DQN with 10 different hyperparameter configurations using Stable Baselines 3.
Saves models, logs, and generates training curves.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from environment.custom_env import AfricanLiteracyTutorEnv


class TrainingMetricsCallback(BaseCallback):
    """Callback to log training metrics for analysis."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.current_episode_reward = 0

    def _on_step(self):
        # Track losses from the logger
        if hasattr(self.model, "logger") and self.model.logger is not None:
            try:
                loss = self.model.logger.name_to_value.get("train/loss", None)
                if loss is not None:
                    self.losses.append(float(loss))
            except Exception:
                pass

        # Track episode rewards
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

        return True


# 10 hyperparameter configurations with meaningful variation
HYPERPARAMS = [
    {
        "run": 1,
        "learning_rate": 1e-3,
        "buffer_size": 50000,
        "exploration_fraction": 0.2,
        "exploration_final_eps": 0.05,
        "gamma": 0.99,
        "batch_size": 64,
        "train_freq": 4,
        "target_update_interval": 1000,
        "learning_starts": 1000,
    },
    {
        "run": 2,
        "learning_rate": 5e-4,
        "buffer_size": 100000,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.02,
        "gamma": 0.99,
        "batch_size": 128,
        "train_freq": 4,
        "target_update_interval": 500,
        "learning_starts": 2000,
    },
    {
        "run": 3,
        "learning_rate": 1e-4,
        "buffer_size": 50000,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.05,
        "gamma": 0.95,
        "batch_size": 32,
        "train_freq": 8,
        "target_update_interval": 2000,
        "learning_starts": 1000,
    },
    {
        "run": 4,
        "learning_rate": 5e-4,
        "buffer_size": 30000,
        "exploration_fraction": 0.15,
        "exploration_final_eps": 0.1,
        "gamma": 0.98,
        "batch_size": 64,
        "train_freq": 4,
        "target_update_interval": 750,
        "learning_starts": 500,
    },
    {
        "run": 5,
        "learning_rate": 1e-3,
        "buffer_size": 100000,
        "exploration_fraction": 0.25,
        "exploration_final_eps": 0.01,
        "gamma": 0.97,
        "batch_size": 256,
        "train_freq": 2,
        "target_update_interval": 1000,
        "learning_starts": 3000,
    },
    {
        "run": 6,
        "learning_rate": 3e-4,
        "buffer_size": 75000,
        "exploration_fraction": 0.2,
        "exploration_final_eps": 0.05,
        "gamma": 0.99,
        "batch_size": 64,
        "train_freq": 4,
        "target_update_interval": 500,
        "learning_starts": 1000,
    },
    {
        "run": 7,
        "learning_rate": 7e-4,
        "buffer_size": 50000,
        "exploration_fraction": 0.35,
        "exploration_final_eps": 0.08,
        "gamma": 0.96,
        "batch_size": 128,
        "train_freq": 1,
        "target_update_interval": 1500,
        "learning_starts": 2000,
    },
    {
        "run": 8,
        "learning_rate": 2e-4,
        "buffer_size": 200000,
        "exploration_fraction": 0.15,
        "exploration_final_eps": 0.03,
        "gamma": 0.995,
        "batch_size": 64,
        "train_freq": 4,
        "target_update_interval": 2000,
        "learning_starts": 5000,
    },
    {
        "run": 9,
        "learning_rate": 8e-4,
        "buffer_size": 30000,
        "exploration_fraction": 0.4,
        "exploration_final_eps": 0.05,
        "gamma": 0.93,
        "batch_size": 32,
        "train_freq": 8,
        "target_update_interval": 500,
        "learning_starts": 1000,
    },
    {
        "run": 10,
        "learning_rate": 5e-4,
        "buffer_size": 50000,
        "exploration_fraction": 0.2,
        "exploration_final_eps": 0.05,
        "gamma": 0.99,
        "batch_size": 64,
        "train_freq": 4,
        "target_update_interval": 1000,
        "learning_starts": 1000,
    },
]

TOTAL_TIMESTEPS = 50000


def train_single_run(params, save_dir):
    """Train a single DQN run with given hyperparameters."""
    run_id = params["run"]
    print(f"\n{'='*60}")
    print(f"DQN Run {run_id}/10")
    print(f"{'='*60}")
    print(f"Hyperparameters: {json.dumps({k: v for k, v in params.items() if k != 'run'}, indent=2)}")

    env = Monitor(AfricanLiteracyTutorEnv())
    callback = TrainingMetricsCallback()

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=params["learning_rate"],
        buffer_size=params["buffer_size"],
        exploration_fraction=params["exploration_fraction"],
        exploration_final_eps=params["exploration_final_eps"],
        gamma=params["gamma"],
        batch_size=params["batch_size"],
        train_freq=params["train_freq"],
        target_update_interval=params["target_update_interval"],
        learning_starts=params["learning_starts"],
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        seed=run_id * 42,
    )

    start_time = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    train_time = time.time() - start_time

    # Save model
    model_path = os.path.join(save_dir, f"dqn_run_{run_id}")
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

    # Find convergence step (first time rolling mean exceeds 50% of final performance)
    convergence_step = TOTAL_TIMESTEPS
    if len(callback.episode_rewards) > 10:
        rolling = np.convolve(callback.episode_rewards, np.ones(10) / 10, mode="valid")
        threshold = mean_reward * 0.5
        converged = np.where(rolling > threshold)[0]
        if len(converged) > 0:
            convergence_step = int(converged[0] * (TOTAL_TIMESTEPS / max(len(callback.episode_rewards), 1)))

    result = {
        "run": run_id,
        "mean_reward": round(float(mean_reward), 2),
        "std_reward": round(float(std_reward), 2),
        "convergence_step": convergence_step,
        "train_time_s": round(train_time, 1),
        "episode_rewards": callback.episode_rewards,
        "losses": callback.losses,
        **{k: v for k, v in params.items() if k != "run"},
    }

    print(f"\nRun {run_id} Results: Mean={mean_reward:.2f} +/- {std_reward:.2f}, "
          f"Convergence~{convergence_step}, Time={train_time:.1f}s")

    env.close()
    eval_env.close()
    return result


def plot_results(all_results, save_dir):
    """Generate training visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("DQN Training Analysis - African Literacy AI Tutor", fontsize=14)

    # 1. Cumulative reward curves
    ax = axes[0, 0]
    for res in all_results:
        if res["episode_rewards"]:
            cumulative = np.cumsum(res["episode_rewards"])
            ax.plot(cumulative, label=f"Run {res['run']} (LR={res['learning_rate']})", alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("DQN Cumulative Reward Curves")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # 2. DQN Objective (Loss) curves
    ax = axes[0, 1]
    for res in all_results:
        if res["losses"]:
            smoothed = np.convolve(res["losses"], np.ones(50) / 50, mode="valid") if len(res["losses"]) > 50 else res["losses"]
            ax.plot(smoothed, label=f"Run {res['run']}", alpha=0.7)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("DQN Objective (Loss) Curves")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # 3. Episode reward over time (convergence)
    ax = axes[1, 0]
    for res in all_results:
        if res["episode_rewards"] and len(res["episode_rewards"]) > 10:
            rolling = np.convolve(res["episode_rewards"], np.ones(10) / 10, mode="valid")
            ax.plot(rolling, label=f"Run {res['run']}", alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (10-ep rolling avg)")
    ax.set_title("DQN Convergence Plot")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # 4. Final performance comparison
    ax = axes[1, 1]
    runs = [r["run"] for r in all_results]
    means = [r["mean_reward"] for r in all_results]
    stds = [r["std_reward"] for r in all_results]
    ax.bar(runs, means, yerr=stds, capsize=4, color="steelblue", alpha=0.8)
    ax.set_xlabel("Run")
    ax.set_ylabel("Mean Evaluation Reward")
    ax.set_title("DQN Performance by Hyperparameter Config")
    ax.set_xticks(runs)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dqn_analysis.png"), dpi=150)
    plt.close()
    print(f"\nPlots saved to {os.path.join(save_dir, 'dqn_analysis.png')}")


def save_results_table(all_results, save_dir):
    """Save hyperparameter results as CSV table."""
    csv_path = os.path.join(save_dir, "dqn_results.csv")
    fieldnames = [
        "run", "learning_rate", "buffer_size", "exploration_fraction",
        "exploration_final_eps", "gamma", "batch_size", "train_freq",
        "target_update_interval", "learning_starts",
        "mean_reward", "std_reward", "convergence_step", "train_time_s"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in all_results:
            writer.writerow({k: res[k] for k in fieldnames})
    print(f"Results table saved to {csv_path}")


def main():
    save_dir = os.path.join(os.path.dirname(__file__), "..", "models", "dqn")
    os.makedirs(save_dir, exist_ok=True)

    all_results = []
    for params in HYPERPARAMS:
        result = train_single_run(params, save_dir)
        all_results.append(result)

    # Find best run
    best = max(all_results, key=lambda x: x["mean_reward"])
    print(f"\n{'='*60}")
    print(f"BEST DQN RUN: Run {best['run']} with Mean Reward = {best['mean_reward']:.2f}")
    print(f"{'='*60}")

    # Save best run indicator
    with open(os.path.join(save_dir, "best_run.json"), "w") as f:
        json.dump({"best_run": best["run"], "mean_reward": best["mean_reward"],
                    "params": {k: v for k, v in best.items()
                               if k not in ["episode_rewards", "losses"]}}, f, indent=2)

    plot_results(all_results, save_dir)
    save_results_table(all_results, save_dir)


if __name__ == "__main__":
    main()
