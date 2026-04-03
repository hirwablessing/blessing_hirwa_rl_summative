"""
Comprehensive Analysis Script - All 4 RL Methods Comparison.

Generates:
1. Cumulative reward curves (all methods in subplots)
2. DQN objective (loss) curves
3. PG entropy curves (REINFORCE, PPO, A2C)
4. Convergence plots
5. Generalization tests (varied start states)
"""

import os
import sys
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.custom_env import AfricanLiteracyTutorEnv


def load_csv_results(path):
    """Load results from a CSV file."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def evaluate_model(model, model_type, env, n_episodes=20):
    """Evaluate a model on a given environment."""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if model_type != "reinforce":
                action = int(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
    return rewards


def generalization_test(save_dir):
    """
    Test best models on modified start conditions to assess generalization.
    Variations: different initial languages, different skill levels, high fatigue starts.
    """
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    dqn_dir = os.path.join(base_dir, "models", "dqn")
    pg_dir = os.path.join(base_dir, "models", "pg")

    # Load best models
    models = {}

    dqn_best_path = os.path.join(dqn_dir, "best_run.json")
    if os.path.exists(dqn_best_path):
        with open(dqn_best_path) as f:
            info = json.load(f)
        from stable_baselines3 import DQN
        model_path = os.path.join(dqn_dir, f"dqn_run_{info['best_run']}")
        if os.path.exists(model_path + ".zip"):
            models["DQN"] = (DQN.load(model_path), "sb3")

    pg_best_path = os.path.join(pg_dir, "best_runs.json")
    if os.path.exists(pg_best_path):
        with open(pg_best_path) as f:
            pg_info = json.load(f)

        if "ppo" in pg_info:
            from stable_baselines3 import PPO
            path = os.path.join(pg_dir, f"ppo_run_{pg_info['ppo']['run']}")
            if os.path.exists(path + ".zip"):
                models["PPO"] = (PPO.load(path), "sb3")

        if "a2c" in pg_info:
            from stable_baselines3 import A2C
            path = os.path.join(pg_dir, f"a2c_run_{pg_info['a2c']['run']}")
            if os.path.exists(path + ".zip"):
                models["A2C"] = (A2C.load(path), "sb3")

        if "reinforce" in pg_info:
            sys.path.insert(0, os.path.join(base_dir, "training"))
            from pg_training import REINFORCEAgent, REINFORCE_PARAMS
            # Find the hidden_size for the best run
            best_run = pg_info["reinforce"]["run"]
            hidden_size = 128  # default
            for p in REINFORCE_PARAMS:
                if p["run"] == best_run:
                    hidden_size = p["hidden_size"]
                    break
            agent = REINFORCEAgent(hidden_size=hidden_size)
            path = os.path.join(pg_dir, f"reinforce_run_{best_run}")
            if os.path.exists(path + ".pt"):
                agent.load(path)
                models["REINFORCE"] = (agent, "reinforce")

    if not models:
        print("No trained models found for generalization test. Skipping.")
        return

    # Define test scenarios
    scenarios = {
        "Default": {},
        "Start Swahili": {"language": 1},
        "Start Yoruba": {"language": 2},
        "Start Amharic": {"language": 3},
        "High Initial Skill": {"high_skill": True},
        "High Fatigue Start": {"high_fatigue": True},
        "Low Engagement": {"low_engagement": True},
    }

    results = {name: {} for name in models}

    for scenario_name, scenario_opts in scenarios.items():
        print(f"\n  Testing scenario: {scenario_name}")
        for model_name, (model, mtype) in models.items():
            env = AfricanLiteracyTutorEnv()
            scenario_rewards = []

            for _ in range(15):
                obs, _ = env.reset()

                # Apply scenario modifications
                if scenario_opts.get("language"):
                    lang_idx = scenario_opts["language"]
                    env.current_language_idx = lang_idx
                    env.state[env.IDX_LANGUAGE] = env.LANGUAGE_VALUES[lang_idx]
                    env.state[env.IDX_L1_TRANSFER] = env.TRANSFER_MATRIX[(0, lang_idx)]
                if scenario_opts.get("high_skill"):
                    rng = np.random.default_rng()
                    for idx in env.SKILL_INDICES:
                        env.state[idx] = rng.uniform(0.3, 0.5)
                if scenario_opts.get("high_fatigue"):
                    env.state[env.IDX_FATIGUE] = 0.6
                if scenario_opts.get("low_engagement"):
                    env.state[env.IDX_ENGAGEMENT] = 0.3

                obs = env.state.copy()
                ep_reward = 0
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    if mtype != "reinforce":
                        action = int(action)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    ep_reward += reward
                    done = terminated or truncated
                scenario_rewards.append(ep_reward)

            results[model_name][scenario_name] = {
                "mean": round(float(np.mean(scenario_rewards)), 2),
                "std": round(float(np.std(scenario_rewards)), 2),
            }
            env.close()

    # Plot generalization results
    fig, ax = plt.subplots(figsize=(12, 6))
    scenario_names = list(scenarios.keys())
    x = np.arange(len(scenario_names))
    width = 0.2
    colors = {"DQN": "#3498db", "REINFORCE": "#e74c3c", "PPO": "#2ecc71", "A2C": "#f39c12"}

    for i, (model_name, scenario_results) in enumerate(results.items()):
        means = [scenario_results[s]["mean"] for s in scenario_names]
        stds = [scenario_results[s]["std"] for s in scenario_names]
        ax.bar(x + i * width, means, width, yerr=stds, label=model_name,
               color=colors.get(model_name, "gray"), alpha=0.8, capsize=3)

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Generalization Test: Agent Performance Across Different Start Conditions")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(scenario_names, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "generalization_test.png"), dpi=150)
    plt.close()
    print(f"\nGeneralization plot saved to {save_dir}/generalization_test.png")

    # Save results as CSV
    csv_path = os.path.join(save_dir, "generalization_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Scenario"] + [f"{m}_mean" for m in results] + [f"{m}_std" for m in results]
        writer.writerow(header)
        for s in scenario_names:
            row = [s]
            row += [results[m][s]["mean"] for m in results]
            row += [results[m][s]["std"] for m in results]
            writer.writerow(row)
    print(f"Generalization CSV saved to {csv_path}")

    return results


def generate_combined_plots(save_dir):
    """Generate the combined comparison plots across all 4 methods."""
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    dqn_dir = os.path.join(base_dir, "models", "dqn")
    pg_dir = os.path.join(base_dir, "models", "pg")

    # Load all CSV results
    dqn_results = load_csv_results(os.path.join(dqn_dir, "dqn_results.csv"))
    reinforce_results = load_csv_results(os.path.join(pg_dir, "reinforce_results.csv"))
    ppo_results = load_csv_results(os.path.join(pg_dir, "ppo_results.csv"))
    a2c_results = load_csv_results(os.path.join(pg_dir, "a2c_results.csv"))

    if not any((dqn_results, reinforce_results, ppo_results, a2c_results)):
        print("No results found. Run training first.")
        return

    # --- Combined Performance Bar Chart ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("All Methods - Hyperparameter Tuning Results (10 Runs Each)", fontsize=14)

    all_data = [
        ("DQN", dqn_results, "#3498db"),
        ("REINFORCE", reinforce_results, "#e74c3c"),
        ("PPO", ppo_results, "#2ecc71"),
        ("A2C", a2c_results, "#f39c12"),
    ]

    for idx, (name, results, color) in enumerate(all_data):
        ax = axes[idx // 2, idx % 2]
        if results:
            runs = [int(r["run"]) for r in results]
            means = [float(r["mean_reward"]) for r in results]
            stds = [float(r["std_reward"]) for r in results]
            ax.bar(runs, means, yerr=stds, capsize=3, color=color, alpha=0.8)
            ax.set_xlabel("Run")
            ax.set_ylabel("Mean Reward")
            # Highlight best
            best_idx = np.argmax(means)
            ax.bar(runs[best_idx], means[best_idx], color="gold", alpha=0.9,
                   edgecolor="black", linewidth=2)
        ax.set_title(f"{name} - Performance by Config")
        ax.set_xticks(range(1, 11))
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "all_methods_comparison.png"), dpi=150)
    plt.close()

    # --- Cross-method best comparison ---
    fig, ax = plt.subplots(figsize=(8, 5))
    best_data = []
    for name, results, color in all_data:
        if results:
            means = [float(r["mean_reward"]) for r in results]
            stds = [float(r["std_reward"]) for r in results]
            best_idx = np.argmax(means)
            best_data.append((name, means[best_idx], stds[best_idx], color))

    if best_data:
        best_names, best_means, best_stds, best_colors = zip(*best_data)
        ax.bar(best_names, best_means, yerr=best_stds, capsize=5, color=best_colors, alpha=0.85,
               edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Mean Evaluation Reward")
        ax.set_title("Best Run: Value-Based (DQN) vs Policy Gradient Methods")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "best_runs_comparison.png"), dpi=150)
    plt.close()

    print(f"Combined plots saved to {save_dir}")


def main():
    save_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("COMPREHENSIVE RL ANALYSIS")
    print("=" * 60)

    print("\n1. Generating combined comparison plots...")
    generate_combined_plots(save_dir)

    print("\n2. Running generalization tests...")
    generalization_test(save_dir)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
