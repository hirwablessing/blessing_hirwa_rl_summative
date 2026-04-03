"""
Main entry point for running the best-performing RL agent
in the African Literacy AI Tutor environment with visualization.

Usage: python main.py
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from environment.custom_env import AfricanLiteracyTutorEnv
from environment.rendering import TutorRenderer


def _find_best_algorithm():
    """Scan saved results and return (algo_name, info_dict, dqn_dir, pg_dir)."""
    base_dir = os.path.dirname(__file__)
    dqn_dir = os.path.join(base_dir, "models", "dqn")
    pg_dir = os.path.join(base_dir, "models", "pg")

    best_algo = None
    best_reward = -float("inf")
    best_info = {}

    # Check DQN best
    dqn_best_path = os.path.join(dqn_dir, "best_run.json")
    if os.path.exists(dqn_best_path):
        with open(dqn_best_path) as f:
            dqn_best = json.load(f)
        if dqn_best["mean_reward"] > best_reward:
            best_reward = dqn_best["mean_reward"]
            best_algo = "dqn"
            best_info = {"run": dqn_best["best_run"], "mean_reward": dqn_best["mean_reward"]}

    # Check PG best
    pg_best_path = os.path.join(pg_dir, "best_runs.json")
    if os.path.exists(pg_best_path):
        with open(pg_best_path) as f:
            pg_best = json.load(f)
        for algo in ["reinforce", "ppo", "a2c"]:
            if algo in pg_best and pg_best[algo]["mean_reward"] > best_reward:
                best_reward = pg_best[algo]["mean_reward"]
                best_algo = algo
                best_info = pg_best[algo]

    return best_algo, best_reward, best_info, dqn_dir, pg_dir


def _load_sb3_model(algo_name, model_dir, run_id):
    """Load a stable-baselines3 model by algorithm name."""
    from stable_baselines3 import DQN, PPO, A2C

    sb3_classes = {"dqn": DQN, "ppo": PPO, "a2c": A2C}
    cls = sb3_classes[algo_name]
    model_path = os.path.join(model_dir, f"{algo_name}_run_{run_id}")
    return cls.load(model_path), algo_name, "sb3"


def load_best_model():
    """Load the best performing model across all algorithms."""
    best_algo, best_reward, best_info, dqn_dir, pg_dir = _find_best_algorithm()

    if best_algo is None:
        print("No trained models found. Please run training first:")
        print("  python training/dqn_training.py")
        print("  python training/pg_training.py")
        sys.exit(1)

    print(f"Best algorithm: {best_algo.upper()} (Run {best_info['run']}, "
          f"Mean Reward: {best_reward:.2f})")

    # Load the model
    if best_algo in ("dqn",):
        return _load_sb3_model(best_algo, dqn_dir, best_info["run"])

    if best_algo in ("ppo", "a2c"):
        return _load_sb3_model(best_algo, pg_dir, best_info["run"])

    if best_algo == "reinforce":
        from training.pg_training import REINFORCEAgent, REINFORCE_PARAMS
        hidden_size = 128
        for p in REINFORCE_PARAMS:
            if p["run"] == best_info["run"]:
                hidden_size = p["hidden_size"]
                break
        agent = REINFORCEAgent(hidden_size=hidden_size)
        model_path = os.path.join(pg_dir, f"reinforce_run_{best_info['run']}")
        agent.load(model_path)
        return agent, best_algo, "reinforce"

    print(f"Unknown algorithm: {best_algo}")
    sys.exit(1)


def _print_banner(algo_name):
    """Print the simulation banner with environment description."""
    print("\n" + "=" * 70)
    print(f"RUNNING BEST AGENT: {algo_name.upper()}")
    print("=" * 70)
    print("\nProblem: Tackling linguistic and literacy barriers in African education")
    print("Agent: AI Tutor that selects optimal pedagogical actions")
    print("Objective: Maximize student literacy mastery while maintaining engagement")
    print("\nReward Structure:")
    print("  + Mastery progress (primary, 10x weight)")
    print("  + High engagement maintenance")
    print("  + Spaced repetition timing bonus")
    print("  + Success streak bonus")
    print("  - Frustration penalty")
    print("  - Fatigue ignoring penalty")
    print("  - Difficulty mismatch penalty")
    print("  - Prerequisite violation penalty")
    print("\nAgent Behavior: The tutor learns to:")
    print("  1. Start with foundational skills (phonemes, letters)")
    print("  2. Gradually increase difficulty as mastery grows")
    print("  3. Use encouragement and breaks to manage affect")
    print("  4. Review vocabulary when spaced repetition is due")
    print("  5. Avoid actions the student isn't ready for")
    print("=" * 70)
    print("\nPress ESC or close window to exit.\n")


def _get_termination_reason(info, obs):
    """Determine why the episode ended."""
    if info["avg_mastery"] > 0.85:
        return "MASTERY ACHIEVED!"
    if obs[6] < 0.1:
        return "Student disengaged"
    if obs[8] > 0.95:
        return "Student too frustrated"
    if obs[7] > 0.95:
        return "Student exhausted"
    return "Session ended (max steps)"


def _print_episode_summary(reason, total_reward, info, step, action_counts, env):
    """Print the summary for a completed episode."""
    print(f"\n  Episode Result: {reason}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Final Mastery: {info['avg_mastery']:.1%}")
    print(f"  Steps: {step}")
    print("  Action Distribution:")
    for action_name, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"    {action_name}: {count} ({count/step:.0%})")
    print("\n  JSON State:")
    print(f"  {json.dumps(env.to_json(), indent=4)}")


def run_simulation():
    """Run the best agent with full visualization and verbose output."""
    model, algo_name, _ = load_best_model()
    env = AfricanLiteracyTutorEnv(render_mode="human")
    renderer = TutorRenderer(width=1000, height=700)

    _print_banner(algo_name)

    n_episodes = 5
    all_rewards = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        step = 0
        action_counts = {}

        print(f"\n--- Episode {ep + 1}/{n_episodes} ---")

        while True:
            raw_action, _ = model.predict(obs, deterministic=True)
            action: int = int(raw_action)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            action_name = env.ACTION_NAMES[action]
            action_counts[action_name] = action_counts.get(action_name, 0) + 1

            print(
                f"  Step {step:3d} | {action_name:22s} | "
                f"{'OK' if info['success'] else 'FAIL':4s} | "
                f"R={reward:+6.2f} | "
                f"Mastery={info['avg_mastery']:.1%} | "
                f"Eng={info['engagement']:.1%} | "
                f"Fat={info['fatigue']:.1%} | "
                f"Fru={info['frustration']:.1%} | "
                f"{info['language']}"
            )

            if not renderer.render(obs, info):
                renderer.close()
                return

            if terminated or truncated:
                reason = _get_termination_reason(info, obs)
                all_rewards.append(total_reward)
                _print_episode_summary(reason, total_reward, info, step, action_counts, env)
                break

    print(f"\n{'='*70}")
    print(f"SIMULATION SUMMARY ({n_episodes} episodes)")
    print(f"  Mean Reward: {np.mean(all_rewards):.2f} +/- {np.std(all_rewards):.2f}")
    print(f"{'='*70}")

    renderer.close()


if __name__ == "__main__":
    run_simulation()
