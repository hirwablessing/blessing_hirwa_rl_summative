"""
Static demonstration of the African Literacy AI Tutor environment
with an agent taking RANDOM actions (no trained model).

This demonstrates the visualization components and environment dynamics
without any RL training involved.
"""

import json
import sys

sys.path.insert(0, ".")
from environment.custom_env import AfricanLiteracyTutorEnv
from environment.rendering import TutorRenderer


def main():
    env = AfricanLiteracyTutorEnv(render_mode="human")
    renderer = TutorRenderer(width=1000, height=700)

    print("=" * 60)
    print("African Literacy AI Tutor - Random Agent Demo")
    print("=" * 60)
    print("\nEnvironment: AfricanLiteracyTutor-v0")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space} ({len(env.ACTION_NAMES)} actions)")
    print("\nActions available:")
    for i, name in enumerate(env.ACTION_NAMES):
        print(f"  {i:2d}: {name}")
    print("\nRunning random agent... (Press ESC or close window to exit)\n")

    obs, info = env.reset()
    total_reward = 0
    step = 0

    while True:
        # Random action selection
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        # Print step info
        print(
            f"Step {step:3d} | Action: {env.ACTION_NAMES[action]:20s} | "
            f"Success: {'Y' if info['success'] else 'N'} | "
            f"Reward: {reward:+6.2f} | "
            f"Mastery: {info['avg_mastery']:.1%} | "
            f"Engagement: {info['engagement']:.1%} | "
            f"Lang: {info['language']}"
        )

        # Render
        if not renderer.render(obs, info):
            break

        if terminated or truncated:
            if info["avg_mastery"] > 0.85:
                reason = "MASTERY ACHIEVED"
            elif obs[6] < 0.1:
                reason = "DISENGAGED"
            elif obs[8] > 0.95:
                reason = "FRUSTRATED"
            elif obs[7] > 0.95:
                reason = "EXHAUSTED"
            else:
                reason = "SESSION ENDED"
            print(f"\n--- Episode ended: {reason} ---")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Final Mastery: {info['avg_mastery']:.1%}")
            print(f"Steps: {step}")

            # Show JSON serialization
            print("\n--- JSON State (API-ready) ---")
            print(json.dumps(env.to_json(), indent=2))

            # Reset for next episode
            obs, info = env.reset()
            total_reward = 0
            step = 0
            print("\n--- New Episode ---\n")

    renderer.close()
    print("\nDemo complete.")


if __name__ == "__main__":
    main()
