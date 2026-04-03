"""
Generate a diagram of the RL agent in the simulated environment
with proper descriptions of all components.

Produces: environment_diagram.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def create_environment_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#f8f9fa")

    # Title
    ax.text(8, 9.6, "African Literacy AI Tutor - RL Environment Architecture",
            fontsize=16, fontweight="bold", ha="center", va="center",
            color="#2c3e50")

    # ---- AGENT BOX ----
    agent_box = FancyBboxPatch((0.5, 5.5), 4.5, 3.2, boxstyle="round,pad=0.2",
                                facecolor="#3498db", edgecolor="#2980b9", linewidth=2, alpha=0.9)
    ax.add_patch(agent_box)
    ax.text(2.75, 8.2, "RL AGENT (AI Tutor)", fontsize=13, fontweight="bold",
            ha="center", color="white")
    ax.text(2.75, 7.6, "Policy Network: MLP [256, 256]", fontsize=9, ha="center", color="#ecf0f1")
    ax.text(2.75, 7.2, "Algorithms: DQN, REINFORCE,", fontsize=9, ha="center", color="#ecf0f1")
    ax.text(2.75, 6.85, "PPO, A2C", fontsize=9, ha="center", color="#ecf0f1")
    ax.text(2.75, 6.3, "Decides: what to teach,", fontsize=9, ha="center", color="#d5f5e3")
    ax.text(2.75, 5.95, "when to rest, difficulty level,", fontsize=9, ha="center", color="#d5f5e3")
    ax.text(2.75, 5.6, "language switching", fontsize=9, ha="center", color="#d5f5e3")

    # ---- ENVIRONMENT BOX ----
    env_box = FancyBboxPatch((6.5, 3.0), 9.0, 5.8, boxstyle="round,pad=0.2",
                              facecolor="#ecf0f1", edgecolor="#95a5a6", linewidth=2)
    ax.add_patch(env_box)
    ax.text(11.0, 8.3, "ENVIRONMENT (Simulated Student)", fontsize=13,
            fontweight="bold", ha="center", color="#2c3e50")

    # Student state sub-box
    student_box = FancyBboxPatch((7.0, 5.8), 4.0, 2.2, boxstyle="round,pad=0.15",
                                  facecolor="#e8f8f5", edgecolor="#1abc9c", linewidth=1.5)
    ax.add_patch(student_box)
    ax.text(9.0, 7.6, "Student State (23 dims)", fontsize=10, fontweight="bold",
            ha="center", color="#1abc9c")
    ax.text(9.0, 7.2, "Mastery: phoneme, letter, syllable,", fontsize=8, ha="center", color="#2c3e50")
    ax.text(9.0, 6.9, "word, sentence, vocabulary", fontsize=8, ha="center", color="#2c3e50")
    ax.text(9.0, 6.5, "Affect: engagement, fatigue,", fontsize=8, ha="center", color="#2c3e50")
    ax.text(9.0, 6.2, "frustration, confidence", fontsize=8, ha="center", color="#2c3e50")

    # Dynamics sub-box
    dynamics_box = FancyBboxPatch((11.5, 5.8), 3.5, 2.2, boxstyle="round,pad=0.15",
                                   facecolor="#fdf2e9", edgecolor="#e67e22", linewidth=1.5)
    ax.add_patch(dynamics_box)
    ax.text(13.25, 7.6, "Learning Dynamics", fontsize=10, fontweight="bold",
            ha="center", color="#e67e22")
    ax.text(13.25, 7.15, "Prerequisite gating", fontsize=8, ha="center", color="#2c3e50")
    ax.text(13.25, 6.8, "Forgetting curve", fontsize=8, ha="center", color="#2c3e50")
    ax.text(13.25, 6.45, "Language transfer", fontsize=8, ha="center", color="#2c3e50")
    ax.text(13.25, 6.1, "Spaced repetition", fontsize=8, ha="center", color="#2c3e50")

    # Languages sub-box
    lang_box = FancyBboxPatch((7.0, 3.5), 4.0, 1.8, boxstyle="round,pad=0.15",
                               facecolor="#ebf5fb", edgecolor="#3498db", linewidth=1.5)
    ax.add_patch(lang_box)
    ax.text(9.0, 4.9, "4 African Languages", fontsize=10, fontweight="bold",
            ha="center", color="#3498db")
    ax.text(9.0, 4.5, "Kinyarwanda | Swahili", fontsize=9, ha="center", color="#2c3e50")
    ax.text(9.0, 4.15, "Yoruba | Amharic", fontsize=9, ha="center", color="#2c3e50")
    ax.text(9.0, 3.75, "(Cross-language transfer effects)", fontsize=8, ha="center",
            color="#7f8c8d", style="italic")

    # Reward sub-box
    reward_box = FancyBboxPatch((11.5, 3.5), 3.5, 1.8, boxstyle="round,pad=0.15",
                                 facecolor="#fdedec", edgecolor="#e74c3c", linewidth=1.5)
    ax.add_patch(reward_box)
    ax.text(13.25, 4.9, "Reward Signal", fontsize=10, fontweight="bold",
            ha="center", color="#e74c3c")
    ax.text(13.25, 4.5, "+ Mastery progress (10x)", fontsize=8, ha="center", color="#27ae60")
    ax.text(13.25, 4.2, "+ Engagement, streaks", fontsize=8, ha="center", color="#27ae60")
    ax.text(13.25, 3.9, "- Frustration, fatigue", fontsize=8, ha="center", color="#c0392b")
    ax.text(13.25, 3.6, "- Difficulty mismatch", fontsize=8, ha="center", color="#c0392b")

    # ---- ARROWS ----
    # Action arrow (Agent -> Environment)
    ax.annotate("", xy=(6.5, 7.8), xytext=(5.0, 7.8),
                arrowprops=dict(arrowstyle="-|>", color="#e74c3c", lw=2.5))
    ax.text(5.75, 8.15, "Action", fontsize=10, fontweight="bold", ha="center", color="#e74c3c")
    ax.text(5.75, 7.45, "15 discrete choices", fontsize=8, ha="center", color="#7f8c8d")

    # Observation arrow (Environment -> Agent)
    ax.annotate("", xy=(5.0, 6.5), xytext=(6.5, 6.5),
                arrowprops=dict(arrowstyle="-|>", color="#27ae60", lw=2.5))
    ax.text(5.75, 6.8, "Observation", fontsize=10, fontweight="bold", ha="center", color="#27ae60")
    ax.text(5.75, 6.15, "23-dim state vector", fontsize=8, ha="center", color="#7f8c8d")

    # Reward arrow (Environment -> Agent, curved)
    ax.annotate("", xy=(2.75, 5.5), xytext=(11.0, 3.0),
                arrowprops=dict(arrowstyle="-|>", color="#f39c12", lw=2.5,
                                connectionstyle="arc3,rad=0.3"))
    ax.text(5.5, 3.3, "Reward", fontsize=10, fontweight="bold", ha="center", color="#f39c12")

    # ---- ACTION SPACE BOX (bottom) ----
    action_box = FancyBboxPatch((0.5, 0.3), 15.0, 2.3, boxstyle="round,pad=0.2",
                                 facecolor="#f5f5f5", edgecolor="#bdc3c7", linewidth=1.5)
    ax.add_patch(action_box)
    ax.text(8.0, 2.25, "Action Space (15 Actions)", fontsize=11, fontweight="bold",
            ha="center", color="#2c3e50")

    actions_left = [
        "0: PHONEME_DRILL", "1: LETTER_TRACING", "2: SYLLABLE_BLENDING",
        "3: WORD_BUILDING", "4: READ_ALOUD_WORD",
    ]
    actions_mid = [
        "5: SENTENCE_READING", "6: VOCABULARY_INTRO", "7: VOCABULARY_REVIEW",
        "8: STORY_LISTENING", "9: CALL_AND_RESPONSE",
    ]
    actions_right = [
        "10: SWITCH_LANGUAGE", "11: INCREASE_DIFFICULTY",
        "12: DECREASE_DIFFICULTY", "13: GIVE_ENCOURAGEMENT",
        "14: TAKE_BREAK",
    ]

    for i, a in enumerate(actions_left):
        ax.text(1.0, 1.85 - i * 0.3, a, fontsize=7.5, ha="left", color="#2c3e50", family="monospace")
    for i, a in enumerate(actions_mid):
        ax.text(5.8, 1.85 - i * 0.3, a, fontsize=7.5, ha="left", color="#2c3e50", family="monospace")
    for i, a in enumerate(actions_right):
        ax.text(10.8, 1.85 - i * 0.3, a, fontsize=7.5, ha="left", color="#2c3e50", family="monospace")

    # ---- TERMINAL CONDITIONS (side note) ----
    ax.text(0.5, 4.8, "Terminal Conditions:", fontsize=9, fontweight="bold", color="#8e44ad")
    ax.text(0.5, 4.45, "• Mastery > 85% (success)", fontsize=8, color="#27ae60")
    ax.text(0.5, 4.15, "• Engagement < 10% (fail)", fontsize=8, color="#c0392b")
    ax.text(0.5, 3.85, "• Frustration > 95% (fail)", fontsize=8, color="#c0392b")
    ax.text(0.5, 3.55, "• Exhaustion (fail)", fontsize=8, color="#c0392b")
    ax.text(0.5, 3.25, "• 200 steps timeout", fontsize=8, color="#7f8c8d")

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), "environment_diagram.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="#f8f9fa")
    plt.close()
    print(f"Environment diagram saved to {save_path}")


if __name__ == "__main__":
    create_environment_diagram()
