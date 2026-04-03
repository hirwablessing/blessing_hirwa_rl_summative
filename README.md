# African Literacy AI Tutor - RL Summative

**Repository:** `blessing_hirwa_rl_summative`

## Mission
Tackling linguistic and literacy barriers in African education by developing voice-first, localized AI tutoring systems using reinforcement learning.

## Environment: AfricanLiteracyTutor-v0

A custom Gymnasium environment where an RL agent acts as an AI tutor, making pedagogical decisions to teach a simulated student literacy skills across four African languages: **Kinyarwanda**, **Swahili**, **Yoruba**, and **Amharic**.

### Observation Space (23 dimensions)
- **Mastery Skills** (6): phoneme, letter recognition, syllable formation, word reading, sentence comprehension, vocabulary
- **Affective States** (4): engagement, fatigue, frustration, confidence
- **Performance Metrics** (3): recent accuracy, response latency, streak
- **Error Patterns** (3): phoneme errors, tone errors, grammar errors
- **Session Variables** (7): progress, language, L1 transfer, spaced repetition, difficulty, review timing, lesson type

### Action Space (15 discrete actions)
| ID | Action | Description |
|----|--------|-------------|
| 0 | PHONEME_DRILL | Practice sound recognition |
| 1 | LETTER_TRACING | Visual letter tracing |
| 2 | SYLLABLE_BLENDING | Combine phonemes into syllables |
| 3 | WORD_BUILDING | Construct words from syllables |
| 4 | READ_ALOUD_WORD | Voice-based word reading |
| 5 | SENTENCE_READING | Sentence comprehension |
| 6 | VOCABULARY_INTRO | Introduce new vocabulary |
| 7 | VOCABULARY_REVIEW | Spaced repetition review |
| 8 | STORY_LISTENING | Listen to a story |
| 9 | CALL_AND_RESPONSE | Interactive engagement game |
| 10 | SWITCH_LANGUAGE | Switch target language |
| 11 | INCREASE_DIFFICULTY | Raise difficulty level |
| 12 | DECREASE_DIFFICULTY | Lower difficulty level |
| 13 | GIVE_ENCOURAGEMENT | Verbal praise |
| 14 | TAKE_BREAK | Rest period |

### Reward Structure
- Primary: mastery improvement (+10x weight)
- Engagement maintenance, spaced repetition timing, streak bonuses
- Penalties: frustration, fatigue ignoring, difficulty mismatch, prerequisite violations

### Environment Dynamics
- **Prerequisite gating**: Can't attempt advanced skills without foundational mastery
- **Forgetting curve**: Skills decay without review
- **Language transfer**: Related languages (Kinyarwanda/Swahili) share positive transfer
- **Affective modeling**: Fatigue, frustration, engagement affect learning rate

## Setup

```bash
pip install -r requirements.txt
```

## Running

### Random Agent Demo (No Training)
```bash
python play_random.py
```

### Train All Models
```bash
python training/dqn_training.py
python training/pg_training.py
```

### Run Best Agent with Visualization
```bash
python main.py
```

### API Demo
```bash
uvicorn api_example:app --reload
```

## Project Structure
```
├── environment/
│   ├── custom_env.py      # Gymnasium environment
│   └── rendering.py       # PyOpenGL + Pygame visualization
├── training/
│   ├── dqn_training.py    # DQN (Stable Baselines 3)
│   └── pg_training.py     # REINFORCE (PyTorch) + PPO + A2C (SB3)
├── models/
│   ├── dqn/               # Saved DQN models & results
│   └── pg/                # Saved PG models & results
├── main.py                # Entry point - best model + visualization
├── play_random.py         # Random agent demo
├── api_example.py         # FastAPI JSON API demo
├── requirements.txt       # Dependencies
└── README.md
```

## RL Algorithms
1. **DQN** (Deep Q-Network) - Value-based, via Stable Baselines 3
2. **REINFORCE** - Policy gradient, manual PyTorch implementation
3. **PPO** (Proximal Policy Optimization) - Policy gradient, via SB3
4. **A2C** (Advantage Actor-Critic) - Actor-critic, via SB3

Each algorithm trained with 10 hyperparameter configurations for comparison.

## Author
Blessing Hirwa - African Leadership University
