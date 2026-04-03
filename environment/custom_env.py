"""
African Literacy AI Tutor Environment

A custom Gymnasium environment simulating an AI tutoring system that teaches
literacy skills across African languages (Kinyarwanda, Swahili, Yoruba, Amharic).

The RL agent acts as the tutor, choosing pedagogical actions to maximize
student learning outcomes while managing engagement, fatigue, and frustration.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class AfricanLiteracyTutorEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    # Action definitions
    ACTION_NAMES = [
        "PHONEME_DRILL",        # 0: Practice individual sound recognition
        "LETTER_TRACING",       # 1: Visual letter/character tracing
        "SYLLABLE_BLENDING",    # 2: Combine phonemes into syllables
        "WORD_BUILDING",        # 3: Construct words from syllables
        "READ_ALOUD_WORD",      # 4: Student reads a word aloud (voice)
        "SENTENCE_READING",     # 5: Read and comprehend a sentence
        "VOCABULARY_INTRO",     # 6: Introduce new vocabulary with audio+picture
        "VOCABULARY_REVIEW",    # 7: Review vocabulary (spaced repetition)
        "STORY_LISTENING",      # 8: Listen to a short story
        "CALL_AND_RESPONSE",    # 9: Interactive call-and-response game
        "SWITCH_LANGUAGE",      # 10: Switch to next target language
        "INCREASE_DIFFICULTY",  # 11: Move to harder material
        "DECREASE_DIFFICULTY",  # 12: Move to easier material (scaffold)
        "GIVE_ENCOURAGEMENT",   # 13: Verbal encouragement / praise
        "TAKE_BREAK",           # 14: Give the student a short rest
    ]

    # Observation indices
    IDX_PHONEME = 0
    IDX_LETTER = 1
    IDX_SYLLABLE = 2
    IDX_WORD = 3
    IDX_SENTENCE = 4
    IDX_VOCAB = 5
    IDX_ENGAGEMENT = 6
    IDX_FATIGUE = 7
    IDX_FRUSTRATION = 8
    IDX_CONFIDENCE = 9
    IDX_ACCURACY = 10
    IDX_LATENCY = 11
    IDX_PROGRESS = 12
    IDX_LANGUAGE = 13
    IDX_L1_TRANSFER = 14
    IDX_SPACED_REP = 15
    IDX_DIFFICULTY = 16
    IDX_STREAK = 17
    IDX_TIME_SINCE_REVIEW = 18
    IDX_ERR_PHONEME = 19
    IDX_ERR_TONE = 20
    IDX_ERR_GRAMMAR = 21
    IDX_LESSON_TYPE = 22

    SKILL_INDICES = [0, 1, 2, 3, 4, 5]

    # Language definitions
    LANGUAGES = ["Kinyarwanda", "Swahili", "Yoruba", "Amharic"]
    LANGUAGE_VALUES = [0.0, 0.33, 0.66, 1.0]

    # Language transfer matrix (from -> to)
    # Kinyarwanda and Swahili are both Bantu => high transfer
    TRANSFER_MATRIX = {
        (0, 0): 1.0, (0, 1): 0.7, (0, 2): 0.3, (0, 3): 0.1,
        (1, 0): 0.7, (1, 1): 1.0, (1, 2): 0.3, (1, 3): 0.1,
        (2, 0): 0.3, (2, 1): 0.3, (2, 2): 1.0, (2, 3): 0.15,
        (3, 0): 0.1, (3, 1): 0.1, (3, 2): 0.15, (3, 3): 1.0,
    }

    # Script characters for each language (used in visualization)
    LANGUAGE_SCRIPTS = {
        0: list("abcdefghijkmnoprstuvwyz"),  # Kinyarwanda (Latin)
        1: list("abcdefghijklmnoprstuvwyz"),  # Swahili (Latin)
        2: list("abcdefghijklmnoprstuvwyẹọṣ"),  # Yoruba (Latin+dots)
        3: list("አበገደሀወዘሐጠየከለመነሰዐፈ"),  # Amharic (Ge'ez script)
    }

    # Prerequisite mapping: action -> (required_skill_index, min_threshold)
    PREREQUISITES = {
        2: (IDX_PHONEME, 0.3),    # syllable blending needs phoneme mastery
        3: (IDX_SYLLABLE, 0.3),   # word building needs syllable mastery
        4: (IDX_WORD, 0.25),      # read aloud needs word reading
        5: (IDX_WORD, 0.4),       # sentence reading needs word reading
        8: (IDX_VOCAB, 0.2),      # story listening needs some vocabulary
    }

    # Action -> primary skill mapping
    ACTION_SKILL_MAP = {
        0: IDX_PHONEME, 1: IDX_LETTER, 2: IDX_SYLLABLE,
        3: IDX_WORD, 4: IDX_WORD, 5: IDX_SENTENCE,
        6: IDX_VOCAB, 7: IDX_VOCAB, 8: IDX_SENTENCE,
    }

    def __init__(self, render_mode=None, max_steps=200, base_lr=0.06):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.base_lr = base_lr

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(23,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(15)

        self.state = np.zeros(23, dtype=np.float32)
        self.current_step = 0
        self.current_language_idx = 0
        self.cumulative_reward = 0.0
        self.last_action = -1
        self.last_action_success = False
        self.last_reward = 0.0
        self.episode_actions = []
        self.renderer = None

    def reset(self, seed=None, options=None):  # noqa: ARG002
        super().reset(seed=seed, options=options)
        self.state = np.zeros(23, dtype=np.float32)

        # Randomized start: student has slight baseline abilities
        self.state[self.IDX_PHONEME] = self.np_random.uniform(0.05, 0.15)
        self.state[self.IDX_LETTER] = self.np_random.uniform(0.0, 0.10)
        self.state[self.IDX_SYLLABLE] = self.np_random.uniform(0.0, 0.05)
        self.state[self.IDX_WORD] = 0.0
        self.state[self.IDX_SENTENCE] = 0.0
        self.state[self.IDX_VOCAB] = self.np_random.uniform(0.02, 0.08)

        # Affective state
        self.state[self.IDX_ENGAGEMENT] = self.np_random.uniform(0.6, 0.9)
        self.state[self.IDX_FATIGUE] = 0.0
        self.state[self.IDX_FRUSTRATION] = self.np_random.uniform(0.0, 0.1)
        self.state[self.IDX_CONFIDENCE] = self.np_random.uniform(0.3, 0.6)

        # Performance
        self.state[self.IDX_ACCURACY] = 0.5
        self.state[self.IDX_LATENCY] = 0.5
        self.state[self.IDX_STREAK] = 0.0

        # Session
        self.state[self.IDX_PROGRESS] = 0.0
        self.current_language_idx = 0
        self.state[self.IDX_LANGUAGE] = 0.0
        self.state[self.IDX_L1_TRANSFER] = 1.0  # starting in native language
        self.state[self.IDX_SPACED_REP] = 0.0
        self.state[self.IDX_DIFFICULTY] = 0.1
        self.state[self.IDX_TIME_SINCE_REVIEW] = 0.0

        # Error patterns
        self.state[self.IDX_ERR_PHONEME] = 0.0
        self.state[self.IDX_ERR_TONE] = 0.0
        self.state[self.IDX_ERR_GRAMMAR] = 0.0

        # Lesson type
        self.state[self.IDX_LESSON_TYPE] = 0.0  # starts with phonics

        self.current_step = 0
        self.cumulative_reward = 0.0
        self.last_action = -1
        self.last_action_success = False
        self.last_reward = 0.0
        self.episode_actions = []

        # Store per-language mastery for transfer effects
        self.language_mastery: dict[int, np.ndarray] = {
            i: np.zeros(6, dtype=np.float32) for i in range(4)
        }
        self.language_mastery[0][:] = self.state[self.SKILL_INDICES]

        return self.state.copy(), {}

    def step(self, action):
        action = int(action)
        old_state = self.state.copy()
        self.current_step += 1
        self.last_action = action
        self.episode_actions.append(action)

        # Update session progress
        self.state[self.IDX_PROGRESS] = self.current_step / self.max_steps

        # Apply action effects
        success = self._apply_action(action)
        self.last_action_success = success

        # Apply passive dynamics (fatigue increase, forgetting curve, etc.)
        self._apply_passive_dynamics()

        # Compute reward
        reward = self._compute_reward(action, old_state, self.state, success)
        self.last_reward = reward
        self.cumulative_reward += reward

        # Check termination
        terminated, truncated = self._check_done()

        # Bonus reward for successful mastery completion
        if terminated and np.mean(self.state[self.SKILL_INDICES]) > 0.85:
            reward += 20.0
            self.last_reward = reward

        # Clip state to valid range
        self.state = np.clip(self.state, 0.0, 1.0)

        info = {
            "action_name": self.ACTION_NAMES[action],
            "success": success,
            "avg_mastery": float(np.mean(self.state[self.SKILL_INDICES])),
            "engagement": float(self.state[self.IDX_ENGAGEMENT]),
            "fatigue": float(self.state[self.IDX_FATIGUE]),
            "frustration": float(self.state[self.IDX_FRUSTRATION]),
            "language": self.LANGUAGES[self.current_language_idx],
            "cumulative_reward": float(self.cumulative_reward),
            "step": self.current_step,
        }

        return self.state.copy(), reward, terminated, truncated, info

    def _apply_action(self, action):
        """Apply the chosen action's effects on the student state."""
        effective_lr = self._get_effective_learning_rate()
        success = True

        # Check prerequisites
        if action in self.PREREQUISITES:
            req_skill, req_threshold = self.PREREQUISITES[action]
            if self.state[req_skill] < req_threshold:
                # Prerequisite not met: minimal learning, increased frustration
                self.state[self.IDX_FRUSTRATION] += 0.1
                self.state[self.IDX_ENGAGEMENT] -= 0.05
                self.state[self.IDX_CONFIDENCE] -= 0.05
                self.last_action_success = False
                return False

        # Determine interaction success probability
        if action in self.ACTION_SKILL_MAP:
            relevant_skill = self.state[self.ACTION_SKILL_MAP[action]]
            difficulty = self.state[self.IDX_DIFFICULTY]
            p_success = self._sigmoid(5 * (relevant_skill - difficulty))
            p_success = np.clip(p_success + self.np_random.normal(0, 0.05), 0.05, 0.95)
            success = self.np_random.random() < p_success

        # Apply learning effects based on action type
        if action == 0:  # PHONEME_DRILL
            gain = effective_lr * 0.08
            self.state[self.IDX_PHONEME] += gain if success else gain * 0.2
            self.state[self.IDX_LESSON_TYPE] = 0.0

        elif action == 1:  # LETTER_TRACING
            gain = effective_lr * 0.07
            self.state[self.IDX_LETTER] += gain if success else gain * 0.2
            self.state[self.IDX_LESSON_TYPE] = 0.0

        elif action == 2:  # SYLLABLE_BLENDING
            gain = effective_lr * 0.07
            self.state[self.IDX_SYLLABLE] += gain if success else gain * 0.15
            # Small phoneme reinforcement
            self.state[self.IDX_PHONEME] += gain * 0.2
            self.state[self.IDX_LESSON_TYPE] = 0.0

        elif action == 3:  # WORD_BUILDING
            gain = effective_lr * 0.06
            self.state[self.IDX_WORD] += gain if success else gain * 0.15
            self.state[self.IDX_SYLLABLE] += gain * 0.15
            self.state[self.IDX_LESSON_TYPE] = 0.5

        elif action == 4:  # READ_ALOUD_WORD
            gain = effective_lr * 0.06
            self.state[self.IDX_WORD] += gain if success else gain * 0.1
            self.state[self.IDX_PHONEME] += gain * 0.15
            self.state[self.IDX_LESSON_TYPE] = 0.5

        elif action == 5:  # SENTENCE_READING
            gain = effective_lr * 0.05
            self.state[self.IDX_SENTENCE] += gain if success else gain * 0.1
            self.state[self.IDX_WORD] += gain * 0.1
            self.state[self.IDX_LESSON_TYPE] = 1.0

        elif action == 6:  # VOCABULARY_INTRO
            gain = effective_lr * 0.07
            self.state[self.IDX_VOCAB] += gain
            self.state[self.IDX_LESSON_TYPE] = 0.5
            # New vocab is always "successful" (just exposure)
            success = True

        elif action == 7:  # VOCABULARY_REVIEW
            spaced_rep_bonus = 1.0 + 1.5 * self.state[self.IDX_SPACED_REP]
            gain = effective_lr * 0.06 * spaced_rep_bonus
            self.state[self.IDX_VOCAB] += gain if success else gain * 0.3
            self.state[self.IDX_SPACED_REP] = 0.0
            self.state[self.IDX_TIME_SINCE_REVIEW] = 0.0
            self.state[self.IDX_LESSON_TYPE] = 0.5

        elif action == 8:  # STORY_LISTENING
            gain = effective_lr * 0.04
            self.state[self.IDX_SENTENCE] += gain * 0.7
            self.state[self.IDX_VOCAB] += gain * 0.5
            self.state[self.IDX_ENGAGEMENT] += 0.05  # Stories are engaging
            self.state[self.IDX_LESSON_TYPE] = 1.0
            success = True  # Passive listening

        elif action == 9:  # CALL_AND_RESPONSE
            gain = effective_lr * 0.03
            self.state[self.IDX_PHONEME] += gain
            self.state[self.IDX_VOCAB] += gain * 0.3
            self.state[self.IDX_ENGAGEMENT] += 0.12
            self.state[self.IDX_FATIGUE] -= 0.03  # Fun activity reduces fatigue
            success = True

        elif action == 10:  # SWITCH_LANGUAGE
            self._switch_language()
            success = True

        elif action == 11:  # INCREASE_DIFFICULTY
            self.state[self.IDX_DIFFICULTY] = min(1.0, self.state[self.IDX_DIFFICULTY] + 0.1)
            success = True

        elif action == 12:  # DECREASE_DIFFICULTY
            self.state[self.IDX_DIFFICULTY] = max(0.0, self.state[self.IDX_DIFFICULTY] - 0.1)
            self.state[self.IDX_FRUSTRATION] -= 0.03
            success = True

        elif action == 13:  # GIVE_ENCOURAGEMENT
            self.state[self.IDX_CONFIDENCE] += 0.08
            self.state[self.IDX_ENGAGEMENT] += 0.06
            self.state[self.IDX_FRUSTRATION] -= 0.05
            success = True

        elif action == 14:  # TAKE_BREAK
            self.state[self.IDX_FATIGUE] -= 0.25
            self.state[self.IDX_ENGAGEMENT] += 0.03
            self.state[self.IDX_FRUSTRATION] -= 0.08
            success = True

        # Update performance metrics based on success/failure
        self._update_performance(action, success)

        return success

    def _switch_language(self):
        """Switch to next language, applying transfer effects."""
        old_lang = self.current_language_idx
        # Save current mastery for this language
        self.language_mastery[old_lang][:] = self.state[self.SKILL_INDICES]

        # Cycle to next language
        self.current_language_idx = (self.current_language_idx + 1) % 4
        new_lang = self.current_language_idx
        self.state[self.IDX_LANGUAGE] = self.LANGUAGE_VALUES[new_lang]

        # Apply transfer effects
        transfer = self.TRANSFER_MATRIX[(old_lang, new_lang)]
        self.state[self.IDX_L1_TRANSFER] = transfer

        # Restore language-specific mastery with transfer bonus
        stored = self.language_mastery[new_lang]
        transferred = self.language_mastery[old_lang] * transfer * 0.3
        for i, skill_idx in enumerate(self.SKILL_INDICES):
            self.state[skill_idx] = min(1.0, max(stored[i], transferred[i]))

        # Amharic has a different script - letter recognition transfers less
        if new_lang == 3:
            self.state[self.IDX_LETTER] *= 0.3

    def _get_effective_learning_rate(self):
        """Calculate effective learning rate based on student state."""
        lr = self.base_lr
        # Engagement boosts learning
        lr *= self.state[self.IDX_ENGAGEMENT]
        # Fatigue reduces learning
        lr *= (1.0 - 0.7 * self.state[self.IDX_FATIGUE])
        # Frustration reduces learning
        lr *= (1.0 - 0.5 * self.state[self.IDX_FRUSTRATION])
        # Confidence slightly boosts learning
        lr *= (0.8 + 0.4 * self.state[self.IDX_CONFIDENCE])
        # L1 transfer helps
        lr *= (0.7 + 0.3 * self.state[self.IDX_L1_TRANSFER])
        return lr

    def _update_performance(self, action, success):
        """Update accuracy, latency, streak, and error patterns."""
        # Rolling accuracy (exponential moving average)
        success_val = 1.0 if success else 0.0
        self.state[self.IDX_ACCURACY] = (
            0.8 * self.state[self.IDX_ACCURACY] + 0.2 * success_val
        )

        # Response latency (lower mastery = slower response)
        if action in self.ACTION_SKILL_MAP:
            skill = self.state[self.ACTION_SKILL_MAP[action]]
            self.state[self.IDX_LATENCY] = np.clip(1.0 - skill + self.np_random.normal(0, 0.05), 0, 1)

        if success:
            # Streak grows
            self.state[self.IDX_STREAK] = min(1.0, self.state[self.IDX_STREAK] + 0.1)
            # Errors decay
            self.state[self.IDX_ERR_PHONEME] = max(0, self.state[self.IDX_ERR_PHONEME] - 0.03)
            self.state[self.IDX_ERR_TONE] = max(0, self.state[self.IDX_ERR_TONE] - 0.03)
            self.state[self.IDX_ERR_GRAMMAR] = max(0, self.state[self.IDX_ERR_GRAMMAR] - 0.03)
            # Confidence grows
            self.state[self.IDX_CONFIDENCE] += 0.03
            # Frustration drops
            self.state[self.IDX_FRUSTRATION] = max(0, self.state[self.IDX_FRUSTRATION] - 0.04)
        else:
            # Streak resets
            self.state[self.IDX_STREAK] = 0.0
            # Frustration grows
            self.state[self.IDX_FRUSTRATION] += 0.07
            # Confidence drops
            self.state[self.IDX_CONFIDENCE] -= 0.04
            # Classify and record error type
            self._record_error(action)

    def _record_error(self, action):
        """Record error pattern based on action type and language."""
        # Phoneme-related actions produce phoneme errors
        if action in [0, 2, 4]:
            self.state[self.IDX_ERR_PHONEME] = min(1, self.state[self.IDX_ERR_PHONEME] + 0.1)
        # Yoruba (tonal language) produces tone errors
        if self.current_language_idx == 2:
            self.state[self.IDX_ERR_TONE] = min(1, self.state[self.IDX_ERR_TONE] + 0.12)
        # Sentence/grammar actions produce grammar errors
        if action in [5, 8]:
            self.state[self.IDX_ERR_GRAMMAR] = min(1, self.state[self.IDX_ERR_GRAMMAR] + 0.1)

    def _apply_passive_dynamics(self):
        """Apply time-based passive effects each step."""
        # Fatigue increases each step
        self.state[self.IDX_FATIGUE] += 0.018 + 0.008 * self.state[self.IDX_DIFFICULTY]

        # Engagement naturally decays
        engagement_decay = -0.012
        # Flow state: engagement increases when difficulty matches ability
        avg_mastery = np.mean(self.state[self.SKILL_INDICES])
        difficulty_match = 1.0 - abs(self.state[self.IDX_DIFFICULTY] - avg_mastery)
        engagement_decay += 0.02 * difficulty_match
        self.state[self.IDX_ENGAGEMENT] += engagement_decay

        # Spaced repetition clock ticks
        self.state[self.IDX_SPACED_REP] = min(1, self.state[self.IDX_SPACED_REP] + 0.012)
        self.state[self.IDX_TIME_SINCE_REVIEW] = min(1, self.state[self.IDX_TIME_SINCE_REVIEW] + 0.015)

        # Forgetting curve: mastery decays slowly without review
        decay = 0.004 * self.state[self.IDX_TIME_SINCE_REVIEW]
        for idx in self.SKILL_INDICES:
            self.state[idx] = max(0, self.state[idx] - decay)

        # High frustration accelerates fatigue
        if self.state[self.IDX_FRUSTRATION] > 0.6:
            self.state[self.IDX_FATIGUE] += 0.01

    def _compute_reward(self, action, old_state, new_state, success):
        """Compute composite reward signal."""
        reward = 0.0

        # 1. MASTERY PROGRESS (primary signal)
        mastery_delta = sum(
            new_state[i] - old_state[i] for i in self.SKILL_INDICES
        )
        reward += 10.0 * mastery_delta

        # 2. ENGAGEMENT MAINTENANCE
        reward += 1.0 * (new_state[self.IDX_ENGAGEMENT] - 0.5)

        # 3. FRUSTRATION PENALTY
        reward -= 2.0 * new_state[self.IDX_FRUSTRATION]

        # 4. FATIGUE PENALTY (ignoring tired student)
        if new_state[self.IDX_FATIGUE] > 0.8 and action != 14:
            reward -= 3.0

        # 5. SPACED REPETITION BONUS
        if action == 7 and old_state[self.IDX_SPACED_REP] > 0.5:
            reward += 2.0

        # 6. DIFFICULTY ALIGNMENT
        avg_mastery = np.mean([new_state[i] for i in self.SKILL_INDICES])
        difficulty_gap = abs(new_state[self.IDX_DIFFICULTY] - avg_mastery)
        reward -= 1.5 * difficulty_gap

        # 7. INAPPROPRIATE ACTION PENALTY (prerequisite violation)
        if action in self.PREREQUISITES:
            req_skill, req_thresh = self.PREREQUISITES[action]
            if old_state[req_skill] < req_thresh:
                reward -= 2.0

        # 8. STREAK BONUS
        reward += 0.5 * new_state[self.IDX_STREAK]

        # 9. LANGUAGE TRANSFER BONUS
        if action == 10 and old_state[self.IDX_L1_TRANSFER] > 0.5:
            reward += 1.5

        # 10. SUCCESS BONUS
        if success and action in self.ACTION_SKILL_MAP:
            reward += 0.5

        return reward

    def _check_done(self):
        """Check terminal conditions."""
        avg_mastery = np.mean(self.state[self.SKILL_INDICES])
        truncated = self.current_step >= self.max_steps

        terminated = (
            avg_mastery > 0.85  # Success: student achieved target mastery
            or self.state[self.IDX_ENGAGEMENT] < 0.1  # Failure: disengaged
            or self.state[self.IDX_FRUSTRATION] > 0.95  # Failure: extreme frustration
            or (self.state[self.IDX_FATIGUE] > 0.95 and self.state[self.IDX_ENGAGEMENT] < 0.3)  # Failure: exhaustion
        )

        return bool(terminated), truncated

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    # ---- JSON Serialization for API integration ----

    def _language_name(self):
        return self.LANGUAGES[self.current_language_idx]

    def _lesson_type_name(self):
        lt = self.state[self.IDX_LESSON_TYPE]
        if lt < 0.25:
            return "phonics"
        elif lt < 0.75:
            return "vocabulary"
        else:
            return "comprehension"

    def to_json(self):
        """Serialize current environment state to JSON-compatible dict."""
        terminated, truncated = self._check_done()
        return {
            "step": int(self.current_step),
            "max_steps": int(self.max_steps),
            "student_state": {
                "mastery": {
                    "phoneme": round(float(self.state[self.IDX_PHONEME]), 4),
                    "letter_recognition": round(float(self.state[self.IDX_LETTER]), 4),
                    "syllable_formation": round(float(self.state[self.IDX_SYLLABLE]), 4),
                    "word_reading": round(float(self.state[self.IDX_WORD]), 4),
                    "sentence_comprehension": round(float(self.state[self.IDX_SENTENCE]), 4),
                    "vocabulary_size": round(float(self.state[self.IDX_VOCAB]), 4),
                },
                "affect": {
                    "engagement": round(float(self.state[self.IDX_ENGAGEMENT]), 4),
                    "fatigue": round(float(self.state[self.IDX_FATIGUE]), 4),
                    "frustration": round(float(self.state[self.IDX_FRUSTRATION]), 4),
                    "confidence": round(float(self.state[self.IDX_CONFIDENCE]), 4),
                },
                "performance": {
                    "recent_accuracy": round(float(self.state[self.IDX_ACCURACY]), 4),
                    "response_latency": round(float(self.state[self.IDX_LATENCY]), 4),
                    "streak": round(float(self.state[self.IDX_STREAK]), 4),
                },
                "error_patterns": {
                    "phoneme_errors": round(float(self.state[self.IDX_ERR_PHONEME]), 4),
                    "tone_errors": round(float(self.state[self.IDX_ERR_TONE]), 4),
                    "grammar_errors": round(float(self.state[self.IDX_ERR_GRAMMAR]), 4),
                },
            },
            "session": {
                "language": self._language_name(),
                "difficulty": round(float(self.state[self.IDX_DIFFICULTY]), 4),
                "progress": round(float(self.state[self.IDX_PROGRESS]), 4),
                "spaced_rep_due": round(float(self.state[self.IDX_SPACED_REP]), 4),
                "lesson_type": self._lesson_type_name(),
            },
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

    def from_json(self, data):
        """Restore environment state from JSON dict."""
        ss = data["student_state"]
        m = ss["mastery"]
        self.state[self.IDX_PHONEME] = m["phoneme"]
        self.state[self.IDX_LETTER] = m["letter_recognition"]
        self.state[self.IDX_SYLLABLE] = m["syllable_formation"]
        self.state[self.IDX_WORD] = m["word_reading"]
        self.state[self.IDX_SENTENCE] = m["sentence_comprehension"]
        self.state[self.IDX_VOCAB] = m["vocabulary_size"]

        a = ss["affect"]
        self.state[self.IDX_ENGAGEMENT] = a["engagement"]
        self.state[self.IDX_FATIGUE] = a["fatigue"]
        self.state[self.IDX_FRUSTRATION] = a["frustration"]
        self.state[self.IDX_CONFIDENCE] = a["confidence"]

        p = ss["performance"]
        self.state[self.IDX_ACCURACY] = p["recent_accuracy"]
        self.state[self.IDX_LATENCY] = p["response_latency"]
        self.state[self.IDX_STREAK] = p["streak"]

        e = ss["error_patterns"]
        self.state[self.IDX_ERR_PHONEME] = e["phoneme_errors"]
        self.state[self.IDX_ERR_TONE] = e["tone_errors"]
        self.state[self.IDX_ERR_GRAMMAR] = e["grammar_errors"]

        s = data["session"]
        self.state[self.IDX_DIFFICULTY] = s["difficulty"]
        self.state[self.IDX_PROGRESS] = s["progress"]
        self.state[self.IDX_SPACED_REP] = s["spaced_rep_due"]
        self.current_language_idx = self.LANGUAGES.index(s["language"])
        self.state[self.IDX_LANGUAGE] = self.LANGUAGE_VALUES[self.current_language_idx]
        self.current_step = data["step"]

    @staticmethod
    def action_from_json(data):
        """Parse an action from a JSON API request."""
        action_map = {name.lower(): i for i, name in enumerate(AfricanLiteracyTutorEnv.ACTION_NAMES)}
        return action_map[data["action"].lower()]

    def get_action_descriptions(self):
        """Return action descriptions for API documentation."""
        descriptions = {
            0: "Practice individual sound recognition with audio prompts",
            1: "Visual letter/character tracing exercise",
            2: "Combine phonemes into syllables interactively",
            3: "Construct words from known syllables",
            4: "Student reads a word aloud (voice exercise)",
            5: "Read and comprehend a full sentence",
            6: "Introduce new vocabulary with picture and audio",
            7: "Review previously learned vocabulary (spaced repetition)",
            8: "Listen to a short story in the target language",
            9: "Interactive call-and-response game (engagement booster)",
            10: "Switch to the next target language",
            11: "Move to harder material",
            12: "Move to easier material (scaffolding)",
            13: "Give verbal encouragement and praise",
            14: "Give the student a short rest period",
        }
        return {
            self.ACTION_NAMES[i]: {"id": i, "description": descriptions[i]}
            for i in range(15)
        }
