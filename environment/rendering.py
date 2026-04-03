"""
PyOpenGL + Pygame Visualization for the African Literacy AI Tutor Environment.

Uses OpenGL for advanced rendering (radar charts, gradient bars, avatar)
and pygame for window management and text rendering.
"""

import math
import numpy as np

import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT, KEYDOWN, K_ESCAPE
from OpenGL.GL import (  # type: ignore[import-untyped]
    glClear, glClearColor, glEnable, glDisable, glBlendFunc, glLineWidth,
    glBegin, glEnd, glVertex2f, glColor3f, glColor4f,
    glMatrixMode, glLoadIdentity, glOrtho, glViewport,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    GL_TRIANGLE_FAN, GL_LINE_LOOP, GL_LINES,
    GL_QUADS, GL_PROJECTION, GL_MODELVIEW,
    glGenTextures, glBindTexture, glTexImage2D, glTexParameteri,
    glDeleteTextures, glTexCoord2f,
    GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
    GL_LINEAR, GL_RGBA, GL_UNSIGNED_BYTE,
)

HAS_OPENGL = True


class TutorRenderer:
    """OpenGL-based renderer for the African Literacy Tutor environment."""

    # Color palette
    BG_COLOR = (0.12, 0.13, 0.18)
    PANEL_COLOR = (0.18, 0.19, 0.25)
    HEADER_COLOR = (0.15, 0.25, 0.45)
    TEXT_COLOR = (230, 230, 240)
    ACCENT_GOLD = (1.0, 0.78, 0.2)
    ACCENT_BLUE = (0.3, 0.6, 1.0)

    # Skill names for radar chart
    SKILL_NAMES = ["Phoneme", "Letter", "Syllable", "Word", "Sentence", "Vocab"]

    # Language flag colors (simplified)
    LANG_COLORS = {
        "Kinyarwanda": (0.0, 0.4, 0.8),
        "Swahili": (0.0, 0.6, 0.3),
        "Yoruba": (0.0, 0.5, 0.0),
        "Amharic": (0.0, 0.7, 0.2),
    }

    def __init__(self, width=1000, height=700):
        self.width = width
        self.height = height
        self.initialized = False
        self.font = None
        self.font_small = None
        self.font_large = None
        self.font_title = None
        self.action_log = []
        self.particles = []
        self._rng = np.random.default_rng()
        self._init_particles()

    def _init_particles(self):
        """Initialize floating letter particles."""
        self.particles = []
        for _ in range(20):
            self.particles.append({
                "x": self._rng.uniform(0, 1000),
                "y": self._rng.uniform(0, 700),
                "vx": self._rng.uniform(-0.3, 0.3),
                "vy": self._rng.uniform(-0.5, -0.1),
                "alpha": self._rng.uniform(0.05, 0.15),
                "char": chr(self._rng.integers(65, 91)),
                "size": self._rng.integers(14, 28),
            })

    def initialize(self):
        """Initialize pygame and OpenGL context."""
        if not HAS_OPENGL:
            raise ImportError("PyOpenGL and pygame are required for rendering.")

        pygame.init()
        pygame.display.set_mode(
            (self.width, self.height), int(DOUBLEBUF) | int(OPENGL)
        )
        pygame.display.set_caption("African Literacy AI Tutor - RL Environment")

        # OpenGL setup
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(*self.BG_COLOR, 1.0)

        # Pygame fonts for text
        self.font = pygame.font.SysFont("Arial", 16)
        self.font_small = pygame.font.SysFont("Arial", 13)
        self.font_large = pygame.font.SysFont("Arial", 22, bold=True)
        self.font_title = pygame.font.SysFont("Arial", 26, bold=True)

        self.initialized = True

    def render(self, env_state, info=None):
        """Render a single frame of the environment."""
        if not self.initialized:
            self.initialize()

        for event in pygame.event.get():
            if event.type == QUIT:
                self.close()
                return False
            if event.type == KEYDOWN and event.key == K_ESCAPE:
                self.close()
                return False

        glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))

        # Update action log
        if info and "action_name" in info:
            success_str = "OK" if info.get("success", False) else "FAIL"
            log_entry = f"Step {info.get('step', '?')}: {info['action_name']} -> {success_str}"
            self.action_log.append(log_entry)
            if len(self.action_log) > 6:
                self.action_log.pop(0)

        # Draw all components
        self._draw_header(env_state, info)
        self._draw_radar_chart(env_state, cx=200, cy=420, radius=120)
        self._draw_student_avatar(env_state, cx=650, cy=450)
        self._draw_affective_bars(env_state, x=420, y=180, w=540, h=130)
        self._draw_action_log(x=30, y=25, w=940, h=145)
        self._draw_particles(env_state)
        self._draw_session_info(env_state)

        # Render all text textures
        self._flush_text()

        pygame.display.flip()
        pygame.time.wait(100)
        return True

    def _draw_header(self, state, info):
        """Draw header bar with title and session info."""
        # Header background
        self._draw_rect(0, self.height - 60, self.width, 60, (*self.HEADER_COLOR, 0.95))

        lang_idx = int(round(state[13] * 3))
        lang_name = ["Kinyarwanda", "Swahili", "Yoruba", "Amharic"][min(lang_idx, 3)]
        step = info.get("step", 0) if info else 0
        cum_reward = info.get("cumulative_reward", 0) if info else 0
        avg_mastery = float(np.mean(state[:6]))

        self._queue_text(
            "African Literacy AI Tutor",
            20, self.height - 42, self.font_title, (255, 220, 100)
        )
        self._queue_text(
            f"Language: {lang_name}    Step: {step}/200    "
            f"Reward: {cum_reward:+.1f}    Mastery: {avg_mastery:.1%}",
            400, self.height - 38, self.font, self.TEXT_COLOR
        )

    def _draw_radar_chart(self, state, cx, cy, radius):
        """Draw a 6-axis radar chart for mastery skills using OpenGL."""
        n = 6
        skills = [float(state[i]) for i in range(6)]
        angles = [2 * math.pi * i / n - math.pi / 2 for i in range(n)]

        # Panel background
        self._draw_rect(cx - radius - 50, cy - radius - 50,
                        2 * radius + 100, 2 * radius + 90,
                        (*self.PANEL_COLOR, 0.8))

        # Label
        self._queue_text("Literacy Skills", cx - 45, cy + radius + 25, self.font_large, self.ACCENT_GOLD)

        # Draw grid rings (OpenGL lines)
        for ring in [0.25, 0.5, 0.75, 1.0]:
            glColor4f(0.4, 0.4, 0.5, 0.3)
            glBegin(GL_LINE_LOOP)
            for angle in angles:
                x = cx + radius * ring * math.cos(angle)
                y = cy + radius * ring * math.sin(angle)
                glVertex2f(x, y)
            glEnd()

        # Draw axis lines
        glColor4f(0.4, 0.4, 0.5, 0.4)
        glBegin(GL_LINES)
        for angle in angles:
            glVertex2f(cx, cy)
            glVertex2f(cx + radius * math.cos(angle), cy + radius * math.sin(angle))
        glEnd()

        # Draw filled mastery polygon
        glColor4f(0.3, 0.7, 1.0, 0.3)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(cx, cy)
        for i, angle in enumerate(angles):
            val = max(skills[i], 0.02)
            x = cx + radius * val * math.cos(angle)
            y = cy + radius * val * math.sin(angle)
            glVertex2f(x, y)
        # Close the polygon
        val = max(skills[0], 0.02)
        glVertex2f(
            cx + radius * val * math.cos(angles[0]),
            cy + radius * val * math.sin(angles[0])
        )
        glEnd()

        # Draw mastery outline
        glColor4f(0.3, 0.7, 1.0, 0.9)
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        for i, angle in enumerate(angles):
            val = max(skills[i], 0.02)
            x = cx + radius * val * math.cos(angle)
            y = cy + radius * val * math.sin(angle)
            glVertex2f(x, y)
        glEnd()
        glLineWidth(1.0)

        # Draw skill labels
        for i, angle in enumerate(angles):
            lx = cx + (radius + 20) * math.cos(angle)
            ly = cy + (radius + 20) * math.sin(angle)
            val_str = f"{skills[i]:.0%}"
            self._queue_text(
                f"{self.SKILL_NAMES[i]} {val_str}",
                int(lx) - 25, int(ly) - 6, self.font_small, (200, 210, 230)
            )

    def _draw_student_avatar(self, state, cx, cy):
        """Draw student avatar with expression based on affective state."""
        engagement = float(state[6])
        fatigue = float(state[7])
        frustration = float(state[8])
        confidence = float(state[9])

        # Panel background
        self._draw_rect(cx - 130, cy - 140, 260, 280, (*self.PANEL_COLOR, 0.8))
        self._queue_text("Student State", cx - 45, cy + 125, self.font_large, self.ACCENT_GOLD)

        # Body color based on engagement (warm=engaged, cool=disengaged)
        r = 0.3 + 0.5 * engagement
        g = 0.4 + 0.3 * engagement
        b = 0.8 - 0.3 * engagement

        # Body (torso) - OpenGL quad
        self._draw_rect(cx - 30, cy - 100, 60, 70, (r * 0.8, g * 0.8, b * 0.8, 0.9))

        # Head - OpenGL circle
        head_r = 35
        self._draw_circle(cx, cy + 10, head_r, (r, g, b, 1.0))

        # Eyes
        eye_y = cy + 18
        eye_droop = fatigue * 8  # Eyes droop when tired
        # Left eye
        self._draw_circle(cx - 12, eye_y, 5, (1.0, 1.0, 1.0, 1.0))
        self._draw_circle(cx - 12, eye_y - eye_droop * 0.3, 2.5, (0.15, 0.15, 0.2, 1.0))
        # Right eye
        self._draw_circle(cx + 12, eye_y, 5, (1.0, 1.0, 1.0, 1.0))
        self._draw_circle(cx + 12, eye_y - eye_droop * 0.3, 2.5, (0.15, 0.15, 0.2, 1.0))

        # Eyelids (fatigue indicator)
        if fatigue > 0.3:
            lid_h = fatigue * 6
            glColor4f(r, g, b, 0.9)
            glBegin(GL_QUADS)
            glVertex2f(cx - 18, eye_y + 5)
            glVertex2f(cx - 6, eye_y + 5)
            glVertex2f(cx - 6, eye_y + 5 - lid_h)
            glVertex2f(cx - 18, eye_y + 5 - lid_h)
            glVertex2f(cx + 6, eye_y + 5)
            glVertex2f(cx + 18, eye_y + 5)
            glVertex2f(cx + 18, eye_y + 5 - lid_h)
            glVertex2f(cx + 6, eye_y + 5 - lid_h)
            glEnd()

        # Mouth - expression based on confidence and frustration
        mouth_y = cy - 2
        if confidence > 0.6 and frustration < 0.3:
            # Smile
            self._draw_arc(cx, mouth_y, 10, 0, math.pi, (0.9, 0.3, 0.3, 0.9))
        elif frustration > 0.6:
            # Frown
            self._draw_arc(cx, mouth_y - 5, 10, math.pi, 2 * math.pi, (0.9, 0.3, 0.3, 0.9))
        else:
            # Neutral
            glColor4f(0.9, 0.3, 0.3, 0.7)
            glBegin(GL_LINES)
            glVertex2f(cx - 8, mouth_y)
            glVertex2f(cx + 8, mouth_y)
            glEnd()

        # Frustration indicator (red aura)
        if frustration > 0.4:
            self._draw_circle(cx, cy + 10, head_r + 8,
                              (1.0, 0.2, 0.1, frustration * 0.3))

        # Status text
        if confidence > 0.6:
            mood = "Happy"
        elif frustration > 0.5:
            mood = "Frustrated"
        elif engagement > 0.5:
            mood = "Focused"
        else:
            mood = "Tired"
        self._queue_text(f"Mood: {mood}", cx - 25, cy - 120, self.font_small, (200, 200, 220))

    def _draw_affective_bars(self, state, x, y, w, h):
        """Draw gradient progress bars for affective states using OpenGL quads."""
        self._draw_rect(x, y, w, h, (*self.PANEL_COLOR, 0.8))
        self._queue_text("Affective & Performance", x + 10, y + h - 22, self.font_large, self.ACCENT_GOLD)

        bars = [
            ("Engagement", float(state[6]), (0.2, 0.8, 0.3), (0.8, 0.2, 0.1)),
            ("Fatigue", float(state[7]), (0.3, 0.6, 0.9), (0.9, 0.2, 0.2)),
            ("Frustration", float(state[8]), (0.9, 0.8, 0.2), (0.9, 0.1, 0.1)),
            ("Confidence", float(state[9]), (0.3, 0.5, 0.9), (0.2, 0.9, 0.3)),
        ]

        bar_h = 16
        bar_x = x + 110
        bar_w = w - 140
        for i, (name, val, color_low, color_high) in enumerate(bars):
            by = y + h - 48 - i * 26

            # Label
            self._queue_text(f"{name}:", x + 10, by - 1, self.font_small, self.TEXT_COLOR)

            # Background bar
            self._draw_rect(bar_x, by - 2, bar_w, bar_h, (0.25, 0.25, 0.3, 0.8))

            # Filled gradient bar (OpenGL quads with color interpolation)
            fill_w = bar_w * np.clip(val, 0, 1)
            if fill_w > 1:
                t = np.clip(val, 0, 1)
                r1, g1, b1 = color_low
                r2, g2, b2 = color_high
                glBegin(GL_QUADS)
                glColor4f(r1, g1, b1, 0.9)
                glVertex2f(bar_x, by - 2)
                glVertex2f(bar_x, by - 2 + bar_h)
                glColor4f(
                    r1 + t * (r2 - r1),
                    g1 + t * (g2 - g1),
                    b1 + t * (b2 - b1),
                    0.9
                )
                glVertex2f(bar_x + fill_w, by - 2 + bar_h)
                glVertex2f(bar_x + fill_w, by - 2)
                glEnd()

            # Value text
            self._queue_text(f"{val:.0%}", bar_x + bar_w + 5, by - 1, self.font_small, self.TEXT_COLOR)

    def _draw_action_log(self, x, y, w, h):
        """Draw the action log panel."""
        self._draw_rect(x, y, w, h, (*self.PANEL_COLOR, 0.8))
        self._queue_text("Action Log", x + 10, y + h - 22, self.font_large, self.ACCENT_GOLD)

        for i, entry in enumerate(self.action_log[-5:]):
            color = (180, 230, 180) if "OK" in entry else (230, 160, 160)
            self._queue_text(entry, x + 15, y + h - 45 - i * 20, self.font_small, color)

    def _draw_session_info(self, state):
        """Draw session info panel."""
        x, y, w, h = 30, 180, 380, 130
        self._draw_rect(x, y, w, h, (*self.PANEL_COLOR, 0.8))
        self._queue_text("Session Info", x + 10, y + h - 22, self.font_large, self.ACCENT_GOLD)

        diff = float(state[16])
        spaced = float(state[15])
        accuracy = float(state[10])
        streak = float(state[17])

        lines = [
            f"Difficulty: {diff:.0%}    Accuracy: {accuracy:.0%}",
            f"Spaced Rep Due: {spaced:.0%}    Streak: {streak:.0%}",
            f"Error Pattern: Ph={state[19]:.0%} Tn={state[20]:.0%} Gr={state[21]:.0%}",
        ]
        for i, line in enumerate(lines):
            self._queue_text(line, x + 15, y + h - 48 - i * 22, self.font_small, (190, 200, 220))

    def _draw_particles(self, state):
        """Draw floating letter particles (from current language script)."""
        lang_idx = int(round(float(state[13]) * 3))
        lang_idx = min(lang_idx, 3)

        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            if p["y"] < -20:
                p["y"] = self.height + 20
                p["x"] = self._rng.uniform(0, self.width)

            glColor4f(0.5, 0.6, 0.8, p["alpha"])
            # Render as a small square (represents a letter)
            s = p["size"] * 0.3
            glBegin(GL_QUADS)
            glVertex2f(p["x"] - s, p["y"] - s)
            glVertex2f(p["x"] + s, p["y"] - s)
            glVertex2f(p["x"] + s, p["y"] + s)
            glVertex2f(p["x"] - s, p["y"] + s)
            glEnd()

    # ---- OpenGL Drawing Helpers ----

    def _draw_rect(self, x, y, w, h, color):
        """Draw a filled rectangle with OpenGL."""
        if len(color) == 4:
            glColor4f(*color)
        else:
            glColor3f(*color)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()

    def _draw_circle(self, cx, cy, r, color, segments=24):
        """Draw a filled circle with OpenGL."""
        if len(color) == 4:
            glColor4f(*color)
        else:
            glColor3f(*color)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(cx, cy)
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            glVertex2f(cx + r * math.cos(angle), cy + r * math.sin(angle))
        glEnd()

    def _draw_arc(self, cx, cy, r, start_angle, end_angle, color, segments=12):
        """Draw an arc using OpenGL lines."""
        if len(color) == 4:
            glColor4f(*color)
        else:
            glColor3f(*color)
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        for i in range(segments + 1):
            angle = start_angle + (end_angle - start_angle) * i / segments
            glVertex2f(cx + r * math.cos(angle), cy + r * math.sin(angle))
        glEnd()
        glLineWidth(1.0)

    # ---- Text Rendering (pygame surface -> displayed via OpenGL) ----

    def __init_text_queue(self):
        if not hasattr(self, "_text_queue"):
            self._text_queue = []

    def _queue_text(self, text, x, y, font, color):
        """Queue text for batch rendering."""
        self.__init_text_queue()
        self._text_queue.append((text, x, y, font, color))

    def _flush_text(self):
        """Render all queued text using pygame on an OpenGL-compatible surface."""
        self.__init_text_queue()
        if not self._text_queue:
            return

        # We need to temporarily switch to a 2D overlay mode
        # Create a pygame surface, render text, then blit via OpenGL texture
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        surface.fill((0, 0, 0, 0))

        for text, x, y, font, color in self._text_queue:
            rendered = font.render(str(text), True, color)
            # Flip y coordinate (OpenGL origin is bottom-left, pygame is top-left)
            py_y = self.height - y - rendered.get_height()
            surface.blit(rendered, (x, py_y))

        self._text_queue.clear()

        # Convert pygame surface to OpenGL texture and render as fullscreen quad
        text_data = pygame.image.tostring(surface, "RGBA", True)

        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height,
            0, GL_RGBA, GL_UNSIGNED_BYTE, text_data
        )

        glEnable(GL_TEXTURE_2D)
        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(self.width, 0)
        glTexCoord2f(1, 1); glVertex2f(self.width, self.height)
        glTexCoord2f(0, 1); glVertex2f(0, self.height)
        glEnd()

        glDisable(GL_TEXTURE_2D)
        glDeleteTextures([tex_id])

    def close(self):
        """Clean up pygame and OpenGL resources."""
        if self.initialized:
            pygame.quit()
            self.initialized = False
