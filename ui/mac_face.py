"""
ui/mac_face.py — PixelFace

Draws a full-screen pixel-art face directly on the display surface.
No Mac body is rendered — the physical screen bezel IS the Mac.

Each facial feature (eyes, nose, mouth) is defined as a 2-D grid of
integer values in expressions.py and rendered as glowing LED dots:

    0 = off (background)
    1 = iris / feature pixel
    2 = pupil centre anchor (eye only)

The face floats vertically on a sine wave and responds to a look-at
target so pupils track keyboard input location.
"""

import math
import random
import time

import pygame

from config import (
    COLOR_BG,
    FLOAT_AMPLITUDE, FLOAT_SPEED,
)
from ui.expressions import (
    Expression, EXPRESSION_COLORS, GLOW_PULSE_SPEED,
    EYE_GRIDS, MOUTH_GRIDS,
    BLINK_INTERVAL_MIN, BLINK_INTERVAL_MAX, BLINK_DURATION,
    TALK_FRAME_RATE, EXPR_BLEND_SPEED, LOOK_LERP_SPEED,
    COLOR_LERP_SPEED, THINK_SWAY_SPEED, THINK_SWAY_PX,
    lerp, lerp_color, smoothstep, pulse,
)


class PixelFace:
    """
    Full-screen pixel-art face.

    Call  update(dt)         each frame to advance animations.
    Call  draw(surface)      each frame to render.
    Call  set_look_target(dx, dy)  to smoothly move pupils.
    Set   .expression        to change state (smooth crossfade).
    """

    # ── Layout constants (dot-grid units, relative to face centre) ────────────
    # Upper-face grid: 10 cols × 6 rows (both eyes + nose in one grid)
    EYE_COLS = 10
    EYE_ROWS = 6
    # Vertical offset from face centre to upper-face grid centre (negative = up)
    EYE_OFFSET_Y = -3.5

    # Mouth is 12 cols × 4 rows
    MOUTH_COLS = 12
    MOUTH_ROWS = 4
    MOUTH_OFFSET_Y = 4.5   # below face centre

    def __init__(self, screen_w: int, screen_h: int):
        self._W = screen_w
        self._H = screen_h

        # dot_px: pixel size of one grid dot — scales to screen
        self._dot = int(screen_h / 24)   # ~37px on 900p, ~45px on 1080p
        self._dot = max(20, self._dot)

        # Face centre in screen pixels
        self._cx = screen_w // 2
        self._cy = screen_h // 2

        # Expression state
        self._expr         = Expression.IDLE
        self._expr_target  = Expression.IDLE
        self._expr_blend   = 1.0            # 1.0 = fully at target

        # Colour interpolation
        self._cur_color    = list(EXPRESSION_COLORS[Expression.IDLE])
        self._tgt_color    = list(EXPRESSION_COLORS[Expression.IDLE])

        # Float animation
        self._phase        = 0.0

        # Blink
        self._next_blink   = self._rand_blink()
        self._blink_t      = 0.0
        self._blinking     = False

        # Talking mouth frame
        self._talk_t       = 0.0
        self._mouth_frame  = 0

        # Look-at (smooth pupil tracking)
        self._look_dx      = 0.0
        self._look_dy      = 0.0
        self._look_tgt_dx  = 0.0
        self._look_tgt_dy  = 0.0

        # Thinking sway
        self._sway_x       = 0.0
        self._time         = 0.0

        # Glow pulse timer (kept for API compatibility, unused with square pixels)
        self._glow_t       = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def expression(self) -> Expression:
        return self._expr_target

    @expression.setter
    def expression(self, expr: Expression):
        if expr == self._expr_target:
            return
        self._expr        = self._expr_target
        self._expr_target = expr
        self._expr_blend  = 0.0
        self._tgt_color   = list(EXPRESSION_COLORS[expr])

    def set_look_target(self, dx: float, dy: float):
        """dx, dy in -1..+1; face smoothly tracks."""
        self._look_tgt_dx = max(-1.0, min(1.0, dx))
        self._look_tgt_dy = max(-1.0, min(1.0, dy))

    def update(self, dt: float):
        self._time    += dt
        self._phase   += dt * FLOAT_SPEED
        self._glow_t  += dt

        # Expression blend
        if self._expr_blend < 1.0:
            self._expr_blend = min(1.0, self._expr_blend + dt * EXPR_BLEND_SPEED)

        # Colour lerp
        t = smoothstep(self._expr_blend)
        for i in range(3):
            self._cur_color[i] = lerp(self._cur_color[i], self._tgt_color[i], dt * COLOR_LERP_SPEED)

        # Blink
        if not self._blinking:
            self._next_blink -= dt
            if self._next_blink <= 0:
                self._blinking = True
                self._blink_t  = 0.0
        else:
            self._blink_t += dt / BLINK_DURATION
            if self._blink_t >= 1.0:
                self._blinking   = False
                self._next_blink = self._rand_blink()
                self._blink_t    = 0.0

        # Talking mouth animation — also runs during blend-out from TALKING
        if self._expr_target == Expression.TALKING or (
            self._expr == Expression.TALKING and self._expr_blend < 1.0
        ):
            self._talk_t += dt * TALK_FRAME_RATE
            frames = MOUTH_GRIDS[Expression.TALKING]
            self._mouth_frame = int(self._talk_t) % len(frames)
        else:
            self._talk_t  = 0.0
            self._mouth_frame = 0

        # Look-at lerp
        spd = LOOK_LERP_SPEED * dt
        self._look_dx += (self._look_tgt_dx - self._look_dx) * min(1.0, spd * 3)
        self._look_dy += (self._look_tgt_dy - self._look_dy) * min(1.0, spd * 3)

        # Thinking sway
        if self._expr_target == Expression.THINKING:
            self._sway_x = math.sin(self._time * THINK_SWAY_SPEED) * THINK_SWAY_PX
        else:
            self._sway_x *= 0.88   # dampen

    def draw(self, surface: pygame.Surface):
        # Global float offset
        float_y = int(math.sin(self._phase) * FLOAT_AMPLITUDE)
        sway_x  = int(self._sway_x)

        cx = self._cx + sway_x
        cy = self._cy + float_y

        color = tuple(int(c) for c in self._cur_color)

        # Blink alpha: sin curve, 0..1 (1=open, 0=closed)
        if self._blinking:
            blink_open = max(0.0, math.sin(self._blink_t * math.pi))
        else:
            blink_open = 1.0

        # Blend weight for expression crossfade
        blend = smoothstep(self._expr_blend)

        # ── Upper face (eyes + nose) ──────────────────────────────────────────
        upper_cy = cy + int(self.EYE_OFFSET_Y * self._dot)
        self._draw_upper_face(surface, cx, upper_cy, color, blink_open, blend)

        # ── Mouth ─────────────────────────────────────────────────────────────
        mouth_cy = cy + int(self.MOUTH_OFFSET_Y * self._dot)
        self._draw_mouth(surface, cx, mouth_cy, color, blend)

    # ── Draw helpers ─────────────────────────────────────────────────────────

    def _draw_upper_face(self, surface, cx, cy, color, blink_open, blend):
        d = self._dot
        grid_from = EYE_GRIDS[self._expr]
        grid_to   = EYE_GRIDS[self._expr_target]
        rows = self.EYE_ROWS   # 6
        cols = self.EYE_COLS   # 8
        ox = cx - (cols / 2) * d
        oy = cy - (rows / 2) * d

        for row in range(rows):
            for col in range(cols):
                val_from = grid_from[row][col]
                val_to   = grid_to[row][col]
                alpha_from = int((1 - blend) * 255 * (val_from > 0))
                alpha_to   = int(blend * 255 * (val_to > 0))
                alpha = max(alpha_from, alpha_to)
                if alpha == 0:
                    continue
                # Blink only on eye rows (0–2), not nose rows (3–5)
                if row <= 2:
                    alpha = int(alpha * blink_open)
                if alpha < 4:
                    continue
                px = int(ox + (col + 0.5) * d)
                py = int(oy + (row + 0.5) * d)
                self._draw_pixel(surface, px, py, color, alpha)

    def _draw_mouth(self, surface, cx, cy, color, blend):
        d = self._dot
        cols = self.MOUTH_COLS
        rows = self.MOUTH_ROWS

        frames_from = MOUTH_GRIDS[self._expr]
        frames_to   = MOUTH_GRIDS[self._expr_target]

        frame_from = frames_from[0]
        if self._expr_target == Expression.TALKING:
            frame_to = frames_to[self._mouth_frame]
        else:
            frame_to = frames_to[0]

        ox = cx - (cols / 2) * d
        oy = cy - (rows / 2) * d

        for row in range(rows):
            for col in range(cols):
                val_from = frame_from[row][col]
                val_to   = frame_to[row][col]

                alpha_from = int((1 - blend) * 255 * val_from)
                alpha_to   = int(blend * 255 * val_to)
                alpha      = max(alpha_from, alpha_to)
                if alpha < 4:
                    continue

                px = int(ox + (col + 0.5) * d)
                py = int(oy + (row + 0.5) * d)
                self._draw_pixel(surface, px, py, color, alpha)

    def _draw_pixel(self, surface, px, py, color, alpha):
        """Draw one retro square pixel — no glow, no round edges, just a crisp block."""
        d = self._dot
        size = max(3, int(d * 0.80))   # 80% fill → 20% gap gives pixel-grid look
        half = size // 2
        if alpha >= 252:
            pygame.draw.rect(surface, color, (px - half, py - half, size, size))
        else:
            s = pygame.Surface((size, size), pygame.SRCALPHA)
            s.fill((*color, alpha))
            surface.blit(s, (px - half, py - half))

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _rand_blink() -> float:
        return random.uniform(BLINK_INTERVAL_MIN, BLINK_INTERVAL_MAX)
