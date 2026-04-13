"""
ui/app.py — pygame app with configurable startup window modes.

Renders only the PixelFace on a dark CRT background.
No HUD, no labels, no status text on screen.
Keyboard input is captured silently; on Enter it fires on_text_input.
While the user is typing, face pupils look downward (toward keyboard).
"""

import threading
import time
from typing import Callable

import pygame

from config import (
    COLOR_BG,
    TARGET_FPS,
    WINDOW_TITLE,
    UI_WINDOW_MODE,
    UI_FORCE_FULLSCREEN,
    UI_WINDOW_WIDTH,
    UI_WINDOW_HEIGHT,
)
from ui.mac_face import PixelFace
from ui.expressions import Expression


class TalkingMACApp:
    """
    pygame UI with configurable startup window mode.

    Thread-safe: set_expression / set_mode_label / show_status can be
    called from any thread — they write to lock-protected attributes.
    The pygame render loop runs on the main thread.
    """

    def __init__(self):
        pygame.init()
        pygame.display.set_caption(WINDOW_TITLE)
        pygame.mouse.set_visible(False)

        info = pygame.display.Info()
        desktop_w = info.current_w
        desktop_h = info.current_h

        mode = (UI_WINDOW_MODE or "fullscreen").strip().lower()
        if UI_FORCE_FULLSCREEN:
            mode = "fullscreen_borderless"

        (self._W, self._H), flags = self._resolve_display_mode(mode, desktop_w, desktop_h)
        self._screen = pygame.display.set_mode((self._W, self._H), flags)
        self._clock = pygame.time.Clock()
        self._running = True

        # PixelFace sizes itself from the selected display mode dimensions.
        self._face = PixelFace(self._W, self._H)

        # State (lock-protected for thread safety)
        self._lock = threading.Lock()
        self._expression = Expression.IDLE
        self._input_buf = ""
        self._typing = False

        # Callbacks
        self.on_text_input: Callable[[str], None] | None = None
        self.on_quit: Callable[[], None] | None = None

        # CRT scanline + vignette overlay (baked once at startup)
        self._overlay = self._make_overlay()

    @staticmethod
    def _resolve_display_mode(mode: str, desktop_w: int, desktop_h: int) -> tuple[tuple[int, int], int]:
        hw_flags = pygame.HWSURFACE | pygame.DOUBLEBUF

        if mode == "fullscreen":
            return (desktop_w, desktop_h), pygame.FULLSCREEN | hw_flags

        if mode == "fullscreen_borderless":
            return (desktop_w, desktop_h), pygame.NOFRAME | hw_flags

        width = min(max(320, UI_WINDOW_WIDTH), desktop_w)
        height = min(max(240, UI_WINDOW_HEIGHT), desktop_h)

        if mode == "windowed":
            return (width, height), pygame.RESIZABLE | hw_flags

        if mode == "floating":
            # Best effort: pygame cannot enforce always-on-top portably.
            float_w = min(width, max(640, desktop_w // 2))
            float_h = min(height, max(420, desktop_h // 2))
            return (float_w, float_h), pygame.NOFRAME | hw_flags

        # "borderless" and unknown values both map to frameless window.
        return (width, height), pygame.NOFRAME | hw_flags

    # ── Public (thread-safe) ─────────────────────────────────────────────────

    def set_expression(self, expr: Expression):
        with self._lock:
            self._expression = expr
        self._face.expression = expr

    def set_look_target(self, dx: float, dy: float):
        self._face.set_look_target(dx, dy)

    # These are kept for API compatibility with main.py but do nothing visual
    def show_status(self, text: str, ttl: float = 5.0):
        pass

    def set_mode_label(self, label: str):
        pass

    def is_running(self) -> bool:
        return self._running

    def stop(self):
        self._running = False

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        last = time.time()
        while self._running:
            now = time.time()
            dt = min(now - last, 0.05)
            last = now

            self._handle_events()
            self._update(dt)
            self._render()
            self._clock.tick(TARGET_FPS)

        pygame.quit()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self._do_quit()

            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    self._do_quit()

                elif ev.key == pygame.K_RETURN:
                    buf = self._input_buf.strip()
                    self._input_buf = ""
                    self._typing = False
                    self._face.set_look_target(0.0, 0.0)
                    if buf and self.on_text_input:
                        threading.Thread(
                            target=self.on_text_input, args=(buf,), daemon=True
                        ).start()

                elif ev.key == pygame.K_BACKSPACE:
                    self._input_buf = self._input_buf[:-1]
                    if not self._input_buf:
                        self._typing = False
                        self._face.set_look_target(0.0, 0.0)

                else:
                    ch = ev.unicode
                    if ch and ch.isprintable():
                        self._input_buf += ch
                        if not self._typing:
                            self._typing = True
                            self._face.set_look_target(0.0, 0.62)

    def _do_quit(self):
        self._running = False
        if self.on_quit:
            self.on_quit()

    def _update(self, dt: float):
        with self._lock:
            self._face.expression = self._expression
        self._face.update(dt)

    def _render(self):
        surf = self._screen
        surf.fill(COLOR_BG)
        self._face.draw(surf)
        surf.blit(self._overlay, (0, 0))
        pygame.display.flip()

    def _make_overlay(self) -> pygame.Surface:
        """Bake CRT scanlines + vignette into one surface."""
        overlay = pygame.Surface((self._W, self._H), pygame.SRCALPHA)

        for y in range(0, self._H, 4):
            pygame.draw.line(overlay, (0, 0, 0, 8), (0, y), (self._W, y))

        for i in range(30, 0, -1):
            frac = i / 30
            alpha = int(frac ** 2.4 * 55)
            rw = int(self._W * (1 - frac * 0.46))
            rh = int(self._H * (1 - frac * 0.46))
            e = pygame.Surface((self._W, self._H), pygame.SRCALPHA)
            pygame.draw.ellipse(e, (0, 0, 0, alpha),
                                ((self._W - rw) // 2, (self._H - rh) // 2, rw, rh))
            overlay.blit(e, (0, 0), special_flags=pygame.BLEND_RGBA_MAX)

        return overlay
