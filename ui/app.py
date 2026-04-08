"""
ui/app.py — Full-screen app: face only.

Renders nothing but the PixelFace on a dark CRT background.
No HUD, no labels, no status text on screen.
Keyboard input is captured silently; on Enter it fires on_text_input.
While the user is typing, face pupils look downward (toward keyboard).
"""

import threading
import time
from typing import Callable

import pygame

from config import COLOR_BG, TARGET_FPS, WINDOW_TITLE
from ui.mac_face import PixelFace
from ui.expressions import Expression


class TalkingMACApp:
    """
    Full-screen pygame UI.

    Thread-safe: set_expression / set_mode_label / show_status can be
    called from any thread — they write to lock-protected attributes.
    The pygame render loop runs on the main thread.
    """

    def __init__(self):
        pygame.init()
        pygame.display.set_caption(WINDOW_TITLE)
        pygame.mouse.set_visible(False)

        info = pygame.display.Info()
        self._W = info.current_w
        self._H = info.current_h

        flags = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
        self._screen = pygame.display.set_mode((self._W, self._H), flags)
        self._clock  = pygame.time.Clock()
        self._running = True

        # PixelFace — full screen, sizes itself from screen dimensions
        self._face = PixelFace(self._W, self._H)

        # State (lock-protected for thread safety)
        self._lock       = threading.Lock()
        self._expression = Expression.IDLE
        self._input_buf  = ""
        self._typing     = False   # True while user has pending keystrokes

        # Callbacks
        self.on_text_input: Callable[[str], None] | None = None
        self.on_quit: Callable[[], None] | None = None

        # CRT scanline + vignette overlay (baked once at startup)
        self._overlay = self._make_overlay()

    # ── Public (thread-safe) ─────────────────────────────────────────────────

    def set_expression(self, expr: Expression):
        with self._lock:
            self._expression = expr
        self._face.expression = expr

    def set_look_target(self, dx: float, dy: float):
        self._face.set_look_target(dx, dy)

    # These are kept for API compatibility with main.py but do nothing visual
    def show_status(self, text: str, ttl: float = 5.0):
        pass   # no HUD — intentional

    def set_mode_label(self, label: str):
        pass   # no HUD — intentional

    def is_running(self) -> bool:
        return self._running

    def stop(self):
        self._running = False

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        last = time.time()
        while self._running:
            now = time.time()
            dt  = min(now - last, 0.05)   # cap dt to prevent spiral on lag
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
                    self._typing    = False
                    # Eyes return to forward gaze
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
                            # Eyes look down toward keyboard
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

        # Dark CRT background
        surf.fill(COLOR_BG)

        # Pixel art face — the only thing rendered
        self._face.draw(surf)

        # CRT overlay (scanlines + vignette)
        surf.blit(self._overlay, (0, 0))

        pygame.display.flip()

    def _make_overlay(self) -> pygame.Surface:
        """Bake CRT scanlines + vignette into one surface."""
        overlay = pygame.Surface((self._W, self._H), pygame.SRCALPHA)

        # Scanlines — every 4 pixels, very subtle
        for y in range(0, self._H, 4):
            pygame.draw.line(overlay, (0, 0, 0, 8), (0, y), (self._W, y))

        # Vignette — elliptical darkening from edges (light, lets pixels breathe)
        for i in range(30, 0, -1):
            frac  = i / 30
            alpha = int(frac ** 2.4 * 55)
            rw    = int(self._W * (1 - frac * 0.46))
            rh    = int(self._H * (1 - frac * 0.46))
            e = pygame.Surface((self._W, self._H), pygame.SRCALPHA)
            pygame.draw.ellipse(e, (0, 0, 0, alpha),
                                ((self._W - rw) // 2, (self._H - rh) // 2, rw, rh))
            overlay.blit(e, (0, 0), special_flags=pygame.BLEND_RGBA_MAX)

        return overlay
