"""
ui/expressions.py — Expression states: pixel bitmap grids + colours.

All timing constants kept from original.  Bezier/ellipse descriptors
replaced with 2-D integer grids (0 = off, 1 = iris, 2 = pupil).
"""

import math
from enum import Enum, auto
from config import (
    COLOR_FACE_IDLE, COLOR_FACE_LISTEN, COLOR_FACE_THINK,
    COLOR_FACE_TALK, COLOR_FACE_HAPPY, COLOR_FACE_ERROR, COLOR_FACE_SLEEP,
)


class Expression(Enum):
    IDLE      = auto()
    LISTENING = auto()
    THINKING  = auto()
    TALKING   = auto()
    HAPPY     = auto()
    ERROR     = auto()
    SLEEPING  = auto()


# ── Expression → foreground colour ───────────────────────────────────────────
EXPRESSION_COLORS: dict[Expression, tuple] = {
    Expression.IDLE:      COLOR_FACE_IDLE,
    Expression.LISTENING: COLOR_FACE_LISTEN,
    Expression.THINKING:  COLOR_FACE_THINK,
    Expression.TALKING:   COLOR_FACE_TALK,
    Expression.HAPPY:     COLOR_FACE_HAPPY,
    Expression.ERROR:     COLOR_FACE_ERROR,
    Expression.SLEEPING:  COLOR_FACE_SLEEP,
}

# Glow pulse speed (radians/sec) per expression
GLOW_PULSE_SPEED: dict[Expression, float] = {
    Expression.IDLE:      0.8,
    Expression.LISTENING: 1.4,
    Expression.THINKING:  1.0,
    Expression.TALKING:   1.8,
    Expression.HAPPY:     1.0,
    Expression.ERROR:     3.5,
    Expression.SLEEPING:  0.3,
}

# ── Upper-face grids (10 cols × 6 rows) ──────────────────────────────────────
# 0 = off, 1 = on
# Each grid covers BOTH eyes + nose in one block, drawn once centred on the face.
# Rows 0–2 = eye region (blink-able); rows 3–5 = nose region (always visible).

EYE_GRIDS: dict[Expression, list[list[int]]] = {

    # Vertical slit eyes + L-shaped Kare nose
    Expression.IDLE: [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    ],

    # Eyes dropped to rows 2–3 (looking down), nose stays
    Expression.LISTENING: [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    ],

    # Eyes shifted right (looking up-right), nose stays
    Expression.THINKING: [
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    ],

    # Same as IDLE — eyes open normally during talking
    Expression.TALKING: [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    ],

    # Squinting eyes (corners curling out), nose stays
    Expression.HAPPY: [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    ],

    # Cross-eyed (both slits pointing inward), no nose
    Expression.ERROR: [
        [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    ],

    # Flat closed-eye lines, no nose
    Expression.SLEEPING: [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    ],
}

# ── Mouth grids (12 cols × 4 rows) ───────────────────────────────────────────
# Each expression is a LIST of frames (for animation).
# TALKING has 8 frames; others have 1.
# All mouths are compact, centred around cols 4–8.

# Subtle Kare smile — corners at cols 4,8; flat bottom arc cols 5-7
_M_IDLE = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

# Wider, deeper smile — corners at cols 3,9
_M_HAPPY = [
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

# Inverted frown — narrow, centred
_M_ERROR = [
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

# Thin flat line — 5-pixel width cols 4-8
_M_THINK = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

# Tiny 3-pixel line — barely awake
_M_SLEEP = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

# ── Talking frames — narrow, restricted to cols 4-8 ───────────────────────────
# T0: closed (same as idle)
_M_T0 = _M_IDLE

# T1: lips just parting — tiny slot
_M_T1 = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
]

# T2: clearly open
_M_T2 = [
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
]

# T3: wide open (AH)
_M_T3 = [
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
]

# T4: O-shape (OO)
_M_T4 = [
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
]

# T5: closing wide
_M_T5 = _M_T3

# T6: closing open
_M_T6 = _M_T2

# T7: closed
_M_T7 = _M_IDLE

MOUTH_GRIDS: dict[Expression, list] = {
    Expression.IDLE:      [_M_IDLE],
    Expression.LISTENING: [_M_IDLE],
    Expression.THINKING:  [_M_THINK],
    Expression.HAPPY:     [_M_HAPPY],
    Expression.ERROR:     [_M_ERROR],
    Expression.SLEEPING:  [_M_SLEEP],
    Expression.TALKING:   [_M_T0, _M_T1, _M_T2, _M_T3, _M_T4, _M_T5, _M_T6, _M_T7],
}

# ── Timing constants (unchanged from original) ────────────────────────────────
BLINK_INTERVAL_MIN  = 2.5    # seconds
BLINK_INTERVAL_MAX  = 6.0
BLINK_DURATION      = 0.14   # seconds
TALK_FRAME_RATE     = 7.0    # mouth shape changes per second
EXPR_BLEND_SPEED    = 3.5    # blend units per second (1/speed = transition time)
LOOK_LERP_SPEED     = 2.5    # units per second for eye look-at tracking
COLOR_LERP_SPEED    = 3.0    # RGB lerp units per second on expression change
THINK_SWAY_SPEED    = 1.4    # radians/sec
THINK_SWAY_PX       = 18     # max horizontal sway in pixels


# ── Helpers ───────────────────────────────────────────────────────────────────

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def lerp_color(c1: tuple, c2: tuple, t: float) -> tuple:
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))

def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)

def pulse(t: float, speed: float = 1.0) -> float:
    return (math.sin(t * speed * math.pi * 2) + 1) * 0.5
