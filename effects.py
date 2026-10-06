# effects.py
#
# Everything cosmetic that reacts to the simulation: comet tails,
# colour changes on bounce, squash-and-stretch, sparks and rings.
# Pure state + math (no pygame, no drawing) - rendering.py draws it.
# None of it ever feeds back into physics.
#
# Driven by two things only: each ball's own state (position,
# velocity) and the events physics.py emits (kind, x, y, normal_x,
# normal_y, speed, ball) with kind "obstacle", "ball" or "pop".
#
# Camera safety: rendering only ever draws these with saturated
# colours in small dim glows, so the camera looking at this same
# monitor never mistakes an effect for a white sticker. Set
# VISUAL_EFFECTS_ENABLED = False in config.py to switch it all off.

import math
import random

from config import (
    VISUAL_EFFECTS_ENABLED,
    BALL_HUE_MIN_DEGREES,
    BALL_HUE_MAX_DEGREES,
    BALL_HUE_SHIFT_ON_BOUNCE,
    BALL_TRAIL_LENGTH,
    BALL_TRAIL_SAMPLE_SECONDS,
    BALL_SQUASH_DURATION_SECONDS,
    IMPACT_EFFECT_MIN_SPEED,
    MAX_PARTICLES,
)


_PARTICLE_GRAVITY = 520.0
_PARTICLE_DRAG = 2.2
_MAX_RINGS = 24

# Impact speed (px/s) at which effects reach full strength.
_FULL_STRENGTH_SPEED = 1400.0


class Effects:
    def __init__(self):
        # Sparks: [x, y, vx, vy, age, life, hue, size]
        self.particles = []

        # Expanding rings: [x, y, age, life, hue, max_radius]
        self.rings = []

        self._random = random.Random()

    def clear(self):
        self.particles = []
        self.rings = []

    # ========================================================
    # Update
    # ========================================================

    def update(self, delta_time, balls, events):
        if not VISUAL_EFFECTS_ENABLED:
            return

        delta_time = max(0.0, min(delta_time, 0.05))

        for event in events:
            self._on_event(event)

        for ball in balls:
            self._update_ball(ball, delta_time)

        self._update_particles(delta_time)
        self._update_rings(delta_time)

    # ========================================================
    # Per-ball state
    # ========================================================

    def _update_ball(self, ball, dt):
        if ball.squash > 0.0:
            ball.squash = max(
                0.0,
                ball.squash
                - dt / max(BALL_SQUASH_DURATION_SECONDS, 1e-3),
            )

        if BALL_TRAIL_LENGTH <= 0:
            ball.trail = []
            return

        ball.trail_timer += dt

        # More than one sample can be due on a slow frame; keep the
        # tail evenly spaced in TIME regardless of frame rate.
        while ball.trail_timer >= BALL_TRAIL_SAMPLE_SECONDS:
            ball.trail_timer -= BALL_TRAIL_SAMPLE_SECONDS

            if ball.dying:
                # A popping ball's tail drains away.
                if ball.trail:
                    ball.trail.pop(0)
            else:
                ball.trail.append((ball.x, ball.y))

        excess = len(ball.trail) - BALL_TRAIL_LENGTH

        if excess > 0:
            del ball.trail[:excess]

    @staticmethod
    def _shift_hue(hue, amount):
        span = BALL_HUE_MAX_DEGREES - BALL_HUE_MIN_DEGREES

        return (
            BALL_HUE_MIN_DEGREES
            + (hue - BALL_HUE_MIN_DEGREES + amount) % span
        )

    # ========================================================
    # Events -> sparks, rings, squash, colour change
    # ========================================================

    def _on_event(self, event):
        kind, x, y, nx, ny, speed, ball = event

        if kind == "pop":
            self._burst(x, y, ball.hue, 16, 80.0, 260.0)
            self._ring(x, y, ball.hue, ball.radius * 3.0, 0.45)
            return

        strength = min(1.0, speed / _FULL_STRENGTH_SPEED)

        ball.squash = 1.0
        ball.squash_nx = nx
        ball.squash_ny = ny

        if kind == "obstacle":
            ball.hue = self._shift_hue(
                ball.hue, BALL_HUE_SHIFT_ON_BOUNCE
            )

            self._sparks(
                x,
                y,
                nx,
                ny,
                ball.hue,
                int(4 + 10 * strength),
                strength,
            )

            self._ring(
                x,
                y,
                ball.hue,
                ball.radius * (1.6 + 1.6 * strength),
                0.32,
            )
        else:
            self._burst(
                x,
                y,
                ball.hue,
                int(3 + 6 * strength),
                60.0,
                140.0 + 200.0 * strength,
            )

            self._ring(
                x,
                y,
                ball.hue,
                ball.radius * (1.3 + 1.0 * strength),
                0.25,
            )

    def _sparks(self, x, y, nx, ny, hue, count, strength):
        rng = self._random

        base = math.atan2(ny, nx)

        for _ in range(count):
            angle = base + rng.uniform(-1.25, 1.25)

            speed = rng.uniform(
                140.0,
                260.0 + 420.0 * strength,
            )

            self._add_particle(
                x,
                y,
                math.cos(angle) * speed,
                math.sin(angle) * speed,
                rng.uniform(0.35, 0.8),
                hue + rng.uniform(-14.0, 14.0),
                rng.uniform(2.5, 5.5),
            )

    def _burst(self, x, y, hue, count, speed_min, speed_max):
        rng = self._random

        for _ in range(count):
            angle = rng.uniform(0.0, 2.0 * math.pi)
            speed = rng.uniform(speed_min, speed_max)

            self._add_particle(
                x,
                y,
                math.cos(angle) * speed,
                math.sin(angle) * speed,
                rng.uniform(0.3, 0.7),
                hue + rng.uniform(-18.0, 18.0),
                rng.uniform(2.5, 5.0),
            )

    def _add_particle(self, x, y, vx, vy, life, hue, size):
        if len(self.particles) >= MAX_PARTICLES:
            # Drop the oldest so a busy moment never grows without
            # bound.
            del self.particles[0]

        self.particles.append(
            [x, y, vx, vy, 0.0, life, hue, size]
        )

    def _ring(self, x, y, hue, max_radius, life):
        if len(self.rings) >= _MAX_RINGS:
            del self.rings[0]

        self.rings.append([x, y, 0.0, life, hue, max_radius])

    # ========================================================
    # Ageing
    # ========================================================

    def _update_particles(self, dt):
        alive = []

        drag = max(0.0, 1.0 - _PARTICLE_DRAG * dt)

        for particle in self.particles:
            particle[4] += dt

            if particle[4] >= particle[5]:
                continue

            particle[3] += _PARTICLE_GRAVITY * dt
            particle[2] *= drag
            particle[3] *= drag
            particle[0] += particle[2] * dt
            particle[1] += particle[3] * dt

            alive.append(particle)

        self.particles = alive

    def _update_rings(self, dt):
        alive = []

        for ring in self.rings:
            ring[2] += dt

            if ring[2] < ring[3]:
                alive.append(ring)

        self.rings = alive
