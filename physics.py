# physics.py
#
# Pure simulation - no pygame, no OpenCV, no drawing. Everything the
# viewer sees (trails, glow, sparks, squash) is derived from this
# module's state and the events it emits, in effects.py.
#
# What is simulated:
#   - balls falling under gravity, with slightly different sizes
#     (bigger = heavier),
#   - bouncing off detected stickers, which are ROTATED rectangles:
#     the bounce uses the real surface normal (flat edge or corner)
#     of the tilted sticker, with restitution (how bouncy) and
#     friction (so balls roll along slopes instead of skating),
#   - balls bouncing off each other,
#   - ALL FOUR screen edges as solid walls - a ball never leaves the
#     screen, it bounces off the top/bottom/left/right forever,
#   - by default none of the above bounces lose any energy
#     (BALL_RESTITUTION = BALL_BALL_RESTITUTION = 1.0, BALL_FRICTION
#     = 0 in config.py), so with gravity a ball just keeps bouncing
#     indefinitely like an ideal rubber ball, instead of gradually
#     settling - lower these if a more "real rubber ball" feel that
#     loses a little height each bounce is wanted instead,
#   - a ball is only ever removed in one of two ways, each
#     independently configurable in config.py and each with its own
#     little pop effect: it reaches BALL_MAX_LIFETIME_SECONDS after
#     being spawned, or it is "stuck" (parked in one spot, usually a
#     pocket between stickers) for BALL_STUCK_TIMEOUT_SECONDS.
#
# Performance notes (this runs 144 times a second):
#   - obstacle dicts are converted ONCE per detection update into
#     lightweight _Solid records with cos/sin already computed,
#   - each ball first does a cheap circle-vs-circle reject against
#     each solid's bounding circle before any real collision math,
#   - time is split into equal slices no longer than
#     PHYSICS_MAX_STEP_SECONDS (and short enough that a fast ball
#     can't skip over a sticker), so motion stays smooth and
#     proportional to real elapsed time.

import math
import random

from geometry import extract_obb
from launcher import Launcher

from config import (
    BALL_RADIUS,
    BALL_RADIUS_VARIATION,
    BALL_GRAVITY,
    BALL_RESTITUTION,
    BALL_REST_SPEED,
    BALL_FRICTION,
    BALL_MAX_SPEED,
    BALL_BALL_COLLISION_ENABLED,
    BALL_BALL_RESTITUTION,
    PHYSICS_MAX_STEP_SECONDS,
    MAX_SIMULTANEOUS_BALLS,
    BALL_STUCK_DISTANCE,
    BALL_STUCK_TIMEOUT_SECONDS,
    BALL_MAX_LIFETIME_SECONDS,
    BALL_HUE_MIN_DEGREES,
    BALL_HUE_MAX_DEGREES,
    IMPACT_EFFECT_MIN_SPEED,
)


# Continuous slow drag (per second) applied to the sliding speed of
# a ball that is resting/rolling ON a surface, so it doesn't glide
# forever on a slope. Impact friction (BALL_FRICTION) is separate.
_ROLLING_DRAG = 0.6

# How long a removed ("popped") ball takes to shrink away.
_DYING_SECONDS = 0.45

_MAX_EVENTS_PER_FRAME = 64


class Ball:
    def __init__(
        self,
        x,
        y,
        radius=15,
        velocity_x=0.0,
        velocity_y=0.0,
        gravity=900.0,
        hue=200.0,
    ):
        self.x = float(x)
        self.y = float(y)

        self.radius = float(radius)

        # Mass grows with area, so a big ball shoves a small one
        # around instead of bouncing off it like an equal.
        self.mass = self.radius * self.radius

        self.velocity_x = float(velocity_x)
        self.velocity_y = float(velocity_y)

        self.gravity = float(gravity)

        self.alive = True

        # True once the ball has fallen down far enough to actually
        # be on screen (y - radius >= 0). Until then the top wall is
        # not enforced, so a ball can still drop in from just above
        # the visible screen the way it always has; from then on,
        # the top wall is a solid ceiling like every other edge.
        self.entered = False

        # Dying balls are on their way out (shrinking away): they
        # no longer collide with anything.
        self.dying = False
        self.fade = 1.0

        self.age = 0.0

        # Purely cosmetic state, owned/updated by effects.py.
        self.hue = float(hue)
        self.trail = []
        self.trail_timer = 0.0
        self.squash = 0.0
        self.squash_nx = 0.0
        self.squash_ny = -1.0

        # Stuck detection.
        self._anchor_x = self.x
        self._anchor_y = self.y
        self._stuck_time = 0.0

    @property
    def speed(self):
        return math.hypot(self.velocity_x, self.velocity_y)


class _Solid:
    """One detected sticker, pre-digested for fast collision tests."""

    __slots__ = (
        "cx",
        "cy",
        "half_long",
        "half_short",
        "cos",
        "sin",
        "bound",
    )

    def __init__(self, cx, cy, rect_long, rect_short, angle):
        theta = math.radians(angle)

        self.cx = cx
        self.cy = cy
        self.half_long = rect_long / 2.0
        self.half_short = rect_short / 2.0
        self.cos = math.cos(theta)
        self.sin = math.sin(theta)

        # Radius of the smallest circle containing the whole
        # rectangle: anything farther away than this (plus a
        # ball's radius) can't possibly touch it.
        self.bound = math.hypot(
            self.half_long,
            self.half_short,
        )


class PhysicsWorld:
    def __init__(self, width, height):
        self.width = float(width)
        self.height = float(height)

        self.launcher = Launcher(width, height)

        self.balls = []

        # Impact/pop/launch events produced by the last update(), consumed
        # by effects.py: (kind, x, y, normal_x, normal_y, speed,
        # ball). kind is "obstacle", "ball", "pop" or "launch".
        self.events = []

        self._solids = []
        self._sticker_solids = []
        self._solid_source = None

        self._random = random.Random()
        self._hue_cursor = self._random.uniform(
            BALL_HUE_MIN_DEGREES,
            BALL_HUE_MAX_DEGREES,
        )

    # ========================================================
    # Balls
    # ========================================================

    def add_ball(self, ball):
        if ball is not None:
            self.balls.append(ball)

    def live_ball_count(self):
        return sum(
            1
            for ball in self.balls
            if ball.alive and not ball.dying
        )

    def _next_hue(self):
        # Golden-ratio hop around the allowed hue range: successive
        # balls always get clearly different colors.
        span = BALL_HUE_MAX_DEGREES - BALL_HUE_MIN_DEGREES

        self._hue_cursor = (
            BALL_HUE_MIN_DEGREES
            + (
                self._hue_cursor
                - BALL_HUE_MIN_DEGREES
                + span * 0.618034
            )
            % span
        )

        return self._hue_cursor

    def spawn_ball(self):
        """
        Spawns a new ball emerging from the launcher nozzle.
        Returns the ball, or None if blocked at nozzle exit or max balls reached.
        """
        if self.live_ball_count() >= MAX_SIMULTANEOUS_BALLS:
            return None

        spawn_x, spawn_y, dir_x, dir_y = self.launcher.get_spawn_transform()

        rng = self._random

        radius = BALL_RADIUS * (
            1.0
            + rng.uniform(
                -BALL_RADIUS_VARIATION,
                BALL_RADIUS_VARIATION,
            )
        )

        min_distance = (radius + BALL_RADIUS * (1.0 + BALL_RADIUS_VARIATION)) * 1.05

        for other in self.balls:
            if not other.alive or other.dying:
                continue

            if math.hypot(spawn_x - other.x, spawn_y - other.y) < min_distance:
                return None

        vel_x = dir_x * self.launcher.spawn_speed
        vel_y = dir_y * self.launcher.spawn_speed

        ball = Ball(
            x=spawn_x,
            y=spawn_y,
            radius=radius,
            velocity_x=vel_x,
            velocity_y=vel_y,
            gravity=BALL_GRAVITY,
            hue=self._next_hue(),
        )
        ball.entered = True

        self.balls.append(ball)

        if len(self.events) < _MAX_EVENTS_PER_FRAME:
            self.events.append(
                (
                    "launch",
                    spawn_x,
                    spawn_y,
                    dir_x,
                    dir_y,
                    self.launcher.spawn_speed,
                    ball,
                )
            )

        return ball

    # ========================================================
    # Obstacles
    # ========================================================

    def set_obstacles(self, obstacles):
        # The vision thread publishes a NEW list object whenever
        # something changed; rebuild sticker solids only then.
        if obstacles is not self._solid_source:
            self._solid_source = obstacles

            solids = []

            for obstacle in obstacles or ():
                obb = extract_obb(obstacle)

                if obb is None:
                    continue

                solids.append(_Solid(*obb))

            self._sticker_solids = solids

        # Combine sticker solids + launcher tube solid
        self._solids = list(self._sticker_solids)
        if self.launcher is not None:
            self._solids.append(_Solid(*self.launcher.get_solid_params()))

    # ========================================================
    # Update
    # ========================================================

    def update(self, delta_time, obstacles=None):
        self.events = []

        if delta_time <= 0:
            return

        delta_time = min(delta_time, 0.05)

        if self.launcher is not None:
            self.launcher.update(delta_time)

        self.set_obstacles(obstacles)

        movers = [
            ball
            for ball in self.balls
            if ball.alive and not ball.dying
        ]

        if movers:
            fastest = max(ball.speed for ball in movers)

            smallest = min(ball.radius for ball in movers)

            # Enough slices that (a) none is longer than the
            # configured maximum and (b) no ball travels more than
            # half its own radius in one slice - which is what
            # guarantees it can't skip through a thin sticker.
            steps = max(
                math.ceil(
                    delta_time / PHYSICS_MAX_STEP_SECONDS
                ),
                math.ceil(
                    fastest * delta_time
                    / (0.5 * smallest)
                ),
                1,
            )

            steps = min(steps, 10)

            step = delta_time / steps

            for _ in range(steps):
                self._step(movers, step)

        self._housekeeping(delta_time)

    def _step(self, movers, h):
        for ball in movers:
            ball.velocity_y += ball.gravity * h

            speed = ball.speed

            if speed > BALL_MAX_SPEED:
                scale = BALL_MAX_SPEED / speed
                ball.velocity_x *= scale
                ball.velocity_y *= scale

            ball.x += ball.velocity_x * h
            ball.y += ball.velocity_y * h

        if BALL_BALL_COLLISION_ENABLED and len(movers) > 1:
            self._collide_balls(movers)

        for ball in movers:
            # Up to 3 passes so a ball squeezed between two
            # stickers (or a sticker and another ball's push) ends
            # up somewhere consistent instead of half inside one.
            for _ in range(3):
                touched = False

                for solid in self._solids:
                    if self._collide_solid(ball, solid, h):
                        touched = True

                if not touched:
                    break

            self._keep_inside_walls(ball)

    # ========================================================
    # Ball vs. sticker (oriented rectangle)
    # ========================================================

    def _collide_solid(self, ball, solid, h):
        radius = ball.radius

        dx = ball.x - solid.cx
        dy = ball.y - solid.cy

        reach = solid.bound + radius

        # Cheap reject: outside the sticker's bounding circle.
        if dx * dx + dy * dy > reach * reach:
            return False

        cos_t = solid.cos
        sin_t = solid.sin

        # Ball center in the sticker's own un-rotated frame, where
        # this is plain circle-vs-axis-aligned-box.
        local_x = dx * cos_t + dy * sin_t
        local_y = -dx * sin_t + dy * cos_t

        half_long = solid.half_long
        half_short = solid.half_short

        closest_x = max(-half_long, min(local_x, half_long))
        closest_y = max(-half_short, min(local_y, half_short))

        diff_x = local_x - closest_x
        diff_y = local_y - closest_y

        distance_sq = diff_x * diff_x + diff_y * diff_y

        if distance_sq > radius * radius:
            return False

        if distance_sq > 1e-9:
            # Center outside the box: (diff) already points from the
            # nearest surface point to the ball - the true surface
            # normal for both a flat face hit and a corner hit.
            distance = math.sqrt(distance_sq)

            normal_x = diff_x / distance
            normal_y = diff_y / distance

            penetration = radius - distance

            local_x += normal_x * penetration
            local_y += normal_y * penetration
        else:
            # Center already inside the box (very fast ball / thin
            # sticker): leave through whichever side is nearest.
            overlap_x = half_long + radius - abs(local_x)
            overlap_y = half_short + radius - abs(local_y)

            if overlap_x < overlap_y:
                normal_x = 1.0 if local_x >= 0 else -1.0
                normal_y = 0.0
                local_x = normal_x * (half_long + radius)
            else:
                normal_x = 0.0
                normal_y = 1.0 if local_y >= 0 else -1.0
                local_y = normal_y * (half_short + radius)

        ball.x = solid.cx + local_x * cos_t - local_y * sin_t
        ball.y = solid.cy + local_x * sin_t + local_y * cos_t

        local_vx = (
            ball.velocity_x * cos_t
            + ball.velocity_y * sin_t
        )
        local_vy = (
            -ball.velocity_x * sin_t
            + ball.velocity_y * cos_t
        )

        normal_speed = local_vx * normal_x + local_vy * normal_y

        # Only react if the ball is actually moving INTO the
        # surface. (Reflecting a ball that is already leaving is
        # what makes bounces flip-flop and jitter.)
        if normal_speed < 0:
            impact = -normal_speed

            tangent_x = -normal_y
            tangent_y = normal_x

            tangent_speed = (
                local_vx * tangent_x + local_vy * tangent_y
            )

            if impact >= BALL_REST_SPEED:
                bounce = BALL_RESTITUTION
                tangent_speed *= 1.0 - BALL_FRICTION
            else:
                # Gentle contact = resting/rolling: no bounce, just
                # a little rolling resistance.
                bounce = 0.0
                tangent_speed *= max(
                    0.0,
                    1.0 - _ROLLING_DRAG * h,
                )

            new_normal_speed = -normal_speed * bounce

            local_vx = (
                new_normal_speed * normal_x
                + tangent_speed * tangent_x
            )
            local_vy = (
                new_normal_speed * normal_y
                + tangent_speed * tangent_y
            )

            ball.velocity_x = local_vx * cos_t - local_vy * sin_t
            ball.velocity_y = local_vx * sin_t + local_vy * cos_t

            if (
                impact >= IMPACT_EFFECT_MIN_SPEED
                and len(self.events) < _MAX_EVENTS_PER_FRAME
            ):
                world_nx = normal_x * cos_t - normal_y * sin_t
                world_ny = normal_x * sin_t + normal_y * cos_t

                self.events.append(
                    (
                        "obstacle",
                        ball.x - world_nx * ball.radius,
                        ball.y - world_ny * ball.radius,
                        world_nx,
                        world_ny,
                        impact,
                        ball,
                    )
                )

        return True

    # ========================================================
    # Ball vs. ball
    # ========================================================

    def _collide_balls(self, movers):
        count = len(movers)

        for i in range(count - 1):
            first = movers[i]

            for j in range(i + 1, count):
                second = movers[j]

                reach = first.radius + second.radius

                dx = second.x - first.x
                dy = second.y - first.y

                if abs(dx) >= reach or abs(dy) >= reach:
                    continue

                distance_sq = dx * dx + dy * dy

                if distance_sq >= reach * reach:
                    continue

                distance = math.sqrt(distance_sq)

                if distance > 1e-6:
                    normal_x = dx / distance
                    normal_y = dy / distance
                else:
                    normal_x = 0.0
                    normal_y = 1.0

                total_mass = first.mass + second.mass

                # Push apart, the lighter ball moving more.
                penetration = reach - distance

                first_share = penetration * (second.mass / total_mass)
                second_share = penetration * (first.mass / total_mass)

                first.x -= normal_x * first_share
                first.y -= normal_y * first_share
                second.x += normal_x * second_share
                second.y += normal_y * second_share

                closing = (
                    (second.velocity_x - first.velocity_x) * normal_x
                    + (second.velocity_y - first.velocity_y) * normal_y
                )

                if closing >= 0:
                    continue

                impact = -closing

                bounce = (
                    BALL_BALL_RESTITUTION
                    if impact >= BALL_REST_SPEED
                    else 0.0
                )

                impulse = (
                    -(1.0 + bounce)
                    * closing
                    / (1.0 / first.mass + 1.0 / second.mass)
                )

                first.velocity_x -= impulse / first.mass * normal_x
                first.velocity_y -= impulse / first.mass * normal_y
                second.velocity_x += impulse / second.mass * normal_x
                second.velocity_y += impulse / second.mass * normal_y

                if (
                    impact >= IMPACT_EFFECT_MIN_SPEED
                    and len(self.events) < _MAX_EVENTS_PER_FRAME
                ):
                    self.events.append(
                        (
                            "ball",
                            first.x + normal_x * first.radius,
                            first.y + normal_y * first.radius,
                            normal_x,
                            normal_y,
                            impact,
                            first,
                        )
                    )

    # ========================================================
    # Walls
    # ========================================================
    #
    # All four screen edges are solid: a ball bounces off every one
    # of them forever and never leaves the screen (the only ways a
    # ball ever disappears are the lifetime/stuck-timeout removal in
    # _housekeeping below, each with its own little pop effect).
    # With BALL_RESTITUTION left at its lossless default (1.0) these
    # bounces never bleed off energy, so together with gravity a
    # ball just keeps bouncing indefinitely, exactly like a perfect
    # rubber ball in a box, until its time is up.

    def _keep_inside_walls(self, ball):
        radius = ball.radius

        if ball.x - radius < 0:
            ball.x = radius

            if ball.velocity_x < 0:
                ball.velocity_x *= -BALL_RESTITUTION

        elif ball.x + radius > self.width:
            ball.x = self.width - radius

            if ball.velocity_x > 0:
                ball.velocity_x *= -BALL_RESTITUTION

        if not ball.entered:
            if ball.y - radius < 0:
                # Still on its way in from above the visible
                # screen - leave the top wall unenforced for now.
                return

            ball.entered = True

        if ball.y - radius < 0:
            ball.y = radius

            if ball.velocity_y < 0:
                ball.velocity_y *= -BALL_RESTITUTION

        elif ball.y + radius > self.height:
            ball.y = self.height - radius

            if ball.velocity_y > 0:
                ball.velocity_y *= -BALL_RESTITUTION

    # ========================================================
    # Lifetime, stuck balls, removal
    # ========================================================

    def _housekeeping(self, delta_time):
        for ball in self.balls:
            if not ball.alive:
                continue

            if ball.dying:
                ball.fade -= delta_time / _DYING_SECONDS

                if ball.fade <= 0.0:
                    ball.alive = False

                continue

            ball.age += delta_time

            if BALL_STUCK_TIMEOUT_SECONDS > 0:
                moved = math.hypot(
                    ball.x - ball._anchor_x,
                    ball.y - ball._anchor_y,
                )

                if moved > BALL_STUCK_DISTANCE:
                    ball._anchor_x = ball.x
                    ball._anchor_y = ball.y
                    ball._stuck_time = 0.0
                else:
                    ball._stuck_time += delta_time

                if ball._stuck_time >= BALL_STUCK_TIMEOUT_SECONDS:
                    self._start_dying(ball)
                    continue

            if (
                BALL_MAX_LIFETIME_SECONDS > 0
                and ball.age >= BALL_MAX_LIFETIME_SECONDS
            ):
                self._start_dying(ball)

        self.balls = [
            ball for ball in self.balls if ball.alive
        ]

    def _start_dying(self, ball):
        ball.dying = True
        ball.fade = 1.0

        if len(self.events) < _MAX_EVENTS_PER_FRAME:
            self.events.append(
                (
                    "pop",
                    ball.x,
                    ball.y,
                    0.0,
                    -1.0,
                    0.0,
                    ball,
                )
            )

    # ========================================================
    # Reset
    # ========================================================

    def clear(self):
        self.balls = []
        self.events = []

    def reset(self):
        self.clear()
