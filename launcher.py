# launcher.py
#
# Represents the top-center launcher tube.
#
# Features:
#   - Pivot point attached at top-center of the screen (x = width / 2, y = -15),
#     extending above the screen top so it is clipped by the screen boundary.
#   - Consists ONLY of a short, elongated solid tube/barrel (no base/tank/turret).
#   - Continuous smooth rotation between LAUNCHER_MIN_ANGLE and LAUNCHER_MAX_ANGLE.
#   - Provides collision parameters for PhysicsWorld (_Solid OBB).
#   - Provides ball spawn transform (position + velocity) outward from nozzle exit.

import math

from config import (
    LAUNCHER_MIN_ANGLE,
    LAUNCHER_MAX_ANGLE,
    LAUNCHER_LENGTH,
    LAUNCHER_WIDTH,
    LAUNCHER_ROTATION_SPEED,
    LAUNCHER_BALL_SPAWN_SPEED,
    BALL_RADIUS,
    BALL_RADIUS_VARIATION,
)


class Launcher:
    def __init__(self, screen_width, screen_height):
        self.screen_width = float(screen_width)
        self.screen_height = float(screen_height)

        # Pivot point at top center. Placed slightly above y=0 (-15px) so the upper part
        # extends beyond the top of the screen and is clipped by the screen boundary.
        self.pivot_x = self.screen_width / 2.0
        self.pivot_y = -15.0

        self.min_angle = float(LAUNCHER_MIN_ANGLE)
        self.max_angle = float(LAUNCHER_MAX_ANGLE)
        self.length = float(LAUNCHER_LENGTH)
        self.tube_width = float(LAUNCHER_WIDTH)
        self.rotation_speed = float(LAUNCHER_ROTATION_SPEED)
        self.spawn_speed = float(LAUNCHER_BALL_SPAWN_SPEED)

        self._time = 0.0
        self.angle = 0.0  # degrees from straight down (0=down, negative=left, positive=right)
        self._update_angle()

    def update_screen_size(self, width, height):
        self.screen_width = float(width)
        self.screen_height = float(height)
        self.pivot_x = self.screen_width / 2.0

    def update(self, delta_time):
        self._time += delta_time
        self._update_angle()

    def _update_angle(self):
        mid_angle = (self.min_angle + self.max_angle) / 2.0
        amplitude = (self.max_angle - self.min_angle) / 2.0

        if amplitude <= 0:
            self.angle = mid_angle
            return

        # Smooth harmonic oscillation within [min_angle, max_angle]
        period = max(0.1, (4.0 * amplitude) / max(1.0, self.rotation_speed))
        omega = (2.0 * math.pi) / period

        sine_val = math.sin(omega * self._time)
        subtle_wave = 0.85 * sine_val + 0.15 * math.sin(omega * 0.5 * self._time)

        self.angle = mid_angle + amplitude * max(-1.0, min(1.0, subtle_wave))

    def get_solid_params(self):
        """
        Returns (cx, cy, rect_long, rect_short, angle) for creating a _Solid in physics.
        """
        rad = math.radians(self.angle)
        dir_x = math.sin(rad)
        dir_y = math.cos(rad)

        cx = self.pivot_x + dir_x * (self.length / 2.0)
        cy = self.pivot_y + dir_y * (self.length / 2.0)

        # OBB angle in geometry convention: 90 degrees minus launcher angle
        obb_angle = 90.0 - self.angle

        return cx, cy, self.length, self.tube_width, obb_angle

    def get_spawn_transform(self):
        """
        Returns (spawn_x, spawn_y, dir_x, dir_y) for spawning a ball outward
        along the nozzle direction, positioned slightly outside the nozzle tip.
        """
        rad = math.radians(self.angle)
        dir_x = math.sin(rad)
        dir_y = math.cos(rad)

        max_ball_radius = BALL_RADIUS * (1.0 + BALL_RADIUS_VARIATION)
        safe_dist = self.length + max_ball_radius + 3.0

        spawn_x = self.pivot_x + dir_x * safe_dist
        spawn_y = self.pivot_y + dir_y * safe_dist

        return spawn_x, spawn_y, dir_x, dir_y

    def get_points(self):
        """
        Returns the 4 corner points of the launcher tube rectangle.
        """
        rad = math.radians(self.angle)
        u_x = math.sin(rad)
        u_y = math.cos(rad)

        v_x = -u_y
        v_y = u_x

        half_w = self.tube_width / 2.0
        length = self.length

        p_top_left = (
            self.pivot_x - v_x * half_w,
            self.pivot_y - v_y * half_w,
        )
        p_top_right = (
            self.pivot_x + v_x * half_w,
            self.pivot_y + v_y * half_w,
        )
        p_bot_right = (
            self.pivot_x + u_x * length + v_x * half_w,
            self.pivot_y + u_y * length + v_y * half_w,
        )
        p_bot_left = (
            self.pivot_x + u_x * length - v_x * half_w,
            self.pivot_y + u_y * length - v_y * half_w,
        )

        return [p_top_left, p_top_right, p_bot_right, p_bot_left]
