# rendering.py
#
# Draws the game onto the monitor.
#
# Speed: with USE_DIRTY_RECTS the screen is split into a grid of
# small cells. Only cells where something was drawn last frame or is
# drawn now get repainted (background, then the sticker patches in
# them, then the effects in them) and only those cells are pushed to
# the display. A few balls touch a few dozen cells instead of all
# 2 million pixels. The calibration border and black interior are
# pre-rendered ONCE into a background surface and just copied in.
#
# Camera safety: the camera watches this very monitor for WHITE
# stickers. So everything here except the required sticker patch is
# drawn in saturated colour, with dim additive glows (never near
# white) and only tiny pastel highlights. The sticker patch is drawn
# slightly smaller than the sticker (see geometry.paint_dimensions)
# and the detector blanks it out, so a removed sticker can't be
# "seen" in its own leftover paint.

import colorsys
import math
import os
import time

import pygame

from geometry import (
    obb_corners,
    extract_obb,
    paint_dimensions,
)

from config import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    BACKGROUND_COLOR,
    CALIBRATION_PATTERN_OUTER_COLOR,
    CALIBRATION_PATTERN_INNER_COLOR,
    CALIBRATION_OUTER_BORDER_SIZE,
    CALIBRATION_INNER_BORDER_SIZE,
    TARGET_FPS,
    USE_DIRTY_RECTS,
    FULL_REPAINT_SECONDS,
    VISUAL_EFFECTS_ENABLED,
    BALL_GLOW_ENABLED,
    BALL_SQUASH_STRETCH_ENABLED,
    SHOW_DETECTED_OBSTACLES,
    DETECTED_OBSTACLE_COLOR,
    DETECTED_OBSTACLE_BORDER_WIDTH,
    DETECTED_OBSTACLE_PAINT_INSET_RATIO,
)


# Side length (px) of one dirty-repaint cell.
_CELL = 96

# Additive glow strengths (fraction of the full hue colour at the
# glow's centre). Deliberately dim: see "camera safety" above.
_GLOW_LEVELS = (0.16, 0.30, 0.48, 0.66)

# Layers, drawn in this order inside every repainted cell.
_LAYER_GLOW = 0
_LAYER_RING = 1
_LAYER_BODY = 2


class Renderer:
    # ========================================================
    # Initialization
    # ========================================================

    def __init__(self):
        # Ask SDL to place the (borderless) window at the top-left
        # corner so it lines up exactly with the screen.
        os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "0,0")

        pygame.init()

        pygame.display.set_caption("Monitor Ball")

        # A borderless window sized to the full screen ("fake
        # fullscreen"/kiosk mode) instead of pygame.FULLSCREEN.
        # Exclusive SDL fullscreen needs a real videofullscreen-mode switch,
        # which is unreliable on several X11/Wayland/XWayland driver
        # combinations (blank frames, crashes on exit).
        self.screen = pygame.display.set_mode(
            (SCREEN_WIDTH, SCREEN_HEIGHT),
            pygame.NOFRAME,
        )

        # pygame.mouse.set_visible(False)

        # try:
        #     pygame.event.set_grab(True)
        # except Exception:
        #     pass

        # SDL2 turns "text input mode" on by default, which on
        # systems with an active IME can swallow certain keys
        # before they reach us as a normal KEYDOWN.
        try:
            pygame.key.stop_text_input()
        except Exception:
            pass

        self.clock = pygame.time.Clock()

        self._screen_rect = pygame.Rect(
            0, 0, SCREEN_WIDTH, SCREEN_HEIGHT
        )

        self._cols = (SCREEN_WIDTH + _CELL - 1) // _CELL
        self._rows = (SCREEN_HEIGHT + _CELL - 1) // _CELL

        self._background = self._build_background()

        self._mode = None
        self._need_full = True
        self._last_full = 0.0

        # Sticker patches, rebuilt only when the detector publishes
        # a new list.
        self._obstacle_source = None
        self._obstacle_polys = []
        self._obstacle_cells = {}
        self._obstacle_dirty = set()

        self._prev_cells = set()

        self._glow_cache = {}
        self._color_cache = {}

    def _build_background(self):
        surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        surface.fill(BACKGROUND_COLOR)
        return surface.convert()

    # ========================================================
    # Calibration screens
    # ========================================================

    def render_calibration(self):
        """
        Solid pink over the whole screen: the simplest possible
        target for the camera's first lock-on. It never changes, so
        it is painted once (and again every FULL_REPAINT_SECONDS as
        a safety net) instead of 144 times a second.
        """
        now = time.monotonic()

        if (
            self._mode != "calibration"
            or now - self._last_full >= FULL_REPAINT_SECONDS
        ):
            self.draw_full_calibration_screen()

            pygame.display.flip()

            self._mode = "calibration"
            self._last_full = now

        self._need_full = True
        self._prev_cells = set()

    def draw_full_calibration_screen(self):
        self.screen.set_clip(None)

        self.screen.fill(CALIBRATION_PATTERN_OUTER_COLOR)

    def draw_calibration_border(self):
        # Border ring + black interior (pre-rendered).
        self.screen.set_clip(None)

        self.screen.blit(self._background, (0, 0))

    def force_full_repaint(self):
        self._need_full = True

    # ========================================================
    # Main rendering
    # ========================================================

    def render(self, balls, obstacles, effects=None, launcher=None):
        now = time.monotonic()

        self._sync_obstacles(obstacles)

        items = self._collect_items(balls, effects, launcher)

        cell_items = {}

        for layer, bounds, draw in items:
            self._add_to_cells(
                cell_items,
                bounds,
                (layer, draw),
            )

        full = (
            not USE_DIRTY_RECTS
            or self._need_full
            or self._mode != "game"
            or now - self._last_full >= FULL_REPAINT_SECONDS
        )

        if full:
            self._paint_full(cell_items, now)
        else:
            self._paint_dirty(cell_items)

        self._prev_cells = set(cell_items)

    def _paint_full(self, cell_items, now):
        screen = self.screen

        screen.set_clip(None)

        screen.blit(self._background, (0, 0))

        for poly in self._obstacle_polys:
            self._draw_obstacle(poly)

        everything = []

        for entries in cell_items.values():
            everything.extend(entries)

        # An item spanning several cells appears once per cell;
        # drawing it once per cell is wasteful when nothing is
        # clipped, so de-duplicate by identity.
        seen = set()
        ordered = []

        for entry in everything:
            key = id(entry[1])

            if key in seen:
                continue

            seen.add(key)
            ordered.append(entry)

        ordered.sort(key=lambda entry: entry[0])

        for _layer, draw in ordered:
            self._draw_item(draw)

        pygame.display.flip()

        self._need_full = False
        self._mode = "game"
        self._last_full = now
        self._obstacle_dirty = set()

    def _paint_dirty(self, cell_items):
        screen = self.screen
        background = self._background

        dirty = (
            set(cell_items)
            | self._prev_cells
            | self._obstacle_dirty
        )

        self._obstacle_dirty = set()

        rects = []

        for cell in dirty:
            rect = self._cell_rect(cell)

            if rect.width <= 0 or rect.height <= 0:
                continue

            rects.append(rect)

            screen.set_clip(rect)

            screen.blit(background, rect.topleft, rect)

            for index in self._obstacle_cells.get(cell, ()):
                self._draw_obstacle(self._obstacle_polys[index])

            entries = cell_items.get(cell)

            if entries:
                entries.sort(key=lambda entry: entry[0])

                for _layer, draw in entries:
                    self._draw_item(draw)

        screen.set_clip(None)

        if rects:
            pygame.display.update(rects)

    # ========================================================
    # Grid helpers
    # ========================================================

    def _cell_rect(self, cell):
        column, row = cell

        return pygame.Rect(
            column * _CELL,
            row * _CELL,
            _CELL,
            _CELL,
        ).clip(self._screen_rect)

    def _cells_for(self, rect):
        left = max(0, rect.left // _CELL)
        top = max(0, rect.top // _CELL)

        right = min(self._cols - 1, (rect.right - 1) // _CELL)
        bottom = min(self._rows - 1, (rect.bottom - 1) // _CELL)

        if rect.right <= 0 or rect.bottom <= 0:
            return ()

        if rect.left >= SCREEN_WIDTH or rect.top >= SCREEN_HEIGHT:
            return ()

        return [
            (column, row)
            for row in range(top, bottom + 1)
            for column in range(left, right + 1)
        ]

    def _add_to_cells(self, cell_items, bounds, entry):
        for cell in self._cells_for(bounds):
            cell_items.setdefault(cell, []).append(entry)

    # ========================================================
    # Sticker patches
    # ========================================================

    def _sync_obstacles(self, obstacles):
        if obstacles is self._obstacle_source:
            return

        self._obstacle_source = obstacles

        # Cells that used to show a patch must be repainted, as
        # must cells that now do.
        self._obstacle_dirty |= set(self._obstacle_cells)

        polys = []
        cells = {}

        if SHOW_DETECTED_OBSTACLES:
            for obstacle in obstacles or ():
                if not obstacle.get("painted", True):
                    continue

                obb = extract_obb(obstacle)

                if obb is None:
                    continue

                cx, cy, long_side, short_side, angle = obb

                long_side, short_side = paint_dimensions(
                    long_side,
                    short_side,
                    DETECTED_OBSTACLE_PAINT_INSET_RATIO,
                )

                if long_side <= 0 or short_side <= 0:
                    continue

                corners = obb_corners(
                    cx,
                    cy,
                    long_side,
                    short_side,
                    angle,
                )

                points = [
                    (int(round(px)), int(round(py)))
                    for px, py in corners
                ]

                xs = [p[0] for p in points]
                ys = [p[1] for p in points]

                bounds = pygame.Rect(
                    min(xs) - 2,
                    min(ys) - 2,
                    max(xs) - min(xs) + 4,
                    max(ys) - min(ys) + 4,
                )

                index = len(polys)

                polys.append(points)

                for cell in self._cells_for(bounds):
                    cells.setdefault(cell, []).append(index)

        self._obstacle_polys = polys
        self._obstacle_cells = cells

        self._obstacle_dirty |= set(cells)

    def _draw_obstacle(self, points):
        if DETECTED_OBSTACLE_BORDER_WIDTH <= 0:
            pygame.draw.polygon(
                self.screen,
                DETECTED_OBSTACLE_COLOR,
                points,
            )
        else:
            pygame.draw.polygon(
                self.screen,
                DETECTED_OBSTACLE_COLOR,
                points,
                int(DETECTED_OBSTACLE_BORDER_WIDTH),
            )

    # ========================================================
    # Collecting what to draw this frame
    # ========================================================
    #
    # Everything is turned into a flat list of
    # (layer, bounding_rect, draw_tuple) ONCE per frame; the
    # per-cell painting then only replays those tuples.

    def _collect_items(self, balls, effects, launcher=None):
        items = []

        if launcher is not None:
            self._collect_launcher(items, launcher)

        effects_on = (
            VISUAL_EFFECTS_ENABLED and effects is not None
        )

        if effects_on:
            for particle in effects.particles:
                self._collect_particle(items, particle)

            for ring in effects.rings:
                self._collect_ring(items, ring)

        for ball in balls or ():
            if ball.alive:
                self._collect_ball(items, ball, effects_on)

        return items

    def _collect_launcher(self, items, launcher):
        points = launcher.get_points()

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        bounds = pygame.Rect(
            int(min(xs)) - 3,
            int(min(ys)) - 3,
            int(max(xs) - min(xs)) + 6,
            int(max(ys) - min(ys)) + 6,
        )

        items.append((_LAYER_BODY, bounds, ("launcher", launcher)))

    # ---- particles / rings -------------------------------------

    def _collect_particle(self, items, particle):
        x, y, _vx, _vy, age, life, hue, size = particle

        fraction = 1.0 - age / life

        level = min(
            len(_GLOW_LEVELS) - 1,
            int(fraction * len(_GLOW_LEVELS)),
        )

        radius = max(2, int(round(size * (0.6 + 0.6 * fraction))) + 2)

        sprite = self._glow(hue, radius, level)

        position = (int(x) - radius, int(y) - radius)

        bounds = pygame.Rect(
            position[0],
            position[1],
            radius * 2,
            radius * 2,
        )

        items.append(
            (
                _LAYER_GLOW,
                bounds,
                ("sprite", sprite, position),
            )
        )

    def _collect_ring(self, items, ring):
        x, y, age, life, hue, max_radius = ring

        fraction = age / life

        radius = max(2, int(max_radius * (0.25 + 0.75 * fraction)))

        fade = (1.0 - fraction) ** 1.5

        color = self._hue_rgb(hue, 0.85, 0.9 * fade)

        width = max(1, int(3 * (1.0 - fraction)) + 1)

        bounds = pygame.Rect(
            int(x) - radius - 2,
            int(y) - radius - 2,
            radius * 2 + 4,
            radius * 2 + 4,
        )

        items.append(
            (
                _LAYER_RING,
                bounds,
                (
                    "ring",
                    color,
                    (int(x), int(y)),
                    radius,
                    width,
                ),
            )
        )

    # ---- balls --------------------------------------------------

    def _collect_ball(self, items, ball, effects_on):
        fade = max(0.0, ball.fade)

        radius = ball.radius * fade

        if radius < 1.0:
            return

        if effects_on:
            self._collect_trail(items, ball, radius)

            if BALL_GLOW_ENABLED:
                self._collect_glow(items, ball, radius, fade)

        points, angle_bounds = self._ball_polygon(
            ball,
            radius,
            effects_on,
        )

        body = self._hue_rgb(ball.hue, 0.85, 1.0)
        rim = self._hue_rgb(ball.hue, 0.95, 0.72)

        # A small pastel highlight (saturated enough that the
        # camera can never count it as "white").
        highlight_color = self._hue_rgb(ball.hue, 0.42, 1.0)

        highlight_radius = max(1, int(round(radius * 0.24)))

        highlight_position = (
            int(round(ball.x - radius * 0.32)),
            int(round(ball.y - radius * 0.34)),
        )

        items.append(
            (
                _LAYER_BODY,
                angle_bounds,
                (
                    "body",
                    points,
                    body,
                    rim,
                    highlight_position,
                    highlight_radius,
                    highlight_color,
                ),
            )
        )

    def _ball_polygon(self, ball, radius, effects_on):
        # Ellipse approximated by a polygon: stretched along the
        # direction of travel with speed, flattened against the
        # surface for a moment after a hit.
        semi_a = radius
        semi_b = radius
        angle = 0.0

        if effects_on and BALL_SQUASH_STRETCH_ENABLED:
            speed = ball.speed

            if speed > 30.0:
                stretch = min(0.35, speed / 4000.0)

                semi_a = radius * (1.0 + stretch)
                semi_b = radius / (1.0 + stretch * 0.9)

                angle = math.atan2(
                    ball.velocity_y,
                    ball.velocity_x,
                )

            weight = ball.squash * ball.squash

            if weight > 0.12:
                k = weight * 0.38

                # First axis lies along the surface normal.
                semi_a = radius * (1.0 - k)
                semi_b = radius * (1.0 + k * 0.7)

                angle = math.atan2(
                    ball.squash_ny,
                    ball.squash_nx,
                )

        count = 18 if radius < 20 else 26

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        points = []

        for i in range(count):
            theta = 2.0 * math.pi * i / count

            ex = semi_a * math.cos(theta)
            ey = semi_b * math.sin(theta)

            points.append(
                (
                    ball.x + ex * cos_a - ey * sin_a,
                    ball.y + ex * sin_a + ey * cos_a,
                )
            )

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        bounds = pygame.Rect(
            int(min(xs)) - 2,
            int(min(ys)) - 2,
            int(max(xs) - min(xs)) + 5,
            int(max(ys) - min(ys)) + 5,
        )

        return points, bounds

    def _collect_glow(self, items, ball, radius, fade):
        glow_radius = int(radius * 2.4)

        level = min(
            len(_GLOW_LEVELS) - 1,
            int(ball.speed / 450.0),
        )

        if fade < 0.6:
            level = max(0, level - 1)

        sprite = self._glow(ball.hue, glow_radius, level)

        position = (
            int(round(ball.x)) - glow_radius,
            int(round(ball.y)) - glow_radius,
        )

        bounds = pygame.Rect(
            position[0],
            position[1],
            glow_radius * 2,
            glow_radius * 2,
        )

        items.append(
            (
                _LAYER_GLOW,
                bounds,
                ("sprite", sprite, position),
            )
        )

    def _collect_trail(self, items, ball, radius):
        trail = ball.trail

        count = len(trail)

        if count < 2:
            return

        sprites = []

        min_x = min_y = 10 ** 9
        max_x = max_y = -(10 ** 9)

        biggest = 0

        for index, (tx, ty) in enumerate(trail):
            fraction = (index + 1) / count

            sprite_radius = max(
                2,
                int(radius * (0.35 + 0.6 * fraction)),
            )

            level = min(
                len(_GLOW_LEVELS) - 1,
                int(fraction * len(_GLOW_LEVELS)),
            )

            position = (
                int(round(tx)) - sprite_radius,
                int(round(ty)) - sprite_radius,
            )

            sprites.append(
                (
                    self._glow(ball.hue, sprite_radius, level),
                    position,
                )
            )

            biggest = max(biggest, sprite_radius)

            min_x = min(min_x, tx)
            max_x = max(max_x, tx)
            min_y = min(min_y, ty)
            max_y = max(max_y, ty)

        bounds = pygame.Rect(
            int(min_x) - biggest - 1,
            int(min_y) - biggest - 1,
            int(max_x - min_x) + biggest * 2 + 3,
            int(max_y - min_y) + biggest * 2 + 3,
        )

        items.append(
            (
                _LAYER_GLOW,
                bounds,
                ("trail", sprites),
            )
        )

    # ========================================================
    # Drawing one collected item
    # ========================================================

    def _draw_item(self, draw):
        kind = draw[0]

        screen = self.screen

        if kind == "sprite":
            screen.blit(
                draw[1],
                draw[2],
                special_flags=pygame.BLEND_RGB_ADD,
            )

        elif kind == "trail":
            for sprite, position in draw[1]:
                screen.blit(
                    sprite,
                    position,
                    special_flags=pygame.BLEND_RGB_ADD,
                )

        elif kind == "ring":
            _kind, color, center, radius, width = draw

            pygame.draw.circle(
                screen,
                color,
                center,
                radius,
                width,
            )

        elif kind == "body":
            (
                _kind,
                points,
                body,
                rim,
                highlight_position,
                highlight_radius,
                highlight_color,
            ) = draw

            pygame.draw.polygon(screen, body, points)

            # Anti-aliased outline in the rim colour smooths the
            # jagged polygon fill edge.
            pygame.draw.aalines(screen, rim, True, points)

            pygame.draw.circle(
                screen,
                highlight_color,
                highlight_position,
                highlight_radius,
            )

        elif kind == "launcher":
            launcher = draw[1]
            points = launcher.get_points()

            # Solid metallic tube fill
            pygame.draw.polygon(screen, (48, 55, 66), points)

            # Highlight & Shadow lines along the tube
            rad = math.radians(launcher.angle)
            u_x = math.sin(rad)
            u_y = math.cos(rad)
            v_x = -u_y
            v_y = u_x

            half_w = launcher.tube_width / 2.0
            length = launcher.length

            # Highlight line along left inner edge
            hl_start = (
                launcher.pivot_x - v_x * (half_w * 0.4),
                launcher.pivot_y - v_y * (half_w * 0.4),
            )
            hl_end = (
                launcher.pivot_x + u_x * length - v_x * (half_w * 0.4),
                launcher.pivot_y + u_y * length - v_y * (half_w * 0.4),
            )
            pygame.draw.line(
                screen,
                (115, 126, 142),
                hl_start,
                hl_end,
                width=max(2, int(half_w * 0.45)),
            )

            # Shadow line along right inner edge
            sh_start = (
                launcher.pivot_x + v_x * (half_w * 0.5),
                launcher.pivot_y + v_y * (half_w * 0.5),
            )
            sh_end = (
                launcher.pivot_x + u_x * length + v_x * (half_w * 0.5),
                launcher.pivot_y + u_y * length + v_y * (half_w * 0.5),
            )
            pygame.draw.line(
                screen,
                (26, 30, 38),
                sh_start,
                sh_end,
                width=max(2, int(half_w * 0.35)),
            )

            # Dark nozzle rim at tip
            p_bot_right = points[2]
            p_bot_left = points[3]
            rim_points = [
                (
                    launcher.pivot_x + u_x * (length - 6.0) - v_x * half_w,
                    launcher.pivot_y + u_y * (length - 6.0) - v_y * half_w,
                ),
                (
                    launcher.pivot_x + u_x * (length - 6.0) + v_x * half_w,
                    launcher.pivot_y + u_y * (length - 6.0) + v_y * half_w,
                ),
                p_bot_right,
                p_bot_left,
            ]
            pygame.draw.polygon(screen, (32, 36, 44), rim_points)

            # Crisp metallic outline
            pygame.draw.polygon(screen, (78, 88, 102), points, width=2)
            pygame.draw.aalines(screen, (95, 108, 124), True, points)

    # ========================================================
    # Colours and glow sprites
    # ========================================================

    def _hue_rgb(self, hue, saturation, value):
        key = (
            int(hue) % 360,
            int(saturation * 50),
            int(value * 50),
        )

        color = self._color_cache.get(key)

        if color is None:
            r, g, b = colorsys.hsv_to_rgb(
                (key[0] % 360) / 360.0,
                saturation,
                max(0.0, min(1.0, value)),
            )

            color = (
                int(r * 255),
                int(g * 255),
                int(b * 255),
            )

            self._color_cache[key] = color

        return color

    def _glow(self, hue, radius, level):
        # Soft radial gradient, cached by (hue bucket, size, level).
        radius = max(2, min(int(radius), 96))

        bucket = int(hue // 6) % 60

        key = (bucket, radius, level)

        sprite = self._glow_cache.get(key)

        if sprite is None:
            color = self._hue_rgb(bucket * 6 + 3, 0.9, 1.0)

            gain = _GLOW_LEVELS[level]

            sprite = pygame.Surface((radius * 2, radius * 2))

            sprite.fill((0, 0, 0))

            for i in range(radius, 0, -1):
                t = 1.0 - i / radius

                k = (t ** 1.8) * gain

                pygame.draw.circle(
                    sprite,
                    (
                        int(color[0] * k),
                        int(color[1] * k),
                        int(color[2] * k),
                    ),
                    (radius, radius),
                    i,
                )

            sprite = sprite.convert()

            self._glow_cache[key] = sprite

        return sprite

    # ========================================================
    # Timing and shutdown
    # ========================================================

    def get_delta_time(self):
        delta_time = self.clock.tick(TARGET_FPS) / 1000.0

        return max(0.0, min(delta_time, 0.05))

    def close(self):
        pygame.mouse.set_visible(True)

        pygame.quit()
