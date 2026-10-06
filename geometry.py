# geometry.py
#
# Small, shared helpers for working with ROTATED rectangles (an
# "OBB" - oriented bounding box) in screen space: a rectangle
# stickers can be glued on crooked, so an obstacle's position is no
# longer just an axis-aligned (x, y, width, height) - it also has
# an angle. This module is the single place that knows how to:
#
#   - turn a raw cv2.minAreaRect() result into a STABLE, canonical
#     (center, long side, short side, angle) representation that
#     doesn't jitter or jump between frames (see normalize_angle
#     and detection.py for why that's needed),
#   - blend/interpolate that angle smoothly across frames
#     (blend_angle),
#   - convert between world space and the rectangle's own local,
#     axis-aligned frame (world_to_local / local_to_world) - this
#     is what lets both collision detection AND collision response
#     be written as plain axis-aligned math that "just works" for
#     any rotation, by doing the rotation once at the boundary
#     instead of re-deriving rotated formulas everywhere,
#   - compute the 4 corners of a rectangle for drawing
#     (obb_corners),
#   - pull all of the above out of a detected-obstacle dict in one
#     place, with one shared definition of what counts as a valid
#     obstacle (extract_obb), so detection/physics/rendering can
#     never disagree about it.
#
# Angle convention: an un-rotated rectangle (long side horizontal)
# has angle 0. Positive angles rotate the long side counter-
# clockwise in standard math coordinates - visually CLOCKWISE on
# screen, since screen y grows downward. Angles are always kept in
# the canonical range (-90, 90], because a rectangle looks
# identical after a 180-degree rotation (it has no "arrow" printed
# on it), so anything outside that range is a redundant duplicate
# of an angle already inside it. Collapsing to one canonical range
# is what keeps blend_angle's shortest-path interpolation well
# defined instead of it occasionally spinning the "wrong" way.

import math


# ============================================================
# Angle canonicalization
# ============================================================

def normalize_angle(angle_degrees, period=180.0):
    """
    Fold an orientation angle into the canonical (-period/2,
    period/2] range.

    A rectangle has 180-degree rotational symmetry, so any angle is
    equivalent to one inside a 180-degree-wide window (period=180).
    A (nearly) SQUARE sticker is even more symmetric - it looks the
    same after every 90 degrees - so for those callers pass
    period=90. Always folding into the SAME window is what makes
    two angles describing the same physical orientation compare and
    blend as equal instead of looking like they are far apart.
    """
    half = period / 2.0

    angle = angle_degrees % period

    if angle > half:
        angle -= period

    return angle


def blend_angle(old_angle, new_angle, alpha, period=180.0):
    """
    Move old_angle toward new_angle by fraction alpha along the
    SHORTEST path, for orientations that repeat every `period`
    degrees.

    A naive `old + alpha * (new - old)` breaks near the wrap-around
    point: 89 degrees and -89 degrees are really only 2 degrees
    apart for a rectangle, but the naive subtraction sees 178 and
    blends the wrong way (and for a near-square sticker, averaging
    across its 90-degree ambiguity would even produce a completely
    wrong in-between angle). Wrapping the raw difference into
    (-period/2, period/2] first avoids both.
    """
    half = period / 2.0

    diff = ((new_angle - old_angle + half) % period) - half

    return normalize_angle(old_angle + alpha * diff, period)


def paint_dimensions(rect_long, rect_short, inset_ratio):
    """
    Size of the white patch painted on the monitor for a detected
    sticker: the sticker trimmed by inset_ratio * short side on every
    edge. Shared by rendering (to draw it) and detection (to blank
    exactly that area out before looking for stickers) so the two can
    never disagree about what was painted.
    """
    inset = inset_ratio * rect_short

    return (
        max(1.0, rect_long - 2.0 * inset),
        max(1.0, rect_short - 2.0 * inset),
    )


# ============================================================
# World <-> local (rectangle-aligned) frame conversion
# ============================================================

def world_to_local(x, y, angle_degrees):
    """
    Rotate a world-space vector (x, y) by -angle_degrees, i.e. into
    the rectangle's own local frame, where its long side lies along
    the local x-axis and its short side along the local y-axis.

    This takes a VECTOR, not a point - to transform a point, first
    subtract the rectangle's center, call this, and the result is
    the point's position relative to the rectangle in its own
    frame. The same function (with no subtraction) also correctly
    rotates direction-only quantities like velocity, since rotation
    doesn't care about position.
    """
    theta = math.radians(angle_degrees)

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    return (
        x * cos_t + y * sin_t,
        -x * sin_t + y * cos_t,
    )


def local_to_world(x, y, angle_degrees):
    """
    The exact inverse of world_to_local: rotate a local-frame
    vector back by +angle_degrees into world space.
    """
    theta = math.radians(angle_degrees)

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    return (
        x * cos_t - y * sin_t,
        x * sin_t + y * cos_t,
    )


# ============================================================
# Corners (for drawing / debugging)
# ============================================================

def obb_corners(center_x, center_y, rect_long, rect_short, angle_degrees):
    """
    Return the 4 corners of a rotated rectangle, in order around
    its perimeter, as world-space (x, y) tuples. Safe to hand
    straight to pygame.draw.polygon or cv2.polylines.
    """
    half_long = rect_long / 2.0
    half_short = rect_short / 2.0

    local_corners = (
        (half_long, half_short),
        (half_long, -half_short),
        (-half_long, -half_short),
        (-half_long, half_short),
    )

    corners = []

    for local_x, local_y in local_corners:
        world_dx, world_dy = local_to_world(
            local_x,
            local_y,
            angle_degrees,
        )

        corners.append(
            (
                center_x + world_dx,
                center_y + world_dy,
            )
        )

    return corners


# ============================================================
# Obstacle dict -> OBB extraction
# ============================================================

def extract_obb(obstacle):
    """
    Pull (center_x, center_y, rect_long, rect_short, angle) out of
    a detected-obstacle dict, or return None if the obstacle is
    missing/malformed/degenerate. This is the ONE place that
    decides what counts as a usable oriented rectangle, so physics,
    rendering, and validity-checking can never disagree with each
    other about it.

    Falls back to treating the obstacle as an axis-aligned
    (angle=0) rectangle built from x/y/width/height if it has no
    "center"/"rect_long"/"rect_short"/"angle" fields, so any code
    that still hands in a plain axis-aligned obstacle dict keeps
    working exactly as before.
    """
    if not isinstance(obstacle, dict):
        return None

    try:
        if "center" in obstacle:
            center_x, center_y = obstacle["center"]
        else:
            center_x = (
                obstacle["x"] + obstacle["width"] / 2.0
            )
            center_y = (
                obstacle["y"] + obstacle["height"] / 2.0
            )

        rect_long = obstacle.get(
            "rect_long",
            obstacle.get("width"),
        )
        rect_short = obstacle.get(
            "rect_short",
            obstacle.get("height"),
        )
        angle = obstacle.get("angle", 0.0)

        center_x = float(center_x)
        center_y = float(center_y)
        rect_long = float(rect_long)
        rect_short = float(rect_short)
        angle = float(angle)
    except (KeyError, TypeError, ValueError):
        return None

    if not all(
        math.isfinite(value)
        for value in (
            center_x,
            center_y,
            rect_long,
            rect_short,
            angle,
        )
    ):
        return None

    if rect_long <= 0 or rect_short <= 0:
        return None

    return center_x, center_y, rect_long, rect_short, angle
