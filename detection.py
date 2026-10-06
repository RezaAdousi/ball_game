# detection.py
#
# Finds the white paper stickers stuck on the monitor, as ROTATED
# rectangles, in a stable, flicker-free way.
#
# Input: a BGR image of the monitor already warped into screen
# space (Calibration.transform_frame) at DETECTION_SCALE of the real
# screen resolution. Output: one dict per sticker, in real SCREEN
# pixels (see geometry.py for the fields/angle convention).
#
# How it decides, in order (each step exists because a real
# exhibition hall breaks the simpler version):
#
#   1. Find generous "maybe" blobs: anything LOCALLY brighter than
#      its own immediate surroundings and not very colored, inside
#      the calibration border (see imaging.py - brightness and color
#      are measured in CIELAB, not HSV, specifically because that
#      separates the two cleanly). "Locally brighter" rather than
#      "brighter than one fixed number for the whole frame" is what
#      lets a sticker sitting in a dimmer part of the hall still be
#      found - a global floor would simply never see it, no matter
#      how good every later step is. The area the game itself
#      painted white for already-detected stickers is blanked out
#      first, otherwise a removed sticker would be "seen" forever in
#      its own painted white (see DETECTED_OBSTACLE_PAINT_*), and any
#      small hole a glare reflection punches through a blob (or a
#      thin glare streak that splits it in two) is patched before
#      measuring it (see imaging.fill_holes).
#
#   2. Re-draw each blob's edge at the half-way lightness between
#      that blob's own paper brightness and the dark screen beside
#      it. A dim sticker in a shadowy corner and an identical one
#      under a spotlight then come out the SAME size - a fixed
#      brightness cutoff can't do that, and it is what made
#      identical stickers seem to be "detected by force" some of the
#      time and not others.
#
#   3. Measure the refined blob: best-fit rotated rectangle (true
#      side lengths and angle), how well it fills that rectangle
#      (rejects round glows), convexity, uniformity of brightness
#      (rejects hot-spot reflections), average saturation (rejects
#      skin/colored objects), and whether the ring of pixels just
#      outside it is dark (a sticker sits on a black screen).
#
#   4. All stickers are identical, so once a couple have been seen
#      the size is learned and a blob of a different size is
#      rejected. Two or three stickers glued edge to edge form one
#      bigger rectangle and are accepted as whole multiples.
#
#   5. Every sticker is tracked separately over time. NEW stickers
#      must pass the strict tests on consecutive passes; stickers
#      already established are kept under slightly relaxed tests, so
#      a real sticker doesn't flicker on borderline frames, and they
#      survive a hand in front of them for a few seconds. Motion is
#      smoothed with a dead-band (no jitter when nothing moves) and
#      catches up fast when a sticker really is moved.

import math
import time
from collections import deque

import cv2
import numpy as np

import imaging

from geometry import (
    normalize_angle,
    blend_angle,
    obb_corners,
    extract_obb,
    paint_dimensions,
)

from config import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    CALIBRATION_OUTER_BORDER_SIZE,
    CALIBRATION_INNER_BORDER_SIZE,
    DETECTION_SCALE,
    MIN_OBSTACLE_AREA,
    MAX_OBSTACLE_AREA,
    OBSTACLE_ADAPTIVE_OFFSET,
    OBSTACLE_MIN_ABSOLUTE_LIGHTNESS,
    OBSTACLE_ADAPTIVE_BLOCK_SPAN,
    OBSTACLE_ADAPTIVE_DEFAULT_SHORT_SIDE,
    OBSTACLE_COARSE_CHROMA_MAX,
    OBSTACLE_EDGE_LEVEL,
    MIN_OBSTACLE_BRIGHTNESS,
    MAX_OBSTACLE_CHROMA,
    MIN_OBSTACLE_RECTANGULARITY,
    MIN_OBSTACLE_SOLIDITY,
    MAX_OBSTACLE_ASPECT_RATIO,
    MAX_OBSTACLE_BRIGHTNESS_STD,
    MIN_OBSTACLE_DARK_SURROUND_RATIO,
    OBSTACLE_BORDER_MASK_MARGIN,
    OBSTACLE_EDGE_MARGIN,
    OBSTACLE_SIZE_TOLERANCE,
    OBSTACLE_EXPECTED_SIZE,
    OBSTACLE_ALLOW_MERGED_STICKERS,
    OBSTACLE_SNAP_TO_EXPECTED_SIZE,
    OBSTACLE_SIZE_FORGET_SECONDS,
    OBSTACLE_RELAXED_FACTOR,
    OBSTACLE_REQUIRED_STABLE_FRAMES,
    OBSTACLE_MAX_MISSING_FRAMES,
    OBSTACLE_PENDING_MAX_MISSING_FRAMES,
    OBSTACLE_STALE_MISSES,
    OBSTACLE_TEMPORAL_MATCH_DISTANCE,
    OBSTACLE_MASK_OPEN_KERNEL,
    OBSTACLE_MASK_CLOSE_KERNEL,
    SHOW_DETECTED_OBSTACLES,
    DETECTED_OBSTACLE_PAINT_INSET_RATIO,
)


# Extra margin (screen px) added around the painted patch when it is
# blanked out, to absorb the small misalignment between the pixels
# we painted and where the camera sees them.
_BLANK_MARGIN = 5.0

# Below this long/short ratio a sticker is "square enough" that its
# orientation only matters modulo 90 degrees.
_SQUARE_ASPECT = 1.15


class _Track:
    """One physical sticker followed over time."""

    __slots__ = (
        "id",
        "cx",
        "cy",
        "angle",
        "long",
        "short",
        "hits",
        "misses",
        "confirmed",
    )

    def __init__(self, track_id, candidate):
        self.id = track_id
        self.cx = candidate["cx"]
        self.cy = candidate["cy"]
        self.angle = candidate["angle"]
        self.long = candidate["long"]
        self.short = candidate["short"]
        self.hits = 1
        self.misses = 0
        self.confirmed = False


class Detector:
    def __init__(self):
        self.scale = float(DETECTION_SCALE)

        self.det_width = int(round(SCREEN_WIDTH * self.scale))
        self.det_height = int(round(SCREEN_HEIGHT * self.scale))

        self._s2 = self.scale * self.scale

        # Kernels, sized in screen pixels then scaled (always odd).
        self._open_kernel = self._kernel(OBSTACLE_MASK_OPEN_KERNEL)
        self._close_kernel = self._kernel(OBSTACLE_MASK_CLOSE_KERNEL)
        self._small_kernel = np.ones((3, 3), np.uint8)

        # Region where stickers are searched: everything inside the
        # calibration border (which is bright by design and would
        # merge with a sticker touching it).
        border = int(
            round(
                (
                    CALIBRATION_OUTER_BORDER_SIZE
                    + CALIBRATION_INNER_BORDER_SIZE
                    + OBSTACLE_BORDER_MASK_MARGIN
                )
                * self.scale
            )
        )

        self._roi = np.zeros(
            (self.det_height, self.det_width), np.uint8
        )
        self._roi[
            border : self.det_height - border,
            border : self.det_width - border,
        ] = 255

        margin = int(round(OBSTACLE_EDGE_MARGIN * self.scale))

        self._roi_bounds = (
            border + margin,
            border + margin,
            self.det_width - border - margin,
            self.det_height - border - margin,
        )

        self._pinned_size = None

        if OBSTACLE_EXPECTED_SIZE:
            long_side, short_side = OBSTACLE_EXPECTED_SIZE
            self._pinned_size = (
                float(max(long_side, short_side)),
                float(min(long_side, short_side)),
            )

        # What "white" currently looks like to the camera, as a LAB
        # (a*, b*) point - see Calibration.white_reference(). Starts
        # at neutral (128, 128) and is kept updated by whoever drives
        # this Detector (vision.py); NOT reset by reset() below,
        # since it describes the room's current lighting, not
        # anything about where the stickers are.
        self._white_a = 128.0
        self._white_b = 128.0

        self.reset()

    # ========================================================
    # Public API
    # ========================================================

    def set_white_reference(self, reference_ab):
        """
        Update what "white" currently looks like to the camera (see
        Calibration.white_reference()). Falls back to LAB neutral
        (128, 128) if None (not measured yet, or adaptive hue/white
        learning is disabled) - the same as the old fixed behavior.
        """
        if reference_ab is None:
            self._white_a = 128.0
            self._white_b = 128.0
        else:
            self._white_a, self._white_b = reference_ab

    def reset(self):
        self.tracks = []
        self._next_id = 1

        self._samples = deque(maxlen=30)
        self._expected = self._pinned_size
        self._established = self._pinned_size is not None
        self._last_size_log = None

        self._last_seen_any = time.monotonic()

        self.obstacles = []
        self._signature = None

        # Last pass's raw candidates, for debugging/tuning.
        self.last_candidates = []

    def detect(self, frame):
        """
        Run one detection pass on a screen-space frame at
        DETECTION_SCALE and return the current list of confirmed
        stickers (screen pixels). The SAME list object is returned
        while nothing changed, so consumers can cheaply notice
        changes with an identity check.
        """
        if (
            frame is None
            or frame.ndim != 3
            or frame.shape[0] != self.det_height
            or frame.shape[1] != self.det_width
        ):
            return self.obstacles

        candidates = self._find_candidates(frame)

        self.last_candidates = candidates

        self._update_tracks(candidates)

        self._forget_stale_size()

        self._publish()

        return self.obstacles

    # ========================================================
    # 1. Coarse blobs
    # ========================================================

    def _find_candidates(self, frame):
        lightness, a_channel, b_channel = imaging.to_lab(frame)

        lightness = cv2.GaussianBlur(lightness, (3, 3), 0)

        chroma = np.clip(
            imaging.chroma(
                a_channel,
                b_channel,
                self._white_a,
                self._white_b,
            ),
            0,
            255,
        ).astype(np.uint8)

        # "Locally brighter than its own neighborhood" (not "above
        # one fixed brightness for the whole frame") - see the
        # module docstring and config.py's OBSTACLE_ADAPTIVE_* for
        # why. A plain floor is still kept underneath it purely as a
        # noise guard: without it, sensor noise inside a totally
        # dark, sticker-free patch of screen can register as
        # "locally brighter than its (equally dark) surroundings"
        # even though nothing is actually there.
        local = cv2.adaptiveThreshold(
            lightness,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self._adaptive_block_size(),
            -OBSTACLE_ADAPTIVE_OFFSET,
        )

        mask = local & cv2.inRange(
            lightness,
            OBSTACLE_MIN_ABSOLUTE_LIGHTNESS,
            255,
        )

        mask &= cv2.inRange(
            chroma,
            0,
            OBSTACLE_COARSE_CHROMA_MAX,
        )

        mask &= self._roi

        blank = self._painted_blank_mask()

        if blank is not None:
            mask[blank > 0] = 0

        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, self._open_kernel
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, self._close_kernel
        )

        # Patches a glare highlight punched through the middle of a
        # sticker, or a thin specular streak that splits one blob
        # into two, without touching a real gap (e.g. a ball or a
        # hand actually covering part of it), which stays open to
        # the mask's edge and so is left alone.
        mask = imaging.fill_holes(mask)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates = []

        for contour in contours:
            candidate = self._analyze(
                contour,
                lightness,
                chroma,
                blank,
            )

            if candidate is not None:
                candidates.append(candidate)

        return candidates

    def _adaptive_block_size(self):
        """
        Neighborhood size (detection-scale px, always odd) for the
        local brightness threshold above. Tied to the sticker's own
        short side (once known) so the neighborhood sampled around a
        sticker is mostly black screen rather than the sticker's own
        bright pixels, which would otherwise drag the local average
        up and make the sticker fail to stand out against itself.
        """
        if self._expected is not None:
            short_side = min(self._expected)
        else:
            short_side = OBSTACLE_ADAPTIVE_DEFAULT_SHORT_SIDE

        size = int(
            round(
                short_side
                * self.scale
                * OBSTACLE_ADAPTIVE_BLOCK_SPAN
            )
        )

        size = max(15, size)

        if size % 2 == 0:
            size += 1

        return size

    def _painted_blank_mask(self):
        """
        Mask of the exact patches the game painted white for the
        stickers it is currently tracking. Those pixels are ignored
        so that a REMOVED sticker's leftover painted white can never
        be mistaken for a sticker (the camera is looking at the same
        screen the game paints on).
        """
        if not SHOW_DETECTED_OBSTACLES:
            return None

        # Keep blanking for two passes after painting stops, to
        # cover the camera's own latency (it still shows the old
        # painted patch for a moment).
        confirmed = [
            t
            for t in self.tracks
            if t.confirmed
            and t.misses < OBSTACLE_STALE_MISSES + 2
        ]

        if not confirmed:
            return None

        blank = np.zeros(
            (self.det_height, self.det_width), np.uint8
        )

        for track in confirmed:
            long_side, short_side = paint_dimensions(
                track.long,
                track.short,
                DETECTED_OBSTACLE_PAINT_INSET_RATIO,
            )

            corners = obb_corners(
                track.cx * self.scale,
                track.cy * self.scale,
                (long_side + 2 * _BLANK_MARGIN) * self.scale,
                (short_side + 2 * _BLANK_MARGIN) * self.scale,
                track.angle,
            )

            cv2.fillConvexPoly(
                blank,
                np.round(corners).astype(np.int32),
                255,
            )

        return blank

    # ========================================================
    # 2 + 3. Refine one blob and measure it
    # ========================================================

    def _analyze(self, contour, lightness, chroma, blank):
        area_px = cv2.contourArea(contour)

        # Loose pre-gate on the coarse blob (the refined blob is
        # gated exactly below).
        loose_area = area_px / self._s2

        if (
            loose_area < MIN_OBSTACLE_AREA * 0.5
            or loose_area > MAX_OBSTACLE_AREA * 1.5
        ):
            return None

        x, y, w, h = cv2.boundingRect(contour)

        ring = max(3, int(round(0.15 * min(w, h))))
        pad = ring + 3

        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(self.det_width, x + w + pad)
        y1 = min(self.det_height, y + h + pad)

        crop_light = lightness[y0:y1, x0:x1]
        crop_chroma = chroma[y0:y1, x0:x1]
        crop_roi = self._roi[y0:y1, x0:x1]

        crop_blank = (
            blank[y0:y1, x0:x1] if blank is not None else None
        )

        shifted = contour - np.array([[[x0, y0]]], np.int32)

        coarse_fill = np.zeros(crop_light.shape, np.uint8)

        cv2.drawContours(coarse_fill, [shifted], -1, 255, -1)

        inside = crop_light[coarse_fill > 0]
        outside = crop_light[coarse_fill == 0]

        if inside.size == 0 or outside.size < 8:
            return None

        plateau = float(np.percentile(inside, 75))
        background = float(np.percentile(outside, 10))

        # 2. Half-way brightness between this blob's own paper level
        # and the dark screen beside it.
        level = max(
            background
            + OBSTACLE_EDGE_LEVEL * (plateau - background),
            float(OBSTACLE_MIN_ABSOLUTE_LIGHTNESS),
        )

        refined = cv2.inRange(crop_light, level, 255)
        refined &= cv2.inRange(
            crop_chroma, 0, OBSTACLE_COARSE_CHROMA_MAX
        )
        refined &= crop_roi

        if crop_blank is not None:
            refined[crop_blank > 0] = 0

        refined = cv2.morphologyEx(
            refined, cv2.MORPH_OPEN, self._small_kernel
        )
        refined = cv2.morphologyEx(
            refined, cv2.MORPH_CLOSE, self._small_kernel
        )

        found, _ = cv2.findContours(
            refined,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not found:
            return None

        contour2 = max(found, key=cv2.contourArea)

        area2_px = cv2.contourArea(contour2)
        area = area2_px / self._s2

        if area < MIN_OBSTACLE_AREA or area > MAX_OBSTACLE_AREA:
            return None

        # ---- 3. measurements ---------------------------------
        rect = cv2.minAreaRect(contour2)

        (rect_cx, rect_cy), (rect_w, rect_h), _raw_angle = rect

        if rect_w <= 0 or rect_h <= 0:
            return None

        rectangularity = area2_px / (rect_w * rect_h)

        # minAreaRect's own angle convention differs between
        # OpenCV versions and jumps for near-square shapes, so the
        # angle is measured from the box's actual long edge.
        box = cv2.boxPoints(rect)

        edge_a = box[1] - box[0]
        edge_b = box[2] - box[1]

        length_a = float(np.hypot(edge_a[0], edge_a[1]))
        length_b = float(np.hypot(edge_b[0], edge_b[1]))

        if length_a <= 0 or length_b <= 0:
            return None

        if length_a >= length_b:
            long_edge, long_px, short_px = edge_a, length_a, length_b
        else:
            long_edge, long_px, short_px = edge_b, length_b, length_a

        aspect = long_px / short_px

        angle = math.degrees(
            math.atan2(long_edge[1], long_edge[0])
        )

        angle = normalize_angle(
            angle,
            90.0 if aspect < _SQUARE_ASPECT else 180.0,
        )

        hull_area = cv2.contourArea(cv2.convexHull(contour2))

        if hull_area <= 0:
            return None

        solidity = area2_px / hull_area

        fill = np.zeros(crop_light.shape, np.uint8)

        cv2.drawContours(fill, [contour2], -1, 255, -1)

        inner = cv2.erode(fill, self._small_kernel)

        if not inner.any():
            inner = fill

        mean_light = float(cv2.mean(crop_light, mask=inner)[0])
        mean_chroma = float(cv2.mean(crop_chroma, mask=inner)[0])

        _, std = cv2.meanStdDev(crop_light, mask=inner)

        light_std = float(std[0][0])

        # Ring just outside the blob must be dark (black screen).
        ring_kernel = np.ones(
            (2 * ring + 1, 2 * ring + 1), np.uint8
        )

        ring_mask = cv2.dilate(fill, ring_kernel) & ~fill

        ring_mask &= crop_roi

        if crop_blank is not None:
            ring_mask[crop_blank > 0] = 0

        ring_count = int(np.count_nonzero(ring_mask))

        if ring_count >= 12:
            dark = crop_light[ring_mask > 0] < (mean_light * 0.5)

            dark_ratio = float(np.count_nonzero(dark)) / ring_count
        else:
            dark_ratio = 1.0

        # Touching the edge of the search area means the sticker is
        # (or may be) cut off - its true size is unknown.
        bx, by, bw, bh = cv2.boundingRect(contour2)

        left, top, right, bottom = self._roi_bounds

        clipped = (
            bx + x0 <= left
            or by + y0 <= top
            or bx + x0 + bw >= right
            or by + y0 + bh >= bottom
        )

        # ---- back to real screen pixels ----------------------
        inverse = 1.0 / self.scale

        candidate = {
            "cx": (rect_cx + x0) * inverse,
            "cy": (rect_cy + y0) * inverse,
            "angle": float(angle),
            "long": long_px * inverse,
            "short": short_px * inverse,
            "area": float(area),
            "aspect": float(aspect),
            "rectangularity": float(rectangularity),
            "solidity": float(solidity),
            "brightness": mean_light,
            "chroma": mean_chroma,
            "std": light_std,
            "dark_ratio": dark_ratio,
            "clipped": clipped,
        }

        candidate["strict_ok"], candidate["snap"] = self._passes(
            candidate, 1.0
        )

        candidate["relaxed_ok"], relaxed_snap = self._passes(
            candidate, OBSTACLE_RELAXED_FACTOR
        )

        candidate["snap"] = candidate["snap"] or relaxed_snap

        return candidate

    # ========================================================
    # Sticker tests (strict for new stickers, relaxed to keep
    # established ones)
    # ========================================================

    def _passes(self, c, factor):
        if c["clipped"]:
            return False, False

        if c["rectangularity"] < MIN_OBSTACLE_RECTANGULARITY * factor:
            return False, False

        if c["solidity"] < MIN_OBSTACLE_SOLIDITY * factor:
            return False, False

        if c["aspect"] > MAX_OBSTACLE_ASPECT_RATIO / factor:
            return False, False

        if c["brightness"] < MIN_OBSTACLE_BRIGHTNESS * factor:
            return False, False

        if c["chroma"] > MAX_OBSTACLE_CHROMA / factor:
            return False, False

        if c["std"] > MAX_OBSTACLE_BRIGHTNESS_STD / factor:
            return False, False

        if (
            c["dark_ratio"]
            < MIN_OBSTACLE_DARK_SURROUND_RATIO * factor
        ):
            return False, False

        return self._size_matches(
            c["long"],
            c["short"],
            OBSTACLE_SIZE_TOLERANCE / factor,
        )

    def _size_matches(self, long_side, short_side, tolerance):
        """
        (matches, is_single_sticker_size). With no learned size yet
        anything passes; with a size learned from just one sighting
        the check is deliberately loose (that one sighting could
        itself have been a false positive).
        """
        if self._expected is None:
            return True, False

        expected_long, expected_short = self._expected

        if not self._established:
            tolerance = max(tolerance, 0.45)

        limit = 3 if OBSTACLE_ALLOW_MERGED_STICKERS else 1

        for along in range(1, limit + 1):
            for across in range(1, limit + 1):
                if along * across > 6:
                    continue

                target_a = along * expected_long
                target_b = across * expected_short

                target_long = max(target_a, target_b)
                target_short = min(target_a, target_b)

                if (
                    abs(long_side - target_long) / target_long
                    <= tolerance
                    and abs(short_side - target_short)
                    / target_short
                    <= tolerance
                ):
                    single = along == 1 and across == 1

                    return True, single

        return False, False

    # ========================================================
    # Learned sticker size
    # ========================================================

    def _learn_size(self, long_side, short_side):
        if self._pinned_size is not None:
            return

        self._samples.append((long_side, short_side))

        best = []

        for a_long, a_short in self._samples:
            members = [
                (b_long, b_short)
                for b_long, b_short in self._samples
                if abs(b_long - a_long) / a_long <= 0.15
                and abs(b_short - a_short) / a_short <= 0.15
            ]

            if len(members) > len(best):
                best = members

        self._expected = (
            float(np.median([m[0] for m in best])),
            float(np.median([m[1] for m in best])),
        )

        self._established = len(best) >= 2

        if self._established:
            rounded = (
                round(self._expected[0]),
                round(self._expected[1]),
            )

            if (
                self._last_size_log is None
                or abs(rounded[0] - self._last_size_log[0]) > 4
                or abs(rounded[1] - self._last_size_log[1]) > 4
            ):
                self._last_size_log = rounded

                print(
                    "[DETECT] sticker size learned: about "
                    f"{rounded[0]} x {rounded[1]} px. To pin it, set "
                    f"OBSTACLE_EXPECTED_SIZE = ({rounded[0]}, "
                    f"{rounded[1]}) in config.py."
                )

    def _forget_stale_size(self):
        if self._pinned_size is not None:
            return

        now = time.monotonic()

        if any(t.confirmed for t in self.tracks):
            self._last_seen_any = now
            return

        if (
            self._expected is not None
            and now - self._last_seen_any
            > OBSTACLE_SIZE_FORGET_SECONDS
        ):
            self._samples.clear()
            self._expected = None
            self._established = False
            self._last_size_log = None

    # ========================================================
    # 5. Tracking
    # ========================================================

    def _update_tracks(self, candidates):
        pairs = []

        for track_index, track in enumerate(self.tracks):
            gate = min(
                OBSTACLE_TEMPORAL_MATCH_DISTANCE,
                max(25.0, 0.75 * track.short),
            )

            for cand_index, candidate in enumerate(candidates):
                if not candidate["relaxed_ok"]:
                    continue

                distance = math.hypot(
                    candidate["cx"] - track.cx,
                    candidate["cy"] - track.cy,
                )

                if distance <= gate:
                    pairs.append((distance, track_index, cand_index))

        pairs.sort()

        used_tracks = set()
        used_candidates = set()

        for _distance, track_index, cand_index in pairs:
            if (
                track_index in used_tracks
                or cand_index in used_candidates
            ):
                continue

            used_tracks.add(track_index)
            used_candidates.add(cand_index)

            self._apply_hit(
                self.tracks[track_index],
                candidates[cand_index],
            )

        survivors = []

        for track_index, track in enumerate(self.tracks):
            if track_index not in used_tracks:
                track.misses += 1

                limit = (
                    OBSTACLE_MAX_MISSING_FRAMES
                    if track.confirmed
                    else OBSTACLE_PENDING_MAX_MISSING_FRAMES
                )

                if track.misses > limit:
                    continue

            survivors.append(track)

        self.tracks = survivors

        # New stickers: strict tests only, and not on top of a
        # sticker we already follow.
        for cand_index, candidate in enumerate(candidates):
            if (
                cand_index in used_candidates
                or not candidate["strict_ok"]
            ):
                continue

            overlaps = any(
                math.hypot(
                    candidate["cx"] - t.cx,
                    candidate["cy"] - t.cy,
                )
                < 0.5 * min(t.short, candidate["short"])
                for t in self.tracks
            )

            if overlaps:
                continue

            track = _Track(self._next_id, candidate)

            self._next_id += 1

            self.tracks.append(track)

            self._maybe_confirm(track, candidate)

    def _apply_hit(self, track, candidate):
        track.misses = 0
        track.hits += 1

        long_side = candidate["long"]
        short_side = candidate["short"]

        if (
            OBSTACLE_SNAP_TO_EXPECTED_SIZE
            and self._established
            and self._expected is not None
            and candidate["snap"]
        ):
            expected_long, expected_short = self._expected

            if (
                abs(long_side - expected_long) / expected_long <= 0.10
                and abs(short_side - expected_short)
                / expected_short
                <= 0.10
            ):
                long_side, short_side = expected_long, expected_short

        # Position: dead-band while nothing moves (no jitter for a
        # ball resting on the sticker), fast catch-up when it does.
        distance = math.hypot(
            candidate["cx"] - track.cx,
            candidate["cy"] - track.cy,
        )

        if distance > 0.8:
            alpha = min(1.0, max(0.25, distance / 8.0))

            track.cx += (candidate["cx"] - track.cx) * alpha
            track.cy += (candidate["cy"] - track.cy) * alpha

        period = (
            90.0
            if track.long / max(track.short, 1e-6) < _SQUARE_ASPECT
            else 180.0
        )

        half = period / 2.0

        angle_diff = abs(
            ((candidate["angle"] - track.angle + half) % period)
            - half
        )

        if angle_diff > 0.6:
            alpha = min(1.0, max(0.25, angle_diff / 4.0))

            track.angle = blend_angle(
                track.angle,
                candidate["angle"],
                alpha,
                period,
            )

        track.long += (long_side - track.long) * 0.35
        track.short += (short_side - track.short) * 0.35

        self._maybe_confirm(track, candidate)

    def _maybe_confirm(self, track, candidate):
        if (
            track.confirmed
            or track.hits < OBSTACLE_REQUIRED_STABLE_FRAMES
        ):
            return

        track.confirmed = True

        # Only single-sticker sightings teach the size (a merged
        # pair would otherwise be learned as "the" sticker size).
        if candidate["snap"] or self._expected is None:
            self._learn_size(candidate["long"], candidate["short"])

    # ========================================================
    # Output
    # ========================================================

    def _publish(self):
        confirmed = [t for t in self.tracks if t.confirmed]

        signature = tuple(
            (
                t.id,
                round(t.cx, 1),
                round(t.cy, 1),
                round(t.angle, 1),
                round(t.long, 1),
                round(t.short, 1),
                t.misses < OBSTACLE_STALE_MISSES,
            )
            for t in confirmed
        )

        if signature == self._signature:
            return

        self._signature = signature

        obstacles = []

        for track in confirmed:
            corners = obb_corners(
                track.cx,
                track.cy,
                track.long,
                track.short,
                track.angle,
            )

            xs = [p[0] for p in corners]
            ys = [p[1] for p in corners]

            obstacles.append(
                {
                    "id": track.id,
                    "center": (track.cx, track.cy),
                    "angle": track.angle,
                    "rect_long": track.long,
                    "rect_short": track.short,
                    # False once the sticker hasn't been seen for a
                    # few passes: rendering then stops painting it.
                    "painted": track.misses < OBSTACLE_STALE_MISSES,
                    "x": min(xs),
                    "y": min(ys),
                    "width": max(xs) - min(xs),
                    "height": max(ys) - min(ys),
                }
            )

        self.obstacles = obstacles

    # ========================================================
    # Debug drawing
    # ========================================================

    def draw(self, frame, obstacles=None):
        """
        Annotated copy of a screen-space detection frame: confirmed
        stickers in green, rejected candidates in red (with the
        first failing reason's numbers), for tuning on site.
        """
        if obstacles is None:
            obstacles = self.obstacles

        output = frame.copy()

        for candidate in self.last_candidates:
            if candidate["relaxed_ok"]:
                continue

            corners = obb_corners(
                candidate["cx"],
                candidate["cy"],
                candidate["long"],
                candidate["short"],
                candidate["angle"],
            )

            points = (
                np.array(corners) * self.scale
            ).astype(np.int32).reshape((-1, 1, 2))

            cv2.polylines(output, [points], True, (0, 0, 255), 1)

            cv2.putText(
                output,
                f"r{candidate['rectangularity']:.2f} "
                f"s{candidate['solidity']:.2f} "
                f"d{candidate['dark_ratio']:.2f}",
                (int(points[0][0][0]), int(points[0][0][1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        for obstacle in obstacles:
            obb = extract_obb(obstacle)

            if obb is None:
                continue

            cx, cy, long_side, short_side, angle = obb

            corners = obb_corners(cx, cy, long_side, short_side, angle)

            points = (
                np.array(corners) * self.scale
            ).astype(np.int32).reshape((-1, 1, 2))

            cv2.polylines(output, [points], True, (0, 255, 0), 2)

            cv2.circle(
                output,
                (int(cx * self.scale), int(cy * self.scale)),
                4,
                (0, 255, 0),
                -1,
            )

        return output

    # ========================================================
    # Helpers
    # ========================================================

    def _kernel(self, size_screen_px):
        size = max(1, int(round(size_screen_px * self.scale)))

        if size % 2 == 0:
            size += 1

        return np.ones((size, size), np.uint8)
