# calibration.py

import math

import cv2
import numpy as np

import imaging

from config import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    CALIBRATION_TARGET_HUE_DEGREES,
    CALIBRATION_TARGET_HUE_TOLERANCE_DEGREES,
    CALIBRATION_MIN_TARGET_SATURATION,
    CALIBRATION_MIN_TARGET_VALUE,
    CALIBRATION_ADAPTIVE_HUE_ENABLED,
    CALIBRATION_ADAPTIVE_HUE_BLEND,
    CALIBRATION_ADAPTIVE_MAX_DRIFT_DEGREES,
    CALIBRATION_WHITE_ADAPTIVE_OFFSET,
    CALIBRATION_WHITE_MIN_ABSOLUTE_LIGHTNESS,
    CALIBRATION_WHITE_MAX_GREEN_DOMINANCE,
    CALIBRATION_WHITE_REFERENCE_BLEND,
    CALIBRATION_WHITE_REFERENCE_MAX_OFFSET,
    CALIBRATION_MIN_SCREEN_AREA_RATIO,
    CALIBRATION_MAX_SCREEN_AREA_RATIO,
    CALIBRATION_MIN_OUTER_TARGET_RATIO,
    CALIBRATION_MIN_INNER_WHITE_RATIO,
    CALIBRATION_MIN_FULL_TARGET_RATIO,
    CALIBRATION_MIN_QUALITY,
    CALIBRATION_REQUIRED_STABLE_FRAMES,
    CALIBRATION_MAX_CORNER_JUMP_RATIO,
    CALIBRATION_SMOOTHING_ALPHA,
    CALIBRATION_MASK_CLOSE_KERNEL,
    CALIBRATION_OUTER_BORDER_SIZE,
    CALIBRATION_INNER_BORDER_SIZE,
)


class Calibration:
    """
    Detects the monitor in two phases, matching what Renderer draws
    in each:

      1. NOT YET CALIBRATED: the whole screen is filled solid pink
         (Renderer.draw_full_calibration_screen). We just look for
         the single largest, best-shaped, (almost) entirely-pink
         quad in the camera frame - the simplest, most unambiguous
         target possible for the very first lock-on.

      2. CALIBRATED (locked on): the screen instead shows a
         saturated green outer ring with a white inner ring around
         a black interior (Renderer.draw_calibration_border), drawn
         every frame during gameplay so the system can keep quietly
         re-verifying the monitor's position without interrupting
         play, while leaving the black interior free for the ball
         and the detected obstacles.

    Compared to the previous four-marker approach this is far more
    robust in a busy, noisy exhibition environment:
      - it's one large, high-contrast shape instead of four tiny
        independent ones, so a person briefly walking in front of
        part of the screen doesn't break detection the way losing
        one out of four small markers would;
      - requiring BOTH the green AND white ring to be present, in
        the right relative position, makes it very unlikely a
        random green/white object or a reflection in the room gets
        mistaken for the screen;
      - once locked, calibration is only re-verified/refined by
        new stable detections and is never silently dropped just
        because a frame or two failed - it only clears on an
        explicit reset() call (bound to the "C" key in main.py).

    Color detection is NOT a single fixed HSV threshold (see
    config.py's "Target-color detection" section for the full
    reasoning). In short: a loose, wide hue pre-filter rules out
    everything nowhere near green, and a per-frame Otsu threshold
    on green-dominance (G compared to R and B) then finds the exact
    cutoff that best separates "clearly green" from "not" IN THE
    CURRENT frame - so it keeps adapting to the room's lighting
    instead of needing a fixed brightness/saturation number that
    only ever matched the lighting at setup time. On top of that,
    once locked on, the system slowly learns the actual hue it is
    seeing and re-centers the pre-filter on it, bounded so it can
    never drift onto some unrelated color.

    Public API is intentionally identical to the previous
    marker-based Calibration class, so main.py needs no changes
    beyond how the pattern itself is drawn.
    """

    def __init__(self):
        self.matrix = None

        self.history = []
        self.hue_history = []
        self.previous_points = None
        self.reference_points = None

        # What hue (0-360 degrees) the camera is ACTUALLY showing
        # for the calibration green right now, learned on the fly.
        # None until the first successful lock, and never reset by
        # reset() below - a "C" key reset is about re-finding the
        # screen's position, not about the room's lighting having
        # changed, so there is no reason to throw this away with it.
        self._learned_hue_center = None

        # What the camera ACTUALLY records for the white inner ring
        # right now, as a (a*, b*) offset from LAB neutral (128,
        # 128) - i.e. exactly what a photographer gets by custom
        # white-balancing off a gray card, except sampled
        # continuously off the ring that is drawn every single
        # frame anyway. detection.py uses this to judge how "white"
        # a candidate sticker really is relative to what white
        # currently looks like to this camera, instead of relative
        # to an absolute assumption that colored ambient light can
        # never actually hold true. Also never cleared by reset().
        self._white_reference_ab = None

        self.failed_frame_count = 0

    # ========================================================
    # Public API
    # ========================================================

    def is_calibrated(self):
        return (
            self.matrix is not None
            and np.isfinite(self.matrix).all()
        )

    def reset(self):
        self.matrix = None

        self.history = []
        self.hue_history = []
        self.previous_points = None
        self.reference_points = None

        self.failed_frame_count = 0

    def update(self, frame):
        if frame is None:
            return self.is_calibrated()

        if frame.ndim != 3:
            return self.is_calibrated()

        # Before we're locked on, the screen is filled solid pink
        # (Renderer.draw_full_calibration_screen) - look for one
        # big pink quad. Once locked on, the screen instead shows
        # the pink/white border ring around a black interior
        # (Renderer.draw_calibration_border) - look for that
        # specific pattern instead, which also lets us tell a real
        # match from a random pink object/reflection in the room.
        if self.is_calibrated():
            candidate = self._find_best_border_candidate(frame)
        else:
            candidate = self._find_best_full_screen_candidate(
                frame
            )

        if candidate is None:
            self._handle_failed_detection()
            return self.is_calibrated()

        self.failed_frame_count = 0

        self._update_white_reference(candidate.get("white_ab"))

        points = candidate["points"]
        hue = candidate.get("hue_degrees")

        if self._stable_against_previous(
            points,
            frame.shape,
        ):
            self.history.append(points)
            self.hue_history.append(hue)

            if (
                len(self.history)
                > CALIBRATION_REQUIRED_STABLE_FRAMES
            ):
                self.history.pop(0)
                self.hue_history.pop(0)
        else:
            self.history = [points]
            self.hue_history = [hue]

        self.previous_points = points.copy()

        if (
            len(self.history)
            >= CALIBRATION_REQUIRED_STABLE_FRAMES
        ):
            self._accept_or_refine(frame.shape)

        return self.is_calibrated()

    def transform_frame(self, frame, scale=1.0):
        """
        Warp the camera frame into screen space. With scale < 1 the
        result is a proportionally smaller (SCREEN_WIDTH*scale by
        SCREEN_HEIGHT*scale) image of the same screen - the warp is
        composed with the scaling, so it's a single cheap pass and
        much faster than warping full size and shrinking after.
        """
        if not self.is_calibrated():
            raise RuntimeError(
                "Calibration has not been completed."
            )

        matrix = self.matrix

        if scale != 1.0:
            matrix = (
                np.diag([scale, scale, 1.0]) @ matrix
            )

        return cv2.warpPerspective(
            frame,
            matrix,
            (
                int(round(SCREEN_WIDTH * scale)),
                int(round(SCREEN_HEIGHT * scale)),
            ),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def transform_point(self, x, y):
        if not self.is_calibrated():
            raise RuntimeError(
                "Calibration has not been completed."
            )

        point = np.float32(
            [[[x, y]]]
        )

        transformed = cv2.perspectiveTransform(
            point,
            self.matrix,
        )

        return tuple(transformed[0][0])

    # ========================================================
    # Candidate detection
    # ========================================================

    def _find_best_border_candidate(self, frame):
        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV,
        )

        lightness, a_channel, b_channel = imaging.to_lab(frame)

        target_mask = self._make_target_mask(frame, hsv)
        white_mask = self._make_white_mask(frame)

        contours, _ = cv2.findContours(
            target_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        outer_depth = float(CALIBRATION_OUTER_BORDER_SIZE)

        inner_depth = float(
            CALIBRATION_OUTER_BORDER_SIZE
            + CALIBRATION_INNER_BORDER_SIZE
        )

        best = None

        for contour in contours:
            quad = self._quad_from_contour(
                contour,
                frame.shape,
            )

            if quad is None:
                continue

            quality = self._quad_quality(
                quad,
                frame.shape,
            )

            if quality <= 0:
                continue

            target_ratio = self._band_ratio(
                target_mask,
                quad,
                0.0,
                outer_depth,
            )

            if (
                target_ratio
                < CALIBRATION_MIN_OUTER_TARGET_RATIO
            ):
                continue

            white_ratio = self._band_ratio(
                white_mask,
                quad,
                outer_depth,
                inner_depth,
            )

            if (
                white_ratio
                < CALIBRATION_MIN_INNER_WHITE_RATIO
            ):
                continue

            score = (
                quality * 0.45
                + target_ratio * 0.30
                + white_ratio * 0.25
            )

            if score < CALIBRATION_MIN_QUALITY:
                continue

            if (
                best is None
                or score > best["score"]
            ):
                best = {
                    "points": quad,
                    "score": float(score),
                    "quality": float(quality),
                    "target_ratio": float(target_ratio),
                    "white_ratio": float(white_ratio),
                    "hue_degrees": self._band_mean_hue(
                        hsv,
                        target_mask,
                        quad,
                        0.0,
                        outer_depth,
                    ),
                    "white_ab": self._band_mean_ab(
                        a_channel,
                        b_channel,
                        white_mask,
                        quad,
                        outer_depth,
                        inner_depth,
                    ),
                }

        return best

    def _find_best_full_screen_candidate(self, frame):
        """
        Used only before we're locked on, while the monitor is
        showing one solid green fill (no white ring, no black
        interior). We just need the largest, best-shaped green quad
        whose own interior is (almost) entirely green - there's no
        separate inner-ring check needed since the "ring" is the
        whole screen.
        """
        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV,
        )

        target_mask = self._make_target_mask(frame, hsv)

        contours, _ = cv2.findContours(
            target_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        best = None

        for contour in contours:
            quad = self._quad_from_contour(
                contour,
                frame.shape,
            )

            if quad is None:
                continue

            quality = self._quad_quality(
                quad,
                frame.shape,
            )

            if quality <= 0:
                continue

            fill_ratio = self._full_quad_ratio(
                target_mask,
                quad,
            )

            if (
                fill_ratio
                < CALIBRATION_MIN_FULL_TARGET_RATIO
            ):
                continue

            score = (
                quality * 0.5
                + fill_ratio * 0.5
            )

            if score < CALIBRATION_MIN_QUALITY:
                continue

            if (
                best is None
                or score > best["score"]
            ):
                best = {
                    "points": quad,
                    "score": float(score),
                    "quality": float(quality),
                    "target_ratio": float(fill_ratio),
                    "white_ratio": 1.0,
                    "hue_degrees": self._quad_mean_hue(
                        hsv,
                        target_mask,
                        quad,
                    ),
                }

        return best

    def _quad_from_contour(
        self,
        contour,
        frame_shape,
    ):
        frame_area = (
            frame_shape[0]
            * frame_shape[1]
        )

        area = cv2.contourArea(contour)

        if area <= 0:
            return None

        area_ratio = area / frame_area

        if area_ratio < CALIBRATION_MIN_SCREEN_AREA_RATIO:
            return None

        if area_ratio > CALIBRATION_MAX_SCREEN_AREA_RATIO:
            return None

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        if perimeter <= 0:
            return None

        for epsilon_ratio in (
            0.005,
            0.008,
            0.012,
            0.018,
            0.025,
            0.035,
        ):
            approx = cv2.approxPolyDP(
                contour,
                epsilon_ratio * perimeter,
                True,
            )

            if len(approx) != 4:
                continue

            points = approx.reshape(
                4, 2
            ).astype(np.float32)

            if not cv2.isContourConvex(
                points.reshape((-1, 1, 2))
            ):
                continue

            return self._order_points(points)

        return None

    # ========================================================
    # Color masks
    # ========================================================

    def _make_target_mask(self, frame_bgr, hsv):
        """
        Finds the calibration green robustly under unpredictable,
        possibly-shifting exhibition lighting, by combining a loose
        hue pre-filter with a per-frame auto-threshold on a
        brightness-INVARIANT color signal, rather than one fixed
        HSV box (see config.py's "Target-color detection" section
        and the class docstring for the full reasoning).
        """
        pre_filter = self._make_hue_prefilter_mask(hsv)

        if cv2.countNonZero(pre_filter) < 64:
            # Nothing even loosely green-ish in view - there is no
            # meaningful split for Otsu to find, so don't ask it to
            # invent one out of noise.
            return self._clean_mask(pre_filter)

        # The "how green is this pixel" signal is measured on LAB's
        # a* axis (green-red), NOT by comparing raw R/G/B levels.
        # LAB deliberately separates brightness (L) from true color
        # (a, b), so a* barely moves when a glare highlight, a
        # shadow, or the camera's own auto-exposure/white-balance
        # changes how BRIGHT a pixel looks - which is most of what
        # "the camera doesn't quite see what the monitor shows"
        # actually is. Comparing raw channel levels (as a plain
        # "G minus R/B" measure would) conflates color with
        # brightness and is far more easily fooled by exactly that.
        _lightness, a_channel, _b_channel = imaging.to_lab(
            frame_bgr
        )

        dominance = imaging.green_dominance(a_channel)

        # Otsu needs to see BOTH classes to find a meaningful split
        # point - "pattern green" and "everything else" - so it is
        # run over the WHOLE frame's dominance histogram (that is
        # genuinely bimodal: a bright green cluster from the
        # pattern vs. a large, much less green-dominant cluster for
        # the room/people/background). Restricting it to only the
        # pixels the loose pre-filter already approved would feed
        # Otsu an almost-uniform population with no real second
        # class to split against, which produces a meaningless
        # threshold instead of an adaptive one.
        threshold, _ = cv2.threshold(
            dominance,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        strong = cv2.inRange(
            dominance,
            int(threshold),
            255,
        )

        # The hue pre-filter is then applied on top as a precision
        # pass: it rules out anything that passed the brightness-
        # based Otsu split but isn't actually the right hue (e.g. a
        # bright white highlight, which has near-zero dominance
        # anyway, or a saturated cyan/yellow object, which does
        # not).
        mask = cv2.bitwise_and(pre_filter, strong)

        mask = self._clean_mask(mask)

        # A specular glare spot on the monitor's glass can punch a
        # small hole clean through the middle of the pattern, or
        # even a thin streak that splits it into two separate
        # blobs. Filling any fully-enclosed hole (see
        # imaging.fill_holes) repairs exactly that without touching
        # a real gap caused by, say, a person actually standing in
        # front of part of the screen (which stays open to the
        # mask's edge and so is left alone).
        return imaging.fill_holes(mask)

    def _make_hue_prefilter_mask(self, hsv):
        center = self._effective_hue_center()

        mask = np.zeros(
            hsv.shape[:2],
            dtype=np.uint8,
        )

        for lower_h, upper_h in self._hue_ranges(
            center,
            CALIBRATION_TARGET_HUE_TOLERANCE_DEGREES,
        ):
            current = cv2.inRange(
                hsv,
                np.array(
                    [
                        lower_h,
                        CALIBRATION_MIN_TARGET_SATURATION,
                        CALIBRATION_MIN_TARGET_VALUE,
                    ],
                    dtype=np.uint8,
                ),
                np.array(
                    [upper_h, 255, 255],
                    dtype=np.uint8,
                ),
            )

            mask = cv2.bitwise_or(mask, current)

        return mask

    @staticmethod
    def _hue_ranges(center_degrees, tolerance_degrees):
        """
        A (center +/- tolerance) window in standard 0-360 degree
        hue, converted to OpenCV's 0-179 H channel and split into
        one or two (lower, upper) ranges so a window that wraps
        past 0/360 (relevant for colors near red, not for the green
        this is tuned for, but kept general) still works correctly.
        """
        raw_low = center_degrees - tolerance_degrees
        raw_high = center_degrees + tolerance_degrees

        if raw_low < 0.0:
            spans = [
                (0.0, raw_high),
                (raw_low + 360.0, 360.0),
            ]
        elif raw_high > 360.0:
            spans = [
                (raw_low, 360.0),
                (0.0, raw_high - 360.0),
            ]
        else:
            spans = [(raw_low, raw_high)]

        return [
            (
                int(round(lo / 2.0)),
                int(round(hi / 2.0)),
            )
            for lo, hi in spans
        ]

    def _make_white_mask(self, frame_bgr):
        """
        Finds the white inner ring the same way detection.py finds
        white stickers (see its module docstring): LOCALLY adaptive
        brightness rather than one fixed V floor for the whole frame
        (so a ring sitting in a dimmer part of the hall is still
        found), and LAB chroma rather than HSV saturation for "is
        this actually neutral" (stable across the whole brightness
        range, unlike HSV's saturation ratio). A fixed absolute HSV
        box here was exactly as fragile under real exhibition
        lighting as the old fixed box was for stickers.
        """
        lightness, a_channel, _b_channel = imaging.to_lab(
            frame_bgr
        )

        block = self._white_ring_adaptive_block_size(
            lightness.shape
        )

        local = cv2.adaptiveThreshold(
            lightness,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block,
            -CALIBRATION_WHITE_ADAPTIVE_OFFSET,
        )

        mask = local & cv2.inRange(
            lightness,
            CALIBRATION_WHITE_MIN_ABSOLUTE_LIGHTNESS,
            255,
        )

        # The only other bright thing near the white ring is the
        # outer green ring itself, bleeding in right at the band
        # boundary - so it is enough to exclude anything clearly
        # GREEN-dominant (one-sided), rather than pinning an exact
        # "how white" chroma number. A strong ambient color cast can
        # push even genuinely white paper's overall chroma quite
        # far (see config.py's comment), but it does not make white
        # look GREEN - a one-sided test stays reliable under warm,
        # cool, or any other cast a two-sided chroma ceiling cannot
        # universally cover.
        mask &= cv2.inRange(
            imaging.green_dominance(a_channel),
            0,
            CALIBRATION_WHITE_MAX_GREEN_DOMINANCE,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (9, 9),
        )

        return cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

    @staticmethod
    def _white_ring_adaptive_block_size(shape):
        # Tied to the frame's own size (calibration has no learned
        # "ring thickness" the way detection.py learns sticker
        # size), large enough that the sampled neighborhood is
        # mostly the ring's dark/green surroundings rather than the
        # ring's own pixels.
        size = int(round(min(shape[:2]) * 0.05))

        size = max(15, size)

        if size % 2 == 0:
            size += 1

        return size

    def _clean_mask(self, mask):
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 5),
        )

        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                CALIBRATION_MASK_CLOSE_KERNEL,
                CALIBRATION_MASK_CLOSE_KERNEL,
            ),
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            open_kernel,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            close_kernel,
        )

        return mask

    # ========================================================
    # Full-quad sampling (used during the solid-pink full-screen
    # calibration phase, before we're locked on)
    # ========================================================

    def _full_quad_ratio(self, mask, points):
        polygon = points.astype(np.int32)

        fill = np.zeros_like(mask)

        cv2.fillPoly(
            fill,
            [polygon],
            255,
        )

        total = cv2.countNonZero(fill)

        if total == 0:
            return 0.0

        inside = cv2.bitwise_and(
            mask,
            fill,
        )

        return cv2.countNonZero(inside) / total

    # ========================================================
    # Band sampling (used to confirm the pink outer / white inner
    # ring at the right relative depth, instead of trusting the
    # outer pink contour alone)
    # ========================================================

    # points is always [top_left, top_right, bottom_right,
    # bottom_left] (see _order_points), so edge 0 (top) and edge 2
    # (bottom) run the screen's WIDTH and edge 1 (right) and edge 3
    # (left) run its HEIGHT. An absolute screen-space pixel depth
    # (e.g. CALIBRATION_OUTER_BORDER_SIZE) converts to that edge's
    # own camera-space pixels via (edge's camera length / that
    # edge's TRUE screen-space length) - using the same reference
    # (SCREEN_WIDTH) for every edge, as if top/bottom and left/right
    # were interchangeable, silently undersizes the left/right band
    # whenever the screen isn't exactly square, which in practice is
    # always: a 1920x1080 screen's left/right edges are only
    # 1080/1920 as long as its top/bottom edges, so a band meant to
    # be the same physical width on every side would come out
    # visibly too shallow on the sides, easily shallow enough to
    # land entirely short of where the white ring actually starts.
    _EDGE_REFERENCE_LENGTH = (
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
    )

    def _band_ratio(
        self,
        mask,
        points,
        depth_start_px,
        depth_end_px,
    ):
        center = np.mean(points, axis=0)

        values = []

        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]

            edge = p2 - p1
            length = np.linalg.norm(edge)

            if length < 1:
                continue

            reference = self._EDGE_REFERENCE_LENGTH[i]

            depth_start = (
                depth_start_px * length / reference
            )
            depth_end = (
                depth_end_px * length / reference
            )

            if depth_end <= depth_start:
                continue

            normal = np.array(
                [-edge[1], edge[0]],
                dtype=np.float32,
            )

            normal_length = np.linalg.norm(normal)

            if normal_length < 1e-6:
                continue

            normal /= normal_length

            midpoint = (p1 + p2) * 0.5

            if (
                np.dot(normal, center - midpoint)
                < 0
            ):
                normal = -normal

            a = p1 + normal * depth_start
            b = p2 + normal * depth_start
            c = p2 + normal * depth_end
            d = p1 + normal * depth_end

            polygon = np.array(
                [a, b, c, d],
                dtype=np.int32,
            )

            band = np.zeros_like(mask)

            cv2.fillPoly(
                band,
                [polygon],
                255,
            )

            total = cv2.countNonZero(band)

            if total == 0:
                continue

            inside = cv2.bitwise_and(
                mask,
                band,
            )

            values.append(
                cv2.countNonZero(inside) / total
            )

        if not values:
            return 0.0

        return float(np.mean(values))

    def _band_polygon_mask(
        self,
        points,
        depth_start_px,
        depth_end_px,
        shape,
    ):
        """
        Same ring-band geometry as _band_ratio above (including the
        same per-edge reference-length correction), but returned as
        a single combined mask (union of the 4 edge trapezoids)
        instead of an averaged ratio - used for pooling pixels
        together (e.g. to measure their mean hue) rather than just
        counting them.
        """
        center = np.mean(points, axis=0)

        mask = np.zeros(shape, dtype=np.uint8)

        drew_any = False

        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]

            edge = p2 - p1
            length = np.linalg.norm(edge)

            if length < 1:
                continue

            reference = self._EDGE_REFERENCE_LENGTH[i]

            depth_start = (
                depth_start_px * length / reference
            )
            depth_end = (
                depth_end_px * length / reference
            )

            if depth_end <= depth_start:
                continue

            normal = np.array(
                [-edge[1], edge[0]],
                dtype=np.float32,
            )

            normal_length = np.linalg.norm(normal)

            if normal_length < 1e-6:
                continue

            normal /= normal_length

            midpoint = (p1 + p2) * 0.5

            if (
                np.dot(normal, center - midpoint)
                < 0
            ):
                normal = -normal

            a = p1 + normal * depth_start
            b = p2 + normal * depth_start
            c = p2 + normal * depth_end
            d = p1 + normal * depth_end

            polygon = np.array(
                [a, b, c, d],
                dtype=np.int32,
            )

            cv2.fillPoly(
                mask,
                [polygon],
                255,
            )

            drew_any = True

        return mask if drew_any else None

    def _band_mean_hue(
        self,
        hsv,
        color_mask,
        points,
        depth_start_px,
        depth_end_px,
    ):
        """
        Mean hue (0-360 degrees) of the pixels that are BOTH inside
        the given ring band AND already matched by color_mask - the
        measurement used to learn what hue the camera is actually
        recording for the calibration green right now (see
        Calibration._update_learned_hue).
        """
        band = self._band_polygon_mask(
            points,
            depth_start_px,
            depth_end_px,
            hsv.shape[:2],
        )

        if band is None:
            return None

        return self._masked_mean_hue(
            hsv,
            cv2.bitwise_and(color_mask, band),
        )

    def _quad_mean_hue(self, hsv, color_mask, points):
        polygon = points.astype(np.int32)

        fill = np.zeros(hsv.shape[:2], dtype=np.uint8)

        cv2.fillPoly(fill, [polygon], 255)

        return self._masked_mean_hue(
            hsv,
            cv2.bitwise_and(color_mask, fill),
        )

    def _band_mean_ab(
        self,
        a_channel,
        b_channel,
        color_mask,
        points,
        depth_start_px,
        depth_end_px,
    ):
        """
        Mean LAB (a*, b*) of the pixels that are BOTH inside the
        given ring band AND already matched by color_mask - used to
        learn what the camera currently records for a KNOWN-white
        surface (see white_reference / _update_white_reference).
        """
        band = self._band_polygon_mask(
            points,
            depth_start_px,
            depth_end_px,
            a_channel.shape[:2],
        )

        if band is None:
            return None

        combined = cv2.bitwise_and(color_mask, band)

        if cv2.countNonZero(combined) < 32:
            return None

        mean_a = cv2.mean(a_channel, mask=combined)[0]
        mean_b = cv2.mean(b_channel, mask=combined)[0]

        return float(mean_a), float(mean_b)

    @staticmethod
    def _masked_mean_hue(hsv, mask):
        # A minimum pixel count keeps a tiny sliver of matched
        # pixels (e.g. a stray reflection) from producing a
        # confident-looking but meaningless hue measurement.
        if cv2.countNonZero(mask) < 32:
            return None

        mean_hue_cv_units = cv2.mean(
            hsv[:, :, 0],
            mask=mask,
        )[0]

        return float(mean_hue_cv_units) * 2.0

    # ========================================================
    # Live white reference ("gray card" for the sticker detector)
    # ========================================================

    def white_reference(self):
        """
        (a*, b*) in LAB's 0-255 encoding (128 = neutral) that the
        camera is CURRENTLY recording for the calibration ring's
        white band, or None before enough samples have come in.
        detection.py measures how "white" a candidate sticker is
        relative to this, instead of relative to absolute LAB
        neutral - exactly like a photographer custom white-balancing
        off a gray card, except this one is sampled continuously off
        the ring that is drawn on the monitor every single frame
        anyway, so it tracks the room's actual current lighting
        automatically rather than needing to be re-tuned on-site.
        """
        return self._white_reference_ab

    def _update_white_reference(self, measured_ab):
        if measured_ab is None:
            return

        measured_a, measured_b = measured_ab

        if self._white_reference_ab is None:
            self._white_reference_ab = (measured_a, measured_b)
            return

        current_a, current_b = self._white_reference_ab

        blended_a = (
            current_a
            + CALIBRATION_WHITE_REFERENCE_BLEND
            * (measured_a - current_a)
        )
        blended_b = (
            current_b
            + CALIBRATION_WHITE_REFERENCE_BLEND
            * (measured_b - current_b)
        )

        # A real color cast from room lighting is a gentle, physical
        # effect - it should never push "white" wildly far from
        # actual neutral. Clamping the offset's magnitude keeps a
        # run of bad samples (someone's bright shirt briefly
        # overlapping the band, say) from teaching the sticker
        # detector a wildly wrong idea of white.
        offset_a = blended_a - 128.0
        offset_b = blended_b - 128.0

        magnitude = math.hypot(offset_a, offset_b)

        limit = CALIBRATION_WHITE_REFERENCE_MAX_OFFSET

        if magnitude > limit:
            scale = limit / magnitude

            offset_a *= scale
            offset_b *= scale

        self._white_reference_ab = (
            128.0 + offset_a,
            128.0 + offset_b,
        )

    # ========================================================
    # Hue learning
    # ========================================================
    #
    # Slowly re-centers the loose hue pre-filter on whatever hue
    # the camera is ACTUALLY recording for the calibration green,
    # instead of forever trusting the single fixed
    # CALIBRATION_TARGET_HUE_DEGREES value from config.py. This is
    # what lets the system track a slow lighting change over a long
    # show without needing to be re-tuned on-site - while
    # ADAPTIVE_MAX_DRIFT_DEGREES keeps it from ever wandering far
    # enough to start matching some unrelated color.

    def _effective_hue_center(self):
        if (
            CALIBRATION_ADAPTIVE_HUE_ENABLED
            and self._learned_hue_center is not None
        ):
            return self._learned_hue_center

        return CALIBRATION_TARGET_HUE_DEGREES

    def _update_learned_hue(self, measured_hue_degrees):
        if measured_hue_degrees is None:
            return

        if not CALIBRATION_ADAPTIVE_HUE_ENABLED:
            return

        current = (
            self._learned_hue_center
            if self._learned_hue_center is not None
            else CALIBRATION_TARGET_HUE_DEGREES
        )

        blended = self._circular_blend_degrees(
            current,
            measured_hue_degrees,
            CALIBRATION_ADAPTIVE_HUE_BLEND,
            period=360.0,
        )

        drift = self._circular_diff_degrees(
            blended,
            CALIBRATION_TARGET_HUE_DEGREES,
            period=360.0,
        )

        max_drift = CALIBRATION_ADAPTIVE_MAX_DRIFT_DEGREES

        if abs(drift) > max_drift:
            blended = (
                CALIBRATION_TARGET_HUE_DEGREES
                + math.copysign(max_drift, drift)
            ) % 360.0

        self._learned_hue_center = blended

    @staticmethod
    def _representative_hue(samples):
        """
        Circular mean (vector average, not a plain arithmetic mean
        - hue wraps around, so naively averaging e.g. 350 degrees
        and 10 degrees must give 0, not 180) of a list of hue
        samples in degrees, skipping any that are None. Returns
        None if nothing usable is in the list.
        """
        valid = [
            value
            for value in samples
            if value is not None
        ]

        if not valid:
            return None

        radians = np.radians(valid)

        mean_sin = float(np.mean(np.sin(radians)))
        mean_cos = float(np.mean(np.cos(radians)))

        return float(
            np.degrees(
                np.arctan2(mean_sin, mean_cos)
            )
            % 360.0
        )

    @staticmethod
    def _circular_diff_degrees(a, b, period):
        """Shortest signed a-b distance on a circle of this period."""
        return (
            (a - b + period / 2.0) % period
        ) - period / 2.0

    @classmethod
    def _circular_blend_degrees(cls, old, new, alpha, period):
        diff = cls._circular_diff_degrees(
            new,
            old,
            period,
        )

        return (old + alpha * diff) % period

    # ========================================================
    # Geometry / quality scoring
    # ========================================================

    def _quad_quality(
        self,
        points,
        frame_shape,
    ):
        height, width = frame_shape[:2]

        frame_area = width * height

        area = self._polygon_area(points)

        area_ratio = area / frame_area

        if area_ratio < CALIBRATION_MIN_SCREEN_AREA_RATIO:
            return 0.0

        if area_ratio > CALIBRATION_MAX_SCREEN_AREA_RATIO:
            return 0.0

        contour = points.reshape(
            (-1, 1, 2)
        ).astype(np.float32)

        if not cv2.isContourConvex(contour):
            return 0.0

        sides = self._side_lengths(points)

        if min(sides) < 50:
            return 0.0

        top_bottom = (
            min(sides[0], sides[2])
            / max(sides[0], sides[2])
        )

        left_right = (
            min(sides[1], sides[3])
            / max(sides[1], sides[3])
        )

        side_quality = (
            top_bottom + left_right
        ) / 2.0

        angle_quality = 1.0

        for i in range(4):
            p1 = points[i - 1]
            p2 = points[i]
            p3 = points[(i + 1) % 4]

            v1 = p1 - p2
            v2 = p3 - p2

            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)

            if n1 < 1e-6 or n2 < 1e-6:
                return 0.0

            cos_angle = np.clip(
                np.dot(v1, v2) / (n1 * n2),
                -1.0,
                1.0,
            )

            angle = np.degrees(
                np.arccos(cos_angle)
            )

            if angle < 45 or angle > 135:
                angle_quality *= 0.3
            elif angle < 60 or angle > 120:
                angle_quality *= 0.7

        area_quality = min(
            area_ratio / 0.40,
            1.0,
        )

        return float(
            area_quality * 0.35
            + side_quality * 0.35
            + angle_quality * 0.30
        )

    @staticmethod
    def _polygon_area(points):
        return abs(
            cv2.contourArea(
                points.astype(np.float32)
            )
        )

    @staticmethod
    def _side_lengths(points):
        return [
            np.linalg.norm(points[1] - points[0]),
            np.linalg.norm(points[2] - points[1]),
            np.linalg.norm(points[3] - points[2]),
            np.linalg.norm(points[0] - points[3]),
        ]

    @staticmethod
    def _order_points(points):
        points = np.asarray(
            points,
            dtype=np.float32,
        )

        s = points.sum(axis=1)
        d = np.diff(points, axis=1).reshape(-1)

        top_left = points[np.argmin(s)]
        top_right = points[np.argmin(d)]
        bottom_right = points[np.argmax(s)]
        bottom_left = points[np.argmax(d)]

        return np.array(
            [top_left, top_right, bottom_right, bottom_left],
            dtype=np.float32,
        )

    # ========================================================
    # Temporal stability
    # ========================================================

    def _stable_against_previous(
        self,
        points,
        frame_shape,
    ):
        if self.previous_points is None:
            return True

        height, width = frame_shape[:2]

        diagonal = np.sqrt(
            width * width + height * height
        )

        distance = self._corners_distance(
            points,
            self.previous_points,
        )

        return (
            distance
            <= diagonal * CALIBRATION_MAX_CORNER_JUMP_RATIO
        )

    @staticmethod
    def _corners_distance(a, b):
        return float(
            np.mean(
                np.linalg.norm(a - b, axis=1)
            )
        )

    # ========================================================
    # Accepting / refining calibration
    # ========================================================

    def _accept_or_refine(self, frame_shape):
        final_points = self._order_points(
            np.median(
                np.stack(self.history, axis=0),
                axis=0,
            )
        )

        quality = self._quad_quality(
            final_points,
            frame_shape,
        )

        if quality < CALIBRATION_MIN_QUALITY:
            return

        area_ratio = self._polygon_area(
            final_points
        ) / (frame_shape[0] * frame_shape[1])

        if area_ratio < CALIBRATION_MIN_SCREEN_AREA_RATIO:
            return

        if self.reference_points is not None:
            final_points = self._smooth(
                self.reference_points,
                final_points,
            )

        matrix = self._calculate_matrix(
            final_points
        )

        if matrix is None:
            return

        self.matrix = matrix
        self.reference_points = final_points

        self._update_learned_hue(
            self._representative_hue(self.hue_history)
        )

    @staticmethod
    def _smooth(old_points, new_points):
        alpha = CALIBRATION_SMOOTHING_ALPHA

        return (
            old_points * (1.0 - alpha)
            + new_points * alpha
        ).astype(np.float32)

    def _calculate_matrix(self, points):
        destination = np.float32(
            [
                [0, 0],
                [SCREEN_WIDTH, 0],
                [SCREEN_WIDTH, SCREEN_HEIGHT],
                [0, SCREEN_HEIGHT],
            ]
        )

        try:
            matrix = cv2.getPerspectiveTransform(
                points.astype(np.float32),
                destination,
            )
        except cv2.error:
            return None

        if matrix is None:
            return None

        if not np.isfinite(matrix).all():
            return None

        return matrix

    # ========================================================
    # Failure handling
    # ========================================================

    def _handle_failed_detection(self):
        self.failed_frame_count += 1

        self.history = []
        self.hue_history = []
        self.previous_points = None

    # ========================================================
    # Debugging
    # ========================================================

    def debug_draw(self, frame):
        """
        Returns an annotated copy of `frame` showing the best
        currently-detected candidate (if any), its quality/color
        scores, the hue center currently being searched around
        (handy for seeing the adaptive learning track the room's
        actual lighting), and whether calibration is locked. Useful
        for tuning config.py's hue/saturation/value floors on-site
        against your real camera and lighting.
        """
        output = frame.copy()

        if self.is_calibrated():
            candidate = self._find_best_border_candidate(frame)
        else:
            candidate = self._find_best_full_screen_candidate(
                frame
            )

        if candidate is not None:
            points = candidate["points"].astype(
                np.int32
            )

            cv2.polylines(
                output,
                [points.reshape((-1, 1, 2))],
                True,
                (0, 255, 0),
                3,
            )

            text = (
                f"quality={candidate['quality']*100:.0f}% "
                f"green={candidate['target_ratio']*100:.0f}% "
                f"white={candidate['white_ratio']*100:.0f}% "
                f"stable={len(self.history)}/"
                f"{CALIBRATION_REQUIRED_STABLE_FRAMES}"
            )
        else:
            text = "searching..."

        status = (
            "CALIBRATED"
            if self.is_calibrated()
            else "NOT CALIBRATED"
        )

        hue_line = (
            f"hue center ~{self._effective_hue_center():.0f} deg "
            f"(anchor {CALIBRATION_TARGET_HUE_DEGREES} deg)"
        )

        cv2.putText(
            output,
            f"{status} | {text}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            output,
            hue_line,
            (20, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return output