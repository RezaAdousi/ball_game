# imaging.py
#
# Small, shared OpenCV helpers for illumination-robust color work,
# used by both calibration.py (finding the monitor) and
# detection.py (finding the white stickers).
#
# Both work in CIELAB rather than HSV, for one concrete reason: LAB
# separates brightness (L) from true color (a, b) far more cleanly
# than HSV does. A glare highlight, a shadow, a dimmer corner of the
# hall, or the camera's own auto white-balance drifting under
# exhibition lighting - all of these move L a lot while barely
# moving a/b. That is exactly the property needed to answer "is
# this green?" or "is this a neutral white/gray?" in a way that
# stays true regardless of how bright, dim, or unevenly lit that
# particular pixel happens to be right now - which is the real
# source of "the camera doesn't quite see what the monitor is
# actually showing": most of that gap is a brightness/exposure
# effect, not a pure color-chemistry one, and LAB is built to shrug
# that off.

import cv2
import numpy as np


def to_lab(frame_bgr):
    """
    (L, a, b), each a 2D uint8 array in OpenCV's 0-255 LAB
    encoding: L is lightness (0=black, 255=white), a is the
    green(low)-red(high) axis centered at 128, b is the
    blue(low)-yellow(high) axis centered at 128.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)

    return lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]


def chroma(a_channel, b_channel, reference_a=128.0, reference_b=128.0):
    """
    How colorful a pixel is, independent of how bright or dark it
    is - near 0 at the reference point, larger for anything further
    away from it in LAB's roughly perceptually-even a/b plane.
    Unlike HSV saturation (a ratio that gets noisy and unstable for
    dark pixels, since it is normalized by brightness itself), this
    stays meaningful whether a pixel is in a bright spot or a
    shadowed corner.

    The reference defaults to (128, 128) - LAB's own neutral gray -
    but callers that have a measured "what does a KNOWN-white
    surface look like to this camera right now" point (see
    Calibration.white_reference) should pass it in instead. Under
    colored ambient light even genuinely white paper is NOT neutral
    gray to the camera - that is unavoidable physics, not a flaw in
    the camera - so testing distance from true neutral would wrongly
    reject it. Testing distance from "whatever white looks like
    right now" sidesteps that entirely, the same way a photographer
    custom white-balances off a gray card instead of trusting a
    fixed assumption about what "white" means.
    """
    da = a_channel.astype(np.float32) - reference_a
    db = b_channel.astype(np.float32) - reference_b

    return np.sqrt(da ** 2 + db ** 2)


def green_dominance(a_channel):
    """
    How far into "green" territory a pixel sits on LAB's red-green
    axis, clipped to [0, 255]. Green is the LOW end of the a
    channel (a < 128), so this is just how far below 128 the pixel
    is. Using this one, roughly illumination-independent channel
    instead of comparing raw R/G/B brightness is what lets the
    calibration pattern still be found once the camera's own white
    balance has shifted its recorded color away from what the
    monitor is actually displaying.
    """
    return np.clip(
        128 - a_channel.astype(np.int16),
        0,
        255,
    ).astype(np.uint8)


def fill_holes(mask):
    """
    Fills any fully-enclosed hole in a binary mask (0/255) - e.g. a
    glare highlight punching a small gap in the middle of an
    otherwise solid colored region, or a reflection splitting a
    sticker's blob in two. Standard flood-fill-from-the-border
    trick: anything NOT reachable from the mask's own border by
    walking through zero pixels must be an enclosed hole, so it
    gets filled in. A real gap that reaches all the way out to the
    mask's edge (e.g. a person actually standing in front of part
    of the pattern) stays untouched, since it IS reachable from the
    border.
    """
    height, width = mask.shape[:2]

    # Padding with a guaranteed-zero 1px border first means the
    # flood-fill seed point is always background, even if the real
    # mask happens to already be foreground right at image corner
    # (0, 0) - otherwise floodFill would treat that corner's own
    # color as the "background" to flood, filling nothing at all.
    padded = cv2.copyMakeBorder(
        mask,
        1,
        1,
        1,
        1,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    flood = padded.copy()

    flood_fill_mask = np.zeros(
        (height + 4, width + 4),
        np.uint8,
    )

    cv2.floodFill(flood, flood_fill_mask, (0, 0), 255)

    flood = flood[1:-1, 1:-1]

    return mask | cv2.bitwise_not(flood)
