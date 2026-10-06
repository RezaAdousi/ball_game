# config.py

# ============================================================
# Screen
# ============================================================

SCREEN_WIDTH = 1530
SCREEN_HEIGHT = 920


# ============================================================
# Camera
# ============================================================

# CAMERA_URL = "http://172.30.2.62:8080/video"
# CAMERA_URL = "http://172.30.1.216:4747/video"
CAMERA_URL = "http://192.168.1.194:4747/video"

# How long to wait between attempts to (re)connect to the camera
# when it is unreachable - both on startup (phone app not open yet)
# and whenever a reconnect is needed mid-show. The game keeps
# retrying forever instead of crashing, since this runs unattended
# in a booth and nobody may be there to relaunch it.
CAMERA_CONNECT_RETRY_SECONDS = 2.0


# ============================================================
# Detection intervals
# ============================================================

# Both of these run on a background vision thread (see vision.py),
# never on the render loop, so they can no longer cause frame
# hitches. Obstacle detection works on a downscaled image (see
# DETECTION_SCALE) and is cheap enough to run 4x per second, which
# makes moving a sticker feel responsive.
MONITOR_DETECTION_INTERVAL = 1.0
OBSTACLE_DETECTION_INTERVAL = 0.25


# ============================================================
# Calibration - screen border pattern
# ============================================================
#
# Instead of four small corner markers, the monitor draws a
# magenta/pink outer ring with a white inner ring around the edge
# of the screen, permanently (during both calibration and normal
# gameplay - see Renderer.draw_calibration_border). This is much
# more robust in a busy, noisy exhibition hall than four small
# markers:
#   - it's a single large, high-contrast shape instead of four
#     tiny independent ones, so a person briefly walking in front
#     of one corner doesn't break detection the way losing one out
#     of four markers would;
#   - requiring BOTH the pink AND the white ring to be present
#     (in the right relative position) makes it very unlikely that
#     a random pink or white object/reflection in the room could
#     be mistaken for the screen.

# Colors drawn on the monitor itself (RGB, used by pygame).
#
# Why GREEN instead of magenta/pink: the outer ring only has to be
# the single easiest possible color for a CAMERA (not a human eye)
# to pick out of a crowded, oddly-lit exhibition hall, and green is
# the best pick available for three concrete reasons:
#   1. Every consumer camera sensor uses a Bayer filter with TWICE
#      as many green photosites as red or blue (the pattern a human
#      eye is most sensitive to), so the green channel is always
#      the least noisy, highest-resolution channel the camera has -
#      this is exactly why film/broadcast green-screens are green
#      and not some other color.
#   2. Pink/magenta sits close to skin tones and to a lot of common
#      clothing under warm (incandescent/halogen) exhibition
#      lighting, which white-balance algorithms can shift toward
#      orange/red - pushing it out of a fixed HSV window entirely.
#      Saturated green has no such close neighbour in a crowd of
#      people, and survives white-balance shifts far better (most
#      auto white balance implementations anchor their correction
#      around making genuinely neutral/gray areas look neutral,
#      which leaves a strongly green-dominant pixel green-dominant
#      under almost any color temperature).
#   3. It pairs well with the detection method below: "is the green
#      channel clearly stronger than red AND blue" is a RELATIVE
#      test, so it survives brightness/exposure changes (someone's
#      shadow, a spotlight) that would otherwise shift an absolute
#      HSV threshold out of range.
CALIBRATION_PATTERN_OUTER_COLOR = (40, 255, 60)   # saturated green
CALIBRATION_PATTERN_INNER_COLOR = (255, 255, 255)  # white

# Thickness of each ring, in screen pixels (at the SCREEN_WIDTH x
# SCREEN_HEIGHT design resolution above).
CALIBRATION_OUTER_BORDER_SIZE = 40
CALIBRATION_INNER_BORDER_SIZE = 26

# ------------------------------------------------------------
# Target-color detection
# ------------------------------------------------------------
#
# The exact color the camera records for CALIBRATION_PATTERN_OUTER_
# COLOR depends on the camera's white balance/exposure and the
# room's lighting, and can drift over the course of a show (the
# hall's lighting changing, the phone's camera re-adjusting
# exposure, etc.) - a single fixed HSV box, tuned once, is exactly
# what used to make detection "work great at setup time, then get
# flaky an hour into the exhibition". So the mask that finds the
# outer ring is now built from two cooperating, auto-adapting
# pieces instead of one fixed threshold (see Calibration.
# _make_target_mask for the implementation):
#
#   1. A wide, loose hue pre-filter (this section) that only rules
#      out colors nowhere near green at all (reds, blues, skin,
#      gray) - it does NOT try to pin down the exact shade.
#   2. Within whatever passes that pre-filter, a per-frame Otsu
#      threshold on how green-dominant each pixel is (how far G
#      exceeds R and B) - Otsu finds the split point that best
#      separates "clearly green" from "not" IN THAT FRAME, so it
#      re-adapts itself to the current lighting automatically,
#      frame by frame, with no fixed brightness/saturation number
#      to go stale.
#
# CALIBRATION_TARGET_HUE_DEGREES/TOLERANCE describe the loose
# pre-filter in standard 0-360 degree hue (green = 120 degrees).
CALIBRATION_TARGET_HUE_DEGREES = 120
CALIBRATION_TARGET_HUE_TOLERANCE_DEGREES = 45

# Loose floor for saturation/brightness in the pre-filter - just
# enough to exclude flat gray/black/white areas, not to pin an
# exact shade (that's Otsu's job).
CALIBRATION_MIN_TARGET_SATURATION = 50
CALIBRATION_MIN_TARGET_VALUE = 35

# Once locked on, the system also WATCHES the hue it is actually
# seeing inside the confirmed ring and slowly re-centers the
# pre-filter on it (see Calibration._update_learned_hue). This is
# what lets it track a slow lighting drift over the course of a
# multi-hour show instead of needing to be re-tuned on-site.
# ADAPTIVE_MAX_DRIFT_DEGREES bounds how far that learned center may
# ever wander from the true CALIBRATION_TARGET_HUE_DEGREES, so a
# run of bad detections can never "learn" its way onto some
# unrelated color.
CALIBRATION_ADAPTIVE_HUE_ENABLED = True
CALIBRATION_ADAPTIVE_HUE_BLEND = 0.08
CALIBRATION_ADAPTIVE_MAX_DRIFT_DEGREES = 30

# The white inner ring is found the same way detection.py finds
# white stickers (see config's "White obstacle detection" section
# for the full reasoning): a LOCAL adaptive brightness test rather
# than one fixed floor for the whole frame, and LAB chroma rather
# than HSV saturation for "is this actually neutral". A fixed HSV
# box here was just as fragile under real exhibition lighting as the
# old fixed box was for stickers.
CALIBRATION_WHITE_ADAPTIVE_OFFSET = 8
CALIBRATION_WHITE_MIN_ABSOLUTE_LIGHTNESS = 50

# One-sided: the only other bright thing nearby is the green ring
# itself, so it's enough to exclude clearly GREEN-dominant pixels
# rather than pin an exact "how white" number - see the comment in
# Calibration._make_white_mask for why a two-sided chroma ceiling
# can't reliably cover every possible lighting cast but this can.
CALIBRATION_WHITE_MAX_GREEN_DOMINANCE = 50

# The white inner ring also serves as a live "gray card": its mean
# LAB (a*, b*) as the camera currently records it is continuously
# learned (slow exponential blend) and handed to detection.py, which
# judges a candidate sticker's whiteness relative to THAT instead of
# absolute neutral - see Calibration.white_reference(). Colored
# ambient light shifts a genuinely white/neutral surface's recorded
# color for physical reasons no fixed threshold can see around; this
# is what lets the sticker detector stay correct anyway, and track
# the room's lighting changing over a long show without re-tuning.
CALIBRATION_WHITE_REFERENCE_BLEND = 0.05

# How far (LAB a/b units) the learned white point may ever drift
# from true neutral, so a run of bad samples can never teach the
# sticker detector a wildly wrong idea of white.
CALIBRATION_WHITE_REFERENCE_MAX_OFFSET = 40.0

# The detected quad's area, as a fraction of the full camera frame,
# must fall in this range. Filters out both tiny false positives
# (reflections, noise) and implausibly large ones.
CALIBRATION_MIN_SCREEN_AREA_RATIO = 0.10
CALIBRATION_MAX_SCREEN_AREA_RATIO = 0.90

# How much of the sampled outer/inner ring band must actually match
# its expected color for a candidate to be accepted.
CALIBRATION_MIN_OUTER_TARGET_RATIO = 0.55
CALIBRATION_MIN_INNER_WHITE_RATIO = 0.40

# Combined quality/color score (0-1) required before a detection is
# even considered as a calibration candidate.
CALIBRATION_MIN_QUALITY = 0.50

# While NOT yet calibrated, the whole screen is filled solid green
# (Renderer.draw_full_calibration_screen) instead of just the thin
# border ring. This gives the very first lock-on the simplest,
# most unambiguous target possible: one huge solid-color rectangle
# with nothing else on screen. This is the fraction of that
# candidate quad's own interior that must actually be green for it
# to count - it naturally stays low while someone is standing in
# front of part of the monitor, so a person walking through frame
# during calibration just delays the lock instead of causing a bad
# one.
CALIBRATION_MIN_FULL_TARGET_RATIO = 0.85

# Temporal stability: how many consecutive matching detections are
# required before (re)locking calibration, and how far (as a
# fraction of the camera frame's diagonal) the corners are allowed
# to drift between frames and still count as "the same" detection.
# This is what lets the system ignore a person briefly walking
# through frame instead of resetting calibration.
CALIBRATION_REQUIRED_STABLE_FRAMES = 5
CALIBRATION_MAX_CORNER_JUMP_RATIO = 0.035

# Exponential smoothing applied to the accepted corner positions on
# every successful re-detection once already calibrated, so small
# frame-to-frame jitter doesn't make the transform wobble.
CALIBRATION_SMOOTHING_ALPHA = 0.25

# Morphological closing kernel (pixels) used to bridge small gaps
# in the color mask caused by glare/reflections on the monitor
# glass. Bigger = more tolerant of glare, but also more likely to
# merge in nearby unrelated green/white blobs - tune together with
# the hue settings above.
CALIBRATION_MASK_CLOSE_KERNEL = 17


# ============================================================
# Camera feed health
# ============================================================
#
# If the raw camera feed barely changes for this long, treat it as
# a stuck/cached/frozen frame (phone screen locked, streaming app
# paused in the background, stale network buffer, dead wifi
# re-serving the last JPEG, lens covered, etc.) rather than a live
# view, and don't trust it for calibration or obstacle detection.
CAMERA_FROZEN_DIFF_THRESHOLD = 0.6
CAMERA_FROZEN_SECONDS = 2.0


# ============================================================
# Rendering
# ============================================================

BACKGROUND_COLOR = (0, 0, 0)

TARGET_FPS = 144

# Only repaint (and push to the display) the small screen regions
# that actually changed since the last frame, instead of redrawing
# and re-uploading all 1920x1080 pixels 144 times a second. On most
# machines this is the single biggest rendering speed-up. If you
# ever see stale "ghost" pixels on the monitor (a driver that
# ignores partial updates), set this to False - everything still
# works, just with more CPU/GPU load.
USE_DIRTY_RECTS = True

# Safety net for USE_DIRTY_RECTS: force one complete repaint every
# this many seconds, so even a misbehaving driver can't leave a
# smudge on the screen for long.
FULL_REPAINT_SECONDS = 10.0


# ============================================================
# Visual effects (purely cosmetic - never affect physics)
# ============================================================
#
# Master switch. Everything below is drawn ONLY with saturated
# colors (never white blobs) and only in small, dim glows, because
# the camera is looking at this same screen and must never mistake
# an effect for a white sticker. If detection ever misbehaves and
# you suspect the visuals, flip this to False first to find out.
VISUAL_EFFECTS_ENABLED = True

# Each ball gets its own hue (degrees, 0-360) from this range. The
# default skips magenta/pink (roughly 280-350) on purpose: that is
# the calibration color, and it must stay unique to the calibration
# border so the camera never confuses the two.
BALL_HUE_MIN_DEGREES = 15
BALL_HUE_MAX_DEGREES = 255

# Every time a ball bounces off a sticker its hue shifts by this
# many degrees (staying inside the range above) - a small, very
# visible "the sticker reacted to me" cue that people love.
BALL_HUE_SHIFT_ON_BOUNCE = 22

# Fading comet tail behind each ball: how many past positions it
# remembers, and how often (seconds) a new one is recorded.
BALL_TRAIL_LENGTH = 16
BALL_TRAIL_SAMPLE_SECONDS = 1.0 / 60.0

# Soft colored glow around each ball (brighter the faster it moves).
BALL_GLOW_ENABLED = True

# Springy squash-and-stretch: balls elongate along their direction
# of travel with speed, and squash flat for a moment on a hard hit.
BALL_SQUASH_STRETCH_ENABLED = True
BALL_SQUASH_DURATION_SECONDS = 0.16

# Sparks + expanding ring on a bounce whose impact speed (px/s)
# exceeds this; gentler taps and resting contact produce nothing,
# so a ball lying still on a sticker stays calm.
IMPACT_EFFECT_MIN_SPEED = 160.0
MAX_PARTICLES = 260


# ============================================================
# Detected obstacles (what is painted back onto the monitor)
# ============================================================
#
# When True, every obstacle the camera currently sees is also
# painted white on the game's own screen (i.e. on the physical
# monitor itself), where it was detected - as the real tilted
# rectangle, not an axis-aligned box. When False, nothing extra is
# drawn for it. Either way this is purely visual - balls always
# collide with every currently-detected obstacle regardless of this
# flag; it never affects physics.
SHOW_DETECTED_OBSTACLES = False

DETECTED_OBSTACLE_COLOR = (255, 255, 255)
DETECTED_OBSTACLE_BORDER_WIDTH = 0

# The painted white is deliberately drawn a little SMALLER than the
# sticker (this fraction of the sticker's short side is trimmed off
# each edge). Reason: the camera looks at this very screen. If the
# painted white were exactly sticker-sized, then after someone
# removes the sticker the camera would keep seeing the painted
# white and "detect" the (now nonexistent) sticker forever. Painting
# inset keeps it hidden under the real sticker, and the detector
# additionally blanks out exactly the area that was painted before
# looking for stickers (see detection.py), so a removed sticker
# really disappears.
DETECTED_OBSTACLE_PAINT_INSET_RATIO = 0.22


# ============================================================
# Ball physics
# ============================================================

BALL_RADIUS = 15

# Each ball's radius is randomly scaled by up to +/- this fraction
# (bigger balls are also heavier in ball-vs-ball collisions).
BALL_RADIUS_VARIATION = 0.25

BALL_GRAVITY = 900

# Fraction of normal speed kept after bouncing off a sticker or a
# screen edge (1.0 = perfectly elastic: with gravity, a ball just
# keeps bouncing at the same energy forever and never settles down,
# which is the default here). Lower it (e.g. ~0.75) for a "real
# rubber ball" feel that loses a little height on every bounce
# instead. Very slow contact against a STICKER is still always
# treated as fully inelastic (BALL_REST_SPEED) so a ball resting on
# one doesn't buzz at the pixel level; screen edges always bounce at
# the full BALL_RESTITUTION regardless of speed.
BALL_RESTITUTION = 1.0
BALL_REST_SPEED = 55.0

# Fraction of sliding (tangential) speed lost per bounce off a
# sticker - what makes a ball roll/slide along a tilted sticker
# instead of skating on ice. 0 = no energy lost this way either (the
# default, paired with BALL_RESTITUTION = 1.0 above); raise it for
# more of a "rolling to a stop on a ramp" feel.
BALL_FRICTION = 0.0

# Safety cap on speed (px/s), so a freak event can never launch a
# ball fast enough to tunnel through a sticker.
BALL_MAX_SPEED = 2000.0

# Balls collide with each other: overlapping balls push apart and
# exchange momentum (heavier balls push lighter ones around). 1.0 =
# no energy lost in the collision either, matching the walls/
# stickers above.
BALL_BALL_COLLISION_ENABLED = True
BALL_BALL_RESTITUTION = 1.0

# Physics runs in equal slices of at most this many seconds, so
# nothing tunnels even on a slow frame.
PHYSICS_MAX_STEP_SECONDS = 1.0 / 240.0


# ============================================================
# Ball spawning and stuck balls
# ============================================================
#
# The two settings that matter most, kept together and right up
# front so they're easy to find and tune:
#
# A new ball drops in every BALL_SPAWN_INTERVAL_SECONDS, on its own
# clock, completely independent of what earlier balls are doing or
# whether they've disappeared yet.
BALL_SPAWN_INTERVAL_SECONDS = 2.5

# ============================================================
# Launcher configuration
# ============================================================
#
# Attached to top-center of the screen.
# Angles are in degrees relative to straight down (0 = straight down,
# negative = left, positive = right).
LAUNCHER_MIN_ANGLE = -70.0
LAUNCHER_MAX_ANGLE = 70.0

# Dimensions of the launcher tube in screen pixels.
LAUNCHER_LENGTH = 75.0
LAUNCHER_WIDTH = 26.0

# Angular rotation speed (controls continuous smooth movement).
LAUNCHER_ROTATION_SPEED = 35.0

# Initial ball speed when exiting the nozzle (px/s).
LAUNCHER_BALL_SPAWN_SPEED = 280.0

# Every ball is removed (with a little pop effect, see effects.py)
# this many seconds after it was spawned, whatever else is
# happening to it. 0 = balls live forever (until/unless the stuck-
# timeout below removes them instead).
BALL_MAX_LIFETIME_SECONDS = 20.0

# --- Everything below is secondary detail for the two settings above ---

# Hard ceiling on balls alive at once - only a safety net so trapped
# balls can never accumulate without limit over a long show.
MAX_SIMULTANEOUS_BALLS = 12

# Random horizontal drop position. With BALL_SPAWN_RANDOM_X = False
# every ball drops from the middle of the screen (the old
# behavior). Margin keeps drops away from the side walls.
BALL_SPAWN_RANDOM_X = True
BALL_SPAWN_MARGIN = 120
BALL_SPAWN_VELOCITY_JITTER = 60.0

# A ball is "stuck" if it has stayed within BALL_STUCK_DISTANCE
# pixels of one spot for BALL_STUCK_TIMEOUT_SECONDS - usually a
# pocket between stickers, but with every screen edge now a solid
# wall (see physics.py) a ball resting on an empty floor with no
# obstacles nearby counts too. A stuck ball stays put for that long
# (so visitors can see it) and is then removed with a small pop,
# freeing its slot, same as the lifetime limit above. 0 = never
# remove stuck balls (BALL_MAX_LIFETIME_SECONDS still applies).
BALL_STUCK_DISTANCE = 30.0
BALL_STUCK_TIMEOUT_SECONDS = 8.0


# ============================================================
# White obstacle detection
# ============================================================

# The camera image is warped to screen space at this fraction of
# the real screen resolution before looking for stickers. 0.5 makes
# detection ~4x cheaper with no meaningful loss of precision (a
# sticker is still tens of pixels wide). All the pixel values below
# stay in real screen pixels - the code converts.
DETECTION_SCALE = 0.5

# Real sticker size limits, in screen pixels squared.
MIN_OBSTACLE_AREA = 1500
MAX_OBSTACLE_AREA = 200000

# --- Finding candidate blobs (deliberately loose) -------------
# Pass 1 finds anything vaguely bright and unsaturated. It is meant
# to be generous; the strict checks below decide what is a sticker.
#
# Both "bright" and "unsaturated" are judged in CIELAB, not HSV (see
# imaging.py): L (lightness) and chroma (how colorful a pixel is)
# are far more independent of each other there than HSV's V and S,
# which is what makes the two tests below behave consistently
# whether a sticker sits in a shadowed corner or under a spotlight.
#
# "Bright" is no longer one fixed brightness floor for the whole
# frame, either. A single global number means a sticker in dimmer
# ambient light can fall below it and never even reach the per-blob
# refinement step below - which is exactly what made otherwise
# identical stickers seem to be "detected by force" some of the
# time. Instead, pass 1 asks a strictly LOCAL question - "is this
# pixel clearly brighter than the small neighborhood around it?"
# (cv2.adaptiveThreshold) - the same question a human eye asks when
# picking out a pale sticker on a dim part of a black screen, and it
# holds regardless of the overall exposure level that frame happens
# to have. OBSTACLE_ADAPTIVE_OFFSET is how much brighter than that
# local neighborhood a pixel must be; OBSTACLE_MIN_ABSOLUTE_LIGHTNESS
# is a sanity floor underneath it, so camera sensor noise sitting in
# a totally dark area can never be mistaken for "locally brighter".
OBSTACLE_ADAPTIVE_OFFSET = 14
OBSTACLE_MIN_ABSOLUTE_LIGHTNESS = 55

# The neighborhood size (screen px) the adaptive threshold above
# looks at is tied to the sticker's own size - large enough that the
# neighborhood is mostly black screen around the sticker rather than
# the sticker's own bright pixels (which would otherwise drag the
# local average up and make the sticker fail to stand out against
# ITSELF). Used as a multiple of the short side once a size is
# known; OBSTACLE_ADAPTIVE_DEFAULT_SHORT_SIDE is only the starting
# guess used before any sticker has been seen yet.
OBSTACLE_ADAPTIVE_BLOCK_SPAN = 2.4
OBSTACLE_ADAPTIVE_DEFAULT_SHORT_SIDE = 90

# Loose chroma ceiling for pass 1 (and for the per-blob edge
# refinement below) - just needs to exclude obviously colored
# objects (clothing, skin, colored lighting splashed on the floor),
# not pin an exact "how white" number (that is MAX_OBSTACLE_CHROMA
# below).
OBSTACLE_COARSE_CHROMA_MAX = 45

# Each blob's edge is then re-drawn at the "half-way" LIGHTNESS
# between the sticker's own paper level and the dark screen next to
# it. That makes a dim sticker in a shadowy corner and an identical
# sticker under a spotlight come out exactly the same size, which is
# what makes all stickers look and behave alike.
OBSTACLE_EDGE_LEVEL = 0.5

# --- Deciding whether a blob really is a sticker --------------
# Average lightness/chroma INSIDE the blob (CIELAB - see above).
MIN_OBSTACLE_BRIGHTNESS = 130
MAX_OBSTACLE_CHROMA = 28

# Fraction of the blob's own best-fit rotated rectangle it fills.
# A disc (a ball's glow, a bulb) scores 0.785, so anything above
# ~0.84 rejects round things outright.
MIN_OBSTACLE_RECTANGULARITY = 0.86

# Fraction of its convex hull the blob fills (rejects L-shapes,
# blobs with bites out of them).
MIN_OBSTACLE_SOLIDITY = 0.90

# Longest side / shortest side. Squares up to index-card shapes.
MAX_OBSTACLE_ASPECT_RATIO = 4.0

# Paper is a uniform, diffuse surface. Reflections and hot-spots
# vary a lot in brightness across their surface.
MAX_OBSTACLE_BRIGHTNESS_STD = 30.0

# A real sticker sits on a black screen: this fraction of the ring
# of pixels just outside it must be clearly darker than the sticker.
MIN_OBSTACLE_DARK_SURROUND_RATIO = 0.55

# Only look for stickers inside the calibration border, and never
# closer than this many extra screen pixels to it (a sticker
# touching the bright border would merge with it).
OBSTACLE_BORDER_MASK_MARGIN = 4
OBSTACLE_EDGE_MARGIN = 5

# --- All stickers are identical: use that -------------------
# Once a few stickers have been seen the system learns their size
# and rejects blobs that don't match (within this fraction) - this
# is what kills "forced" detections of things that merely look
# roughly rectangular. Set OBSTACLE_EXPECTED_SIZE = (long, short)
# in screen pixels to pin the size instead of learning it (the
# learned size is printed to the console once it has settled).
OBSTACLE_SIZE_TOLERANCE = 0.22
OBSTACLE_EXPECTED_SIZE = None

# Two or three identical stickers glued edge to edge form one
# bigger rectangle; accept whole multiples of the sticker size.
# (For physics a merged rectangle behaves exactly like the parts.)
OBSTACLE_ALLOW_MERGED_STICKERS = True

# Draw every sticker at exactly the learned size when the measured
# size is within ~10% of it, so identical stickers give identical
# obstacles instead of wobbling by a pixel or two.
OBSTACLE_SNAP_TO_EXPECTED_SIZE = True

# Forget the learned size after this many seconds with no stickers
# at all, so a different sticker size can be used later.
OBSTACLE_SIZE_FORGET_SECONDS = 30.0

# Tracks that are already established are kept with slightly
# relaxed rules (this fraction of the strict thresholds) so a real
# sticker never flickers in and out on borderline frames, while NEW
# stickers still have to pass the full strict tests.
OBSTACLE_RELAXED_FACTOR = 0.90

# --- Following stickers over time (each one tracked separately) ---
# Consecutive detections needed before a NEW sticker counts.
OBSTACLE_REQUIRED_STABLE_FRAMES = 2

# An established sticker survives this many passes without being
# seen (someone's hand in the way). At 0.25 s per pass, 12 = 3 s.
OBSTACLE_MAX_MISSING_FRAMES = 12

# After this many passes without seeing an established sticker, the
# game stops PAINTING its white patch (and, a moment later, stops
# blanking that area for the detector). Without this, a sticker that
# was moved would be hidden by its own old painted patch until it
# timed out. If it was moved, it is found at the new spot within a
# pass or two; if it was removed, it is dropped at
# OBSTACLE_MAX_MISSING_FRAMES as before. Physics keeps using it
# either way until then.
OBSTACLE_STALE_MISSES = 3

# Not-yet-confirmed candidates are dropped after this many misses.
OBSTACLE_PENDING_MAX_MISSING_FRAMES = 1

# Max distance (screen px) a sticker may move between two passes
# and still be "the same" sticker.
OBSTACLE_TEMPORAL_MATCH_DISTANCE = 80

# Kernel sizes (screen pixels) for cleaning up the blob mask.
OBSTACLE_MASK_OPEN_KERNEL = 3
OBSTACLE_MASK_CLOSE_KERNEL = 5
