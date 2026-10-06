# vision.py
#
# Runs everything that looks at the camera - screen calibration and
# sticker detection - on its own background thread, so the render /
# physics loop never waits on OpenCV. (Before this, a detection pass
# ran inside the game loop and every pass was a small visible hitch.)
# OpenCV releases the GIL while it works, so the two really do run
# side by side.
#
# The game loop only ever calls snapshot(), which is a cheap lock +
# a few attribute reads. The obstacle list it returns is the SAME
# list object until something actually changed (Detector publishes a
# new list only on change), so physics/rendering can detect changes
# with a plain identity check.

import threading
import time

from config import (
    DETECTION_SCALE,
    MONITOR_DETECTION_INTERVAL,
    OBSTACLE_DETECTION_INTERVAL,
)


class Vision:
    def __init__(self, camera, calibration, detector):
        self.camera = camera
        self.calibration = calibration
        self.detector = detector

        self._lock = threading.Lock()

        self._calibrated = False
        self._frozen = False
        self._obstacles = []
        self._message = "Camera not connected yet."

        # Bumped by request_reset(); a result computed against an
        # older generation is discarded instead of being published.
        self._generation = 0
        self._reset_pending = False

        self._stop_event = threading.Event()

        self._thread = threading.Thread(
            target=self._run,
            name="vision",
            daemon=True,
        )

    # ========================================================
    # Public API (called from the game loop)
    # ========================================================

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def request_reset(self):
        """Forget calibration and stickers; start over."""
        with self._lock:
            self._generation += 1
            self._reset_pending = True
            self._calibrated = False
            self._obstacles = []

    def snapshot(self):
        """(calibrated, camera_frozen, obstacles, camera_message)."""
        with self._lock:
            return (
                self._calibrated,
                self._frozen,
                self._obstacles,
                self._message,
            )

    # ========================================================
    # Thread
    # ========================================================

    def _run(self):
        last_frame_id = None
        last_monitor_pass = 0.0
        last_obstacle_pass = 0.0
        last_error_print = 0.0

        while not self._stop_event.is_set():
            try:
                with self._lock:
                    reset = self._reset_pending
                    self._reset_pending = False
                    generation = self._generation

                if reset:
                    self.calibration.reset()
                    self.detector.reset()

                    last_frame_id = None
                    last_monitor_pass = 0.0
                    last_obstacle_pass = 0.0

                try:
                    frame, frame_id = self.camera.read_with_id()
                except RuntimeError as error:
                    self._publish(
                        generation,
                        message=str(error),
                    )

                    time.sleep(0.05)
                    continue

                frozen = self.camera.is_frozen()

                if frame_id == last_frame_id:
                    # Same frame as last time: nothing new to
                    # analyse (the camera delivers ~30 fps, the
                    # loop spins faster). Still keep the frozen
                    # flag fresh in case frames stopped entirely.
                    self._publish(generation, frozen=frozen)

                    time.sleep(0.004)
                    continue

                last_frame_id = frame_id

                if frozen:
                    # A stuck feed must not be trusted for
                    # calibration or stickers.
                    self._publish(
                        generation,
                        frozen=True,
                        message="",
                    )
                    continue

                now = time.monotonic()

                calibrated = self.calibration.is_calibrated()

                # Until locked on, look at every new frame; once
                # locked, only re-verify the monitor occasionally.
                if (
                    not calibrated
                    or now - last_monitor_pass
                    >= MONITOR_DETECTION_INTERVAL
                ):
                    self.calibration.update(frame)

                    last_monitor_pass = now

                    calibrated = self.calibration.is_calibrated()

                obstacles = None

                if (
                    calibrated
                    and now - last_obstacle_pass
                    >= OBSTACLE_DETECTION_INTERVAL
                ):
                    small = self.calibration.transform_frame(
                        frame,
                        DETECTION_SCALE,
                    )

                    # Keep the sticker detector's idea of "what does
                    # white look like right now" in sync with what
                    # calibration is continuously learning off the
                    # ring - see Calibration.white_reference().
                    self.detector.set_white_reference(
                        self.calibration.white_reference()
                    )

                    obstacles = self.detector.detect(small)

                    last_obstacle_pass = time.monotonic()

                self._publish(
                    generation,
                    calibrated=calibrated,
                    frozen=False,
                    message="",
                    obstacles=obstacles,
                )

            except Exception as error:
                # Never let one bad frame kill the thread (and with
                # it, all vision until the next restart).
                now = time.monotonic()

                if now - last_error_print > 5.0:
                    print(
                        "[VISION] unexpected error, "
                        f"continuing: {error!r}"
                    )

                    last_error_print = now

                time.sleep(0.1)

    def _publish(
        self,
        generation,
        calibrated=None,
        frozen=None,
        message=None,
        obstacles=None,
    ):
        with self._lock:
            if generation != self._generation:
                # A reset happened while this pass was running; its
                # results describe the old state.
                return

            if calibrated is not None:
                self._calibrated = calibrated

            if frozen is not None:
                self._frozen = frozen

            if message is not None:
                self._message = message

            if obstacles is not None:
                self._obstacles = obstacles
