# main.py

import time

import cv2
import pygame

from camera import Camera
from calibration import Calibration
from detection import Detector
from vision import Vision
from physics import PhysicsWorld
from effects import Effects
from rendering import Renderer

from config import (
    CAMERA_URL,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    BALL_SPAWN_INTERVAL_SECONDS,
)


# If a spawn attempt fails (too many balls in play, or another ball
# is sitting right at the drop point), try again after this short
# delay instead of waiting a whole spawn interval.
_SPAWN_RETRY_SECONDS = 0.25

_WARNING_INTERVAL_SECONDS = 5.0


class Game:
    # ========================================================
    # Initialization
    # ========================================================

    def __init__(self):
        # Connecting happens on the camera's own thread, so the
        # window below comes up (and stays responsive to the
        # keyboard) even if the phone isn't streaming yet.
        self.camera = Camera(CAMERA_URL)

        self.calibration = Calibration()
        self.detector = Detector()

        # Camera -> calibration -> sticker detection, all on a
        # background thread; the loop below only ever reads its
        # latest published result.
        self.vision = Vision(
            self.camera,
            self.calibration,
            self.detector,
        )

        self.physics = PhysicsWorld(
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
        )

        self.effects = Effects()

        self.renderer = Renderer()

        self.running = True

        self.was_calibrated = False

        # Very negative so the first ball drops the moment
        # calibration locks on.
        self.last_ball_spawn = -1e9

        self.last_warning = 0.0

        self.vision.start()

    # ========================================================
    # Main loop
    # ========================================================

    def run(self):
        try:
            while self.running:
                self._handle_events()

                if not self.running:
                    break

                # The per-frame pipeline is wrapped defensively:
                # this runs unattended in a busy exhibition hall,
                # and one unexpected error must never take the
                # installation down. Worst case we skip a frame.
                try:
                    self._run_one_frame()
                except Exception as error:
                    print(
                        "[ERROR] unexpected failure in main "
                        f"loop, continuing: {error!r}"
                    )

                    time.sleep(0.05)

        finally:
            self.close()

    def _run_one_frame(self):
        (
            calibrated,
            frozen,
            obstacles,
            message,
        ) = self.vision.snapshot()

        current_time = time.monotonic()

        self._maybe_warn(current_time, frozen, message)

        if not calibrated:
            if self.was_calibrated:
                print(
                    "[CALIBRATION] lost/reset - "
                    "waiting for the screen to be recognized..."
                )

                self.was_calibrated = False

            self.renderer.render_calibration()

            # Nothing moves while calibrating; no need to spin.
            self.renderer.clock.tick(30)

            return

        if not self.was_calibrated:
            print("[CALIBRATION] locked on - starting game.")

            self.was_calibrated = True

        delta_time = self.renderer.get_delta_time()

        self.physics.update(delta_time, obstacles)

        self._spawn_ball(current_time)

        self.effects.update(
            delta_time,
            self.physics.balls,
            self.physics.events,
        )

        self.renderer.render(
            self.physics.balls,
            obstacles,
            self.effects,
        )

    # ========================================================
    # Warnings
    # ========================================================

    def _maybe_warn(self, current_time, frozen, message):
        if (
            current_time - self.last_warning
            < _WARNING_INTERVAL_SECONDS
        ):
            return

        if frozen:
            print(
                "[WARNING] Camera feed looks FROZEN "
                "(no motion detected for a while) - "
                "check the phone/app/connection. "
                "Calibration and obstacle detection are "
                "paused until a live frame is seen again."
            )
        elif message:
            print(f"[WARNING] {message}")
        else:
            return

        self.last_warning = current_time

    # ========================================================
    # Ball spawning
    # ========================================================

    def _spawn_ball(self, current_time):
        # Balls arrive on their OWN clock, one every
        # BALL_SPAWN_INTERVAL_SECONDS, whether or not earlier balls
        # have left. A ball wedged between stickers just stays
        # (until the physics removes it as stuck) and the next one
        # still comes on schedule.
        if (
            current_time - self.last_ball_spawn
            < BALL_SPAWN_INTERVAL_SECONDS
        ):
            return

        if self.physics.spawn_ball() is not None:
            self.last_ball_spawn = current_time
        else:
            self.last_ball_spawn = (
                current_time
                - BALL_SPAWN_INTERVAL_SECONDS
                + _SPAWN_RETRY_SECONDS
            )

    # ========================================================
    # Events
    # ========================================================

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False

                elif event.key in (
                    pygame.K_c,
                    pygame.K_ESCAPE,
                ):
                    self._reset_calibration()

    # ========================================================
    # Calibration reset
    # ========================================================

    def _reset_calibration(self):
        self.vision.request_reset()

        self.physics.clear()

        self.effects.clear()

        self.renderer.force_full_repaint()

        self.last_ball_spawn = -1e9

    # ========================================================
    # Shutdown
    # ========================================================

    def close(self):
        try:
            self.vision.stop()
        except Exception:
            pass

        try:
            self.camera.release()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        try:
            self.renderer.close()
        except Exception:
            pass


def main():
    game = Game()
    game.run()


if __name__ == "__main__":
    main()
