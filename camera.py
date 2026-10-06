import threading
import time

import cv2
import numpy as np

from config import (
    CAMERA_FROZEN_DIFF_THRESHOLD,
    CAMERA_FROZEN_SECONDS,
    CAMERA_CONNECT_RETRY_SECONDS,
)


class Camera:
    """
    Wrapper around cv2.VideoCapture for an HTTP/MJPEG camera stream,
    with a few exhibition-day survival features:

    - Frame grabbing runs on its own background thread that reads as
      fast as the stream delivers and keeps only the SINGLE most
      recent frame. cv2.VideoCapture's own buffering on network
      streams often ignores CAP_PROP_BUFFERSIZE; if the consumer is
      ever briefly slower than the stream, frames queue up upstream
      and every later read() returns an ever-staler frame ("the game
      gets slower and slower"). Draining continuously and handing
      out only the newest frame makes that impossible.

    - Connecting also happens on that thread and never blocks the
      caller: the game window comes up immediately (and stays
      responsive to the keyboard) even if the phone/app isn't
      streaming yet. It keeps retrying forever, both at startup and
      after a mid-show drop.

    - Every frame gets an increasing id (see read_with_id), so the
      vision thread can skip a frame it has already processed
      instead of redoing the same work on it.

    - Frozen-frame detection: apps that pause in the background
      often keep re-sending the last JPEG, which cv2.VideoCapture
      happily "reads" forever. We track whether the image content
      actually changes and expose is_frozen() so callers can refuse
      to trust a feed that looks alive but isn't.
    """

    MAX_CONSECUTIVE_FAILURES = 5
    RECONNECT_BACKOFF_SECONDS = 1.0

    def __init__(self, url: str):
        self.url = url
        self.cap = None

        self._stop_event = threading.Event()

        self._state_lock = threading.Lock()
        self._last_thumb = None
        self._last_change_time = time.monotonic()

        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._latest_id = 0
        self._latest_error = "Camera not connected yet."

        self._thread = threading.Thread(
            target=self._run,
            name="camera-reader",
            daemon=True,
        )
        self._thread.start()

    # ========================================================
    # Connection
    # ========================================================

    def _connect_with_retry(self):
        """
        Keeps trying to open the stream until it works or the
        camera is released. This runs unattended in a booth: the
        phone/app may not be streaming yet, or wifi may take a
        moment, and nobody may be there to relaunch anything.
        """
        attempt = 0

        while not self._stop_event.is_set():
            attempt += 1

            try:
                self._open()
                print("[CAMERA] connected.")
                return
            except Exception as error:
                with self._frame_lock:
                    self._latest_error = (
                        f"Camera not connected ({error})"
                    )

                print(
                    f"[CAMERA] connect attempt {attempt} failed "
                    f"({error}) - retrying in "
                    f"{CAMERA_CONNECT_RETRY_SECONDS:.1f}s..."
                )

                self._stop_event.wait(
                    CAMERA_CONNECT_RETRY_SECONDS
                )

    def _open(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        self.cap = cv2.VideoCapture(self.url)

        # Cheap extra defense on backends that honor it; the reader
        # thread is what actually guarantees we never fall behind.
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not connect to camera stream: {self.url}"
            )

    # ========================================================
    # Background thread
    # ========================================================

    def _run(self):
        self._connect_with_retry()

        consecutive_failures = 0

        while not self._stop_event.is_set():
            try:
                success, frame = self.cap.read()
            except Exception as error:
                success, frame = False, None

                with self._frame_lock:
                    self._latest_error = (
                        f"Camera read error: {error}"
                    )

            if not success or frame is None:
                consecutive_failures += 1

                with self._frame_lock:
                    self._latest_error = (
                        "Failed to read frame from camera."
                    )

                if (
                    consecutive_failures
                    >= self.MAX_CONSECUTIVE_FAILURES
                ):
                    self._stop_event.wait(
                        self.RECONNECT_BACKOFF_SECONDS
                    )

                    self._connect_with_retry()

                    consecutive_failures = 0
                else:
                    # A failed read on some backends returns
                    # instantly; don't spin a core on it.
                    self._stop_event.wait(0.01)

                continue

            consecutive_failures = 0

            self._update_freeze_state(frame)

            # Publish only the newest frame; anything the consumer
            # hasn't taken yet is simply replaced.
            with self._frame_lock:
                self._latest_frame = frame
                self._latest_id += 1
                self._latest_error = None

    # ========================================================
    # Public API
    # ========================================================

    def read_with_id(self):
        """Newest frame and its id; raises RuntimeError if none yet."""
        with self._frame_lock:
            frame = self._latest_frame
            frame_id = self._latest_id
            error = self._latest_error

        if frame is None:
            raise RuntimeError(
                error or "No frame received from camera yet."
            )

        return frame, frame_id

    def read(self):
        return self.read_with_id()[0]

    def is_frozen(self):
        with self._state_lock:
            last_change = self._last_change_time

        return (
            time.monotonic() - last_change
        ) > CAMERA_FROZEN_SECONDS

    def release(self):
        self._stop_event.set()

        # Let the reader thread notice and finish before the
        # capture object is torn down underneath it.
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

    # ========================================================
    # Frozen-frame detection
    # ========================================================

    def _update_freeze_state(self, frame):
        # Shrink FIRST (INTER_AREA averages, so it also suppresses
        # pixel noise) and only then convert to gray - converting a
        # full frame just to throw almost all of it away is wasted
        # work on the capture thread.
        thumb = cv2.cvtColor(
            cv2.resize(
                frame,
                (32, 18),
                interpolation=cv2.INTER_AREA,
            ),
            cv2.COLOR_BGR2GRAY,
        ).astype(np.int16)

        now = time.monotonic()

        with self._state_lock:
            if self._last_thumb is None:
                self._last_change_time = now
            else:
                diff = float(
                    np.mean(np.abs(thumb - self._last_thumb))
                )

                if diff > CAMERA_FROZEN_DIFF_THRESHOLD:
                    self._last_change_time = now

            self._last_thumb = thumb
