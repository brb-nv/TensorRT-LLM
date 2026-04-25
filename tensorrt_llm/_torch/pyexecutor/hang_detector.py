import asyncio
import os
import threading
from contextlib import contextmanager
from typing import Callable, Optional

from tensorrt_llm._utils import print_all_stacks
from tensorrt_llm.logger import logger


class HangDetector:
    def __init__(
        self, timeout: Optional[int] = None, on_detected: Optional[Callable[[], None]] = None
    ):
        # DEBUG harmony-hang: env override wins, then explicit kwarg, then 300s default.
        env_timeout = os.environ.get("TLLM_HANG_DETECTION_TIMEOUT")
        if env_timeout is not None:
            try:
                self.timeout = int(env_timeout)
            except ValueError:
                self.timeout = timeout if timeout is not None else 300
        else:
            self.timeout = timeout if timeout is not None else 300
        assert self.timeout > 0, "timeout must be greater than 0"
        self.on_detected = on_detected or (lambda: None)
        self.task = None
        self.loop = None
        self.loop_thread = None
        self.lock = threading.Lock()
        self.active = False
        self._detected = False
        # DEBUG harmony-hang: periodic soft stack dumps at fractions of the timeout,
        # so we observe how the stall evolves rather than getting one snapshot at +timeout.
        # Disable by setting TLLM_HANG_DETECTOR_SOFT_DUMPS=0.
        self._soft_dumps_enabled = os.environ.get("TLLM_HANG_DETECTOR_SOFT_DUMPS", "1") != "0"
        # DEBUG harmony-hang: comma-separated seconds list, e.g. "30,60,90,120,180,240".
        soft_dump_env = os.environ.get("TLLM_HANG_DETECTOR_SOFT_DUMP_AT", "")
        if soft_dump_env:
            try:
                self._soft_dump_at = sorted({int(s) for s in soft_dump_env.split(",") if s.strip()})
            except ValueError:
                self._soft_dump_at = []
        else:
            # Default: dump at 30s, 60s, then every 60s up to (timeout - 30s).
            stops = [30, 60]
            t = 120
            while t < self.timeout:
                stops.append(t)
                t += 60
            self._soft_dump_at = [s for s in stops if 0 < s < self.timeout]

    def start(self):
        """Enable hang detection."""

        def run_loop():
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        self.active = True
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=run_loop, daemon=True, name="hang_detector_loop")
        self.loop_thread.start()

    async def _detect_hang(self):
        # DEBUG harmony-hang: periodic soft dumps before the kill threshold.
        if self._soft_dumps_enabled and self._soft_dump_at:
            elapsed = 0
            for stop in self._soft_dump_at:
                await asyncio.sleep(max(0, stop - elapsed))
                elapsed = stop
                logger.warning(
                    f"[hang_detector] No checkpoint for {elapsed}s "
                    f"(soft dump, kill at {self.timeout}s):"
                )
                print_all_stacks()
            await asyncio.sleep(max(0, self.timeout - elapsed))
        else:
            await asyncio.sleep(self.timeout)
        with self.lock:
            self._detected = True
            logger.error(f"Hang detected after {self.timeout} seconds.")
            print_all_stacks()
            self.on_detected()

    def detected(self):
        """Return True if hang is detected."""
        with self.lock:
            return self._detected

    def checkpoint(self):
        """Reset hang detection timer."""
        self.cancel_task()
        if self.active:
            self.task = asyncio.run_coroutine_threadsafe(self._detect_hang(), self.loop)

    def cancel_task(self):
        """Cancel the hang detection task."""
        if self.task is not None and not self.task.done():
            self.task.cancel()
            self.task = None

    @contextmanager
    def pause(self):
        """Pause hang detection in scope."""
        try:
            self.cancel_task()
            yield
        finally:
            self.checkpoint()

    def stop(self):
        """Stop hang detection."""
        self.active = False
        self.cancel_task()
        if self.loop is not None:
            # Cancel all pending tasks before stopping the loop
            def cancel_all_tasks():
                for task in asyncio.all_tasks(self.loop):
                    if not task.done():
                        task.cancel()
                self.loop.call_soon(self.loop.stop)

            self.loop.call_soon_threadsafe(cancel_all_tasks)

            if self.loop_thread is not None and self.loop_thread.is_alive():
                self.loop_thread.join()

            self.loop = None
            self.loop_thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        return False
