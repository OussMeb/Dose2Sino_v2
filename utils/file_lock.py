#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File locking utilities for safe multiprocess access.

Prevents multiple workers from writing to the same patient directory simultaneously.
"""

import os
import time
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)


class FileLock:
    """
    Simple file-based lock for multiprocess synchronization.

    Uses os.rename() which is atomic on POSIX systems.
    """

    def __init__(self, lock_path: Path, timeout: float = 300.0, poll_interval: float = 0.1):
        """
        Initialize file lock.

        Args:
            lock_path: Path to lock file (typically patient_out_dir / ".lock")
            timeout: Maximum time to wait for lock (seconds)
            poll_interval: Time between lock acquisition attempts (seconds)
        """
        self.lock_path = Path(lock_path)
        self.temp_lock_path = self.lock_path.parent / f".lock.{os.getpid()}.tmp"
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.acquired = False
        self.start_time = None

    def acquire(self) -> bool:
        """
        Acquire the lock with timeout.

        Returns:
            True if lock acquired, False if timeout
        """
        self.start_time = time.time()

        while True:
            try:
                # Create lock file atomically (fails if exists)
                # First create temp file with unique name
                with open(self.temp_lock_path, 'w') as f:
                    f.write(f"{os.getpid()}\n{time.time()}\n")

                # Try to rename to final lock path (atomic)
                os.rename(self.temp_lock_path, self.lock_path)

                self.acquired = True
                logger.debug(f"[LOCK] Acquired: {self.lock_path} (pid={os.getpid()})")
                return True

            except FileExistsError:
                # Lock already exists
                elapsed = time.time() - self.start_time

                if elapsed > self.timeout:
                    logger.error(f"[LOCK] Timeout acquiring {self.lock_path} after {elapsed:.1f}s")
                    return False

                # Check if lock holder is still alive
                try:
                    if self.lock_path.exists():
                        with open(self.lock_path, 'r') as f:
                            lock_pid = int(f.readline().strip())

                        # Check if process still exists (Unix only)
                        try:
                            os.kill(lock_pid, 0)  # Signal 0 = check if process exists
                        except (OSError, ProcessLookupError):
                            # Process dead, remove stale lock
                            logger.warning(f"[LOCK] Removing stale lock (dead pid {lock_pid})")
                            self.lock_path.unlink()
                            continue
                except Exception as e:
                    logger.warning(f"[LOCK] Error checking lock holder: {e}")

                # Wait before retry
                time.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"[LOCK] Error acquiring lock: {e}")
                return False

    def release(self):
        """Release the lock."""
        if self.acquired and self.lock_path.exists():
            try:
                self.lock_path.unlink()
                logger.debug(f"[LOCK] Released: {self.lock_path} (pid={os.getpid()})")
                self.acquired = False
            except Exception as e:
                logger.error(f"[LOCK] Error releasing lock: {e}")

    def __enter__(self):
        """Context manager entry."""
        if not self.acquire():
            raise TimeoutError(f"Failed to acquire lock {self.lock_path} within {self.timeout}s")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


@contextmanager
def acquire_patient_lock(patient_out_dir: Path, timeout: float = 300.0) -> Generator[FileLock, None, None]:
    """
    Context manager for acquiring patient directory lock.

    Usage:
        with acquire_patient_lock(patient_out_dir):
            # Safe to write to patient_out_dir
            pass

    Args:
        patient_out_dir: Patient output directory
        timeout: Maximum time to wait for lock

    Yields:
        FileLock instance

    Raises:
        TimeoutError: If lock cannot be acquired within timeout
    """
    lock_path = Path(patient_out_dir) / ".lock"
    lock = FileLock(lock_path, timeout=timeout)

    try:
        if not lock.acquire():
            raise TimeoutError(f"Failed to acquire lock for {patient_out_dir}")
        yield lock
    finally:
        lock.release()

