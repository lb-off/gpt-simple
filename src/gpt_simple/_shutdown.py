"""
ShutdownCoordinator: graceful, distributed-safe stop for gpt_simple.

Listens for orchestrator signals (``SIGTERM``, ``SIGINT``, ``SIGUSR1``,
``SIGUSR2``), a walltime budget (auto-detected from
``SLURM_JOB_END_TIME`` or set explicitly via
``TrainingConfig.max_runtime_seconds``), and a local flag file
(``output_dir/.shutdown_requested``).  Coordinates a SINGLE shutdown
decision across all ranks via an all-reduce on every check, so
multi-GPU runs cannot half-stop with one rank still in the forward pass.

Typical usage from the training loop::

    coord = ShutdownCoordinator(
        accelerator=accelerator,
        output_dir=cfg.training.output_dir,
        max_runtime_seconds=cfg.training.max_runtime_seconds,
        walltime_reserve_seconds=cfg.training.walltime_reserve_seconds,
    )
    coord.install_signal_handlers()
    coord.clear_flag_file()   # wipe stale flag from a previous run

    while step < max_steps:
        if coord.should_shutdown():
            save_shutdown_checkpoint(...)
            coord.clear_flag_file()
            break
        ...  # forward / backward / step

A *second* ``SIGINT`` (Ctrl-C twice) forces an immediate ``os._exit`` so
the user can always bail out if the graceful path itself hangs.
"""

from __future__ import annotations

import logging
import os
import signal
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

logger = logging.getLogger("gpt_simple")


# Signals we listen for.  Order is irrelevant; all of them mean
# "save and exit cleanly", except a second SIGINT which force-exits.
_HANDLED_SIGNALS: Dict[int, str] = {
    signal.SIGTERM: "SIGTERM",
    signal.SIGINT: "SIGINT",
    signal.SIGUSR1: "SIGUSR1",  # SLURM walltime warning by convention
    signal.SIGUSR2: "SIGUSR2",
}


_FLAG_FILENAME = ".shutdown_requested"
_DEFAULT_RESERVE_SECONDS = 300  # 5 minutes; enough to save a large checkpoint
_FORCE_EXIT_CODE = 130


class ShutdownCoordinator:
    """Coordinates graceful shutdown across signals, walltime, and ranks.

    Parameters
    ----------
    accelerator
        The ``Accelerator`` instance (or ``None`` for unit tests / pure
        single-process runs).  Used only to read ``device`` and
        ``is_main_process``; all-reduce uses ``torch.distributed`` directly.
    output_dir
        Where to look for the ``.shutdown_requested`` flag file.
    max_runtime_seconds
        Optional explicit wall-clock budget.  If ``None``, we try to auto
        detect from ``SLURM_JOB_END_TIME`` (Unix timestamp set by SLURM).
        If neither is present, the walltime watchdog is disabled.
    walltime_reserve_seconds
        How far in front of the deadline to trigger shutdown, giving the
        loop enough time to save before the orchestrator kills us.
    check_flag_file
        If ``True`` (default), poll ``output_dir/.shutdown_requested`` on
        every ``should_shutdown()`` call.  Disable for tests that don't
        care about that path.
    loop_start_monotonic
        Optional override for the monotonic clock anchor used to compute
        the deadline from ``max_runtime_seconds``.  Defaults to "now".
    """

    def __init__(
        self,
        accelerator: Any = None,
        output_dir: Union[str, Path] = ".",
        max_runtime_seconds: Optional[int] = None,
        walltime_reserve_seconds: int = _DEFAULT_RESERVE_SECONDS,
        check_flag_file: bool = True,
        loop_start_monotonic: Optional[float] = None,
    ):
        self.accelerator = accelerator
        self.output_dir = Path(output_dir)
        self.walltime_reserve_seconds = max(0, int(walltime_reserve_seconds))
        self.check_flag_file = check_flag_file

        self._loop_start_monotonic = (
            float(loop_start_monotonic)
            if loop_start_monotonic is not None
            else time.monotonic()
        )
        self._deadline_monotonic = self._compute_deadline(max_runtime_seconds)

        self._local_flag = False
        self._reason: Optional[str] = None
        self._sigint_count = 0
        self._installed_handlers: Dict[int, Any] = {}
        self._flag_file = self.output_dir / _FLAG_FILENAME

        if self._deadline_monotonic is not None and self._is_main():
            remaining = self._deadline_monotonic - time.monotonic()
            logger.info(
                f"Walltime watchdog armed: deadline in {remaining:.0f}s "
                f"(reserve={self.walltime_reserve_seconds}s)"
            )

    # ------------------------------------------------------------------ #
    #  Deadline resolution
    # ------------------------------------------------------------------ #

    def _compute_deadline(
        self, explicit_max: Optional[int]
    ) -> Optional[float]:
        """Resolve the monotonic deadline (or ``None`` if no walltime).

        Resolution order:
          1. ``max_runtime_seconds`` argument (explicit config)
          2. ``SLURM_JOB_END_TIME`` env var (SLURM clusters)
          3. ``GPT_SIMPLE_MAX_RUNTIME`` env var (generic / Kubernetes /
             Docker)
          4. None — watchdog disabled
        """
        if explicit_max is not None and explicit_max > 0:
            return self._loop_start_monotonic + float(explicit_max)

        slurm_end = os.environ.get("SLURM_JOB_END_TIME")
        if slurm_end:
            try:
                unix_deadline = int(slurm_end)
            except (TypeError, ValueError):
                logger.warning(
                    f"Could not parse SLURM_JOB_END_TIME={slurm_end!r}; "
                    "walltime watchdog disabled."
                )
                return None
            # Convert wall-clock Unix → monotonic equivalent
            now_unix = time.time()
            now_mono = time.monotonic()
            return now_mono + (unix_deadline - now_unix)

        generic = os.environ.get("GPT_SIMPLE_MAX_RUNTIME")
        if generic:
            try:
                seconds = int(generic)
            except (TypeError, ValueError):
                logger.warning(
                    f"Could not parse GPT_SIMPLE_MAX_RUNTIME={generic!r}; "
                    "walltime watchdog disabled."
                )
                return None
            if seconds > 0:
                return self._loop_start_monotonic + float(seconds)

        return None

    # ------------------------------------------------------------------ #
    #  Signal handling
    # ------------------------------------------------------------------ #

    def install_signal_handlers(self) -> None:
        """Install handlers for SIGTERM, SIGINT, SIGUSR1, SIGUSR2.

        Saves the previous handlers so they can be restored by
        :py:meth:`uninstall_signal_handlers`.  Silently skips signals that
        cannot be installed (e.g. when called from a non-main thread).
        """
        for sig, name in _HANDLED_SIGNALS.items():
            try:
                prev = signal.signal(sig, self._handle_signal)
                self._installed_handlers[sig] = prev
            except (OSError, ValueError) as exc:
                logger.debug(f"Could not install {name} handler: {exc}")

    def uninstall_signal_handlers(self) -> None:
        """Restore the signal handlers that were active before install."""
        for sig, prev in self._installed_handlers.items():
            try:
                signal.signal(sig, prev)
            except (OSError, ValueError):
                pass
        self._installed_handlers.clear()

    def _handle_signal(self, signum: int, frame) -> None:  # noqa: ARG002
        name = _HANDLED_SIGNALS.get(signum, str(signum))

        # Two consecutive SIGINTs (Ctrl-C twice) means "I really mean it".
        # We bypass the graceful path so the user can always escape a hang.
        if signum == signal.SIGINT:
            self._sigint_count += 1
            if self._sigint_count >= 2:
                logger.error(
                    "Second SIGINT received — force-exiting (use Ctrl-\\ if "
                    "this also hangs)."
                )
                os._exit(_FORCE_EXIT_CODE)

        if not self._local_flag:
            logger.warning(
                f"Shutdown signal {name} received — will save a shutdown "
                "checkpoint and exit cleanly."
            )
            self._local_flag = True
            self._reason = f"signal:{name}"

    # ------------------------------------------------------------------ #
    #  Programmatic and passive triggers
    # ------------------------------------------------------------------ #

    def request_shutdown(self, reason: str) -> None:
        """Programmatic shutdown trigger (e.g. dataset exhaustion)."""
        if not self._local_flag:
            logger.warning(f"Shutdown requested: {reason}")
            self._local_flag = True
            self._reason = reason

    def _walltime_exceeded(self) -> bool:
        if self._deadline_monotonic is None:
            return False
        threshold = self._deadline_monotonic - self.walltime_reserve_seconds
        return time.monotonic() >= threshold

    def _flag_file_present(self) -> bool:
        if not self.check_flag_file:
            return False
        try:
            return self._flag_file.is_file()
        except OSError:
            return False

    # ------------------------------------------------------------------ #
    #  Coordination
    # ------------------------------------------------------------------ #

    def should_shutdown(self) -> bool:
        """Return ``True`` if any rank wants to shut down.

        Updates the local flag from passive sources (walltime, flag file)
        and then performs a small all-reduce so the answer is consistent
        across ranks.  Safe to call every step — the overhead is one
        ``int32`` all-reduce.
        """
        # Passive sources update local flag first.
        if not self._local_flag and self._walltime_exceeded():
            self._local_flag = True
            self._reason = "walltime"
            if self._is_main():
                logger.warning(
                    "Walltime deadline reached "
                    f"(reserve={self.walltime_reserve_seconds}s) — shutting down."
                )

        if not self._local_flag and self._flag_file_present():
            self._local_flag = True
            self._reason = "flag_file"
            if self._is_main():
                logger.warning(
                    f"Shutdown flag file detected at {self._flag_file}."
                )

        # All-reduce across ranks.  Any rank set ⇒ everyone shuts down.
        local_val = 1 if self._local_flag else 0
        global_val = self._all_reduce_max(local_val)
        if global_val > 0 and not self._local_flag:
            self._local_flag = True
            self._reason = self._reason or "peer_request"

        return self._local_flag

    def _all_reduce_max(self, value: int) -> int:
        try:
            import torch.distributed as dist
        except ImportError:
            return value

        if not dist.is_available() or not dist.is_initialized():
            return value

        try:
            t = torch.tensor(value, dtype=torch.int32)
            device = getattr(self.accelerator, "device", None)
            if device is not None:
                t = t.to(device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            return int(t.item())
        except Exception as exc:
            logger.warning(
                f"all_reduce for shutdown coordination failed ({exc}); "
                "falling back to local-only decision."
            )
            return value

    # ------------------------------------------------------------------ #
    #  Introspection
    # ------------------------------------------------------------------ #

    @property
    def reason(self) -> Optional[str]:
        """Why this coordinator is requesting shutdown (or ``None``)."""
        return self._reason

    @property
    def deadline_monotonic(self) -> Optional[float]:
        return self._deadline_monotonic

    def remaining_seconds(self) -> Optional[float]:
        """Seconds until the walltime deadline (``None`` if no deadline)."""
        if self._deadline_monotonic is None:
            return None
        return max(0.0, self._deadline_monotonic - time.monotonic())

    # ------------------------------------------------------------------ #
    #  Housekeeping
    # ------------------------------------------------------------------ #

    def clear_flag_file(self) -> None:
        """Remove the flag file if it exists; quietly ignore failures."""
        try:
            self._flag_file.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning(
                f"Could not remove flag file {self._flag_file}: {exc}"
            )

    # ------------------------------------------------------------------ #
    #  Internals
    # ------------------------------------------------------------------ #

    def _is_main(self) -> bool:
        if self.accelerator is None:
            return True
        return bool(getattr(self.accelerator, "is_main_process", True))


__all__ = ["ShutdownCoordinator"]
