"""
``gpt-simple stop`` subcommand.

Gracefully stops a running training job by creating a shutdown flag file
and sending SIGTERM.  ``--force`` sends SIGKILL instead.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time

from gpt_simple._run_state import RunState

logger = logging.getLogger("gpt_simple")

_GRACEFUL_TIMEOUT = 600  # 10 minutes


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


class StopCommand:
    @staticmethod
    def register(subparsers: argparse._SubParsersAction) -> None:
        p = subparsers.add_parser(
            "stop",
            help="Stop a running training job",
            description=(
                "Gracefully stop a training job (flag file + SIGTERM). "
                "Use --force for immediate SIGKILL."
            ),
        )
        p.add_argument(
            "--output_dir", type=str, default="./outputs",
            help="Training output directory (default: ./outputs)",
        )
        p.add_argument(
            "--force", action="store_true",
            help="Send SIGKILL instead of graceful shutdown",
        )
        p.set_defaults(func=StopCommand.run)

    @staticmethod
    def run(args: argparse.Namespace) -> None:
        state = RunState.read(args.output_dir)
        if state is None:
            logger.error(
                f"No run state found in {args.output_dir}. "
                "Has a training job been started?"
            )
            sys.exit(1)

        pid = state.pid
        if not _pid_alive(pid):
            logger.warning(f"Process {pid} is not running (already exited).")
            return

        if args.force:
            logger.warning(f"Sending SIGKILL to PID {pid}...")
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError as exc:
                logger.error(f"Failed to kill process: {exc}")
                sys.exit(1)
            state.status = "stopped"
            state.write(args.output_dir)
            logger.info(f"Process {pid} killed.")
            return

        # Graceful shutdown: create flag file + SIGTERM
        flag_file = os.path.join(args.output_dir, ".shutdown_requested")
        with open(flag_file, "w") as f:
            f.write(str(os.getpid()))

        logger.info(f"Sending SIGTERM to PID {pid} (graceful shutdown)...")
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as exc:
            logger.error(f"Failed to send SIGTERM: {exc}")
            sys.exit(1)

        # Poll until the process exits
        logger.info(f"Waiting for process to exit (timeout: {_GRACEFUL_TIMEOUT}s)...")
        deadline = time.monotonic() + _GRACEFUL_TIMEOUT
        while time.monotonic() < deadline:
            if not _pid_alive(pid):
                break
            time.sleep(2)
        else:
            logger.warning(
                f"Process {pid} did not exit within {_GRACEFUL_TIMEOUT}s. "
                "Use --force to send SIGKILL."
            )
            return

        # Refresh state from disk (the training process should have written it)
        updated = RunState.read(args.output_dir)
        if updated and updated.status == "running":
            updated.status = "stopped"
            updated.write(args.output_dir)

        logger.info(f"Process {pid} has exited.")
