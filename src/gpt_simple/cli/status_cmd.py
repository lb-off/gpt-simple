"""
``gpt-simple status`` subcommand.

Reads multiple sources to give a complete picture of a training job:

  - ``.run_state.json``       (live; refreshed every ``logging_steps``)
  - ``trainer_state.json``    (canonical; one per checkpoint, written on save)
  - ``.shutdown_requested``   (set by ``gpt-simple stop``; advisory)
  - ``checkpoints/``          (list of recent checkpoints + final/)

The display falls back gracefully when files are missing (e.g. a job
that crashed before any checkpoint was saved still gets a useful
RunState-only display).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from gpt_simple._checkpoint import CheckpointManager, TrainerState
from gpt_simple._run_state import RunState

logger = logging.getLogger("gpt_simple")


def _pid_alive(pid: int) -> bool:
    """Check whether *pid* is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _human_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f}m"
    hours = minutes / 60
    if hours < 24:
        m = int(minutes) % 60
        return f"{int(hours)}h {m}m"
    days = int(hours) // 24
    h = int(hours) % 24
    return f"{days}d {h}h"


def _format_tokens(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# ---------------------------------------------------------------------------
# Information gathering
# ---------------------------------------------------------------------------


def _load_latest_trainer_state(
    output_dir: str,
) -> Tuple[Optional[TrainerState], Optional[Path]]:
    """Return (TrainerState, dir) for the most recent checkpoint, or (None, None).

    Uses the same ranking logic as the trainer's auto-resume: latest by
    step, with ``final/`` always last.  Returns ``None`` when the output
    directory doesn't exist yet or has no completed checkpoints.
    """
    out = Path(output_dir)
    if not out.exists():
        return None, None
    try:
        mgr = CheckpointManager(output_dir=out)
        ckpts = mgr.list_checkpoints()
    except Exception:
        return None, None
    if not ckpts:
        return None, None
    _step, _name, ckpt_path = ckpts[-1]
    try:
        return TrainerState.load(ckpt_path), ckpt_path
    except Exception:
        return None, None


def _list_recent_checkpoints(
    output_dir: str, limit: int = 5
) -> List[Tuple[int, str]]:
    """Return ``[(step, name), ...]`` for the *limit* most-recent checkpoints."""
    out = Path(output_dir)
    if not out.exists():
        return []
    try:
        mgr = CheckpointManager(output_dir=out)
        ckpts = mgr.list_checkpoints()
    except Exception:
        return []
    if not ckpts:
        return []
    tail = ckpts[-limit:]
    return [(step, name) for step, name, _ in tail]


def _shutdown_flag_path(output_dir: str) -> Optional[Path]:
    """Return the path to ``.shutdown_requested`` if it exists, else None."""
    p = Path(output_dir) / ".shutdown_requested"
    return p if p.is_file() else None


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _display_state(
    state: Optional[RunState],
    trainer_state: Optional[TrainerState],
    latest_ckpt_dir: Optional[Path],
    recent_ckpts: List[Tuple[int, str]],
    shutdown_requested: Optional[Path],
) -> None:
    """Print the state to stdout (rich or plain)."""

    alive = (
        _pid_alive(state.pid) if (state is not None and state.pid) else False
    )
    now = datetime.now(timezone.utc)

    started_str = "unknown"
    elapsed_str = ""
    if state is not None and state.started_at:
        try:
            started = datetime.fromisoformat(state.started_at)
            elapsed = (now - started).total_seconds()
            started_str = state.started_at
            elapsed_str = f" ({_human_duration(elapsed)} ago)"
        except (ValueError, TypeError):
            started_str = state.started_at

    # Status line --------------------------------------------------------
    if state is None:
        if trainer_state is not None:
            status_line = "NO RUN STATE (checkpoint only)"
        else:
            status_line = "EMPTY"
    elif state.status == "running" and not alive:
        status_line = f"CRASHED (PID {state.pid}, exited)"
    elif state.status == "running":
        status_line = f"RUNNING (PID {state.pid})"
    elif state.status == "error":
        pid_note = "exited" if not alive else "still alive"
        status_line = f"ERROR (PID {state.pid}, {pid_note})"
    elif state.status == "completed":
        status_line = "COMPLETED"
    elif state.status == "stopped":
        status_line = "STOPPED"
    elif state.status == "halted":
        status_line = "HALTED (bucket exhausted; needs attention)"
    else:
        status_line = state.status.upper()

    try:
        import rich  # noqa: F401
        _display_rich(
            state,
            trainer_state,
            latest_ckpt_dir,
            recent_ckpts,
            shutdown_requested,
            status_line,
            started_str,
            elapsed_str,
        )
    except ImportError:
        _display_plain(
            state,
            trainer_state,
            latest_ckpt_dir,
            recent_ckpts,
            shutdown_requested,
            status_line,
            started_str,
            elapsed_str,
        )


def _display_rich(
    state: Optional[RunState],
    trainer_state: Optional[TrainerState],
    latest_ckpt_dir: Optional[Path],
    recent_ckpts: List[Tuple[int, str]],
    shutdown_requested: Optional[Path],
    status_line: str,
    started_str: str,
    elapsed_str: str,
) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan", min_width=14)
    table.add_column()

    color_map = {
        "RUNNING": "green",
        "COMPLETED": "bold green",
        "STOPPED": "yellow",
        "ERROR": "bold red",
        "CRASHED": "bold red",
        "EMPTY": "dim",
        "NO RUN STATE": "yellow",
    }
    status_color = "white"
    for key, c in color_map.items():
        if key in status_line:
            status_color = c
            break

    table.add_row("Status", f"[{status_color}]{status_line}[/{status_color}]")
    if started_str and started_str != "unknown":
        table.add_row("Started", f"{started_str}{elapsed_str}")

    if shutdown_requested is not None:
        table.add_row(
            "Shutdown",
            f"[yellow]requested[/yellow] ({shutdown_requested})",
        )

    # Error display --------------------------------------------------
    if state is not None and state.status == "error" and state.error:
        table.add_row("Failed at", f"Step {state.global_step}")
        table.add_row("Error", state.error.strip().split("\n")[-1])
    else:
        _add_progress_rows_rich(table, state, trainer_state)

    if latest_ckpt_dir is not None:
        table.add_row("Checkpoint", str(latest_ckpt_dir))
    elif state is not None and state.latest_checkpoint:
        table.add_row("Checkpoint", state.latest_checkpoint)

    if recent_ckpts:
        table.add_row(
            "History", ", ".join(f"{name}@{step}" for step, name in recent_ckpts)
        )

    console.print(table)

    if state is not None and state.status == "error" and state.error:
        console.print("\n[bold red]Full traceback:[/bold red]")
        console.print(state.error)


def _add_progress_rows_rich(table, state: Optional[RunState], ts: Optional[TrainerState]) -> None:
    """Pick the best progress numbers from RunState first, falling back to TrainerState."""
    step = state.global_step if state is not None else (ts.step if ts is not None else 0)
    max_steps = state.max_steps if state is not None else 0
    loss = state.loss if state is not None else (ts.metrics.loss if ts is not None else float("inf"))
    lr = state.learning_rate if state is not None else (
        ts.metrics.learning_rate if ts is not None else 0.0
    )
    tok_per_sec = state.tokens_per_sec if state is not None else (
        ts.metrics.tokens_per_sec if ts is not None else 0.0
    )
    tokens_trained = state.tokens_trained if state is not None else (
        ts.tokens_trained if ts is not None else 0
    )

    if max_steps > 0:
        pct = f" ({step / max_steps * 100:.1f}%)"
        table.add_row("Progress", f"{step} / {max_steps} steps{pct}")
    else:
        table.add_row("Progress", f"{step} steps")
    if loss < float("inf"):
        table.add_row("Loss", f"{loss:.4f}")
    if lr > 0:
        table.add_row("LR", f"{lr:.2e}")
    if tok_per_sec > 0:
        table.add_row("Throughput", f"{tok_per_sec:,.0f} tok/s")
    if tokens_trained > 0:
        table.add_row("Tokens", f"{_format_tokens(tokens_trained)} trained")


def _display_plain(
    state: Optional[RunState],
    trainer_state: Optional[TrainerState],
    latest_ckpt_dir: Optional[Path],
    recent_ckpts: List[Tuple[int, str]],
    shutdown_requested: Optional[Path],
    status_line: str,
    started_str: str,
    elapsed_str: str,
) -> None:
    lines = [f"Status:     {status_line}"]
    if started_str and started_str != "unknown":
        lines.append(f"Started:    {started_str}{elapsed_str}")

    if shutdown_requested is not None:
        lines.append(f"Shutdown:   requested ({shutdown_requested})")

    if state is not None and state.status == "error" and state.error:
        lines.append(f"Failed at:  Step {state.global_step}")
        lines.append(f"Error:      {state.error.strip().split(chr(10))[-1]}")
    else:
        step = state.global_step if state is not None else (
            trainer_state.step if trainer_state is not None else 0
        )
        max_steps = state.max_steps if state is not None else 0
        loss = state.loss if state is not None else (
            trainer_state.metrics.loss if trainer_state is not None else float("inf")
        )
        lr = state.learning_rate if state is not None else (
            trainer_state.metrics.learning_rate if trainer_state is not None else 0.0
        )
        tok_per_sec = state.tokens_per_sec if state is not None else (
            trainer_state.metrics.tokens_per_sec if trainer_state is not None else 0.0
        )
        tokens_trained = state.tokens_trained if state is not None else (
            trainer_state.tokens_trained if trainer_state is not None else 0
        )

        if max_steps > 0:
            pct = f" ({step / max_steps * 100:.1f}%)"
            lines.append(f"Progress:   {step} / {max_steps} steps{pct}")
        else:
            lines.append(f"Progress:   {step} steps")
        if loss < float("inf"):
            lines.append(f"Loss:       {loss:.4f}")
        if lr > 0:
            lines.append(f"LR:         {lr:.2e}")
        if tok_per_sec > 0:
            lines.append(f"Throughput: {tok_per_sec:,.0f} tok/s")
        if tokens_trained > 0:
            lines.append(f"Tokens:     {_format_tokens(tokens_trained)} trained")

    if latest_ckpt_dir is not None:
        lines.append(f"Checkpoint: {latest_ckpt_dir}")
    elif state is not None and state.latest_checkpoint:
        lines.append(f"Checkpoint: {state.latest_checkpoint}")

    if recent_ckpts:
        history = ", ".join(f"{name}@{step}" for step, name in recent_ckpts)
        lines.append(f"History:    {history}")

    sys.stdout.write("\n".join(lines) + "\n")

    if state is not None and state.status == "error" and state.error:
        sys.stdout.write(f"\nFull traceback:\n{state.error}\n")


class StatusCommand:
    @staticmethod
    def register(subparsers: argparse._SubParsersAction) -> None:
        p = subparsers.add_parser(
            "status",
            help="Show training job status",
            description=(
                "Show the live status of a training job by reading "
                ".run_state.json, the most recent trainer_state.json, "
                "and the list of saved checkpoints."
            ),
        )
        p.add_argument(
            "--output_dir", type=str, default="./outputs",
            help="Training output directory (default: ./outputs)",
        )
        p.set_defaults(func=StatusCommand.run)

    @staticmethod
    def run(args: argparse.Namespace) -> None:
        state = RunState.read(args.output_dir)
        trainer_state, latest_ckpt_dir = _load_latest_trainer_state(
            args.output_dir
        )
        recent_ckpts = _list_recent_checkpoints(args.output_dir)
        shutdown_requested = _shutdown_flag_path(args.output_dir)

        if state is None and trainer_state is None:
            logger.error(
                f"No run state or checkpoints found in {args.output_dir}. "
                "Has a training job been started?"
            )
            sys.exit(1)

        _display_state(
            state,
            trainer_state,
            latest_ckpt_dir,
            recent_ckpts,
            shutdown_requested,
        )
