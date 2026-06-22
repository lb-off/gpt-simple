"""
Run-state persistence for gpt_simple training jobs.

This file is a thin LIVE mirror of the latest progress, written to
``output_dir/.run_state.json`` every ``logging_steps`` interval.  It is
read by ``gpt-simple status`` and ``gpt-simple stop`` to inspect a
running (or finished) job without touching any checkpoint files.

The canonical source of truth for resume is ``trainer_state.json`` next
to each checkpoint (see ``_checkpoint.py``).  RunState exists so the
status command can run in milliseconds without unpickling a model.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional


_STATE_FILENAME = ".run_state.json"


@dataclass
class RunState:
    """Live snapshot of the currently-running training job.

    Updated on every ``logging_steps`` interval (cheap JSON dump on the
    main rank).  Treat as advisory metadata; do not use this file for
    resume — use ``TrainerState`` from the latest checkpoint instead.
    """

    status: str = "running"  # running | completed | stopped | halted | error
    pid: int = 0
    started_at: str = ""
    updated_at: str = ""

    global_step: int = 0
    max_steps: int = 0
    loss: float = float("inf")
    learning_rate: float = 0.0
    tokens_trained: int = 0
    tokens_per_sec: float = 0.0

    error: Optional[str] = None
    latest_checkpoint: Optional[str] = None
    config_path: Optional[str] = None

    # -- I/O ----------------------------------------------------------------

    def write(self, output_dir: str) -> None:
        """Atomically write state as JSON (write-tmp then rename)."""
        os.makedirs(output_dir, exist_ok=True)
        self.updated_at = _now_iso()
        data = asdict(self)
        target = os.path.join(output_dir, _STATE_FILENAME)
        fd, tmp = tempfile.mkstemp(dir=output_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @classmethod
    def read(cls, output_dir: str) -> Optional["RunState"]:
        """Read state from disk, returning *None* when no file exists."""
        path = os.path.join(output_dir, _STATE_FILENAME)
        if not os.path.isfile(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            known = {f.name for f in cls.__dataclass_fields__.values()}
            # Tolerate the legacy ``checkpoint`` field name in case an old
            # state file is encountered.
            if "checkpoint" in data and "latest_checkpoint" not in data:
                data["latest_checkpoint"] = data.pop("checkpoint")
            return cls(**{k: v for k, v in data.items() if k in known})
        except (json.JSONDecodeError, TypeError):
            return None

    @classmethod
    def state_path(cls, output_dir: str) -> str:
        return os.path.join(output_dir, _STATE_FILENAME)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_state(
    *,
    max_steps: int,
    config_path: Optional[str] = None,
) -> RunState:
    """Create a fresh RunState for the beginning of a training job."""
    now = _now_iso()
    return RunState(
        status="running",
        pid=os.getpid(),
        started_at=now,
        updated_at=now,
        global_step=0,
        max_steps=max_steps,
        config_path=config_path,
    )
