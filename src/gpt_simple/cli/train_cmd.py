"""
``gpt-simple train`` subcommand.

Single-GPU:   gpt-simple train --config config.yaml
Multi-GPU:    gpt-simple train --config config.yaml --nproc_per_node 4

When ``--nproc_per_node`` > 1, the command launches workers via
``torch.distributed.run.run()`` (the torchrun library API) instead
of calling ``train()`` in-process.
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger("gpt_simple")


class TrainCommand:
    @staticmethod
    def register(subparsers: argparse._SubParsersAction) -> None:
        p = subparsers.add_parser(
            "train",
            help="Run a pretraining job",
            description="Launch a GPT-Simple pretraining run (single or multi-GPU).",
        )
        p.add_argument(
            "--config", type=str, required=True,
            help="Path to YAML/JSON config file",
        )
        p.add_argument(
            "--force", action="store_true",
            help=(
                "Wipe any existing checkpoints/run-state in output_dir and "
                "start training from scratch.  Equivalent to manually "
                "deleting the output dir and passing --training.resume scratch."
            ),
        )
        p.add_argument(
            "--allow-bucket-exhaustion", action="store_true",
            help=(
                "Override data.allow_bucket_exhaustion to true for this run. "
                "Permits buckets to run out of tokens mid-curriculum (the "
                "data loader drops them and renormalises the mix)."
            ),
        )
        p.add_argument(
            "--allow-budget-mismatch", action="store_true",
            help=(
                "Override data.allow_budget_mismatch to true for this run. "
                "Permits a curriculum whose total token demand differs from "
                "what max_steps will consume."
            ),
        )
        p.add_argument(
            "--skip-runtime-probe", action="store_true",
            help=(
                "Skip the synthetic forward/backward/step that runs after "
                "accelerator init.  Use when very GPU-memory-tight and the "
                "2x model footprint during the probe would itself OOM."
            ),
        )
        p.add_argument(
            "--nproc_per_node", type=int, default=1,
            help="Number of GPUs per node (>1 uses torchrun under the hood)",
        )
        p.add_argument(
            "--nnodes", type=int, default=1,
            help="Number of nodes for distributed training",
        )
        p.add_argument(
            "--node_rank", type=int, default=0,
            help="Rank of this node",
        )
        p.add_argument(
            "--master_addr", type=str, default="127.0.0.1",
            help="Master address for distributed training",
        )
        p.add_argument(
            "--master_port", type=str, default="29500",
            help="Master port for distributed training",
        )

        # Config overrides
        p.add_argument("--training.max_steps", type=int, default=None)
        p.add_argument("--training.output_dir", type=str, default=None)
        p.add_argument(
            "--training.resume",
            type=str,
            default=None,
            help="'auto' (default), 'scratch', or a checkpoint path",
        )
        p.add_argument(
            "--training.keep_last_k",
            type=int,
            default=None,
            help="Rolling-buffer size for checkpoint retention",
        )
        p.add_argument(
            "--training.max_runtime_seconds",
            type=int,
            default=None,
            help=(
                "Max wall-clock seconds before saving a shutdown checkpoint "
                "and exiting (auto-detected from SLURM_JOB_END_TIME if unset)"
            ),
        )
        p.add_argument(
            "--training.walltime_reserve_seconds",
            type=int,
            default=None,
            help="Buffer seconds before the walltime deadline (default 300)",
        )
        p.add_argument("--training.seed", type=int, default=None)
        p.add_argument("--training.wandb_project", type=str, default=None)
        p.add_argument("--training.wandb_run_name", type=str, default=None)
        p.add_argument("--optimizer.learning_rate", type=float, default=None)
        p.add_argument("--data.path", type=str, default=None)
        p.add_argument("--data.tokenizer", type=str, default=None)
        p.add_argument("--data.format", type=str, default=None)

        p.set_defaults(func=TrainCommand.run)

    @staticmethod
    def run(args: argparse.Namespace) -> None:
        if args.nproc_per_node > 1 or args.nnodes > 1:
            _launch_distributed(args)
        else:
            _launch_single(args)


def _collect_overrides(args: argparse.Namespace) -> dict[str, object]:
    """Extract all dotted CLI overrides that were explicitly set.

    Convention: any argparse attribute whose name contains a ``.`` is a
    config override (e.g. ``training.max_steps``).  Attributes left at
    their ``None`` default are excluded.
    """
    return {
        key: val
        for key, val in vars(args).items()
        if "." in key and val is not None
    }


def _apply_overrides(cfg, overrides: dict[str, object]) -> None:
    """Apply dotted CLI overrides to a Config object."""
    for key, val in overrides.items():
        section, attr = key.split(".", 1)
        setattr(getattr(cfg, section), attr, val)
    cfg._validate_schedule()


def _clobber_output_dir(output_dir: str) -> None:
    """Delete checkpoints and ephemeral state in ``output_dir`` (in-place).

    Used by ``--force`` to make ``resume='scratch'`` safe even when prior
    runs left checkpoints behind.  We do NOT delete the ``output_dir``
    itself (the user may have logs, configs or wandb metadata in there)
    — only the artifacts that would block a fresh run:

      - ``checkpoints/`` (every saved checkpoint)
      - ``.run_state.json`` (live status snapshot)
      - ``.shutdown_requested`` (any stale graceful-shutdown flag)
      - ``.stop_chain`` (any stale resume-chain stop marker)
      - ``tokenizer/`` (re-saved on first checkpoint anyway)
    """
    import os
    import shutil
    from pathlib import Path

    out = Path(output_dir)
    if not out.exists():
        return

    targets = [
        out / "checkpoints",
        out / "tokenizer",
        out / ".run_state.json",
        out / ".shutdown_requested",
        out / ".stop_chain",
    ]
    for t in targets:
        if t.is_dir():
            shutil.rmtree(t)
        elif t.exists():
            os.remove(t)


def _launch_single(args: argparse.Namespace) -> None:
    """In-process single-GPU training."""
    from gpt_simple.config import Config
    from gpt_simple.train import train, write_error_state

    cfg = Config.from_file(args.config)
    _apply_overrides(cfg, _collect_overrides(args))

    if getattr(args, "force", False):
        logger.warning(
            f"--force: wiping prior checkpoints and run state in "
            f"{cfg.training.output_dir} (resume forced to 'scratch')."
        )
        _clobber_output_dir(cfg.training.output_dir)
        cfg.training.resume = "scratch"

    if getattr(args, "allow_bucket_exhaustion", False):
        cfg.data.allow_bucket_exhaustion = True
    if getattr(args, "allow_budget_mismatch", False):
        cfg.data.allow_budget_mismatch = True

    try:
        result = train(
            config=cfg,
            skip_runtime_probe=bool(getattr(args, "skip_runtime_probe", False)),
        )
        logger.info(
            f"Training complete — {result.total_steps} steps, "
            f"final loss {result.final_loss:.4f}"
        )
    except Exception:
        try:
            write_error_state(cfg.training.output_dir, sys.exc_info()[1])
        except Exception:
            pass
        raise


def _resolve_output_dir(
    args: argparse.Namespace, overrides: dict[str, object]
) -> str | None:
    """Best-effort resolution of the run's ``output_dir`` in the launcher.

    Prefers the CLI override (cheap); falls back to loading the config.
    Used only so the launcher knows where to drop the
    ``.shutdown_requested`` flag when it receives a walltime signal.
    """
    od = overrides.get("training.output_dir")
    if isinstance(od, str) and od:
        return od
    try:
        from gpt_simple.config import Config

        cfg = Config.from_file(args.config)
        _apply_overrides(cfg, overrides)
        return cfg.training.output_dir
    except Exception:
        return None


def _install_launcher_signal_forwarding(output_dir: str) -> None:
    """Make SLURM's walltime ``SIGUSR1`` trigger a graceful, ranked stop.

    SLURM delivers ``--signal=USR1@N`` to the srun task — i.e. *this*
    launcher process — not to the worker ranks where the
    ``ShutdownCoordinator`` signal handler lives, and ``torchrun`` does not
    forward SIGUSR1.  Left unhandled, the default action (terminate) kills
    the launcher and tears down the workers mid-step, so no shutdown
    checkpoint is written and the resume-chain dies.

    Instead of killing anything, we catch the signal here and drop the
    ``.shutdown_requested`` flag that every worker already polls once per
    step (see ``ShutdownCoordinator.should_shutdown``).  Each rank then
    saves a shutdown checkpoint, clears the flag, and exits 0, so the
    launcher returns 0 and the chain can resubmit cleanly.
    """
    import signal
    from pathlib import Path

    flag = Path(output_dir) / ".shutdown_requested"

    def _handler(signum, frame):  # noqa: ARG001
        try:
            flag.parent.mkdir(parents=True, exist_ok=True)
            flag.write_text(f"launcher:signal:{signum}\n")
        except OSError:
            pass

    # SIGUSR1 is the SLURM walltime convention; SIGUSR2 is handled too for
    # sites that configure it instead.  SIGTERM/SIGINT are left to the
    # elastic agent (scancel / Ctrl-C path).
    for sig in (signal.SIGUSR1, signal.SIGUSR2):
        try:
            signal.signal(sig, _handler)
        except (OSError, ValueError):
            pass


def _ensure_hostname_resolves() -> None:
    """Ensure the local hostname resolves to an IP address.

    PyTorch's elastic agent uses ``socket.getfqdn()`` for its internal
    TCPStore, ignoring ``--master_addr``.  On macOS the hostname often
    has no DNS entry, causing an infinite retry loop.  Patching
    ``getfqdn`` to fall back to ``localhost`` fixes this.
    """
    import socket

    _orig_getfqdn = socket.getfqdn

    def _safe_getfqdn(name: str = "") -> str:
        fqdn = _orig_getfqdn(name)
        try:
            socket.getaddrinfo(fqdn, None)
            return fqdn
        except socket.gaierror:
            return "localhost"

    socket.getfqdn = _safe_getfqdn  # type: ignore[assignment]

    _orig_gethostname = socket.gethostname

    def _safe_gethostname() -> str:
        hostname = _orig_gethostname()
        try:
            socket.getaddrinfo(hostname, None)
            return hostname
        except socket.gaierror:
            return "localhost"

    socket.gethostname = _safe_gethostname  # type: ignore[assignment]


def _launch_distributed(args: argparse.Namespace) -> None:
    """Multi-GPU via ``torch.distributed.run``."""
    import os
    try:
        from torch.distributed.run import run, get_args_parser
    except ImportError:
        from gpt_simple.errors import GptSimpleError
        raise GptSimpleError(
            "torch.distributed.run is required for multi-GPU training. "
            "Install PyTorch >= 1.9."
        )

    _ensure_hostname_resolves()

    os.environ.setdefault("MASTER_ADDR", args.master_addr)
    os.environ.setdefault("MASTER_PORT", str(args.master_port))

    import torch
    if not torch.cuda.is_available():
        # Without CUDA, Accelerator needs ACCELERATE_USE_CPU=1 to even
        # recognise the distributed env vars torchrun sets (RANK, WORLD_SIZE,
        # LOCAL_RANK).  Otherwise every rank thinks it's a solo process and
        # we get racy file writes when checkpoints save in parallel.
        os.environ.setdefault("ACCELERATE_USE_CPU", "1")
        os.environ.setdefault("ACCELERATE_TORCH_DEVICE", "cpu")
        os.environ.setdefault("ACCELERATE_MIXED_PRECISION", "no")
        # On macOS, dist.barrier() under gloo schedules its sentinel tensor
        # on the default device, which is MPS when available; c10d::barrier
        # isn't implemented for MPS.  Falling back to CPU is correct here
        # (we're already CPU-only) and silences the NotImplementedError.
        if torch.backends.mps.is_available():
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    # --force needs to wipe state once, before workers spawn, to avoid races.
    # We do it here on the launcher process (which is exactly one per node);
    # on multi-node runs the node_rank=0 launcher is the canonical wiper.
    if getattr(args, "force", False) and int(getattr(args, "node_rank", 0)) == 0:
        from gpt_simple.config import Config
        cfg = Config.from_file(args.config)
        _apply_overrides(cfg, _collect_overrides(args))
        logger.warning(
            f"--force: wiping prior checkpoints and run state in "
            f"{cfg.training.output_dir} (resume forced to 'scratch')."
        )
        _clobber_output_dir(cfg.training.output_dir)
        # Workers inherit the override via --training.resume below.

    torchrun_parser = get_args_parser()
    torchrun_argv = [
        f"--nproc_per_node={args.nproc_per_node}",
        f"--nnodes={args.nnodes}",
        f"--node_rank={args.node_rank}",
        f"--master_addr={args.master_addr}",
        f"--master_port={args.master_port}",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint={args.master_addr}:{args.master_port}",
    ]

    overrides = _collect_overrides(args)
    if getattr(args, "force", False):
        # Workers don't see --force; tell them via the config-override channel.
        overrides["training.resume"] = "scratch"
    script_args = ["--config", args.config]
    for key, val in overrides.items():
        script_args.extend([f"--{key}", str(val)])
    if getattr(args, "allow_bucket_exhaustion", False):
        script_args.append("--allow-bucket-exhaustion")
    if getattr(args, "allow_budget_mismatch", False):
        script_args.append("--allow-budget-mismatch")
    if getattr(args, "skip_runtime_probe", False):
        script_args.append("--skip-runtime-probe")

    torchrun_argv.extend([
        "-m", "gpt_simple",
        *script_args,
    ])

    torchrun_args = torchrun_parser.parse_args(torchrun_argv)
    torchrun_args.module = True

    # Forward SLURM's walltime SIGUSR1 to a graceful shutdown.  Installed on
    # the launcher (the srun task) because that is where the signal lands.
    out_dir = _resolve_output_dir(args, overrides)
    if out_dir:
        _install_launcher_signal_forwarding(out_dir)

    logger.info(
        f"Launching distributed training: "
        f"{args.nproc_per_node} GPU(s) x {args.nnodes} node(s)"
    )
    run(torchrun_args)
