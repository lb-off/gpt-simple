"""
``gpt-simple validate`` subcommand.

Runs all offline pre-flight checks and prints a human-readable summary.
Designed to be called on the login node before submitting a SLURM job —
catches typos, missing buckets, token-budget shortfalls, and curriculum/
training-loop mismatches before any GPU time is consumed.

Exit codes:
    0 — all good (info-only findings)
    2 — warnings present (or any finding with ``--strict``)
    1 — errors / budget issues present
"""

from __future__ import annotations

import argparse
import logging
import sys

from gpt_simple.config import Config
from gpt_simple.validate import (
    format_report,
    run_offline_validation,
)

logger = logging.getLogger("gpt_simple")


class ValidateCommand:
    @staticmethod
    def register(subparsers: argparse._SubParsersAction) -> None:
        p = subparsers.add_parser(
            "validate",
            help="Pre-flight check a config before submitting a training job",
            description=(
                "Run every offline check the trainer would otherwise only "
                "discover after SLURM allocation: config validity, data dir "
                "structure, tokenizer load, curriculum bucket existence + "
                "token budget, training-loop / curriculum mismatch, output "
                "dir writability, resume drift.  Prints a human-readable "
                "summary and exits with a non-zero code on errors."
            ),
        )
        p.add_argument(
            "--config", type=str, required=True,
            help="Path to YAML/JSON config file",
        )
        p.add_argument(
            "--world-size", type=int, default=1,
            help=(
                "Number of GPUs the real training job will use.  Affects "
                "the effective batch size and per-rank shard count checks. "
                "Default 1."
            ),
        )
        p.add_argument(
            "--strict", action="store_true",
            help="Treat warnings as errors (exit 1 instead of 2).",
        )
        p.add_argument(
            "--allow-bucket-exhaustion", action="store_true",
            help=(
                "Override data.allow_bucket_exhaustion to true for this "
                "validation run.  Downgrades per-bucket token-budget "
                "shortfalls from error to warning."
            ),
        )
        p.add_argument(
            "--allow-budget-mismatch", action="store_true",
            help=(
                "Override data.allow_budget_mismatch to true for this "
                "validation run.  Downgrades curriculum-vs-max_steps "
                "mismatch from error to warning."
            ),
        )

        # Dotted overrides applied on top of the loaded config — useful when
        # validating the same config with a different output_dir, etc.
        p.add_argument("--training.max_steps", type=int, default=None)
        p.add_argument("--training.output_dir", type=str, default=None)
        p.add_argument("--training.resume", type=str, default=None)
        p.add_argument("--data.path", type=str, default=None)
        p.add_argument("--data.tokenizer", type=str, default=None)
        p.add_argument("--data.format", type=str, default=None)

        p.set_defaults(func=ValidateCommand.run)

    @staticmethod
    def run(args: argparse.Namespace) -> None:
        try:
            cfg = Config.from_file(args.config)
        except Exception as exc:
            logger.error(f"Could not load config {args.config}: {exc}")
            sys.exit(1)

        overrides = {
            key: val
            for key, val in vars(args).items()
            if "." in key and val is not None
        }
        for key, val in overrides.items():
            section, attr = key.split(".", 1)
            setattr(getattr(cfg, section), attr, val)
        cfg._validate_schedule()

        if args.allow_bucket_exhaustion:
            cfg.data.allow_bucket_exhaustion = True
        if args.allow_budget_mismatch:
            cfg.data.allow_budget_mismatch = True

        report = run_offline_validation(cfg, world_size=args.world_size)

        sys.stdout.write(format_report(report, config_path=args.config) + "\n")

        # Exit code policy:
        #   errors or budget issues -> 1
        #   warnings only           -> 2 (or 1 in --strict mode)
        #   otherwise               -> 0
        if report.has_errors() or report.has_budget_issues():
            sys.exit(1)
        if report.has_warnings():
            sys.exit(1 if args.strict else 2)
        sys.exit(0)
