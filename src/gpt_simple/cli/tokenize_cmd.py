"""
``gpt-simple tokenize`` subcommand.

Thin wrapper around ``gpt_simple.pretokenize.main()``.
"""

from __future__ import annotations

import argparse
import sys


class TokenizeCommand:
    @staticmethod
    def register(subparsers: argparse._SubParsersAction) -> None:
        p = subparsers.add_parser(
            "tokenize",
            help="Pre-tokenize a dataset into binary shards",
            description="Convert raw JSONL/text data into pre-tokenized .bin/.idx shards.",
        )
        p.add_argument(
            "--input_dir", type=str, required=True,
            help="Directory containing .jsonl or .txt files",
        )
        p.add_argument(
            "--output_dir", type=str, required=True,
            help="Directory to write .bin/.idx shards",
        )
        p.add_argument(
            "--tokenizer_path", type=str, default="gpt2",
            help="Tokenizer name or path (default: gpt2)",
        )
        p.add_argument(
            "--max_length", type=int, default=3072,
            help="Maximum sequence length in tokens",
        )
        p.add_argument(
            "--overlap_size", type=int, default=256,
            help="Overlap between consecutive chunks",
        )
        p.add_argument(
            "--probabilistic_overlap", action="store_true",
            help="Randomly skip overlaps",
        )
        p.add_argument(
            "--overlap_probability", type=float, default=0.7,
            help="Probability of using overlap when --probabilistic_overlap is set",
        )
        p.add_argument(
            "--min_text_length", type=int, default=200,
            help="Skip documents shorter than this (characters)",
        )
        p.add_argument(
            "--num_workers", type=int, default=1,
            help="Number of parallel workers",
        )
        p.set_defaults(func=TokenizeCommand.run)

    @staticmethod
    def run(args: argparse.Namespace) -> None:
        # Forward to pretokenize.main() by rewriting sys.argv
        argv_backup = sys.argv
        try:
            sys.argv = ["gpt-simple-tokenize"]
            for key in (
                "input_dir", "output_dir", "tokenizer_path", "max_length",
                "overlap_size", "overlap_probability", "min_text_length",
                "num_workers",
            ):
                val = getattr(args, key)
                sys.argv.extend([f"--{key}", str(val)])
            if args.probabilistic_overlap:
                sys.argv.append("--probabilistic_overlap")

            from gpt_simple.pretokenize import main
            main()
        finally:
            sys.argv = argv_backup
