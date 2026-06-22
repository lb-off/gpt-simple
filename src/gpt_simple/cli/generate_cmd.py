"""
``gpt-simple generate`` subcommand.

Single prompt::

    gpt-simple generate --checkpoint ./outputs/checkpoints/checkpoint-12000 \\
        --prompt "Once upon a time"

Batch from JSONL::

    gpt-simple generate --output-dir ./outputs \\
        --prompts-file prompts.jsonl --output completions.jsonl

Input JSONL format: one object per line with at minimum a ``prompt`` field.
Other fields are echoed back unchanged.

Output JSONL: input fields + ``completion`` + a ``generation`` block with
the sampling parameters used.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterator

logger = logging.getLogger("gpt_simple")


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{lineno}: invalid JSON ({exc}); "
                    "each line must be a JSON object"
                ) from exc
            if not isinstance(obj, dict):
                raise ValueError(
                    f"{path}:{lineno}: expected a JSON object, got {type(obj).__name__}"
                )
            if "prompt" not in obj:
                raise ValueError(
                    f"{path}:{lineno}: object is missing required key 'prompt'"
                )
            yield obj


class GenerateCommand:
    @staticmethod
    def register(subparsers: argparse._SubParsersAction) -> None:
        p = subparsers.add_parser(
            "generate",
            help="Run prompts through a trained model",
            description=(
                "Load a checkpoint and generate completions for one prompt "
                "(--prompt) or a batch (--prompts-file)."
            ),
        )

        src = p.add_mutually_exclusive_group(required=True)
        src.add_argument(
            "--checkpoint", type=Path,
            help="Path to a specific checkpoint directory (e.g. .../checkpoint-12000).",
        )
        src.add_argument(
            "--output-dir", type=Path,
            help=(
                "Path to a run's output_dir. The latest checkpoint inside "
                "checkpoints/ is used (same rule as `train --training.resume auto`)."
            ),
        )

        inp = p.add_mutually_exclusive_group(required=True)
        inp.add_argument("--prompt", type=str, help="Single prompt string.")
        inp.add_argument(
            "--prompts-file", type=Path,
            help="JSONL file with one {\"prompt\": ...} object per line.",
        )

        p.add_argument(
            "--output", type=Path, default=None,
            help=(
                "Write completions as JSONL to this file. Default: emit the "
                "same JSONL records to stdout."
            ),
        )
        p.add_argument(
            "--tokenizer", type=Path, default=None,
            help=(
                "Override the tokenizer location. By default the tokenizer is "
                "discovered next to the checkpoint (run-root tokenizer/ dir)."
            ),
        )

        # Placement
        p.add_argument(
            "--device", type=str, default=None,
            help="cuda | cpu | cuda:N. Default: cuda if available, else cpu.",
        )
        p.add_argument(
            "--dtype", type=str, default="bf16",
            choices=["fp32", "fp16", "bf16"],
            help="Cast weights after load. Default: bf16.",
        )

        # Sampling
        p.add_argument("--max-new-tokens", type=int, default=100)
        p.add_argument("--temperature", type=float, default=0.8)
        p.add_argument("--top-k", type=int, default=50)
        p.add_argument("--top-p", type=float, default=0.95)
        p.add_argument(
            "--greedy", action="store_true",
            help="Argmax decoding (overrides --temperature/--top-k/--top-p).",
        )
        p.add_argument("--repetition-penalty", type=float, default=1.0)
        p.add_argument("--seed", type=int, default=None)
        p.add_argument(
            "--return-full-text", action="store_true",
            help="Include the prompt in each completion.",
        )

        p.set_defaults(func=GenerateCommand.run)

    @staticmethod
    def run(args: argparse.Namespace) -> None:
        # Import torch + the generate module lazily so `gpt-simple --help`
        # stays snappy and doesn't pay the torch import cost.
        from gpt_simple.generate import generate, load_for_inference

        path = args.checkpoint if args.checkpoint is not None else args.output_dir

        sampling: dict = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "do_sample": not args.greedy,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
            "return_full_text": args.return_full_text,
        }
        if args.greedy:
            sampling["temperature"] = 1.0
            sampling["top_k"] = None
            sampling["top_p"] = None

        model, tokenizer, _cfg = load_for_inference(
            path,
            device=args.device,
            dtype=args.dtype,
            tokenizer_path=args.tokenizer,
        )

        meta = _gen_meta(sampling)
        if args.prompt is not None:
            completions = generate(model, tokenizer, args.prompt, **sampling)
            out_records = [{
                "prompt": args.prompt,
                "completion": completions[0],
                "generation": meta,
            }]
        else:
            records = list(_iter_jsonl(args.prompts_file))
            prompts = [r["prompt"] for r in records]
            completions = generate(model, tokenizer, prompts, **sampling)
            out_records = [
                {**r, "completion": c, "generation": meta}
                for r, c in zip(records, completions)
            ]

        if args.output is not None:
            _write_jsonl(args.output, out_records)
            logger.info("Wrote %d completions to %s", len(out_records), args.output)
        else:
            for rec in out_records:
                sys.stdout.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _gen_meta(sampling: dict) -> dict:
    """Strip non-serialisable / housekeeping fields from sampling args."""
    return {k: v for k, v in sampling.items() if k != "return_full_text"}


def _write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
