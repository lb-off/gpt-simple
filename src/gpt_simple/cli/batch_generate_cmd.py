"""
``gpt-simple batch-generate`` subcommand.

Runs a *self-describing* JSONL through one or more checkpoints.  Unlike
``gpt-simple generate`` — where the model and sampling parameters are fixed
for the whole invocation — every record here may carry its own ``model``
block and ``generation`` overrides::

    {"id": "ex-1",
     "prompt": "Once upon a time",
     "model": {"checkpoint": "/runs/r1/checkpoints/checkpoint-12000", "dtype": "bf16"},
     "generation": {"max_new_tokens": 200, "temperature": 0.8, "top_k": 50}}

Only ``prompt`` is required.  Anything omitted falls back to the CLI
defaults (``--checkpoint``/``--output-dir``, ``--dtype``, ``--temperature``,
…), so a *homogeneous* batch need not repeat the model on every line.  Extra
fields are echoed back unchanged.

Records are grouped by ``(model, dtype, tokenizer)`` and each distinct model
is loaded exactly once — a 2B+ checkpoint costs minutes to load, so we
amortise that across all of its prompts and only ever hold one model in
memory at a time.  Output order matches input order regardless of grouping.

**Validation is a GPU-free pre-flight.**  Before any weights are loaded the
command parses every record, checks the sampling parameters, and confirms
each distinct checkpoint resolves (config parses, weights + tokenizer
present).  ``--dry-run`` stops there and prints the execution plan — run it
on a cluster login node to gate ``sbatch`` before spending a GPU
allocation.  Structural problems are hard errors (exit 2, nothing loaded);
failures that only surface *during* generation are soft (the record gets an
``error`` field instead of ``completion`` and the job continues).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

logger = logging.getLogger("gpt_simple")

# User-facing sampling keys allowed inside a record's ``generation`` block.
_SAMPLING_KEYS = {
    "max_new_tokens", "temperature", "top_k", "top_p",
    "greedy", "repetition_penalty", "seed", "return_full_text",
}
# Keys allowed inside a record's ``model`` block.
_MODEL_KEYS = {"checkpoint", "output_dir", "dtype", "tokenizer"}
# Mirror of gpt_simple.generate._DTYPE_ALIASES — duplicated so validation
# stays torch-free-ish and self-contained; kept in sync deliberately.
_DTYPE_NAMES = {
    "fp32", "float32", "f32",
    "fp16", "float16", "half",
    "bf16", "bfloat16",
}


class ModelSpec(NamedTuple):
    """A record's *literal* model reference, before path resolution."""
    path: str
    dtype: str
    tokenizer: Optional[str]


class CanonicalModel(NamedTuple):
    """Resolved grouping key — one distinct value == one model load.

    Built from the *resolved* checkpoint + tokenizer dirs, so different
    literal references to the same model collapse to a single load.
    """
    checkpoint_dir: str
    dtype: str
    tokenizer_dir: str


class ResolvedRecord(NamedTuple):
    index: int                 # position in the input file (for output ordering)
    lineno: int
    raw: Dict[str, Any]        # original record, echoed through
    prompt: str
    model: ModelSpec
    sampling: Dict[str, Any]   # ready-to-splat kwargs for generate()


class BatchGenerateCommand:
    @staticmethod
    def register(subparsers: argparse._SubParsersAction) -> None:
        p = subparsers.add_parser(
            "batch-generate",
            help="Run a self-describing JSONL through one or more checkpoints",
            description=(
                "Generate completions for a JSONL where each record may carry "
                "its own model and sampling parameters. Models are loaded once "
                "each and reused across their prompts."
            ),
        )

        p.add_argument(
            "--input", type=Path, required=True,
            help="Input JSONL: one record per line, each with at least a 'prompt'.",
        )
        p.add_argument(
            "--output", type=Path, default=None,
            help="Write completions as JSONL here. Default: emit to stdout.",
        )

        # Job-level default model (used by records without their own 'model').
        dflt = p.add_mutually_exclusive_group()
        dflt.add_argument(
            "--checkpoint", type=Path, default=None,
            help="Default checkpoint dir for records that omit a 'model' block.",
        )
        dflt.add_argument(
            "--output-dir", type=Path, default=None,
            help="Default run dir (latest checkpoint auto-picked) for records "
                 "that omit a 'model' block.",
        )
        p.add_argument(
            "--tokenizer", type=Path, default=None,
            help="Default tokenizer override (records may override per-model).",
        )

        # Placement (job-level: one process, one device).
        p.add_argument(
            "--device", type=str, default=None,
            help="cuda | cpu | cuda:N. Default: cuda if available, else cpu.",
        )
        p.add_argument(
            "--dtype", type=str, default="bf16", choices=sorted(_DTYPE_NAMES),
            help="Default weight cast for records that omit model.dtype. Default: bf16.",
        )

        # Default sampling (records override per-field via their 'generation').
        p.add_argument("--max-new-tokens", type=int, default=100)
        p.add_argument("--temperature", type=float, default=0.8)
        p.add_argument("--top-k", type=int, default=50)
        p.add_argument("--top-p", type=float, default=0.95)
        p.add_argument("--greedy", action="store_true",
                       help="Default to argmax decoding (records may override).")
        p.add_argument("--repetition-penalty", type=float, default=1.0)
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--return-full-text", action="store_true",
                       help="Include the prompt in each completion.")

        p.add_argument(
            "--dry-run", action="store_true",
            help="Validate the input and print the execution plan without "
                 "loading any model. GPU-free; safe to run on a login node.",
        )

        p.set_defaults(func=BatchGenerateCommand.run)

    @staticmethod
    def run(args: argparse.Namespace) -> None:
        # Lazy import: keep `gpt-simple --help` snappy and avoid the torch
        # cost on a pure --dry-run that errors out during parsing.
        from gpt_simple.errors import CheckpointError
        from gpt_simple.generate import generate, load_for_inference, validate_checkpoint

        if not args.input.is_file():
            logger.error("input JSONL not found: %s", args.input)
            sys.exit(2)

        default_model = args.checkpoint or args.output_dir
        default_model = str(default_model) if default_model is not None else None
        default_tokenizer = str(args.tokenizer) if args.tokenizer is not None else None
        default_sampling = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "greedy": args.greedy,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
            "return_full_text": args.return_full_text,
        }

        records, errors = _parse_and_resolve(
            args.input, default_model, args.dtype, default_tokenizer, default_sampling,
        )

        if not records and not errors:
            logger.error("%s contains no records", args.input)
            sys.exit(2)

        # GPU-free pre-flight, once per distinct literal model reference:
        # resolve the checkpoint, parse its config, and load its tokenizer.
        # The *resolved* checkpoint dir becomes the grouping key, so two
        # references that point at the same model (e.g. a run dir and the
        # checkpoint dir it auto-picks) are loaded only once.
        literal_specs: Dict[ModelSpec, None] = {}
        for r in records:
            literal_specs.setdefault(r.model, None)
        canon_of: Dict[ModelSpec, CanonicalModel] = {}
        for spec in literal_specs:
            try:
                info = validate_checkpoint(spec.path, tokenizer_path=spec.tokenizer)
            except CheckpointError as exc:
                errors.append(f"model {spec.path!r}: {exc}")
                continue
            canon_of[spec] = CanonicalModel(
                checkpoint_dir=str(info.checkpoint_dir),
                dtype=spec.dtype,
                tokenizer_dir=str(info.tokenizer_dir),
            )

        if errors:
            logger.error("pre-flight failed with %d problem(s):", len(errors))
            for msg in errors:
                logger.error("  %s", msg)
            sys.exit(2)

        # Group records by canonical (resolved) model — this is the set of
        # actual loads that will happen.
        groups: Dict[CanonicalModel, List[ResolvedRecord]] = {}
        for r in records:
            groups.setdefault(canon_of[r.model], []).append(r)

        # Literal references that collapsed onto each canonical model — lets
        # the report surface auto-pick (run dir -> checkpoint) and dedup.
        refs_of: Dict[CanonicalModel, List[str]] = {}
        for spec, canon in canon_of.items():
            refs = refs_of.setdefault(canon, [])
            if spec.path not in refs:
                refs.append(spec.path)

        # Explicit pre-flight report. Runs for both --dry-run and real runs:
        # for a SLURM gate the value is spelling out exactly what passed, so a
        # long queue wait isn't wasted on a typo that could've been caught.
        merged = len(literal_specs) - len(groups)
        logger.info("Pre-flight validation (no GPU, no weights loaded):")
        logger.info(
            "  records  : %d parsed — all valid JSON, all carry a 'prompt' "
            "string, all sampling params in range",
            len(records),
        )
        logger.info(
            "  models   : %d reference(s) -> %d distinct checkpoint(s) to load%s",
            len(literal_specs), len(groups),
            f" ({merged} duplicate reference(s) merged)" if merged else "",
        )
        for key, recs in groups.items():
            logger.info("    %s  [dtype=%s]  — %d prompt(s)",
                        key.checkpoint_dir, key.dtype, len(recs))
            logger.info(
                "        config.json parses, pytorch_model.bin present, "
                "tokenizer at %s", key.tokenizer_dir,
            )
            if refs_of[key] != [key.checkpoint_dir]:
                logger.info("        referenced by: %s",
                            ", ".join(repr(x) for x in refs_of[key]))
        logger.info("  prompts  : %d total", len(records))

        if args.dry_run:
            logger.info("--dry-run OK — input valid, nothing loaded. Safe to submit.")
            return

        # --- Execute: load each model once, generate, soft per-record errors.
        results: List[Optional[dict]] = [None] * len(records)
        for key, recs in groups.items():
            logger.info("Loading %s (dtype=%s) for %d prompt(s) …",
                        key.checkpoint_dir, key.dtype, len(recs))
            model, tokenizer, _cfg = load_for_inference(
                key.checkpoint_dir, device=args.device, dtype=key.dtype,
                tokenizer_path=key.tokenizer_dir,
            )
            for r in recs:
                meta = _gen_meta(r.sampling)
                try:
                    completions = generate(model, tokenizer, r.prompt, **r.sampling)
                    results[r.index] = {**r.raw, "completion": completions[0], "generation": meta}
                except Exception as exc:  # soft: one bad prompt doesn't sink the job
                    logger.warning(
                        "%s:%d: generation failed: %s: %s",
                        args.input, r.lineno, type(exc).__name__, exc,
                    )
                    results[r.index] = {
                        **r.raw,
                        "error": f"{type(exc).__name__}: {exc}",
                        "generation": meta,
                    }
            # Drop references here (not in a helper) so the model is freed
            # before the next group's load — never two models co-resident.
            del model, tokenizer
            _empty_cache()

        if args.output is not None:
            _write_jsonl(args.output, results)
            logger.info("Wrote %d completions to %s", len(results), args.output)
        else:
            for rec in results:
                sys.stdout.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Parsing + validation (GPU-free)
# ---------------------------------------------------------------------------


def _parse_and_resolve(
    path: Path,
    default_model: Optional[str],
    default_dtype: str,
    default_tokenizer: Optional[str],
    default_sampling: Dict[str, Any],
) -> Tuple[List[ResolvedRecord], List[str]]:
    """Read the JSONL and resolve every record against the CLI defaults.

    Returns ``(records, errors)``.  All structural problems are collected
    (not raised) so the caller can report them together and exit once.
    """
    records: List[ResolvedRecord] = []
    errors: List[str] = []

    with open(path) as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            where = f"{path}:{lineno}"
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{where}: invalid JSON ({exc})")
                continue
            if not isinstance(obj, dict):
                errors.append(f"{where}: expected a JSON object, got {type(obj).__name__}")
                continue

            rec_errors: List[str] = []

            prompt = obj.get("prompt")
            if "prompt" not in obj:
                rec_errors.append("missing required key 'prompt'")
            elif not isinstance(prompt, str):
                rec_errors.append(f"'prompt' must be a string, got {type(prompt).__name__}")

            spec = _resolve_model(
                obj.get("model"), default_model, default_dtype, default_tokenizer, rec_errors,
            )
            sampling = _resolve_sampling(obj.get("generation"), default_sampling, rec_errors)

            if rec_errors:
                for msg in rec_errors:
                    errors.append(f"{where}: {msg}")
                continue

            # index = position among *valid* records, so it indexes cleanly
            # into the results array regardless of interleaved errored lines.
            records.append(ResolvedRecord(
                index=len(records), lineno=lineno, raw=obj,
                prompt=prompt, model=spec, sampling=sampling,
            ))

    return records, errors


def _resolve_model(
    block: Any,
    default_model: Optional[str],
    default_dtype: str,
    default_tokenizer: Optional[str],
    errors: List[str],
) -> Optional[ModelSpec]:
    """Build a ModelSpec from a record's ``model`` block + CLI defaults."""
    path: Optional[str] = default_model
    dtype = default_dtype
    tokenizer = default_tokenizer

    if block is not None:
        if not isinstance(block, dict):
            errors.append(f"'model' must be an object, got {type(block).__name__}")
            return None
        unknown = set(block) - _MODEL_KEYS
        if unknown:
            errors.append(f"unknown 'model' key(s): {sorted(unknown)}")
        if "checkpoint" in block and "output_dir" in block:
            errors.append("'model' has both 'checkpoint' and 'output_dir'; pick one")
        rec_path = block.get("checkpoint", block.get("output_dir"))
        if rec_path is not None:
            path = str(rec_path)
        if "dtype" in block:
            dtype = block["dtype"]
        if "tokenizer" in block and block["tokenizer"] is not None:
            tokenizer = str(block["tokenizer"])

    if path is None:
        errors.append(
            "no model: record has no 'model' block and no --checkpoint/"
            "--output-dir default was given"
        )
        return None
    if not isinstance(dtype, str) or dtype not in _DTYPE_NAMES:
        errors.append(f"model.dtype must be one of {sorted(_DTYPE_NAMES)}, got {dtype!r}")
        return None

    return ModelSpec(path=path, dtype=dtype, tokenizer=tokenizer)


def _resolve_sampling(
    block: Any,
    default_sampling: Dict[str, Any],
    errors: List[str],
) -> Dict[str, Any]:
    """Merge a record's ``generation`` block over defaults → generate() kwargs."""
    merged = dict(default_sampling)
    if block is not None:
        if not isinstance(block, dict):
            errors.append(f"'generation' must be an object, got {type(block).__name__}")
            return {}
        unknown = set(block) - _SAMPLING_KEYS
        if unknown:
            errors.append(f"unknown 'generation' key(s): {sorted(unknown)}")
        merged.update({k: v for k, v in block.items() if k in _SAMPLING_KEYS})

    _validate_sampling(merged, errors)
    return _to_generate_kwargs(merged)


def _validate_sampling(s: Dict[str, Any], errors: List[str]) -> None:
    def _is_num(x: Any) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    if not isinstance(s["max_new_tokens"], int) or isinstance(s["max_new_tokens"], bool) \
            or s["max_new_tokens"] < 1:
        errors.append(f"max_new_tokens must be an int >= 1, got {s['max_new_tokens']!r}")
    if not _is_num(s["temperature"]) or s["temperature"] <= 0:
        errors.append(f"temperature must be a number > 0, got {s['temperature']!r}")
    if s["top_k"] is not None and (not isinstance(s["top_k"], int)
                                   or isinstance(s["top_k"], bool) or s["top_k"] < 0):
        errors.append(f"top_k must be a non-negative int or null, got {s['top_k']!r}")
    if s["top_p"] is not None and (not _is_num(s["top_p"]) or not 0 < s["top_p"] <= 1):
        errors.append(f"top_p must be a number in (0, 1] or null, got {s['top_p']!r}")
    if not _is_num(s["repetition_penalty"]) or s["repetition_penalty"] <= 0:
        errors.append(f"repetition_penalty must be a number > 0, got {s['repetition_penalty']!r}")
    if s["seed"] is not None and (not isinstance(s["seed"], int) or isinstance(s["seed"], bool)):
        errors.append(f"seed must be an int or null, got {s['seed']!r}")
    if not isinstance(s["greedy"], bool):
        errors.append(f"greedy must be a bool, got {s['greedy']!r}")
    if not isinstance(s["return_full_text"], bool):
        errors.append(f"return_full_text must be a bool, got {s['return_full_text']!r}")


def _to_generate_kwargs(s: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a user-facing sampling block to generate() kwargs.

    ``greedy`` maps to ``do_sample=False`` and neutralises
    temperature/top_k/top_p (same rule as ``gpt-simple generate``).
    """
    greedy = bool(s["greedy"])
    kw = {
        "max_new_tokens": s["max_new_tokens"],
        "temperature": s["temperature"],
        "top_k": s["top_k"],
        "top_p": s["top_p"],
        "do_sample": not greedy,
        "repetition_penalty": s["repetition_penalty"],
        "seed": s["seed"],
        "return_full_text": s["return_full_text"],
    }
    if greedy:
        kw["temperature"] = 1.0
        kw["top_k"] = None
        kw["top_p"] = None
    return kw


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _gen_meta(sampling: Dict[str, Any]) -> Dict[str, Any]:
    """Strip housekeeping fields from generate() kwargs for the output record."""
    return {k: v for k, v in sampling.items() if k != "return_full_text"}


def _write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _empty_cache() -> None:
    """Return freed GPU memory to the allocator between model loads."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
