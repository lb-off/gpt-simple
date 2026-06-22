"""
Regression tests for distributed-tensor placement invariants.

The bug this file guards against
--------------------------------

NCCL all_reduce / all_gather / broadcast only accept CUDA tensors.  Gloo
accepts CPU tensors too.  A tensor created with the default constructor
(``torch.tensor(0)``) lands on CPU, which works under our gloo E2E tests
but raises ``RuntimeError: No backend type associated with device type
cpu`` the moment the same code runs under NCCL (multi-GPU, real production
hardware).

That trap bit us once on an 8x V100 box even though all 247 prior tests
were green, because none of those tests exercised the NCCL backend.  This
file enforces a static invariant on the trainer source so that the bug
class cannot return silently:

    Any tensor passed to a torch.distributed collective (all_reduce,
    all_gather, broadcast) must be constructed with an explicit
    ``device=`` argument.

The check parses ``train.py`` and ``_shutdown.py`` as plain text — it's a
lint, not a runtime test — so it works on any host (no CUDA, no
distributed, no GPUs).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import pytest

SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "gpt_simple"

# Files that perform collective ops on hand-built tensors.  If a new file
# joins them, add it here AND make sure each collective in that file has
# an explicit device on its tensor.
_FILES_TO_CHECK = [
    SRC_ROOT / "train.py",
    SRC_ROOT / "_shutdown.py",
]

# Match e.g. ``dist.all_reduce(loss_t, op=...)`` or
# ``_dist_exhaust.all_reduce(local_exhausted)`` — anything ending in
# ``.all_reduce(<NAME>``, where NAME is the python identifier of the
# tensor being reduced.
_COLLECTIVE_CALL_RE = re.compile(
    r"""
    \b(?:[a-zA-Z_]\w*\.)?      # optional dist-module prefix (dist., _dist.)
    (all_reduce|all_gather|broadcast)
    \s*\(\s*
    \[?                        # all_gather can take a list — skip leading [
    ([a-zA-Z_]\w*)             # the tensor's variable name (group 2)
    """,
    re.VERBOSE,
)

# Match ``<NAME> = torch.tensor(...)`` — possibly multi-line.
_TENSOR_ASSIGN_RE = re.compile(
    r"""
    ^\s*
    ([a-zA-Z_]\w*)             # LHS variable name (group 1)
    \s*=\s*
    torch\.tensor\s*\(
    """,
    re.VERBOSE,
)


def _collect_offences(path: Path) -> List[Tuple[int, str, str]]:
    """Return ``(line_no, tensor_name, source_snippet)`` for every
    tensor-construction-without-device feeding a distributed collective
    inside *path*.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")

    # Step 1: index EVERY assignment of the form ``<name> = torch.tensor(...)``
    # so that ``if/else`` branches that both bind the same tensor name are
    # both checked.  We grab up to 8 continuation lines so multi-line
    # ``torch.tensor(..., device=...)`` is seen as a single blob.
    assignments: dict[str, List[Tuple[int, str]]] = {}
    i = 0
    while i < len(lines):
        m = _TENSOR_ASSIGN_RE.match(lines[i])
        if m:
            name = m.group(1)
            blob = lines[i]
            depth = blob.count("(") - blob.count(")")
            j = i
            while depth > 0 and j + 1 < len(lines) and j - i < 8:
                j += 1
                blob += "\n" + lines[j]
                depth += lines[j].count("(") - lines[j].count(")")
            assignments.setdefault(name, []).append((i, blob))
        i += 1

    # Step 2: scan for distributed collective calls and check every
    # assignment of the callee's tensor argument that precedes the call.
    # If ANY preceding assignment was built without a device (and no
    # subsequent .to(device) moved it), flag it — because that branch may
    # produce a CPU tensor at runtime.
    offences: List[Tuple[int, str, str]] = []
    for i, line in enumerate(lines):
        m = _COLLECTIVE_CALL_RE.search(line)
        if not m:
            continue
        tensor_name = m.group(2)
        for assign_line, blob in assignments.get(tensor_name, []):
            if assign_line > i:
                continue
            if "device=" in blob or ".to(" in blob:
                continue
            # Look for a subsequent ``t.to(...)`` / ``t = ....to(...)`` /
            # ``t.to_(...)`` between construction and the collective call.
            moved = False
            for k in range(assign_line + 1, i):
                mid = lines[k]
                if re.search(rf"\b{re.escape(tensor_name)}\.to[_]?\(", mid):
                    moved = True
                    break
                if re.search(
                    rf"^\s*{re.escape(tensor_name)}\s*=\s*.*\.to\(", mid
                ):
                    moved = True
                    break
            if moved:
                continue
            offences.append((assign_line + 1, tensor_name, blob.strip()))
    return offences


@pytest.mark.parametrize("path", _FILES_TO_CHECK, ids=lambda p: p.name)
def test_no_cpu_tensors_into_distributed_collectives(path: Path) -> None:
    """Every tensor handed to all_reduce / all_gather / broadcast must
    have been built with an explicit ``device=`` or ``.to(...)``.

    Otherwise it lands on CPU and crashes under NCCL with
    ``RuntimeError: No backend type associated with device type cpu``.
    """
    offences = _collect_offences(path)
    if offences:
        report = "\n".join(
            f"  line {ln}: tensor '{name}' built without device=\n"
            f"      {blob}"
            for ln, name, blob in offences
        )
        pytest.fail(
            f"{path.name}: tensor(s) constructed without device= are "
            f"passed to a distributed collective.  This will crash under "
            f"NCCL.  Add ``device=accelerator.device`` (or equivalent) "
            f"to the torch.tensor(...) call:\n{report}"
        )


def test_collective_regex_actually_matches_known_call() -> None:
    """Smoke test: make sure the regex catches an obvious case so a
    typo in the regex can't silently defang every check above."""
    sample = "                _dist.all_reduce(step_tokens_t, op=ReduceOp.SUM)"
    m = _COLLECTIVE_CALL_RE.search(sample)
    assert m is not None, "regex failed to match a canonical all_reduce call"
    assert m.group(2) == "step_tokens_t"


def test_tensor_assign_regex_actually_matches_known_construction() -> None:
    sample = "    foo = torch.tensor(0, dtype=torch.int32, device=dev)"
    m = _TENSOR_ASSIGN_RE.match(sample)
    assert m is not None, "regex failed to match a canonical tensor assign"
    assert m.group(1) == "foo"


# ---------------------------------------------------------------------------
# accelerator.prepare must NOT receive dataloaders / schedulers
# ---------------------------------------------------------------------------
#
# Background: Accelerate's ``prepare()`` automatically wraps IterableDatasets
# in ``IterableDatasetShard`` and schedulers in ``AcceleratedScheduler``.
# Both wrappers re-implement distribution semantics that we already own
# inside ``StreamingDataModule`` and the train loop:
#
#   - ``IterableDatasetShard`` buffers ``batch_size * num_processes`` items
#     from the dataset and emits only ``batch_size`` per rank, discarding
#     ``(num_processes - 1) / num_processes`` of the data.  On 8 GPUs this
#     made the 8x V100 smoke test exhaust the dataset at step 2 instead of
#     step 16.
#   - ``AcceleratedScheduler`` calls ``scheduler.step()`` ``num_processes``
#     times per ``.step()`` invocation when ``split_batches=False``,
#     racing through warmup and decay ``num_processes``x faster.
#
# Neither matches what we want.  We prepare only the model and optimizer.
# This lint locks that in: any ``accelerator.prepare(...)`` call must not
# pass any argument whose name suggests dataloader / scheduler.

_PREPARE_CALL_RE = re.compile(
    r"""
    \baccelerator\.prepare\s*\(    # the call site
    (?P<args>[^)]*)                # crude one-line capture of arguments
    \)
    """,
    re.VERBOSE,
)

# Identifier-name heuristics for the kinds of objects we *forbid* from
# being passed through ``accelerator.prepare``.  Anything matching one of
# these substrings (case-insensitive, on the argument identifier) is a
# violation.
_FORBIDDEN_PREPARE_ARG_PATTERNS = (
    "_dl",            # train_dl, eval_dl
    "dataloader",     # any *dataloader* name
    "data_loader",
    "scheduler",      # lr_scheduler, scheduler
    "_sched",         # lr_sched, sched
)


def _strip_strings_and_comments(text: str) -> str:
    """Blank out triple-quoted strings and line comments so a regex scan
    won't false-positive on documentation that *mentions* a pattern we're
    forbidding.  Preserves line numbers (replaces with spaces, not nothing).
    """
    out = []
    i = 0
    in_triple = None  # one of None, '"""', "'''"
    while i < len(text):
        if in_triple is None:
            if text[i:i + 3] in ('"""', "'''"):
                in_triple = text[i:i + 3]
                out.append("   ")
                i += 3
                continue
            if text[i] == "#":
                # rest-of-line comment — skip to newline
                while i < len(text) and text[i] != "\n":
                    out.append(" ")
                    i += 1
                continue
            out.append(text[i])
            i += 1
        else:
            if text[i:i + 3] == in_triple:
                out.append("   ")
                in_triple = None
                i += 3
                continue
            # preserve newlines so line numbers stay aligned
            out.append("\n" if text[i] == "\n" else " ")
            i += 1
    return "".join(out)


def test_accelerator_prepare_only_takes_model_and_optimizer() -> None:
    """``train.py`` must not route dataloaders or schedulers through
    ``accelerator.prepare``.

    Doing so causes Accelerate to install ``IterableDatasetShard`` (which
    re-shards the already-rank-sharded stream and silently discards 7/8 of
    the data on 8 GPUs) and ``AcceleratedScheduler`` (which advances the
    schedule ``num_processes`` times per step).  Both are correctness
    bugs in our setup, not perf footguns.
    """
    text = (SRC_ROOT / "train.py").read_text(encoding="utf-8")
    # Strip docstrings + comments so we only inspect executable code.
    text = _strip_strings_and_comments(text)
    offences: List[Tuple[int, str, str]] = []
    for m in _PREPARE_CALL_RE.finditer(text):
        args_blob = m.group("args")
        # Compute the line number of the call.
        line_no = text[: m.start()].count("\n") + 1
        # Split on commas at top level (none of our prepare calls have
        # nested call expressions, so a naive split is fine).
        for raw_arg in args_blob.split(","):
            ident = raw_arg.strip().lower()
            # Strip out trailing ``=value`` kwarg forms.
            if "=" in ident:
                ident = ident.split("=", 1)[0].strip()
            if not ident:
                continue
            for forbidden in _FORBIDDEN_PREPARE_ARG_PATTERNS:
                if forbidden in ident:
                    offences.append((line_no, ident, args_blob.strip()))
                    break
    if offences:
        report = "\n".join(
            f"  line {ln}: forbidden arg '{ident}' in accelerator.prepare({blob})"
            for ln, ident, blob in offences
        )
        pytest.fail(
            "train.py: accelerator.prepare(...) must not receive "
            "dataloaders or schedulers — they get silently re-wrapped "
            "by Accelerate's IterableDatasetShard / AcceleratedScheduler, "
            "which double-shards data and races the LR schedule.\n"
            f"{report}"
        )


def test_prepare_regex_actually_matches_known_call() -> None:
    sample = "    llm, opt = accelerator.prepare(llm, opt)"
    m = _PREPARE_CALL_RE.search(sample)
    assert m is not None, "regex failed to match a canonical prepare call"
    assert "llm" in m.group("args")
    assert "opt" in m.group("args")


def test_prepare_lint_actually_catches_the_buggy_pattern(tmp_path) -> None:
    """Sanity check: feed the regression checker source code that
    *does* route a dataloader through ``accelerator.prepare`` and confirm
    that we flag it.  Without this, a typo in the regex or the forbidden
    list could silently defang the real check above.
    """
    buggy = """
    def train():
        llm, opt, train_dl, lr_scheduler = accelerator.prepare(
            llm, opt, train_dl, lr_scheduler
        )
    """
    stripped = _strip_strings_and_comments(buggy)
    matches = list(_PREPARE_CALL_RE.finditer(stripped))
    assert matches, "regex didn't find the buggy prepare call"
    args_blob = matches[0].group("args")
    flagged = []
    for raw_arg in args_blob.split(","):
        ident = raw_arg.strip().lower()
        if "=" in ident:
            ident = ident.split("=", 1)[0].strip()
        for forbidden in _FORBIDDEN_PREPARE_ARG_PATTERNS:
            if forbidden in ident:
                flagged.append(ident)
                break
    assert "train_dl" in flagged, f"failed to flag train_dl: {flagged}"
    assert "lr_scheduler" in flagged, f"failed to flag lr_scheduler: {flagged}"
