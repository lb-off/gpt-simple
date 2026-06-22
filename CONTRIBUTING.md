# Contributing to GPT-Simple

Thanks for your interest in contributing! This project aims to be a clean,
readable reference implementation of an LLM pretraining stack, so
contributions are judged as much on clarity as on correctness.

## Development setup

```bash
git clone <your-fork-url>
cd GPT-Simple
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.10+ is required.

## Before you open a pull request

Run the linter and the test suite locally:

```bash
ruff check src/ tests/
pytest -m "not e2e"     # fast unit tests
pytest                  # full suite, including slow end-to-end tests
```

End-to-end tests (marked `e2e`) spawn the `gpt-simple` CLI as a subprocess
and run tiny real training/resume jobs. They are slower; the fast lane
(`-m "not e2e"`) is enough for most changes.

CI runs `ruff check src/ tests/` and `pytest -m "not e2e"` on every push
and pull request — keep both green.

## Coding guidelines

- **Match the surrounding style.** Line length is 120 (configured in
  `pyproject.toml`); `ruff` enforces the rest.
- **Comment the "why", not the "what".** Explain non-obvious intent,
  trade-offs, or constraints. Don't narrate what the code plainly does,
  and don't leave behind comments that only justify a past bug fix.
- **Keep public behavior covered by tests.** Add or update tests for any
  change in behavior.
- **Update the docs in the same PR.** If you change a config field, the
  data format, or a CLI command, update the relevant page in `docs/`.

## Two project conventions worth knowing

**Config is the single source of truth.** Default values and the exact
field list live in `src/gpt_simple/config.py`. Documentation describes
*meaning and intent* and deliberately avoids duplicating default values,
so the docs can't drift out of sync. Please preserve that split.

**The library stays vendor-neutral.** The core package must not reference
any specific cluster, scheduler, or site (no SLURM/Kubernetes/Jean Zay
specifics in `src/`). Orchestration is handled through generic mechanisms
(`resume: auto`, walltime budgets, POSIX signals). Site-specific or
scheduler-specific material belongs in `examples/`, and example files
should use placeholders (`/path/to/...`, `CHANGEME`) rather than real
accounts or paths.

## Commit and PR style

- Write focused commits with clear messages explaining the *why*.
- Keep PRs scoped to one logical change where possible.
- Describe what changed and how you tested it in the PR description.

## Reporting bugs

Open an issue with a minimal config and the command you ran, the expected
vs. actual behavior, and your environment (OS, Python, PyTorch, GPU). For
training/resume issues, the contents of `output_dir/.run_state.json` and
the relevant log lines are very helpful.

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
