"""
Logging setup for gpt_simple.

Configures the ``"gpt_simple"`` logger hierarchy.  When ``rich`` is
installed the CLI uses a coloured handler; otherwise falls back to a
plain ``StreamHandler``.
"""

from __future__ import annotations

import logging
import sys


def setup_logging(
    *,
    verbose: bool = False,
    quiet: bool = False,
    use_rich: bool = True,
) -> None:
    """Configure the ``gpt_simple`` logger.

    Parameters
    ----------
    verbose : bool
        Set level to DEBUG.
    quiet : bool
        Set level to WARNING.
    use_rich : bool
        Use ``rich.logging.RichHandler`` when available.
    """
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = logging.INFO

    logger = logging.getLogger("gpt_simple")
    logger.setLevel(level)

    if logger.handlers:
        return

    handler: logging.Handler
    if use_rich:
        try:
            from rich.logging import RichHandler
            handler = RichHandler(
                level=level,
                show_time=True,
                show_path=False,
                markup=True,
                rich_tracebacks=True,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
        except ImportError:
            handler = _plain_handler(level)
    else:
        handler = _plain_handler(level)

    logger.addHandler(handler)


def _plain_handler(level: int) -> logging.StreamHandler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("[gpt_simple] %(message)s"))
    return handler
