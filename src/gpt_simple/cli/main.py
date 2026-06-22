"""
``gpt-simple`` CLI entry point.

Dispatches to subcommands: train, tokenize, init, status, stop, validate,
generate, batch-generate.
"""

from __future__ import annotations

import argparse
import logging
import sys

from gpt_simple._logging import setup_logging
from gpt_simple.errors import GptSimpleError

logger = logging.getLogger("gpt_simple")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="gpt-simple",
        description="GPT-Simple — lightweight LLM pretraining toolkit",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Only show warnings and errors"
    )
    parser.set_defaults(func=lambda _args: parser.print_help())

    sub = parser.add_subparsers(title="commands", dest="command")

    from gpt_simple.cli.train_cmd import TrainCommand
    from gpt_simple.cli.tokenize_cmd import TokenizeCommand
    from gpt_simple.cli.init_cmd import InitCommand
    from gpt_simple.cli.status_cmd import StatusCommand
    from gpt_simple.cli.stop_cmd import StopCommand
    from gpt_simple.cli.validate_cmd import ValidateCommand
    from gpt_simple.cli.generate_cmd import GenerateCommand
    from gpt_simple.cli.batch_generate_cmd import BatchGenerateCommand

    TrainCommand.register(sub)
    TokenizeCommand.register(sub)
    InitCommand.register(sub)
    StatusCommand.register(sub)
    StopCommand.register(sub)
    ValidateCommand.register(sub)
    GenerateCommand.register(sub)
    BatchGenerateCommand.register(sub)

    args = parser.parse_args(argv)

    setup_logging(
        verbose=args.verbose,
        quiet=args.quiet,
    )

    try:
        args.func(args)
    except GptSimpleError as exc:
        logger.error(str(exc))
        sys.exit(exc.exit_code)
    except KeyboardInterrupt:
        logger.warning("Interrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
