"""
Typed exceptions for gpt_simple.

The CLI layer catches these and maps them to user-facing error messages
and exit codes.  Library code should never call ``sys.exit()`` directly.
"""


class GptSimpleError(Exception):
    """Base exception for all gpt_simple errors."""
    exit_code = 1


class ConfigError(GptSimpleError):
    """Invalid or missing configuration."""
    exit_code = 2


class DataError(GptSimpleError):
    """Data path missing, wrong format, no shards found, etc."""
    exit_code = 3


class CheckpointError(GptSimpleError):
    """Checkpoint corrupt, incompatible, or not found."""
    exit_code = 4
