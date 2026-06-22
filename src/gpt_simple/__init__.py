"""
gpt_simple — a clean, hackable GPT pretraining library.
"""

from importlib.metadata import PackageNotFoundError, version

from gpt_simple.config import Config, CurriculumPhase, DataConfig, ModelConfig, OptimizerConfig, TrainingConfig
from gpt_simple.errors import CheckpointError, ConfigError, DataError, GptSimpleError
from gpt_simple.generate import generate, load_for_inference, validate_checkpoint
from gpt_simple.model import SimpleLLM
from gpt_simple.tokenizer import SimpleLLMTokenizer
from gpt_simple.train import TrainingResult, train

# Single source of truth is pyproject.toml; read it from the installed
# metadata so the two can't drift. Falls back when running from a source
# tree that hasn't been installed.
try:
    __version__ = version("gpt-simple")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "train",
    "generate",
    "load_for_inference",
    "validate_checkpoint",
    "Config",
    "CurriculumPhase",
    "TrainingResult",
    "ModelConfig",
    "DataConfig",
    "OptimizerConfig",
    "TrainingConfig",
    "SimpleLLM",
    "SimpleLLMTokenizer",
    "GptSimpleError",
    "ConfigError",
    "DataError",
    "CheckpointError",
]
