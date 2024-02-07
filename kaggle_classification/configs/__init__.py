"""Config files."""

from argparse import ArgumentTypeError
from .classification_config import ClassificationConfig
from .petals_config import PetalsConfig

CONFIGS_MAPPING = {"petals_config": PetalsConfig()}


def config_argparse(config_name: str) -> ClassificationConfig:
    try:
        return CONFIGS_MAPPING[config_name]
    except KeyError:
        raise ArgumentTypeError(f"invalid choice: {config_name} found (choose from {', '.join(CONFIGS_MAPPING.keys())})")


__all__ = ["ClassificationConfig", "PetalsConfig"]
