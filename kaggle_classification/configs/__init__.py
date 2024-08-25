"""Config files."""

from argparse import ArgumentTypeError
from typing import Any, Dict

from .catsdogs_config import CatsdogsConfig
from .classification_config import ClassificationConfig
from .petals_config import PetalsConfig

CONFIGS_MAPPING: Dict[str, ClassificationConfig[Any]] = {
    "catsdogs_config": CatsdogsConfig(),
    "petals_config": PetalsConfig(),
}


def config_argparse(config_name: str) -> ClassificationConfig[Any]:
    """Parse string to config object."""
    try:
        return CONFIGS_MAPPING[config_name]
    except KeyError as exc:
        raise ArgumentTypeError(
            f"invalid choice: {config_name} found (choose from {', '.join(CONFIGS_MAPPING.keys())})"
        ) from exc


__all__ = ["ClassificationConfig", "PetalsConfig"]
