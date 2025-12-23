"""
manifest provides an 'expert' level control interface for
the training loop.
"""
from __future__ import annotations

from pathlib import Path
import json
import yaml
from pydantic import BaseModel
from caramba.config.model import ModelConfig
from caramba.config.defaults import Defaults
from caramba.config.group import Group


class Manifest(BaseModel):
    """
    Manifest turns the manifest.json file into a control structure.
    """
    version: int
    name: str | None = None
    notes: str
    defaults: Defaults
    model: ModelConfig
    groups: list[Group]

    @classmethod
    def from_path(cls, path: Path) -> Manifest:
        """
        load the manifest from a JSON or YAML file.
        """
        suffix = path.suffix.lower()

        match suffix:
            case ".json":
                return cls.model_validate(
                    json.loads(path.read_text(encoding="utf-8")),
                )
            case ".yml" | ".yaml":
                return cls.model_validate(yaml.safe_load(
                    path.read_text(encoding="utf-8"),
                ))
            case _:
                raise ValueError(
                    f"Unsupported manifest format '{suffix}'. Expected .json, .yml, or .yaml."
                )
