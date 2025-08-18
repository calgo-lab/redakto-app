from __future__ import annotations
from pydantic import BaseModel
from typing import List, Optional
import json
from pathlib import Path


class SupportedModel(BaseModel):
    model_name: str
    model_id: str
    model_type: str
    model_type_description: str
    model_directory_name: List[str]
    model_version: str

class EntitySetModel(BaseModel):
    entity_set_id: str
    corpus_name: str
    corpus_doctype: str
    corpus_description: str
    corpus_version: str
    corpus_language: str
    links: List[str]
    supported_models_root_dir: List[str]
    supported_models: List[SupportedModel]

class AppInfoData(BaseModel):
    app_name: str
    entity_set_models: List[EntitySetModel]

class AppInfo:
    """Utility for loading and accessing app information and configuration."""

    DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config/app_info.json"

    def __init__(self, config: AppInfoData):
        self._config = config

    @classmethod
    def load(cls, config_path: Optional[str | Path] = None) -> AppInfo:
        """
        Load configuration from a file.
        If no path is given, loads from DEFAULT_CONFIG_PATH.
        """
        path = Path(config_path) if config_path else cls.DEFAULT_CONFIG_PATH
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return cls(AppInfoData.model_validate(data))

    @property
    def app_name(self) -> str:
        return self._config.app_name

    @property
    def entity_sets(self) -> List[EntitySetModel]:
        return self._config.entity_set_models

    def get_entity_set(self, entity_set_id: str) -> EntitySetModel | None:
        """Find an entity set by its ID."""
        return next((es for es in self._config.entity_set_models if es.entity_set_id == entity_set_id), None)

    def to_dict(self) -> dict:
        """Convert back to dictionary."""
        return self._config.model_dump()

    def to_json(self) -> str:
        """Convert back to JSON string."""
        return self._config.model_dump_json(indent=4)