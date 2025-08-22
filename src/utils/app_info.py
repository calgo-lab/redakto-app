from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional

import yaml

class SupportedModel(BaseModel):
    model_name: str
    model_id: str
    model_type: str
    model_type_description: str
    model_loading_strategy: str
    model_directory_name: List[str]
    model_version: str
    model_impl: str
    model_system_requirements: List[str]

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
    """
    Utility for loading and accessing app information and configuration.
    """

    DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config/app_info.yml"

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
            data = yaml.safe_load(f)
        
        # Parse entity sets and their models with full validation
        entity_sets = []
        for es_data in data["entity_set_models"]:
            models = [SupportedModel.model_validate(model) for model in es_data["supported_models"]]
            entity_sets.append(EntitySetModel(
                entity_set_id=es_data["entity_set_id"],
                corpus_name=es_data["corpus_name"],
                corpus_doctype=es_data["corpus_doctype"],
                corpus_description=es_data["corpus_description"],
                corpus_version=es_data["corpus_version"],
                corpus_language=es_data["corpus_language"],
                links=es_data["links"],
                supported_models_root_dir=es_data["supported_models_root_dir"],
                supported_models=models
            ))
        
        app_info_data = AppInfoData(
            app_name=data["app_name"],
            entity_set_models=entity_sets
        )
        return cls(app_info_data)

    @property
    def app_name(self) -> str:
        return self._config.app_name

    @property
    def entity_sets(self) -> List[EntitySetModel]:
        return self._config.entity_set_models

    def get_entity_set(self, entity_set_id: str) -> EntitySetModel | None:
        """
        Find an entity set by its ID.
        """
        return next((es for es in self._config.entity_set_models if es.entity_set_id == entity_set_id), None)