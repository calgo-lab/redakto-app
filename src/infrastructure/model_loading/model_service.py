from pathlib import Path
from src.domain.exceptions import ModelNotFoundError, UnsupportedModelTypeError
from src.domain.repositories.model_repository import ModelRepository
from src.infrastructure.frameworks import FlairTaggerLoader, MT5Loader
from typing import Any, Dict
from src.utils.app_info import AppInfo

class ModelService(ModelRepository):
    def __init__(self):
        self.app_info = AppInfo.load()
        self.flair_tagger_loader = FlairTaggerLoader()
        self.mtfive_loader = MT5Loader()
        self._models: Dict[str, Dict[str, Any]] = {}
        self._load_supported_models()
        print(f"Loaded models: {self._models}")
    
    def _load_supported_models(self):
        self._models.clear()
        for es in self.app_info.entity_sets:
            es_id = es.entity_set_id
            self._models[es_id] = {}
            for m_cfg in es.supported_models:
                m_path = self._get_model_path(es, m_cfg)
                self._models[es_id][m_cfg.model_id] = self._load_model(m_cfg, m_path)

    def _load_model(self, model_cfg, model_path):
        if model_cfg.model_type == "NER-PG":
            return MT5Loader.load_model(model_path)
        elif model_cfg.model_type == "NER":
            return FlairTaggerLoader.load_model(model_path)
        
        raise UnsupportedModelTypeError(model_cfg.entity_set_id,
                                        model_cfg.model_id,
                                        model_cfg.model_type)
    
    def _get_model_path(self, entity_set, model_cfg) -> Path:
        return Path(
            *entity_set.supported_models_root_dir
            ).joinpath(
                *model_cfg.model_directory_name
            ).joinpath(
                model_cfg.model_version
            )
    
    def get_model(self, entity_set_id: str, model_id: str) -> Dict[str, Any]:
        try:
            return self._models[entity_set_id][model_id]
        except KeyError:
            raise ModelNotFoundError(entity_set_id, model_id)

    def list_models(self, entity_set_id: str) -> Dict[str, str]:
        return {
            m.model_id: m.model_type
            for m in self.app_info.get_entity_set(entity_set_id).supported_models
        }

    def get_model_config(self, entity_set_id: str, model_id: str) -> Dict[str, Any]:
        entity_set = self.app_info.get_entity_set(entity_set_id)
        model_config = next(
            (m for m in entity_set.supported_models if m.model_id == model_id), 
            None
        )
        if not model_config:
            raise ModelNotFoundError(entity_set_id, model_id)
        return model_config.dict()

    def reload_models(self) -> None:
        self._models = self._load_supported_models()
    
    # Existing model loading implementation...