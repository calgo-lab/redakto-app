import importlib.metadata

from packaging.requirements import Requirement
from packaging.version import Version, InvalidVersion
from pathlib import Path
from src.infrastructure.services.model_service import ModelService
from src.infrastructure.frameworks.model_loader import ModelLoader
from src.infrastructure.frameworks.sequence_tagger_loader import SequenceTaggerLoader
from src.infrastructure.frameworks.mt5_for_conditional_generation_loader import MT5ForConditionalGenerationLoader
from src.infrastructure.frameworks.sequence_tagger_inference_maker import SequenceTaggerInferenceMaker
from src.infrastructure.frameworks.mt5_for_conditional_generation_inference_maker import MT5ForConditionalGenerationInferenceMaker
from src.infrastructure.frameworks.model_inference_maker import ModelInferenceMaker
from src.domain.exceptions import ModelNotFoundError, UnsupportedModelLoadingStrategyError, UnsupportedModelImplTypeError
from src.utils import AppInfo
from typing import Any, Dict, List, Tuple


class ModelServiceImpl(ModelService):
    """
    Implementation of the ModelService interface.
    This service manages the loading and retrieval of models based on entity sets and model IDs.
    It uses the AppInfo utility to load application configuration and supported models.
    """
    _models_registry: Dict[str, Dict[str, Tuple[ModelLoader, ModelInferenceMaker]]] = {}
    def __init__(self):
        self._app_info = AppInfo.load()
        self._load_model_registry()
    
    def _load_model_registry(self):
        """
        Load the model registry from the app info configuration.
        This method initializes the _models_registry dictionary with models from the app info.
        It clears any existing models and reloads them based on the current configuration.
        """
        self._models_registry.clear()
        for entity_set_cfg in self._app_info.entity_sets:
            entity_set_id = entity_set_cfg.entity_set_id
            self._models_registry[entity_set_id] = {}
            for model_cfg in entity_set_cfg.supported_models:
                self._models_registry[entity_set_id][model_cfg.model_id] = self._load(entity_set_cfg, model_cfg)

    def check_requirements(self, requirements: List[str]) -> bool:
        """
        Check if all given requirements (like in requirements.txt format) 
        are installed in the current environment.
        :param requirements: List of requirement strings, e.g. ["flair==0.11.1", "torch>=2.0"]
        :return: True if all requirements are satisfied, False otherwise.
        """
        for req_str in requirements:
            try:
                req = Requirement(req_str)
                installed_version = importlib.metadata.version(req.name)
                if req.specifier and not req.specifier.contains(Version(installed_version), prereleases=True):
                    print(f"{req.name} {installed_version} does not satisfy {req.specifier}")
                    return False
                else:
                    print(f"{req.name} {installed_version} satisfies {req.specifier}")
            except importlib.metadata.PackageNotFoundError:
                print(f"{req_str} not installed")
                return False
            except InvalidVersion:
                print(f"Could not parse version for {req_str}")
                return False
            
        return True
    
    def _load(self, entity_set_cfg, model_cfg) -> Tuple[ModelLoader, ModelInferenceMaker]:
        """
        Load a model based on the entity set configuration and model configuration.
        :param entity_set_cfg: The configuration for the entity set.
        :param model_cfg: The configuration for the model.
        :return: A tuple containing the ModelLoader and ModelInferenceMaker instances.
        """
        model_loader: ModelLoader = None
        model_inference_maker: ModelInferenceMaker = None
        
        if not self.check_requirements(model_cfg.model_system_requirements):
            print(f"Requirements not satisfied, skipping model loading in this instance for model {model_cfg.model_id} in entity set {entity_set_cfg.entity_set_id}")
            return None, None
        
        if model_cfg.model_loading_strategy not in ["local_disk_storage"]:
            print(f"Unsupported model loading strategy {model_cfg.model_loading_strategy} for model {model_cfg.model_id} in entity set {entity_set_cfg.entity_set_id}")
            raise UnsupportedModelLoadingStrategyError(entity_set_cfg.entity_set_id, model_cfg.model_id, model_cfg.model_loading_strategy)
        
        if model_cfg.model_impl not in ["SequenceTagger", "MT5ForConditionalGeneration"]:
            print(f"Unsupported model impl type {model_cfg.model_impl} for model {model_cfg.model_id} in entity set {entity_set_cfg.entity_set_id}")
            raise UnsupportedModelImplTypeError(entity_set_cfg.entity_set_id, model_cfg.model_id, model_cfg.model_impl)
        
        if model_cfg.model_loading_strategy == "local_disk_storage":
            model_path = self._get_model_path(entity_set_cfg, model_cfg)
            if model_cfg.model_impl == "SequenceTagger":
                model_loader = SequenceTaggerLoader(model_name_or_path=model_path, loading_strategy=model_cfg.model_loading_strategy)
                model_inference_maker = SequenceTaggerInferenceMaker(model_loader=model_loader)
            elif model_cfg.model_impl == "MT5ForConditionalGeneration":
                model_loader = MT5ForConditionalGenerationLoader(model_name_or_path=model_path, loading_strategy=model_cfg.model_loading_strategy)
                model_inference_maker = MT5ForConditionalGenerationInferenceMaker(model_loader=model_loader)
        
        return model_loader, model_inference_maker
    
    def _get_model_path(self, entity_set_cfg, model_cfg) -> Path:
        """
        Construct the path to the model based on the entity set configuration and model configuration.
        :param entity_set_cfg: The configuration for the entity set.
        :param model_cfg: The configuration for the model.
        :return: The path to the model.
        """
        return Path(
            *entity_set_cfg.supported_models_root_dir
            ).joinpath(
                *model_cfg.model_directory_name
            ).joinpath(
                model_cfg.model_version
            )
    
    def get_model_inference_maker(self, entity_set_id: str, model_id: str) -> ModelInferenceMaker:
        """
        Retrieve the ModelInferenceMaker for the specified entity set and model ID.
        :param entity_set_id: The ID of the entity set for which the model is requested.
        :param model_id: The ID of the model to be used for inference.
        :return: An instance of ModelInferenceMaker for the specified model.
        """
        try:
            _, model_inference_maker = self._models_registry[entity_set_id][model_id]
            if not model_inference_maker:
                raise ModelNotFoundError(entity_set_id, model_id)
            return model_inference_maker
        except KeyError:
            raise ModelNotFoundError(entity_set_id, model_id)

    def list_models(self, entity_set_id: str) -> Dict[str, str]:
        """
        List available models for a given entity set.
        :param entity_set_id: The ID of the entity set for which models are listed.
        :return: A dictionary mapping model IDs to model types.
        """
        return {
            m.model_id: m.model_type
            for m in self._app_info.get_entity_set(entity_set_id).supported_models
        }

    def get_model_config(self, entity_set_id: str, model_id: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model.
        :param entity_set_id: The ID of the entity set for which the model configuration is requested.
        :param model_id: The ID of the model for which the configuration is requested.
        :return: A dictionary containing the model configuration.
        """
        entity_set = self._app_info.get_entity_set(entity_set_id)
        model_config = next(
            (m for m in entity_set.supported_models if m.model_id == model_id), 
            None
        )
        if not model_config:
            raise ModelNotFoundError(entity_set_id, model_id)
        return model_config.dict()

    def reload_model_registry(self) -> None:
        """
        Reload the model registry from the application configuration.
        This method should be called to refresh the model registry, typically after configuration changes.
        :return: None
        """
        self._models_registry = self._load_model_registry()

    