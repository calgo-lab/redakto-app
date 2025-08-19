from abc import ABC, abstractmethod
from typing import Any, Dict

class ModelRepository(ABC):
    @abstractmethod
    def get_model(self, entity_set_id: str, model_id: str) -> Dict[str, Any]:
        """
        Returns loaded model components
        """
        pass

    @abstractmethod
    def list_models(self, entity_set_id: str) -> Dict[str, str]:
        """
        List available models for an entity set
        """
        pass

    @abstractmethod
    def get_model_config(self, entity_set_id: str, model_id: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model
        """
        pass

    @abstractmethod
    def reload_models(self) -> None:
        """
        Reload model components from configuration
        """
        pass