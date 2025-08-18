from abc import ABC, abstractmethod
from typing import Dict, Any
from ..exceptions import ModelNotFoundError

class ModelRepository(ABC):
    @abstractmethod
    def get_model(self, entity_set_id: str, model_id: str) -> Dict[str, Any]:
        """
        Returns loaded model components
        :raises ModelNotFoundError: if model not found
        """
        pass

    @abstractmethod
    def list_models(self, entity_set_id: str) -> Dict[str, str]:
        """List available models for an entity set"""
        pass

    @abstractmethod
    def get_model_config(self, entity_set_id: str, model_id: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        pass

    @abstractmethod
    def reload_models(self) -> None:
        """Reload models from configuration"""
        pass