from abc import abstractmethod
from src.infrastructure.frameworks.model_loader import ModelLoader
from typing import Any


class CachedModelLoader(ModelLoader):
    """
    Base class ensuring caching across an app run.
    """
    _model = None

    def load(self) -> Any:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @abstractmethod
    def _load_model(self) -> Any:
        pass