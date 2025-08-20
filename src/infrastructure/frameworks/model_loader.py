from abc import ABC, abstractmethod
from typing import Any


class ModelLoader(ABC):
    """
    Interface for all model loaders.
    This interface defines the contract for loading models in different frameworks.
    Each loader should implement the load method to return the model object.
    """

    @abstractmethod
    def load(self) -> Any:
        """
        Load the model (only once) and return the model object.
        """
        pass
