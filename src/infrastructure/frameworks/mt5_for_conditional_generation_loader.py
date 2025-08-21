from src.infrastructure.frameworks.cached_model_loader import CachedModelLoader
from transformers import MT5ForConditionalGeneration, MT5TokenizerFast

class MT5ForConditionalGenerationLoader(CachedModelLoader):
    """
    Responsible for loading MT5 model and tokenizer.
    """
    def __init__(self, 
                 model_name_or_path: str,
                 loading_strategy: str = "local_disk_storage"):
        """
        :param model_name_or_path: The path or name of the MT5 model to load.
        :param loading_strategy: The strategy to use for loading the model (e.g., local_disk_storage).
        """
        super().__init__(model_name_or_path, loading_strategy)
        self._tokenizer: MT5TokenizerFast = None

    def _load_model(self) -> MT5ForConditionalGeneration:
        """
        Load the MT5ForConditionalGeneration model and return the model object.
        :return: The loaded MT5ForConditionalGeneration model object."""
        try:
            if self.loading_strategy == "local_disk_storage":
                self._tokenizer = MT5TokenizerFast.from_pretrained(self.model_name_or_path)
                return MT5ForConditionalGeneration.from_pretrained(self.model_name_or_path)
            else:
                raise ValueError(f"Unsupported loading strategy: {self.loading_strategy} for model at {self._model_name_or_path}")
        except Exception as e:
            print(f"Failed to load MT5 model for {self._model_name_or_path}: {e}")
    
    @property
    def tokenizer(self) -> MT5TokenizerFast:
        """
        Get the tokenizer for the loaded MT5 model.
        :return: The MT5TokenizerFast
        """
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded yet. Call load() first.")
        return self._tokenizer