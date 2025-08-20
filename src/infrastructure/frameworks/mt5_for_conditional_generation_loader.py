from src.infrastructure.frameworks.cached_model_loader import CachedModelLoader
from transformers import MT5ForConditionalGeneration, MT5TokenizerFast

class MT5ForConditionalGenerationLoader(CachedModelLoader):
    """
    Responsible for loading mT5 model and tokenizer.
    """
    def __init__(self, model_name_or_path: str):
        self._model_name_or_path: str = model_name_or_path
        self._model: MT5ForConditionalGeneration = None
        self._tokenizer: MT5TokenizerFast = None

    def _load_model(self) -> MT5ForConditionalGeneration:
        try:
            self._tokenizer = MT5TokenizerFast.from_pretrained(self._model_name_or_path)
            return MT5ForConditionalGeneration.from_pretrained(self._model_name_or_path)
        except Exception as e:
            print(f"Failed to load MT5 model for {self._model_name_or_path}: {e}")
    
    @property
    def tokenizer(self) -> MT5TokenizerFast:
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded yet. Call load() first.")
        return self._tokenizer