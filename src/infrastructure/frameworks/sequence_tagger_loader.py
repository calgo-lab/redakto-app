from flair.models import SequenceTagger
from pathlib import Path
from src.infrastructure.frameworks.cached_model_loader import CachedModelLoader

import flair


class SequenceTaggerLoader(CachedModelLoader):
    """
    Responsible for loading a Flair SequenceTagger model - given a model_path.
    """
    def __init__(self, model_name_or_path: str, cache_root: Path = Path("/app/flair_cache_root")):
        self._model_name_or_path: str = model_name_or_path / "model.pt"
        self._cache_root: Path = cache_root

    def _load_model(self) -> SequenceTagger:
        flair.cache_root = self._cache_root
        try:
            return SequenceTagger.load(self._model_name_or_path)
        except Exception as e:
            print(f"Failed to load Flair model for {self._model_name_or_path}: {e}")