from flair.data import Sentence
from flair.models import SequenceTagger
from pathlib import Path
from typing import List

import flair
import os


class FlairTaggerLoader:
    
    @staticmethod
    def load_model(model_path: str):
        
        flair.cache_root = Path(os.path.join(*['/app', 'flair_cache_root']))
        try:
            return SequenceTagger.load(model_path / "model.pt")
        except Exception as e:
            print(f"Model loading failed for model_path, {model_path}: {str(e)}")
    
    @staticmethod
    def predict(tagger: SequenceTagger, sentences: List[Sentence]):
        tagger.predict(sentences)
        return sentences