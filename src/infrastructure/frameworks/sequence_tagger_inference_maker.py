from flair.data import Sentence
from src.infrastructure.frameworks.model_inference_maker import ModelInferenceMaker
from src.infrastructure.frameworks.sequence_tagger_loader import SequenceTaggerLoader
from src.infrastructure.frameworks.somajo_tokenizer import SoMaJoTokenizer
from typing import Any, List

class SequenceTaggerInferenceMaker(ModelInferenceMaker):
    """
    This class implements the infer method to return the inference result with a Flair SequenceTagger model.
    """
    _somajo_tokenizer: SoMaJoTokenizer = None

    def __init__(self, model_loader: SequenceTaggerLoader):
        """
        :param model_loader: The SequenceTaggerLoader instance with the model object.
        """
        super().__init__(model_loader)
        self._load_somajo_tokenizer()

    def infer(self, input_text: str, **kwargs) -> List[Sentence]:
        """
        Make inference using the loaded SequenceTagger model.
        :param input_text: The input text for which inference is to be made.
        :param **kwargs: Additional keyword arguments for inference.
        :return: The inference result as a list of Flair Sentence objects.
        """
        tagger = self.model_loader.load()
        sentences: List[Sentence] = self._get_somajo_tokenized_flair_sentences(input_text)
        tagger.predict(sentences)
        return sentences
    
    def _get_somajo_tokenized_flair_sentences(self, text: str) -> List[Sentence]:
        sentences: List[Sentence] = list()
        for tokenized_sentence in self._somjo_tokenizer.tokenize(text):
            sentences.append(Sentence(tokenized_sentence))
        return sentences
    
    def _load_somajo_tokenizer(self):
        """
        Load the SoMaJo tokenizer.
        """
        if self._somajo_tokenizer is None:
            self._somajo_tokenizer = SoMaJoTokenizer()
        return self._somajo_tokenizer