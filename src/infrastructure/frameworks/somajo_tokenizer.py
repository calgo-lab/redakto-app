from somajo import SoMaJo
from typing import List

class SoMaJoTokenizer:
    """
    This class provides methods to tokenize text using the SoMaJo tokenizer.
    """

    def __init__(self, 
                 language: str = "de_CMC", 
                 split_camel_case: bool = False):
        """
        Initialize the SoMaJo tokenizer.
        :param language: The language model to use for tokenization.
        :param split_camel_case: Whether to split camel case words.
        """
        self._tokenizer = SoMaJo(language, split_camel_case=split_camel_case)

    def tokenize(self, text: str) -> List[List[str]]:
        """
        Tokenize the input text into sentences.
        :param text: The input text to tokenize.
        :return: A list of tokenized sentences.
        """
        tokenized_sentences: List[List[str]] = list()
        for sentence in self._tokenizer.tokenize_text([text]):
            tokenized_sentences.append([token.text for token in sentence])
        return tokenized_sentences