import torch

from flair.data import Dictionary
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from typing import List


class FlairTaggerLoader:
    @staticmethod
    def load_model(model_path: str):
        try:
            return SequenceTagger.load(model_path / "model.pt")
        except AttributeError as e:
            
            # TODO: Need to fix for flair new version fine-tuned sequence tagger, flair totally messed up with backward compatibility :(
            if "embedding_length" in str(e):
                print(f"[WARN] Detected Flair new version style model at {model_path}, converting...")

                saved = torch.load(model_path / "model.pt", map_location="cpu")

                state_dict = saved.get("state_dict", None)
                if state_dict is None:
                    raise RuntimeError("Unsupported model format: missing state_dict")

                tag_dictionary_items = saved.get("tag_dictionary")
                if isinstance(tag_dictionary_items, list):
                    tag_dictionary = Dictionary(tag_dictionary_items)
                else:
                    tag_dictionary = Dictionary(tag_dictionary_items.get_items())
                
                print(f"saved['embeddings']: {saved['embeddings']}")

                tagger = SequenceTagger(
                    hidden_size=saved["hidden_size"],
                    embeddings=TransformerWordEmbeddings(saved["embeddings"]),
                    tag_dictionary=tag_dictionary,
                    tag_type=saved["tag_type"]
                )
                tagger.load_state_dict(state_dict)

                print("[INFO] Model converted and loaded successfully.")
                return tagger

            raise
    
    @staticmethod
    def predict(tagger: SequenceTagger, sentences: List[Sentence]):
        tagger.predict(sentences)
        return sentences