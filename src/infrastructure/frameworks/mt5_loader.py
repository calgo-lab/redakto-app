from transformers import MT5ForConditionalGeneration, MT5TokenizerFast
from typing import Any, Dict, Tuple

class MT5Loader:
    
    @staticmethod
    def load_model(model_path: str) -> Tuple[MT5ForConditionalGeneration, MT5TokenizerFast]:
        return (
            MT5ForConditionalGeneration.from_pretrained(model_path),
            MT5TokenizerFast.from_pretrained(model_path)
        )
    
    @staticmethod
    def tokenize_text(tokenizer: MT5TokenizerFast, input_text: str, **kwargs) -> Dict[str, Any]:
        return tokenizer(
            input_text,
            return_tensors="pt",
            **kwargs
        )
    
    @staticmethod
    def generate(model: MT5ForConditionalGeneration, 
                 tokenizer: MT5TokenizerFast, 
                 tokenized_inputs: Dict[str, Any],
                 **kwargs) -> str:
        
        outputs = model.generate(**tokenized_inputs, **kwargs)
        return tokenizer.decode(outputs[0])