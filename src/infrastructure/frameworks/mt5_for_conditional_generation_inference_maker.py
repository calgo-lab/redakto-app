from src.infrastructure.frameworks.model_inference_maker import ModelInferenceMaker
from src.infrastructure.frameworks.mt5_for_conditional_generation_loader import MT5ForConditionalGenerationLoader
from typing import Any, Dict, List

class MT5ForConditionalGenerationInferenceMaker(ModelInferenceMaker):
    """
    This class implements the infer method to return the inference result with a MT5ForConditionalGeneration model.
    """
    def __init__(self, model_loader: MT5ForConditionalGenerationLoader):
        """
        :param model_loader: The MT5ForConditionalGenerationLoader instance with the model object.
        """
        super().__init__(model_loader)

    def infer(self, input_text: str, **kwargs) -> List[str]:
        """
        Make inference using the loaded SequenceTagger model.
        :param input_text: The input text for which inference is to be made.
        :param **kwargs: Additional keyword arguments for inference.
        :return: The inference result as a list of output str.
        """
        model = self.model_loader.load()
        tokenizer = self.model_loader.tokenizer
        
        # extract tokenization kwargs
        max_length: int = kwargs.get("max_length", 512)
        padding: str = kwargs.get("padding", "max_length")
        truncation: bool = kwargs.get("truncation", True)

        inputs: Dict[str, Any] = tokenizer(input_text,
                                           return_tensors="pt",
                                           max_length=max_length,
                                           padding=padding,
                                           truncation=truncation)
        # extract repeat_count kwarg
        repeat_count: int = kwargs.get("repeat_count", 1)
        
        # extract generate kwargs
        max_length: int = kwargs.get("max_length", 512)
        temperature: float = kwargs.get("temperature", 0.8)
        do_sample: bool = kwargs.get("do_sample", True)
        top_k: int = kwargs.get("top_k", 100)

        output_texts: List[str] = list()
        for _ in range(1, repeat_count+1):
            outputs = model.generate(**inputs,
                                     max_length=max_length,
                                     temperature=temperature,
                                     do_sample=do_sample,
                                     top_k=top_k)
            output_texts.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        
        return output_texts