from transformers import MT5ForConditionalGeneration, MT5TokenizerFast

class MT5Loader:
    @staticmethod
    def load_model(model_path: str):
        return (
            MT5ForConditionalGeneration.from_pretrained(model_path),
            MT5TokenizerFast.from_pretrained(model_path)
        )
    
    @staticmethod
    def generate(model, tokenizer, input_text: str):
        
        inputs = tokenizer(input_text,
                           return_tensors="pt",
                           max_length=512,
                           padding="max_length",
                           truncation=True)
        
        outputs = model.generate(**inputs,
                                 max_length=512,
                                 temperature=0.8,
                                 do_sample=True,
                                 top_k=100)
        
        return tokenizer.decode(outputs[0])