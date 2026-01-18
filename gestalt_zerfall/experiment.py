import torch
from transformers import AutoModel, AutoTokenizer

class ExperimentLoader:
    def __init__(self, model_name="gpt2"):
        """
        Loads the model and tokenizer.
        Supports: GPT-2, BERT, Mamba
        """
        print(f"Loading model: {model_name}...")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer
    
    def is_causal(self):
        """
        Returns True if the model is causal (GPT, Mamba), False if bidirectional (BERT).
        """
        # Heuristic based on model type or name
        if "bert" in self.model_name.lower() and "roberta" not in self.model_name.lower():
             # BERT is bidirectional. 
             # Note: RoBERTa is also bidirectional but we focus on BERT for now.
             return False
        return True

class StimuliGenerator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def generate_repetition(self, word: str, n: int) -> torch.Tensor:
        """
        Generates a sequence with the word repeated n times.
        Returns input_ids tensor of shape (1, T).
        """
        text = (word + " ") * n
        # Strip trailing space to be clean
        text = text.strip()
        return self.tokenizer(text, return_tensors="pt")['input_ids']

    def generate_control(self, text: str) -> torch.Tensor:
        """
        Generates a sequence from natural text.
        """
        return self.tokenizer(text, return_tensors="pt")['input_ids']
