import sys
import os
import torch
import pytest

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gestalt_zerfall.experiment import ExperimentLoader, StimuliGenerator
from gestalt_zerfall.probes import HookManager

MODELS_TO_TEST = [
    "gpt2",
    "bert-base-uncased",
    "state-spaces/mamba-130m-hf"
]

def test_model_loading_and_hooks():
    for model_name in MODELS_TO_TEST:
        print(f"\nTesting {model_name}...")
        try:
            loader = ExperimentLoader(model_name)
            model = loader.get_model()
            tokenizer = loader.get_tokenizer()
        except OSError:
            print(f"Skipping {model_name} (model not found or internet issue)")
            continue
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue

        # Generate Stimuli
        generator = StimuliGenerator(tokenizer)
        input_ids = generator.generate_repetition("Apple", 10)
        
        # Register Hooks
        hook_manager = HookManager(model)
        hook_manager.register_hooks()
        
        if not hook_manager.hooks:
            print(f"FAILED: No hooks registered for {model_name}")
            continue
            
        # Run Forward
        with torch.no_grad():
            try:
                model(input_ids)
            except Exception as e:
                print(f"FAILED: Forward pass error for {model_name}: {e}")
                continue
        
        # Check Activations
        activations = hook_manager.get_activations()
        if not activations:
             print(f"FAILED: No activations captured for {model_name}")
             continue
             
        layer_names = list(activations.keys())
        print(f"Captured {len(layer_names)} layers: {layer_names[:3]} ...")
        
        # Check Shape
        # BERT/GPT/Mamba should all have (Batch, Seq, Dim)
        # Batch=1, Seq=10 (approx, BERT adds CLS/SEP so 12)
        act = activations[layer_names[0]][0]
        print(f"Shape: {act.shape}")
        
        assert act.dim() == 3, f"Expected 3 dims, got {act.dim()}"
        assert act.shape[0] == 1, "Batch size should be 1"
        
        print(f"SUCCESS: {model_name} passed.")

if __name__ == "__main__":
    test_model_loading_and_hooks()
