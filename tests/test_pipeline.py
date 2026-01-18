import sys
import os
import torch

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gestalt_zerfall.experiment import ExperimentLoader, StimuliGenerator
from gestalt_zerfall.probes import HookManager
from gestalt_zerfall.metrics import compute_effective_rank, compute_cosine_drift, compute_l2_norm

def test_pipeline():
    print("Initializing Experiment...")
    # Use a very small model for testing if possible, or just GPT-2 small
    loader = ExperimentLoader("gpt2")
    model = loader.get_model()
    tokenizer = loader.get_tokenizer()
    
    print("Generating Stimuli...")
    generator = StimuliGenerator(tokenizer)
    # "Apple" repeated 10 times
    input_ids = generator.generate_repetition("Apple", 10)
    print(f"Input shape: {input_ids.shape}")
    
    print("Registering Hooks...")
    hook_manager = HookManager(model)
    hook_manager.register_hooks()
    
    print("Running Forward Pass...")
    with torch.no_grad():
        outputs = model(input_ids)
    
    print("Extracting Activations...")
    activations = hook_manager.get_activations()
    print(f"Captured layers: {list(activations.keys())}")
    
    # Check shape of first layer
    first_layer = list(activations.keys())[0]
    h_0 = activations[first_layer][0] # List of tensors, get first batch
    print(f"Layer {first_layer} shape: {h_0.shape}")
    
    # Verify shape: (1, T, D) -> (1, 10, 768) for GPT-2 small
    assert h_0.shape[1] == 10
    assert h_0.shape[2] == 768
    
    print("Computing Metrics...")
    er = compute_effective_rank(h_0)
    print(f"Effective Rank: {er}")
    
    drift = compute_cosine_drift(h_0)
    print(f"Drift (Cosine Sim): {drift}")
    
    l2 = compute_l2_norm(h_0)
    print(f"L2 Norm: {l2}")
    
    print("Test Passed!")

if __name__ == "__main__":
    test_pipeline()
