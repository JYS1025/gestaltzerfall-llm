import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from gestalt_zerfall.experiment import ExperimentLoader, StimuliGenerator
from gestalt_zerfall.probes import HookManager
from gestalt_zerfall.metrics import compute_effective_rank, compute_cosine_drift, compute_l2_norm, compute_attention_entropy
from gestalt_zerfall.visualize import plot_effective_rank, plot_drift, plot_attention_entropy

def run_single_model(model_name, cfg):
    print(f"\n=== Running Experiment for {model_name} ===")
    
    try:
        loader = ExperimentLoader(model_name)
        model = loader.get_model()
        tokenizer = loader.get_tokenizer()
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None

    # Stimuli
    N_REPEATS = cfg.experiment.n_repeats
    WORD = cfg.experiment.stimulus_word
    generator = StimuliGenerator(tokenizer)
    input_ids = generator.generate_repetition(WORD, N_REPEATS)
    
    # Hooks
    hook_manager = HookManager(model)
    hook_manager.register_hooks()
    
    # Run
    print("Running Forward Pass...")
    with torch.no_grad():
        try:
            outputs = model(input_ids, output_attentions=True)
            attentions = outputs.attentions if hasattr(outputs, 'attentions') else None
        except Exception as e:
            print(f"Warning: Could not get attentions for {model_name} ({e}). Using default forward.")
            model(input_ids)
            attentions = None
    
    activations = hook_manager.get_activations()
    
    # Analysis
    er_per_layer = []
    layer_names = []
    
    # Drift (Last Layer)
    if not activations:
        print(f"No activations captured for {model_name}!")
        return None

    last_layer_name = list(activations.keys())[-1]
    last_layer_act = activations[last_layer_name][0].squeeze(0) # (T, D)
    
    drift = compute_cosine_drift(last_layer_act)
    l2 = compute_l2_norm(last_layer_act)
    
    # ER
    for name, acts in activations.items():
        act = acts[0].squeeze(0)
        er = compute_effective_rank(act)
        er_per_layer.append(er)
        layer_names.append(name)

    # Entropy
    entropy_per_layer = []
    if attentions is not None:
        for i, layer_attn in enumerate(attentions):
            entropy = compute_attention_entropy(layer_attn)
            avg_entropy = entropy.mean(dim=1).squeeze(0)
            entropy_per_layer.append(avg_entropy.numpy())
    
    return {
        "model_name": model_name,
        "er": er_per_layer,
        "drift": drift.numpy(),
        "l2": l2.numpy(),
        "entropy": entropy_per_layer,
        "layer_names": layer_names
    }

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    RESULTS_DIR = cfg.experiment.results_dir
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    models = cfg.models
    results = {}
    
    for model_name in models:
        res = run_single_model(model_name, cfg)
        if res:
            results[model_name] = res
            
    # Comparative Plots
    print("\nGenerating Comparative Plots...")
    
    # 1. Drift Comparison
    plt.figure(figsize=(12, 6))
    for name, res in results.items():
        plt.plot(res["drift"], label=name)
    plt.xlabel("Token Position")
    plt.ylabel("Cosine Similarity (to t=0)")
    plt.title(f"Semantic Drift Comparison ('{cfg.experiment.stimulus_word}' x {cfg.experiment.n_repeats})")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "compare_drift.png"))
    plt.close()
    
    # 2. ER Comparison
    plt.figure(figsize=(12, 6))
    
    if cfg.visualization.truncate_layers:
        min_layers = min(len(res["er"]) for res in results.values())
        print(f"Truncating ER plot to first {min_layers} layers for consistency.")
        title_suffix = f"(First {min_layers} Layers)"
    else:
        title_suffix = "(All Layers)"
    
    for name, res in results.items():
        er_data = res["er"]
        if cfg.visualization.truncate_layers:
            er_data = er_data[:min_layers]
        plt.plot(er_data, label=name, marker='o')
        
    plt.xlabel("Layer Depth")
    plt.ylabel("Effective Rank")
    plt.title(f"Effective Rank Comparison {title_suffix}")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "compare_er.png"))
    plt.close()

    # 3. Attention Entropy Comparison
    plt.figure(figsize=(12, 6))
    has_entropy = False
    for name, res in results.items():
        if res["entropy"] and len(res["entropy"]) > 0:
            has_entropy = True
            ent_stack = np.stack(res["entropy"])
            global_avg_ent = np.mean(ent_stack, axis=0)
            plt.plot(global_avg_ent, label=f"{name} (Avg)")
            
    if has_entropy:
        plt.xlabel("Token Position")
        plt.ylabel("Average Attention Entropy")
        plt.title("Attention Entropy Comparison (Global Average)")
        plt.legend()
        plt.savefig(os.path.join(RESULTS_DIR, "compare_entropy.png"))
    plt.close()

    print(f"Experiment Complete. Results saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    main()
