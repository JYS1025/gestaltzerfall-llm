import matplotlib.pyplot as plt
import numpy as np

def plot_effective_rank(ranks_per_layer, save_path=None):
    """
    ranks_per_layer: dict {layer_name: [er_t0, er_t1, ...]} or similar.
    Actually, we probably want ER vs Repetition Count (Sequence Length).
    """
    plt.figure(figsize=(10, 6))
    for layer, ranks in ranks_per_layer.items():
        plt.plot(ranks, label=layer)
    
    plt.xlabel("Layer Depth") # Or Sequence Position? 
    # Wait, ER is a property of the whole matrix or a window?
    # In the plan: "Effective Rank of the hidden state matrix H".
    # H is (T, d). So one ER value per layer for the whole sequence.
    # OR, we can do windowed ER.
    # For now, let's assume we plot ER per layer.
    
    plt.ylabel("Effective Rank")
    plt.title("Effective Rank across Layers")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_drift(drift_metrics, save_path=None):
    """
    drift_metrics: Tensor (T,)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(drift_metrics.numpy())
    plt.xlabel("Token Position")
    plt.ylabel("Metric Value")
    plt.title("Drift Metric over Time")
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_attention_entropy(entropy_per_layer, save_path=None):
    """
    entropy_per_layer: List of tensors or arrays, one per layer.
                       Each shape: (Seq_Len,) (averaged over heads)
    """
    plt.figure(figsize=(10, 6))
    for i, entropy in enumerate(entropy_per_layer):
        # entropy is (T,)
        plt.plot(entropy, label=f"Layer {i}")
    
    plt.xlabel("Token Position")
    plt.ylabel("Attention Entropy")
    plt.title("Attention Entropy over Time (Avg over Heads)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
