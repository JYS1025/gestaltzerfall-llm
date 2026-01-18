import torch
import numpy as np

def compute_effective_rank(tensor: torch.Tensor) -> float:
    """
    Computes the Effective Rank (ER) of a matrix H (T x D).
    ER(H) = exp(Entropy(Normalized Singular Values))
    """
    # tensor shape: (Batch, Seq, Dim) -> We assume Batch=1 for now, or flatten
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0) # (Seq, Dim)
    
    # Center the data? 
    # Usually ER is defined on the covariance or the raw matrix. 
    # For "isotropization", we look at the raw geometry relative to origin or mean.
    # Let's use the raw singular values of the matrix H.
    
    # SVD
    try:
        # float32 might be unstable for very small values, but usually fine
        U, S, Vh = torch.linalg.svd(tensor, full_matrices=False)
    except RuntimeError:
        # Fallback for stability if needed
        return 1.0

    # Normalize singular values to form a probability distribution
    # p_i = sigma_i / sum(sigma_j)
    sigma_sum = S.sum()
    if sigma_sum == 0:
        return 0.0
        
    p = S / sigma_sum
    
    # Entropy: - sum(p * log(p))
    # Add epsilon to avoid log(0)
    entropy = -torch.sum(p * torch.log(p + 1e-12))
    
    # Effective Rank
    er = torch.exp(entropy).item()
    return er

def compute_cosine_drift(tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes Cosine Similarity between h_t and h_0 for all t.
    Returns tensor of shape (T,).
    """
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)
        
    h_0 = tensor[0].unsqueeze(0) # (1, D)
    
    # Cosine Similarity
    # F.cosine_similarity computes along dim=1 by default
    sim = torch.nn.functional.cosine_similarity(h_0, tensor, dim=1)
    return sim

def compute_l2_norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes L2 norm of h_t for all t.
    """
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    
    return torch.norm(tensor, p=2, dim=1)

def compute_attention_entropy(attn_matrix: torch.Tensor) -> torch.Tensor:
    """
    Computes entropy of attention distribution.
    attn_matrix: (Batch, Heads, Seq, Seq) or (Heads, Seq, Seq)
    Returns entropy per head per token: (Batch, Heads, Seq)
    """
    # We want entropy over the last dimension (attention to previous tokens)
    # H(p) = - sum(p * log(p))
    
    # Avoid log(0)
    entropy = -torch.sum(attn_matrix * torch.log(attn_matrix + 1e-12), dim=-1)
    return entropy
