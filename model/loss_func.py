import torch
import torch.nn.functional as F

def nt_xent_loss(z_i, z_j, z_negatives, tau=0.5):
    """
    Compute the NT-Xent loss for a given positive pair (z_i, z_j) and negative pairs z_negatives.
    
    Parameters:
    z_i (torch.Tensor): Embedding vector of the anchor (positive pair part 1).
    z_j (torch.Tensor): Embedding vector of the positive counterpart (positive pair part 2).
    z_negatives (torch.Tensor): Embedding vectors of negative samples.
    tau (float): Temperature parameter for scaling.
    
    Returns:
    torch.Tensor: Computed NT-Xent loss.
    """
    # Normalize the embeddings
    z_i = F.normalize(z_i, dim=0)
    z_j = F.normalize(z_j, dim=0)
    z_negatives = F.normalize(z_negatives, dim=1)

    # Compute similarity between positive pair
    sim_ij = torch.exp(sim(z_i, z_j,tau))
    
    # Compute similarity between anchor and all negatives
    sim_in = torch.exp( sim(z_i.unsqueeze(0), z_negatives.t(), tau) ).sum()
    
    # Total similarity including positive and negative pairs
    total_sim = sim_ij + sim_in
    
    # Compute NT-Xent loss
    loss = -torch.log(sim_ij / total_sim)
    
    return loss

# def sim(u, v, tau):
#     return torch.dot(u, v) / (torch.norm(u) * torch.norm(v)) / tau

def sim(u, v, tau):
    """
    Compute cosine similarity for batches of vectors.

    Args:
    u (torch.Tensor): Tensor of shape (batch_size, features).
    v (torch.Tensor): Tensor of shape (batch_size, features).
    tau (float): Temperature scaling factor.

    Returns:
    torch.Tensor: Cosine similarities scaled by temperature, shape (batch_size,).
    """

    u_norm = u / u.norm(dim=(u.ndim-1), keepdim=True)
    v_norm = v / v.norm(dim=(v.ndim-1), keepdim=True)
    similarity = (u_norm @ v_norm) / tau
    return similarity