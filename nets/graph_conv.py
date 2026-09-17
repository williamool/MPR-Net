"""
Top-K Graph Construction (GC) and Graph-Based Feature Propagation (GFP)
of the Prototype-Guided Graph Reasoning (PGR) module, Sec. III-C-2/3 in the paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GraphUpdate(nn.Module):
    """Two-layer GCN, Eq. (26): H1 = ReLU(A_hat X W1), H2 = A_hat H1 W2."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        edge_index = edge_index.to(x.device)
        q = self.gcn1(x, edge_index)
        x = F.relu(q)
        x = self.gcn2(x, edge_index)
        return x


def select_top_k_pixels(score, k_ratio=0.04):
    """Select the Top-K spatial locations of the prototype-guided score map, K_g = floor(r * H * W)."""
    batch_size, height, width = score.shape
    num_pixels = height * width
    num_select = int(num_pixels * k_ratio)

    flat_score = score.view(batch_size, -1)                     # [B, H*W]
    _, top_indices = torch.topk(flat_score, num_select, dim=1)  # [B, K_g]

    h_indices = top_indices // width
    w_indices = top_indices % width
    top_indices_2d = torch.stack([h_indices, w_indices], dim=-1)  # [B, K_g, 2]

    return top_indices_2d, top_indices


def build_edge_index_from_features(features, threshold=0.55):
    """Similarity graph among candidate nodes, Eq. (25): A_ij = 1 if cos(x_i, x_j) > tau else 0."""
    norm_features = F.normalize(features, p=2, dim=1)
    similarity_matrix = torch.mm(norm_features, norm_features.t())  # [N, N]

    # one column per edge: [2, num_edges]
    edges = torch.nonzero(similarity_matrix > threshold, as_tuple=False).t()

    # degenerate case (no edge at all): fall back to self-loops so that every node is kept
    if edges.size(1) == 0:
        num_nodes = features.size(0)
        edges = torch.arange(num_nodes).repeat(2, 1)

    return edges


def update_features_with_gcn(z, score, graph_update_module, k_ratio=0.001, similarity_threshold=0.6):
    """
    Graph-Based Feature Propagation.

    z     : prototype-enhanced feature F_p, [B, C, H, W]
    score : prototype-guided score map S,  [B, H, W]
    Returns the refined feature F_r = Scatter(F_p, H2, T), Eq. (27).
    """
    batch_size, channels, height, width = z.shape

    top_indices_2d, flat_indices = select_top_k_pixels(score, k_ratio)

    # gather node features X of the selected Top-K candidates
    flat_z = z.view(batch_size, channels, -1)                                                  # [B, C, H*W]
    top_features = flat_z.gather(2, flat_indices.unsqueeze(1).expand(-1, channels, -1))        # [B, C, K_g]
    top_features = top_features.permute(0, 2, 1).reshape(-1, channels)                         # [B*K_g, C]

    # graph construction (Eq. 25) + two-layer GCN propagation (Eq. 26)
    edge_index = build_edge_index_from_features(top_features, similarity_threshold)
    updated_features = graph_update_module(top_features, edge_index)                           # [B*K_g, C]

    # scatter the refined node features back to their spatial locations (Eq. 27)
    updated_features = updated_features.view(batch_size, -1, channels).permute(0, 2, 1)        # [B, C, K_g]
    updated_features = updated_features.to(z.dtype)
    new_z_flat = z.clone().view(batch_size, channels, -1)
    new_z_flat = new_z_flat.scatter(2, flat_indices.unsqueeze(1).expand(-1, channels, -1), updated_features)
    new_z = new_z_flat.view(batch_size, channels, height, width)

    return new_z
