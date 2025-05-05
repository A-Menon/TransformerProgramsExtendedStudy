import math
import torch
from torch import nn
from torch.nn import functional as F

# 1. PrefixSumCounts
# Convert one‑hot tokens → running count of that same token position‑wise.
class PrefixSumCounts(nn.Module):
    def __init__(self, d_vocab):
        super().__init__()
        self.d_vocab = d_vocab

    def forward(self, x):
        # x: (B, L) token IDs
        one_hot = F.one_hot(x, num_classes=self.d_vocab).float()  # use self.d_vocab
        cumsum = torch.cumsum(one_hot, dim=1)
        idx = x.unsqueeze(-1)  # (B, L, 1)
        counts = torch.gather(cumsum, dim=2, index=idx).float()  # (B, L, 1)
        return counts

# 2. SparseExpertCountingNetwork
# Sparse mixture of experts for counting-related tasks
# MoE layer learns to pick one of several purpose‑built counting experts per token
class HistogramExpert(nn.Module):
    def forward(self, x):
        return x.sum(-1, keepdim=True)

class FrequencyExpert(nn.Module):
    def forward(self, x):
        total = x.sum(-1, keepdim=True) + 1e-6
        return x / total

class UniquenessExpert(nn.Module):
    def forward(self, x):
        return (x != 0).float().sum(-1, keepdim=True)

class PatternCountExpert(nn.Module):
    def forward(self, x):
        # x: (N, hist_dim)
        diff = (x[:, 1:] != x[:, :-1]).float()
        pad = torch.zeros(x.size(0), 1, device=x.device)
        return torch.cat([pad, diff], dim=1).cumsum(1)

class SparseExpertCountingNetwork(nn.Module):
    def __init__(self, hist_dim, n_experts=4):
        super().__init__()
        self.experts = nn.ModuleList([
            HistogramExpert(),
            FrequencyExpert(),
            UniquenessExpert(),
            PatternCountExpert()
        ])
        self.router = nn.Linear(hist_dim, n_experts)

    def forward(self, histograms):
        # histograms: (B*n_blocks, hist_dim)
        logits = self.router(histograms)
        routes = F.gumbel_softmax(logits, hard=True, dim=-1)
        out = torch.zeros_like(histograms[..., :1])
        for i, exp in enumerate(self.experts):
            mask = routes[..., i] > 0
            if mask.any():
                expert_in = histograms[mask]
                expert_out = exp(expert_in)
                out[mask] = expert_out
        return out.squeeze(-1)