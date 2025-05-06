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
        one_hot = F.one_hot(x, num_classes=self.d_vocab).float()
        cumsum = torch.cumsum(one_hot, dim=1)
        idx = x.unsqueeze(-1)
        counts = torch.gather(cumsum, dim=2, index=idx).float()
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
        maxcount,_ = x.max(-1, keepdim=True)
        return maxcount / total

class UniquenessExpert(nn.Module):
    def forward(self, x):
        return (x != 0).float().sum(-1, keepdim=True)

class PatternCountExpert(nn.Module):
    def forward(self, x):
        diff = (x[:, 1:] != x[:, :-1]).float()
        return diff.sum(-1, keepdim=True)

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
        logits = self.router(histograms)
        routes = F.gumbel_softmax(logits, tau=1.0, hard=not self.training, dim=-1)
        expert_outs = torch.stack(
            [exp(histograms) for exp in self.experts],
            dim=1
        )
        out = (routes.unsqueeze(-1) * expert_outs).sum(dim=1)
        return out.squeeze(-1)