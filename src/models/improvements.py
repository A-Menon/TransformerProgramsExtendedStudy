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

# 2. HashSketchEmbed
# H universal hashes → smaller categorical variables.
# Each variable size k' ≈ sqrt(|V|).  Collisions are expected.
class HashSketchEmbed(nn.Module):
    def __init__(self, d_vocab, H=4):
        super().__init__()
        self.H = H
        self.d_vocab = d_vocab
        self.p = 2**31 - 1
        self.register_buffer("a", torch.randint(1, self.p, (H,)))
        self.register_buffer("b", torch.randint(0, self.p, (H,)))
        self.kp = int(math.ceil(math.sqrt(d_vocab)))

    def forward(self, x):
        # x: (B, L)
        v = x.long().unsqueeze(0)  # (1, B, L)
        h = (self.a.view(self.H,1,1) * v + self.b.view(self.H,1,1)) % self.p
        h = (h % self.kp).permute(1,2,0)  # (B, L, H)
        return h

# 3. ChunkAggregator
# Produces *extra* categorical & numeric summary tokens prepended to sequence.
class ChunkAggregator(nn.Module):
    def __init__(self, block_size, bos_id=0):
        super().__init__()
        self.block = block_size
        self.bos_id = bos_id

    def forward(self, tokens, cat_embed_f, num_embed_f, token_embed_f):
        # tokens: (B, L)
        B, L = tokens.shape
        pad_len = (-L) % self.block
        if pad_len > 0:
            tokens = F.pad(tokens, (0, pad_len), value=self.bos_id)
            L += pad_len
        n_blocks = L // self.block

        # block-level category IDs: first token in each block
        blocks = tokens.view(B, n_blocks, self.block)
        cat_ids = blocks[:, :, 0]  # (B, n_blocks)
        cat_emb = cat_embed_f(cat_ids)  # (B, n_blocks, D_cat)

        # block histograms
        vocab = num_embed_f.get_W().size(0)
        one_hot = F.one_hot(blocks, num_classes=vocab).float()
        hist = one_hot.sum(dim=2)  # (B, n_blocks, vocab)
        num_emb = num_embed_f(hist)  # (B, n_blocks, D_num)

        # original token embeddings
        token_embs = token_embed_f(tokens)  # (B, L, D_token)

        # concatenate [cat_emb, num_emb] at start of sequence
        new_seq = torch.cat([cat_emb, num_emb, token_embs], dim=1)
        return new_seq, cat_ids, hist

# 4. SparseExpertCountingNetwork
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