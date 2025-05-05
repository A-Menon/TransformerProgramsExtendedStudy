import math, torch, numpy as np
from torch import nn
from torch.nn import functional as F


# Convert one‑hot tokens → running count of that same token position‑wise.
# Output shape: (B, L, 1)
class PrefixSumCounts(nn.Module):
    def __init__(self, d_vocab):
        super().__init__()
        self.d_vocab = d_vocab

    def forward(self, x):
        one_hot = F.one_hot(x, self.d_vocab).float()
        cumsum = torch.cumsum(one_hot, dim=1)
        idx = x.unsqueeze(-1)
        counts = torch.gather(cumsum, 2, idx).float()
        return counts

# H universal hashes → smaller categorical variables.
# Each variable size k' ≈ sqrt(|V|).  Collisions are expected.
class HashSketchEmbed(nn.Module):
    def __init__(self, d_vocab, H=4):
        super().__init__()
        self.H = H
        self.d_vocab = d_vocab
        p = 2 ** 31 - 1
        self.register_buffer("a", torch.randint(1, p, (H,)))
        self.register_buffer("b", torch.randint(0, p, (H,)))
        k_root = int(math.ceil(math.sqrt(d_vocab)))
        self.kp = k_root

    def forward(self, x):
        v = x.long().unsqueeze(0)
        h = (self.a.view(self.H,1,1) * v + self.b.view(self.H,1,1)) % (2**31-1)
        h = (h % self.kp).permute(1,2,0)
        return h

# Fixed-size block histogram.  If L not divisible by block_size, pads BOS.
# Produces *extra* categorical & numeric summary tokens prepended to sequence.
class ChunkAggregator(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block = block_size

    def forward(self, tokens, cat_embed_f, num_embed_f):
        if tokens.ndim == 3:
            B, L, H = tokens.shape
            tokens = tokens.view(B, L*H)
        else:
            B, L = tokens.shape
        pad = (-L) % self.block
        if pad:
            tokens = torch.cat([tokens,
                                torch.zeros(B,pad,dtype=tokens.dtype,
                                            device=tokens.device)], dim=1)
        B, L = tokens.shape
        blocks = tokens.view(B, L//self.block, self.block)
        vocab_size = num_embed_f.get_W().size(0)
        one_hot = F.one_hot(blocks, vocab_size).float()
        hist = one_hot.sum(-2)
        cat_dummy = blocks[..., 0]
        cat_emb = cat_embed_f(cat_dummy)
        idx = torch.clamp(hist.argmax(-1), max=vocab_size - 1).long()
        _ = num_embed_f(idx) * 0.0
        new_tokens = torch.cat([blocks[..., 0], tokens], dim=1)
        return new_tokens, cat_emb, hist
    
# Sparse mixture of experts for counting-related tasks
# MoE layer learns to pick one of several purpose‑built counting experts per token
class _BaseCountExpert(nn.Module):
    def __init__(self, d_model): super().__init__()
    def forward(self, x): raise NotImplementedError()

class HistogramExpert(_BaseCountExpert):
    def forward(self, x):
        return x.sum(-1, keepdim=True)

class FrequencyExpert(_BaseCountExpert):
    def forward(self, x):
        return x / (x.sum(-1, keepdim=True) + 1e-6)

class UniquenessExpert(_BaseCountExpert):
    def forward(self, x):
        return (x != 0).float().sum(-1, keepdim=True)

class PatternCountExpert(_BaseCountExpert):
    def forward(self, x):
        diff = (x[:, 1:] != x[:, :-1]).float()
        pad = torch.zeros_like(diff[:, :1])
        return torch.cat([pad, diff], 1).cumsum(1)

class SparseExpertCountingNetwork(nn.Module):
    def __init__(self, d_model, n_experts=4, capacity_factor=1.5):
        super().__init__()
        self.experts = nn.ModuleList([
            HistogramExpert(d_model),
            FrequencyExpert(d_model),
            UniquenessExpert(d_model),
            PatternCountExpert(d_model)
        ])
        self.router = nn.Linear(d_model, n_experts)

    def forward(self, x, **kwargs):
        logits = self.router(x)
        routes = F.gumbel_softmax(logits, hard=True, dim=-1)
        out = torch.zeros_like(x)
        for i, exp in enumerate(self.experts):
            m = routes[..., i] > 0
            if m.any():
                out[m] = exp(x[m])
        return out
    
# Makes token representations more semantic while maintaining discreteness for interpretability
class ContrastiveTokenRepresentations(nn.Module):
    def __init__(self, d_vocab, n_buckets=32, temperature=0.07):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(n_buckets, d_vocab))
        self.temperature = temperature
        self.n_buckets = n_buckets

    def forward(self, onehot_tokens):
        sims = torch.matmul(onehot_tokens, self.prototypes.t())
        sims = sims / self.temperature
        hard = F.gumbel_softmax(sims, hard=True, dim=-1)
        return hard
    
# Memory network for longer sequences
# Vaguely inspired by RAG
class PositionalNgramMemoryNetwork(nn.Module):
    def __init__(self, d_model, max_ngram=3, memory_slots=64):
        super().__init__()
        self.max_ngram = max_ngram
        self.d_model = d_model
        self.memory = nn.Parameter(torch.randn(memory_slots, max_ngram, d_model))
        self.pos_bias = nn.Parameter(torch.randn(memory_slots, max_ngram))

    def extract_ngrams(self, x):
        B, L, d = x.shape
        pad = torch.zeros(B, self.max_ngram-1, d, device=x.device)
        xpad = torch.cat([pad, x], 1)
        ngrams = []
        for n in range(self.max_ngram):
            ngrams.append(xpad[:, n:n+L])
        return torch.stack(ngrams, 2)

    def forward(self, x):
        ngrams = self.extract_ngrams(x)
        mem = self.memory.unsqueeze(0).unsqueeze(0)
        sims = (ngrams.unsqueeze(3) * mem).sum(-1)
        scores = sims + self.pos_bias.view(1,1,*self.pos_bias.shape)
        best = scores.argmax(-1)
        enhanced = torch.gather(
            self.memory, 0,
            best.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,1,self.d_model)
        ).sum(2)
        return enhanced