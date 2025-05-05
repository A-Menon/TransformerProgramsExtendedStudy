from functools import partial
import itertools
import math

import einops
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from models.improvements import PrefixSumCounts, HashSketchEmbed, ChunkAggregator, SparseExpertCountingNetwork, ContrastiveTokenRepresentations, PositionalNgramMemoryNetwork

from utils import logging

logger = logging.get_logger(__name__)


def softmax(X, dim=-1, tau=1.0):
    return F.softmax(X / tau, dim=dim)


def softmax_no_temp(X, dim=-1, **kwargs):
    return F.softmax(X, dim=dim)


def gumbel_hard(X, dim=-1, tau=1.0):
    return F.gumbel_softmax(X, tau=tau, dim=dim, hard=True)


def gumbel_soft(X, dim=-1, tau=1.0):
    return F.gumbel_softmax(X, tau=tau, dim=dim, hard=False)


def argmax(X, dim=-1, **kwargs):
    pred = X.argmax(dim=dim)
    return torch.zeros_like(X).scatter_(dim, pred.unsqueeze(dim), 1.0)


def subsoftmax(X, dim, k, tau=1.0, f=softmax):
    shape = X.shape
    dim = len(shape) - 1 if dim == -1 else dim
    new_shape = shape[:dim] + (shape[dim] // k, k) + shape[dim + 1 :]
    return f(X.view(new_shape), dim=dim + 1, tau=tau).view(shape)


# A helper class to get access to intermediate activations (inspired by Garcon)
# It's a dummy module that is the identity function by default
# Source:
# https://colab.research.google.com/drive/19gn2tavBGDqOYHLatjSROhABBD5O_JyZ
class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.fwd_hooks = []
        self.bwd_hooks = []

    def give_name(self, name):
        # Called by the model at initialisation
        self.name = name

    def add_hook(self, hook, dir="fwd"):
        # Hook format is fn(activation, hook_name)
        # Change it into PyTorch hook format (this includes input and output,
        # which are the same for a HookPoint)
        def full_hook(module, module_input, module_output):
            return hook(module_output, name=self.name)

        if dir == "fwd":
            handle = self.register_forward_hook(full_hook)
            self.fwd_hooks.append(handle)
        elif dir == "bwd":
            handle = self.register_backward_hook(full_hook)
            self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Invalid direction {dir}")

    def remove_hooks(self, dir="fwd"):
        if (dir == "fwd") or (dir == "both"):
            for hook in self.fwd_hooks:
                hook.remove()
            self.fwd_hooks = []
        if (dir == "bwd") or (dir == "both"):
            for hook in self.bwd_hooks:
                hook.remove()
            self.bwd_hooks = []
        if dir not in ["fwd", "bwd", "both"]:
            raise ValueError(f"Invalid direction {dir}")

    def forward(self, x, **kwargs):
        return x


# Define network architecture. Adapted from
# https://colab.research.google.com/drive/19gn2tavBGDqOYHLatjSROhABBD5O_JyZ
class CatEmbed(nn.Module):
    def __init__(
        self,
        d_vocab,
        n_vars,
        d_var,
        temp=1.0,
        init_emb=None,
        sample_fn=None,
        one_hot_embed=False,
        **kwargs,
    ):
        super().__init__()
        d_model = n_vars * d_var
        if one_hot_embed:
            assert n_vars == 1
            self.W_E = nn.Parameter(torch.eye(d_var)[:, :d_vocab] * 20.0 - 10.0)
            self.W_E.requires_grad = False
        elif init_emb is not None:
            assert init_emb.shape[0] == d_model
            self.W_E = nn.Parameter(init_emb)
        else:
            self.W_E = nn.Parameter(
                (torch.randn(d_model, d_vocab) / np.sqrt(d_model))
            )
        self.d_var = d_var
        self.n_vars = n_vars
        self.temp = temp
        self.sample_fn = sample_fn

    def set_temp(self, temp, sample_fn=None):
        self.temp = temp
        if sample_fn is not None:
            self.sample_fn = sample_fn

    def get_W(self):
        return subsoftmax(
            self.W_E, 0, self.d_var, tau=self.temp, f=self.sample_fn
        )

    def forward(self, x, **kwargs):
        W_E = self.get_W()
        return F.embedding(x, W_E.T)

    @property
    def device(self):
        return self.W_E.device


class NumEmbed(nn.Module):
    def __init__(
        self,
        d_vocab,
        n_vars,
        temp=1.0,
        init_emb=None,
        count_only=False,
        **kwargs,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.init_emb = None
        if count_only:
            self.W_E = nn.Parameter(torch.ones(d_vocab, n_vars))
            self.W_E.requires_grad = False
        elif init_emb is not None:
            assert init_emb.shape[0] == n_vars
            self.W_E = nn.Parameter(init_emb.T)
        else:
            self.W_E = nn.Parameter(
                (torch.randn(d_vocab, n_vars) / np.sqrt(n_vars))
            )

    def get_W(self):
        return self.W_E

    def forward(self, x, **kwargs):
        W_E = self.get_W()
        return F.embedding(x, W_E)


class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model, mask=None):
        super().__init__()
        self.W_U = nn.Parameter(
            torch.randn(d_model, d_vocab) / np.sqrt(d_vocab)
        )
        if mask is not None:
            self.register_buffer(
                "mask", torch.tensor(mask).view(1, 1, len(mask))
            )
        else:
            self.mask = None

    def forward(self, x, **kwargs):
        logits = x @ self.W_U
        if self.mask is not None:
            logits = logits.masked_fill(self.mask, -1e30)
        return logits


class OneHotPosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model, d_var=None, temp=1.0):
        super().__init__()
        self.d_model = d_model
        self.max_ctx = max_ctx
        self.register_buffer(
            "W",
            torch.cat(
                [torch.eye(max_ctx), torch.zeros(max_ctx, d_model - max_ctx)],
                -1,
            ),
        )
        self.temp = temp

    def set_temp(self, temp, sample_fn=None):
        self.temp = temp
        if sample_fn is not None:
            self.sample_fn = sample_fn

    def get_W(self):
        return self.W

    def forward(self, x, **kwargs):
        W_pos = self.get_W()
        seq_len = x.shape[1]
        
        if seq_len > self.max_ctx:
            pos_emb = W_pos.unsqueeze(0).repeat(x.size(0), 1, 1)
            padding = torch.zeros(
                x.size(0), seq_len - self.max_ctx, W_pos.shape[-1],
                device=x.device
            )
            return torch.cat([pos_emb, padding], dim=1)
        else:
            return torch.zeros_like(x[:, :, :W_pos.shape[-1]]) + W_pos[:seq_len]


class ConstrainedRead(nn.Module):
    def __init__(
        self,
        d_in,
        d_var,
        n_heads,
        n_vars=None,
        temp=1.0,
        sample_fn=softmax,
    ):
        super().__init__()
        self.d_in, self.d_var = d_in, d_var
        self.n_vars = max(n_vars or d_in // d_var, 1)
        self.n_heads = n_heads
        self.W = nn.Parameter(
            torch.randn(n_heads, self.n_vars) / np.sqrt(self.n_vars)
        )
        self.register_buffer(
            "E",
            torch.stack(
                [torch.eye(d_in // self.n_vars) for _ in range(self.n_vars)], 0
            ),
        )
        self.sample_fn = sample_fn
        self.temp = temp

    def fix_read(self, var_idxs):
        if type(var_idxs) == int:
            var_idxs = [var_idxs] * self.n_heads
        W = torch.ones_like(self.W.data) * -10
        for h, idx in enumerate(var_idxs):
            W[h, idx] = 10
        self.W.data = W
        self.W.requires_grad = False

    def set_temp(self, temp, sample_fn=None):
        self.temp = temp
        if sample_fn is not None:
            self.sample_fn = sample_fn

    def get_W(self):
        return self.sample_fn(self.W, tau=self.temp, dim=-1)

    def forward(self, **kwargs):
        var_probs = self.get_W()
        read_matrix = torch.einsum("ab,bcd->abcd", var_probs, self.E)
        return einops.rearrange(read_matrix, "a b c d -> a d (b c)")


class WPred(nn.Module):
    def __init__(self, n_heads, d_head, temp=1.0, sample_fn=softmax):
        super().__init__()
        self.temp = temp
        self.W = nn.Parameter(
            torch.randn(n_heads, d_head, d_head) / np.sqrt(d_head)
        )
        self.sample_fn = sample_fn

    def get_W(self):
        return self.sample_fn(self.W, tau=self.temp, dim=-1)

    def set_temp(self, temp, sample_fn=None):
        self.temp = temp
        if sample_fn is not None:
            self.sample_fn = sample_fn

    def forward(self, W_Q):
        # W_Q -- (n_heads, d_head, d_in)
        W = self.get_W()
        return (torch.matmul(W_Q.permute(0, 2, 1), W)).permute(0, 2, 1)


class WScore(nn.Module):
    def __init__(self, n_heads, d_head, temp=1.0):
        super().__init__()
        self.temp = temp
        self.W = nn.Parameter(
            torch.randn(n_heads, d_head, d_head) / np.sqrt(d_head)
        )

    def get_W(self):
        return self.W

    def set_temp(self, temp, sample_fn=None):
        self.temp = temp

    def forward(self, W_Q):
        W = self.get_W()
        return (torch.matmul(W_Q.permute(0, 2, 1), W)).permute(0, 2, 1)


class ScoreAttention(nn.Module):
    def __init__(
        self,
        d_in,
        n_heads,
        d_head,
        n_ctx,
        model,
        temp=1.0,
        n_vars=None,
        sample_fn=softmax,
        attention_temp=None,
        **kwargs,
    ):
        super().__init__()
        d_out = d_head * n_heads
        self.model = model
        # n_heads, d_head, d_model
        self.W_K = ConstrainedRead(
            d_in,
            d_head,
            n_heads,
            n_vars=n_vars,
            temp=temp,
            sample_fn=sample_fn,
        )
        self.W_Q = ConstrainedRead(
            d_in,
            d_head,
            n_heads,
            n_vars=n_vars,
            temp=temp,
            sample_fn=sample_fn,
        )
        self.W_pred = WScore(n_heads, d_head, temp=temp)
        self.W_V = ConstrainedRead(
            d_in,
            d_head,
            n_heads,
            n_vars=n_vars,
            temp=temp,
            sample_fn=sample_fn,
        )
        self.register_buffer("mask", torch.tril(torch.ones((n_ctx, n_ctx))))
        self.d_head = d_head
        self.n_heads = n_heads
        self.hook_k = HookPoint()
        self.hook_q = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn = HookPoint()
        self.hook_attn_pre = HookPoint()
        self.register_buffer(
            "pos_bias", torch.arange(start=n_ctx, end=0, step=-1)
        )
        self.temp = temp
        self.sample_fn = sample_fn
        self.attention_temp = attention_temp or np.sqrt(self.d_head)

    def set_temp(self, temp, sample_fn=None):
        if sample_fn is not None:
            self.sample_fn = sample_fn
        self.temp = temp
        for module in (self.W_Q, self.W_K, self.W_V, self.W_pred):
            module.set_temp(temp, sample_fn=sample_fn)

    def forward(self, x, mask=None, **kwargs):
        k = self.hook_k(torch.einsum("ihd,bpd->biph", self.W_K(), x))
        q = self.hook_q(
            torch.einsum("ihd,bpd->biph", self.W_pred(self.W_Q()), x)
        )
        v = self.hook_v(torch.einsum("ihd,bpd->biph", self.W_V(), x))
        attn_scores_pre = torch.einsum("biph,biqh->biqp", k, q)
        if mask is not None:
            attn_scores_masked = attn_scores_pre.masked_fill(
                (~mask).unsqueeze(1), -1e10
            )
        else:
            attn_scores_masked = torch.tril(attn_scores_pre) - 1e10 * (
                1 - self.mask[: x.shape[-2], : x.shape[-2]]
            )
        attn_matrix = self.hook_attn(
            self.sample_fn(
                self.hook_attn_pre(attn_scores_masked) / self.attention_temp,
                tau=self.temp,
                dim=-1,
            )
        )
        z = self.hook_z(torch.einsum("biph,biqp->biqh", v, attn_matrix))
        z_flat = einops.rearrange(z, "b i q h -> b q (i h)")
        out = z_flat
        return out


class ClippedRelPosBias(nn.Module):
    def __init__(self, n_heads, max_ctx, one_hot_input=False):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(n_heads, 2 * max_ctx))
        self.register_buffer("idxs", torch.arange(2 * max_ctx))
        self.one_hot_input = one_hot_input
        self.max_ctx = max_ctx

    def get_bias(self, nq=None, nk=None):
        if nq is None:
            nq = self.max_ctx
        if nk is None:
            nk = self.max_ctx
        idxs = self.idxs[:nk].unsqueeze(0) - self.idxs[:nq].unsqueeze(-1)
        bias = self.bias
        bias = F.threshold(-F.threshold(-bias, -10.0, -10.0), -10.0, -10.0)
        return bias[:, idxs]

    def forward(self, scores):
        nq, nk = scores.shape[-2:]
        if nq > self.max_ctx or nk > self.max_ctx:
            extended_bias = torch.zeros(
                self.bias.size(0),
                nq, nk,
                device=self.bias.device
            )
            
            orig_nq = min(nq, self.max_ctx)
            orig_nk = min(nk, self.max_ctx)
            orig_bias = self.get_bias(nq=orig_nq, nk=orig_nk)
            
            extended_bias[:, :orig_nq, :orig_nk] = orig_bias
            
            if self.one_hot_input:
                return (scores + 1e-20).log() + extended_bias.unsqueeze(0)
            return scores + extended_bias.unsqueeze(0)
        else:
            bias = self.get_bias(nq=nq, nk=nk)
            if self.one_hot_input:
                return (scores + 1e-20).log() + bias.unsqueeze(0)
            return scores + bias.unsqueeze(0)


class FixedRelPosBias(nn.Module):
    def __init__(self, n_heads, max_ctx, one_hot_input=False):
        super().__init__()
        self.register_buffer(
            "bias",
            (
                torch.cat(
                    [
                        torch.tensor([-2]),
                        torch.linspace(0.0, -max_ctx, max_ctx - 1) / max_ctx,
                        torch.linspace(-max_ctx, 0.0, max_ctx) / max_ctx,
                    ],
                    -1,
                )
            )
            .unsqueeze(0)
            .repeat(n_heads, 1),
        )
        self.register_buffer("idxs", torch.arange(2 * max_ctx))
        self.one_hot_input = one_hot_input
        self.max_ctx = max_ctx

    def get_bias(self, nq=None, nk=None):
        if nq is None:
            nq = self.max_ctx
        if nk is None:
            nk = self.max_ctx
        idxs = self.idxs[:nk].unsqueeze(0) - self.idxs[:nq].unsqueeze(-1)
        bias = self.bias
        return bias[:, idxs]

    def forward(self, scores):
        nq, nk = scores.shape[-2:]
        if nq > self.max_ctx or nk > self.max_ctx:
            extended_bias = torch.zeros(
                self.bias.size(0),
                nq, nk,
                device=self.bias.device
            )
            
            orig_nq = min(nq, self.max_ctx)
            orig_nk = min(nk, self.max_ctx)
            orig_bias = self.get_bias(nq=orig_nq, nk=orig_nk)
            
            extended_bias[:, :orig_nq, :orig_nk] = orig_bias
            
            if self.one_hot_input:
                return (scores + 1e-20).log() + extended_bias.unsqueeze(0)
            return scores + extended_bias.unsqueeze(0)
        else:
            bias = self.get_bias(nq=nq, nk=nk)
            if self.one_hot_input:
                return (scores + 1e-20).log() + bias.unsqueeze(0)
            return scores + bias.unsqueeze(0)


class NoRelPosBias(nn.Module):
    def __init__(self, n_heads, max_ctx, one_hot_input=False):
        super().__init__()
        self.register_buffer("bias", torch.zeros(n_heads, 2 * max_ctx))
        self.register_buffer("idxs", torch.arange(2 * max_ctx))
        self.one_hot_input = one_hot_input
        self.max_ctx = max_ctx

    def get_bias(self, nq=None, nk=None):
        if nq is None:
            nq = self.max_ctx
        if nk is None:
            nk = self.max_ctx
        idxs = self.idxs[:nk].unsqueeze(0) - self.idxs[:nq].unsqueeze(-1)
        bias = self.bias
        return bias[:, idxs].softmax(-1)

    def forward(self, scores):
        nq, nk = scores.shape[-2:]
        if nq > self.max_ctx or nk > self.max_ctx:
            extended_bias = torch.zeros(
                self.bias.size(0),
                nq, nk,
                device=self.bias.device
            )
            
            orig_nq = min(nq, self.max_ctx)
            orig_nk = min(nk, self.max_ctx)
            orig_bias = self.get_bias(nq=orig_nq, nk=orig_nk)
            
            extended_bias[:, :orig_nq, :orig_nk] = orig_bias
            
            if self.one_hot_input:
                return (scores + 1e-20).log() + extended_bias.unsqueeze(0)
            return scores + extended_bias.unsqueeze(0)
        else:
            bias = self.get_bias(nq=nq, nk=nk)
            if self.one_hot_input:
                return (scores + 1e-20).log() + bias.unsqueeze(0)
            return scores + bias.unsqueeze(0)


class CatAttention(nn.Module):
    def __init__(
        self,
        d_in,
        n_heads,
        d_head,
        n_ctx,
        model,
        temp=1.0,
        n_vars=None,
        sample_fn=softmax,
        attention_temp=None,
        rel_pos_bias="fixed",
        **kwargs,
    ):
        super().__init__()
        d_out = d_head * n_heads
        self.model = model
        self.W_K = ConstrainedRead(
            d_in,
            d_head,
            n_heads,
            n_vars=n_vars,
            temp=temp,
            sample_fn=sample_fn,
        )
        self.W_Q = ConstrainedRead(
            d_in,
            d_head,
            n_heads,
            n_vars=n_vars,
            temp=temp,
            sample_fn=sample_fn,
        )
        self.W_pred = WPred(n_heads, d_head, temp=temp, sample_fn=sample_fn)
        self.W_V = ConstrainedRead(
            d_in,
            d_head,
            n_heads,
            n_vars=n_vars,
            temp=temp,
            sample_fn=sample_fn,
        )
        rel_pos_types = {
            "fixed": FixedRelPosBias,
            "clipped": ClippedRelPosBias,
            "none": NoRelPosBias,
        }
        if rel_pos_bias not in rel_pos_types:
            raise NotImplementedError(rel_pos_bias)
        self.rel_pos_bias = rel_pos_types[rel_pos_bias](
            n_heads, n_ctx, one_hot_input=True
        )
        self.register_buffer("mask", torch.tril(torch.ones((n_ctx, n_ctx))))
        self.d_head = d_head
        self.n_heads = n_heads
        self.hook_k = HookPoint()
        self.hook_q = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn = HookPoint()
        self.hook_attn_pre = HookPoint()
        self.hook_attn_pos = HookPoint()
        self.register_buffer(
            "pos_bias", torch.arange(start=n_ctx, end=0, step=-1)
        )
        self.temp = temp
        self.sample_fn = sample_fn
        self.attention_temp = attention_temp or np.sqrt(self.d_head)

    def set_temp(self, temp, sample_fn=None):
        if sample_fn is not None:
            self.sample_fn = sample_fn
        self.temp = temp
        for module in (self.W_Q, self.W_K, self.W_V, self.W_pred):
            module.set_temp(temp, sample_fn=sample_fn)

    def forward(self, x, mask=None, **kwargs):
        k = self.hook_k(torch.einsum("ihd,bpd->biph", self.W_K(), x))
        q = self.hook_q(
            torch.einsum("ihd,bpd->biph", self.W_pred(self.W_Q()), x)
        )
        v = self.hook_v(torch.einsum("ihd,bpd->biph", self.W_V(), x))
        attn_scores_pre = torch.einsum("biph,biqh->biqp", k, q)
        
        seq_len = attn_scores_pre.size(-1)
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        ).unsqueeze(0).unsqueeze(1)
        
        attn_scores_masked = attn_scores_pre.masked_fill(~causal_mask, -1e30)
        
        attn_scores_masked = self.hook_attn_pre(attn_scores_masked)
        attn_scores_pos = self.hook_attn_pos(attn_scores_masked)
        
        attn_matrix = self.hook_attn(
            self.sample_fn(
                attn_scores_pos / self.attention_temp,
                tau=self.temp,
                dim=-1,
            )
        )
        
        z = self.hook_z(torch.einsum("biph,biqp->biqh", v, attn_matrix))
        z_flat = einops.rearrange(z, "b i q h -> b q (i h)")
        
        return z_flat

class NumAttention(nn.Module):
    def __init__(
        self,
        d_in_cat,
        d_in_num,
        n_heads,
        d_head,
        n_ctx,
        model,
        temp=1.0,
        sample_fn=softmax,
        selector_width=False,
        **kwargs,
    ):
        super().__init__()
        self.temp = temp
        d_out = d_head * n_heads
        self.model = model
        # n_heads, d_head, d_model
        self.W_K = ConstrainedRead(
            d_in_cat,
            d_head,
            n_heads,
            n_vars=d_in_cat // d_head,
            temp=temp,
            sample_fn=sample_fn,
        )
        self.W_Q = ConstrainedRead(
            d_in_cat,
            d_head,
            n_heads,
            n_vars=d_in_cat // d_head,
            temp=temp,
            sample_fn=sample_fn,
        )
        self.W_pred = WPred(n_heads, d_head, temp=temp, sample_fn=sample_fn)
        self.W_V = ConstrainedRead(
            d_in_num,
            1,
            n_heads=n_heads,
            n_vars=d_in_num,
            temp=temp,
            sample_fn=sample_fn,
        )
        self.selector_width = selector_width
        if selector_width:
            self.W_V.fix_read(0)

        self.register_buffer("mask", torch.tril(torch.ones((n_ctx, n_ctx))))
        self.d_head = d_head
        self.n_heads = n_heads
        self.hook_k = HookPoint()
        self.hook_q = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn = HookPoint()
        self.hook_attn_pre = HookPoint()
        self.sample_fn = sample_fn

    def set_temp(self, temp, sample_fn=None):
        if sample_fn is not None:
            self.sample_fn = sample_fn
        for module in (self.W_Q, self.W_K, self.W_V, self.W_pred):
            module.set_temp(temp, sample_fn=sample_fn)

    def forward(self, x_cat, x_num, mask=None, **kwargs):
        k = self.hook_k(torch.einsum("ihd,bpd->biph", self.W_K(), x_cat))
        q = self.hook_q(
            torch.einsum("ihd,bpd->biph", self.W_pred(self.W_Q()), x_cat)
        )
        v = self.hook_v(torch.einsum("ihd,bpd->biph", self.W_V(), x_num))
        
        B, I, P, H = k.shape
        _, _, Q, _ = q.shape
        
        attn_scores_pre = torch.einsum("biph,biqh->biqp", k, q)
        
        causal_mask = torch.ones(Q, P, device=x_cat.device, dtype=torch.bool).triu(diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
        
        attn_scores_masked = attn_scores_pre.masked_fill(causal_mask, -1e30)
        attn_scores_masked = self.hook_attn_pre(attn_scores_masked)
        
        attn_matrix = self.hook_attn(self.sample_fn(attn_scores_masked, dim=-1))
        
        if attn_matrix.shape[-1] != v.shape[-2]:
            P_v = v.shape[-2]
            if attn_matrix.shape[-1] > P_v:
                attn_matrix = attn_matrix[..., :P_v]
            else:
                padding = torch.zeros(
                    B, I, Q, P_v - attn_matrix.shape[-1],
                    device=attn_matrix.device
                )
                attn_matrix = torch.cat([attn_matrix, padding], dim=-1)
        
        z = self.hook_z(torch.einsum("biph,biqp->biqh", v, attn_matrix))
        z_flat = einops.rearrange(z, "b i q h -> b q (i h)")
        
        return z_flat


# MLP Layers
class CatMLP(nn.Module):
    def __init__(
        self,
        d_in,
        d_mlp,
        d_out,
        act_type,
        model,
        temp=1.0,
        sample_fn=softmax_no_temp,
        mlp_vars_in=2,
        n_vars=None,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        # Read `mlp_vars_in` variables. d_out is variable dimension.
        self.W_read = ConstrainedRead(
            d_in=d_in,
            d_var=d_out,
            n_heads=mlp_vars_in,
            n_vars=n_vars,
            temp=temp,
            sample_fn=sample_fn,
        )
        self.d_in = d_in
        self.d_out = d_out
        self.dim_out = dim_out = d_out
        self.mlp_vars_in = mlp_vars_in
        d_var_in = (d_in // n_vars) if n_vars is not None else d_out
        self.W_in = nn.Parameter(
            torch.randn(d_mlp, d_var_in * mlp_vars_in) / np.sqrt(d_mlp)
        )
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(dim_out, d_mlp) / np.sqrt(d_out))
        self.b_out = nn.Parameter(torch.zeros(dim_out))
        self.act_type = act_type
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()
        assert act_type in ["ReLU", "GeLU"]
        self.temp = temp
        self.sample_fn = sample_fn
        self.dropout = nn.Dropout(dropout)

    @property
    def device(self):
        return self.W_out.device

    def forward(self, x, **kwargs):
        W_read = self.W_read().reshape(-1, self.d_in)
        x = torch.einsum("md,bpd->bpm", W_read, x)
        x = self.hook_pre(torch.einsum("md,bpd->bpm", self.W_in, x) + self.b_in)
        act = F.relu if self.act_type == "ReLU" else F.gelu
        x = self.dropout(act(x))
        x = self.hook_post(x)
        x = torch.einsum("dm,bpm->bpd", self.W_out, x) + self.b_out
        x = x.log_softmax(-1)
        sample_fn = self.sample_fn
        return sample_fn(x, dim=-1, tau=self.temp)

    def set_temp(self, temp, sample_fn=None):
        if sample_fn is not None:
            self.sample_fn = sample_fn
        self.temp = temp
        for module in (self.W_read,):
            module.set_temp(temp, sample_fn=sample_fn)


class CatMLPs(nn.Module):
    def __init__(self, n_mlps, **kwargs):
        super().__init__()
        self.mlps = nn.ModuleList([CatMLP(**kwargs) for _ in range(n_mlps)])
        self.n_mlps = n_mlps

    @property
    def device(self):
        return self.mlps[0].W_out.device

    def forward(self, x, **kwargs):
        out = []
        for mlp in self.mlps:
            out.append(mlp(x, **kwargs))
        return torch.cat(out, -1)

    def set_temp(self, temp, sample_fn=None):
        for mlp in self.mlps:
            mlp.set_temp(temp, sample_fn=sample_fn)


class TransformerProgramBlock(nn.Module):
    def __init__(
        self,
        d_in_cat,
        d_in_num,
        d_mlp,
        d_head,
        n_heads_num,
        n_heads_cat,
        n_ctx,
        act_type,
        model,
        temp=1.0,
        n_cat_mlps=1,
        n_num_mlps=1,
        attention_type="cat",
        rel_pos_bias="fixed",
        mlp_vars_in=2,
        selector_width=True,
        sample_fn=softmax,
        attention_temp=None,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.sample_fn = sample_fn
        attns = {"cat": CatAttention, "score": ScoreAttention}
        self.cat_attn = (attns.get(attention_type, CatAttention))(
            d_in=d_in_cat,
            n_heads=n_heads_cat,
            d_head=d_head,
            n_ctx=n_ctx,
            model=self.model,
            temp=temp,
            sample_fn=sample_fn,
            attention_temp=attention_temp,
            rel_pos_bias=rel_pos_bias,
        )
        self.num_attn = None
        if n_heads_num:
            self.num_attn = NumAttention(
                d_in_cat=d_in_cat,
                d_in_num=d_in_num,
                n_heads=n_heads_num,
                d_head=d_head,
                n_ctx=n_ctx,
                model=self.model,
                temp=temp,
                sample_fn=sample_fn,
                attention_temp=attention_temp,
                selector_width=selector_width,
            )
        self.n_heads_cat, self.n_heads_num = n_heads_cat, n_heads_num
        self.cat_mlp = CatMLPs(
            n_mlps=n_cat_mlps,
            d_in=d_in_cat + d_head * n_heads_cat,
            d_mlp=d_mlp,
            d_out=d_head,
            act_type=act_type,
            model=self.model,
            sample_fn=sample_fn,
            mlp_vars_in=mlp_vars_in,
            dropout=dropout,
        )
        self.num_mlp = CatMLPs(
            n_mlps=n_num_mlps,
            d_in=d_in_num + n_heads_num,
            d_mlp=d_mlp,
            d_out=d_head,
            n_vars=d_in_num + n_heads_num,
            act_type=act_type,
            model=self.model,
            sample_fn=sample_fn,
            mlp_vars_in=mlp_vars_in,
            dropout=dropout,
        )
        self.hook_attn_out_cat = HookPoint()
        self.hook_attn_out_num = HookPoint()
        self.hook_cat_mlp_out = HookPoint()
        self.hook_num_mlp_out = HookPoint()
        self.hook_resid_pre_cat = HookPoint()
        self.hook_resid_pre_num = HookPoint()
        self.hook_resid_mid_cat = HookPoint()
        self.hook_resid_mid_num = HookPoint()
        self.hook_resid_post_cat = HookPoint()
        self.hook_resid_post_num = HookPoint()
        self.n_cat_mlps = n_cat_mlps
        self.n_num_mlps = n_num_mlps
        self.sample_fn = sample_fn
        self.dropout = nn.Dropout(dropout)

    def set_temp(self, temp, sample_fn=None):
        if sample_fn is not None:
            self.sample_fn = sample_fn
        for module in (
            self.cat_attn,
            self.num_attn,
            self.cat_mlp,
            self.num_mlp,
        ):
            if module is not None:
                module.set_temp(temp, sample_fn=sample_fn)

    def forward(self, x_cat, x_num, mask=None, **kwargs):
        x_cat = self.hook_resid_pre_cat(x_cat)
        x_num = self.hook_resid_pre_num(x_num)
        drop = self.dropout

        if self.n_heads_cat:
            attn_out_cat = self.hook_attn_out_cat(
                self.cat_attn(x_cat, mask=mask)
            )
        if self.n_heads_num:
            attn_out_num = self.hook_attn_out_num(
                self.num_attn(x_cat, x_num, mask=mask)
            )
        if self.n_heads_cat:
            x_cat = self.hook_resid_mid_cat(
                torch.cat([x_cat, drop(attn_out_cat)], -1)
            )
        if self.n_heads_num:
            x_num = self.hook_resid_mid_num(
                torch.cat([x_num, drop(attn_out_num)], -1)
            )
        if self.n_cat_mlps:
            cat_mlp_out = self.hook_cat_mlp_out(self.cat_mlp(x_cat))
            x_cat = torch.cat([x_cat, drop(cat_mlp_out)], -1)
        if self.n_num_mlps:
            num_mlp_out = self.hook_num_mlp_out(self.num_mlp(x_num))
            x_cat = torch.cat([x_cat, drop(num_mlp_out)], -1)
        x_cat = self.hook_resid_post_cat(x_cat)
        x_num = self.hook_resid_post_num(x_num)
        return x_cat, x_num


class TransformerProgramModel(nn.Module):
    def __init__(
        self,
        d_vocab,
        n_layers=2,
        n_vars=2,
        n_vars_cat=None,
        n_vars_num=None,
        d_var=4,
        d_mlp=8,
        n_heads=2,
        n_heads_cat=None,
        n_heads_num=None,
        n_ctx=5,
        d_pos=None,
        act_type="ReLU",
        use_cache=False,
        use_ln=False,
        d_vocab_out=None,
        temp=1.0,
        init_emb=None,
        n_cat_mlps=True,
        n_num_mlps=True,
        attention_type="cat",
        rel_pos_bias="fixed",
        mlp_vars_in=2,
        sample_fn=softmax,
        attention_temp=None,
        dropout=0.0,
        unembed_num=True,
        unembed_mask=None,
        pool_outputs=False,
        one_hot_embed=False,
        count_only=False,
        selector_width=False,
        improvements=None,
        **kwargs,
    ):
        super().__init__()
        improvements = improvements or {}
        self.use_prefix = "prefixsum" in improvements
        self.use_sketch = "sketch" in improvements
        self.use_chunks = "chunks" in improvements
        self.use_experts = "experts" in improvements
        self.use_contrast = "contrast" in improvements
        self.use_mem = "memory" in improvements

        self.cache = {}
        self.use_cache = use_cache
        self.d_pos = d_pos or d_var
        self.d_var = d_var

        self.d_head = d_head = d_var

        if n_vars_cat is None:
            n_vars_cat = n_vars
        if n_vars_num is None:
            n_vars_num = n_vars

        self.n_vars_cat, self.n_vars_num = n_vars_cat, n_vars_num

        self.d_model = d_model = d_var * n_vars_cat
        self.n_layers = n_layers

        if self.use_sketch:
            self.hash_embed = HashSketchEmbed(d_vocab)
            self.embed = CatEmbed(
                self.hash_embed.kp,
                n_vars=n_vars_cat,
                d_var=d_var,
                temp=temp,
                sample_fn=sample_fn,
                one_hot_embed=one_hot_embed,
            )
        else:
            self.embed = CatEmbed(
                d_vocab,
                n_vars=n_vars_cat,
                d_var=d_var,
                temp=temp,
                init_emb=init_emb,
                sample_fn=sample_fn,
                one_hot_embed=one_hot_embed,
            )
        self.num_embed = NumEmbed(
            d_vocab,
            n_vars=n_vars_num,
            count_only=count_only,
        )
        if self.use_prefix:
            self.prefix_layer = PrefixSumCounts(d_vocab)
        if self.use_chunks:
            self.chunker = ChunkAggregator(block_size=8)
        self.pos_embed = OneHotPosEmbed(
            n_ctx,
            d_var,
            temp=temp,
        )
        self.d_pos = d_pos = d_var

        if n_heads_cat is None:
            n_heads_cat = n_heads
        if n_heads_num is None:
            n_heads_num = n_heads
        self.n_heads_cat, self.n_heads_num = n_heads_cat, n_heads_num
        extra_cat = 0
        extra_num = 0
        if self.use_contrast:
            self.contrast_layer = ContrastiveTokenRepresentations(d_vocab)
            extra_cat += self.contrast_layer.n_buckets
        if self.use_experts:
            expert_in = d_model + extra_cat
            self.expert_layer = SparseExpertCountingNetwork(expert_in, n_experts=4)
            extra_num += 1
        if self.use_prefix:
            extra_num += 1
        if self.use_mem:
            self.mem_net = PositionalNgramMemoryNetwork(d_model + extra_cat)
        layer_out_cat = d_var * n_heads_cat + (
            d_head * (n_cat_mlps + n_num_mlps)
        )
        layer_out_num = n_heads_num
        total_width = (d_model + d_pos + extra_cat + n_layers * layer_out_cat + 
                       (n_vars_num * int(unembed_num)) + extra_num + n_layers * (layer_out_num * int(unembed_num)))
        self.blocks = nn.ModuleList(
            [
                TransformerProgramBlock(
                    d_in_cat=d_model + d_pos + i * layer_out_cat + extra_cat,
                    d_in_num=n_vars_num + i * layer_out_num + extra_num,
                    d_mlp=d_mlp,
                    d_head=d_head,
                    n_heads_num=n_heads_num,
                    n_heads_cat=n_heads_cat,
                    n_ctx=n_ctx,
                    act_type=act_type,
                    model=[self],
                    temp=temp,
                    n_cat_mlps=n_cat_mlps,
                    n_num_mlps=n_num_mlps,
                    attention_type=attention_type,
                    rel_pos_bias=rel_pos_bias,
                    mlp_vars_in=mlp_vars_in,
                    selector_width=selector_width,
                    sample_fn=sample_fn,
                    attention_temp=attention_temp,
                    dropout=dropout,
                )
                for i in range(n_layers)
            ]
        )
        self.n_cat_mlps, self.n_num_mlps = n_cat_mlps, n_num_mlps
        self.dropout = nn.Dropout(dropout)
        self.unembed = Unembed(
            d_vocab_out or d_vocab,
            total_width,
            mask=unembed_mask,
        )

        for name, module in self.named_modules():
            if type(module) == HookPoint:
                module.give_name(name)

        self.unembed_num = unembed_num
        self.pool_outputs = pool_outputs
        self.hook_final = HookPoint()
        self.hook_final.name = "hook_final"
        self.hook_pool = HookPoint()
        self.hook_pool.name = "hook_pool"

    def set_temp(self, temp, sample_fn=None):
        if sample_fn is not None:
            self.sample_fn = sample_fn
        for module in [self.embed, self.pos_embed] + [b for b in self.blocks]:
            if module is not None:
                module.set_temp(temp, sample_fn=sample_fn)

    def forward(self, x, mask=None, **kwargs):
        B, L = x.size()
        if mask is None:
            mask = torch.ones(B, L, L, device=x.device, dtype=torch.bool)
            
        x_hashed = self.hash_embed(x) if self.use_sketch else x

        original_seq_length = L
        num_chunk_tokens = 0

        if self.use_chunks:
            x_hashed, chunk_cat_ids, _ = self.chunker(
                x_hashed,
                cat_embed_f=self.embed,
                num_embed_f=self.num_embed,
            )

            chunk_cat = self.embed(chunk_cat_ids)
            num_chunk_tokens = chunk_cat.size(1)

            Bk = chunk_cat.size(1)
            top = torch.ones(B, Bk, Bk + L, device=mask.device, dtype=torch.bool)
            bottom = torch.cat(
                [torch.ones(B, L, Bk, device=mask.device, dtype=torch.bool), mask], 2
            )
            mask = torch.cat([top, bottom], 1)
        else:
            chunk_cat = None

        contrast_cat = None
        if self.use_contrast:
            tokens_flat = x_hashed.reshape(x_hashed.size(0), -1) if x_hashed.dim() == 3 else x_hashed
            if self.use_chunks:
                chunk_ids_flat = chunk_cat_ids.view(chunk_cat_ids.size(0), -1)
                tokens_for_contrast = torch.cat([chunk_ids_flat, tokens_flat], dim=1)
            else:
                tokens_for_contrast = tokens_flat
            one_hot = F.one_hot(
                tokens_for_contrast,
                num_classes=self.contrast_layer.prototypes.size(1)
            )
            contrast_cat = self.contrast_layer(one_hot.float())

        x_cat = self.embed(x_hashed)
        if chunk_cat is not None:
            x_cat = torch.cat([chunk_cat, x_cat], dim=1)
        if contrast_cat is not None:
            x_cat = torch.cat([contrast_cat, x_cat], dim=-1)

        x_num = self.num_embed(x)
        if self.use_prefix:
            x_num = torch.cat([x_num, self.prefix_layer(x)], dim=-1)
        if chunk_cat is not None:
            zeros_num = torch.zeros(B, Bk, x_num.size(-1), device=x_num.device)
            x_num = torch.cat([zeros_num, x_num], 1)

        if self.use_mem:
            x_cat = x_cat + self.mem_net(x_cat)
        
        if self.use_experts:
            expert_feat = self.expert_layer(x_cat)
            if expert_feat.size(-1) != 1:
                expert_feat = expert_feat.mean(-1, keepdim=True)
            if expert_feat.size(1) != x_num.size(1):
                fixed_expert = torch.zeros(
                    expert_feat.size(0), 
                    x_num.size(1),
                    expert_feat.size(2),
                    device=expert_feat.device
                )
                min_seq_len = min(expert_feat.size(1), x_num.size(1))
                fixed_expert[:, :min_seq_len] = expert_feat[:, :min_seq_len]
                expert_feat = fixed_expert
            x_num = torch.cat([x_num, expert_feat], dim=-1)

        if self.pos_embed is not None:
            x_cat = torch.cat([x_cat, self.pos_embed(x_cat)], dim=-1)

        for block in self.blocks:
            x_cat, x_num = block(x_cat, x_num, mask=mask)

        x_out = torch.cat([x_cat, x_num], dim=-1) if self.unembed_num else x_cat
        x_out = self.hook_final(x_out)

        if self.pool_outputs:
            pooled = x_out.masked_fill(~mask[:, 0].unsqueeze(-1), 0).mean(1, keepdims=True)
            x_out = torch.cat([pooled, x_out[:, 1:]], dim=1)
            x_out = self.hook_pool(x_out)

        return self.unembed(x_out)

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def hook_points(self):
        return [
            module for name, module in self.named_modules() if "hook" in name
        ]

    def remove_all_hooks(self):
        for hp in self.hook_points():
            hp.remove_hooks("fwd")
            hp.remove_hooks("bwd")

    def cache_all(self, cache, incl_bwd=False):
        # Caches all activations wrapped in a HookPoint
        def save_hook(tensor, name):
            cache[name] = tensor.detach()

        def save_hook_back(tensor, name):
            cache[name + "_grad"] = tensor[0].detach()

        for hp in self.hook_points():
            hp.add_hook(save_hook, "fwd")
            if incl_bwd:
                hp.add_hook(save_hook_back, "bwd")

    @property
    def device(self):
        return self.embed.device
