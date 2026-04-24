"""
DracoAI V1 — PyTorch Transformer (for training)
================================================
Qwen 3.5 9B Instruct — NO BIAS throughout.

Used by trainer_v1.py for GPU training.
After training, weights are synced back to NumPy model via _full_sync().

MoE source: SINGLE Qwen 3.5 9B Instruct checkpoint.
    No separate "coder" or "instruct" split model.
    layer_idx % 8 → expert slot:
        0-3: Code group  (FFN activations skew toward code/math tokens)
        4-7: Language group (FFN activations skew toward language/instruction tokens)

ALL FIXES:
    ✅ Qwen 3.5 9B Instruct (was 7B — typo corrected)
    ✅ n_layers=36 in default config (9B has 36 layers)
    ✅ Expert naming: Code group (0-3) / Language group (4-7)
    ✅ bias=False on ALL nn.Linear layers (Qwen 3.5 compatible)
    ✅ RMSNorm eps=1e-6
    ✅ rope_theta from config (not hardcoded)
    ✅ 8-expert MoE with router temperature + fallback to least-loaded
    ✅ Per-expert RMSNorm (scale only, no bias)
    ✅ MoE symmetry-breaking routing_bias (additive signal, not Linear bias)
    ✅ Flash Attention (F.scaled_dot_product_attention)
    ✅ ignore_index = -100 consistent
    ✅ MTP t+2 masked loss correct
    ✅ GQA repeat_interleave correct
    ✅ torch.compile compatible
    ✅ _full_sync: ALL weights synced including expert_norm, router_bias
    ✅ print statement base model corrected to Qwen 3.5 9B Instruct
"""
# Copyright 2026 The Draco Studio and DUCNGUYEN-creator
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from transformer_v1 import DracoConfig

# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════
def _rms(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm — NO bias, NO mean subtraction (Qwen 3.5 standard)."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

# ══════════════════════════════════════════════════════════════════════
# RoPE
# ══════════════════════════════════════════════════════════════════════
class RoPETorch(nn.Module):
    def __init__(self, head_dim: int, max_seq: int, theta: float = 1_000_000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t     = torch.arange(max_seq).float()
        mat   = torch.outer(t, freqs)
        emb   = torch.cat([mat, mat], dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        seq  = x.shape[2]
        cos  = self.cos[offset:offset + seq].unsqueeze(0).unsqueeze(0)
        sin  = self.sin[offset:offset + seq].unsqueeze(0).unsqueeze(0)
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos[..., :half] - x2 * sin[..., :half],
                          x1 * sin[..., half:] + x2 * cos[..., half:]], dim=-1)

# ══════════════════════════════════════════════════════════════════════
# GQA — NO BIAS
# ══════════════════════════════════════════════════════════════════════
class GQATorch(nn.Module):
    def __init__(self, cfg: DracoConfig):
        super().__init__()
        d, hq, hkv, hd = cfg.d_model, cfg.n_heads_q, cfg.n_heads_kv, cfg.head_dim
        self.hq  = hq; self.hkv = hkv; self.hd = hd; self.rep = cfg.gqa_repeat
        # bias=False — Qwen 3.5 9B has no attention bias
        self.wq  = nn.Linear(d,       hq * hd,  bias=False)
        self.wk  = nn.Linear(d,       hkv * hd, bias=False)
        self.wv  = nn.Linear(d,       hkv * hd, bias=False)
        self.wo  = nn.Linear(hq * hd, d,         bias=False)
        self.rope = RoPETorch(hd, cfg.context_len, cfg.rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        hq, hkv, hd = self.hq, self.hkv, self.hd
        Q = self.wq(x).view(B, T, hq,  hd).transpose(1, 2)
        K = self.wk(x).view(B, T, hkv, hd).transpose(1, 2)
        V = self.wv(x).view(B, T, hkv, hd).transpose(1, 2)
        Q = self.rope(Q); K = self.rope(K)
        K = K.repeat_interleave(self.rep, dim=1)
        V = V.repeat_interleave(self.rep, dim=1)
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, hq * hd)
        return self.wo(out)  # bias=False

# ══════════════════════════════════════════════════════════════════════
# SwiGLU — NO BIAS
# ══════════════════════════════════════════════════════════════════════
class SwiGLUTorch(nn.Module):
    def __init__(self, d_in: int, d_ff: int):
        super().__init__()
        # bias=False — Qwen 3.5 9B gate_proj/up_proj/down_proj all no bias
        self.gate_proj = nn.Linear(d_in, d_ff, bias=False)
        self.up_proj   = nn.Linear(d_in, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_in, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

# ══════════════════════════════════════════════════════════════════════
# MoE — 8 experts from single Qwen 3.5 9B Instruct
# ══════════════════════════════════════════════════════════════════════
class MoETorch(nn.Module):
    """
    8 experts, all from ONE Qwen 3.5 9B Instruct checkpoint.
    No separate "coder" or "instruct" model split.

    Expert assignment via layer_idx % 8:
        Code group (0-3):     FFN from layers whose idx % 8 in {0,1,2,3}
        Language group (4-7): FFN from layers whose idx % 8 in {4,5,6,7}

    Router temperature prevents expert collapse.
    Fallback: tokens with zero output → least-loaded expert.
    """
    def __init__(self, cfg: DracoConfig):
        super().__init__()
        d = cfg.d_model
        self.n_exp      = cfg.n_experts          # 8
        self.top_k      = cfg.n_experts_top      # 2
        self.cap        = cfg.moe_capacity
        self.dep_thresh = cfg.moe_dep_threshold
        self.router_t   = cfg.moe_router_temp    # temperature

        # Router: NO BIAS (bias=False)
        self.W_router   = nn.Linear(d, cfg.n_experts, bias=False)

        # Routing bias as a learnable parameter (NOT a Linear bias — symmetry breaking)
        self.router_bias = nn.Parameter(
            torch.tensor([0.02 * (i - cfg.n_experts / 2.0)
                          for i in range(cfg.n_experts)], dtype=torch.float32)
        )

        # Expert FFN: full d_ff for Qwen 3.5 9B checkpoint compatibility
        d_exp = cfg.d_ff
        self.experts = nn.ModuleList([SwiGLUTorch(d, d_exp) for _ in range(cfg.n_experts)])
        self.shared  = SwiGLUTorch(d, max(cfg.d_ff // 4, 32))

        # Per-expert RMSNorm (scale only, no bias)
        self.expert_norm = nn.ParameterList([
            nn.Parameter(torch.ones(d)) for _ in range(cfg.n_experts)
        ])

    def forward(
        self,
        x: torch.Tensor,
        intent_boost: Optional[Dict[int, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, d = x.shape
        x_flat  = x.reshape(B * T, d)
        S       = x_flat.shape[0]

        # Router with temperature — NO BIAS on W_router
        logits = self.W_router(x_flat) / max(self.router_t, 1e-4)
        logits = logits + self.router_bias.unsqueeze(0)  # symmetry breaking

        if intent_boost:
            for eid, boost in intent_boost.items():
                if 0 <= eid < self.n_exp:
                    logits[:, eid] += boost

        probs     = torch.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values
        top_vals, top_idx = torch.topk(probs, self.top_k, dim=-1)
        gates = top_vals / (top_vals.sum(dim=-1, keepdim=True) + 1e-8)

        capacity = max(1, int(S * self.cap / self.n_exp))
        output   = torch.zeros_like(x_flat)
        tok_load = torch.zeros(self.n_exp, device=x.device, dtype=torch.float32)

        for k in range(self.top_k):
            exp_ids = top_idx[:, k]
            g       = gates[:, k]
            for ei in range(self.n_exp):
                mask = (exp_ids == ei).nonzero(as_tuple=True)[0]
                if len(mask) == 0:
                    continue
                # DEP: high-confidence tokens skip extra experts
                if k > 0:
                    dep_mask = max_probs[mask] > self.dep_thresh
                    mask     = mask[~dep_mask] if dep_mask.any() else mask
                if len(mask) == 0:
                    continue
                # Capacity check
                avail = capacity - int(tok_load[ei].item())
                if avail <= 0:
                    continue
                mask = mask[:avail]
                tok_load[ei] += len(mask)
                # Per-expert RMSNorm
                x_norm = _rms(x_flat[mask], self.expert_norm[ei])
                output[mask] += g[mask].unsqueeze(-1) * self.experts[ei](x_norm)

        # Fallback: any tokens with zero output get least-loaded expert
        zero_mask = (output.abs().sum(dim=-1) == 0)
        if zero_mask.any():
            fb    = int(tok_load.argmin().item())
            x_fb  = _rms(x_flat[zero_mask], self.expert_norm[fb])
            output[zero_mask] = self.experts[fb](x_fb)
            tok_load[fb]     += zero_mask.sum()

        # Shared expert always runs
        output = output + 0.25 * self.shared(x_flat)
        output = output.view(B, T, d)

        # Load balancing loss
        importance = probs.sum(dim=0)
        load       = tok_load / max(S, 1)
        lb_loss    = 0.01 * (importance.std() + load.std())

        return output, lb_loss

# ══════════════════════════════════════════════════════════════════════
# Transformer Block
# ══════════════════════════════════════════════════════════════════════
class BlockTorch(nn.Module):
    def __init__(self, cfg: DracoConfig):
        super().__init__()
        d = cfg.d_model
        # RMSNorm weights (scale only — NO bias)
        self.norm      = nn.Parameter(torch.ones(d))
        self.post_norm = nn.Parameter(torch.ones(d))
        self.attn      = GQATorch(cfg)
        self.moe       = MoETorch(cfg)

    def forward(self, x: torch.Tensor, intent_boost=None) -> Tuple[torch.Tensor, torch.Tensor]:
        x       = x + self.attn(_rms(x, self.norm))
        moe_out, lb = self.moe(_rms(x, self.post_norm), intent_boost)
        return x + moe_out, lb

# ══════════════════════════════════════════════════════════════════════
# Multi-Token Head
# ══════════════════════════════════════════════════════════════════════
class MultiTokenHeadTorch(nn.Module):
    IGNORE_INDEX = -100

    def __init__(self, cfg: DracoConfig, token_emb: nn.Embedding):
        super().__init__()
        d = cfg.d_model
        self.mtp_w     = cfg.mtp_weight
        self.h1_norm   = nn.Parameter(torch.ones(d))
        self.h2_norm   = nn.Parameter(torch.ones(d))
        self.h2_ffn    = SwiGLUTorch(d, max(d // 2, 32))
        self.token_emb = token_emb  # weight-tied

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        l1 = _rms(x, self.h1_norm) @ self.token_emb.weight.T
        h2 = x + self.h2_ffn(x)
        l2 = _rms(h2, self.h2_norm) @ self.token_emb.weight.T
        return l1, l2

    def loss(self, l1: torch.Tensor, l2: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, T, V = l1.shape
        t1      = targets[:, 1:T + 1]
        t2_raw  = targets[:, 2:T + 2]
        pad     = torch.full((B, T - t2_raw.shape[1]), self.IGNORE_INDEX,
                             device=targets.device, dtype=torch.long)
        t2      = torch.cat([t2_raw, pad], dim=1)
        loss1   = F.cross_entropy(l1.reshape(-1, V), t1.reshape(-1),
                                  ignore_index=self.IGNORE_INDEX)
        loss2   = F.cross_entropy(l2.reshape(-1, V), t2.reshape(-1),
                                  ignore_index=self.IGNORE_INDEX)
        return loss1 + self.mtp_w * loss2

# ══════════════════════════════════════════════════════════════════════
# DracoTransformerTorch — full model for training
# ══════════════════════════════════════════════════════════════════════
class DracoTransformerTorch(nn.Module):
    def __init__(self, cfg: DracoConfig):
        super().__init__()
        self.cfg        = cfg
        self.token_emb  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks     = nn.ModuleList([BlockTorch(cfg) for _ in range(cfg.n_layers)])
        self.norm_final = nn.Parameter(torch.ones(cfg.d_model))
        self.mtp        = MultiTokenHeadTorch(cfg, self.token_emb)
        self.apply(self._init_weights)
        n = sum(p.numel() for p in self.parameters())
        print(f"[DracoTorch] {n/1e9:.3f}B | {cfg.n_layers}L "
              f"{cfg.n_heads_q}Qh/{cfg.n_heads_kv}KVh MoE×{cfg.n_experts} "
              f"θ={cfg.rope_theta:.0f} | Base: Qwen 3.5 9B Instruct")

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            # No bias to init (bias=False everywhere)

    def forward(
        self,
        ids:          torch.Tensor,
        targets:      Optional[torch.Tensor] = None,
        intent_boost: Optional[Dict]         = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x        = self.token_emb(ids)
        total_lb = torch.tensor(0.0, device=ids.device)
        for blk in self.blocks:
            x, lb = blk(x, intent_boost)
            total_lb = total_lb + lb
        x      = _rms(x, self.norm_final)
        l1, l2 = self.mtp(x)
        loss   = None
        if targets is not None:
            loss = self.mtp.loss(l1, l2, targets) + total_lb
        return l1, loss

# ══════════════════════════════════════════════════════════════════════
# _full_sync — sync ALL weights from PyTorch model back to NumPy model
# ══════════════════════════════════════════════════════════════════════
def _full_sync(torch_model: DracoTransformerTorch, numpy_model) -> int:
    """
    Sync ALL trainable parameters from torch_model → numpy_model.
    Call this after every training checkpoint.
    Returns number of tensors synced.
    """
    import numpy as np
    synced = 0

    def _s(t_tensor, n_array):
        nonlocal synced
        arr = t_tensor.detach().cpu().float().numpy()
        if n_array is not None and n_array.shape == arr.shape:
            n_array[:] = arr; synced += 1

    _s(torch_model.token_emb.weight, numpy_model.token_emb)
    _s(torch_model.norm_final,       numpy_model.norm_final)

    for i, (tb, nb) in enumerate(zip(torch_model.blocks, numpy_model.blocks)):
        _s(tb.attn.wq.weight, nb.attn.wq)
        _s(tb.attn.wk.weight, nb.attn.wk)
        _s(tb.attn.wv.weight, nb.attn.wv)
        _s(tb.attn.wo.weight, nb.attn.wo)
        _s(tb.norm,      nb.norm)
        _s(tb.post_norm, nb.post_norm)
        _s(tb.moe.W_router.weight, nb.moe.W_router)
        _s(tb.moe.router_bias,     nb.moe.router_bias)
        for j, (te, ne) in enumerate(zip(tb.moe.experts, nb.moe.experts)):
            _s(te.gate_proj.weight, ne.W_g)
            _s(te.up_proj.weight,   ne.W_u)
            _s(te.down_proj.weight, ne.W_d)
            _s(tb.moe.expert_norm[j], nb.moe.expert_norm[j])
        _s(tb.moe.shared.gate_proj.weight, nb.moe.shared.W_g)
        _s(tb.moe.shared.up_proj.weight,   nb.moe.shared.W_u)
        _s(tb.moe.shared.down_proj.weight, nb.moe.shared.W_d)

    _s(torch_model.mtp.h1_norm, numpy_model.mtp.h1_norm)
    _s(torch_model.mtp.h2_norm, numpy_model.mtp.h2_norm)

    # h2_proj synced as transpose of token_emb (weight-tied)
    h2p_arr = torch_model.mtp.token_emb.weight.detach().cpu().float().numpy().T
    if numpy_model.mtp.h2_proj.shape == h2p_arr.shape:
        numpy_model.mtp.h2_proj[:] = h2p_arr; synced += 1

    _s(torch_model.mtp.h2_ffn.gate_proj.weight, numpy_model.mtp.h2_ffn.W_g)
    _s(torch_model.mtp.h2_ffn.up_proj.weight,   numpy_model.mtp.h2_ffn.W_u)
    _s(torch_model.mtp.h2_ffn.down_proj.weight, numpy_model.mtp.h2_ffn.W_d)

    # Sync lm_head (weight-tied to token_emb)
    numpy_model._sync_head()

    return synced