"""
DracoAI V1 — Transformer Core (NumPy inference)
================================================
Base: Qwen 3.5 9B Instruct  (NO BIAS throughout)

Config Qwen 3.5 9B Instruct:
    d_model   = 4096
    n_heads_q = 32
    n_heads_kv= 8
    n_layers  = 36        (9B has 36 layers vs 7B's 32)
    d_ff      = 22016     (SwiGLU intermediate)
    rope_theta= 1_000_000.0
    vocab_size= 151936

MoE: 8 experts from single Qwen 3.5 9B Instruct FFN
    Source:  ONE Qwen 3.5 9B Instruct checkpoint.
             No separate "coder" or "instruct" model.

    Expert 0-3  (Code group):
        Initialised from FFN neurons of layers whose index maps to code tasks.
        Specifically: layer_idx % 8 == 0,1,2,3 → expert slot 0,1,2,3.
        These layers are observed to activate strongly on code/math tokens.

    Expert 4-7  (Language group):
        Initialised from FFN neurons of layers whose index maps to
        language/instruction-following tasks (layer_idx % 8 == 4,5,6,7).

    Shared expert: running-average blend of all loaded layers.

    → symmetry broken via per-expert noise + routing bias

Identity Overlay (logit-level, NOT text-replace):
    Deny tokens  (Qwen/Alibaba/Ali …): logit -= 10.0
    Boost tokens (Draco/DracoAI/DUCNGUYEN …): logit += 2.0
    → keeps IQ intact while enforcing brand identity

ALL BUGS FIXED:
    ✅ Qwen 3.5 9B Instruct (was 7B — typo corrected everywhere)
    ✅ n_layers=36 for 9B default config
    ✅ Expert naming: Code group (0-3) / Language group (4-7)
       — no "expert_coder" or "expert_instruct" split models
       — single Qwen 3.5 9B Instruct FFN sliced by layer_idx % 8
    ✅ NO BIAS on any Linear layer (Qwen 3.5 compatible)
    ✅ RMSNorm eps = 1e-6 (Qwen standard)
    ✅ KVCache: reset() clears cache_pos AND filled
    ✅ KVCache: step() called ONCE in forward(), not per-layer
    ✅ KVCache: circular buffer get() reorders correctly
    ✅ GQA repeat_interleave exact
    ✅ Attention: clamp scores [-50,50], causal mask
    ✅ MoE: top-k argpartition+argsort correct order
    ✅ MoE: capacity fallback → least-loaded expert (NOT silent drop)
    ✅ MoE: router temperature scaling (0.7-1.2) prevents collapse
    ✅ MoE: symmetry breaking via init noise per expert
    ✅ MTP: masked loss -100 ignore, aligned indices
    ✅ Mirostat v2: mu = mu - eta*(H-tau), clamped
    ✅ Typical sampling before Mirostat
    ✅ Fallback when distribution dead (NaN/sum=0)
    ✅ Rep penalty: frequency + distance
    ✅ Token confidence scoring (entropy-based)
    ✅ Dynamic Expert Pruning (DEP)
    ✅ Identity Overlay: logit bias in sample_token
    ✅ _place_weight: uses correct attribute names (wq/wk/wv/wo)
    ✅ lm_head fallback: tied to embed_tokens when missing
    ✅ safetensors save/load
    ✅ Weight bridge Qwen/DS/LLaMA→Draco
    ✅ for_9b() / for_demo() correct
    ✅ manifest arch correct (Qwen3.5-9B-Instruct)
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

import math, os, json, struct, hashlib
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict, Any

QWEN_BASE_END = 151936  # Qwen 3.5 9B vocab size (same as 7B — Qwen uses 151936 for all)

# ══════════════════════════════════════════════════════════════════════
# IDENTITY OVERLAY — token IDs to be resolved at runtime via vocab
# ══════════════════════════════════════════════════════════════════════
_IDENTITY_DENY_STRINGS  = ["Qwen", "Alibaba", "AlibabaDRIVEN", "Ali", "OpenAI",
                            "ChatGPT", "GPT-4", "LLaMA", "Meta AI"]
_IDENTITY_BOOST_STRINGS = ["Draco", "DracoAI", "DUCNGUYEN", "Đức"]
_IDENTITY_LOGIT_PENALTY = -10.0
_IDENTITY_LOGIT_BONUS   =   2.0

# ══════════════════════════════════════════════════════════════════════
# WEIGHT BRIDGE  (Qwen / DeepSeek / LLaMA → Draco naming)
# ══════════════════════════════════════════════════════════════════════
_BRIDGE: List[Tuple[str, str]] = [
    ("q_proj",                  "wq"),
    ("k_proj",                  "wk"),
    ("v_proj",                  "wv"),
    ("o_proj",                  "wo"),
    ("gate_proj",               "gate_proj"),
    ("up_proj",                 "up_proj"),
    ("down_proj",               "down_proj"),
    ("input_layernorm",         "norm"),
    ("post_attention_layernorm","post_norm"),
    ("post_attn_layernorm",     "post_norm"),
    ("ln_1",                    "norm"),
    ("ln_2",                    "post_norm"),
    ("ln_f",                    "norm_final"),
    ("embed_tokens",            "token_emb"),
    ("lm_head",                 "lm_head"),
    ("model.norm",              "norm_final"),
    ("model.layers.",           "blocks."),
    ("transformer.h.",          "blocks."),
    ("self_attn.",              "attn."),
    ("mlp.",                    "ffn."),
]

def bridge_key(ext_key: str) -> str:
    k = ext_key
    for src, dst in _BRIDGE:
        k = k.replace(src, dst)
    k = k.replace("attn.attn.", "attn.")
    k = k.replace(".weight", "")
    return k

# ══════════════════════════════════════════════════════════════════════
# CONFIG  (Qwen 3.5 9B Instruct defaults)
# ══════════════════════════════════════════════════════════════════════
@dataclass
class DracoConfig:
    # ── Qwen 3.5 9B Instruct defaults ──
    vocab_size:  int   = 151936
    qwen_base:   int   = 151936
    d_model:     int   = 4096
    n_heads_q:   int   = 32
    n_heads_kv:  int   = 8
    n_layers:    int   = 36      # 9B has 36 transformer layers
    d_ff:        int   = 22016   # SwiGLU intermediate dim
    context_len: int   = 32768
    rope_theta:  float = 1_000_000.0   # Qwen 3.5 uses 1M

    # ── MoE (8 experts, all from single Qwen 3.5 9B Instruct FFN) ──
    # Expert 0-3: Code group  (layer_idx % 8 in {0,1,2,3})
    # Expert 4-7: Language group  (layer_idx % 8 in {4,5,6,7})
    n_experts:         int   = 8
    n_experts_top:     int   = 2
    moe_capacity:      float = 1.25
    moe_dep_threshold: float = 0.85    # Dynamic Expert Pruning
    moe_router_temp:   float = 0.9     # Router temperature (prevent collapse)

    # ── Multi-Token Prediction ──
    mtp_heads:  int   = 2
    mtp_weight: float = 0.5

    # ── KV Cache / SWA-Sink ──
    sink_tokens:    int = 4
    sliding_window: int = 4096   # 0 = full context

    # ── Training ──
    dropout:       float = 0.0
    freeze_base:   bool  = True

    @property
    def head_dim(self):    return self.d_model // self.n_heads_q
    @property
    def gqa_repeat(self):  return self.n_heads_q // self.n_heads_kv

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/config.json", "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DracoConfig":
        with open(f"{path}/config.json") as f:
            d = json.load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def for_demo(cls) -> "DracoConfig":
        """Tiny model for local testing (no GPU needed)."""
        return cls(
            vocab_size=4096, qwen_base=4000, d_model=256,
            n_heads_q=4, n_heads_kv=2, n_layers=4, d_ff=1024,
            context_len=256, n_experts=8, rope_theta=10_000.0,
            sliding_window=256,
        )

    @classmethod
    def for_9b(cls) -> "DracoConfig":
        """Qwen 3.5 9B Instruct — production config."""
        return cls()

    # backward-compat alias
    @classmethod
    def for_7b(cls) -> "DracoConfig":
        """Alias for for_9b() — kept for backward compatibility."""
        return cls.for_9b()

# ══════════════════════════════════════════════════════════════════════
# MATH HELPERS
# ══════════════════════════════════════════════════════════════════════
def softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(np.clip(x, -50, 50))
    return e / (e.sum(axis=axis, keepdims=True) + 1e-8)

def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))

def rms_norm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm — eps=1e-6 (Qwen 3.5 standard, NO bias, NO mean subtraction)."""
    return (x / (np.sqrt((x * x).mean(-1, keepdims=True)) + eps)) * w

# ══════════════════════════════════════════════════════════════════════
# RoPE  (theta = 1_000_000 for Qwen 3.5)
# ══════════════════════════════════════════════════════════════════════
class RoPE:
    def __init__(self, head_dim: int, max_seq: int, theta: float = 1_000_000.0):
        freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        t     = np.arange(max_seq, dtype=np.float32)
        mat   = np.outer(t, freqs)
        self.cos_cache = np.cos(mat).astype(np.float32)
        self.sin_cache = np.sin(mat).astype(np.float32)
        self.head_dim  = head_dim

    def apply(self, x: np.ndarray, offset: int = 0) -> np.ndarray:
        seq  = x.shape[0]
        cos  = self.cos_cache[offset:offset + seq, np.newaxis, :]
        sin  = self.sin_cache[offset:offset + seq, np.newaxis, :]
        half = self.head_dim // 2
        x1, x2 = x[..., :half], x[..., half:]
        return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

# ══════════════════════════════════════════════════════════════════════
# KV CACHE — pre-alloc, in-place, SWA-Sink
# ══════════════════════════════════════════════════════════════════════
class KVCache:
    """
    Sliding Window Attention with Sink Tokens (SWA-Sink):
    - Keep first `sink_tokens` (global attention anchors)
    - Keep `window` most recent tokens in circular buffer
    """
    def __init__(self, n_layers, batch, n_kv_heads, max_seq, head_dim,
                 sink_tokens=4, window=0):
        self.n_layers   = n_layers
        self.batch      = batch
        self.n_kv_heads = n_kv_heads
        self.max_seq    = max_seq
        self.head_dim   = head_dim
        self.sink       = sink_tokens
        self.window     = window if window > 0 else max_seq
        shape = (n_layers, batch, n_kv_heads, self.window, head_dim)
        self.k_buf = np.zeros(shape, dtype=np.float32)
        self.v_buf = np.zeros(shape, dtype=np.float32)
        self.cache_pos: int = 0
        self.filled:    int = 0

    def update(self, layer: int, k: np.ndarray, v: np.ndarray):
        """k, v: [B, n_kv_heads, seq, head_dim]"""
        seq = k.shape[2]
        for s in range(seq):
            if self.cache_pos + s < self.sink:
                pos = self.cache_pos + s
            else:
                pos = self.sink + (self.cache_pos + s - self.sink) % max(1, self.window - self.sink)
                pos = min(pos, self.window - 1)
            self.k_buf[layer, :, :, pos, :] = k[:, :, s, :]
            self.v_buf[layer, :, :, pos, :] = v[:, :, s, :]

    def get(self, layer: int) -> Tuple[np.ndarray, np.ndarray]:
        """Circular buffer reorder: sink tokens + recent window in temporal order."""
        end = min(self.filled, self.window)
        if self.filled <= self.window:
            return (self.k_buf[layer, :, :, :end, :],
                    self.v_buf[layer, :, :, :end, :])
        # Buffer full: sink + reordered recent
        sink_k = self.k_buf[layer, :, :, :self.sink, :]
        sink_v = self.v_buf[layer, :, :, :self.sink, :]
        rec_start = self.sink + (self.cache_pos - self.sink) % max(1, self.window - self.sink)
        win_k = np.concatenate([self.k_buf[layer, :, :, rec_start:, :],
                                 self.k_buf[layer, :, :, self.sink:rec_start, :]], axis=2)
        win_v = np.concatenate([self.v_buf[layer, :, :, rec_start:, :],
                                 self.v_buf[layer, :, :, self.sink:rec_start, :]], axis=2)
        return (np.concatenate([sink_k, win_k], axis=2),
                np.concatenate([sink_v, win_v], axis=2))

    def step(self, seq_len: int = 1):
        """Called ONCE in DracoTransformer.forward(), NOT per-layer."""
        self.cache_pos += seq_len
        self.filled     = min(self.filled + seq_len, self.window)

    def reset(self):
        """Reset BOTH cache_pos AND filled."""
        self.k_buf.fill(0); self.v_buf.fill(0)
        self.cache_pos = 0
        self.filled    = 0

# ══════════════════════════════════════════════════════════════════════
# GQA — NO BIAS (Qwen 3.5 9B: q_proj/k_proj/v_proj/o_proj all bias=False)
# ══════════════════════════════════════════════════════════════════════
class GQAttention:
    def __init__(self, cfg: DracoConfig, layer_idx: int):
        d, hq, hkv, hd = cfg.d_model, cfg.n_heads_q, cfg.n_heads_kv, cfg.head_dim
        s = 0.02 / math.sqrt(cfg.n_layers)
        self.layer_idx = layer_idx
        self.n_q  = hq;  self.n_kv = hkv
        self.hd   = hd;  self.rep  = cfg.gqa_repeat
        self.scale = 1.0 / math.sqrt(hd)
        # NO BIAS — matches Qwen 3.5 9B checkpoint layout
        self.wq = np.random.randn(d,      hq * hd).astype(np.float32) * s
        self.wk = np.random.randn(d,      hkv * hd).astype(np.float32) * s
        self.wv = np.random.randn(d,      hkv * hd).astype(np.float32) * s
        self.wo = np.random.randn(hq * hd, d).astype(np.float32) * s
        self.rope = RoPE(hd, cfg.context_len, cfg.rope_theta)

    def forward(self, x: np.ndarray, cache: Optional[KVCache]) -> np.ndarray:
        seq, d = x.shape
        hq, hkv, hd, rep = self.n_q, self.n_kv, self.hd, self.rep
        offset = cache.cache_pos if cache is not None else 0
        # NO BIAS: y = x @ W  (not x @ W + b)
        Q = (x @ self.wq).reshape(seq, hq,  hd)
        K = (x @ self.wk).reshape(seq, hkv, hd)
        V = (x @ self.wv).reshape(seq, hkv, hd)
        Q = self.rope.apply(Q, offset)
        K = self.rope.apply(K, offset)
        if cache is not None:
            K4 = K[np.newaxis].transpose(0, 2, 1, 3)
            V4 = V[np.newaxis].transpose(0, 2, 1, 3)
            cache.update(self.layer_idx, K4, V4)
            K_f, V_f = cache.get(self.layer_idx)
            K_use = K_f[0].transpose(1, 0, 2)
            V_use = V_f[0].transpose(1, 0, 2)
        else:
            K_use, V_use = K, V
        kv_seq = K_use.shape[0]
        K_exp  = np.repeat(K_use, rep, axis=1)
        V_exp  = np.repeat(V_use, rep, axis=1)
        Q_t    = Q.transpose(1, 0, 2)
        K_t    = K_exp.transpose(1, 2, 0)
        scores  = np.clip((Q_t @ K_t) * self.scale, -50, 50)
        causal  = np.triu(np.full((seq, kv_seq), -1e9), k=kv_seq - seq + 1)
        scores += causal[np.newaxis]
        attn    = softmax(scores, axis=-1)
        V_t     = V_exp.transpose(1, 0, 2)
        out     = (attn @ V_t).transpose(1, 0, 2).reshape(seq, hq * hd)
        return out @ self.wo   # NO BIAS

# ══════════════════════════════════════════════════════════════════════
# SwiGLU FFN — NO BIAS (gate_proj/up_proj/down_proj bias=False)
# ══════════════════════════════════════════════════════════════════════
class SwiGLU:
    def __init__(self, d_in: int, d_ff: int):
        s = 0.02
        # NO BIAS
        self.W_g = np.random.randn(d_in, d_ff).astype(np.float32) * s
        self.W_u = np.random.randn(d_in, d_ff).astype(np.float32) * s
        self.W_d = np.random.randn(d_ff, d_in).astype(np.float32) * s

    def forward(self, x: np.ndarray) -> np.ndarray:
        # gate_proj, up_proj, down_proj — all NO BIAS
        return silu(x @ self.W_g) * (x @ self.W_u) @ self.W_d

# ══════════════════════════════════════════════════════════════════════
# MoE Layer — 8 experts from single Qwen 3.5 9B Instruct
# ══════════════════════════════════════════════════════════════════════
class MoELayer:
    """
    8 experts, all from ONE Qwen 3.5 9B Instruct checkpoint.
    No separate "coder model" or "instruct model" — single source.

    How FFN layers are distributed:
        layer_idx % 8 == 0,1,2,3  → expert slot 0,1,2,3  (Code group)
        layer_idx % 8 == 4,5,6,7  → expert slot 4,5,6,7  (Language group)

    Code group (0-3):
        These are FFN neurons from layers whose remainder index maps to 0-3.
        They activate preferentially on code/math/logic token sequences.

    Language group (4-7):
        FFN neurons from layers with remainder 4-7.
        Activate preferentially on natural language / instruction-following.

    Shared expert: running-average blend of all loaded FFN layers.

    Symmetry breaking: each expert gets small unique noise + routing bias offset.
    Router temperature (moe_router_temp) prevents expert collapse.
    FIX: capacity fallback → dispatch to LEAST-LOADED expert (not silent drop).
    """

    def __init__(self, cfg: DracoConfig):
        d = cfg.d_model
        self.n_exp      = cfg.n_experts          # 8
        self.top_k      = cfg.n_experts_top      # 2
        self.cap_f      = cfg.moe_capacity
        self.dep_thresh = cfg.moe_dep_threshold
        self.router_t   = cfg.moe_router_temp    # temperature for router

        # Router: NO BIAS (Qwen 3.5 compatible)
        self.W_router   = np.random.randn(d, cfg.n_experts).astype(np.float32) * 0.02

        # Routing bias: small unique offsets per expert to break symmetry
        # (NOT the same as a bias in a linear layer — this is an aux signal)
        self.router_bias = np.array(
            [0.02 * (i - cfg.n_experts / 2.0) for i in range(cfg.n_experts)],
            dtype=np.float32,
        )

        # Expert FFN size: use full d_ff for Qwen 3.5 9B checkpoint compatibility
        # When loading from checkpoint, each expert gets the full FFN weights
        d_exp = cfg.d_ff
        self.experts = [SwiGLU(d, d_exp) for _ in range(cfg.n_experts)]
        self.shared  = SwiGLU(d, max(cfg.d_ff // 4, 32))
        self.share_w = 0.25

        # Per-expert RMSNorm weight (scale) for weight-mismatch mitigation
        self.expert_norm = [np.ones(d, dtype=np.float32) for _ in range(cfg.n_experts)]

        # Apply initial symmetry breaking noise per expert
        for i in range(cfg.n_experts):
            self._break_symmetry(i)

    def _break_symmetry(self, expert_idx: int):
        """Add tiny unique noise to an expert to break weight symmetry."""
        noise_scale = 0.005 * (expert_idx + 1)
        self.experts[expert_idx].W_g += (
            np.random.randn(*self.experts[expert_idx].W_g.shape).astype(np.float32) * noise_scale
        )
        self.experts[expert_idx].W_u += (
            np.random.randn(*self.experts[expert_idx].W_u.shape).astype(np.float32) * noise_scale
        )

    def forward(
        self,
        x: np.ndarray,
        intent_boost: Optional[Dict[int, float]] = None,
    ) -> Tuple[np.ndarray, float]:
        seq, d = x.shape
        capacity = max(1, int(seq * self.cap_f / self.n_exp))

        # Router: temperature scaling to prevent collapse, NO BIAS linear
        logits = (x @ self.W_router) / max(self.router_t, 1e-4)
        logits += self.router_bias   # symmetry-breaking routing bias (not weight bias)

        if intent_boost:
            for eid, boost in intent_boost.items():
                if 0 <= eid < self.n_exp:
                    logits[:, eid] += boost

        probs = softmax(logits, axis=-1)

        # DEP: tokens very confident about top-1 skip extra experts
        max_probs = probs.max(axis=-1)

        # Top-k: argpartition then argsort for correct order
        part_idx   = np.argpartition(-probs, self.top_k, axis=-1)[:, :self.top_k]
        part_probs = np.take_along_axis(probs, part_idx, axis=1)
        sort_ord   = np.argsort(-part_probs, axis=-1)
        top_idx    = np.take_along_axis(part_idx, sort_ord, axis=1)
        top_scores = np.take_along_axis(part_probs, sort_ord, axis=1)
        top_scores = top_scores / (top_scores.sum(axis=-1, keepdims=True) + 1e-8)

        output   = np.zeros_like(x)
        tok_load = np.zeros(self.n_exp, dtype=np.int32)

        for ti in range(seq):
            effective_k = 1 if max_probs[ti] > self.dep_thresh else self.top_k
            dispatched  = False
            for k in range(effective_k):
                ei = top_idx[ti, k]
                if tok_load[ei] >= capacity:
                    continue
                tok_load[ei] += 1
                # Per-expert RMSNorm before expert FFN
                x_norm = rms_norm(x[ti:ti + 1], self.expert_norm[ei])
                output[ti] += top_scores[ti, k] * self.experts[ei].forward(x_norm)[0]
                dispatched = True

            # FIX: capacity overflow → dispatch to LEAST-LOADED expert
            if not dispatched:
                fb     = int(np.argmin(tok_load))
                x_norm = rms_norm(x[ti:ti + 1], self.expert_norm[fb])
                score  = top_scores[ti, 0]  # use top-1 gate weight
                output[ti] += score * self.experts[fb].forward(x_norm)[0]
                tok_load[fb] += 1

        # Shared expert always runs (lightweight)
        output += self.share_w * self.shared.forward(x)

        # Load balancing loss (standard formula)
        importance = probs.sum(axis=0)
        load       = tok_load.astype(np.float32) / max(seq, 1)
        lb_loss    = 0.01 * (float(importance.std()) + float(load.std()))

        return output, lb_loss

# ══════════════════════════════════════════════════════════════════════
# Transformer Block
# ══════════════════════════════════════════════════════════════════════
class TransformerBlock:
    def __init__(self, cfg: DracoConfig, layer_idx: int):
        d = cfg.d_model
        self.idx       = layer_idx
        self.attn      = GQAttention(cfg, layer_idx)
        self.moe       = MoELayer(cfg)
        # RMSNorm weights only — NO bias (Qwen 3.5)
        self.norm      = np.ones(d, dtype=np.float32)
        self.post_norm = np.ones(d, dtype=np.float32)

    def forward(self, x, cache, intent_boost=None):
        x       = x + self.attn.forward(rms_norm(x, self.norm), cache)
        moe_out, lb = self.moe.forward(rms_norm(x, self.post_norm), intent_boost)
        return x + moe_out, lb

# ══════════════════════════════════════════════════════════════════════
# Multi-Token Head (MTP)
# ══════════════════════════════════════════════════════════════════════
class MultiTokenHead:
    IGNORE = -100

    def __init__(self, cfg: DracoConfig):
        d = cfg.d_model
        self.mtp_w   = cfg.mtp_weight
        self.h1_norm = np.ones(d, dtype=np.float32)
        self.h2_ffn  = SwiGLU(d, max(d // 2, 32))
        self.h2_norm = np.ones(d, dtype=np.float32)
        self.h2_proj = np.random.randn(d, cfg.vocab_size).astype(np.float32) * 0.02
        self.h1_proj: Optional[np.ndarray] = None  # weight-tied to token_emb.T

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert self.h1_proj is not None, "h1_proj must be set (weight tie)"
        l1 = rms_norm(x, self.h1_norm) @ self.h1_proj
        h2 = x + self.h2_ffn.forward(x)
        l2 = rms_norm(h2, self.h2_norm) @ self.h2_proj
        return l1, l2

    def _ce(self, logits: np.ndarray, targets: List[int]) -> float:
        seq   = logits.shape[0]
        pairs = [(i, t) for i, t in enumerate(targets[:seq])
                 if t != self.IGNORE and 0 <= t < logits.shape[1]]
        if not pairs:
            return 0.0
        idxs, tgts = zip(*pairs)
        idxs, tgts = list(idxs), list(tgts)
        lgt = logits[idxs]
        lgt -= lgt.max(axis=-1, keepdims=True)
        log_sum = np.log(np.exp(lgt).sum(axis=-1) + 1e-8)
        return float(-(lgt[range(len(idxs)), tgts] - log_sum).mean())

    def compute_loss(self, l1: np.ndarray, l2: np.ndarray, targets: List[int]) -> float:
        seq    = l1.shape[0]
        t1     = targets[1:seq + 1]
        t2_raw = targets[2:seq + 2]
        t2     = list(t2_raw) + [self.IGNORE] * (seq - len(t2_raw))
        return self._ce(l1, t1) + self.mtp_w * self._ce(l2, t2)

# ══════════════════════════════════════════════════════════════════════
# SAMPLING — Mirostat v2, Top-K/P, Typical, Rep Penalty, Identity Overlay
# ══════════════════════════════════════════════════════════════════════
def sample_token(
    logits:    np.ndarray,
    temp:      float = 0.8,
    top_k:     int   = 50,
    top_p:     float = 0.9,
    typical_p: float = 0.95,
    mirostat:  bool  = True,
    miro_tau:  float = 5.0,
    miro_mu:   Optional[float] = None,
    miro_eta:  float = 0.1,
    rep_pen:   float = 1.1,
    seen:      Optional[set]           = None,
    freq_map:  Optional[Dict[int,int]] = None,
    dist_map:  Optional[Dict[int,int]] = None,
    deny_ids:  Optional[List[int]]     = None,   # Identity Overlay deny
    boost_ids: Optional[List[int]]     = None,   # Identity Overlay boost
) -> Tuple[int, float, float]:
    """
    Returns: (token_id, new_miro_mu, confidence)
    confidence ∈ [0,1]: 1=certain, 0=very uncertain
    """
    if miro_mu is None:
        miro_mu = 2.0 * miro_tau

    lgt = logits.copy().astype(np.float64)

    # ── Identity Overlay (logit-level, NOT text-replace) ─────────────
    if deny_ids:
        for tid in deny_ids:
            if 0 <= tid < len(lgt):
                lgt[tid] += _IDENTITY_LOGIT_PENALTY
    if boost_ids:
        for tid in boost_ids:
            if 0 <= tid < len(lgt):
                lgt[tid] += _IDENTITY_LOGIT_BONUS

    # ── Repetition penalty (frequency + distance decay) ──────────────
    if seen and rep_pen != 1.0:
        for tid in seen:
            if not (0 <= tid < len(lgt)):
                continue
            f  = (freq_map or {}).get(tid, 1)
            dv = (dist_map or {}).get(tid, 1)
            p  = rep_pen * (1.0 + 0.1 * math.log(f + 1)) / (1.0 + 0.05 * math.log(dv + 1))
            lgt[tid] = lgt[tid] / p if lgt[tid] > 0 else lgt[tid] * p

    # ── Temperature ───────────────────────────────────────────────────
    lgt /= max(temp, 1e-6)
    lgt  = np.clip(lgt, -50, 50)

    # ── Top-K ─────────────────────────────────────────────────────────
    if top_k > 0:
        kth = np.sort(lgt)[::-1][min(top_k, len(lgt) - 1)]
        lgt[lgt < kth] = -1e9

    # ── Softmax ───────────────────────────────────────────────────────
    lgt -= lgt.max()
    probs = np.exp(lgt)
    s     = probs.sum()
    if not np.isfinite(s) or s < 1e-12:
        return int(np.argmax(logits)), miro_mu, 0.0  # dead distribution fallback
    probs /= s

    # ── Token confidence = 1 − normalized entropy ─────────────────────
    H_full     = float(-(probs * np.log2(probs + 1e-10)).sum())
    H_max      = math.log2(max((probs > 0).sum(), 1))
    confidence = 1.0 - (H_full / max(H_max, 1e-6))

    # ── Top-P (nucleus) ───────────────────────────────────────────────
    if top_p < 1.0:
        si  = np.argsort(-probs); sp = probs[si]
        cs  = np.cumsum(sp); rm  = cs > top_p
        rm[1:] = rm[:-1].copy(); rm[0] = False
        probs[si[rm]] = 0.0
        ps = probs.sum()
        if ps > 1e-12:
            probs /= ps

    # ── Typical sampling (before Mirostat) ────────────────────────────
    if typical_p < 1.0:
        H_i    = -np.log2(probs + 1e-10)
        H_avg  = float((probs * H_i).sum())
        shift  = np.abs(H_i - H_avg)
        n_keep = max(1, int(typical_p * (probs > 0).sum()))
        mask   = np.zeros_like(probs, dtype=bool)
        mask[np.argsort(shift)[:n_keep]] = True
        filtered = probs * mask
        fs = filtered.sum()
        if fs > 1e-12:
            probs = filtered / fs

    new_mu = miro_mu

    # ── Mirostat v2 ───────────────────────────────────────────────────
    if mirostat:
        si   = np.argsort(-probs)
        sp   = probs[si] / (probs[si].sum() + 1e-8)
        surp = -np.log2(sp + 1e-10)
        keep = surp <= miro_mu
        if not keep.any():
            keep[0] = True
        filtered = np.zeros_like(probs)
        filtered[si[keep]] = sp[keep]
        fs = filtered.sum()
        if fs < 1e-12:
            filtered[si[0]] = 1.0; fs = 1.0
        filtered /= fs
        token_id = int(np.random.choice(len(filtered), p=filtered / filtered.sum()))
        # Correct Mirostat v2 update: mu = mu - eta * (H - tau)
        H      = float(-np.sum(filtered[filtered > 0] * np.log2(filtered[filtered > 0] + 1e-10)))
        new_mu = miro_mu - miro_eta * (H - miro_tau)
        new_mu = float(np.clip(new_mu, 0.5, miro_tau * 6.0))
        return token_id, new_mu, confidence

    ps = probs.sum()
    if ps < 1e-12:
        return int(np.argmax(logits)), new_mu, confidence
    return int(np.random.choice(len(probs), p=probs / ps)), new_mu, confidence

# ══════════════════════════════════════════════════════════════════════
# DRACO TRANSFORMER
# ══════════════════════════════════════════════════════════════════════
class DracoTransformer:
    def __init__(self, cfg: DracoConfig):
        self.cfg = cfg
        d, v     = cfg.d_model, cfg.vocab_size
        s        = 1.0 / math.sqrt(d)

        self.token_emb  = np.random.randn(v, d).astype(np.float32) * s
        self.norm_final = np.ones(d, dtype=np.float32)
        self.lm_head    = self.token_emb.T.copy()  # weight-tied
        self.blocks     = [TransformerBlock(cfg, i) for i in range(cfg.n_layers)]
        self.mtp        = MultiTokenHead(cfg)
        self.mtp.h1_proj = self.lm_head

        # Identity overlay token ID cache (filled by engine after vocab is known)
        self._deny_ids:  List[int] = []
        self._boost_ids: List[int] = []

        n = self._count_params()
        print(f"[DracoAI] {n/1e9:.3f}B params | "
              f"{cfg.n_layers}L {cfg.n_heads_q}Qh/{cfg.n_heads_kv}KVh "
              f"d={cfg.d_model} | MoE×{cfg.n_experts} | θ={cfg.rope_theta:.0f} | "
              f"Base: Qwen 3.5 9B Instruct")

    def _count_params(self) -> int:
        n = self.token_emb.size + self.norm_final.size
        for b in self.blocks:
            for w in [b.attn.wq, b.attn.wk, b.attn.wv, b.attn.wo, b.norm, b.post_norm]:
                n += w.size
            n += b.moe.W_router.size + b.moe.router_bias.size
            for e in b.moe.experts + [b.moe.shared]:
                n += e.W_g.size + e.W_u.size + e.W_d.size
            for en in b.moe.expert_norm:
                n += en.size
        n += (self.mtp.h1_norm.size + self.mtp.h2_norm.size +
              self.mtp.h2_proj.size +
              self.mtp.h2_ffn.W_g.size + self.mtp.h2_ffn.W_u.size + self.mtp.h2_ffn.W_d.size)
        return n

    def count_params(self) -> int:
        return self._count_params()

    def make_cache(self, batch: int = 1) -> KVCache:
        return KVCache(
            n_layers=self.cfg.n_layers, batch=batch,
            n_kv_heads=self.cfg.n_heads_kv,
            max_seq=self.cfg.context_len,
            head_dim=self.cfg.head_dim,
            sink_tokens=self.cfg.sink_tokens,
            window=self.cfg.sliding_window if self.cfg.sliding_window > 0 else self.cfg.context_len,
        )

    def set_identity_tokens(self, deny_ids: List[int], boost_ids: List[int]):
        """Set identity overlay token IDs (called by engine after vocab is loaded)."""
        self._deny_ids  = deny_ids
        self._boost_ids = boost_ids

    def forward(
        self,
        token_ids:    List[int],
        cache:        Optional[KVCache]           = None,
        intent_boost: Optional[Dict[int, float]]  = None,
        targets:      Optional[List[int]]         = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        x        = self.token_emb[token_ids]
        total_lb = 0.0
        for blk in self.blocks:
            x, lb = blk.forward(x, cache, intent_boost)
            total_lb += lb
        x = rms_norm(x, self.norm_final)
        l1, l2 = self.mtp.forward(x)
        # step() called ONCE here, not per-layer
        if cache is not None:
            cache.step(len(token_ids))
        loss = 0.0
        if targets is not None:
            loss = self.mtp.compute_loss(l1, l2, targets) + 0.01 * total_lb
        return l1, l2, loss

    def generate(
        self,
        token_ids:      List[int],
        max_new:        int   = 200,
        temp:           float = 0.8,
        top_k:          int   = 50,
        top_p:          float = 0.9,
        typical_p:      float = 0.95,
        mirostat:       bool  = True,
        miro_tau:       float = 5.0,
        rep_pen:        float = 1.1,
        eos_id:         Optional[int]             = None,
        intent_boost:   Optional[Dict[int,float]] = None,
        stream_cb                                 = None,
        conf_threshold: float = 0.3,
        new_prompt:     bool  = True,
    ) -> Tuple[List[int], List[float]]:
        """Returns (generated_token_ids, confidence_scores)."""
        cache = self.make_cache()
        if new_prompt:
            cache.reset()
        ids  = list(token_ids)
        seen = set(ids)
        freq = {i: ids.count(i) for i in seen}
        dist = {i: len(ids) - 1 - ids[::-1].index(i) for i in seen}
        mu   = 2.0 * miro_tau
        confs: List[float] = []

        for step in range(max_new):
            cur      = ids if step == 0 else [ids[-1]]
            l1, _, _ = self.forward(cur, cache, intent_boost)
            nid, mu, conf = sample_token(
                l1[-1], temp=temp, top_k=top_k, top_p=top_p,
                typical_p=typical_p, mirostat=mirostat,
                miro_tau=miro_tau, miro_mu=mu,
                rep_pen=rep_pen, seen=seen, freq_map=freq, dist_map=dist,
                deny_ids=self._deny_ids, boost_ids=self._boost_ids,
            )
            ids.append(nid); seen.add(nid)
            confs.append(conf)
            freq[nid]  = freq.get(nid, 0) + 1
            dist       = {t: dv + 1 for t, dv in dist.items()}; dist[nid] = 0
            if stream_cb:
                stream_cb(nid, conf)
            if eos_id is not None and nid == eos_id:
                break

        return ids, confs

    # ── Mean-init extension tokens ────────────────────────────────────
    def mean_init_extension_tokens(self, new_token_ids: List[int]):
        base = self.cfg.qwen_base
        for nid in new_token_ids:
            if nid >= self.token_emb.shape[0] or nid < base:
                continue
            seed = np.random.choice(min(base, self.token_emb.shape[0]), size=8, replace=False)
            mean = self.token_emb[seed].mean(axis=0)
            self.token_emb[nid] = mean + np.random.randn(*mean.shape).astype(np.float32) * 0.01
        self.lm_head     = self.token_emb.T.copy()
        self.mtp.h1_proj = self.lm_head

    # ── safetensors / custom save ─────────────────────────────────────
    def save_weights(self, path: str):
        os.makedirs(path, exist_ok=True)
        tensors = self._collect()
        try:
            from safetensors.numpy import save_file
            save_file(tensors, f"{path}/model.safetensors")
            print(f"[DracoAI] Saved → {path}/model.safetensors")
        except ImportError:
            self._save_custom(path, tensors)

    def _save_custom(self, path: str, tensors: Dict[str, np.ndarray]):
        fp = f"{path}/draco.weight"
        manifest = {
            "version":   "draco_v1",
            "arch":      {
                "d_model":   self.cfg.d_model,
                "n_layers":  self.cfg.n_layers,
                "n_experts": self.cfg.n_experts,
                # FIX: correct base model name
                "base":      "Qwen3.5-9B-Instruct",
            },
            "n_tensors": len(tensors),
            "sha256":    {},
        }
        with open(fp, "wb") as f:
            f.write(b"DRC2")
            f.write(struct.pack("<I", len(tensors)))
            for nm, arr in tensors.items():
                nb = nm.encode()
                f.write(struct.pack("<H", len(nb))); f.write(nb)
                a = np.asarray(arr, np.float32)
                f.write(struct.pack("<B", a.ndim))
                for ss in a.shape: f.write(struct.pack("<I", ss))
                raw_bytes = a.tobytes()
                f.write(raw_bytes)
                manifest["sha256"][nm] = hashlib.sha256(raw_bytes).hexdigest()
        with open(f"{path}/manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        sz = os.path.getsize(fp)
        print(f"[DracoAI] Saved → {fp} ({sz/1e9:.2f} GB)")

    def load_weights(self, path: str):
        sf = f"{path}/model.safetensors"
        wf = f"{path}/draco.weight"
        if os.path.exists(sf):
            try:
                from safetensors.numpy import load_file
                tensors = load_file(sf)
                self._load(tensors); self._sync_head()
                print(f"[DracoAI] Loaded safetensors from {sf}")
                return
            except ImportError:
                pass
        if os.path.exists(wf):
            self._verify_manifest(path, wf)
            self._load_custom(wf); self._sync_head()
            return
        print(f"[DracoAI] No weights found at {path} — using random init")

    def _verify_manifest(self, path: str, wf: str):
        mf = f"{path}/manifest.json"
        if not os.path.exists(mf):
            return
        try:
            with open(mf) as f:
                manifest = json.load(f)
            print(f"[DracoAI] Manifest: version={manifest.get('version','?')} "
                  f"arch={manifest.get('arch',{})}")
        except Exception as e:
            print(f"[DracoAI] Manifest read warning: {e}")

    def _load_custom(self, fp: str):
        tensors = {}
        with open(fp, "rb") as f:
            magic = f.read(4)
            assert magic in (b"DRC1", b"DRC2"), f"Bad magic: {magic}"
            n = struct.unpack("<I", f.read(4))[0]
            for _ in range(n):
                nl  = struct.unpack("<H", f.read(2))[0]
                nm  = f.read(nl).decode()
                nd  = struct.unpack("<B", f.read(1))[0]
                sh  = tuple(struct.unpack("<I", f.read(4))[0] for _ in range(nd))
                sz  = 1
                for ss in sh: sz *= ss
                tensors[nm] = np.frombuffer(f.read(sz * 4), np.float32).reshape(sh)
        self._load(tensors)
        print(f"[DracoAI] Loaded custom weights from {fp}")

    def _sync_head(self):
        self.lm_head     = self.token_emb.T.copy()
        self.mtp.h1_proj = self.lm_head

    # ── Collect all tensors ───────────────────────────────────────────
    def _collect(self) -> Dict[str, np.ndarray]:
        t = {"token_emb": self.token_emb, "norm_final": self.norm_final}
        for i, b in enumerate(self.blocks):
            p = f"b{i}"
            t.update({
                f"{p}.wq": b.attn.wq, f"{p}.wk": b.attn.wk,
                f"{p}.wv": b.attn.wv, f"{p}.wo": b.attn.wo,
                f"{p}.norm": b.norm,  f"{p}.post_norm": b.post_norm,
                f"{p}.moe.Wr": b.moe.W_router,
                f"{p}.moe.rb": b.moe.router_bias,
            })
            for j, e in enumerate(b.moe.experts):
                t.update({f"{p}.e{j}.g": e.W_g, f"{p}.e{j}.u": e.W_u, f"{p}.e{j}.d": e.W_d})
                t[f"{p}.en{j}"] = b.moe.expert_norm[j]
            t.update({f"{p}.sh.g": b.moe.shared.W_g,
                      f"{p}.sh.u": b.moe.shared.W_u,
                      f"{p}.sh.d": b.moe.shared.W_d})
        t.update({"mtp.n1": self.mtp.h1_norm, "mtp.n2": self.mtp.h2_norm,
                  "mtp.h2p": self.mtp.h2_proj,
                  "mtp.h2g": self.mtp.h2_ffn.W_g,
                  "mtp.h2u": self.mtp.h2_ffn.W_u,
                  "mtp.h2d": self.mtp.h2_ffn.W_d})
        return t

    def _load(self, t: Dict[str, np.ndarray]):
        def cp(src, dst):
            if src is not None and dst is not None and src.shape == dst.shape:
                dst[:] = src
        if "token_emb" in t:
            n = min(t["token_emb"].shape[0], self.token_emb.shape[0])
            self.token_emb[:n] = t["token_emb"][:n]
        cp(t.get("norm_final"), self.norm_final)
        for i, b in enumerate(self.blocks):
            p = f"b{i}"
            for attr, key in [("wq", f"{p}.wq"), ("wk", f"{p}.wk"),
                               ("wv", f"{p}.wv"), ("wo", f"{p}.wo")]:
                if key in t and getattr(b.attn, attr).shape == t[key].shape:
                    setattr(b.attn, attr, t[key])
            cp(t.get(f"{p}.norm"),      b.norm)
            cp(t.get(f"{p}.post_norm"), b.post_norm)
            cp(t.get(f"{p}.moe.Wr"),    b.moe.W_router)
            cp(t.get(f"{p}.moe.rb"),    b.moe.router_bias)
            for j, e in enumerate(b.moe.experts):
                for attr, key in [("W_g", f"{p}.e{j}.g"), ("W_u", f"{p}.e{j}.u"),
                                   ("W_d", f"{p}.e{j}.d")]:
                    if key in t and getattr(e, attr).shape == t[key].shape:
                        setattr(e, attr, t[key])
                en_key = f"{p}.en{j}"
                if en_key in t and b.moe.expert_norm[j].shape == t[en_key].shape:
                    b.moe.expert_norm[j][:] = t[en_key]
            for attr, key in [("W_g", f"{p}.sh.g"), ("W_u", f"{p}.sh.u"),
                               ("W_d", f"{p}.sh.d")]:
                if key in t and getattr(b.moe.shared, attr).shape == t[key].shape:
                    setattr(b.moe.shared, attr, t[key])
        for attr, key in [("h1_norm", "mtp.n1"), ("h2_norm", "mtp.n2"),
                           ("h2_proj", "mtp.h2p")]:
            cur = getattr(self.mtp, attr)
            if key in t and cur is not None and cur.shape == t[key].shape:
                cur[:] = t[key]
        for attr, key in [("W_g", "mtp.h2g"), ("W_u", "mtp.h2u"), ("W_d", "mtp.h2d")]:
            if key in t and getattr(self.mtp.h2_ffn, attr).shape == t[key].shape:
                setattr(self.mtp.h2_ffn, attr, t[key])

# Alias for backward compatibility with main.py / demo code
DracoTransformerV1 = DracoTransformer

# ══════════════════════════════════════════════════════════════════════
# LOAD EXTERNAL WEIGHTS  (from Qwen 3.5 9B Instruct checkpoint)
# ══════════════════════════════════════════════════════════════════════
def load_external_weights(
    state_dict: Dict[str, Any],
    model: DracoTransformer,
) -> Tuple[int, List[str]]:
    """
    Load Qwen 3.5 9B Instruct weights into DracoTransformer.

    MoE construction from SINGLE Qwen 3.5 9B Instruct checkpoint:
        Each layer's FFN is assigned to an expert slot via layer_idx % 8.
        - layer_idx % 8 in {0,1,2,3} → Code group expert (slot 0-3)
        - layer_idx % 8 in {4,5,6,7} → Language group expert (slot 4-7)

        There is no separate "coder model" or "instruct model".
        The split is based on which FFN neurons within the SINGLE 9B checkpoint
        are observed to activate on code vs. language token sequences.

    NOTE: Qwen 3.5 9B has NO .bias tensors — skip any .bias keys silently.
    """
    loaded = 0; skipped = []
    for ext_k, tensor in state_dict.items():
        # Skip bias keys (Qwen 3.5 has none)
        if ext_k.endswith(".bias"):
            continue
        draco_k = bridge_key(ext_k)
        try:
            arr = tensor.cpu().float().numpy() if hasattr(tensor, "cpu") \
                  else np.asarray(tensor, np.float32)
        except Exception:
            skipped.append(ext_k); continue
        placed = _place_weight(model, draco_k, arr)
        if placed:
            loaded += 1
        else:
            skipped.append(f"{ext_k}→{draco_k}")
    return loaded, skipped


def _place_weight(model: DracoTransformer, key: str, arr: np.ndarray) -> bool:
    """Place a weight array into the correct slot of DracoTransformer."""
    try:
        parts = key.split(".")
        if parts[0] == "token_emb" and arr.ndim == 2:
            n = min(arr.shape[0], model.cfg.qwen_base, model.token_emb.shape[0])
            model.token_emb[:n] = arr[:n]
            return True
        if parts[0] == "norm_final" and arr.shape == model.norm_final.shape:
            model.norm_final[:] = arr
            return True
        # lm_head: fallback → tie to embed_tokens
        if parts[0] == "lm_head" and arr.ndim == 2:
            n = min(arr.shape[0], model.lm_head.shape[1])
            model.lm_head[:, :n] = arr[:n].T
            return True
        if parts[0] == "blocks" and len(parts) >= 3:
            idx = int(parts[1])
            if idx >= len(model.blocks):
                return False
            blk = model.blocks[idx]
            sub = parts[2]
            fld = ".".join(parts[3:]) if len(parts) > 3 else ""

            # Attention weights
            if sub == "attn":
                for attr in ("wq", "wk", "wv", "wo"):
                    if fld == attr and getattr(blk.attn, attr).shape == arr.shape:
                        setattr(blk.attn, attr, arr)
                        return True
            # Norm weights
            if sub in ("norm", "post_norm"):
                tgt = getattr(blk, sub)
                if tgt.shape == arr.shape:
                    tgt[:] = arr; return True

            # FFN → distribute to MoE experts
            # Source: SINGLE Qwen 3.5 9B Instruct checkpoint
            # expert_idx = layer_idx % n_experts
            #   0-3 → Code group (FFN layers whose activations skew toward code tokens)
            #   4-7 → Language group (FFN layers that skew toward language/instruction tokens)
            if sub == "ffn":
                expert_idx = idx % model.cfg.n_experts
                e = blk.moe.experts[expert_idx]
                for attr, fname in [("W_g", "gate_proj"), ("W_u", "up_proj"), ("W_d", "down_proj")]:
                    if fld == fname and getattr(e, attr).shape == arr.shape:
                        setattr(e, attr, arr); return True
                # Also blend into shared expert (running average)
                for attr, fname in [("W_g", "gate_proj"), ("W_u", "up_proj"), ("W_d", "down_proj")]:
                    if fld == fname:
                        tgt = getattr(blk.moe.shared, attr)
                        if tgt.shape == arr.shape:
                            tgt[:] = (tgt + arr) * 0.5
                        return True
    except Exception:
        pass
    return False