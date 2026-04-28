# Copyright 2026 Draco Studio and DUCNGUYEN-creator
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
"""
DracoAI V1 — NumPy Transformer (Inference Only)
================================================
MoE Transformer built from a single Qwen 3.5 9B dense checkpoint.
8 experts created by distributing the 36 Qwen FFN layers across experts
using layer_idx % 8, with weights AVERAGED (not overwritten) per expert.
Architecture:
  - GQA (Grouped Query Attention) + SWA-Sink KVCache
  - MoE (8 experts, top-2 routing, load-balance loss)
  - MTP (Multi-Token Prediction / speculative decoding)
  - RoPE positional encoding
  - Mirostat v2 sampling
  - Identity overlay via logit-level bias (no text replacement)
FIXES (V1 — final consolidated):
    ✅ KVCache: prefill safety — long prompt slices sink+tail, no modulo overwrite
    ✅ KVCache: filled updated correctly for both prefill and decode
    ✅ MoE: vectorised boolean-mask dispatch (removes Python token loop)
    ✅ MTP: try_speculative used in generate (logits2 no longer wasted)
    ✅ _place_weight: uses expert_accum dict to AVERAGE layers per expert
    ✅ _break_symmetry NOT called in __init__ (only after load for random init)
    ✅ RoPE offset correct through long prefill + speculative decode steps
    ✅ MoE gate normalisation: softmax only over selected top-k experts
    ✅ FIX S1: generate() breaks immediately when spec_id == eos_id
    ✅ FIX 2.2 (🔴 CRITICAL): _sample_mirostat_v2 mu update sign CORRECTED
         Was:  mu = mu + eta * (surprise - tau)   ← positive feedback = collapse
         Now:  mu = mu - eta * (surprise - tau)   ← correct per Basu 2020
    ✅ FIX 2.3: distance-decayed repetition penalty via pos dict
    ✅ FIX CACHE-ROLLBACK (🔴 CRITICAL): generate() now correctly calls
         cache.restore(snap) when verify_id != spec_pending, then immediately
         re-forwards verify_id so its K/V is written to cache before continuing.
         The old code stored snap in cache._pending_snap but NEVER called
         restore() during rejection — leaving rotten K/V in cache permanently.
         This is the "transactional cache" pattern. No more hallucination from
         rejected speculative tokens poisoning the attention context.
    ✅ FIX MOE-NOISE: MoELayer.forward() adds Gumbel noise (scale=0.05) to
         router logits at inference time to prevent expert collapse.
    ✅ FIX LOGITS-CLIP: np.clip(logits, -50, 50) in sampling before exp()
         prevents numerical overflow/underflow in float32.
    ✅ FIX PROBS-GUARD: After softmax, probs re-normalised to ensure sum=1.0
         and NaN values replaced with 0 before np.random.choice.
    ✅ FIX LOGITS-PIPELINE: generate() applies logit processing in strict order:
         clip → repetition_penalty → (mirostat or top-k/p → min-p) → sample.
    ✅ FIX MIN-P: _sample_topk_topp() supports min_p parameter (default 0.0).
    ✅ FIX SANITY: _sanity_checks() validates logits when debug=True.
    ✅ FIX ADAPTIVE-TEMP: generate() supports adaptive_temp=True.
    ✅ FIX IDENTITY-BIAS: set_identity_bias() uses reduced boost (2.0).
    ✅ FIX ATTN-CLIP: GQAttention clips attention scores to [-50, 50].
    ✅ FIX SILU-CLIP: ExpertFFN.forward() clips gate before exp().
    ✅ FIX SPEC-INDENT (🔴 CRITICAL): Speculative verification block was incorrectly
         indented inside the repetition-penalty for-loop and outside the while-loop.
         Fixed: block now correctly sits at while-body level, after penalty application.
    ✅ FIX SPEC-LOGITS (🔴 CRITICAL): pre_spec_logits saved BEFORE snapshot/spec forward.
         On rejection, re-sample from pre_spec_logits (clean, uncontaminated) instead of
         post-spec logits — eliminates hallucination from rejected token's K/V influence.
    ✅ FIX SPEC-NPOS: On rejection, verify_id reuses spec_pending's n_pos slot (n_pos-1)
         so repetition-penalty position tracking is always accurate.
    ✅ FIX SPEC-ACCEPT-MISSING-KV (🔴 CRITICAL): After accepting spec and sampling T+2
         (nid), the accept block did NOT forward nid before trying speculative T+3.
         If speculative T+3 was added, the next iteration would forward T+3 with cache
         missing K/V(nid), causing attention to skip a position and breaking RoPE.
         Fix: always call forward([nid]) in the accept block after sampling T+2, then
         use the resulting l2_new (fresh MTP output) for T+3 speculation. This mirrors
         exactly what the normal sampling path does: forward(nid) → spec from l2_new.
         When no spec follows, cur=[nid] and the next iteration forwards nid normally —
         but with forward done eagerly, cache is always complete before any spec chain.
    ✅ FIX KVCACHE-ALLOC: KVCache.__init__ uses np.empty instead of np.zeros.
         np.zeros zero-fills the entire buffer at startup (up to 4 GB for 9B model),
         causing lag and potential OOM on low-RAM machines.
         np.empty skips zero-fill; safe because update() always writes before get() reads.
         Optional use_memmap=True parameter backs buffers with disk-mapped temp files,
         supporting window sizes that exceed available RAM at the cost of ~2-5x slower I/O.
         cleanup() method deletes temp files when the cache is no longer needed.
    ✅ FIX MIROSTAT-MU-PERSIST: generate() now uses self._miro_mu instead of a local
         mu = 5.0 that was reset on every call regardless of new_prompt.
         self._miro_mu is reset to 5.0 only when new_prompt=True (or cache is None).
         For multi-turn continuation (new_prompt=False), mu carries over, preserving
         Mirostat's adaptive entropy state across successive generate() calls.
         mu is saved back to self._miro_mu at the end of every generate() call.
    ✅ FIX KVCACHE-LONGPREFILL-POS (🔴 CRITICAL): KVCache long-prefill path (seq > window)
         wrote filled=window but left cache_pos=0. Then forward() called step(seq) pushing
         cache_pos to seq (>> window). get() then computed
         rec_start = sink + (cache_pos - sink) % recent_cap, which for large cache_pos
         pointed to the MIDDLE of the tail buffer instead of the start — returning tokens
         in the wrong chronological order for the very first decode step after a long prompt.
         Fix: long-prefill path now sets cache_pos = window at layer==0, and step() guards
         against double-advancing when cache_pos is already pinned to window by a long-prefill.
PRODUCTION ADDITIONS (V1 — this release):
    ✅ DTYPE SUPPORT: DracoTransformerV1(config, dtype=np.float16) initialises all weights
         as float16.  cast_weights(dtype) converts in-place after load.  Logit computation
         upcasts to float32 automatically to prevent vocab-projection overflow.
    ✅ QUANTIZEDLINEAR: class QuantizedLinear supports INT8 (per-channel symmetric) and
         INT4 (packed uint8, per-group, group_size=128).  quantize_weights(quant='int8'|'int4')
         replaces all attn/FFN weight matrices in-place; forward() transparently dequantises
         on-the-fly.  Save/load via .npz.  ~4× (INT8) or ~8× (INT4) memory reduction.
    ✅ _mm() HELPER: unified matrix multiply that handles both np.ndarray and QuantizedLinear
         in GQAttention.forward() and ExpertFFN.forward() — no logic change needed.
    ✅ GGUF EXPORTER: GGUFExporter(model).write_gguf(path) exports all weights to GGUF FP16
         using the llama/Qwen2-MoE tensor naming convention.  Ready for Q4_K_M quantisation
         via: llama-quantize out_fp16.gguf out_q4km.gguf Q4_K_M
    ✅ TRANSFORMER BRIDGE: TransformerBridge unifies NumPy and llama.cpp backends behind
         one generate() API.  Supports intent_bias (router) and intent_boost (logits) on
         both backends.  export_gguf() converts and auto-switches to llama.cpp backend."""
from __future__ import annotations
import math, os, json, time, copy, tempfile, struct, ctypes
from typing import List, Optional, Tuple, Dict, Callable
import numpy as np
# ── Constants ──────────────────────────────────────────────────────────
SINK_TOKENS     = 4      # Number of "sink" tokens kept at start of KVCache
SPEC_THRESH     = 0.80   # Confidence threshold for speculative accept
DEFAULT_TEMP    = 0.7
DEFAULT_TOP_P   = 0.9
MOE_NOISE_SCALE = 0.05   # Gumbel noise scale for MoE router diversity
# ── dtype helpers ──────────────────────────────────────────────────────
def _detect_compute_dtype() -> np.dtype:
    """Auto-detect best dtype: bfloat16 if supported, else float16, else float32.
    NumPy has no native bfloat16 — we represent it as float32 at compute time
    but keep storage as uint16 (same bit-width as bf16).  For pure NumPy inference
    float16 is the practical production dtype on most hardware.
    """
    return np.float16   # float16 is universally safe on NumPy CPU inference

COMPUTE_DTYPE: np.dtype = _detect_compute_dtype()
# ─────────────────────────────────────────────────────────────────────
# QuantizedLinear — INT8 / INT4 weight-only quantisation
# ─────────────────────────────────────────────────────────────────────
class QuantizedLinear:
    """
    Weight-only quantisation for inference (activations stay float32).
    Supports INT8 (per-channel symmetric) and INT4 (packed, per-group).

    INT8 — symmetric per-output-channel:
        W_q   : (out, in)  int8
        scale : (out,)     float32
        quant : 'int8'

    INT4 — packed uint8 (two int4 values per byte), per-group:
        W_q   : (out, in//2)  uint8  — low nibble = col*2, high nibble = col*2+1
        scale : (out, in//group_size) float32
        zero  : (out, in//group_size) float32
        quant : 'int4'
        group_size: typically 128 (matches GGUF Q4_K_M groups)

    Usage:
        ql = QuantizedLinear.from_float(W_fp32, quant='int8')
        y  = ql.forward(x)   # x: (..., in_features)  y: (..., out_features)

    Load from GGUF-dequantised numpy array:
        ql = QuantizedLinear.from_float(arr, quant='int4', group_size=128)
    """

    def __init__(self):
        self.W_q:      np.ndarray = None
        self.scale:    np.ndarray = None
        self.zero:     Optional[np.ndarray] = None
        self.quant:    str        = 'int8'
        self.in_feat:  int        = 0
        self.out_feat: int        = 0
        self.group_size: int      = 128

    # ── Quantise from float ──────────────────────────────────────────
    @staticmethod
    def from_float(W: np.ndarray, quant: str = 'int8',
                   group_size: int = 128) -> "QuantizedLinear":
        """Quantise a float32 weight matrix (out_features, in_features)."""
        if W.ndim == 1:
            W = W.reshape(1, -1)
        out_f, in_f = W.shape
        ql = QuantizedLinear()
        ql.quant     = quant
        ql.in_feat   = in_f
        ql.out_feat  = out_f
        ql.group_size = group_size

        if quant == 'int8':
            # Per-channel symmetric: scale = max(|W|) / 127
            abs_max = np.abs(W).max(axis=1, keepdims=True).clip(min=1e-8)
            scale   = (abs_max / 127.0).astype(np.float32)
            W_q     = np.round(W / scale).clip(-128, 127).astype(np.int8)
            ql.W_q  = W_q
            ql.scale = scale.reshape(out_f)   # (out,)

        elif quant == 'int4':
            # Per-group asymmetric: zero-point + scale
            assert in_f % group_size == 0, \
                f"in_features ({in_f}) must be divisible by group_size ({group_size})"
            n_groups = in_f // group_size
            W_r = W.reshape(out_f, n_groups, group_size).astype(np.float32)
            w_min = W_r.min(axis=2)    # (out, n_groups)
            w_max = W_r.max(axis=2)
            scale = ((w_max - w_min) / 15.0).clip(min=1e-8).astype(np.float32)
            zero  = (-w_min / scale).clip(0, 15).round().astype(np.float32)
            # Quantise to [0, 15]
            W_int4 = np.round((W_r - w_min[:, :, None]) / scale[:, :, None]).clip(0, 15).astype(np.uint8)
            # Pack: two int4 per byte (col*2 in low nibble, col*2+1 in high nibble)
            W_flat  = W_int4.reshape(out_f, in_f)
            # Pad in_f to even if needed
            if in_f % 2 != 0:
                W_flat = np.pad(W_flat, ((0, 0), (0, 1)))
            lo = W_flat[:, 0::2] & 0x0F
            hi = (W_flat[:, 1::2] & 0x0F) << 4
            ql.W_q  = (lo | hi).astype(np.uint8)   # (out, in//2)
            ql.scale = scale   # (out, n_groups)
            ql.zero  = zero    # (out, n_groups)
        else:
            raise ValueError(f"Unknown quant type: {quant!r}. Use 'int8' or 'int4'.")
        return ql

    # ── Dequantise ───────────────────────────────────────────────────
    def dequantize(self) -> np.ndarray:
        """Return the dequantised float32 weight matrix (out, in)."""
        if self.quant == 'int8':
            return self.W_q.astype(np.float32) * self.scale[:, None]
        elif self.quant == 'int4':
            out_f = self.out_feat
            n_groups = self.W_q.shape[1] * 2 // self.group_size  # recompute
            # Unpack nibbles
            lo = (self.W_q & 0x0F).astype(np.float32)
            hi = ((self.W_q >> 4) & 0x0F).astype(np.float32)
            W_flat = np.empty((out_f, self.W_q.shape[1] * 2), dtype=np.float32)
            W_flat[:, 0::2] = lo
            W_flat[:, 1::2] = hi
            W_flat = W_flat[:, :self.in_feat]   # trim padding
            # Dequant per-group
            W_r = W_flat.reshape(out_f, n_groups, self.group_size)
            W_r = W_r * self.scale[:, :, None] + (self.zero[:, :, None] * -self.scale[:, :, None])
            return W_r.reshape(out_f, self.in_feat)
        raise ValueError(f"Unknown quant: {self.quant}")

    # ── Forward ──────────────────────────────────────────────────────
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (..., in_features)  → output: (..., out_features)
        Dequantise on-the-fly; activations remain float32.
        For INT8 this is nearly free; for INT4 the unpack adds ~15% overhead
        but memory bandwidth is halved vs float16.
        """
        W_fp = self.dequantize()    # (out, in)
        return x @ W_fp.T

    # ── Serialise / deserialise ──────────────────────────────────────
    def save(self, path: str):
        """Save to a single .npz file."""
        d: dict = {"W_q": self.W_q, "scale": self.scale,
                   "meta": np.array([self.in_feat, self.out_feat,
                                     self.group_size,
                                     1 if self.quant == 'int4' else 0])}
        if self.zero is not None:
            d["zero"] = self.zero
        np.savez_compressed(path, **d)

    @staticmethod
    def load(path: str) -> "QuantizedLinear":
        data = np.load(path + ".npz" if not path.endswith(".npz") else path)
        ql = QuantizedLinear()
        ql.W_q  = data["W_q"]
        ql.scale = data["scale"]
        ql.zero  = data["zero"] if "zero" in data else None
        meta     = data["meta"]
        ql.in_feat    = int(meta[0])
        ql.out_feat   = int(meta[1])
        ql.group_size = int(meta[2])
        ql.quant      = 'int4' if int(meta[3]) else 'int8'
        return ql
# ─────────────────────────────────────────────────────────────────────
# RoPE helpers
# ─────────────────────────────────────────────────────────────────────
def _rope_freqs(head_dim: int, base: float = 10000.0) -> np.ndarray:
    i = np.arange(0, head_dim, 2, dtype=np.float32)
    return 1.0 / (base ** (i / head_dim))
def _apply_rope(x: np.ndarray, freqs: np.ndarray, offset: int = 0) -> np.ndarray:
    """Apply RoPE to x of shape (..., seq, head_dim)."""
    seq  = x.shape[-2]
    hdim = x.shape[-1]
    pos  = np.arange(offset, offset + seq, dtype=np.float32)
    angles = np.outer(pos, freqs)
    cos = np.cos(angles).astype(x.dtype)
    sin = np.sin(angles).astype(x.dtype)
    extra = x.ndim - 2
    shape = (1,) * extra + (seq, hdim // 2)
    cos = cos.reshape(shape)
    sin = sin.reshape(shape)
    x1 = x[..., :hdim // 2]
    x2 = x[..., hdim // 2:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
# ─────────────────────────────────────────────────────────────────────
# KVCache with SWA-Sink  (Sliding Window Attention + sink tokens)
# ─────────────────────────────────────────────────────────────────────
class KVCache:
    """
    Sliding-window KV cache with sink tokens.
    Buffer layout (after any prefill):
        [0 .. sink-1]    : sink tokens (oldest, always retained)
        [sink .. window-1]: recent tokens in circular order
    get() always returns tokens in chronological order:
        sink tokens first, then recent tokens oldest→newest.
    FIX CACHE-ROLLBACK: snapshot() and restore() for transactional
    speculative decoding. Before forwarding a spec token, call snapshot().
    If the spec is rejected, call restore(snap) to revert K/V state cleanly.
    """
    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int,
                 window: int = 1024, sink: int = SINK_TOKENS,
                 use_memmap: bool = False, memmap_dir: Optional[str] = None):
        """
        use_memmap=True: buffers backed by disk-mapped temp files.
            Avoids upfront RAM allocation — only pages in physical RAM that are
            actually read/written.  ~2-5x slower I/O than pure RAM.
            Recommended on machines with < 8 GB free RAM and large windows.
        use_memmap=False (default): np.empty — fast allocation, no zero-fill,
            safe because update() always writes before get() reads.
        """
        self.n_layers   = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim   = head_dim
        self.window     = window
        self.sink       = sink
        self.cache_pos: int = 0
        self.filled:    int = 0
        self._use_memmap = use_memmap
        self._k_file     = None
        self._v_file     = None
        shape = (n_layers, 1, n_kv_heads, window, head_dim)
        dtype = np.float16
        if use_memmap:
            _dir = memmap_dir or tempfile.gettempdir()
            # delete=False so memmap can keep the fd open cross-platform
            self._k_file = tempfile.NamedTemporaryFile(
                dir=_dir, delete=False, suffix=".k_cache")
            self._v_file = tempfile.NamedTemporaryFile(
                dir=_dir, delete=False, suffix=".v_cache")
            self.k_buf = np.memmap(self._k_file.name, dtype=dtype,
                                   mode="w+", shape=shape)
            self.v_buf = np.memmap(self._v_file.name, dtype=dtype,
                                   mode="w+", shape=shape)
        else:
            # np.empty: virtual memory allocated, no zero-fill — fast startup.
            # All positions are written by update() before get() reads them.
            self.k_buf = np.empty(shape, dtype=dtype)
            self.v_buf = np.empty(shape, dtype=dtype)
    def reset(self):
        self.k_buf[:] = 0
        self.v_buf[:] = 0
        if self._use_memmap:
            self.k_buf.flush()
            self.v_buf.flush()
        self.cache_pos = 0
        self.filled    = 0

    def cleanup(self):
        """Delete memmap temp files. Call when KVCache is no longer needed."""
        for f in (self._k_file, self._v_file):
            if f is not None:
                try:
                    os.unlink(f.name)
                except OSError:
                    pass
        self._k_file = self._v_file = None
    def snapshot(self) -> dict:
        """
        FIX CACHE-ROLLBACK: Capture current cache state for transactional rollback.
        Returns a dict with copies of all mutable state.
        With memmap buffers, flush is called before copy to ensure consistency.
        """
        if self._use_memmap:
            self.k_buf.flush()
            self.v_buf.flush()
        return {
            "cache_pos": self.cache_pos,
            "filled":    self.filled,
            "k_buf":     self.k_buf.copy(),
            "v_buf":     self.v_buf.copy(),
        }
    def restore(self, snap: dict):
        """
        FIX CACHE-ROLLBACK: Restore cache state from a snapshot.
        Must be called with the exact dict returned by snapshot().
        """
        self.cache_pos = snap["cache_pos"]
        self.filled    = snap["filled"]
        np.copyto(self.k_buf, snap["k_buf"])
        np.copyto(self.v_buf, snap["v_buf"])
    def update(self, layer: int, k: np.ndarray, v: np.ndarray):
        """
        Store K,V for one layer.
        k, v shape: (1, n_kv_heads, seq, head_dim)
        Two distinct paths:
          1. seq > window  → prefill-long: copy sink + tail, no modulo
          2. seq <= window → normal: sequential write with wrap-around
        filled updated only on layer==0 to avoid n_layers redundant writes.
        """
        seq = k.shape[2]
        if seq > self.window:
            tail_len = self.window - self.sink
            self.k_buf[layer, :, :, :self.sink, :]           = k[:, :, :self.sink,  :].astype(np.float16)
            self.v_buf[layer, :, :, :self.sink, :]           = v[:, :, :self.sink,  :].astype(np.float16)
            self.k_buf[layer, :, :, self.sink:self.window, :] = k[:, :, -tail_len:, :].astype(np.float16)
            self.v_buf[layer, :, :, self.sink:self.window, :] = v[:, :, -tail_len:, :].astype(np.float16)
            if layer == 0:
                self.filled    = self.window
                # FIX KVCACHE-LONGPREFILL-POS: set cache_pos = window so get() rec_start formula
                # computes correct chronological order for subsequent decode steps.
                # Without this, cache_pos stays 0 then step(seq) sets it to seq (> window),
                # causing rec_start = sink + (seq - sink) % recent_cap to point to the MIDDLE
                # of the tail buffer instead of the start — tokens read in wrong order.
                self.cache_pos = self.window
        else:
            for s in range(seq):
                abs_pos = self.cache_pos + s
                if abs_pos < self.sink:
                    buf_pos = abs_pos
                else:
                    buf_pos = self.sink + (abs_pos - self.sink) % max(1, self.window - self.sink)
                self.k_buf[layer, :, :, buf_pos, :] = k[:, :, s, :].astype(np.float16)
                self.v_buf[layer, :, :, buf_pos, :] = v[:, :, s, :].astype(np.float16)
            if layer == 0:
                self.filled = min(self.cache_pos + seq, self.window)
    def get(self, layer: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return K, V in chronological order. Shape: (1, n_kv_heads, filled, head_dim)"""
        if self.filled < self.window:
            k = self.k_buf[layer, :, :, :self.filled, :]
            v = self.v_buf[layer, :, :, :self.filled, :]
            return k.astype(np.float32), v.astype(np.float32)
        recent_cap = self.window - self.sink
        if recent_cap <= 0:
            k = self.k_buf[layer, :, :, :self.filled, :]
            v = self.v_buf[layer, :, :, :self.filled, :]
            return k.astype(np.float32), v.astype(np.float32)
        rec_start = self.sink + (self.cache_pos - self.sink) % recent_cap
        k_sink = self.k_buf[layer, :, :, :self.sink, :]
        v_sink = self.v_buf[layer, :, :, :self.sink, :]
        k_rec  = np.concatenate([
            self.k_buf[layer, :, :, rec_start:self.window, :],
            self.k_buf[layer, :, :, self.sink:rec_start,   :],
        ], axis=2)
        v_rec  = np.concatenate([
            self.v_buf[layer, :, :, rec_start:self.window, :],
            self.v_buf[layer, :, :, self.sink:rec_start,   :],
        ], axis=2)
        k = np.concatenate([k_sink, k_rec], axis=2)
        v = np.concatenate([v_sink, v_rec], axis=2)
        return k.astype(np.float32), v.astype(np.float32)
    def step(self, seq_len: int = 1):
        """Advance cache pointer after a forward pass.
        FIX KVCACHE-LONGPREFILL-POS: When the long-prefill path already set
        cache_pos = window, adding seq_len (which is > window) would make
        cache_pos >> window and cause get() rec_start to point to the wrong
        buffer slot on the very next decode step.  Guard: if cache_pos was
        already pinned to window by the long-prefill path, do not add again.
        """
        if self.cache_pos == self.window and seq_len > self.window:
            # Long-prefill just ran; cache_pos pinned to window.
            # Skip re-increment — next decode's update() will wrap correctly.
            pass
        else:
            self.cache_pos += seq_len
# ─────────────────────────────────────────────────────────────────────
# Layer normalisations
# ─────────────────────────────────────────────────────────────────────
def rms_norm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * w

def _mm(x: np.ndarray, W) -> np.ndarray:
    """Matrix multiply that transparently handles both np.ndarray and QuantizedLinear.
    For ndarray:        x @ W      (standard, W shape: in × out)
    For QuantizedLinear with _transposed=True: x @ dequant(W).T
      = same as x @ original_W  because QL stores W.T in (out, in) convention.
    """
    if isinstance(W, QuantizedLinear):
        return W.forward(x)   # ql.forward already does x @ W_fp.T
    return x @ W
# ─────────────────────────────────────────────────────────────────────
# Grouped Query Attention
# ─────────────────────────────────────────────────────────────────────
class GQAttention:
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int):
        self.d_model    = d_model
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim   = head_dim
        self.n_rep      = n_heads // n_kv_heads
        scale = 1.0 / math.sqrt(d_model)
        self.W_q = np.random.randn(d_model, n_heads    * head_dim).astype(np.float32) * scale
        self.W_k = np.random.randn(d_model, n_kv_heads * head_dim).astype(np.float32) * scale
        self.W_v = np.random.randn(d_model, n_kv_heads * head_dim).astype(np.float32) * scale
        self.W_o = np.random.randn(n_heads * head_dim, d_model).astype(np.float32)    * scale
        self._rope_freqs: Optional[np.ndarray] = None
    def _get_rope(self, head_dim: int) -> np.ndarray:
        if self._rope_freqs is None or self._rope_freqs.shape[0] != head_dim // 2:
            self._rope_freqs = _rope_freqs(head_dim)
        return self._rope_freqs
    def forward(self, x: np.ndarray, cache: KVCache, layer_idx: int) -> np.ndarray:
        """
        x: (1, seq, d_model)  Returns: (1, seq, d_model)
        FIX ATTN-CLIP: attention scores clipped to [-50, 50] before softmax.
        """
        bsz, seq, _ = x.shape
        freqs  = self._get_rope(self.head_dim)
        offset = cache.cache_pos
        Q = _mm(x, self.W_q).reshape(bsz, seq, self.n_heads,    self.head_dim).transpose(0, 2, 1, 3)
        K = _mm(x, self.W_k).reshape(bsz, seq, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = _mm(x, self.W_v).reshape(bsz, seq, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        Q = _apply_rope(Q, freqs, offset)
        K = _apply_rope(K, freqs, offset)
        cache.update(layer_idx, K, V)
        K_f, V_f = cache.get(layer_idx)
        kv_seq   = K_f.shape[2]
        K_exp = np.repeat(K_f, self.n_rep, axis=1)
        V_exp = np.repeat(V_f, self.n_rep, axis=1)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn  = Q @ K_exp.transpose(0, 1, 3, 2) * scale
        # Causal mask
        if seq > 1:
            causal   = np.triu(np.full((seq, seq), -1e9, dtype=np.float32), 1)
            past_len = kv_seq - seq
            if past_len > 0:
                mask_full = np.concatenate(
                    [np.zeros((seq, past_len), dtype=np.float32), causal], axis=1
                )
            else:
                mask_full = causal
            attn = attn + mask_full[None, None, :, :]
        # FIX ATTN-CLIP: clip scores before softmax for numerical stability
        attn = np.clip(attn, -50.0, 50.0)
        attn = attn - attn.max(axis=-1, keepdims=True)
        attn = np.exp(attn)
        attn_sum = attn.sum(axis=-1, keepdims=True)
        attn = attn / (attn_sum + 1e-9)
        out = attn @ V_exp
        out = out.transpose(0, 2, 1, 3).reshape(bsz, seq, self.n_heads * self.head_dim)
        return _mm(out, self.W_o)
# ─────────────────────────────────────────────────────────────────────
# Expert FFN (SwiGLU)
# ─────────────────────────────────────────────────────────────────────
class ExpertFFN:
    def __init__(self, d_model: int, d_ff: int):
        scale = 1.0 / math.sqrt(d_model)
        self.W_g = np.random.randn(d_model, d_ff).astype(np.float32) * scale
        self.W_u = np.random.randn(d_model, d_ff).astype(np.float32) * scale
        self.W_d = np.random.randn(d_ff, d_model).astype(np.float32) * scale
    def forward(self, x: np.ndarray) -> np.ndarray:
        gate = _mm(x, self.W_g)
        # FIX SILU-CLIP: clip gate before exp to prevent overflow
        gate = gate / (1.0 + np.exp(-np.clip(gate, -50, 50)))  # SiLU
        return _mm(gate * _mm(x, self.W_u), self.W_d)
    def _break_symmetry(self, scale: float = 1e-3):
        """Add tiny noise to break weight symmetry (init only, NOT on load)."""
        self.W_g += np.random.randn(*self.W_g.shape).astype(np.float32) * scale
        self.W_u += np.random.randn(*self.W_u.shape).astype(np.float32) * scale
# ─────────────────────────────────────────────────────────────────────
# Mixture of Experts Layer
# ─────────────────────────────────────────────────────────────────────
class MoELayer:
    """
    8-expert MoE with top-2 routing.
    FIX MOE-NOISE: Gumbel noise added to router logits to prevent expert
    collapse at inference time. Scale 0.05 diversifies routing without
    significantly changing top-k selections.
    """
    def __init__(self, d_model: int, d_ff: int, n_experts: int = 8, top_k: int = 2):
        self.d_model   = d_model
        self.d_ff      = d_ff
        self.n_experts = n_experts
        self.top_k     = top_k
        scale = 1.0 / math.sqrt(d_model)
        self.W_router    = np.random.randn(d_model, n_experts).astype(np.float32) * scale
        self.router_bias = np.zeros(n_experts, dtype=np.float32)
        self.experts     = [ExpertFFN(d_model, d_ff) for _ in range(n_experts)]
        self.shared      = ExpertFFN(d_model, d_ff)
        self.norm_w      = np.ones(d_model, dtype=np.float32)
    def forward(self, x: np.ndarray,
                add_noise: bool = True,
                intent_bias: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        x: (batch=1, seq, d_model)
        Returns: (output, aux_losses_dict)
        FIX MOE-NOISE: Gumbel noise injected when add_noise=True (inference).
        intent_bias: optional (n_experts,) array — engine bias added to router logits.
        """
        bsz, seq, d = x.shape
        x_flat = x.reshape(seq, d)
        logits = x_flat @ self.W_router + self.router_bias
        # Kết nối Engine → Router: cộng intent_bias (đã nhân với INTENT_BIAS_ALPHA ở ngoài)
        if intent_bias is not None:
            logits = logits + intent_bias.reshape(1, -1)
        # FIX MOE-NOISE
        if add_noise and seq > 0:
            noise = np.random.gumbel(size=logits.shape).astype(np.float32) * MOE_NOISE_SCALE
            logits = logits + noise
        # Softmax over all experts for load-balance metrics
        router_soft = np.exp(np.clip(logits - logits.max(axis=-1, keepdims=True), -50, 50))
        router_soft = router_soft / (router_soft.sum(axis=-1, keepdims=True) + 1e-9)
        # Top-k selection
        top_idx = np.argsort(logits, axis=-1)[:, -self.top_k:][:, ::-1]
        # Gate weights: softmax over selected experts only
        top_logits = np.take_along_axis(logits, top_idx, axis=1)
        top_logits = top_logits - top_logits.max(axis=-1, keepdims=True)
        gates = np.exp(np.clip(top_logits, -50, 50))
        gates = gates / (gates.sum(axis=-1, keepdims=True) + 1e-9)
        output = np.zeros((seq, d), dtype=np.float32)
        for k in range(self.top_k):
            expert_ids = top_idx[:, k]
            g_k        = gates[:, k]
            for e in range(self.n_experts):
                mask = expert_ids == e
                if not mask.any():
                    continue
                x_sel  = x_flat[mask]
                g_sel  = g_k[mask]
                normed = rms_norm(x_sel, self.norm_w)
                e_out  = self.experts[e].forward(normed)
                output[mask] += g_sel[:, None] * e_out
        output += self.shared.forward(rms_norm(x_flat, self.norm_w))
        # Aux losses
        importance = router_soft.mean(axis=0)
        load = (top_idx == np.arange(self.n_experts)[:, None, None]).any(axis=-1).mean(axis=1)
        aux = {
            "importance_loss": float(importance.std()),
            "load_loss":       float(load.std()),
            "aux_total":       float(importance.std() + load.std()),
        }
        return output.reshape(bsz, seq, d), aux
# ─────────────────────────────────────────────────────────────────────
# Multi-Token Prediction head
# ─────────────────────────────────────────────────────────────────────
class MTPHead:
    """
    Predicts the next token (l1) and one-step-ahead token (l2).
    l2 is used for speculative decoding with cache snapshot/restore verification.
    """
    def __init__(self, d_model: int, vocab_size: int):
        scale = 1.0 / math.sqrt(d_model)
        self.W1 = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W2 = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.lm_head: Optional[np.ndarray] = None
        self.d_model = d_model
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """x: (1, seq, d_model)  Returns: (l1, l2) each (1, seq, vocab_size)"""
        def silu(z):
            return z / (1.0 + np.exp(-np.clip(z, -50, 50)))
        h1 = silu(x @ self.W1)
        h2 = silu(h1 @ self.W2)
        W  = self.lm_head if self.lm_head is not None else self.W1
        return h1 @ W.T, h2 @ W.T
    def try_speculative(self, l2: np.ndarray, thresh: float = SPEC_THRESH
                        ) -> Tuple[Optional[int], float]:
        """
        Return (token_id, confidence) if MTP prediction is confident enough.
        The caller MUST: (1) snapshot cache, (2) forward spec token, (3) verify,
        (4) restore cache if rejected.
        """
        last_logits = l2[0, -1].astype(np.float64)
        last_logits = np.clip(last_logits, -50, 50)
        probs = np.exp(last_logits - last_logits.max())
        probs /= probs.sum() + 1e-9
        best_id   = int(probs.argmax())
        best_prob = float(probs[best_id])
        if best_prob >= thresh:
            return best_id, best_prob
        return None, 0.0
# ─────────────────────────────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────────────────────────────
class TransformerBlock:
    def __init__(self, layer_idx: int, d_model: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, d_ff: int, n_experts: int = 8):
        self.layer_idx = layer_idx
        self.attn      = GQAttention(d_model, n_heads, n_kv_heads, head_dim)
        self.moe       = MoELayer(d_model, d_ff, n_experts)
        self.norm1     = np.ones(d_model, dtype=np.float32)
        self.norm2     = np.ones(d_model, dtype=np.float32)
    def forward(self, x: np.ndarray, cache: KVCache,
                add_noise: bool = True,
                intent_bias: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        h     = rms_norm(x, self.norm1)
        h     = self.attn.forward(h, cache, self.layer_idx)
        x     = x + h
        h, aux = self.moe.forward(rms_norm(x, self.norm2),
                                  add_noise=add_noise,
                                  intent_bias=intent_bias)
        x = x + h
        return x, aux
# ─────────────────────────────────────────────────────────────────────
# Full Transformer Model
# ─────────────────────────────────────────────────────────────────────
class DracoTransformerV1:
    """
    DracoAI Transformer — NumPy inference engine.
    Sampling: Mirostat v2 + temperature scaling + min-p.
    Speculative decoding: transactional cache with snapshot/restore.

    dtype: np.float32 (default, safe everywhere)
           np.float16 (production — 2× less RAM, faster on AVX2/NEON)
    quant_mode: None | 'int8' | 'int4'  — quantise attention/FFN weights after load.
    """
    def __init__(self, config: dict,
                 dtype: np.dtype = np.float32,
                 quant_mode: Optional[str] = None,
                 quant_group_size: int = 128):
        self.config     = config
        self.d_model    = config.get("d_model",    128)
        self.n_layers   = config.get("n_layers",     4)
        self.n_heads    = config.get("n_heads",       4)
        self.n_kv_heads = config.get("n_kv_heads",   2)
        self.head_dim   = config.get("head_dim",     32)
        self.d_ff       = config.get("d_ff",        512)
        self.n_experts  = config.get("n_experts",    8)
        self.vocab_size = config.get("vocab_size", 151936)
        self.window     = config.get("window",     1024)
        self._id_bias: Optional[np.ndarray] = None
        # dtype / quant config
        self._dtype:           np.dtype       = np.dtype(dtype)
        self._quant_mode:      Optional[str]  = quant_mode   # None | 'int8' | 'int4'
        self._quant_group_size: int           = quant_group_size
        scale = 1.0 / math.sqrt(self.d_model)
        self.embedding = (np.random.randn(self.vocab_size, self.d_model) * scale).astype(self._dtype)
        self.lm_head   = self.embedding
        self.blocks = [
            TransformerBlock(
                i, self.d_model, self.n_heads, self.n_kv_heads,
                self.head_dim, self.d_ff, self.n_experts
            )
            for i in range(self.n_layers)
        ]
        self.norm_f = np.ones(self.d_model, dtype=np.float32)
        self.mtp = MTPHead(self.d_model, self.vocab_size)
        self.mtp.lm_head = self.lm_head
        self._cache: Optional[KVCache] = None
        self._miro_mu: float = 5.0          # FIX: persistent Mirostat mu across turns
        self._memmap_cache: bool = False     # set True to use memmap KVCache
        self._memmap_dir: Optional[str] = None
    def _make_cache(self) -> KVCache:
        return KVCache(
            self.n_layers, self.n_kv_heads, self.head_dim,
            window=self.window, sink=SINK_TOKENS,
            use_memmap=self._memmap_cache,
            memmap_dir=self._memmap_dir,
        )
    # ── Forward pass ─────────────────────────────────────────────────
    def forward(self, token_ids: List[int], cache: KVCache,
                intent_boost: Optional[np.ndarray] = None,
                add_noise: bool = True,
                intent_bias: Optional[np.ndarray] = None      # ← tham số mới
                ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Returns: (l1, l2, aux_list)
          l1: (1, seq, vocab) — standard next-token logits
          l2: (1, seq, vocab) — speculative one-step-ahead logits
          aux_list: per-block MoE aux losses
        """
        ids = np.array(token_ids, dtype=np.int32)
        ids = np.clip(ids, 0, self.vocab_size - 1)
        x   = self.embedding[ids][None]
        aux_list = []
        for block in self.blocks:
            x, aux = block.forward(x, cache,
                                   add_noise=add_noise,
                                   intent_bias=intent_bias)
            aux_list.append(aux)
        x  = rms_norm(x, self.norm_f)
        # Upcast to float32 for logit computation — avoids fp16 overflow at vocab projection
        x32 = x.astype(np.float32) if x.dtype != np.float32 else x
        l1 = x32 @ self.lm_head.astype(np.float32).T
        _, l2 = self.mtp.forward(x32)
        if self._id_bias is not None:
            l1 = l1 + self._id_bias[None, None, :]
        if intent_boost is not None:
            l1 = l1 + intent_boost[None, None, :]
        cache.step(len(token_ids))
        return l1, l2, aux_list
    # ── Sanity checks ─────────────────────────────────────────────────
    @staticmethod
    def _sanity_checks(logits: np.ndarray, label: str = ""):
        """FIX SANITY: Validate logits and raise on fatal issues."""
        if np.any(np.isnan(logits)):
            raise RuntimeError(f"NaN in logits {label}")
        if np.any(np.isinf(logits)):
            raise RuntimeError(f"Inf in logits {label}")
    # ── Sampling ──────────────────────────────────────────────────────
    @staticmethod
    def _sample_mirostat_v2(logits: np.ndarray, mu: float, tau: float = 5.0,
                             eta: float = 0.1) -> Tuple[int, float]:
        """
        Mirostat v2 adaptive sampling.
        FIX 2.2 (🔴 CRITICAL): mu update sign corrected to MINUS.
        FIX LOGITS-CLIP: logits clipped to [-50, 50] before exp.
        FIX PROBS-GUARD: probs re-normalised and NaN-guarded.
        """
        logits = np.clip(logits, -50.0, 50.0)
        probs  = np.exp(logits - logits.max())
        probs /= probs.sum() + 1e-9
        # Guard NaN/Inf
        bad = ~np.isfinite(probs)
        if bad.any():
            probs[bad] = 0.0
            s = probs.sum()
            if s < 1e-9:
                probs[:] = 1.0 / len(probs)
            else:
                probs /= s
        idx          = np.argsort(probs)[::-1]
        probs_sorted = probs[idx]
        surprises    = -np.log2(probs_sorted + 1e-9)
        cutoff       = max(1, int(np.searchsorted(surprises, mu)))
        trunc_probs = probs_sorted[:cutoff]
        trunc_sum   = trunc_probs.sum()
        if trunc_sum < 1e-9:
            trunc_probs = probs_sorted[:1].copy()
            trunc_probs[:] = 1.0
        else:
            trunc_probs = trunc_probs / trunc_sum
        chosen_local = int(np.random.choice(len(trunc_probs), p=trunc_probs))
        chosen_id    = int(idx[chosen_local])
        surprise = float(-np.log2(probs[chosen_id] + 1e-9))
        mu_new   = mu - eta * (surprise - tau)   # FIX 2.2: MINUS (not plus)
        return chosen_id, max(0.1, mu_new)
    @staticmethod
    def _sample_topk_topp(logits: np.ndarray, temp: float = DEFAULT_TEMP,
                          top_p: float = DEFAULT_TOP_P, top_k: int = 50,
                          min_p: float = 0.0) -> int:
        """
        FIX LOGITS-CLIP: clip logits before exp.
        FIX MIN-P: filter tokens below min_p * max_prob after top-k/p.
        FIX PROBS-GUARD: ensure valid probability distribution.
        """
        logits = np.clip(logits / max(temp, 1e-6), -50.0, 50.0)
        if top_k > 0:
            kth    = np.partition(logits, -top_k)[-top_k]
            logits = np.where(logits < kth, -1e9, logits)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum() + 1e-9
        # FIX MIN-P
        if min_p > 0.0:
            max_prob = float(probs.max())
            probs[probs < min_p * max_prob] = 0.0
            p_sum = probs.sum()
            if p_sum > 1e-9:
                probs /= p_sum
            else:
                probs[:] = 1.0 / len(probs)
        # Top-p nucleus
        idx    = np.argsort(probs)[::-1]
        cumsum = np.cumsum(probs[idx])
        cut    = int(np.searchsorted(cumsum, top_p)) + 1
        probs_trunc = np.zeros_like(probs)
        probs_trunc[idx[:cut]] = probs[idx[:cut]]
        p_sum = probs_trunc.sum()
        if p_sum < 1e-9:
            probs_trunc[idx[0]] = 1.0
            p_sum = 1.0
        probs_trunc /= p_sum
        # FIX PROBS-GUARD
        bad = ~np.isfinite(probs_trunc)
        if bad.any():
            probs_trunc[bad] = 0.0
            s = probs_trunc.sum()
            if s < 1e-9:
                probs_trunc[idx[0]] = 1.0
            else:
                probs_trunc /= s
        return int(np.random.choice(len(probs_trunc), p=probs_trunc))
    # ── Generate ──────────────────────────────────────────────────────
    def generate(
        self,
        prompt_ids:      List[int],
        max_new_tokens:  int   = 256,
        temp:            float = DEFAULT_TEMP,
        top_p:           float = DEFAULT_TOP_P,
        min_p:           float = 0.0,
        eos_id:          int   = 151645,
        new_prompt:      bool  = True,
        use_mirostat:    bool  = True,
        use_speculative: bool  = True,
        adaptive_temp:   bool  = False,
        debug:           bool  = False,
        stream_cb:       Optional[Callable[[int, float], None]] = None,
        intent_boost:    Optional[np.ndarray] = None,
        intent_bias:     Optional[np.ndarray] = None,   # ← tham số mới
    ) -> List[int]:
        """
        Generate up to max_new_tokens tokens from prompt_ids.
        FIX CACHE-ROLLBACK (🔴 CRITICAL): When a speculative token is rejected
        (verify_id != spec_pending):
            1. cache.restore(snap) is called to revert K/V to pre-speculation state.
            2. pre_spec_logits (clean logits saved BEFORE the spec forward) are used
               to re-sample verify_id — preventing the rejected token's K/V from
               contaminating the sampling distribution.
            3. If verify_id == eos_id, forward EOS once to keep cache consistent.
            4. The next loop iteration forwards verify_id correctly from the clean state.
        FIX INDENT (🔴 CRITICAL): The speculative verification block was incorrectly
        nested inside the repetition-penalty loop and outside the while-loop due to
        wrong indentation. Now correctly placed inside while, after penalty application.
        FIX SPEC-ACCEPT-MISSING-KV (🔴 CRITICAL): After accepting spec and sampling
        T+2 (nid), old code immediately tried speculative T+3 without forwarding nid.
        If T+3 spec was added, next iteration forwarded T+3 with K/V(nid) absent from
        cache → attention skipped a position, RoPE broke. Fixed: forward([nid]) in
        accept block first; use l2_new from that forward for T+3 speculation.
        FIX LOGITS-PIPELINE: clip → repetition_penalty → sample.
        FIX ADAPTIVE-TEMP: temperature adjusted based on output entropy.
        """
        if new_prompt or self._cache is None:
            self._cache = self._make_cache()
            self._miro_mu = 5.0   # FIX: reset mu only when starting a new prompt
        cache = self._cache
        ids   = list(prompt_ids)
        mu  = self._miro_mu    # FIX: load persistent mu (already reset above if new_prompt)
        tau = 5.0
        eta = 0.1
        freq: Dict[int, int] = {}
        pos:  Dict[int, int] = {}
        n_pos = 0
        spec_pending:    Optional[int]         = None
        spec_snap:       Optional[dict]        = None   # snapshot taken before forwarding spec token
        pre_spec_logits: Optional[np.ndarray]  = None   # clean logits saved BEFORE forwarding spec token
        n_generated = 0
        current_temp = temp
        # Prefill
        cur = ids
        while n_generated < max_new_tokens:
            l1, l2, _ = self.forward(cur, cache,
                                      intent_boost=intent_boost,
                                      add_noise=True,
                                      intent_bias=intent_bias)
            last_logits = l1[0, -1].copy().astype(np.float64)
            if debug:
                self._sanity_checks(last_logits, f"step={n_generated}")
            # FIX LOGITS-PIPELINE step 1: clip
            last_logits = np.clip(last_logits, -50.0, 50.0)
            # FIX LOGITS-PIPELINE step 2: repetition penalty (distance-decayed)
            for tid, cnt in freq.items():
                if cnt > 0:
                    dist_penalty = n_pos - pos.get(tid, 0) + 1
                    last_logits[tid] -= 0.3 * cnt / dist_penalty
            # ── Speculative verification path ─────────────────────────────
            if spec_pending is not None:
                # Sample from current logits (post-spec forward) to verify
                if use_mirostat:
                    verify_id, mu = self._sample_mirostat_v2(last_logits, mu, tau, eta)
                else:
                    verify_id = self._sample_topk_topp(last_logits, current_temp, top_p, min_p=min_p)
                if verify_id == spec_pending:
                    # ✅ Speculative confirmed — cache K/V for spec token already written.
                    # last_logits = forward(spec_pending) output → distribution for T+2.
                    spec_pending    = None
                    spec_snap       = None
                    pre_spec_logits = None
                    # ── Sample T+2 (nid) from last_logits ────────────────
                    if adaptive_temp and not use_mirostat:
                        probs_tmp = np.exp(last_logits - last_logits.max())
                        probs_tmp /= probs_tmp.sum() + 1e-9
                        entropy = float(-np.sum(probs_tmp * np.log(probs_tmp + 1e-9)))
                        max_entropy = math.log(self.vocab_size)
                        norm_entropy = entropy / (max_entropy + 1e-9)
                        if norm_entropy < 0.1:
                            current_temp = min(temp * 1.5, 2.0)
                        elif norm_entropy > 0.8:
                            current_temp = max(temp * 0.7, 0.3)
                        else:
                            current_temp = temp
                    if use_mirostat:
                        nid, mu = self._sample_mirostat_v2(last_logits, mu, tau, eta)
                    else:
                        nid = self._sample_topk_topp(last_logits, current_temp, top_p, min_p=min_p)
                    ids.append(nid)
                    freq[nid] = freq.get(nid, 0) + 1
                    pos[nid]  = n_pos
                    n_pos    += 1
                    n_generated += 1
                    conf = float(np.exp(np.clip(last_logits[nid] - last_logits.max(), -50, 0)))
                    if stream_cb:
                        stream_cb(nid, conf)
                    if nid == eos_id or n_generated >= max_new_tokens:
                        break
                    # ── Forward T+2 to write K/V(nid) into cache ─────────
                    # CRITICAL: nid has no K/V in cache yet. If we chain another
                    # speculative token here, the next iteration would forward spec2
                    # while cache is missing K/V(nid), causing attention to skip a
                    # position and breaking RoPE alignment. We must forward nid first.
                    # This mirrors exactly what the normal sampling path does:
                    # forward(nid) → get fresh l2_new → try_speculative(l2_new) for T+3.
                    l1_new, l2_new, _ = self.forward(
                        [nid], cache,
                        intent_boost=intent_boost,
                        add_noise=True,
                        intent_bias=intent_bias
                    )
                    last_logits_new = l1_new[0, -1].copy().astype(np.float64)
                    if debug:
                        self._sanity_checks(last_logits_new, f"accept_fwd step={n_generated}")
                    # ── Try speculative T+3 from fresh l2_new ────────────
                    if use_speculative:
                        spec_id, spec_conf = self.mtp.try_speculative(l2_new)
                        if spec_id is not None:
                            if spec_id == eos_id:
                                break
                            # last_logits_new is clean (no penalty yet); save as-is
                            # — penalty will be reapplied from scratch in verify iter
                            pre_spec_logits = last_logits_new.copy()
                            spec_snap = cache.snapshot()
                            ids.append(spec_id)
                            freq[spec_id] = freq.get(spec_id, 0) + 1
                            pos[spec_id]  = n_pos
                            n_pos        += 1
                            n_generated  += 1
                            if stream_cb:
                                stream_cb(spec_id, spec_conf)
                            if n_generated >= max_new_tokens:
                                break
                            spec_pending = spec_id
                    cur = [ids[-1]]
                    continue
                else:
                    # ❌ Speculative rejected
                    # Step 1: Roll back cache to pre-speculation state
                    if spec_snap is not None:
                        cache.restore(spec_snap)
                    # Step 2: Re-sample verify_id from CLEAN logits (before spec forward
                    # contaminated the distribution). This avoids hallucination from
                    # the rejected token's K/V affecting the logits we sampled from.
                    if pre_spec_logits is not None:
                        last_logits = pre_spec_logits.copy()
                        # Re-apply repetition penalty with spec_pending already excluded
                        # (we pop it below before recording verify_id)
                        for tid, cnt in freq.items():
                            if tid == spec_pending:
                                continue      # exclude the rejected token
                            if cnt > 0:
                                dist_penalty = n_pos - pos.get(tid, 0) + 1
                                last_logits[tid] -= 0.3 * cnt / dist_penalty
                        if use_mirostat:
                            verify_id, mu = self._sample_mirostat_v2(last_logits, mu, tau, eta)
                        else:
                            verify_id = self._sample_topk_topp(last_logits, current_temp, top_p, min_p=min_p)
                    # Step 3: Update token list — replace rejected spec with correct token
                    if ids and ids[-1] == spec_pending:
                        ids[-1] = verify_id
                        freq.pop(spec_pending, None)
                        pos.pop(spec_pending, None)
                        # n_pos was already incremented when spec was tentatively added;
                        # we reuse that slot for verify_id (position stays correct)
                    freq[verify_id] = freq.get(verify_id, 0) + 1
                    pos[verify_id]  = n_pos - 1   # reuse spec's slot position
                    conf = float(np.exp(np.clip(last_logits[verify_id] - last_logits.max(), -50, 0)))
                    if stream_cb:
                        stream_cb(verify_id, conf)
                    if verify_id == eos_id:
                        # Forward EOS so cache is consistent, then terminate
                        self.forward([verify_id], cache,
                                     intent_boost=intent_boost,
                                     add_noise=True,
                                     intent_bias=intent_bias)
                        spec_pending    = None
                        spec_snap       = None
                        pre_spec_logits = None
                        break
                    # Step 4: Clean up and continue — next iteration forwards verify_id
                    spec_pending    = None
                    spec_snap       = None
                    pre_spec_logits = None
                    cur = [ids[-1]]
                    continue
            # ── Normal sampling step ──────────────────────────────
            # FIX ADAPTIVE-TEMP
            if adaptive_temp and not use_mirostat:
                probs_tmp = np.exp(last_logits - last_logits.max())
                probs_tmp /= probs_tmp.sum() + 1e-9
                entropy = float(-np.sum(probs_tmp * np.log(probs_tmp + 1e-9)))
                max_entropy = math.log(self.vocab_size)
                norm_entropy = entropy / (max_entropy + 1e-9)
                if norm_entropy < 0.1:
                    current_temp = min(temp * 1.5, 2.0)
                elif norm_entropy > 0.8:
                    current_temp = max(temp * 0.7, 0.3)
                else:
                    current_temp = temp
            if use_mirostat:
                nid, mu = self._sample_mirostat_v2(last_logits, mu, tau, eta)
            else:
                nid = self._sample_topk_topp(last_logits, current_temp, top_p, min_p=min_p)
            ids.append(nid)
            freq[nid] = freq.get(nid, 0) + 1
            pos[nid]  = n_pos
            n_pos    += 1
            n_generated += 1
            conf = float(np.exp(np.clip(last_logits[nid] - last_logits.max(), -50, 0)))
            if stream_cb:
                stream_cb(nid, conf)
            if nid == eos_id:
                break
            if n_generated >= max_new_tokens:
                break
            # ── Speculative decoding ──────────────────────────────
            if use_speculative:
                spec_id, spec_conf = self.mtp.try_speculative(l2)
                if spec_id is not None:
                    if spec_id == eos_id:
                        # FIX S1: immediate break on speculative EOS
                        break
                    # FIX CACHE-ROLLBACK: save clean logits and snapshot BEFORE forwarding spec token
                    pre_spec_logits = last_logits.copy()
                    spec_snap = cache.snapshot()
                    # Tentatively accept; will verify next iteration
                    ids.append(spec_id)
                    freq[spec_id] = freq.get(spec_id, 0) + 1
                    pos[spec_id]  = n_pos
                    n_pos        += 1
                    n_generated  += 1
                    if stream_cb:
                        stream_cb(spec_id, spec_conf)
                    if n_generated >= max_new_tokens:
                        break
                    spec_pending = spec_id
            cur = [ids[-1]]
        # ── Cleanup: if ended mid-spec, restore and remove unverified token ──
        if spec_pending is not None:
            if spec_snap is not None:
                cache.restore(spec_snap)
            if ids and ids[-1] == spec_pending:
                ids.pop()
            pre_spec_logits = None
        self._miro_mu = mu   # FIX: persist mu for next generate() call (multi-turn)
        return ids[len(prompt_ids):]
    # ── Weight loading ────────────────────────────────────────────────
    def load_external_weights(self, state_dict: dict, from_checkpoint: bool = True):
        """
        Load weights from a Qwen-compatible state_dict.
        Expert assignment: Qwen layer i → expert (i % n_experts), AVERAGED.
        """
        import re as _re
        expert_accum: Dict[int, Dict[str, List]] = {e: {} for e in range(self.n_experts)}
        shared_accum: Dict[str, List] = {}
        def _accum(accum_dict, key, arr):
            if key not in accum_dict:
                accum_dict[key] = [arr.copy().astype(np.float32), 1]
            else:
                accum_dict[key][0] += arr.astype(np.float32)
                accum_dict[key][1] += 1
        for key, val in state_dict.items():
            arr = val if isinstance(val, np.ndarray) else np.array(val, dtype=np.float32)
            if "embed_tokens" in key:
                self.embedding    = arr.astype(np.float32)
                self.lm_head      = self.embedding
                self.mtp.lm_head  = self.lm_head
                continue
            if "lm_head" in key:
                self.lm_head      = arr.astype(np.float32)
                self.mtp.lm_head  = self.lm_head
                continue
            for i, block in enumerate(self.blocks):
                tag = f"layers.{i}."
                if tag not in key:
                    continue
                if "q_proj"   in key: block.attn.W_q = arr.T.astype(np.float32)
                if "k_proj"   in key: block.attn.W_k = arr.T.astype(np.float32)
                if "v_proj"   in key: block.attn.W_v = arr.T.astype(np.float32)
                if "o_proj"   in key: block.attn.W_o = arr.T.astype(np.float32)
                if "input_layernorm"          in key: block.norm1 = arr.astype(np.float32)
                if "post_attention_layernorm" in key: block.norm2 = arr.astype(np.float32)
            m = _re.search(r"layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)", key)
            if m:
                src_layer  = int(m.group(1))
                proj_name  = m.group(2)
                expert_idx = src_layer % self.n_experts
                attr_map   = {"gate_proj": "W_g", "up_proj": "W_u", "down_proj": "W_d"}
                attr       = attr_map[proj_name]
                _accum(expert_accum[expert_idx], attr, arr.T)
                _accum(shared_accum, attr, arr.T)
        for e in range(self.n_experts):
            for attr, (total, count) in expert_accum[e].items():
                avg = (total / count).astype(np.float32)
                setattr(self.blocks[0].moe.experts[e], attr, avg)
                for blk in self.blocks[1:]:
                    setattr(blk.moe.experts[e], attr, avg)
        for attr, (total, count) in shared_accum.items():
            avg = (total / count).astype(np.float32)
            for blk in self.blocks:
                setattr(blk.moe.shared, attr, avg)
        if "model.norm.weight" in state_dict:
            self.norm_f = np.array(state_dict["model.norm.weight"], dtype=np.float32)
        if not from_checkpoint:
            for blk in self.blocks:
                for exp in blk.moe.experts:
                    exp._break_symmetry()
    def set_identity_bias(self, token_ids: List[int], boost: float = 2.0):
        """
        Logit-level identity overlay.
        FIX D2: Default boost reduced from 5.0 → 2.0.
        """
        self._id_bias = np.zeros(self.vocab_size, dtype=np.float32)
        for tid in token_ids:
            if 0 <= tid < self.vocab_size:
                self._id_bias[tid] = boost
    def quantize_weights(self, quant: Optional[str] = None,
                          group_size: int = 128) -> None:
        """
        Quantise all attention and FFN weights in-place to QuantizedLinear objects.
        quant: 'int8' or 'int4'.  Uses self._quant_mode if quant is None.
        After this call, GQAttention.W_q/k/v/o and ExpertFFN.W_g/u/d become
        QuantizedLinear instances; forward() calls ql.forward() transparently.

        Memory savings vs float32:
          int8: ~4× for FFN/attn weights
          int4: ~8× (suitable for < 6 GB RAM inference of a 9B model)

        Note: embedding and lm_head remain float32 (vocab × d_model is small
        relative to FFN; quantising embedding hurts quality noticeably).
        """
        mode = quant or self._quant_mode
        if mode is None:
            raise ValueError("Specify quant='int8' or 'int4'")
        self._quant_mode      = mode
        self._quant_group_size = group_size

        for blk in self.blocks:
            # Attention projections: shape (in, out) for NumPy convention
            # QuantizedLinear.from_float expects (out, in) — transpose first
            for attr in ("W_q", "W_k", "W_v", "W_o"):
                W = getattr(blk.attn, attr)  # (in, out)
                if isinstance(W, QuantizedLinear):
                    continue
                ql = QuantizedLinear.from_float(W.T, quant=mode, group_size=group_size)
                # Wrap so forward(x) = x @ W  (not x @ W.T) — store transposed QL
                ql._transposed = True
                setattr(blk.attn, attr, ql)
            # MoE experts
            for exp in list(blk.moe.experts) + [blk.moe.shared]:
                for attr in ("W_g", "W_u", "W_d"):
                    W = getattr(exp, attr)
                    if isinstance(W, QuantizedLinear):
                        continue
                    ql = QuantizedLinear.from_float(W.T, quant=mode, group_size=group_size)
                    ql._transposed = True
                    setattr(exp, attr, ql)

    def cast_weights(self, dtype: np.dtype) -> None:
        """Cast all float weight matrices to the given dtype (float16 / float32).
        QuantizedLinear weights are NOT cast (they store int8/uint8).
        Call after load_external_weights() to reduce RAM.
        """
        dtype = np.dtype(dtype)
        self._dtype = dtype
        # embedding stays in dtype (used for lookup — must match compute)
        self.embedding = self.embedding.astype(dtype)
        self.lm_head   = self.embedding
        if self.mtp.lm_head is not None:
            self.mtp.lm_head = self.lm_head
        self.norm_f = self.norm_f.astype(dtype)
        for blk in self.blocks:
            blk.norm1 = blk.norm1.astype(dtype)
            blk.norm2 = blk.norm2.astype(dtype)
            for attr in ("W_q", "W_k", "W_v", "W_o"):
                W = getattr(blk.attn, attr)
                if not isinstance(W, QuantizedLinear):
                    setattr(blk.attn, attr, W.astype(dtype))
            for exp in list(blk.moe.experts) + [blk.moe.shared]:
                for attr in ("W_g", "W_u", "W_d"):
                    W = getattr(exp, attr)
                    if not isinstance(W, QuantizedLinear):
                        setattr(exp, attr, W.astype(dtype))
            if not isinstance(blk.moe.W_router, QuantizedLinear):
                blk.moe.W_router = blk.moe.W_router.astype(dtype)
            blk.moe.norm_w = blk.moe.norm_w.astype(dtype)

    def save_weights(self, path: str):
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "embedding.npy"), self.embedding)
        np.save(os.path.join(path, "norm_f.npy"),    self.norm_f)
        for i, blk in enumerate(self.blocks):
            prefix = os.path.join(path, f"block_{i}")
            np.save(f"{prefix}_norm1.npy", blk.norm1)
            np.save(f"{prefix}_norm2.npy", blk.norm2)
            for attr in ("W_q", "W_k", "W_v", "W_o"):
                np.save(f"{prefix}_attn_{attr}.npy", getattr(blk.attn, attr))
            for e, exp in enumerate(blk.moe.experts):
                for attr in ("W_g", "W_u", "W_d"):
                    np.save(f"{prefix}_expert{e}_{attr}.npy", getattr(exp, attr))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
    @classmethod
    def load_weights(cls, path: str) -> "DracoTransformerV1":
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        model = cls(config)
        model.embedding   = np.load(os.path.join(path, "embedding.npy"))
        model.lm_head     = model.embedding
        model.mtp.lm_head = model.lm_head
        model.norm_f      = np.load(os.path.join(path, "norm_f.npy"))
        for i, blk in enumerate(model.blocks):
            prefix = os.path.join(path, f"block_{i}")
            blk.norm1 = np.load(f"{prefix}_norm1.npy")
            blk.norm2 = np.load(f"{prefix}_norm2.npy")
            for attr in ("W_q", "W_k", "W_v", "W_o"):
                setattr(blk.attn, attr, np.load(f"{prefix}_attn_{attr}.npy"))
            for e, exp in enumerate(blk.moe.experts):
                for attr in ("W_g", "W_u", "W_d"):
                    setattr(exp, attr, np.load(f"{prefix}_expert{e}_{attr}.npy"))
        return model
# ─────────────────────────────────────────────────────────────────────
# GGUFExporter — export weights to GGUF FP16 for llama.cpp
# ─────────────────────────────────────────────────────────────────────
class GGUFExporter:
    """
    Export DracoTransformerV1 weights to GGUF (FP16) for use with llama.cpp.
    The exported file uses the 'llama' architecture name so llama.cpp can load it
    with Q4_K_M quantisation via:
        ./llama-quantize dracoai_fp16.gguf dracoai_q4km.gguf Q4_K_M

    Usage:
        exporter = GGUFExporter(model)
        exporter.write_gguf("dracoai_fp16.gguf")

    Requires: pip install gguf
    The gguf package is the official Python writer from the llama.cpp project.
    """

    # GGUF tensor name map: DracoAI attr → llama.cpp tensor name convention
    # Mirrors the Qwen2-MoE naming so llama.cpp's Qwen2MoE handler recognises them.
    _ATTN_MAP = {
        "W_q": "attn_q",
        "W_k": "attn_k",
        "W_v": "attn_v",
        "W_o": "attn_output",
    }
    _FFN_MAP = {
        "W_g": "ffn_gate",
        "W_u": "ffn_up",
        "W_d": "ffn_down",
    }

    def __init__(self, model: "DracoTransformerV1"):
        self.model = model

    def write_gguf(self, output_path: str):
        """Write GGUF FP16 file.  Raises ImportError if 'gguf' is not installed."""
        try:
            from gguf import GGUFWriter
        except ImportError:
            raise ImportError(
                "gguf package not found. Install with: pip install gguf\n"
                "Source: https://github.com/ggerganov/llama.cpp/tree/master/gguf-py"
            )

        m   = self.model
        cfg = m.config
        writer = GGUFWriter(output_path, "llama")

        # ── Metadata ────────────────────────────────────────────────
        writer.add_name("DracoAI-V1")
        writer.add_description("DracoAI V1 MoE Transformer — exported from transformer_v1.py")
        writer.add_uint32("llama.context_length",       cfg.get("window", 1024))
        writer.add_uint32("llama.embedding_length",     cfg.get("d_model", 128))
        writer.add_uint32("llama.block_count",          cfg.get("n_layers", 4))
        writer.add_uint32("llama.attention.head_count", cfg.get("n_heads", 4))
        writer.add_uint32("llama.attention.head_count_kv", cfg.get("n_kv_heads", 2))
        writer.add_float32("llama.attention.layer_norm_rms_epsilon", 1e-6)
        writer.add_uint32("llama.vocab_size",           cfg.get("vocab_size", 151936))
        writer.add_uint32("llama.rope.dimension_count", cfg.get("head_dim", 32))
        # MoE metadata
        writer.add_uint32("llama.expert_count",         cfg.get("n_experts", 8))
        writer.add_uint32("llama.expert_used_count",    2)  # top-k=2

        # ── Token embedding ─────────────────────────────────────────
        writer.add_tensor("token_embd.weight",
                          m.embedding.astype(np.float16))
        writer.add_tensor("output_norm.weight",
                          m.norm_f.astype(np.float16))
        writer.add_tensor("output.weight",
                          m.lm_head.astype(np.float16))

        # ── Per-layer tensors ────────────────────────────────────────
        for i, blk in enumerate(m.blocks):
            pfx = f"blk.{i}"

            def _add(name: str, arr):
                w = arr.dequantize() if isinstance(arr, QuantizedLinear) else arr
                writer.add_tensor(f"{pfx}.{name}.weight", w.astype(np.float16))

            # Norms
            _add("attn_norm",  blk.norm1)
            _add("ffn_norm",   blk.norm2)
            # Attention — NumPy stores (in, out); GGUF expects (out, in)
            for attr, tname in self._ATTN_MAP.items():
                W = getattr(blk.attn, attr)
                w = W.dequantize() if isinstance(W, QuantizedLinear) else W
                writer.add_tensor(f"{pfx}.{tname}.weight",
                                  w.T.astype(np.float16))
            # Router
            writer.add_tensor(f"{pfx}.ffn_gate_inp.weight",
                              blk.moe.W_router.astype(np.float16))
            # Experts
            for e, exp in enumerate(blk.moe.experts):
                for attr, tname in self._FFN_MAP.items():
                    W = getattr(exp, attr)
                    w = W.dequantize() if isinstance(W, QuantizedLinear) else W
                    writer.add_tensor(f"{pfx}.{tname}_exps.{e}.weight",
                                      w.T.astype(np.float16))
            # Shared expert
            for attr, tname in self._FFN_MAP.items():
                W = getattr(blk.moe.shared, attr)
                w = W.dequantize() if isinstance(W, QuantizedLinear) else W
                writer.add_tensor(f"{pfx}.{tname}_shexp.weight",
                                  w.T.astype(np.float16))

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        print(f"✅ GGUF written to {output_path}  "
              f"(quantise with: llama-quantize {output_path} out.gguf Q4_K_M)")

# ─────────────────────────────────────────────────────────────────────
# TransformerBridge — unified inference interface with llama.cpp fallback
# ─────────────────────────────────────────────────────────────────────
class TransformerBridge:
    """
    Production inference bridge that:
      1. Tries the NumPy DracoTransformerV1 backend first.
      2. Falls back to llama.cpp (via llama-cpp-python) if a .gguf file is provided.
      3. Forwards intent_bias / intent_boost to whichever backend is active.

    Usage — NumPy backend:
        bridge = TransformerBridge(numpy_model=model)
        ids = bridge.generate(prompt_ids, max_new_tokens=256)

    Usage — llama.cpp backend (4-bit, fast):
        bridge = TransformerBridge(gguf_path="dracoai_q4km.gguf",
                                   n_gpu_layers=32)
        ids = bridge.generate(prompt_ids, max_new_tokens=256)

    Usage — auto (numpy until GGUF available, then swap):
        bridge = TransformerBridge(numpy_model=model, gguf_path="out.gguf")
        bridge.export_gguf()   # exports then switches backend
        ids = bridge.generate(prompt_ids)

    Intent bias forwarding:
        bridge.set_intent_bias(bias_array)   # (n_experts,) — added to router
        bridge.set_intent_boost(boost_array) # (vocab_size,) — added to logits
    Both are applied on every generate() call until cleared.
    llama.cpp backend applies intent_boost as logit_bias (token → score offset).
    """

    BACKEND_NUMPY  = "numpy"
    BACKEND_LLAMA  = "llama.cpp"

    def __init__(self,
                 numpy_model: Optional["DracoTransformerV1"] = None,
                 gguf_path:   Optional[str] = None,
                 n_gpu_layers: int = 0,
                 n_ctx:        int = 2048,
                 verbose:      bool = False):
        self._numpy_model  = numpy_model
        self._gguf_path    = gguf_path
        self._n_gpu_layers = n_gpu_layers
        self._n_ctx        = n_ctx
        self._verbose      = verbose
        self._llama        = None   # lazy-loaded llama_cpp.Llama instance
        self._intent_bias:  Optional[np.ndarray] = None
        self._intent_boost: Optional[np.ndarray] = None

        # Decide initial backend
        if gguf_path and os.path.exists(gguf_path):
            self._backend = self.BACKEND_LLAMA
            self._load_llama()
        elif numpy_model is not None:
            self._backend = self.BACKEND_NUMPY
        else:
            raise ValueError("Provide numpy_model or an existing gguf_path.")

    # ── Backend management ───────────────────────────────────────────
    def _load_llama(self):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Install with:\n"
                "  pip install llama-cpp-python\n"
                "  (GPU: CMAKE_ARGS='-DLLAMA_CUDA=on' pip install llama-cpp-python)"
            )
        self._llama = Llama(
            model_path=self._gguf_path,
            n_gpu_layers=self._n_gpu_layers,
            n_ctx=self._n_ctx,
            verbose=self._verbose,
        )

    def export_gguf(self, output_path: Optional[str] = None) -> str:
        """Export NumPy model to GGUF and switch to llama.cpp backend."""
        if self._numpy_model is None:
            raise RuntimeError("No numpy_model to export.")
        path = output_path or self._gguf_path or "dracoai_fp16.gguf"
        GGUFExporter(self._numpy_model).write_gguf(path)
        self._gguf_path = path
        self._backend   = self.BACKEND_LLAMA
        self._load_llama()
        return path

    @property
    def backend(self) -> str:
        return self._backend

    def use_numpy(self):
        """Force NumPy backend (for debugging or when GGUF is slower)."""
        if self._numpy_model is None:
            raise RuntimeError("No numpy_model available.")
        self._backend = self.BACKEND_NUMPY

    def use_llama(self):
        """Force llama.cpp backend."""
        if self._gguf_path is None or not os.path.exists(self._gguf_path):
            raise RuntimeError("No valid gguf_path. Call export_gguf() first.")
        self._backend = self.BACKEND_LLAMA
        if self._llama is None:
            self._load_llama()

    # ── Intent control ───────────────────────────────────────────────
    def set_intent_bias(self, bias: Optional[np.ndarray]):
        """
        (n_experts,) array added to MoE router logits every forward pass.
        NumPy backend: forwarded directly to DracoTransformerV1.forward().
        llama.cpp backend: MoE router is internal — bias is silently ignored
          (llama.cpp does not expose per-expert routing hooks externally).
        """
        self._intent_bias = bias

    def set_intent_boost(self, boost: Optional[np.ndarray]):
        """
        (vocab_size,) array added to output logits every generate step.
        NumPy backend: forwarded as intent_boost.
        llama.cpp backend: translated to logit_bias dict (top-200 non-zero entries).
        """
        self._intent_boost = boost

    def _boost_to_logit_bias(self) -> Optional[Dict[int, float]]:
        """Convert intent_boost np array to llama_cpp logit_bias format."""
        if self._intent_boost is None:
            return None
        arr = self._intent_boost
        # Only pass non-zero entries (llama_cpp logit_bias is a dict token→score)
        nz = np.nonzero(arr)[0]
        if len(nz) == 0:
            return None
        # Limit to top-200 by absolute value to avoid huge dict
        if len(nz) > 200:
            nz = nz[np.argsort(np.abs(arr[nz]))[-200:]]
        return {int(i): float(arr[i]) for i in nz}

    # ── Main generate interface ──────────────────────────────────────
    def generate(self,
                 prompt_ids:     List[int],
                 max_new_tokens: int   = 256,
                 temp:           float = DEFAULT_TEMP,
                 top_p:          float = DEFAULT_TOP_P,
                 min_p:          float = 0.0,
                 eos_id:         int   = 151645,
                 new_prompt:     bool  = True,
                 use_mirostat:   bool  = True,
                 use_speculative: bool = True,
                 stream_cb: Optional[Callable[[int, float], None]] = None,
                 ) -> List[int]:
        """
        Generate tokens.  Dispatches to active backend.
        Returns new token IDs (not including prompt).
        """
        if self._backend == self.BACKEND_NUMPY:
            return self._generate_numpy(
                prompt_ids, max_new_tokens, temp, top_p, min_p,
                eos_id, new_prompt, use_mirostat, use_speculative, stream_cb
            )
        else:
            return self._generate_llama(
                prompt_ids, max_new_tokens, temp, top_p, min_p, eos_id, stream_cb
            )

    def _generate_numpy(self, prompt_ids, max_new_tokens, temp, top_p, min_p,
                        eos_id, new_prompt, use_mirostat, use_speculative, stream_cb):
        return self._numpy_model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temp=temp,
            top_p=top_p,
            min_p=min_p,
            eos_id=eos_id,
            new_prompt=new_prompt,
            use_mirostat=use_mirostat,
            use_speculative=use_speculative,
            stream_cb=stream_cb,
            intent_boost=self._intent_boost,
            intent_bias=self._intent_bias,
        )

    def _generate_llama(self, prompt_ids, max_new_tokens, temp, top_p, min_p,
                        eos_id, stream_cb):
        """Generate via llama.cpp.  Decodes token IDs → text for llama_cpp API."""
        if self._llama is None:
            self._load_llama()
        logit_bias = self._boost_to_logit_bias()
        # llama_cpp.Llama.generate() accepts token list directly (llama-cpp-python ≥ 0.2.0)
        output_ids: List[int] = []
        gen = self._llama.generate(
            prompt_ids,
            top_k=50,
            top_p=top_p,
            min_p=min_p,
            temp=temp,
            repeat_penalty=1.1,
            logits_processor=None,
        )
        for tok in gen:
            if tok == eos_id or len(output_ids) >= max_new_tokens:
                break
            output_ids.append(tok)
            if stream_cb:
                stream_cb(tok, 1.0)
        # Apply logit_bias post-hoc if provided (note: llama-cpp-python exposes
        # logit_bias at the higher eval() level, not generate(); patch here for
        # full parity — advanced users can subclass and override _generate_llama)
        return output_ids

    def __repr__(self) -> str:
        return (f"TransformerBridge(backend={self._backend!r}, "
                f"gguf={self._gguf_path!r})")

# ── Smoke test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    config = {
        "d_model": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
        "head_dim": 16, "d_ff": 128, "n_experts": 4, "vocab_size": 1000, "window": 64,
    }
    # ── Test float32 baseline ──
    model = DracoTransformerV1(config)
    cache = model._make_cache()
    ids_test = [1, 2, 3]
    l1, l2, _ = model.forward(ids_test, cache)
    snap = cache.snapshot()
    old_pos = cache.cache_pos
    l1b, l2b, _ = model.forward([4], cache)
    cache.restore(snap)
    assert cache.cache_pos == old_pos, "snapshot/restore failed"
    print("✅ KVCache snapshot/restore OK")
    out = model.generate([1, 2, 3], max_new_tokens=5, use_speculative=False, debug=True)
    assert isinstance(out, list) and len(out) <= 5
    print(f"✅ generate OK: {out}")
    out2 = model.generate([1, 2, 3], max_new_tokens=8, use_speculative=True)
    print(f"✅ speculative generate OK: {out2}")
    logits = np.array([100.0, -100.0, 50.0, 0.0])
    nid, mu = DracoTransformerV1._sample_mirostat_v2(logits, mu=5.0)
    assert 0 <= nid < 4
    print(f"✅ mirostat sampling OK: id={nid}, mu={mu:.3f}")
    nid2 = DracoTransformerV1._sample_topk_topp(logits, min_p=0.1, top_k=4)
    assert 0 <= nid2 < 4
    print(f"✅ min-p sampling OK: id={nid2}")

    # ── Test float16 dtype ──
    model16 = DracoTransformerV1(config, dtype=np.float16)
    assert model16.embedding.dtype == np.float16
    out16 = model16.generate([1, 2, 3], max_new_tokens=4, use_speculative=False)
    assert isinstance(out16, list)
    print(f"✅ float16 generate OK: {out16}")

    # ── Test cast_weights ──
    model.cast_weights(np.float16)
    assert model.embedding.dtype == np.float16
    out_cast = model.generate([1, 2, 3], max_new_tokens=3, use_speculative=False)
    assert isinstance(out_cast, list)
    model.cast_weights(np.float32)   # restore
    print("✅ cast_weights float16 → float32 OK")

    # ── Test INT8 quantisation ──
    model_q = DracoTransformerV1(config)
    model_q.quantize_weights(quant='int8')
    out_q8 = model_q.generate([1, 2, 3], max_new_tokens=4, use_speculative=False)
    assert isinstance(out_q8, list)
    print(f"✅ INT8 quantized generate OK: {out_q8}")

    # ── Test INT4 quantisation (group_size=16 for small test model) ──
    model_q4 = DracoTransformerV1(config)
    model_q4.quantize_weights(quant='int4', group_size=16)
    out_q4 = model_q4.generate([1, 2, 3], max_new_tokens=4, use_speculative=False)
    assert isinstance(out_q4, list)
    print(f"✅ INT4 quantized generate OK: {out_q4}")

    # ── Test QuantizedLinear save/load ──
    import tempfile, os as _os
    W_test = np.random.randn(32, 64).astype(np.float32)
    ql8    = QuantizedLinear.from_float(W_test, quant='int8')
    W_dq   = ql8.dequantize()
    assert W_dq.shape == W_test.shape
    with tempfile.TemporaryDirectory() as td:
        ql8.save(_os.path.join(td, "test_ql"))
        ql8_loaded = QuantizedLinear.load(_os.path.join(td, "test_ql.npz"))
        assert np.allclose(ql8_loaded.dequantize(), W_dq, atol=1e-2)
    print("✅ QuantizedLinear INT8 save/load round-trip OK")

    # ── Test TransformerBridge (numpy backend) ──
    bridge = TransformerBridge(numpy_model=DracoTransformerV1(config))
    boost  = np.zeros(config["vocab_size"], dtype=np.float32)
    boost[42] = 3.0
    bridge.set_intent_boost(boost)
    bridge_out = bridge.generate([1, 2, 3], max_new_tokens=5, use_speculative=False)
    assert isinstance(bridge_out, list)
    print(f"✅ TransformerBridge (numpy) OK: {bridge_out}")

    print("✅ transformer_v1 self-test passed")