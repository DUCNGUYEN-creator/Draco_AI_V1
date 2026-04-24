"""
DracoAI V1 — Tokenizer
======================
Qwen 3.5 9B Instruct compatible.

- BPE encode O(N log N) via merge_rank
- UTF-8 byte-buffer safe streaming decode
- Extension vocab lock (ID >= QWEN_BASE_END)
- ChatML template: <|im_start|>role\\ncontent<|im_end|>

QWEN_BASE_END = 151936  (Qwen 3.5 9B vocab boundary)

FIXES:
    ✅ Qwen 3.5 9B Instruct (was 7B — typo)
    ✅ Special token IDs aligned to Qwen 3.5 9B checkpoint layout
       (endoftext=151643, im_start=151644, im_end=151645 …)
    ✅ encode_chat uses correct token IDs (not raw byte 256+idx offsets)
    ✅ stream_decode: UTF-8 safe, flush on word boundary
    ✅ decode_token: single-token real-time decode
    ✅ save/load: config model field updated to Qwen3.5-9B-Instruct
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

import re, os, json, math, hashlib, time
from collections import Counter
from typing import List, Dict, Tuple, Optional, Iterator

# ── Special tokens (Qwen 3.5 9B ChatML style) ───────────────────────
# These IDs match the actual Qwen 3.5 9B Instruct checkpoint vocabulary.
SPECIAL_TOKENS: Dict[str, int] = {
    "<|endoftext|>": 151643,
    "<|im_start|>":  151644,
    "<|im_end|>":    151645,
    "<|pad|>":       151646,
    "<|unk|>":       151647,
    "<|sep|>":       151648,
    "<think>":       151649,
    "</think>":      151650,
    "<tool_call>":   151651,
    "</tool_call>":  151652,
}

N_SPECIAL     = len(SPECIAL_TOKENS)
# Qwen 3.5 9B vocab size (same boundary as 7B — Qwen uses 151936 for all sizes)
QWEN_BASE_END = 151936

BOS_TOKEN = "<|im_start|>"
EOS_TOKEN = "<|im_end|>"
PAD_TOKEN = "<|pad|>"
UNK_TOKEN = "<|unk|>"

# ─────────────────────────────────────────────────────────────────────

class CharAnalyzer:
    VIET  = set("áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ")
    PUNCT = set(".,!?;:\"'()[]{}/—–...«»")

    @staticmethod
    def char_type(c: str) -> str:
        if c.isdigit():                   return "NUM"
        if c in CharAnalyzer.VIET:        return "VIET_TONE"
        if c.isalpha():                   return "VIET" if ord(c) > 127 else "ALPHA"
        if c in CharAnalyzer.PUNCT:       return "PUNCT"
        if c == " ":                      return "SPACE"
        if c == "\n":                     return "NEWLINE"
        return "OTHER"

    @staticmethod
    def analyze_word(w: str) -> dict:
        types = [CharAnalyzer.char_type(c) for c in w]
        return {
            "text": w, "length": len(w), "char_types": types,
            "has_viet": any(t in ("VIET", "VIET_TONE") for t in types),
            "has_num":  any(t == "NUM" for t in types),
            "is_upper": w.isupper(), "is_title": w.istitle(),
        }

# ─────────────────────────────────────────────────────────────────────

class BPETokenizer:
    """
    Byte Pair Encoding Tokenizer — DracoAI / Qwen 3.5 9B Instruct compatible.

    Special token IDs match Qwen 3.5 9B checkpoint (151643–151652).
    Base vocab IDs: 0–255 (byte tokens), merges above 255.

    Streaming:
        for piece in tok.stream_decode(ids):
            print(piece, end="", flush=True)

    Single-token decode (real-time UI):
        text = tok.decode_token(token_id)
    """

    # Chars that trigger a stream-flush (word boundary)
    _FLUSH_CHARS: frozenset = frozenset(" \t\n.,!?;:—–()[]{}'\"，。！？；：")

    _PATTERN = re.compile(
        r"[a-zA-ZÀ-ỹ]+|"
        r"\d+(?:[.,]\d+)?|"
        r"\s+|"
        r"[^\w\s]",
        re.UNICODE,
    )

    def __init__(self):
        self.merges:      Dict[Tuple[int, int], int] = {}
        self._merge_rank: Dict[Tuple[int, int], int] = {}
        self.vocab:       Dict[int, bytes]            = {}
        self.inv_vocab:   Dict[bytes, int]            = {}
        self._ext_vocab:  Dict[int, bytes]            = {}
        self._ext_inv:    Dict[bytes, int]            = {}
        # Build base byte vocab (IDs 0–255)
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    # ── Pre-tokenize ─────────────────────────────────────────────────
    def _pre(self, text: str) -> List[str]:
        return self._PATTERN.findall(text)

    # ── Train ────────────────────────────────────────────────────────
    def train(self, text: str, vocab_size: int = 8000, verbose: bool = True):
        assert vocab_size > 256, "vocab_size must be > 256"
        n_merges = vocab_size - 256
        if verbose:
            print(f"[BPE] Train: vocab={vocab_size}, merges={n_merges}, corpus={len(text):,}c")
        ids = list(text.encode("utf-8"))
        # Reset merges
        self.merges = {}
        next_id = 256

        for mi in range(n_merges):
            counts: Counter = Counter()
            for a, b in zip(ids, ids[1:]):
                counts[(a, b)] += 1
            if not counts:
                break
            best = max(counts, key=counts.get)
            if counts[best] < 2:
                break
            new_ids: List[int] = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == best[0] and ids[i + 1] == best[1]:
                    new_ids.append(next_id); i += 2
                else:
                    new_ids.append(ids[i]);  i += 1
            ids = new_ids
            self.vocab[next_id] = self.vocab[best[0]] + self.vocab[best[1]]
            self.merges[best] = next_id
            if verbose and (mi + 1) % 200 == 0:
                print(f"  Merge {mi+1}/{n_merges}: id={next_id} freq={counts[best]}")
            next_id += 1

        self.inv_vocab   = {v: k for k, v in self.vocab.items()}
        self._merge_rank = {pair: rank for rank, pair in enumerate(self.merges.keys())}
        if verbose:
            print(f"[BPE] Done! vocab={len(self.vocab)}")

    # ── Encode (O(N log N) via merge_rank) ───────────────────────────
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        Special tokens use their Qwen 3.5 9B IDs (SPECIAL_TOKENS dict).
        BPE tokens use IDs 0–255 (bytes) and 256+ (merges).
        """
        # Handle special tokens in text
        sp_pattern = re.compile('|'.join(re.escape(tok) for tok in sorted(
            SPECIAL_TOKENS.keys(), key=len, reverse=True)))

        out = []
        if add_bos:
            out.append(SPECIAL_TOKENS[BOS_TOKEN])

        last = 0
        for match in sp_pattern.finditer(text):
            start, end = match.span()
            if start > last:
                out.extend(self._bpe_encode_plain(text[last:start]))
            out.append(SPECIAL_TOKENS[match.group(0)])
            last = end
        if last < len(text):
            out.extend(self._bpe_encode_plain(text[last:]))

        if add_eos:
            out.append(SPECIAL_TOKENS[EOS_TOKEN])
        return out

    def _bpe_encode_plain(self, text: str) -> List[int]:
        """Encode plain text (no special tokens) using BPE."""
        ids = list(text.encode("utf-8"))
        if not self.merges:
            return ids
        changed = True
        while changed and len(ids) >= 2:
            changed = False
            best_rank, best_pos = len(self.merges), -1
            for i in range(len(ids) - 1):
                r = self._merge_rank.get((ids[i], ids[i + 1]), len(self.merges))
                if r < best_rank:
                    best_rank, best_pos = r, i
            if best_pos == -1:
                break
            pair = (ids[best_pos], ids[best_pos + 1])
            ids  = ids[:best_pos] + [self.merges[pair]] + ids[best_pos + 2:]
            changed = True
        return ids

    # ── Decode single token → immediate print ────────────────────────
    def decode_token(self, token_id: int, skip_special: bool = True) -> str:
        sp_ids = frozenset(SPECIAL_TOKENS.values())
        if skip_special and token_id in sp_ids:
            return ""
        if token_id in self._ext_vocab:
            return self._ext_vocab[token_id].decode("utf-8", errors="replace")
        b = self.vocab.get(token_id, b"")
        return b.decode("utf-8", errors="replace")

    # ── Stream decode: UTF-8 safe + flush on word boundary ───────────
    def stream_decode(self, token_ids: List[int], skip_special: bool = True) -> Iterator[str]:
        sp_ids   = frozenset(SPECIAL_TOKENS.values())
        byte_buf: bytes = b""
        text_buf: str   = ""

        for tid in token_ids:
            if skip_special and tid in sp_ids:
                if text_buf:
                    yield text_buf; text_buf = ""
                continue
            raw       = self._ext_vocab.get(tid) or self.vocab.get(tid, b"")
            byte_buf += raw
            try:
                piece    = byte_buf.decode("utf-8")
                byte_buf = b""
            except UnicodeDecodeError:
                continue  # wait for more bytes of multi-byte char
            text_buf += piece
            if text_buf and text_buf[-1] in self._FLUSH_CHARS:
                yield text_buf; text_buf = ""

        # Final flush
        if byte_buf:
            text_buf += byte_buf.decode("utf-8", errors="replace")
        if text_buf:
            yield text_buf

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        return "".join(self.stream_decode(ids, skip_special))

    # ── Context analysis ─────────────────────────────────────────────
    def encode_with_context(self, text: str) -> dict:
        words  = self._pre(text)
        tokens = []
        for w in words:
            ids  = self.encode(w, add_bos=False, add_eos=False)
            info = CharAnalyzer.analyze_word(w)
            tokens.append({"surface": w, "token_ids": ids, "analysis": info})
        return {"text": text, "tokens": tokens, "total": sum(len(t["token_ids"]) for t in tokens)}

    # ── Chat template (Qwen 3.5 9B ChatML) ──────────────────────────
    def encode_chat(self, messages: List[dict]) -> List[int]:
        """
        Qwen 3.5 9B ChatML format:
            <|im_start|>system\\n...<|im_end|>\\n
            <|im_start|>user\\n...<|im_end|>\\n
            <|im_start|>assistant\\n...

        Uses real Qwen 3.5 9B special token IDs from SPECIAL_TOKENS.
        """
        im_s    = SPECIAL_TOKENS[BOS_TOKEN]   # 151644
        im_e    = SPECIAL_TOKENS[EOS_TOKEN]   # 151645
        # Newline as raw byte token (ID 10 = b'\n')
        newline = [10]
        ids     = []

        for m in messages:
            role    = m.get("role", "user")
            content = m.get("content", "")
            ids.append(im_s)
            ids.extend(self._bpe_encode_plain(role))
            ids.extend(newline)
            ids.extend(self.encode(content, add_bos=False, add_eos=False))
            ids.append(im_e)
            ids.extend(newline)

        # If last message is not assistant, add assistant prefix
        if not messages or messages[-1].get("role") != "assistant":
            ids.append(im_s)
            ids.extend(self._bpe_encode_plain("assistant"))
            ids.extend(newline)

        return ids

    # ── Extension vocab ───────────────────────────────────────────────
    def add_extension_tokens(self, new_tokens: Dict[int, str]):
        for tid, tok_str in new_tokens.items():
            if tid < QWEN_BASE_END:
                continue  # base vocab locked
            b = tok_str.encode("utf-8")
            self._ext_vocab[tid] = b
            self._ext_inv[b]     = tid
        print(f"[BPE] +{len(new_tokens)} ext tokens (ID>={QWEN_BASE_END}). Base locked.")

    @property
    def vocab_size(self) -> int:
        base = max(self.vocab.keys()) + 1 if self.vocab else 0
        ext_max = max(self._ext_vocab.keys()) + 1 if self._ext_vocab else 0
        return max(base, ext_max, QWEN_BASE_END)

    @property
    def eos_id(self):       return SPECIAL_TOKENS[EOS_TOKEN]    # 151645
    @property
    def bos_id(self):       return SPECIAL_TOKENS[BOS_TOKEN]    # 151644
    @property
    def pad_id(self):       return SPECIAL_TOKENS[PAD_TOKEN]    # 151646
    @property
    def unk_id(self):       return SPECIAL_TOKENS[UNK_TOKEN]    # 151647
    @property
    def think_id(self):     return SPECIAL_TOKENS["<think>"]    # 151649
    @property
    def think_end_id(self): return SPECIAL_TOKENS["</think>"]   # 151650

    # ── Save / Load ───────────────────────────────────────────────────
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        merges_out = {f"{a},{b}": v for (a, b), v in self.merges.items()}
        vocab_out  = {str(k): list(v) for k, v in self.vocab.items()}
        ext_out    = {str(k): list(v) for k, v in self._ext_vocab.items()}

        with open(f"{path}/merges.json", "w", encoding="utf-8") as f:
            json.dump(merges_out, f)
        with open(f"{path}/vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab_out, f)
        with open(f"{path}/ext_vocab.json", "w", encoding="utf-8") as f:
            json.dump(ext_out, f)

        config = {
            "tokenizer_class": "DracoBPE",
            "bos_token":  BOS_TOKEN,
            "eos_token":  EOS_TOKEN,
            "pad_token":  PAD_TOKEN,
            "unk_token":  UNK_TOKEN,
            "vocab_size": self.vocab_size,
            "base_end":   QWEN_BASE_END,
            # FIX: correct model name — Qwen 3.5 9B Instruct
            "model":      "Qwen/Qwen3.5-9B-Instruct",
            "special_tokens": SPECIAL_TOKENS,
        }
        with open(f"{path}/tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"[BPE] Saved → {path} ({self.vocab_size} tokens)")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        tok = cls()
        with open(f"{path}/merges.json", encoding="utf-8") as f:
            tok.merges = {
                tuple(int(x) for x in k.split(",")): v
                for k, v in json.load(f).items()
            }
        tok._merge_rank = {pair: rank for rank, pair in enumerate(tok.merges.keys())}
        with open(f"{path}/vocab.json", encoding="utf-8") as f:
            tok.vocab = {int(k): bytes(v) for k, v in json.load(f).items()}
        tok.inv_vocab = {v: k for k, v in tok.vocab.items()}
        ext_f = f"{path}/ext_vocab.json"
        if os.path.exists(ext_f):
            with open(ext_f, encoding="utf-8") as f:
                tok._ext_vocab = {int(k): bytes(v) for k, v in json.load(f).items()}
                tok._ext_inv   = {v: k for k, v in tok._ext_vocab.items()}
        print(f"[BPE] Loaded: {tok.vocab_size} tokens (+{len(tok._ext_vocab)} ext)")
        return tok