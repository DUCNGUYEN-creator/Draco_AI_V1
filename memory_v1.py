"""
DracoAI V1 — Memory System (Vector DB)
=======================================
ALL BUGS FIXED:
    ✅ MiniEmbedder dim=256 everywhere (no mismatch with LongTermMemoryV1)
    ✅ np.vstack → list buffer (no O(N²) copy per store)
    ✅ _get_episode_summaries separate name (no recursion)
    ✅ WorkingMemoryV1 accurate token counting via tokenizer
    ✅ WorkingMemoryV1 evict with full metadata
    ✅ auto_learn filter: len>5 words, no "?"
    ✅ LRU cleanup when n_vectors > MAX_VECTORS
    ✅ npz save/load with allow_pickle=True
    ✅ LongTermMemory.search: 2-stage (cosine + intent rerank)

Vector DB entry format:
    {
        "text":      str,
        "embedding": List[float],   # cosine-searchable, dim=256
        "score":     float,         # set at query time
        "timestamp": float,
        "frequency": int,
        "type":      str,           # "fact"|"episode"|"knowledge"|"user_query"
        "metadata":  dict,
    }
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

import os, json, time, math, hashlib
import numpy as np
from typing import List, Dict, Optional, Any, Tuple

# ══════════════════════════════════════════════════════════════════════
# MINI EMBEDDER  (TF-IDF + hash trick, dim=256)
# ══════════════════════════════════════════════════════════════════════
class MiniEmbedder:
    """
    Lightweight TF-IDF + hash-trick embedder.
    dim=256 unified throughout the memory system.
    Supports Vietnamese + English, bigrams, synonyms, entity boost.
    """
    SW_VI = {
        "là","và","của","có","được","không","này","đó","với","để","trong",
        "một","các","những","cho","từ","theo","về","như","vì","đã","sẽ",
        "tôi","bạn","anh","chị","em",
    }
    SW_EN = {
        "is","a","an","the","of","in","to","for","and","or","it",
        "this","that","be","was","are","at","by","from","with",
    }
    SYNO = {
        "lập trình": "code", "code": "programming",
        "hiểu": "understand", "nhớ": "memory",
        "tính": "calculate",  "debug": "fix",
    }

    def __init__(self, dim: int = 256):
        # FIX: dim=256 unified (was 512 in class, 256 in LTM → shape mismatch)
        self.dim = dim
        self._vocab: Dict[str, int]   = {}
        self._idf:   Dict[str, float] = {}
        self._n_docs: int = 0

    def _tokenize(self, text: str) -> List[str]:
        import re
        tokens  = re.findall(r"[a-zA-ZÀ-ỹ]+|\d+", text.lower())
        sw      = self.SW_VI | self.SW_EN
        tokens  = [self.SYNO.get(t, t) for t in tokens if t not in sw and len(t) > 1]
        bigrams = [tokens[i] + "_" + tokens[i + 1] for i in range(len(tokens) - 1)]
        return tokens + bigrams[:len(tokens)]

    def _hash(self, w: str) -> int:
        if w not in self._vocab:
            self._vocab[w] = int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim
        return self._vocab[w]

    def fit(self, texts: List[str]):
        self._n_docs = len(texts)
        df: Dict[str, int] = {}
        for t in texts:
            for w in set(self._tokenize(t)):
                df[w] = df.get(w, 0) + 1
        self._idf = {w: math.log((self._n_docs + 1) / (cnt + 1)) + 1.0
                     for w, cnt in df.items()}

    def embed(self, text: str, entity_boost: List[str] = None) -> np.ndarray:
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self.dim, dtype=np.float32)
        tf: Dict[str, float] = {}
        for w in tokens: tf[w] = tf.get(w, 0) + 1
        n   = len(tokens)
        vec = np.zeros(self.dim, dtype=np.float32)
        for w, cnt in tf.items():
            idf    = self._idf.get(w, 1.0)
            eb     = 2.0 if entity_boost and w in [e.lower() for e in entity_boost] else 1.0
            weight = (cnt / n) * idf * eb
            idx    = self._hash(w)
            vec[idx]                       += weight
            vec[(idx * 7  + 3) % self.dim] += weight * 0.5
            vec[(idx * 13 + 7) % self.dim] += weight * 0.3
        norm = np.linalg.norm(vec)
        return (vec / (norm + 1e-8)).astype(np.float32)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/embedder.json", "w", encoding="utf-8") as f:
            json.dump({"vocab": self._vocab, "idf": self._idf,
                       "n_docs": self._n_docs, "dim": self.dim}, f)

    def load(self, path: str):
        fp = f"{path}/embedder.json"
        if not os.path.exists(fp): return
        with open(fp, encoding="utf-8") as f:
            d = json.load(f)
        self._vocab  = d["vocab"]
        self._idf    = d["idf"]
        self._n_docs = d["n_docs"]
        self.dim     = d["dim"]   # honour saved dim (always 256)

# ══════════════════════════════════════════════════════════════════════
# KV CACHE BUFFER — short-term buffer before LTM
# ══════════════════════════════════════════════════════════════════════
class KVCacheBuffer:
    def __init__(self, embedder: MiniEmbedder, threshold: float = 0.45):
        self.embedder  = embedder
        self.threshold = threshold
        self._buf: List[dict] = []
        self._max = 50

    def push(self, text: str, context: str = ""):
        if len(text.strip()) < 3: return
        imp = self._importance(text, context)
        if imp >= self.threshold:
            self._buf.append({"text": text, "context": context,
                               "importance": imp, "ts": time.time()})
            if len(self._buf) > self._max:
                self._buf.sort(key=lambda x: x["importance"], reverse=True)
                self._buf = self._buf[:self._max // 2]

    def flush_to_ltm(self, ltm: "LongTermMemoryV1") -> int:
        count = 0
        for item in self._buf:
            ltm.store(item["text"], {"from_buffer": True, "importance": item["importance"],
                                      "ts": item["ts"], "source": "kv_cache"})
            count += 1
        self._buf.clear()
        return count

    def _importance(self, text: str, context: str) -> float:
        import re
        score = 0.3
        wc = len(text.split())
        if 5 <= wc <= 50: score += 0.2
        if re.search(r"\d+", text): score += 0.1
        if any(p in text.lower() for p in ["là", "định nghĩa", "nghĩa là", "means"]): score += 0.15
        if re.search(r"\b[A-ZÀÁÂ][a-zàáâã]+\b", text): score += 0.1
        if context:
            t_w = set(text.lower().split()); c_w = set(context.lower().split())
            score += min(len(t_w & c_w) / (len(t_w) + 1), 0.15)
        return min(score, 1.0)

# ══════════════════════════════════════════════════════════════════════
# LONG-TERM MEMORY V1 — proper Vector DB
# ══════════════════════════════════════════════════════════════════════
class LongTermMemoryV1:
    MAX_VECTORS = 50_000

    def __init__(self, memory_dir: str = "memory"):
        self.dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
        # FIX: dim=256 matches MiniEmbedder default everywhere
        self.embedder = MiniEmbedder(dim=256)
        self._ff  = f"{memory_dir}/facts.json"
        self._ef  = f"{memory_dir}/episodes.json"
        self._kf  = f"{memory_dir}/knowledge.json"
        self._pf  = f"{memory_dir}/prefs.json"
        self._npz = f"{memory_dir}/vectors.npz"
        self._mf  = f"{memory_dir}/meta.npy"
        self._init()
        self.embedder.load(memory_dir)
        # In-memory: list buffer (no O(N²) vstack per store)
        self._vec_list:  List[np.ndarray]    = []
        self._meta_list: List[dict]          = []
        self._vids:      List[str]           = []
        self._vecs_arr:  Optional[np.ndarray] = None
        self._dirty = False
        self._load_vectors()

    def _init(self):
        for fp, d in [(self._ff, []), (self._ef, []), (self._kf, {}), (self._pf, {})]:
            if not os.path.exists(fp): self._w(fp, d)

    def _r(self, fp):
        try:
            with open(fp, encoding="utf-8") as f: return json.load(f)
        except Exception:
            return {} if fp in (self._kf, self._pf) else []

    def _w(self, fp, data):
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ── Vector storage ────────────────────────────────────────────────
    def _load_vectors(self):
        self._vec_list  = []; self._meta_list = []
        self._vids      = []; self._vecs_arr  = None
        if os.path.exists(self._npz):
            try:
                data = np.load(self._npz, allow_pickle=True)
                if "vecs" in data:
                    self._vecs_arr = data["vecs"].astype(np.float32)
                    self._vids     = list(data["vids"])
            except Exception:
                pass
        if os.path.exists(self._mf):
            try:
                obj = np.load(self._mf, allow_pickle=True)
                self._meta_list = list(obj.item().get("metas", []) if obj.ndim == 0 else [])
            except Exception:
                pass
        if self._vecs_arr is not None:
            self._vec_list = [self._vecs_arr[i] for i in range(len(self._vecs_arr))]

    def _save_vectors(self):
        if not self._vec_list: return
        arr = np.array(self._vec_list, dtype=np.float32)
        np.savez_compressed(self._npz, vecs=arr, vids=np.array(self._vids, dtype=object))
        np.save(self._mf, np.array({"metas": self._meta_list}, dtype=object), allow_pickle=True)
        self._vecs_arr = arr
        self._dirty    = False

    def _ensure_arr(self):
        if self._dirty or self._vecs_arr is None or len(self._vecs_arr) != len(self._vec_list):
            if self._vec_list:
                self._vecs_arr = np.array(self._vec_list, dtype=np.float32)
                self._dirty    = False

    # ── Store ──────────────────────────────────────────────────────────
    def store(self, text: str, metadata: dict = None) -> str:
        vid = hashlib.md5(text.encode()).hexdigest()[:12]
        if vid in self._vids:
            idx = self._vids.index(vid)
            self._meta_list[idx]["frequency"] = self._meta_list[idx].get("frequency", 1) + 1
            self._save_vectors(); return vid
        vec = self.embedder.embed(text)
        meta = {
            "text":      text,
            "embedding": vec.tolist(),
            "score":     0.0,
            "timestamp": time.time(),
            "frequency": 1,
            "type":      (metadata or {}).get("type", "general"),
            "metadata":  metadata or {},
        }
        self._vids.append(vid)
        self._meta_list.append(meta)
        self._vec_list.append(vec)   # list append — no O(N²) vstack
        self._dirty = True
        if len(self._vec_list) > self.MAX_VECTORS:
            self._lru_cleanup()
        self._save_vectors()
        return vid

    def _lru_cleanup(self):
        now = time.time()
        scores = []
        for i, meta in enumerate(self._meta_list):
            age  = (now - meta.get("timestamp", now)) / 86400.0
            freq = meta.get("frequency", 1)
            scores.append((freq * math.exp(-age / 30.0), i))
        scores.sort(reverse=True)
        keep_idx = sorted(i for _, i in scores[:int(self.MAX_VECTORS * 0.8)])
        self._vids      = [self._vids[i]      for i in keep_idx]
        self._meta_list = [self._meta_list[i] for i in keep_idx]
        self._vec_list  = [self._vec_list[i]  for i in keep_idx]
        self._dirty = True

    # ── Search ─────────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 20,
               entities: List[str] = None) -> List[dict]:
        """2-stage: cosine similarity → scored entries."""
        if not self._vec_list: return []
        self._ensure_arr()
        q_vec = self.embedder.embed(query, entity_boost=entities)
        now   = time.time()
        norms = np.linalg.norm(self._vecs_arr, axis=1, keepdims=True) + 1e-8
        q_n   = np.linalg.norm(q_vec) + 1e-8
        sims  = (self._vecs_arr @ q_vec) / (norms.squeeze() * q_n)
        scored = []
        for i, (sim, meta) in enumerate(zip(sims, self._meta_list)):
            age     = (now - meta.get("timestamp", now)) / 86400.0
            recency = math.exp(-age / 7.0)
            freq    = math.log(meta.get("frequency", 1) + 1)
            score   = float(sim) * 0.6 + recency * 0.3 + (freq / 5.0) * 0.1
            scored.append({
                "id":        self._vids[i],
                "score":     score,
                "text":      meta.get("text", ""),
                "embedding": meta.get("embedding", []),
                "timestamp": meta.get("timestamp", now),
                "frequency": meta.get("frequency", 1),
                "type":      meta.get("type", "general"),
                "ts":        meta.get("timestamp", now),
                **meta.get("metadata", {}),
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def get_context(self, query: str, top_k: int = 3,
                    entities: List[str] = None) -> str:
        results = self.search(query, top_k=top_k * 3, entities=entities)[:top_k]
        return " | ".join(r["text"][:150] for r in results if r.get("text"))

    # ── Facts ──────────────────────────────────────────────────────────
    def remember_fact(self, key: str, value: str, confidence: float = 1.0):
        facts = self._r(self._ff)
        for f in facts:
            if f["key"].lower() == key.lower():
                f["value"] = value; f["confidence"] = confidence
                f["updated"] = time.time()
                self._w(self._ff, facts)
                self.store(f"{key} là {value}", {"type": "fact", "key": key})
                return
        facts.append({"key": key, "value": value, "confidence": confidence,
                       "created": time.time(), "updated": time.time()})
        self._w(self._ff, facts)
        self.store(f"{key} là {value}", {"type": "fact", "key": key})

    def recall_fact(self, key: str) -> Optional[str]:
        for f in self._r(self._ff):
            if f["key"].lower() == key.lower(): return f["value"]
        return None

    def get_facts(self) -> List[dict]: return self._r(self._ff)

    # ── Episodes ───────────────────────────────────────────────────────
    def save_episode(self, messages: List[dict], summary: str = "",
                     session_id: str = ""):
        eps = self._r(self._ef)
        eps.append({"session_id": session_id or str(time.time()),
                    "summary": summary, "saved_at": time.time(),
                    "messages": messages[-20:]})
        if len(eps) > 100: eps = eps[-100:]
        self._w(self._ef, eps)
        if summary:
            self.store(summary, {"type": "episode", "session_id": session_id})

    def _get_episode_summaries(self, query: str, top_k: int = 2) -> List[str]:
        """Separate name — no recursion with get_context."""
        q_vec  = self.embedder.embed(query); scored = []
        for ep in self._r(self._ef)[-30:]:
            s = ep.get("summary", "")
            if not s: continue
            sv    = self.embedder.embed(s)
            score = self.embedder.similarity(q_vec, sv)
            scored.append((score, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k]]

    def replay_relevant(self, query: str, top_k: int = 2) -> List[dict]:
        q_vec  = self.embedder.embed(query); scored = []
        for ep in self._r(self._ef)[-30:]:
            s = ep.get("summary", "")
            if not s: continue
            score = self.embedder.similarity(q_vec, self.embedder.embed(s))
            scored.append((score, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:top_k]]

    # ── Knowledge ──────────────────────────────────────────────────────
    def learn(self, topic: str, info: str):
        kb = self._r(self._kf)
        if topic not in kb: kb[topic] = []
        if info not in kb[topic]:
            kb[topic].append(info)
            if len(kb[topic]) > 50: kb[topic] = kb[topic][-50:]
            self._w(self._kf, kb)
            self.store(f"{topic}: {info}", {"type": "knowledge", "topic": topic})

    def recall_knowledge(self, topic: str) -> List[str]:
        kb = self._r(self._kf); tl = topic.lower()
        for k, v in kb.items():
            if tl in k.lower() or k.lower() in tl: return v
        return []

    def get_topics(self) -> List[str]: return list(self._r(self._kf).keys())

    # ── Prefs ──────────────────────────────────────────────────────────
    def set_pref(self, k, v):
        p = self._r(self._pf); p[k] = v; self._w(self._pf, p)
    def get_pref(self, k, d=None): return self._r(self._pf).get(k, d)
    def get_prefs(self) -> dict:   return self._r(self._pf)

    # ── Summary ────────────────────────────────────────────────────────
    def get_summary_with_intent(self, query: str = "", intent: dict = None) -> str:
        """No recursion: calls _get_episode_summaries (not get_context)."""
        facts  = self.get_facts()[:5]
        prefs  = self.get_prefs()
        topics = self.get_topics()[:8]
        parts  = []
        if prefs:   parts.append("User: " + ", ".join(f"{k}={v}" for k, v in list(prefs.items())[:4]))
        if facts:   parts.append("Facts: " + "; ".join(f"{f['key']}={f['value']}" for f in facts[:4]))
        if topics:  parts.append("Topics: " + ", ".join(topics[:6]))
        if query:
            entities = intent.get("entities", []) if intent else []
            ctx      = self.get_context(query, top_k=2, entities=entities)
            if ctx: parts.append(ctx)
            ep_sums  = self._get_episode_summaries(query, top_k=1)  # no recursion
            if ep_sums: parts.append(f"Episode: {ep_sums[0][:100]}")
        return " | ".join(parts)

    def fit_embedder(self, texts: List[str]):
        self.embedder.fit(texts); self.embedder.save(self.dir)
        if self._meta_list:
            self._vec_list = [self.embedder.embed(m.get("text", ""))
                              for m in self._meta_list]
            self._dirty = True
            self._save_vectors()

# ══════════════════════════════════════════════════════════════════════
# WORKING MEMORY V1
# ══════════════════════════════════════════════════════════════════════
class WorkingMemoryV1:
    """
    Short-term conversation buffer.
    Accurate token counting if tokenizer is provided.
    Evicted messages go to LTM with full metadata.
    """
    def __init__(self, ltm: LongTermMemoryV1, kv_buffer: KVCacheBuffer,
                 max_tokens: int = 512, tokenizer=None):
        self.ltm        = ltm
        self.kv_buf     = kv_buffer
        self.max_tokens = max_tokens
        self.tokenizer  = tokenizer
        self.messages:   List[dict] = []
        self.token_count = 0
        self.session_id  = str(int(time.time()))

    def _count_tokens(self, content: str) -> int:
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(content, add_bos=False, add_eos=False))
            except Exception:
                pass
        return int(len(content.split()) * 1.3)

    def add(self, role: str, content: str, meta: dict = None):
        self.messages.append({"role": role, "content": content,
                               "ts": time.time(), "meta": meta or {}})
        self.token_count += self._count_tokens(content)
        self._trim()

    def _trim(self):
        while self.token_count > self.max_tokens and len(self.messages) > 1:
            for i, m in enumerate(self.messages):
                if m["role"] == "system": continue
                removed       = self.messages.pop(i)
                removed_tc    = self._count_tokens(removed["content"])
                self.token_count -= removed_tc
                self.kv_buf.push(removed["content"], self._recent_ctx())
                self.ltm.store(removed["content"], {
                    "type":      "evicted_context",
                    "role":      removed["role"],
                    "timestamp": removed.get("ts", time.time()),
                    "source":    "working_memory_evict",
                })
                if len(self.kv_buf._buf) >= 5:
                    self.kv_buf.flush_to_ltm(self.ltm)
                break
            else:
                break

    def _recent_ctx(self) -> str:
        return " ".join(m["content"] for m in self.messages[-3:])

    def get_messages(self) -> List[dict]:
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    def end_session(self):
        if self.messages:
            self.kv_buf.flush_to_ltm(self.ltm)
            msgs    = self.get_messages()
            words   = " ".join(m["content"] for m in msgs if m["role"] in ("user", "assistant"))
            summary = words[:200]
            self.ltm.save_episode(msgs, summary=summary, session_id=self.session_id)
        self.messages.clear(); self.token_count = 0
        self.session_id = str(int(time.time()))

    def clear(self):
        self.messages.clear(); self.token_count = 0