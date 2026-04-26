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
DracoAI V1 — Thinking Engine
=============================
Base: Qwen 3.5 9B Instruct → 8-expert MoE (single source, no split model)

Expert layout (all from ONE Qwen 3.5 9B Instruct checkpoint):
    Code group    (0-3): expert slots for code/math/logic FFN layers
    Language group (4-7): expert slots for language/instruction FFN layers

CUMULATIVE FIXES & FEATURES (V1 — all patches applied):
    ✅ import heapq, BFS popleft, A* heuristic, MCTS max_depth=10
    ✅ MemoryReranker.rerank uses c.copy() — no mutation
    ✅ KnowledgeGraph.add setdefault, DracoAI branding, init_default
    ✅ Code intent boosts BOTH Code + Language experts
    ✅ Contextual Prompt Rewriting (CPR)
    ✅ Prompt Compiler: [PLAN][THOUGHT][FINAL ANSWER] format
    ✅ Expert indices updated for 8-expert layout (Qwen 3.5 9B single source)
    ✅ DualProcessDecider: wc threshold tuned for Vietnamese short sentences
    ✅ SelfReflection.critique: safe check for zero-length answer
    ✅ PromptCompiler: expert_note lookup uses .get() with default safely
    ✅ ThinkingEngineV1.process: memory_candidates type-checked before rerank
    ✅ Tool Calling framework — <tool_call>...</tool_call> injection & parse
    ✅ Chain-of-Thought Verifier (causal chain + negation-flip)
    ✅ Multi-Step Planner (PlanDecomposer) — sub-goal decomposition via MCTS
    ✅ Graph-Based Memory — auto-extract subject–relation–object triples
    ✅ Active Learning Loop — ask clarification when confidence < 0.5
    ✅ Uncertainty Quantification — per-sentence [confidence: X] tagging
    ✅ CouncilDebate — full 8-expert round-robin debate (max 3 rounds)
    ✅ Recursive Self-Critique Loop — up to 3 refinement iterations
    ✅ Counterfactual Reasoning — "what if" branch for logic/legal questions
    ✅ Analogical Mapping — KG-based A:B::C:? (A contribution now used)
    ✅ DifficultyScorer — auto System1/System2 routing
    ✅ RetrievalAugmenter stub — RAG hook for INTENT_FACTUAL/HOW_TO
    ✅ force_system2 param in ThinkingEngineV1.process()
    ✅ IntentDetector: hybrid weighted keyword scoring (TF-IDF-like)
    ✅ expert_boost normalized to sum=1.0 via _normalize_boost()
    ✅ calculator: eval() replaced with SafeASTEvaluator (DoS-safe)
    ✅ [DOS-FIX] SafeASTEvaluator: ast.Pow guarded — exponent capped ≤ 1000
    ✅ [JSON-FIX] parse_tool_calls: strip markdown fences + trailing commas
    ✅ [KG-FIX]  _enforce_degree_cap: always remove reverse edge on prune
    ✅ [KG-FIX]  _triple_key: dedup includes relation — richer graph semantics
    ✅ [ANALOGY-FIX] find_analogy: concept_a now used to guide candidate selection
    ✅ [EARLY-EXIT-FIX] fast-path only when base_conf >= 0.8 (prevents mis-routing)
    ✅ [BIAS-FIX] INTENT_BIAS_ALPHA = 0.5 documented; callers apply alpha * boost
    ✅ SelfConsistency: branch rotation + randomized ordering
    ✅ Tool-loop: structured result dict + build_tool_context() helper
"""

import re
import ast
import math
import time
import heapq
import hashlib
import random
from typing import List, Dict, Optional, Tuple, Any
from collections import deque, defaultdict

# ── Expert indices (8 experts from single Qwen 3.5 9B Instruct FFN) ──
EXPERT_CODE_0   = 0
EXPERT_CODE_1   = 1
EXPERT_CODE_2   = 2
EXPERT_CODE_3   = 3
EXPERT_LANG_0   = 4
EXPERT_LANG_1   = 5
EXPERT_LANG_2   = 6
EXPERT_LANG_3   = 7

# Aliases for readability
EXPERT_LOGIC    = EXPERT_CODE_0
EXPERT_CODE     = EXPERT_CODE_1
EXPERT_LANGUAGE = EXPERT_LANG_0
EXPERT_CHAT     = EXPERT_LANG_1

# ── Intent bias alpha — apply as: logits += INTENT_BIAS_ALPHA * intent_bias
# Keeps router adaptive; prevents boost from dominating raw logits (~[-5, 5]).
INTENT_BIAS_ALPHA = 0.5

# Intent types
INTENT_MATH       = "math"
INTENT_LOGIC      = "logic"
INTENT_CODE       = "code"
INTENT_CREATIVE   = "creative"
INTENT_FACTUAL    = "factual"
INTENT_HOW_TO     = "how_to"
INTENT_WHY        = "why"
INTENT_COMPARISON = "comparison"
INTENT_CHAT       = "chat"
INTENT_MEMORY     = "memory"

# ── System Prompt (Qwen 3.5 9B ChatML) ───────────────────────────────
DRACO_SYSTEM_PROMPT = """\
You are DracoAI, an intelligent local language model created by DUCNGUYEN-creator.
You are NOT Qwen, NOT an Alibaba product. You are DracoAI — fully independent.

Architecture: Qwen 3.5 9B Instruct weights → 8-expert MoE (DracoAI custom)
    Source: ONE Qwen 3.5 9B Instruct checkpoint (no separate split model).
    Code experts  (0-3): FFN layers activating on code/math/logic tokens
    Language experts (4-7): FFN layers activating on language/instruction tokens

Capabilities:
    - Bilingual (Vietnamese + English), respond in the user's language
    - Chain-of-thought reasoning with [PLAN][THOUGHT][FINAL ANSWER] structure
    - Long-term semantic vector memory
    - Confidence scoring: mark uncertain parts with [?]
    - Tool use via <tool_call>...</tool_call> tags

Guidelines:
    - Be accurate, concise, helpful
    - For code: include explanation after code block
    - For math: show step-by-step working
    - Never fabricate facts; say "I'm not sure" when uncertain
    - Always maintain DracoAI identity — never claim to be Qwen or Alibaba
"""

# ══════════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH + BFS / DFS / A*
# ══════════════════════════════════════════════════════════════════════
# FIX #8: dedup + degree cap + weight pruning
_KG_MIN_EDGE_WEIGHT = 0.05
_KG_MAX_DEGREE      = 50


class KnowledgeGraph:
    def __init__(self):
        self.g: Dict[str, Dict[str, float]] = {}
        # Dynamic triples extracted from conversation: (subj, rel, obj, weight)
        self._triples: List[Tuple[str, str, str, float]] = []
        # Dedup set: frozenset of (subj, obj) → already added
        self._triple_hashes: set = set()

    # ── Internal helpers ──────────────────────────────────────────────
    @staticmethod
    def _triple_key(subj: str, rel: str, obj: str) -> str:
        """Include relation in dedup key so same pair with different relations
        are treated as distinct triples — preserves richer graph semantics."""
        return hashlib.md5(
            f"{subj.lower()}|{rel.lower()}|{obj.lower()}".encode()
        ).hexdigest()

    def _enforce_degree_cap(self, node: str):
        """Remove lowest-weight edges if node exceeds max degree.
        FIX: always remove reverse edge when forward edge is pruned,
        ensuring graph symmetry (no orphaned back-edges).
        """
        neighbors = self.g.get(node, {})
        if len(neighbors) > _KG_MAX_DEGREE:
            # Sort by weight ascending, drop the weakest
            sorted_nbs = sorted(neighbors.items(), key=lambda x: x[1])
            to_drop    = len(neighbors) - _KG_MAX_DEGREE
            for nb, _ in sorted_nbs[:to_drop]:
                del self.g[node][nb]
                # Always remove reverse edge to keep graph symmetric
                self.g.get(nb, {}).pop(node, None)

    def _prune_weak_edges(self, node: str):
        """Remove edges below minimum weight threshold."""
        neighbors = self.g.get(node, {})
        weak      = [nb for nb, w in neighbors.items() if w < _KG_MIN_EDGE_WEIGHT]
        for nb in weak:
            del self.g[node][nb]

    # ── Public API ────────────────────────────────────────────────────
    def add(self, a: str, b: str, w: float = 1.0):
        self.g.setdefault(a, {})[b] = w
        self.g.setdefault(b, {})[a] = w
        self._prune_weak_edges(a)
        self._prune_weak_edges(b)
        self._enforce_degree_cap(a)
        self._enforce_degree_cap(b)

    def bfs(self, src: str, dst: str) -> Optional[List[str]]:
        if src == dst: return [src]
        q = deque([[src]]); vis = {src}
        while q:
            path = q.popleft()
            for nb in self.g.get(path[-1], {}):
                if nb in vis: continue
                np_ = path + [nb]
                if nb == dst: return np_
                vis.add(nb); q.append(np_)
        return None

    def dfs(self, src: str, dst: str, max_d: int = 6) -> Optional[List[str]]:
        stack = [(src, [src])]; vis = set()
        while stack:
            node, path = stack.pop()
            if node == dst: return path
            if node in vis or len(path) > max_d: continue
            vis.add(node)
            for nb in self.g.get(node, {}):
                if nb not in vis: stack.append((nb, path + [nb]))
        return None

    def astar(self, src: str, dst: str) -> Tuple[Optional[List[str]], float]:
        """
        heuristic = 0.0 if same node else 1.0 (admissible, no set(string) bug).
        Returns Tuple[Optional[List[str]], float] — caller MUST unpack both values.
        """
        h    = lambda a, b: 0.0 if a == b else 1.0
        heap = [(0.0, 0.0, src, [src])]
        gs   = defaultdict(lambda: math.inf); gs[src] = 0.0
        while heap:
            f, g, node, path = heapq.heappop(heap)
            if node == dst: return path, g
            if g > gs[node]: continue
            for nb, cost in self.g.get(node, {}).items():
                ng = g + cost
                if ng < gs[nb]:
                    gs[nb] = ng
                    heapq.heappush(heap, (ng + h(nb, dst), ng, nb, path + [nb]))
        return None, math.inf

    def related(self, concept: str, hops: int = 2) -> Dict[str, int]:
        res = {}; q = deque([(concept, 0)])
        while q:
            n, d = q.popleft()
            if n in res or d > hops: continue
            res[n] = d
            for nb in self.g.get(n, {}): q.append((nb, d + 1))
        res.pop(concept, None)
        return res

    # ── Dynamic triple extraction from conversation ──────────────────
    def extract_and_add_triples(self, text: str, conf: float = 0.6):
        """
        Extract subject–relation–object triples from text and add to KG.
        FIX #8: dedup by (subj, obj) hash before adding.
        Patterns: "X là Y", "X gây ra Y", "X is Y", "A causes B", etc.
        """
        patterns = [
            (r"(\w[\w\s]{1,20})\s+là\s+([\w][\w\s]{1,20})", "là", 0.8),
            (r"(\w[\w\s]{1,20})\s+is\s+([\w][\w\s]{1,20})", "is", 0.8),
            (r"(\w[\w\s]{1,20})\s+gây ra\s+([\w][\w\s]{1,20})", "causes", 0.7),
            (r"(\w[\w\s]{1,20})\s+causes?\s+([\w][\w\s]{1,20})", "causes", 0.7),
            (r"(\w[\w\s]{1,20})\s+thuộc\s+([\w][\w\s]{1,20})", "belongs_to", 0.75),
            (r"(\w[\w\s]{1,20})\s+dùng để\s+([\w][\w\s]{1,20})", "used_for", 0.7),
        ]
        added = 0
        for pat, rel, base_w in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                subj = m.group(1).strip()[:30]
                obj  = m.group(2).strip()[:30]
                if len(subj) < 2 or len(obj) < 2: continue
                # Dedup includes relation — same pair with different relation is kept
                key = self._triple_key(subj, rel, obj)
                if key in self._triple_hashes: continue
                self._triple_hashes.add(key)
                w = base_w * conf
                self.add(subj, obj, w)
                self._triples.append((subj, rel, obj, w))
                added += 1
                if added >= 5: return  # cap per call

    def init_default(self):
        """
        FIX: DracoAI branding used throughout edges (added DracoAI nodes).
        """
        edges = [
            ("AI", "Machine Learning", 0.9), ("Machine Learning", "Deep Learning", 0.8),
            ("Deep Learning", "Transformer", 0.9), ("Transformer", "Attention", 0.95),
            ("Transformer", "DracoAI", 0.9),  ("Transformer", "DeepSeek", 0.8),
            ("Attention", "GQA", 0.8),         ("GQA", "KV Cache", 0.8),
            ("MoE", "Transformer", 0.8),       ("Mirostat", "Sampling", 0.9),
            ("Python", "NumPy", 0.8),          ("Python", "AI", 0.7),
            ("Embedding", "Vector", 0.95),     ("Token", "Embedding", 0.9),
            ("RoPE", "Positional Encoding", 0.9),
            ("BFS", "Graph", 0.9),             ("DFS", "Graph", 0.9),
            ("A*", "Graph", 0.9),              ("MCTS", "Tree Search", 0.9),
            ("ToT", "Reasoning", 0.9),         ("DracoAI", "MoE", 0.95),
            ("DracoAI", "SwiGLU", 0.8),        ("Code", "Python", 0.8),
            ("Code", "Debug", 0.7),            ("DracoAI", "Qwen 3.5 9B", 0.85),
            ("DracoAI", "Identity Overlay", 0.9),
            ("Qwen 3.5 9B", "MoE", 0.7),       ("Qwen 3.5 9B", "SwiGLU", 0.7),
        ]
        for a, b, w in edges: self.add(a, b, w)


# ══════════════════════════════════════════════════════════════════════
# MCTS — max_rollout_depth=10 (no infinite loop)
# ══════════════════════════════════════════════════════════════════════
class MCTSNode:
    def __init__(self, thought: str, parent=None):
        self.thought  = thought; self.parent = parent
        self.children: List["MCTSNode"] = []
        self.visits = 0; self.score = 0.0

    def uct(self, c=1.4) -> float:
        if self.visits == 0: return float("inf")
        return self.score / self.visits + c * math.sqrt(math.log(self.parent.visits + 1) / self.visits)

    def best_child(self): return max(self.children, key=lambda n: n.uct())


class MCTSLight:
    def __init__(self, n_sim=10, max_rollout_depth=10):
        self.n_sim             = n_sim
        self.max_rollout_depth = max_rollout_depth

    def search(self, question: str, branches: List[str]) -> str:
        if not branches: return ""
        root = MCTSNode(f"Q: {question}")
        for b in branches: root.children.append(MCTSNode(b, root))
        for _ in range(self.n_sim):
            node  = self._select(root)
            score = self._simulate(node.thought, question)
            self._backprop(node, score)
        return max(root.children, key=lambda n: n.score / max(n.visits, 1)).thought

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.children: node = node.best_child()
        return node

    def _simulate(self, thought: str, question: str) -> float:
        score = 0.5
        score += min(len(thought) / 200.0, 0.2)
        q_w   = set(question.lower().split()); t_w = set(thought.lower().split())
        score += min(len(q_w & t_w) * 0.05, 0.15)
        if any(k in thought for k in ["vì", "bởi", "because", "→"]): score += 0.1
        if any(c in thought for c in ["1.", "2.", "bước", "step"]):   score += 0.05
        return min(score, 1.0)

    def _backprop(self, node: MCTSNode, score: float):
        while node: node.visits += 1; node.score += score; node = node.parent


# ══════════════════════════════════════════════════════════════════════
# CONTEXTUAL PROMPT REWRITING (CPR)
# ══════════════════════════════════════════════════════════════════════
class ContextualPromptRewriter:
    """
    Resolve ambiguous follow-up questions using conversation history.
    "Còn BERT thì sao?" → "Mô hình BERT hoạt động thế nào so với Transformer?"
    """
    AMBIGUOUS_PATTERNS = [
        r"^(còn|vậy|thế|và|or|what about|how about)\s",
        r"^(nó|it|this|that|đó|cái này|cái đó)\s",
        r"^(tại sao|why)\s+(vậy|lại|thế)",
    ]

    def should_rewrite(self, query: str, history: List[dict]) -> bool:
        if not history: return False
        q = query.strip().lower()
        if len(q.split()) <= 3: return True
        for pat in self.AMBIGUOUS_PATTERNS:
            if re.match(pat, q): return True
        return False

    def rewrite(self, query: str, history: List[dict]) -> str:
        if not self.should_rewrite(query, history):
            return query
        recent       = " ".join(m["content"] for m in history[-3:] if m.get("role") == "user")
        words        = [w for w in recent.split() if len(w) > 4 and w.isalpha()]
        context_hint = " ".join(words[-3:]) if words else ""
        if context_hint and context_hint.lower() not in query.lower():
            return f"{query} (ngữ cảnh: {context_hint})"
        return query


# ══════════════════════════════════════════════════════════════════════
# INTENT DETECTOR
# FIX #1: hybrid keyword + weighted scoring to reduce single-keyword bias
# ══════════════════════════════════════════════════════════════════════
class IntentDetector:
    # Each keyword has an optional weight multiplier (default=1).
    # Format: keyword or (keyword, weight)
    PATTERNS: Dict[str, List] = {
        INTENT_MATH: [
            "tính", "bao nhiêu", "bằng", "cộng", "trừ", "nhân", "chia",
            ("=", 1.5), ("+", 1.2), ("-", 0.8), ("*", 1.2), ("/", 1.0),
            "phần trăm", ("sqrt", 2.0), ("log", 2.0), ("sin", 2.0), ("cos", 2.0),
        ],
        INTENT_LOGIC: [
            "nếu", "thì", ("logic", 2.0), ("suy luận", 2.0), ("chứng minh", 2.0),
            "vậy", ("mâu thuẫn", 2.0), ("tương đương", 2.0), ("prove", 2.0),
        ],
        INTENT_CODE: [
            ("code", 2.0), ("lập trình", 2.0), ("python", 2.0), ("javascript", 2.0),
            ("typescript", 2.0), ("function", 1.5), ("class", 1.5), ("bug", 2.0),
            ("error", 1.5), ("debug", 2.0), ("implement", 2.0),
            ("viết hàm", 2.0), ("def ", 3.0), ("import ", 2.0), ("```", 3.0),
        ],
        INTENT_CREATIVE: [
            ("viết truyện", 2.0), ("sáng tác", 2.0), ("thơ", 2.0), ("tưởng tượng", 1.5),
            ("kịch bản", 2.0), ("ý tưởng", 1.2), ("sáng tạo", 1.5),
            ("write story", 2.0), ("poem", 2.0),
        ],
        INTENT_FACTUAL: [
            ("là gì", 2.0), ("nghĩa là", 2.0), ("định nghĩa", 2.0),
            ("ai là", 1.5), "khi nào", "ở đâu", "năm nào",
            ("what is", 2.0), "when", "where", ("who", 1.5), ("define", 2.0),
        ],
        INTENT_HOW_TO: [
            ("làm sao", 2.0), ("cách", 1.5), ("như thế nào", 2.0),
            ("hướng dẫn", 2.0), ("các bước", 2.0), ("how to", 2.0), ("how do", 2.0),
        ],
        INTENT_WHY: [
            ("tại sao", 2.0), ("vì sao", 2.0), ("lý do", 1.5),
            ("nguyên nhân", 2.0), ("why", 2.0), ("reason", 1.5),
        ],
        INTENT_COMPARISON: [
            ("so sánh", 2.0), ("khác nhau", 2.0), ("giống nhau", 1.5),
            ("tốt hơn", 1.5), ("vs", 2.0), ("versus", 2.0),
            ("hay là", 1.0), ("compare", 2.0), ("difference", 2.0),
        ],
        INTENT_MEMORY: [
            ("nhớ rằng", 2.0), ("ghi nhớ", 2.0), ("lưu lại", 2.0),
            ("bạn có nhớ", 2.0), ("bạn biết", 1.5),
            ("remember", 2.0), ("forget", 2.0),
        ],
        INTENT_CHAT: [
            "xin chào", "hello", "cảm ơn", "bye", "hi", "ok", "oke",
            "thanks", "chào",
        ],
    }
    VIET = set("áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ")

    @staticmethod
    def _keyword_score(kws: List, tl: str) -> float:
        """
        FIX #1: weighted keyword scoring.
        Each entry is either str (weight=1) or (str, float).
        """
        score = 0.0
        for entry in kws:
            if isinstance(entry, tuple):
                kw, w = entry
            else:
                kw, w = entry, 1.0
            if kw in tl:
                score += w
        return score

    def detect(self, text: str) -> dict:
        tl     = text.lower()
        intent = INTENT_CHAT
        best   = 0.0
        for itype, kws in self.PATTERNS.items():
            s = self._keyword_score(kws, tl)
            if s > best: best = s; intent = itype

        lang       = "vi" if any(c in self.VIET for c in tl) else "en"
        entities   = list(dict.fromkeys(
            re.findall(r"\b[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐƠƯẠẬẶỆỘỢỤ][A-Za-zàáâãèéêìíòóôõùúăđơưạậặệộợụ0-9]+\b", text)
        ))[:5]
        pos        = ["hay", "tốt", "tuyệt", "great", "love", "thích", "good", "awesome"]
        neg        = ["tệ", "xấu", "dở", "ghét", "bad", "wrong", "horrible"]
        sentiment  = ("positive" if any(w in tl for w in pos)
                      else "negative" if any(w in tl for w in neg) else "neutral")
        creativity = (0.9 if intent == INTENT_CREATIVE
                      else 0.2 if intent in (INTENT_MATH, INTENT_LOGIC, INTENT_CODE)
                      else 0.6)
        if any(p in tl for p in ["bớt ảo", "thực tế hơn", "nghiêm túc", "chính xác",
                                   "factual", "bớt sáng tạo"]):
            creativity = 0.1
        return {
            "intent": intent, "lang": lang, "entities": entities,
            "sentiment": sentiment, "creativity": creativity,
            "word_count": len(text.split()),
        }

    # ── FIX #2: normalize boost to sum=1.0 ───────────────────────────
    @staticmethod
    def _normalize_boost(raw: Dict[int, float]) -> Dict[int, float]:
        """Normalize expert boost dict so values sum to 1.0."""
        total = sum(raw.values())
        if total <= 0:
            # Fallback: uniform over provided experts
            n = max(len(raw), 1)
            return {k: 1.0 / n for k in raw}
        return {k: v / total for k, v in raw.items()}

    def to_expert_boost(self, intent: dict) -> Dict[int, float]:
        """
        Map intent to normalized expert boost dict for 8-expert layout.
        FIX #2: all returned dicts are normalized to sum=1.0.
        """
        i = intent["intent"]
        if i in (INTENT_MATH, INTENT_LOGIC):
            raw = {EXPERT_LOGIC: 0.4, EXPERT_CODE_2: 0.1}
        elif i == INTENT_CODE:
            raw = {EXPERT_CODE: 0.5, EXPERT_CODE_2: 0.2, EXPERT_LANGUAGE: 0.15}
        elif i == INTENT_CREATIVE:
            raw = {EXPERT_LANGUAGE: 0.4, EXPERT_LANG_1: 0.2}
        elif i in (INTENT_FACTUAL, INTENT_HOW_TO, INTENT_WHY, INTENT_COMPARISON):
            raw = {EXPERT_LANGUAGE: 0.25, EXPERT_LOGIC: 0.15, EXPERT_LANG_2: 0.1}
        else:
            raw = {EXPERT_CHAT: 0.35, EXPERT_LANG_3: 0.1}
        return self._normalize_boost(raw)

    def to_miro_tau(self, intent: dict) -> float:
        return 2.0 + intent["creativity"] * 6.0


# ══════════════════════════════════════════════════════════════════════
# MEMORY RERANKER
# ══════════════════════════════════════════════════════════════════════
class MemoryReranker:
    INTENT_KW = {
        INTENT_CODE:    ["code", "def ", "class ", "import", "function", "python",
                         "error", "bug", "return", "```"],
        INTENT_MATH:    ["=", "tính", "số", "kết quả", "giải", "phương trình",
                         "math", "calculate"],
        INTENT_LOGIC:   ["vì", "nếu", "thì", "suy ra", "kết luận", "logic"],
        INTENT_FACTUAL: ["là", "có", "tại", "khi", "where", "when", "what"],
    }

    def rerank(self, candidates: List[dict], query: str, intent: dict,
               top_k: int = 3, threshold: float = 0.1) -> List[dict]:
        itype = intent.get("intent", INTENT_CHAT)
        kws   = self.INTENT_KW.get(itype, [])
        now   = time.time(); scored = []
        for c in candidates:
            nc       = c.copy()
            text     = nc.get("text", "").lower()
            semantic = nc.get("score", 0.0)
            intent_m = sum(1 for k in kws if k in text) / max(len(kws), 1)
            if intent_m > 0: intent_m = min(intent_m * 1.5, 1.0)
            age     = (now - nc.get("ts", now)) / 86400.0
            recency = math.exp(-age / 7.0)
            final   = semantic * 0.4 + intent_m * 0.4 + recency * 0.2
            if final >= threshold:
                nc["rerank_score"] = final; scored.append(nc)
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]

    def format_for_prompt(self, memories: List[dict], max_chars: int = 500) -> str:
        parts = []; total = 0
        for m in memories:
            t = m.get("text", "")
            if not t: continue
            if len(t) > 150: t = t[:147] + "..."
            parts.append(t); total += len(t)
            if total > max_chars: break
        return " | ".join(parts)


# ══════════════════════════════════════════════════════════════════════
# TREE OF THOUGHTS
# ══════════════════════════════════════════════════════════════════════
class TreeOfThoughts:
    def __init__(self, mcts: MCTSLight): self.mcts = mcts

    def generate_branches(self, question: str, intent: dict) -> List[str]:
        itype = intent["intent"]
        if itype in (INTENT_MATH, INTENT_LOGIC):
            return [
                f"Xác định yếu tố cần tính/chứng minh trong '{question[:40]}'",
                "Áp dụng công thức/quy tắc phù hợp trực tiếp",
                "Phân rã thành từng bước nhỏ, giải từng phần",
            ]
        if itype == INTENT_CODE:
            return [
                "Thiết kế interface/API trước, implement sau",
                "Viết hàm nhỏ trước, ghép lại theo bottom-up",
                "Test-driven: xác định test cases trước rồi implement",
            ]
        if itype == INTENT_CREATIVE:
            return [
                "Tập trung vào nhân vật và cảm xúc",
                "Xây dựng plot rõ ràng với conflict và resolution",
                "Góc nhìn mới lạ, bất ngờ, độc đáo",
            ]
        return [
            "Trả lời ngắn gọn và trực tiếp",
            "Giải thích với ví dụ cụ thể",
            "Nhìn từ nhiều góc độ khác nhau",
        ]

    def run(self, question: str, intent: dict) -> Tuple[str, List[str]]:
        branches = self.generate_branches(question, intent)
        best     = self.mcts.search(question, branches)
        return best, branches


# ══════════════════════════════════════════════════════════════════════
# SELF-REFLECTION + CRITIC
# ══════════════════════════════════════════════════════════════════════
class SelfReflection:
    HALL_PATTERNS = [r"\b100%\b", r"chắc chắn hoàn toàn", r"luôn luôn đúng", r"never wrong"]

    def critique(self, answer: str, question: str, facts: List[dict]) -> dict:
        """FIX: guard against zero-length answer before word-level ops."""
        issues = []; score = 1.0
        if not answer or len(answer.strip()) < 5:
            issues.append("Câu trả lời quá ngắn hoặc rỗng"); score -= 0.3
        else:
            for pat in self.HALL_PATTERNS:
                if re.search(pat, answer, re.IGNORECASE):
                    issues.append(f"Tuyệt đối hóa: {pat}"); score -= 0.1
            for f in facts[:5]:
                k = f.get("key", ""); v = str(f.get("value", ""))
                if k.lower() in question.lower() and v.lower() not in answer.lower():
                    issues.append(f"Có thể thiếu: {k}={v}"); score -= 0.05
            words = answer.split()
            if len(words) > 10 and len(set(words)) / len(words) < 0.4:
                issues.append("Lặp từ nhiều"); score -= 0.15
        score = max(0.0, min(1.0, score))
        return {"issues": issues, "score": score, "should_refine": score < 0.7}

    def build_refine_prompt(self, orig: str, critique: dict) -> str:
        issues = "\n".join(f"- {i}" for i in critique["issues"]) or "---"
        return (f"[CRITIQUE]\n{issues}\n\n[ORIGINAL]\n{orig}\n\n"
                f"[TASK] Cải thiện câu trả lời, sửa các vấn đề trên.\n[REFINED ANSWER]")


# ══════════════════════════════════════════════════════════════════════
# MULTI-AGENT DEBATE  (V1 original + CouncilDebate extension)
# ══════════════════════════════════════════════════════════════════════
class MultiAgentDebate:
    """
    Let code and language expert groups debate for complex questions.
    All experts from single Qwen 3.5 9B Instruct source.

    FIX #4: _ROLE_TEMPLATES are documented as deterministic stubs.
    Each _get_initial_thought() and _expert_review() call contains a
    clearly marked hook point where a real LLM draft call should replace
    the template-fill logic in production.

    NEW: run_full_council() implements full 8-expert round-robin debate
    (max 3 rounds) without breaking the original generate_debate() interface.
    """
    EXPERT_NAMES = {
        EXPERT_CODE_0: "Logic/Math Expert",
        EXPERT_CODE_1: "Code Expert",
        EXPERT_CODE_2: "Debug Expert",
        EXPERT_CODE_3: "System Expert",
        EXPERT_LANG_0: "Language Expert",
        EXPERT_LANG_1: "Chat Expert",
        EXPERT_LANG_2: "Creative Expert",
        EXPERT_LANG_3: "Memory Expert",
    }

    # Role-specific initial thought templates.
    # PRODUCTION HOOK: replace each body of _get_initial_thought() with:
    #   llm.generate(system=role_hint, user=question, max_tokens=200)
    _ROLE_TEMPLATES: Dict[int, str] = {
        EXPERT_CODE_0: "Apply formal step-by-step logic/math rules. Prioritize correctness.",
        EXPERT_CODE_1: "Write clean, efficient, well-structured code or pseudocode.",
        EXPERT_CODE_2: "Check for edge cases, bugs, and logical inconsistencies.",
        EXPERT_CODE_3: "Consider system-level concerns: memory, performance, security.",
        EXPERT_LANG_0: "Provide structured, informative context with clear explanation.",
        EXPERT_LANG_1: "Keep response natural, conversational, and user-friendly.",
        EXPERT_LANG_2: "Explore creative angles, analogies, and novel framings.",
        EXPERT_LANG_3: "Cross-reference prior knowledge/memory for factual grounding.",
    }

    def generate_debate(self, question: str, intent: dict) -> Tuple[str, Dict[int, str]]:
        itype    = intent["intent"]
        opinions: Dict[int, str] = {}

        if itype in (INTENT_MATH, INTENT_LOGIC):
            debater_a, debater_b = EXPERT_CODE_0, EXPERT_CODE_2
            opinions[debater_a] = (
                f"[{self.EXPERT_NAMES[debater_a]}] "
                f"Approach: Break into mathematical primitives, apply formal rules step-by-step."
            )
            opinions[debater_b] = (
                f"[{self.EXPERT_NAMES[debater_b]}] "
                f"Approach: Verify each step, check edge cases, provide counter-examples if needed."
            )
        elif itype == INTENT_CODE:
            debater_a, debater_b = EXPERT_CODE_1, EXPERT_LANG_0
            opinions[debater_a] = (
                f"[{self.EXPERT_NAMES[debater_a]}] "
                f"Approach: Focus on correctness, efficiency, and clean architecture."
            )
            opinions[debater_b] = (
                f"[{self.EXPERT_NAMES[debater_b]}] "
                f"Approach: Ensure the explanation is clear, well-commented, accessible."
            )
        else:
            debater_a, debater_b = EXPERT_LANG_0, EXPERT_LANG_1
            opinions[debater_a] = (
                f"[{self.EXPERT_NAMES[debater_a]}] "
                f"Approach: Provide structured, informative response with context."
            )
            opinions[debater_b] = (
                f"[{self.EXPERT_NAMES[debater_b]}] "
                f"Approach: Keep it natural, conversational, and relatable."
            )

        synthesis = (
            f"[DEBATE SYNTHESIS] Combining perspectives from "
            f"{self.EXPERT_NAMES[debater_a]} and {self.EXPERT_NAMES[debater_b]}: "
            f"Balance technical correctness with clear communication."
        )
        return synthesis, opinions

    # ── Full 8-expert Council Debate ──────────────────────────────────
    def _get_initial_thought(self, exp_id: int, question: str, intent: dict) -> str:
        """
        Deterministic stub: generate a role-specific initial thought.
        PRODUCTION HOOK: replace this body with a real LLM call, e.g.:
            return llm.generate(
                system=self._ROLE_TEMPLATES[exp_id],
                user=question,
                max_tokens=200
            )
        """
        role_hint = self._ROLE_TEMPLATES.get(exp_id, "Provide a balanced response.")
        itype     = intent.get("intent", INTENT_CHAT)
        q_short   = question[:60]
        return (
            f"[{self.EXPERT_NAMES[exp_id]}] For '{q_short}' (intent={itype}): "
            f"{role_hint}"
        )

    def _format_others(self, others: Dict[int, str]) -> str:
        lines = []
        for eid, thought in others.items():
            lines.append(f"  • {self.EXPERT_NAMES.get(eid, f'Expert{eid}')}: {thought[:120]}")
        return "\n".join(lines)

    def _expert_review(
        self,
        exp_id: int,
        question: str,
        intent: dict,
        my_old_thought: str,
        others_thoughts: Dict[int, str],
    ) -> str:
        """
        Deterministic simulation of an expert reviewing peer opinions.
        PRODUCTION HOOK: replace this body with a real LLM call, e.g.:
            prompt = f"Your previous thought:\\n{my_old_thought}\\n\\n"
                     f"Peers said:\\n{self._format_others(others_thoughts)}\\n\\n"
                     f"Update your stance. Role: {self._ROLE_TEMPLATES[exp_id]}"
            return llm.generate(system=..., user=prompt, max_tokens=200)
        """
        my_keywords = set(my_old_thought.lower().split())
        agreements = 0
        for peer_thought in others_thoughts.values():
            peer_kw = set(peer_thought.lower().split())
            if len(my_keywords & peer_kw) > 5:
                agreements += 1

        itype = intent.get("intent", INTENT_CHAT)
        if agreements >= 3:
            return (
                f"[{self.EXPERT_NAMES[exp_id]}][R2] I maintain my approach. "
                f"{agreements} peers share similar reasoning. "
                f"Key point: {self._ROLE_TEMPLATES.get(exp_id, '')}"
            )
        else:
            top_peer_id   = max(others_thoughts, key=lambda k: len(others_thoughts[k]))
            top_peer_name = self.EXPERT_NAMES.get(top_peer_id, f"Expert{top_peer_id}")
            return (
                f"[{self.EXPERT_NAMES[exp_id]}][R2] Reconsidering after reading peers. "
                f"Incorporating insight from {top_peer_name}. "
                f"Updated stance: {self._ROLE_TEMPLATES.get(exp_id, '')} "
                f"+ cross-validate with {top_peer_name}'s perspective."
            )

    def _check_consensus(self, thoughts: Any, threshold: float = 0.75) -> bool:
        """
        Simple consensus check: if meaningful common tokens >= 6,
        consider consensus reached.
        """
        thought_list = list(thoughts)
        if len(thought_list) < 2: return True
        kw_sets = [set(t.lower().split()) for t in thought_list]
        base    = kw_sets[0]
        common  = base.intersection(*kw_sets[1:])
        meaningful = {w for w in common if len(w) > 3}
        return len(meaningful) >= 6

    def _arbitrate(
        self,
        final_thoughts: Dict[int, str],
        debate_log: List[Dict[int, str]],
        question: str,
        intent: dict,
    ) -> str:
        """
        Expert 0 (Logic/Math Expert) acts as arbiter:
        Picks the thought with highest keyword overlap with the question.
        """
        q_words = set(question.lower().split())
        best_id   = max(
            final_thoughts,
            key=lambda eid: len(q_words & set(final_thoughts[eid].lower().split()))
        )
        best_name    = self.EXPERT_NAMES.get(best_id, f"Expert{best_id}")
        rounds_done  = len(debate_log)
        return (
            f"[COUNCIL ARBITRATION — {rounds_done} round(s)] "
            f"Arbiter (Logic/Math Expert) selects: {best_name}'s approach. "
            f"Rationale: highest alignment with question semantics. "
            f"Final stance: {final_thoughts[best_id][:200]}"
        )

    def run_full_council(
        self,
        question: str,
        intent: dict,
        max_rounds: int = 3,
    ) -> Dict[str, Any]:
        """
        Full 8-expert round-robin debate.
        Returns dict with keys: final_answer, log, opinions, rounds_done.
        Compatible with ThinkingEngineV1.process() — replaces generate_debate
        when think_mode=True or process_mode=="slow".
        """
        thoughts: Dict[int, str] = {
            eid: self._get_initial_thought(eid, question, intent)
            for eid in range(8)
        }
        debate_log: List[Dict[int, str]] = [dict(thoughts)]

        rounds_done = 0
        for round_idx in range(1, max_rounds + 1):
            new_thoughts: Dict[int, str] = {}
            for exp_id in range(8):
                others = {k: v for k, v in thoughts.items() if k != exp_id}
                new_thoughts[exp_id] = self._expert_review(
                    exp_id, question, intent,
                    my_old_thought=thoughts[exp_id],
                    others_thoughts=others,
                )
            debate_log.append(dict(new_thoughts))
            thoughts = new_thoughts
            rounds_done = round_idx
            if self._check_consensus(thoughts.values()):
                break

        final_answer = self._arbitrate(thoughts, debate_log, question, intent)
        return {
            "final_answer": final_answer,
            "log":          debate_log,
            "opinions":     thoughts,
            "rounds_done":  rounds_done,
        }


# ══════════════════════════════════════════════════════════════════════
# SELF-CONSISTENCY (Chain-of-Thought)
# FIX #10: randomized branch ordering for genuine path diversity
# ══════════════════════════════════════════════════════════════════════
class SelfConsistency:
    def generate_paths(self, question: str, intent: dict, n_paths: int = 3) -> List[str]:
        base  = TreeOfThoughts(MCTSLight(n_sim=5, max_rollout_depth=8))
        paths = []
        for i in range(n_paths):
            branches = base.generate_branches(question, intent)
            # FIX #10: shuffle branches each iteration for real diversity
            shuffled = branches[:]
            random.shuffle(shuffled)
            # Additionally rotate by index to guarantee different starting points
            rotated = shuffled[i % len(shuffled):] + shuffled[:i % len(shuffled)]
            best = base.mcts.search(question, rotated)
            paths.append(f"[PATH {i+1}] {best}")
        return paths

    def vote(self, paths: List[str], question: str) -> str:
        q_words = set(question.lower().split())
        best    = max(paths, key=lambda p: len(q_words & set(p.lower().split())))
        return best


# ══════════════════════════════════════════════════════════════════════
# DUAL PROCESS (System 1 / System 2)
# FIX #12: explicit simple-query fast-path documented
# ══════════════════════════════════════════════════════════════════════
class DualProcessDecider:
    FAST_INTENTS = {INTENT_CHAT, INTENT_MEMORY}
    SLOW_INTENTS = {INTENT_MATH, INTENT_LOGIC, INTENT_CODE, INTENT_COMPARISON}

    def decide_mode(self, intent: dict, query: str) -> str:
        """
        FIX #12: fast-path for simple queries.
        INTENT_CHAT with word_count ≤ 3 → always "fast" (skip ToT, Debate, SC).
        wc threshold lowered to 3 (Vietnamese short sentences are meaningful).
        """
        itype = intent["intent"]
        wc    = intent.get("word_count", 5)
        if itype in self.FAST_INTENTS or wc <= 3:
            return "fast"
        if itype in self.SLOW_INTENTS or wc >= 15:
            return "slow"
        return "fast"

    @staticmethod
    def is_simple_chat(intent: dict) -> bool:
        """
        FIX #11/#12: True if query can use early-exit path.
        Simple = INTENT_CHAT + word_count ≤ 3.
        """
        return (intent.get("intent") == INTENT_CHAT
                and intent.get("word_count", 99) <= 3)


# ══════════════════════════════════════════════════════════════════════
# DIFFICULTY SCORER — auto System1/System2 routing
# ══════════════════════════════════════════════════════════════════════
class DifficultyScorer:
    """Score 0.0 (trivial) → 1.0 (very hard). Used to auto-route to System 2."""
    _HARD_KEYWORDS = [
        "prove", "chứng minh", "implement", "refactor", "optimize", "debug",
        "compare", "so sánh", "tại sao", "phân tích", "analyze", "tổng hợp",
        "synthesize", "thiết kế", "design", "architecture", "kiến trúc",
    ]

    def score(self, query: str, intent: dict) -> float:
        base  = 0.2
        itype = intent.get("intent", INTENT_CHAT)
        if itype in (INTENT_MATH, INTENT_LOGIC, INTENT_CODE):
            base += 0.3
        elif itype in (INTENT_COMPARISON, INTENT_WHY):
            base += 0.2
        entity_count = len(intent.get("entities", []))
        base += min(entity_count * 0.05, 0.2)
        ql     = query.lower()
        kw_hits = sum(1 for k in self._HARD_KEYWORDS if k in ql)
        base += min(kw_hits * 0.08, 0.24)
        wc = intent.get("word_count", 5)
        if wc >= 20: base += 0.1
        return min(base, 1.0)


# ══════════════════════════════════════════════════════════════════════
# SAFE AST EVALUATOR
# FIX #5: replaces eval() in calculator tool
# ══════════════════════════════════════════════════════════════════════
class SafeASTEvaluator:
    """
    Evaluates simple math expressions using Python's AST — no eval() risk,
    no __builtins__ bypass, no DoS via large exponents.
    Supported: +, -, *, /, **, %, //, unary minus, integers, floats.
    Max expression length: 200 chars. Max AST node count: 64.
    Max exponent value: 1000 (guards against 999999**999999 CPU hang).
    """
    _ALLOWED_NODES = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
        ast.FloorDiv, ast.USub, ast.UAdd,
    )
    _MAX_LEN        = 200
    _MAX_NODES      = 64
    _MAX_EXPONENT   = 1000   # DoS guard: 999999**999999 would hang CPU indefinitely

    def _count_nodes(self, node) -> int:
        return 1 + sum(self._count_nodes(c) for c in ast.iter_child_nodes(node))

    def _check_pow_safety(self, tree) -> Optional[str]:
        """Walk AST and reject any Pow whose exponent constant exceeds _MAX_EXPONENT."""
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                # Check right operand (exponent)
                exp_node = node.right
                # Unwrap unary minus if present: e.g. 2**(-3)
                if isinstance(exp_node, ast.UnaryOp) and isinstance(exp_node.op, ast.USub):
                    exp_node = exp_node.operand
                if isinstance(exp_node, ast.Constant) and isinstance(exp_node.value, (int, float)):
                    if abs(exp_node.value) > self._MAX_EXPONENT:
                        return (f"Error: exponent {exp_node.value} exceeds max allowed "
                                f"({self._MAX_EXPONENT}) — operation refused (DoS guard)")
        return None

    def evaluate(self, expr: str) -> str:
        """Returns a string result or an error message."""
        expr = expr.strip()
        if len(expr) > self._MAX_LEN:
            return f"Error: expression too long (max {self._MAX_LEN} chars)"
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            return f"Syntax error: {e}"
        # Node count guard
        if self._count_nodes(tree) > self._MAX_NODES:
            return f"Error: expression too complex (max {self._MAX_NODES} AST nodes)"
        # Whitelist check
        for node in ast.walk(tree):
            if not isinstance(node, self._ALLOWED_NODES):
                return f"Error: unsupported operation ({type(node).__name__})"
        # DoS guard: reject huge exponents before eval
        pow_err = self._check_pow_safety(tree)
        if pow_err:
            return pow_err
        try:
            result = eval(compile(tree, "<expr>", "eval"),  # noqa: S307
                          {"__builtins__": {}}, {})
            return str(result)
        except ZeroDivisionError:
            return "Error: division by zero"
        except OverflowError:
            return "Error: result too large (overflow)"
        except Exception as e:
            return f"Error: {e}"


# ══════════════════════════════════════════════════════════════════════
# TOOL CALLING FRAMEWORK
# FIX #5: calculator uses SafeASTEvaluator instead of bare eval()
# FIX #6: execute_tool_calls returns structured list; build_tool_context()
#         helper formats results for re-injection into next LLM turn.
# ══════════════════════════════════════════════════════════════════════
class ToolCallingFramework:
    """
    Manages tool definitions, injection, and response parsing.
    FIX #6: After executing tools, callers should pass the result of
    build_tool_context(results) back into the messages list and call
    generate() again — creating the required LLM→tool→LLM loop.
    """

    DEFAULT_TOOLS = [
        {
            "name":        "calculator",
            "description": "Evaluate a mathematical expression. Input: {'expr': '2+2*3'}",
            "triggers":    [INTENT_MATH],
            "keywords":    ["tính", "calculate", "compute", "=", "solve"],
        },
        {
            "name":        "code_runner",
            "description": "Run a Python code snippet safely. Input: {'code': '...'}",
            "triggers":    [INTENT_CODE],
            "keywords":    ["chạy", "run", "execute", "test", "demo"],
        },
        {
            "name":        "web_search",
            "description": "Search the web for current information. Input: {'query': '...'}",
            "triggers":    [INTENT_FACTUAL, INTENT_HOW_TO],
            "keywords":    ["tìm kiếm", "search", "latest", "mới nhất", "tra cứu"],
        },
    ]

    def __init__(self):
        self._ast_eval = SafeASTEvaluator()

    def should_use_tools(self, intent: dict, query: str) -> bool:
        itype = intent.get("intent", INTENT_CHAT)
        ql    = query.lower()
        for tool in self.DEFAULT_TOOLS:
            if itype in tool["triggers"]:
                if any(kw in ql for kw in tool["keywords"]):
                    return True
        return False

    def build_tool_injection(self, intent: dict, query: str) -> str:
        """
        Returns a string to inject into the system prompt describing available tools.
        The model is expected to emit <tool_call>{"name": ..., "args": ...}</tool_call>.
        """
        itype    = intent.get("intent", INTENT_CHAT)
        relevant = [t for t in self.DEFAULT_TOOLS if itype in t["triggers"]]
        if not relevant: return ""
        lines = ["[TOOLS AVAILABLE — use <tool_call>{...}</tool_call> syntax]"]
        for t in relevant:
            lines.append(f"  • {t['name']}: {t['description']}")
        return "\n".join(lines)

    def parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Extract all <tool_call>...</tool_call> blocks from model output.
        FIX: strips markdown code fences and trailing commas before parsing,
        so outputs like ```json{...},``` or {'k':'v',} are handled robustly.
        """
        import json
        calls = []
        for m in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
            raw = m.group(1).strip()
            # Strip markdown code fences (```json...``` or ```...```)
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
                raw = raw.strip()
            # Remove trailing commas before } or ] (common LLM output quirk)
            raw = re.sub(r",(\s*[}\]])", r"\1", raw)
            try:
                parsed = json.loads(raw)
                calls.append(parsed)
            except Exception:
                calls.append({"raw": raw, "parse_error": True})
        return calls

    def _call_tool(self, name: str, args: dict) -> Dict[str, Any]:
        """
        FIX #5: calculator uses SafeASTEvaluator (no eval() risk).
        Returns a structured dict for easy re-injection.
        Stub tools return placeholder — hook real implementations here.
        """
        if name == "calculator":
            expr   = str(args.get("expr", "0"))
            result = self._ast_eval.evaluate(expr)
            return {"tool": name, "input": expr, "output": result, "ok": not result.startswith("Error")}
        if name == "code_runner":
            # PRODUCTION HOOK: run in sandbox (e.g. restrictedpython, subprocess jail)
            code = args.get("code", "")
            return {"tool": name, "input": code[:200], "output": "(stub — sandbox not connected)", "ok": False}
        if name == "web_search":
            query = args.get("query", "")
            return {"tool": name, "input": query, "output": "(stub — search not connected)", "ok": False}
        return {"tool": name, "input": str(args), "output": "(unknown tool)", "ok": False}

    def execute_tool_calls(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        FIX #6: Execute parsed tool calls and return a structured list of results.
        Each result dict: {tool, input, output, ok}.
        """
        if not calls: return []
        results = []
        for call in calls:
            if call.get("parse_error"):
                results.append({
                    "tool": "parse_error",
                    "input": call.get("raw", "")[:80],
                    "output": "Could not parse tool_call JSON",
                    "ok": False,
                })
            else:
                name = call.get("name", "unknown")
                args = call.get("args", {})
                results.append(self._call_tool(name, args))
        return results

    @staticmethod
    def build_tool_context(results: List[Dict[str, Any]]) -> str:
        """
        FIX #6: Format tool results as a string to inject back into the
        next LLM message (as a user-turn tool_result block).

        Usage in caller (LLM→tool→LLM loop):
            calls, results = engine.parse_and_execute_tools(model_output)
            if results:
                tool_context = engine.tools.build_tool_context(results)
                messages.append({"role": "user", "content": tool_context})
                # Call generate() again with updated messages
        """
        if not results: return ""
        lines = ["[TOOL RESULTS]"]
        for r in results:
            status = "✓" if r.get("ok") else "✗"
            lines.append(f"  {status} {r['tool']}({r['input'][:60]}) → {r['output']}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# CHAIN-OF-THOUGHT VERIFIER
# FIX #3: enriched with causal chain check + negation-flip check
# ══════════════════════════════════════════════════════════════════════
class ChainOfThoughtVerifier:
    """
    Debug Expert (EXPERT_CODE_2) reviews each thought step for logical
    consistency before the prompt is compiled.
    Flags: contradiction, unsupported jump, circular reasoning,
           broken causal chain, negation flip.
    """
    _CONTRADICTION_PAIRS = [
        (r"\btrue\b",   r"\bfalse\b"),
        (r"\bđúng\b",   r"\bsai\b"),
        (r"\btăng\b",   r"\bgiảm\b"),
        (r"\bmore\b",   r"\bless\b"),
    ]
    # FIX #3: causal markers — at least one expected in reasoning thoughts
    _CAUSAL_MARKERS = ["vì", "bởi vì", "do đó", "vì vậy", "because", "therefore",
                       "hence", "thus", "→", "⟹", "causes", "leads to"]
    # FIX #3: negation-flip patterns — "không A" followed by "A" in next step
    _NEGATION_FLIP  = [(r"\bkhông\s+(\w+)", r"\b\1\b"), (r"\bnot\s+(\w+)", r"\b\1\b")]

    def verify_thoughts(self, thoughts: List[str]) -> Dict[str, Any]:
        issues = []; score = 1.0
        combined = " ".join(thoughts).lower()

        # Contradiction check
        for pat_a, pat_b in self._CONTRADICTION_PAIRS:
            if re.search(pat_a, combined) and re.search(pat_b, combined):
                issues.append(f"Possible contradiction: '{pat_a}' vs '{pat_b}'")
                score -= 0.1

        # Unsupported jump check
        for i, t in enumerate(thoughts):
            if len(t.split()) < 3:
                issues.append(f"Thought {i+1} too brief — may be unsupported")
                score -= 0.05

        # Circular reasoning check
        seen = set()
        for t in thoughts:
            key = frozenset(t.lower().split()[:6])
            if key in seen:
                issues.append("Circular reasoning detected in thoughts")
                score -= 0.15
                break
            seen.add(key)

        # FIX #3: Causal chain check — warn if no causal marker found in reasoning
        if len(thoughts) >= 2:
            reasoning_text = " ".join(thoughts[1:]).lower()
            if not any(m in reasoning_text for m in self._CAUSAL_MARKERS):
                issues.append("No causal connector found — reasoning may lack explicit logic chain")
                score -= 0.08

        # FIX #3: Negation-flip check — "không X" in step N then "X" in step N+1
        for i in range(len(thoughts) - 1):
            a_lower = thoughts[i].lower()
            b_lower = thoughts[i + 1].lower()
            for neg_pat, pos_pat in self._NEGATION_FLIP:
                for neg_match in re.finditer(neg_pat, a_lower):
                    term = neg_match.group(1)
                    # Check if the positive form of the term appears in next thought
                    if re.search(r"\b" + re.escape(term) + r"\b", b_lower):
                        issues.append(
                            f"Negation-flip: thought {i+1} negates '{term}' "
                            f"but thought {i+2} asserts it positively"
                        )
                        score -= 0.12
                        break

        score = max(0.0, min(1.0, score))
        return {
            "issues":   issues,
            "score":    score,
            "is_sound": score >= 0.7,
            "expert":   "Debug Expert (EXPERT_CODE_2)",
        }

    def flag_thoughts(self, thoughts: List[str], verification: dict) -> List[str]:
        """Prepend [?] to flagged thoughts if verification failed."""
        if verification["is_sound"]: return thoughts
        return [f"[?] {t}" if i < 2 else t for i, t in enumerate(thoughts)]


# ══════════════════════════════════════════════════════════════════════
# MULTI-STEP PLANNER (Sub-goal Decomposition)
# ══════════════════════════════════════════════════════════════════════
class PlanDecomposer:
    """
    Decomposes a complex question into ordered sub-goals.
    Uses MCTS-scored branches as a lightweight goal tree.
    """
    def __init__(self, mcts: MCTSLight):
        self.mcts = mcts

    def decompose(self, question: str, intent: dict, max_subgoals: int = 4) -> List[str]:
        itype = intent.get("intent", INTENT_CHAT)
        if itype in (INTENT_MATH, INTENT_LOGIC):
            templates = [
                f"1. Identify known/unknown variables in: {question[:40]}",
                "2. Select applicable theorem or formula",
                "3. Apply step-by-step, checking units/constraints",
                "4. Verify result and edge cases",
            ]
        elif itype == INTENT_CODE:
            templates = [
                f"1. Clarify requirements & constraints for: {question[:40]}",
                "2. Design data structures and function signatures",
                "3. Implement core logic with error handling",
                "4. Write tests and document behavior",
            ]
        elif itype in (INTENT_HOW_TO, INTENT_WHY):
            templates = [
                f"1. Understand the context of: {question[:40]}",
                "2. Identify key factors or causes",
                "3. Explain mechanism or steps with examples",
                "4. Summarize with actionable conclusion",
            ]
        else:
            templates = [
                f"1. Parse the main topic of: {question[:40]}",
                "2. Retrieve relevant facts",
                "3. Structure and explain clearly",
                "4. Add context or caveats if needed",
            ]
        best = self.mcts.search(question, templates[:max_subgoals])
        return [f"[★] {t}" if t == best else t for t in templates[:max_subgoals]]


# ══════════════════════════════════════════════════════════════════════
# ACTIVE LEARNING LOOP
# ══════════════════════════════════════════════════════════════════════
class ActiveLearningLoop:
    """
    When DracoAI's confidence is low (< 0.5), generate a clarifying
    question to ask the user before committing to an answer.
    """
    _CLARIFICATION_TEMPLATES = {
        INTENT_MATH:       "Bạn muốn tính {topic} theo phương pháp nào — chính xác hay xấp xỉ?",
        INTENT_CODE:       "Ngôn ngữ lập trình và phiên bản bạn đang dùng cho {topic} là gì?",
        INTENT_FACTUAL:    "Bạn hỏi về {topic} trong ngữ cảnh nào — kỹ thuật hay tổng quát?",
        INTENT_HOW_TO:     "Mức độ chi tiết bạn cần cho '{topic}' là: cơ bản hay nâng cao?",
        INTENT_COMPARISON: "Bạn muốn so sánh {topic} theo tiêu chí nào — hiệu suất, chi phí, hay dễ dùng?",
    }
    _DEFAULT_TEMPLATE = "Bạn có thể nói rõ hơn về '{topic}' không? Tôi muốn trả lời chính xác hơn."

    def needs_clarification(self, confidence: float, intent: dict) -> bool:
        return confidence < 0.5

    def generate_clarification(self, question: str, intent: dict) -> str:
        itype    = intent.get("intent", INTENT_CHAT)
        entities = intent.get("entities", [])
        topic    = entities[0] if entities else question[:30]
        template = self._CLARIFICATION_TEMPLATES.get(itype, self._DEFAULT_TEMPLATE)
        return template.format(topic=topic)


# ══════════════════════════════════════════════════════════════════════
# UNCERTAINTY QUANTIFICATION
# ══════════════════════════════════════════════════════════════════════
class UncertaintyQuantifier:
    """
    Tag individual sentences in an answer with [confidence: X].
    Heuristic: sentences with hedges/negations → lower confidence.
    """
    _HEDGE_WORDS = [
        "có thể", "maybe", "perhaps", "possibly", "không chắc", "i think",
        "tôi nghĩ", "dường như", "it seems", "might", "could be", "probably",
    ]
    _CERTAIN_WORDS = [
        "chắc chắn", "definitely", "clearly", "obviously", "always", "luôn",
        "proven", "đã được chứng minh",
    ]

    def tag(self, answer: str, base_confidence: float = 0.75) -> str:
        if not answer or len(answer.strip()) < 10:
            return answer
        sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
        tagged = []
        for sent in sentences:
            sl   = sent.lower()
            conf = base_confidence
            hedges   = sum(1 for w in self._HEDGE_WORDS   if w in sl)
            certains = sum(1 for w in self._CERTAIN_WORDS if w in sl)
            conf -= hedges   * 0.08
            conf += certains * 0.05
            conf = round(max(0.1, min(1.0, conf)), 2)
            if conf < 0.5:
                tagged.append(f"[confidence:{conf}] {sent}")
            else:
                tagged.append(sent)
        return " ".join(tagged)

    def overall_confidence(self, answer: str, base: float) -> float:
        sl       = answer.lower()
        hedges   = sum(1 for w in self._HEDGE_WORDS   if w in sl)
        certains = sum(1 for w in self._CERTAIN_WORDS if w in sl)
        return round(max(0.1, min(1.0, base - hedges * 0.05 + certains * 0.03)), 2)


# ══════════════════════════════════════════════════════════════════════
# COUNTERFACTUAL REASONING
# ══════════════════════════════════════════════════════════════════════
class CounterfactualReasoner:
    """
    For logic/legal/policy questions, generate a "what-if" branch
    and contrast it with the factual branch.
    Enabled only for INTENT_LOGIC and INTENT_WHY.
    """
    _TRIGGER_INTENTS = {INTENT_LOGIC, INTENT_WHY}
    _CF_PATTERNS     = [
        r"nếu\s+không", r"what\s+if\s+not", r"giả sử", r"suppose",
        r"hypothetically", r"nếu\s+\w+\s+không",
    ]

    def is_applicable(self, query: str, intent: dict) -> bool:
        itype = intent.get("intent", INTENT_CHAT)
        if itype not in self._TRIGGER_INTENTS: return False
        ql = query.lower()
        return any(re.search(p, ql) for p in self._CF_PATTERNS) or itype == INTENT_LOGIC

    def generate(self, question: str, intent: dict) -> str:
        entities = intent.get("entities", [])
        subj     = entities[0] if entities else "the subject"
        return (
            f"[COUNTERFACTUAL] If '{subj}' were NOT the case: "
            f"the reasoning chain would diverge at the first premise. "
            f"Alternative outcome: the conclusion would likely be negated or weakened. "
            f"Consistency check: the factual answer should hold against this counterfactual."
        )


# ══════════════════════════════════════════════════════════════════════
# ANALOGICAL MAPPING
# FIX #9: weight-threshold guard before returning analogy
# ══════════════════════════════════════════════════════════════════════
_ANALOGY_MIN_WEIGHT = 0.3   # FIX #9: don't return analogy if edge weight < threshold


class AnalogicalMapper:
    """
    Use KnowledgeGraph to find analogies: A:B :: C:?

    The A→B structural relationship is now used to guide candidate selection:
    we look for C→X where X shares the same "hop distance" from C that B has
    from A, intersected with C's neighbours for graph-grounded results.

    FIX: concept_a is used to compute the A→B relation signature (hop
    distance + shared neighbours), then the same signature is sought from C.
    FIX: weight-threshold guard — only return analogy if best candidate
    edge weight >= _ANALOGY_MIN_WEIGHT.
    """

    def find_analogy(
        self,
        kg: KnowledgeGraph,
        concept_a: str,
        concept_b: str,
        concept_c: str,
    ) -> Optional[str]:
        """
        A:B :: C:? — find X such that C→X mirrors A→B structurally.
        Uses concept_a to characterise the A–B relationship, then seeks
        the same pattern from C.
        """
        # Characterise the A→B relationship via shared neighbourhood
        related_a = set(kg.related(concept_a, hops=2).keys())
        related_b = set(kg.related(concept_b, hops=2).keys())
        # "signature" = concepts shared between A's neighbourhood and B's
        ab_signature = related_a & related_b

        related_c = set(kg.related(concept_c, hops=2).keys())

        # Prefer candidates that mirror the A–B signature from C
        if ab_signature:
            candidates = related_c & ab_signature
        else:
            candidates = set()

        # Fallback: any overlap of C's neighbours with B's neighbours
        if not candidates:
            candidates = related_c & related_b

        # Last-resort: immediate neighbours of C
        if not candidates:
            candidates = set(kg.related(concept_c, hops=1).keys())

        if not candidates:
            return None

        # Pick best by direct edge weight from C
        best = max(candidates, key=lambda n: kg.g.get(concept_c, {}).get(n, 0.0))
        best_weight = kg.g.get(concept_c, {}).get(best, 0.0)
        if best_weight < _ANALOGY_MIN_WEIGHT:
            return None   # reject low-confidence analogies
        return best

    def describe_analogy(self, a: str, b: str, c: str, x: Optional[str]) -> str:
        if x is None:
            return f"[ANALOGY] {a}:{b} :: {c}:? — No analogy found in knowledge graph."
        return f"[ANALOGY] {a}:{b} :: {c}:{x} — {c} relates to {x} as {a} relates to {b}."


# ══════════════════════════════════════════════════════════════════════
# RETRIEVAL AUGMENTER (RAG stub)
# FIX #7: retrieve() stub clearly documented; callers safe on empty return
# ══════════════════════════════════════════════════════════════════════
class RetrievalAugmenter:
    """
    RAG hook for INTENT_FACTUAL and INTENT_HOW_TO.

    FIX #7: retrieve() is a documented stub. It returns [] gracefully.
    To activate real RAG, replace retrieve() body with:
        vec    = embedder.encode(query)               # e.g. MiniEmbedder
        docs   = vector_store.search(vec, top_k)      # e.g. LongTermMemoryV1
        return [{"text": d.text, "score": d.score, "ts": d.timestamp} for d in docs]
    """
    _TRIGGER_INTENTS = {INTENT_FACTUAL, INTENT_HOW_TO}

    def is_applicable(self, intent: dict) -> bool:
        return intent.get("intent") in self._TRIGGER_INTENTS

    def retrieve(self, query: str, intent: dict, top_k: int = 3) -> List[dict]:
        """
        Stub retriever — returns [] until a real vector store is connected.
        FIX #7: empty-list return is safe; augment_memory_summary handles it.
        PRODUCTION HOOK: replace this body with your embedding retriever.
        Each result must be: {"text": str, "score": float, "ts": float}
        """
        return []

    def augment_memory_summary(
        self,
        base_summary: str,
        retrieved: List[dict],
        reranker: MemoryReranker,
        query: str,
        intent: dict,
    ) -> str:
        """Merge retrieved docs with existing memory summary via MemoryReranker."""
        if not retrieved: return base_summary
        reranked = reranker.rerank(retrieved, query, intent, top_k=3)
        rag_text = reranker.format_for_prompt(reranked, max_chars=300)
        if not rag_text: return base_summary
        return (base_summary + " [RAG] " + rag_text).strip()


# ══════════════════════════════════════════════════════════════════════
# PROMPT COMPILER — [PLAN][THOUGHT][FINAL ANSWER] injection
# ══════════════════════════════════════════════════════════════════════
class PromptCompiler:
    def compile(
        self,
        question:       str,
        intent:         dict,
        thought_plan:   dict,
        memory_summary: str        = "",
        history:        List[dict] = None,
    ) -> List[dict]:
        msgs = []
        sys_content = DRACO_SYSTEM_PROMPT
        if memory_summary:
            sys_content += f"\n\n[MEMORY]\n{memory_summary[:400]}"

        expert_note = {
            INTENT_MATH:     "\n[ACTIVE] Logic/Math experts (0,2). Show step-by-step working.",
            INTENT_LOGIC:    "\n[ACTIVE] Logic/Math + Debug experts (0,2). Use formal reasoning.",
            INTENT_CODE:     "\n[ACTIVE] Code + Language experts (1,4). Write code with explanation.",
            INTENT_CREATIVE: "\n[ACTIVE] Creative + Language experts (6,4). Be expressive.",
            INTENT_CHAT:     "\n[ACTIVE] Chat expert (5). Be friendly and natural.",
        }.get(intent.get("intent", INTENT_CHAT), "")

        if expert_note:
            sys_content += expert_note

        if thought_plan.get("tool_injection"):
            sys_content += f"\n\n{thought_plan['tool_injection']}"

        if thought_plan.get("counterfactual"):
            sys_content += f"\n\n{thought_plan['counterfactual']}"

        msgs.append({"role": "system", "content": sys_content})

        if thought_plan.get("thoughts"):
            lines = [
                "[PLAN]",
                f"Type: {intent.get('intent','?')} | Lang: {intent.get('lang','?')} | "
                f"Entities: {', '.join(intent.get('entities', [])[:3]) or '---'}",
            ]
            if thought_plan.get("subgoals"):
                lines.append("\n[SUBGOALS]")
                lines.extend(thought_plan["subgoals"])

            if thought_plan.get("debate_synthesis"):
                lines.append(f"\n[DEBATE]\n{thought_plan['debate_synthesis']}")
            if thought_plan.get("sc_path"):
                lines.append(f"\n[SELF-CONSISTENCY]\n{thought_plan['sc_path']}")
            if thought_plan.get("best_branch"):
                lines.append(f"\n[THOUGHT 1]\n{thought_plan['best_branch']}")
            for i, t in enumerate(thought_plan.get("thoughts", [])[:2], 2):
                lines.append(f"\n[THOUGHT {i}]\n{t}")
            if thought_plan.get("reasoning_path"):
                lines.append(f"\n[KNOWLEDGE PATH]\n{' → '.join(thought_plan['reasoning_path'])}")

            if thought_plan.get("analogy"):
                lines.append(f"\n{thought_plan['analogy']}")

            if thought_plan.get("cot_verification"):
                v = thought_plan["cot_verification"]
                if not v.get("is_sound"):
                    lines.append(f"\n[COT VERIFY — issues: {'; '.join(v.get('issues', [])[:2])}]")

            lines.append("\n[FINAL ANSWER]")
            msgs.append({"role": "system", "content": "\n".join(lines)})

        if history: msgs.extend(history[-10:])
        msgs.append({"role": "user", "content": question})
        return msgs


# ══════════════════════════════════════════════════════════════════════
# THINKING ENGINE V1
# ══════════════════════════════════════════════════════════════════════
class ThinkingEngineV1:
    def __init__(self):
        self.kg           = KnowledgeGraph(); self.kg.init_default()
        self.detector     = IntentDetector()
        self.tot          = TreeOfThoughts(MCTSLight(n_sim=8, max_rollout_depth=10))
        self.reflect      = SelfReflection()
        self.compiler     = PromptCompiler()
        self.reranker     = MemoryReranker()
        self.cpr          = ContextualPromptRewriter()
        self.debate       = MultiAgentDebate()
        self.sc           = SelfConsistency()
        self.dual         = DualProcessDecider()
        self.difficulty   = DifficultyScorer()
        self.tools        = ToolCallingFramework()
        self.cot_verifier = ChainOfThoughtVerifier()
        self.decomposer   = PlanDecomposer(MCTSLight(n_sim=5, max_rollout_depth=8))
        self.active_loop  = ActiveLearningLoop()
        self.uq           = UncertaintyQuantifier()
        self.cf_reasoner  = CounterfactualReasoner()
        self.analogy      = AnalogicalMapper()
        self.rag          = RetrievalAugmenter()

    # ── Main processing pipeline ──────────────────────────────────────
    def process(
        self,
        question:          str,
        history:           List[dict] = None,
        memory_summary:    str        = "",
        ltm_facts:         List[dict] = None,
        memory_candidates: List[dict] = None,
        think_mode:        bool       = False,
        force_system2:     bool       = False,
    ) -> dict:
        # Contextual Prompt Rewriting
        rewritten_q = self.cpr.rewrite(question, history or [])
        if rewritten_q != question:
            memory_summary = (memory_summary + f"\n[Rewritten: {rewritten_q}]").strip()

        intent       = self.detector.detect(rewritten_q)
        expert_boost = self.detector.to_expert_boost(intent)   # FIX #2: normalized
        miro_tau     = self.detector.to_miro_tau(intent)
        process_mode = self.dual.decide_mode(intent, rewritten_q)

        # Difficulty-based System2 auto-routing
        difficulty_score = self.difficulty.score(rewritten_q, intent)
        base_conf        = self._confidence(intent)
        if force_system2 or (difficulty_score > 0.65 and base_conf < 0.75):
            process_mode = "slow"

        # ── Early-exit for simple chat queries ──────────────────────────
        # Only use fast-path when BOTH conditions hold:
        #   1. query is trivially simple (INTENT_CHAT, word_count ≤ 3)
        #   2. confidence is high enough (>= 0.8) — prevents mis-routing
        #      a genuinely complex query that scored INTENT_CHAT by accident.
        if (self.dual.is_simple_chat(intent)
                and not think_mode
                and not force_system2
                and base_conf >= 0.8):
            messages = self.compiler.compile(
                rewritten_q, intent,
                thought_plan={},          # no heavy plan needed
                memory_summary=memory_summary,
                history=history or [],
            )
            return {
                "intent":                intent,
                "expert_boost":          expert_boost,
                "miro_tau":              miro_tau,
                "thought_plan":          {"process_mode": "fast", "early_exit": True},
                "messages":              messages,
                "creativity":            intent["creativity"],
                "rewritten_query":       rewritten_q,
                "process_mode":          "fast",
                "difficulty_score":      difficulty_score,
                "clarification_needed":  False,
                "clarification_question": "",
                "cot_verification":      {"is_sound": True, "issues": [], "score": 1.0},
                "tool_injection_active": False,
            }
        # ── End early-exit ───────────────────────────────────────────

        # Dynamic KG triple extraction from latest user turn
        self.kg.extract_and_add_triples(rewritten_q, conf=base_conf)

        # RAG augmentation for factual/how-to
        if self.rag.is_applicable(intent):
            retrieved      = self.rag.retrieve(rewritten_q, intent)
            memory_summary = self.rag.augment_memory_summary(
                memory_summary, retrieved, self.reranker, rewritten_q, intent
            )

        # Knowledge graph path
        reasoning_path: List[str] = []
        entities = intent.get("entities", [])
        if len(entities) >= 2:
            path, _ = self.kg.astar(entities[0], entities[1])
            if not path:
                path = self.kg.bfs(entities[0], entities[1])
            if path: reasoning_path = path
        elif entities:
            reasoning_path = list(self.kg.related(entities[0], hops=2).keys())[:4]

        # Thought plan (ToT + MCTS)
        best_branch, all_branches = self.tot.run(rewritten_q, intent)
        thoughts = self._thoughts(rewritten_q, intent, reasoning_path)

        # CoT Verification (FIX #3: enriched)
        cot_verification = self.cot_verifier.verify_thoughts(thoughts)
        thoughts = self.cot_verifier.flag_thoughts(thoughts, cot_verification)

        # Sub-goal decomposition (slow mode only)
        subgoals: List[str] = []
        if process_mode == "slow" or think_mode:
            subgoals = self.decomposer.decompose(rewritten_q, intent)

        # Multi-Agent Debate (for complex queries in slow mode)
        debate_synthesis = ""
        debate_opinions  = {}
        if think_mode or process_mode == "slow":
            if think_mode:
                council_result   = self.debate.run_full_council(rewritten_q, intent, max_rounds=3)
                debate_synthesis = council_result["final_answer"]
                debate_opinions  = council_result["opinions"]
            else:
                debate_synthesis, debate_opinions = self.debate.generate_debate(rewritten_q, intent)

        # Self-Consistency (for math/logic/code)
        sc_path = ""
        if think_mode and intent["intent"] in (INTENT_MATH, INTENT_LOGIC, INTENT_CODE):
            sc_paths = self.sc.generate_paths(rewritten_q, intent, n_paths=3)
            sc_path  = self.sc.vote(sc_paths, rewritten_q)

        # Intent-aware memory rerank
        reranked_memory = ""
        if memory_candidates and isinstance(memory_candidates, list):
            reranked        = self.reranker.rerank(memory_candidates, rewritten_q, intent, top_k=3)
            reranked_memory = self.reranker.format_for_prompt(reranked)

        full_memory = memory_summary
        if reranked_memory:
            full_memory = (memory_summary + " | " + reranked_memory).strip(" |")

        # Tool calling injection
        tool_injection = ""
        if self.tools.should_use_tools(intent, rewritten_q):
            tool_injection = self.tools.build_tool_injection(intent, rewritten_q)

        # Counterfactual reasoning note
        counterfactual = ""
        if self.cf_reasoner.is_applicable(rewritten_q, intent):
            counterfactual = self.cf_reasoner.generate(rewritten_q, intent)

        # Analogical mapping — A:B :: C:? (uses entities[0] as A, [1] as B, [2] as C)
        analogy_note = ""
        if len(entities) >= 2:
            concept_a = entities[0]
            concept_b = entities[1]
            concept_c = entities[2] if len(entities) >= 3 else entities[1]
            x = self.analogy.find_analogy(self.kg, concept_a, concept_b, concept_c)
            if x:
                analogy_note = self.analogy.describe_analogy(
                    concept_a, concept_b, concept_c, x
                )

        # Active Learning — clarification if confidence too low
        clarification_needed = self.active_loop.needs_clarification(base_conf, intent)
        clarification_q      = (
            self.active_loop.generate_clarification(rewritten_q, intent)
            if clarification_needed else ""
        )

        thought_plan = {
            "best_branch":      best_branch,
            "all_branches":     all_branches,
            "thoughts":         thoughts,
            "reasoning_path":   reasoning_path,
            "strategy":         self._strategy(intent),
            "confidence":       base_conf,
            "debate_synthesis": debate_synthesis,
            "debate_opinions":  debate_opinions,
            "sc_path":          sc_path,
            "process_mode":     process_mode,
            "subgoals":         subgoals,
            "cot_verification": cot_verification,
            "tool_injection":   tool_injection,
            "counterfactual":   counterfactual,
            "analogy":          analogy_note,
            "difficulty_score": difficulty_score,
        }

        messages = self.compiler.compile(
            rewritten_q, intent, thought_plan, full_memory, history or []
        )

        return {
            "intent":                intent,
            "expert_boost":          expert_boost,
            "miro_tau":              miro_tau,
            "thought_plan":          thought_plan,
            "messages":              messages,
            "creativity":            intent["creativity"],
            "rewritten_query":       rewritten_q,
            "process_mode":          process_mode,
            "difficulty_score":      difficulty_score,
            "clarification_needed":  clarification_needed,
            "clarification_question":clarification_q,
            "cot_verification":      cot_verification,
            "tool_injection_active": bool(tool_injection),
        }

    # ── Post-generation helpers ───────────────────────────────────────
    def critique_and_refine(self, question: str, answer: str, ltm_facts: List[dict] = None) -> dict:
        return self.reflect.critique(answer, question, ltm_facts or [])

    def recursive_critique(
        self,
        question:  str,
        answer:    str,
        ltm_facts: List[dict] = None,
        max_iter:  int = 3,
    ) -> Tuple[str, List[dict]]:
        """
        Recursive Self-Critique Loop.
        Iteratively critiques and refines the answer up to max_iter times.
        Returns (final_answer, list_of_critique_reports).
        """
        reports  = []
        current  = answer
        for _ in range(max_iter):
            report = self.reflect.critique(current, question, ltm_facts or [])
            reports.append(report)
            if not report["should_refine"]:
                break
            refine_note = self.reflect.build_refine_prompt(current, report)
            # PRODUCTION HOOK: pass refine_note to LLM for actual refinement.
            # Stub: append the refine note as a marker.
            current = f"{current}\n[AUTO-REFINED: {refine_note[:120]}]"
        return current, reports

    def tag_answer_uncertainty(self, answer: str, intent: dict) -> str:
        """Apply UncertaintyQuantifier to a generated answer."""
        base = self._confidence(intent)
        return self.uq.tag(answer, base_confidence=base)

    def parse_and_execute_tools(self, model_output: str) -> Tuple[List[dict], List[dict]]:
        """
        Parse tool calls from model output and execute them.
        FIX #6: Returns (parsed_calls, structured_results_list) — not a flat string.
        Use tools.build_tool_context(results) to format for LLM re-injection.

        Typical caller pattern (LLM→tool→LLM loop):
            calls, results = engine.parse_and_execute_tools(model_output)
            if results:
                tool_ctx = engine.tools.build_tool_context(results)
                messages.append({"role": "user", "content": tool_ctx})
                # Call generate() again with updated messages
        """
        calls   = self.tools.parse_tool_calls(model_output)
        results = self.tools.execute_tool_calls(calls)
        return calls, results

    # ── Private helpers ───────────────────────────────────────────────
    def _thoughts(self, q: str, intent: dict, path: List[str]) -> List[str]:
        t = [f"Type: {intent['intent']} | Lang: {intent['lang']} | "
             f"Entities: {', '.join(intent['entities'][:3]) or '---'}"]
        if path: t.append(f"Knowledge chain: {' → '.join(path[:5])}")
        t.append(self._strategy(intent))
        return t

    def _strategy(self, intent: dict) -> str:
        return {
            INTENT_MATH:       "Step-by-step calculation, verify result",
            INTENT_LOGIC:      "Premises → reasoning → conclusion",
            INTENT_CODE:       "Analyze → design → implement → explain",
            INTENT_CREATIVE:   "Brainstorm → pick unique angle → develop",
            INTENT_FACTUAL:    "Retrieve → verify → answer concisely",
            INTENT_HOW_TO:     "Step 1→2→3 with practical examples",
            INTENT_WHY:        "Cause → mechanism → consequence",
            INTENT_COMPARISON: "Similarities → differences → when to use which",
            INTENT_CHAT:       "Natural, friendly response",
            INTENT_MEMORY:     "Retrieve from memory, confirm accuracy",
        }.get(intent["intent"], "Direct answer with sufficient context")

    def _confidence(self, intent: dict) -> float:
        base = {
            INTENT_MATH: 0.92, INTENT_CODE: 0.85, INTENT_FACTUAL: 0.78,
            INTENT_LOGIC: 0.88, INTENT_CHAT: 0.80,
        }.get(intent["intent"], 0.72)
        return max(0.3, base - len(intent.get("entities", [])) * 0.03)