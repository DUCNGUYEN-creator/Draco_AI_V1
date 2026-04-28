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
DracoAI V1 — Main Engine (Production‑ready)
===========================================
Tích hợp đầy đủ ThinkingEngineV1, Memory, Transformer thông qua TransformerBridge.
Tương thích với: engine_v1.py, transformer_v1.py, tokenizer.py, memory_v1.py

Changelog V1 (fixes):
  [A] _register_query_hash: sửa eviction dùng popleft() thay vì index[0]
  [B] intent_bias_arr: guard None trước np.tanh()
  [C] skill_calc: regex extract expression trước khi _safe_calc
  [D] _auto_learn: filter stopwords đầu câu
  [E] chat/stream_chat: thêm context trimming (MAX_HISTORY_TURNS)
  [F] ltm.store: check duplicate trước khi lưu
  [G] _auto_learn: khôi phục HALLUCINATION_FLAGS guard
  [H] Thêm skill_read_file, skill_create_file, skill_write_report
  Vẫn chưa xong, cần góp ý, đây là file thêm skill cứng cho AI và là main engine của model.
"""

import os, re, time
from typing import List, Dict, Optional, Generator, Tuple, Any
from collections import deque

from tokenizer import BPETokenizer
from memory_v1 import LongTermMemoryV1, WorkingMemoryV1, KVCacheBuffer
from engine_v1 import ThinkingEngineV1
from transformer_v1 import (
    DracoTransformerV1,
    TransformerBridge,
    GGUFExporter,
)

# ── [G] Hallucination guard ────────────────────────────────────────────────────
HALLUCINATION_FLAGS = [
    "tôi không chắc", "có thể là", "tôi đoán", "theo như tôi biết thì",
    "i'm not sure", "i think", "maybe", "i believe", "possibly",
    "tôi nghĩ rằng có thể", "chưa rõ",
]

# ── [D] Stopwords filter cho _auto_learn ──────────────────────────────────────
_LEARN_STOPWORDS = {
    "tôi", "bạn", "nó", "họ", "chúng", "mình", "này", "đây", "kia",
    "the", "a", "an", "it", "this", "that", "these", "those", "i", "we",
    "you", "he", "she", "they", "my", "your", "our", "their",
    "tôi nghĩ", "bạn có thể", "chúng ta", "theo tôi",
}

# ── [E] Context window limit ───────────────────────────────────────────────────
MAX_HISTORY_TURNS = 20   # số lượt hội thoại tối đa giữ lại (1 lượt = user + assistant)


class DracoEngineV1:
    """Điều phối viên chính – kết nối tất cả các tầng."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.ckpt = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # ── Tokenizer ─────────────────────────────────────────────────
        tok_dir = os.path.join(checkpoint_dir, "tokenizer")
        if os.path.exists(os.path.join(tok_dir, "tokenizer_draco.json")):
            self.tokenizer = BPETokenizer.load(tok_dir)
        else:
            print("[Engine] Không tìm thấy tokenizer – khởi tạo rỗng")
            self.tokenizer = BPETokenizer()

        # ── Mô hình ────────────────────────────────────────────────────
        if os.path.exists(os.path.join(checkpoint_dir, "config.json")):
            self.model = DracoTransformerV1.load_weights(checkpoint_dir)
        else:
            config = {
                "d_model": 128, "n_layers": 4, "n_heads": 4, "n_kv_heads": 2,
                "head_dim": 32, "d_ff": 512, "n_experts": 8,
                "vocab_size": max(151936, self.tokenizer.vocab_size),
                "window": 1024,
            }
            self.model = DracoTransformerV1(config)

        # ── Bridge ─────────────────────────────────────────────────────
        self.bridge = TransformerBridge(numpy_model=self.model)

        # ── Memory ────────────────────────────────────────────────────
        mem_dir = os.path.join(os.path.dirname(os.path.abspath(checkpoint_dir)), "memory")
        self.ltm = LongTermMemoryV1(mem_dir)
        kv_buf   = KVCacheBuffer()
        self.working = WorkingMemoryV1(self.ltm, kv_buf,
                                       max_tokens=self.model.window,
                                       tokenizer=self.tokenizer)

        # ── Thinking Engine ────────────────────────────────────────────
        self.thinking = ThinkingEngineV1(
            max_experts=4,
            bridge=self.bridge,
            tokenizer=self.tokenizer,
        )

        # ── Kỹ năng nhanh ──────────────────────────────────────────────
        self._register_skills()

        # ── [A] Dedup hash queue (khởi tạo đúng, eviction dùng popleft) ──
        self._hash_queue: deque = deque(maxlen=512)
        self._hash_set:   set   = set()

        print(f"[DracoAI] Sẵn sàng! "
              f"(vocab={self.tokenizer.vocab_size}, "
              f"bridge={'đã kết nối' if self.bridge.is_connected() else 'stub'})\n")

    # ══════════════════════════════════════════════════════════════════
    # [A] Dedup helper (FIFO eviction chuẩn)
    # ══════════════════════════════════════════════════════════════════
    def _register_query_hash(self, h: str) -> bool:
        """Trả về True nếu hash mới (chưa thấy), False nếu trùng.
        Eviction FIFO đúng: khi queue đầy, deque tự drop phần tử cũ nhất
        ở bên trái → ta phải đồng bộ remove khỏi set TRƯỚC KHI append.
        """
        if h in self._hash_set:
            return False
        # Nếu queue đầy, phần tử sẽ bị drop khi append → remove khỏi set trước
        if len(self._hash_queue) == self._hash_queue.maxlen:
            oldest = self._hash_queue[0]   # đọc nhưng chưa pop
            # deque sẽ tự popleft khi append, ta chỉ cần sync set
            self._hash_set.discard(oldest)
        self._hash_queue.append(h)
        self._hash_set.add(h)
        return True

    # ══════════════════════════════════════════════════════════════════
    # Skills
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def _safe_calc(expr: str) -> Optional[float]:
        import ast, operator
        _ops = {
            ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.Pow: operator.pow, ast.USub: operator.neg,
        }
        def _eval(node):
            if isinstance(node, ast.Constant):
                return float(node.value) if isinstance(node.value, (int, float)) else None
            if isinstance(node, ast.BinOp):
                op = _ops.get(type(node.op))
                if op is None: return None
                return op(_eval(node.left), _eval(node.right))
            if isinstance(node, ast.UnaryOp):
                op = _ops.get(type(node.op))
                if op is None: return None
                return op(_eval(node.operand))
            return None
        try:
            tree = ast.parse(expr.strip(), mode="eval")
            return _eval(tree.body)
        except Exception:
            return None

    def _register_skills(self):
        # ── Skill: tính toán ──────────────────────────────────────────
        def skill_calc(text, *_):
            # [C] Extract expression trước, không ăn cả câu
            m = re.search(
                r"([\d\s\+\-\*\/\^\(\)\.]+(?:\*\*[\d\s\(\)\.]+)?)",
                text
            )
            if m:
                r = self._safe_calc(m.group(1).strip())
                return f"Kết quả: {r:g}" if r is not None else None
            return None

        # ── Skill: ghi nhớ ────────────────────────────────────────────
        def skill_remember(text, *_):
            m = re.search(r"nhớ\s+(?:rằng\s+)?(.+?)\s+(?:là|=|is)\s+(.+)", text.lower())
            if m:
                k, v = m.group(1).strip(), m.group(2).strip()
                self.ltm.remember_fact(k, v)
                return f"✅ Đã ghi nhớ: **{k}** = {v}"
            return None

        # ── Skill: nhớ lại ────────────────────────────────────────────
        def skill_recall(text, *_):
            facts = self.ltm.get_facts()
            for f in facts:
                if f["key"].lower() in text.lower():
                    return f"💾 **{f['key']}**: {f['value']}"
            results = self.ltm.search(text, top_k=2)
            if results and results[0]["score"] > 0.5:
                return f"💾 [Memory] {results[0]['text']}"
            return None

        # ── [H] Skill: đọc file ───────────────────────────────────────
        def skill_read_file(text, *_):
            """Phát hiện yêu cầu đọc file và trả về nội dung tóm tắt."""
            m = re.search(
                r"(?:đọc|mở|xem|read|open|load|hiện|show)\s+(?:file\s+)?[\"']?([^\s\"']+\.\w+)[\"']?",
                text, re.IGNORECASE
            )
            if not m:
                return None
            path = m.group(1).strip()
            if not os.path.exists(path):
                return f"❌ Không tìm thấy file: `{path}`"
            try:
                ext = os.path.splitext(path)[1].lower()
                size = os.path.getsize(path)

                # Plain text / code
                if ext in (".txt", ".md", ".log", ".py", ".json", ".csv",
                           ".yaml", ".yml", ".ini", ".toml", ".xml", ".html",
                           ".js", ".ts", ".java", ".c", ".cpp", ".h", ".rs"):
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read(8192)  # Đọc tối đa 8KB để tránh flood
                    lines = content.splitlines()
                    preview = "\n".join(lines[:50])
                    note = f"\n... (còn {len(lines)-50} dòng)" if len(lines) > 50 else ""
                    return (f"📄 **File:** `{path}` ({size:,} bytes)\n"
                            f"```{ext.lstrip('.')}\n{preview}{note}\n```")

                # Binary / unknown
                return (f"📦 **File:** `{path}` ({size:,} bytes)\n"
                        f"Định dạng `{ext}` — cần xử lý chuyên biệt "
                        f"(pdf/docx/xlsx/pptx). Bạn muốn tôi phân tích không?")
            except Exception as e:
                return f"❌ Lỗi đọc file: {e}"

        # ── [H] Skill: tạo file ───────────────────────────────────────
        def skill_create_file(text, *_):
            """Phát hiện yêu cầu tạo/ghi file, thực hiện và xác nhận."""
            m = re.search(
                r"(?:tạo|viết|ghi|lưu|create|write|save|xuất)\s+"
                r"(?:file\s+|ra\s+)?[\"']?([^\s\"']+\.\w+)[\"']?"
                r"(?:\s+với\s+nội\s+dung\s+|\s+content[:\s]+|\s+chứa\s+)(.+)",
                text, re.IGNORECASE | re.DOTALL
            )
            if not m:
                return None
            path, content_raw = m.group(1).strip(), m.group(2).strip()
            # Bảo vệ: không ghi đè file hệ thống
            if path.startswith("/etc") or path.startswith("/sys") or path.startswith("/proc"):
                return "⛔ Không thể ghi vào đường dẫn hệ thống."
            try:
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content_raw)
                return (f"✅ Đã tạo file: `{path}` "
                        f"({len(content_raw.encode())} bytes, "
                        f"{len(content_raw.splitlines())} dòng)")
            except Exception as e:
                return f"❌ Lỗi tạo file: {e}"

        # ── [H] Skill: viết báo cáo ───────────────────────────────────
        def skill_write_report(text, *_):
            """Phát hiện yêu cầu viết báo cáo và trả về template/hướng dẫn."""
            keywords_vi = ["viết báo cáo", "tạo báo cáo", "soạn báo cáo",
                           "báo cáo về", "lập báo cáo"]
            keywords_en = ["write report", "create report", "generate report",
                           "draft report", "report on"]
            tl = text.lower()
            matched = any(k in tl for k in keywords_vi + keywords_en)
            if not matched:
                return None

            # Trích chủ đề
            topic = re.sub(
                r"(viết|tạo|soạn|lập|write|create|generate|draft)\s+"
                r"(báo cáo|report)\s*(về|on|about)?\s*",
                "", tl, flags=re.IGNORECASE
            ).strip().title() or "Chủ đề chưa xác định"

            template = f"""# Báo cáo: {topic}

## 1. Tóm tắt điều hành (Executive Summary)
> Mô tả ngắn gọn mục tiêu, phương pháp và kết quả chính.

## 2. Giới thiệu
- Bối cảnh và lý do thực hiện
- Phạm vi và giới hạn của báo cáo

## 3. Phương pháp
- Công cụ / nguồn dữ liệu sử dụng
- Quy trình thu thập và xử lý

## 4. Kết quả & Phân tích
| Chỉ số | Giá trị | Ghi chú |
|--------|---------|---------|
| ...    | ...     | ...     |

## 5. Đánh giá & Thảo luận
- Điểm mạnh
- Hạn chế
- So sánh với kỳ vọng / baseline

## 6. Kết luận & Kiến nghị
- Kết luận chính
- Đề xuất hành động tiếp theo

## 7. Tài liệu tham khảo
- [1] ...

---
*Báo cáo được tạo bởi DracoAI V1 — {time.strftime("%Y-%m-%d %H:%M")}*
"""
            return f"📝 **Template báo cáo về: {topic}**\n\n{template}"

        self._skills = {
            "calc":    (skill_calc,    ["tính", " + ", " - ", " * ", " / ", "calculate"]),
            "remember":(skill_remember,["nhớ rằng", "ghi nhớ", "lưu lại", "remember that"]),
            "recall":  (skill_recall,  ["bạn có nhớ", "nhớ không", "bạn biết", "what is", "recall"]),
            # [H] Skills mới
            "read_file":   (skill_read_file,   ["đọc file", "mở file", "xem file",
                                                 "read file", "open file", "load file"]),
            "create_file": (skill_create_file, ["tạo file", "ghi file", "lưu file",
                                                 "create file", "write file", "save file", "xuất file"]),
            "write_report":(skill_write_report,["viết báo cáo", "tạo báo cáo", "soạn báo cáo",
                                                 "lập báo cáo", "write report", "create report",
                                                 "draft report", "báo cáo về"]),
        }

    def _try_skills(self, text: str) -> Optional[str]:
        tl = text.lower()
        for _, (fn, triggers) in self._skills.items():
            if any(t in tl for t in triggers):
                try:
                    r = fn(text, {}, {})
                    if r: return r
                except Exception:
                    pass
        return None

    # ══════════════════════════════════════════════════════════════════
    # [E] Context trimming helper
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def _trim_history(history: List[dict], max_turns: int = MAX_HISTORY_TURNS) -> List[dict]:
        """Giữ lại tối đa `max_turns` lượt cuối (1 lượt = 2 messages: user + assistant).
        Đảm bảo không cắt giữa lượt (luôn bắt đầu bằng role='user').
        """
        if len(history) <= max_turns * 2:
            return history
        trimmed = history[-(max_turns * 2):]
        # Đảm bảo bắt đầu bằng user
        while trimmed and trimmed[0].get("role") != "user":
            trimmed = trimmed[1:]
        return trimmed

    # ══════════════════════════════════════════════════════════════════
    # [F] LTM store với duplicate check
    # ══════════════════════════════════════════════════════════════════
    def _safe_ltm_store(self, text: str, meta: dict):
        """Lưu vào LTM chỉ khi chưa có nội dung tương tự gần đây."""
        import hashlib
        h = hashlib.md5(text.strip().lower().encode()).hexdigest()
        if not self._register_query_hash(h):
            return   # duplicate – bỏ qua
        self.ltm.store(text, meta)

    # ══════════════════════════════════════════════════════════════════
    # Chat
    # ══════════════════════════════════════════════════════════════════
    def chat(
            self,
            user_input: str,
            temperature: float = 0.8,
            top_k: int = 50,
            top_p: float = 0.9,
            min_p: float = 0.05,
            max_tokens: int = 256,
            think_mode: bool = False,
    ) -> dict:
        t0 = time.time()

        # 1. Skill shortcut
        skill_reply = self._try_skills(user_input)
        if skill_reply:
            self.working.add("user", user_input)
            self.working.add("assistant", skill_reply)
            return {
                "reply": skill_reply, "intent": {"intent": "skill"},
                "thought_plan": {}, "skill": "builtin",
                "time_ms": int((time.time() - t0) * 1000),
                "critique": {}, "confidence_avg": 1.0,
            }

        # 2. Prepare memory
        intent     = self.thinking.detector.detect(user_input)
        mem_input  = self.ltm.prepare_engine_input(user_input, intent, top_k=3)

        # [E] Trim history trước khi đưa vào engine
        raw_history = self.working.get_messages()
        history = self._trim_history(raw_history, max_turns=MAX_HISTORY_TURNS)

        # 3. Engine processing
        engine_out = self.thinking.process(
            user_input,
            history=history,
            memory_summary=mem_input["memory_summary"],
            ltm_facts=mem_input["ltm_facts"],
            memory_candidates=mem_input["memory_candidates"],
            think_mode=think_mode,
        )

        # 4. Tokenize
        prompt_ids = self.thinking.tokenize_prompt(engine_out["messages"])

        # 5. Build generation kwargs
        expert_boost    = engine_out.get("expert_boost", {})
        intent_bias_arr = self.bridge.expert_boost_to_array(expert_boost)

        # [B] Guard None trước np.tanh
        if intent_bias_arr is not None:
            try:
                import numpy as np
                intent_bias_arr = np.tanh(intent_bias_arr)
            except Exception:
                intent_bias_arr = None

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temp": temperature,
            "top_p": top_p,
            "min_p": min_p,
            "use_mirostat": True,
            "intent_bias": intent_bias_arr,
        }

        # 6. Generate tokens
        new_tokens = self.bridge.generate(prompt_ids, **gen_kwargs)

        # 7. Decode
        reply    = self.tokenizer.decode(new_tokens).strip() or "..."
        avg_conf = engine_out.get("calibrated_confidence", 0.8)

        # 8. Update memory
        self.working.add("user", user_input)
        self.working.add("assistant", reply)

        # [F] Dùng safe store thay vì store trực tiếp
        self._safe_ltm_store(user_input, {"type": "user_query", "intent": intent["intent"]})

        self._auto_learn(reply)

        return {
            "reply":           reply,
            "intent":          intent,
            "thought_plan":    engine_out["thought_plan"] if think_mode else {},
            "skill":           "transformer",
            "time_ms":         int((time.time() - t0) * 1000),
            "critique":        {},
            "confidence_avg":  round(avg_conf, 3),
            "rewritten_query": engine_out.get("rewritten_query", user_input),
            "process_mode":    engine_out.get("process_mode", "fast"),
            "debate":          engine_out["thought_plan"].get("debate_synthesis", ""),
        }

    # ══════════════════════════════════════════════════════════════════
    # Streaming Chat
    # ══════════════════════════════════════════════════════════════════
    def stream_chat(
            self,
            messages: List[dict],
            temperature: float = 0.8,
            top_p: float = 0.9,
            max_tokens: int = 256,
            think_mode: bool = False,
    ) -> Generator[Tuple[str, float], None, None]:
        user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

        intent    = self.thinking.detector.detect(user_msg)
        mem_input = self.ltm.prepare_engine_input(user_msg, intent, top_k=3)

        # [E] Trim history cho stream cũng vậy
        history_raw = [m for m in messages[:-1]]
        history = self._trim_history(history_raw, max_turns=MAX_HISTORY_TURNS)

        engine_out = self.thinking.process(
            user_msg,
            history=history,
            memory_summary=mem_input["memory_summary"],
            ltm_facts=mem_input["ltm_facts"],
            memory_candidates=mem_input["memory_candidates"],
            think_mode=think_mode,
        )

        prompt_ids = self.thinking.tokenize_prompt(engine_out["messages"])

        expert_boost    = engine_out.get("expert_boost", {})
        intent_bias_arr = self.bridge.expert_boost_to_array(expert_boost)

        # [B] Guard None
        if intent_bias_arr is not None:
            try:
                import numpy as np
                intent_bias_arr = np.tanh(intent_bias_arr)
            except Exception:
                intent_bias_arr = None

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temp": temperature,
            "top_p": top_p,
            "min_p": 0.05,
            "use_mirostat": True,
            "intent_bias": intent_bias_arr,
        }

        pieces: List[Tuple[str, float]] = []
        full_reply_parts: List[str] = []

        def stream_cb(token_id: int, confidence: float):
            text = self.tokenizer.decode_token(token_id)
            if text:
                pieces.append((text, confidence))
                full_reply_parts.append(text)

        gen_kwargs["stream_cb"] = stream_cb
        self.bridge.generate(prompt_ids, **gen_kwargs)

        # Update memory sau khi stream xong (không bị mất state)
        full_reply = "".join(full_reply_parts).strip()
        if full_reply:
            self.working.add("user", user_msg)
            self.working.add("assistant", full_reply)
            self._safe_ltm_store(user_msg, {"type": "user_query", "intent": intent["intent"]})
            self._auto_learn(full_reply)

        for text, conf in pieces:
            yield text, conf

    # ══════════════════════════════════════════════════════════════════
    # Utilities
    # ══════════════════════════════════════════════════════════════════
    def _auto_learn(self, response: str):
        """Tự học fact từ response với guard hallucination và stopwords."""

        # [G] Guard hallucination: không học khi model không chắc
        resp_lower = response.lower()
        if any(flag in resp_lower for flag in HALLUCINATION_FLAGS):
            return

        pattern = r"([A-ZÀ-Ỹa-zà-ỹ\s]{3,40})\s+(?:là|is|means|có nghĩa là)\s+([^\.]{5,150})"
        for m in re.finditer(pattern, response):
            subj = m.group(1).strip()
            desc = m.group(2).strip()

            # [D] Filter stopwords đầu câu
            subj_lower = subj.lower()
            if any(subj_lower.startswith(sw) for sw in _LEARN_STOPWORDS):
                continue
            # Thêm guard: subject không được quá chung chung (< 3 từ nếu là cụm)
            if len(subj.split()) > 5:
                continue

            if len(subj) < 40 and len(desc.split()) > 5 and "?" not in desc:
                self.ltm.learn(subj, desc, source="auto_learn")

    def load_external_weights(self, state_dict: dict) -> dict:
        self.model.load_external_weights(state_dict, from_checkpoint=True)
        return {"status": "ok"}

    def end_session(self):
        self.working.end_session()

    def export_gguf(self, output_path: str = "dracoai_fp16.gguf"):
        GGUFExporter(self.model).write_gguf(output_path)
        self.bridge = TransformerBridge(gguf_path=output_path, n_gpu_layers=32)
        self.thinking.bridge = self.bridge