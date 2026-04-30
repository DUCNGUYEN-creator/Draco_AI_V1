"""
DracoAI V1 — Advanced Trainer
==============================
Huấn luyện đa chế độ cho DracoTransformerTorchV1.

Các chế độ:
  • full      – Huấn luyện toàn bộ mô hình
  • router    – Chỉ huấn luyện router MoE (freeze experts)
  • skill     – Huấn luyện một nhóm expert (code / language)
  • thinking  – Huấn luyện chuỗi suy nghĩ (logic + debug experts)
  • lora      – Huấn luyện nhẹ với LoRA
  • qlora     – Huấn luyện cực nhẹ với QLoRA (4‑bit)
  • chat      – Huấn luyện trên dữ liệu hội thoại (ChatML)
  • pretrain  – Huấn luyện trên văn bản thuần / code hỗn hợp (tương tự full)

Điểm nổi bật:
  - Gradient accumulation thực sự (sửa lỗi accumulation bị bỏ quên)
  - Resume an toàn với LoRA/QLoRA
  - Hỗ trợ dữ liệu chat đầu vào dạng danh sách message
  - Đồng bộ trọng số với NumPy model sau mỗi save
"""

import os, json, math, time, random
import numpy as np
from typing import Optional, List, Dict, Any, Union, Tuple

# ── Import PyTorch (nếu có) ──────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from transformer_torch_v1 import (
        DracoTransformerTorchV1,
        train_step,          # vẫn dùng tham khảo, nhưng tự viết vòng lặp
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ── Import NumPy model (để đồng bộ) ──────────────────────────────────
try:
    from transformer_v1 import DracoTransformerV1, GGUFExporter
except ImportError:
    DracoTransformerV1 = None
    GGUFExporter = None


class TrainerV1:
    """
    Trình huấn luyện đa năng cho DracoAI V1.

    Args:
        checkpoint_dir: thư mục lưu checkpoint
        config:         dictionary cấu hình mô hình
        numpy_model:    mô hình NumPy inference để đồng bộ (có thể None)
        tokenizer:      tokenizer (BPETokenizer) để tokenize văn bản
        device:         thiết bị ('cpu', 'cuda', …)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        config:         dict,
        numpy_model:    Optional[DracoTransformerV1] = None,
        tokenizer:      Any = None,
        device:         Optional[str] = None,
    ):
        self.ckpt_dir    = checkpoint_dir
        self.config      = config
        self.numpy_model = numpy_model
        self.tokenizer   = tokenizer
        self.device      = device or (
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else "cpu"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────
    # API chính: train() – dành cho dữ liệu text thuần hoặc code
    # ──────────────────────────────────────────────────────────────────
    def train(
        self,
        data:             List[Union[str, List[int]]],
        steps:            int = 5000,
        batch_size:       int = 1,
        accumulation:     int = 1,        # Số batch tích luỹ trước khi update
        mode:             str = "full",
        skill_group:      Optional[str] = None,
        lora_rank:        int = 8,
        lora_alpha:       float = 16.0,
        max_lr:           float = 3e-4,
        min_lr:           float = 3e-5,
        warmup_steps:     int = 200,
        grad_clip:        float = 1.0,
        log_every:        int = 50,
        save_every:       int = 500,
        resume:           bool = False,
    ):
        """
        Huấn luyện trên văn bản thô (hoặc token ids).
        data: list các string hoặc list các token id.
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for training.")

        print(f"\n{'='*60}\n DracoAI V1 Training | mode={mode} | steps={steps}\n{'='*60}")

        # Tokenize nếu cần
        data_ids = self._prepare_data(data)

        # Khởi tạo hoặc resume model
        model = self._init_model(mode, skill_group, lora_rank, lora_alpha, resume)

        # Optimizer & scaler
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1)
        scaler = torch.cuda.amp.GradScaler(enabled=(self.device == "cuda"))

        # Resume step
        start_step = self._get_resume_step(resume)

        # DataLoader cho batch
        ctx = self.config["window"]
        if batch_size > 1:
            dataset = self._create_dataset(data_ids, ctx)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            loader_iter = iter(loader)
        else:
            loader = loader_iter = None

        loss_hist = []
        t0 = time.time()
        step = start_step
        accum_steps = 0  # Số batch đã tích luỹ trong optimizer

        while step < steps:
            # Lấy batch
            if batch_size > 1:
                try:
                    x, y = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader)
                    x, y = next(loader_iter)
            else:
                s = random.randint(0, len(data_ids) - ctx - 2)
                x = torch.tensor(data_ids[s:s+ctx], dtype=torch.long).unsqueeze(0).to(self.device)
                y = torch.tensor(data_ids[s:s+ctx+2], dtype=torch.long).unsqueeze(0).to(self.device)

            # Tính learning rate
            lr = self._cosine_lr(step, steps, max_lr, min_lr, warmup_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr

            # Forward & loss
            loss = self._compute_loss(model, x, y)
            if loss is None or not torch.isfinite(loss):
                continue

            # Backward (tích luỹ gradient)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_steps += 1
            loss_hist.append(loss.item())

            # Chỉ update optimizer khi đã đủ batch tích luỹ
            if accum_steps >= accumulation:
                # Clip gradient
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                accum_steps = 0

                # Sau khi update mới tăng step (để step đếm số lần update)
                step += 1

            # Logging
            if step % log_every == 0 and step > 0:
                avg_loss = sum(loss_hist[-log_every:]) / min(len(loss_hist), log_every)
                elapsed = time.time() - t0
                print(f"  Step {step:5d}/{steps} | loss={avg_loss:.4f} | "
                      f"ppl={math.exp(min(avg_loss,20)):.1f} | lr={lr:.2e} | "
                      f"{step*ctx/elapsed/1000:.1f}K tok/s")

            # Checkpoint & sync
            if step % save_every == 0 and step > 0:
                self._save_checkpoint(model, step, loss_hist[-1] if loss_hist else 0.0)
                if self.numpy_model is not None:
                    self._sync_to_numpy(model)

        # Lưu cuối cùng
        self._save_checkpoint(model, step, loss_hist[-1] if loss_hist else 0.0)
        if self.numpy_model is not None:
            self._sync_to_numpy(model)
        print(f"[Trainer] Training complete ({step} steps).")

    # ──────────────────────────────────────────────────────────────────
    # API cho dữ liệu chat (hội thoại)
    # ──────────────────────────────────────────────────────────────────
    def train_chat(
        self,
        conversations:    List[List[Dict[str, str]]],  # Mỗi phần tử là list các message
        steps:            int = 5000,
        batch_size:       int = 1,
        accumulation:     int = 1,
        mode:             str = "full",
        skill_group:      Optional[str] = None,
        lora_rank:        int = 8,
        lora_alpha:       float = 16.0,
        max_lr:           float = 3e-4,
        min_lr:           float = 3e-5,
        warmup_steps:     int = 200,
        grad_clip:        float = 1.0,
        log_every:        int = 50,
        save_every:       int = 500,
        resume:           bool = False,
    ):
        """
        Huấn luyện trên dữ liệu hội thoại (ChatML).
        Mỗi cuộc hội thoại là một list các dict {"role": ..., "content": ...}.
        Tokenizer phải hỗ trợ encode_chat (ChatML).
        """
        if self.tokenizer is None or not hasattr(self.tokenizer, "encode_chat"):
            raise RuntimeError("Tokenizer with encode_chat support is required for chat training.")

        print(f"\n{'='*60}\n DracoAI V1 Chat Training | mode={mode} | steps={steps}\n{'='*60}")

        # Chuyển đổi conversations thành danh sách các token ids (đã được encode_chat)
        all_ids = []
        for conv in conversations:
            ids = self.tokenizer.encode_chat(conv, add_generation_prompt=True)
            all_ids.append(ids)
        # Ghép tất cả thành một chuỗi dài (có thể thêm token phân cách nếu cần)
        flat_ids = []
        for seg in all_ids:
            flat_ids.extend(seg)
        # Thêm token EOS giữa các đoạn? Tokenizer tự lo.

        # Dùng chung logic train với dữ liệu đã tokenize
        self.train(
            data=[flat_ids],
            steps=steps,
            batch_size=batch_size,
            accumulation=accumulation,
            mode=mode,
            skill_group=skill_group,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            max_lr=max_lr,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
            grad_clip=grad_clip,
            log_every=log_every,
            save_every=save_every,
            resume=resume,
        )

    # ──────────────────────────────────────────────────────────────────
    # Helper: chuẩn bị dữ liệu
    # ──────────────────────────────────────────────────────────────────
    def _prepare_data(self, data):
        if self.tokenizer and data and isinstance(data[0], str):
            print("[Trainer] Tokenizing data...")
            all_ids = []
            for text in data:
                ids = self.tokenizer.encode(text, add_bos=False, add_eos=False)
                all_ids.extend(ids)
            return all_ids
        return data

    # ──────────────────────────────────────────────────────────────────
    # Khởi tạo model, áp dụng mode và resume
    # ──────────────────────────────────────────────────────────────────
    def _init_model(self, mode, skill_group, lora_rank, lora_alpha, resume):
        if resume:
            # Tìm checkpoint mới nhất
            model = self._load_latest_checkpoint(mode, skill_group, lora_rank, lora_alpha)
        else:
            model = DracoTransformerTorchV1(self.config).to(self.device)
            self._apply_mode(model, mode, skill_group, lora_rank, lora_alpha)
        return model

    def _apply_mode(self, model, mode, skill_group, lora_rank, lora_alpha):
        if mode == "full" or mode == "pretrain" or mode == "mixed":
            print(f"[Trainer] Mode: {mode} (all parameters trainable)")
        elif mode == "router":
            print("[Trainer] Mode: Router-only training")
            model.freeze_experts()
        elif mode == "skill":
            if skill_group not in ("code", "language"):
                raise ValueError("skill_group must be 'code' or 'language'")
            print(f"[Trainer] Mode: Skill training — {skill_group}")
            self._freeze_non_skill(model, skill_group)
        elif mode == "thinking":
            print("[Trainer] Mode: Thinking (CoT) training")
            self._freeze_non_thinking(model)
        elif mode == "lora":
            print(f"[Trainer] Mode: LoRA (rank={lora_rank}, alpha={lora_alpha})")
            model.enable_lora(rank=lora_rank, alpha=lora_alpha)
        elif mode == "qlora":
            print(f"[Trainer] Mode: QLoRA (rank={lora_rank}, alpha={lora_alpha})")
            model.enable_qlora(rank=lora_rank, alpha=lora_alpha)
        else:
            raise ValueError(f"Unknown training mode: {mode}")

    def _freeze_non_skill(self, model, skill: str):
        keep = set(range(0, 4)) if skill == "code" else set(range(4, 8))
        for blk in model.blocks:
            for e, exp in enumerate(blk.moe.experts):
                for p in exp.parameters():
                    p.requires_grad = (e in keep)
            for p in blk.moe.shared.parameters():
                p.requires_grad = False

    def _freeze_non_thinking(self, model):
        keep = {0, 2}
        for blk in model.blocks:
            for e, exp in enumerate(blk.moe.experts):
                for p in exp.parameters():
                    p.requires_grad = (e in keep)
            for p in blk.moe.shared.parameters():
                p.requires_grad = False
            for p in blk.attn.parameters():
                p.requires_grad = False

    # ──────────────────────────────────────────────────────────────────
    # Resume step
    # ──────────────────────────────────────────────────────────────────
    def _get_resume_step(self, resume: bool) -> int:
        if not resume:
            return 0
        state_path = os.path.join(self.ckpt_dir, "trainer_state.json")
        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
            return state.get("step", 0)
        return 0

    # ──────────────────────────────────────────────────────────────────
    # Load checkpoint (an toàn với LoRA/QLoRA)
    # ──────────────────────────────────────────────────────────────────
    def _load_latest_checkpoint(self, mode: str, skill_group, lora_rank, lora_alpha):
        ckpts = [f for f in os.listdir(self.ckpt_dir) if f.startswith("trainer_") and f.endswith(".pt")]
        if not ckpts:
            print("[Trainer] No checkpoint found — starting from scratch.")
            model = DracoTransformerTorchV1(self.config).to(self.device)
            self._apply_mode(model, mode, skill_group, lora_rank, lora_alpha)
            return model

        ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(self.ckpt_dir, x)))
        filename = ckpts[-1]
        print(f"[Trainer] Loading checkpoint: {filename}")
        model, _ = DracoTransformerTorchV1.load_checkpoint(
            self.ckpt_dir, filename=filename, current_config=self.config
        )
        model = model.to(self.device)
        # Áp dụng mode training hiện tại (có thể khác với mode khi lưu checkpoint)
        self._apply_mode(model, mode, skill_group, lora_rank, lora_alpha)
        return model

    # ──────────────────────────────────────────────────────────────────
    # Tính loss (tương tự train_step nhưng không step)
    # ──────────────────────────────────────────────────────────────────
    def _compute_loss(self, model, input_ids: torch.Tensor, labels: torch.Tensor):
        """Tính CE loss cho l1 và MTP loss cho l2."""
        base_model = model.module if hasattr(model, "module") else model
        l1, l2, aux_total = model(input_ids, return_aux=True)
        ce_loss = F.cross_entropy(
            l1[:, :-1].reshape(-1, base_model.vocab_size),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        mtp_coeff = getattr(base_model, "_mtp_aux_coeff", 0.1)
        if mtp_coeff > 0 and l2.shape[1] > 1:
            mtp_loss = F.cross_entropy(
                l2[:, :-1].reshape(-1, base_model.vocab_size),
                labels[:, 2:].reshape(-1) if labels.shape[1] > 2 else labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            loss = ce_loss + 0.01 * aux_total + mtp_coeff * mtp_loss
        else:
            loss = ce_loss + 0.01 * aux_total
        return loss

    # ──────────────────────────────────────────────────────────────────
    # Dataset đơn giản cho văn bản
    # ──────────────────────────────────────────────────────────────────
    class _SimpleDataset(Dataset):
        def __init__(self, ids, ctx):
            self.ids = ids
            self.ctx = ctx
        def __len__(self):
            return len(self.ids) - self.ctx - 2
        def __getitem__(self, idx):
            x = self.ids[idx : idx + self.ctx]
            y = self.ids[idx : idx + self.ctx + 2]
            return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def _create_dataset(self, ids, ctx):
        return self._SimpleDataset(ids, ctx)

    # ──────────────────────────────────────────────────────────────────
    # Checkpoint & sync
    # ──────────────────────────────────────────────────────────────────
    def _save_checkpoint(self, model, step, loss):
        path = os.path.join(self.ckpt_dir, f"trainer_{step:06d}.pt")
        torch.save({
            "step":   step,
            "model":  model.state_dict(),
            "config": self.config,
        }, path)
        state = {"step": step, "loss": loss}
        with open(os.path.join(self.ckpt_dir, "trainer_state.json"), "w") as f:
            json.dump(state, f)
        print(f"[Trainer] Checkpoint saved: {path}")

    def _sync_to_numpy(self, torch_model):
        if self.numpy_model is None:
            return
        try:
            torch_model._full_sync(self.numpy_model)
            print("[Trainer] Weights synced to NumPy model.")
        except Exception as e:
            print(f"[Trainer] Sync failed: {e}")

    # ──────────────────────────────────────────────────────────────────
    # Learning rate schedule
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _cosine_lr(step, total, max_lr, min_lr, warmup):
        if step < warmup:
            return max_lr * step / max(warmup, 1)
        if step >= total:
            return min_lr
        progress = (step - warmup) / (total - warmup)
        return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)

    # ──────────────────────────────────────────────────────────────────
    # Export GGUF
    # ──────────────────────────────────────────────────────────────────
    def export_gguf(self, torch_model, output_path: str):
        self._sync_to_numpy(torch_model)
        if GGUFExporter is None:
            raise RuntimeError("transformer_v1.py not found.")
        from transformer_v1 import GGUFExporter
        GGUFExporter(self.numpy_model).write_gguf(output_path)
        print(f"[Trainer] GGUF exported to {output_path}")


# Alias
DracoTrainer = TrainerV1