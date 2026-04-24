# Draco AI V1 🐉

**Draco AI V1** is a high-performance, localized Large Language Model (LLM) built upon the **Qwen 3.5 9B** base. This project focuses on architectural optimization, deep reasoning capabilities, and a sophisticated personalization system.

Developed with pride by **Draco Studio (Vietnam)**, led by **DUCNGUYEN-creator**.

---

## 🧠 Technical Architecture & Logic

The Draco AI V1 incorporates state-of-the-art algorithms and custom optimizations across several domains:

### 1. Core Transformer Architecture
| Algorithm / Technique | Description |
| :--- | :--- |
| **Mixture of Experts (MoE)** | Transforms Dense architecture into 8 specialized experts (0-3: Code, 4-7: Language), maximizing knowledge capacity while minimizing inference costs. |
| **SwiGLU Activation** | Implemented in FFN layers for each expert (bias-free) for better non-linear representation. |
| **RMSNorm** | Root Mean Square Layer Normalization (eps=1e-6) for stable training, compatible with Qwen 3.5. |
| **RoPE** | Rotary Position Embeddings (theta=1,000,000) for enhanced relative position awareness. |
| **GQA & SWA** | Grouped Query Attention and Sliding Window Attention (4096 tokens) to optimize KV-cache and memory. |
| **Sink Tokens** | Retains initial tokens as global attention anchors to prevent long-context forgetfulness. |

### 2. MoE Strategies
| Technique | Purpose |
| :--- | :--- |
| **Router Temp Scaling** | Prevents expert collapse by scaling logits (0.7-1.2). |
| **Routing Bias** | Breaks symmetry during initialization to ensure experts learn distinct roles. |
| **Dynamic Expert Pruning** | Automatically prunes experts based on an 85% threshold to reduce load. |
| **Capacity Fallback** | Re-routes tokens to under-loaded experts when capacity is reached. |

### 3. Generation & Reasoning
| Technique | Description |
| :--- | :--- |
| **Mirostat v2** | Adaptive sampling based on entropy to maintain target perplexity. |
| **Typical Sampling** | Ensures consistent text by selecting tokens near the distribution's average entropy. |
| **Repetition Penalty** | Penalties based on frequency and distance to prevent loopy text. |
| **Multi-Token Prediction** | Predicts two tokens simultaneously to boost generation speed and coherence. |
| **Identity Overlay** | Logit-level adjustments to reinforce "Draco" brand identity. |

### 4. Memory System
| Technique | Description |
| :--- | :--- |
| **TF-IDF + Hash Trick** | Lightweight 256-dim embedding for efficient memory retrieval. |
| **Intent Rerank** | Secondary ranking layer to match user intent beyond simple cosine similarity. |
| **LRU Cleanup** | Least Recently Used eviction policy to manage memory constraints. |

---

## 🛠 Project Structure

* `transformer_v1.py`: Core Transformer definitions and `DracoConfig` setup.
* `transformer_torch_v1.py`: PyTorch implementation optimized for training and inference.
* `memory_v1.py`: The management system for personalized long-term memory storage.
* `tokenizer.py`: Handles input processing based on the Qwen 3.5 vocabulary (151,936 tokens).

---

## ⚙️ Installation

To set up the environment for Draco AI V1, please install the following dependencies:

```bash
pip install numpy
pip install torch torchvision torchaudio
```
---

## 🤝 Community & Contribution

We believe in the power of open-source and community-driven AI. **Draco AI V1** is a product of **Draco Studio (Vietnam)**. We welcome all developers, researchers, and AI enthusiasts to contribute!

* **Lead Developer:** DUCNGUYEN-creator
* **💬 Discord Community:** [Join Draco Studio Discord](https://discord.gg/JfStzfkTH)

---

## 📄 License

This project is licensed under the **Apache License 2.0**.  
Copyright 2026 **The Draco Studio** and **DUCNGUYEN-creator**.

---
*Built with ❤️ in Vietnam by Draco Studio.*