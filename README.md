# Draco AI V1 🐉

**Draco AI V1** is a high-performance, localized Large Language Model (LLM) built upon the **Qwen 3.5 9B** base. This project focuses on architectural optimization, deep reasoning capabilities, and a sophisticated personalization system.

Developed with pride by **Draco Studio (Vietnam)**, led by **DUCNGUYEN-creator**.

---

## 🚀 Key Features

### 1. MoE-ization (Mixture of Experts)
We have successfully transformed the original **Dense** architecture into a **Mixture of Experts (MoE)** structure.
* **Efficiency:** Maintains massive knowledge capacity while significantly reducing computational overhead during inference.
* **Performance:** Optimized for running high-level AI on consumer-grade hardware.

### 2. Deep Reasoning Integration
Draco AI V1 incorporates advanced logical reasoning workflows, allowing the model to "think" through complex problems step-by-step before generating a final response.

### 3. Advanced Long-Term Memory
Unlike standard LLMs, Draco AI features a dedicated memory system:
* **Deep Personalization:** Adapts to user habits, preferences, and historical data.
* **Consistency:** Maintains long-term conversational context for a truly human-like interaction.

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

🤝 Community & Contribution
We believe in the power of open-source and community-driven AI. Draco AI V1 is a product of Draco Studio (Vietnam). We welcome all developers, researchers, and AI enthusiasts to contribute!

Lead Developer: DUCNGUYEN-creator

💬Discord Community: [Join Draco Studio Discord](https://discord.gg/JfStzfkTH)

📄 License
This project is licensed under the Apache License 2.0.
Copyright 2026 The Draco Studio and DUCNGUYEN-creator.

Built with ❤️ in Vietnam by Draco Studio.