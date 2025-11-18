# MERLIN: Multimodal Embedding Reasoning for Low-level Imitation and Navigation

MERLIN is a **Vision–Language–Action (VLA)**-style transformer
implemented in **PyTorch**.

It takes as input:
- an RGB image of the scene  
- a natural language instruction  
- proprioceptive state (e.g., end-effector position)  

and predicts a **continuous action** (e.g., Δx, Δy) for a toy reaching task.

The goal of this project is to **study multimodal fusion and transformer-based policies**
on a small, fully reproducible benchmark that runs on a laptop.

---

## 🔍 Research Questions

MERLIN is designed to explore:

- How well can a small multimodal transformer learn a reaching policy?
- Does adding language instruction help, even when it’s simple?
- What is the effect of:
  - RGB vs RGB + depth
  - Frozen vs trainable text encoder
  - Number of transformer layers / heads
  - Different token orderings (e.g., `[readout, proprio, text, image]`)

---

## 🧱 Architecture

High-level architecture:

```text
        +-----------------+        +-------------------+
image ->| ImageTokenizer  |----+   |                   |
        +-----------------+    |   |                   |
                               +-->|                   |
text -->[ T5 Encoder + proj ]--+   | Merlin Transformer|--> [readout token] -> Action MLP
                               +-->|   Backbone        |
proprio->[ Proprio MLP ]-------+   |                   |
                                   |                   |
[learnable readout token] -------->+-------------------+
