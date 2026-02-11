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

---

## Installation

### Install uv:

`curl -LsSf https://astral.sh/uv/install.sh | sh`

### Verify the installation:

`echo 'export PATH="$HOME/snap/code/221/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc`    

`uv --version`

## Clone the repository or Download as zip file

`git clone https://github.com/yoursrealkiran/merlin_vla.git`

`cd merlin_vla`

## Environment Setup

### Create a Virual Environment

#### In the terminal, run the below command to create a virtual environment

`uv venv`

#### Activate the environment

`source .venv/bin/activate`

#### Install Dependencies

`uv sync`

#### Running evaluation

`uv run python merlin/eval/eval_reach.py --ckpt checkpoints/merlin_toy_reach_rgb/<your_checkpoint_file>`


