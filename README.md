# Building an LLM from Scratch

An interactive textbook teaching how to build a large language model from scratch, using only NumPy (Part I) and PyTorch (Part II). By the end, you will have implemented every component needed for a working GPT-style language model.

## The North Star

An LLM is a **next-word predictor**. That's the entire idea. Everything in this textbook answers two questions:

1. **What does it mean to predict?** How does a machine guess the next thing in a sequence?
2. **What is a "word" to a machine?** How do we turn language into numbers?

## Project Structure

```
llms-from-scratch/
├── notebooks/
│   ├── part1_text_to_prediction.ipynb      # Chapters 1-6 (available now)
│   └── part2_sequences_to_transformers.ipynb # Chapters 7-11 (coming soon)
├── llmkit/                                   # Supporting library
│   ├── data.py                              # Data generators + WikiText-2 loader
│   ├── viz.py                               # All visualizations
│   └── solutions/                           # Reference implementations
│       ├── ch01_regression.py
│       ├── ch02_onehot.py
│       ├── ch03_logistic.py
│       ├── ch04_word2vec.py
│       ├── ch05_mlp.py
│       └── ch06_bpe.py
├── pyproject.toml
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- Comfortable with linear algebra (vectors, matrices, dot products)
- Comfortable with calculus (derivatives, chain rule)
- Basic Python/NumPy

### Installation

```bash
# Clone or download this repository
cd llms-from-scratch

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh  # On Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Sync dependencies (creates venv automatically)
uv sync

# Launch Jupyter
uv run jupyter notebook notebooks/part1_text_to_prediction.ipynb
```

## Part I: From Text to Prediction

**Status:** Complete

Build a complete (if naive) language model pipeline. By the end, you can predict the next token from a bag of embeddings—but with no sense of order.

### Chapters

1. **Linear Regression** — Learn prediction via gradient descent
2. **One-Hot Encoding** — Turn words into numbers (terrible numbers)
3. **Logistic Regression** — Combine representation + prediction (the LLM skeleton!)
4. **Word2Vec** — Learn meaningful embeddings from context
5. **MLP + Backpropagation** — Capture non-linear patterns
6. **BPE Tokenization** — Handle any word with subword units

**Data:** Synthetic data (Ch 1-3, 5) for clarity, WikiText-2 (Ch 4, 6) for real language

**Tools:** Pure NumPy — every gradient computed by hand

**Output:** A working next-token predictor (order-blind)

## Part II: From Sequences to Transformers

**Status:** In Progress

Give the model a sense of sequence, then progressively upgrade until you have a transformer.

### Chapters (Planned)

7. **RNN** — Process sequences with hidden state
8. **LSTM** — Long-term memory via gates
9. **Attention** — Look at all positions, not just compressed state
10. **Transformer** — Attention is all you need (multi-head, causal masking, residual connections)
11. **Autoregressive Generation** — Turn predictions into actual text

**Data:** WikiText-2 throughout

**Tools:** PyTorch (handwritten forward pass, autograd for backward pass)

**Output:** A working GPT-style language model that generates coherent text

## How It Works

Interactive Jupyter notebooks with hands-on exercises. Each chapter includes:
- Concept explanation
- Math formulation
- Coding exercises with reference implementations
- Visualizations

The `llmkit/` library provides:
- **`data.py`** — Data generators and WikiText-2 loader
- **`viz.py`** — Visualization utilities
- **`solutions/`** — Reference implementations for each chapter

## Learning Path

### If you're new to ML:

Start with Part I, Chapter 1. Take it slow. The concepts build on each other.

### If you know ML basics:

Skim Chapters 1-3, focus on Chapters 4-6 (embeddings, backprop, tokenization).

### If you know deep learning:

Jump to Part II for RNNs → Transformers. Part I is foundation review.

## Curriculum

This textbook follows a two-track approach:

**Representation Track:**
One-hot → Word2Vec → BPE → Contextual embeddings (via Transformer)

**Prediction Track:**
Linear regression → Logistic regression → MLP → RNN → LSTM → Attention → Transformer

Both tracks converge at Chapter 10, where the transformer architecture produces contextual embeddings as a byproduct of its prediction mechanism.

## Dependencies

All dependencies are managed via `pyproject.toml`:

- **NumPy** (1.24+) — For all math in Part I
- **Matplotlib** (3.7+) — For visualizations
- **PyTorch** (2.0+) — For Part II (autograd, optimizers)
- **Jupyter** — For running interactive notebooks

Minimal by design. No TensorFlow, no Hugging Face, no black boxes.


