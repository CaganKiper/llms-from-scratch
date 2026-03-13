"""Chapter 7: Recurrent Neural Network

Core concepts:
- Hidden state: A vector that carries information forward through the sequence
- RNN step: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b)
- Sequence processing: one token at a time, accumulating context
- Vanishing gradient: gradients shrink when multiplied across many timesteps
- PyTorch autograd replaces manual backprop from here on
"""

import torch
import torch.nn as nn


# ============================================================================
# Data Preparation
# ============================================================================

def prepare_sequences(token_ids, seq_len=32):
    """Create (input, target) pairs using a sliding window over token ids.

    Each input is a sequence of length seq_len; the target is that same
    sequence shifted one step to the right (next-token prediction).

    Args:
        token_ids: List or 1-D tensor of integer token ids
        seq_len: Length of each input sequence

    Returns:
        inputs: LongTensor (n_sequences, seq_len)
        targets: LongTensor (n_sequences, seq_len)  — inputs shifted by 1
    """
    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor(token_ids, dtype=torch.long)

    inputs, targets = [], []
    for i in range(len(token_ids) - seq_len):
        inputs.append(token_ids[i : i + seq_len])
        targets.append(token_ids[i + 1 : i + seq_len + 1])

    return torch.stack(inputs), torch.stack(targets)


# ============================================================================
# RNN Primitives (handwritten, no nn.RNN)
# ============================================================================

def rnn_step(x_t, h_prev, W_xh, W_hh, b_h):
    """Single RNN timestep.

    h_t = tanh(x_t @ W_xh + h_prev @ W_hh + b_h)

    Args:
        x_t:    (batch, input_size)  — current token embedding
        h_prev: (batch, hidden_size) — previous hidden state
        W_xh:   (input_size, hidden_size)
        W_hh:   (hidden_size, hidden_size)
        b_h:    (hidden_size,)

    Returns:
        h_t: (batch, hidden_size)
    """
    return torch.tanh(x_t @ W_xh + h_prev @ W_hh + b_h)


def rnn_forward(x_seq, h0, W_xh, W_hh, b_h):
    """Run RNN over a full sequence.

    Args:
        x_seq: (batch, seq_len, input_size) — sequence of embeddings
        h0:    (batch, hidden_size)          — initial hidden state
        W_xh, W_hh, b_h: parameters (see rnn_step)

    Returns:
        h_seq: (batch, seq_len, hidden_size) — all hidden states
        h_T:   (batch, hidden_size)          — final hidden state
    """
    batch, seq_len, _ = x_seq.shape
    h = h0
    h_seq = []
    for t in range(seq_len):
        h = rnn_step(x_seq[:, t, :], h, W_xh, W_hh, b_h)
        h_seq.append(h)
    h_seq = torch.stack(h_seq, dim=1)   # (batch, seq_len, hidden)
    return h_seq, h


# ============================================================================
# RNN Language Model
# ============================================================================

class RNNLanguageModel(nn.Module):
    """Language model built from a handwritten RNN.

    Architecture:
        token ids  →  embedding  →  RNN (handwritten)  →  linear  →  logits
    """

    def __init__(self, vocab_size, embed_dim=64, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # RNN weights stored as nn.Parameter so the optimizer sees them
        self.W_xh = nn.Parameter(torch.randn(embed_dim, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h  = nn.Parameter(torch.zeros(hidden_size))
        self.output_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h0=None):
        """Forward pass.

        Args:
            x:  (batch, seq_len) — token id sequences
            h0: (batch, hidden_size) or None (zeros)

        Returns:
            logits: (batch, seq_len, vocab_size)
            h_T:    (batch, hidden_size) — final hidden state
        """
        batch = x.shape[0]
        if h0 is None:
            h0 = torch.zeros(batch, self.hidden_size, device=x.device)
        emb = self.embedding(x)                           # (batch, seq, embed)
        h_seq, h_T = rnn_forward(emb, h0,
                                 self.W_xh, self.W_hh, self.b_h)
        logits = self.output_head(h_seq)                  # (batch, seq, vocab)
        return logits, h_T


# ============================================================================
# Training
# ============================================================================

def train_rnn(model, inputs, targets, n_epochs=10, batch_size=32,
              lr=1e-3, verbose=True):
    """Train an RNNLanguageModel.

    Args:
        model:    RNNLanguageModel instance
        inputs:   LongTensor (n, seq_len)
        targets:  LongTensor (n, seq_len)
        n_epochs: Number of passes over the data
        batch_size: Mini-batch size
        lr:       Learning rate for Adam
        verbose:  Print loss every epoch

    Returns:
        model:        Trained model
        loss_history: List of per-epoch average losses
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    n = len(inputs)
    loss_history = []

    for epoch in range(n_epochs):
        # Shuffle
        perm = torch.randperm(n)
        inputs  = inputs[perm]
        targets = targets[perm]

        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, n, batch_size):
            xb = inputs[i : i + batch_size]
            yb = targets[i : i + batch_size]

            logits, _ = model(xb)                         # (batch, seq, vocab)
            batch_sz, seq_len, vocab = logits.shape
            loss = criterion(logits.view(batch_sz * seq_len, vocab),
                             yb.view(batch_sz * seq_len))

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping — prevents exploding gradients common in RNNs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg = epoch_loss / n_batches
        loss_history.append(avg)
        if verbose:
            print(f"Epoch {epoch + 1}/{n_epochs}  loss: {avg:.4f}")

    return model, loss_history
