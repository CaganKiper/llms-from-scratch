"""Chapter 9: Attention Mechanism

Core concepts:
- Attention score: how relevant is position j when processing position i?
- Scaled dot-product: softmax(Q K^T / sqrt(d_k)) V
- Query (Q): "What am I looking for?"
- Key   (K): "What do I contain?"
- Value (V): "What information do I carry?"
- Context vector: weighted sum of all values — resolves the bottleneck

Still bolted on top of an LSTM here. The sequential backbone is the next
thing we replace (Chapter 10).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmkit.solutions.ch08_lstm import lstm_step


# ============================================================================
# Attention Primitives
# ============================================================================

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled dot-product attention.

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

    Args:
        Q:    (batch, q_len, d_k)
        K:    (batch, k_len, d_k)
        V:    (batch, k_len, d_v)
        mask: (batch, q_len, k_len) bool mask — True positions are ignored

    Returns:
        output:  (batch, q_len, d_v)
        weights: (batch, q_len, k_len)  — attention distribution
    """
    d_k = Q.shape[-1]
    # Raw scores: (batch, q_len, k_len)
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    output  = weights @ V
    return output, weights


# ============================================================================
# Attention Layer
# ============================================================================

class AttentionLayer(nn.Module):
    """Single-head attention with learned Q, K, V projections.

    Takes a sequence of hidden states and returns context-aware representations.
    """

    def __init__(self, hidden_size, attention_dim=64):
        super().__init__()
        self.W_q = nn.Linear(hidden_size, attention_dim, bias=False)
        self.W_k = nn.Linear(hidden_size, attention_dim, bias=False)
        self.W_v = nn.Linear(hidden_size, attention_dim, bias=False)
        self.W_o = nn.Linear(attention_dim, hidden_size, bias=False)

    def forward(self, hidden_states, mask=None):
        """Apply attention to a sequence of hidden states.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            mask:          (batch, seq_len, seq_len) or None

        Returns:
            output:  (batch, seq_len, hidden_size)
            weights: (batch, seq_len, seq_len)
        """
        Q = self.W_q(hidden_states)
        K = self.W_k(hidden_states)
        V = self.W_v(hidden_states)
        context, weights = scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(context)
        return output, weights


# ============================================================================
# LSTM + Attention Language Model
# ============================================================================

class RNNWithAttention(nn.Module):
    """LSTM language model augmented with self-attention over hidden states.

    The LSTM still processes the sequence one step at a time, but instead of
    predicting from the final hidden state only, we apply attention over all
    hidden states to build a richer context vector at each position.
    """

    def __init__(self, vocab_size, embed_dim=64, hidden_size=128, attention_dim=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding   = nn.Embedding(vocab_size, embed_dim)
        # LSTM weights
        self.W_x = nn.Parameter(torch.randn(embed_dim,   4 * hidden_size) * 0.01)
        self.W_h = nn.Parameter(torch.randn(hidden_size, 4 * hidden_size) * 0.01)
        self.b   = nn.Parameter(torch.zeros(4 * hidden_size))
        # Attention
        self.attention   = AttentionLayer(hidden_size, attention_dim)
        self.output_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h0=None, c0=None):
        """Forward pass.

        Args:
            x: (batch, seq_len)

        Returns:
            logits: (batch, seq_len, vocab_size)
            weights: (batch, seq_len, seq_len) — attention weights
        """
        batch, seq_len = x.shape
        device = x.device
        if h0 is None:
            h0 = torch.zeros(batch, self.hidden_size, device=device)
        if c0 is None:
            c0 = torch.zeros(batch, self.hidden_size, device=device)

        # LSTM forward (handwritten loop)
        emb = self.embedding(x)
        h, c = h0, c0
        h_list = []
        for t in range(seq_len):
            h, c = lstm_step(emb[:, t, :], h, c, self.W_x, self.W_h, self.b)
            h_list.append(h)
        h_seq = torch.stack(h_list, dim=1)   # (batch, seq, hidden)

        # Attention over all hidden states
        context, weights = self.attention(h_seq)

        logits = self.output_head(context)
        return logits, weights


# ============================================================================
# Training
# ============================================================================

def train_attention(model, inputs, targets, n_epochs=10, batch_size=32,
                    lr=1e-3, verbose=True):
    """Train an RNNWithAttention model.

    Args:
        model:    RNNWithAttention instance
        inputs:   LongTensor (n, seq_len)
        targets:  LongTensor (n, seq_len)
        n_epochs, batch_size, lr, verbose: standard training args

    Returns:
        model, loss_history
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    n = len(inputs)
    loss_history = []

    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        inputs  = inputs[perm]
        targets = targets[perm]

        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, n, batch_size):
            xb = inputs[i : i + batch_size]
            yb = targets[i : i + batch_size]

            logits, _ = model(xb)
            bs, sl, v = logits.shape
            loss = criterion(logits.view(bs * sl, v), yb.view(bs * sl))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg = epoch_loss / n_batches
        loss_history.append(avg)
        if verbose:
            print(f"Epoch {epoch + 1}/{n_epochs}  loss: {avg:.4f}")

    return model, loss_history
