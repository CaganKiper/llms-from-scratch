"""Chapter 10: The Transformer

Core concepts:
- Self-attention: every token attends to every other token in the same sequence
- Causal mask: prevent attending to future positions during language modeling
- Positional encoding: inject position info since there is no recurrence
- Multi-head attention: parallel attention heads learn different relationships
- Residual connections + LayerNorm: enable training deep stacks
- Transformer block: MHA → Add&Norm → FFN → Add&Norm

Contextual embeddings emerge naturally: "bank" in "river bank" vs "bank
account" gets different representations after passing through attention layers.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from llmkit.solutions.ch09_attention import scaled_dot_product_attention


# ============================================================================
# Positional Encoding & Causal Mask
# ============================================================================

def make_causal_mask(seq_len, device=None):
    """Upper-triangular causal mask.

    True  = "do not attend here" (future position).
    False = "allowed to attend" (current or past).

    Args:
        seq_len: Sequence length
        device:  Target device

    Returns:
        mask: BoolTensor (1, seq_len, seq_len)
    """
    # torch.triu with diagonal=1 marks everything strictly above the diagonal
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0)   # (1, seq, seq) — broadcasts over batch


def positional_encoding(seq_len, d_model, device=None):
    """Sinusoidal positional encoding.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        seq_len: Sequence length
        d_model: Model dimension
        device:  Target device

    Returns:
        pe: (1, seq_len, d_model)  — ready to add to embeddings
    """
    pe  = torch.zeros(seq_len, d_model, device=device)
    pos = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float, device=device)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)   # (1, seq_len, d_model)


# ============================================================================
# Multi-Head Attention (handwritten)
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention.

    Splits the model dimension into n_heads parallel attention heads,
    each learning different types of token relationships.
    Outputs are concatenated and projected back.

    MultiHead(Q,K,V) = Concat(head_1, ..., head_h) W_O
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        """Apply multi-head attention.

        Args:
            x:    (batch, seq_len, d_model)
            mask: (1 or batch, seq_len, seq_len) bool mask

        Returns:
            output:      (batch, seq_len, d_model)
            all_weights: (batch, n_heads, seq_len, seq_len)
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V and split into heads
        Q = self.W_q(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        # Now Q, K, V: (batch, n_heads, seq_len, d_head)

        # Expand mask for the heads dimension if needed
        if mask is not None:
            # mask: (1 or batch, seq_len, seq_len) → (batch, 1, seq_len, seq_len)
            mask = mask.unsqueeze(1)

        # Reshape to (batch*n_heads, seq, d_head) to reuse our primitive
        bh = batch * self.n_heads
        Q_ = Q.contiguous().view(bh, seq_len, self.d_head)
        K_ = K.contiguous().view(bh, seq_len, self.d_head)
        V_ = V.contiguous().view(bh, seq_len, self.d_head)
        mask_ = mask.expand(batch, self.n_heads, seq_len, seq_len).contiguous().view(
            bh, seq_len, seq_len
        ) if mask is not None else None

        context, weights_flat = scaled_dot_product_attention(Q_, K_, V_, mask_)
        # context: (batch*n_heads, seq, d_head)
        # Reassemble heads
        context = context.view(batch, self.n_heads, seq_len, self.d_head)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        output  = self.W_o(context)

        weights = weights_flat.view(batch, self.n_heads, seq_len, seq_len)
        return output, weights


# ============================================================================
# Feed-Forward Network
# ============================================================================

class FeedForward(nn.Module):
    """Position-wise feed-forward network.

    Two linear layers with GELU activation in between.
    Expands to 4 * d_model then projects back.
    """

    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


# ============================================================================
# Transformer Block
# ============================================================================

class TransformerBlock(nn.Module):
    """Single transformer decoder block.

    1. Multi-head self-attention
    2. Residual add + LayerNorm
    3. Feed-forward network
    4. Residual add + LayerNorm
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn   = MultiHeadAttention(d_model, n_heads)
        self.ff     = FeedForward(d_model)
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """Apply one transformer block.

        Args:
            x:    (batch, seq_len, d_model)
            mask: causal mask

        Returns:
            x:       (batch, seq_len, d_model)
            weights: (batch, n_heads, seq_len, seq_len)
        """
        # Self-attention + residual + norm
        attn_out, weights = self.attn(x, mask)
        x = self.norm1(x + attn_out)
        # Feed-forward + residual + norm
        x = self.norm2(x + self.ff(x))
        return x, weights


# ============================================================================
# Full Transformer Language Model
# ============================================================================

class TransformerLanguageModel(nn.Module):
    """GPT-style autoregressive transformer language model.

    Architecture:
        token ids
          → token embedding + positional encoding
          → N × TransformerBlock (with causal mask)
          → LayerNorm
          → linear output head
          → logits over vocabulary
    """

    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, max_seq_len=256):
        super().__init__()
        self.d_model     = d_model
        self.max_seq_len = max_seq_len
        self.embedding   = nn.Embedding(vocab_size, d_model)
        # Learned positional embedding (GPT-style)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks       = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.norm         = nn.LayerNorm(d_model)
        self.output_head  = nn.Linear(d_model, vocab_size, bias=False)
        # Tie output weights to input embedding (common practice)
        self.output_head.weight = self.embedding.weight

    def forward(self, x):
        """Forward pass with causal masking.

        Args:
            x: (batch, seq_len) — token ids

        Returns:
            logits:      (batch, seq_len, vocab_size)
            all_weights: list of (batch, n_heads, seq_len, seq_len) per layer
        """
        batch, seq_len = x.shape
        device = x.device

        # Token + positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(positions)

        # Causal mask: (1, seq_len, seq_len)
        mask = make_causal_mask(seq_len, device=device)

        all_weights = []
        for block in self.blocks:
            h, weights = block(h, mask)
            all_weights.append(weights)

        h      = self.norm(h)
        logits = self.output_head(h)
        return logits, all_weights


# ============================================================================
# Training
# ============================================================================

def train_transformer(model, inputs, targets, n_epochs=10, batch_size=32,
                      lr=3e-4, verbose=True):
    """Train a TransformerLanguageModel.

    Args:
        model:    TransformerLanguageModel instance
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
