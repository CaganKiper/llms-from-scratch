"""Chapter 8: Long Short-Term Memory (LSTM)

Core concepts:
- Cell state: a separate highway for long-term information (c_t)
- Forget gate:  f_t = sigmoid(x_t @ W_xf + h_{t-1} @ W_hf + b_f)
- Input gate:   i_t = sigmoid(x_t @ W_xi + h_{t-1} @ W_hi + b_i)
- Gate values:  g_t = tanh(x_t @ W_xg + h_{t-1} @ W_hg + b_g)
- Output gate:  o_t = sigmoid(x_t @ W_xo + h_{t-1} @ W_ho + b_o)
- Cell update:  c_t = f_t * c_{t-1} + i_t * g_t
- Hidden state: h_t = o_t * tanh(c_t)

PyTorch autograd handles the backward pass.
"""

import torch
import torch.nn as nn


# ============================================================================
# LSTM Primitives (handwritten, no nn.LSTM)
# ============================================================================

def lstm_step(x_t, h_prev, c_prev, W_x, W_h, b):
    """Single LSTM timestep.

    All four gate linear maps are batched into two large matrix multiplications
    then split: this is faster and keeps the code readable.

    Args:
        x_t:    (batch, input_size)
        h_prev: (batch, hidden_size)
        c_prev: (batch, hidden_size)
        W_x:    (input_size,  4 * hidden_size)  — input→gates
        W_h:    (hidden_size, 4 * hidden_size)  — hidden→gates
        b:      (4 * hidden_size,)

    Returns:
        h_t: (batch, hidden_size)
        c_t: (batch, hidden_size)
    """
    gates = x_t @ W_x + h_prev @ W_h + b   # (batch, 4*hidden)
    hidden_size = h_prev.shape[1]

    f = torch.sigmoid(gates[:, 0 * hidden_size : 1 * hidden_size])   # forget
    i = torch.sigmoid(gates[:, 1 * hidden_size : 2 * hidden_size])   # input
    g = torch.tanh(   gates[:, 2 * hidden_size : 3 * hidden_size])   # gate (cell candidate)
    o = torch.sigmoid(gates[:, 3 * hidden_size : 4 * hidden_size])   # output

    c_t = f * c_prev + i * g
    h_t = o * torch.tanh(c_t)
    return h_t, c_t


def lstm_forward(x_seq, h0, c0, W_x, W_h, b):
    """Run LSTM over a full sequence.

    Args:
        x_seq: (batch, seq_len, input_size)
        h0:    (batch, hidden_size)
        c0:    (batch, hidden_size)
        W_x, W_h, b: parameters

    Returns:
        h_seq: (batch, seq_len, hidden_size)
        h_T:   (batch, hidden_size)
        c_T:   (batch, hidden_size)
    """
    h, c = h0, c0
    h_seq = []
    for t in range(x_seq.shape[1]):
        h, c = lstm_step(x_seq[:, t, :], h, c, W_x, W_h, b)
        h_seq.append(h)
    h_seq = torch.stack(h_seq, dim=1)
    return h_seq, h, c


# ============================================================================
# LSTM Language Model
# ============================================================================

class LSTMLanguageModel(nn.Module):
    """Language model built from a handwritten LSTM.

    Architecture:
        token ids  →  embedding  →  LSTM (handwritten)  →  linear  →  logits
    """

    def __init__(self, vocab_size, embed_dim=64, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding   = nn.Embedding(vocab_size, embed_dim)
        # Single pair of big matrices covers all four gates
        self.W_x = nn.Parameter(torch.randn(embed_dim,    4 * hidden_size) * 0.01)
        self.W_h = nn.Parameter(torch.randn(hidden_size,  4 * hidden_size) * 0.01)
        self.b   = nn.Parameter(torch.zeros(4 * hidden_size))
        self.output_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h0=None, c0=None):
        """Forward pass.

        Args:
            x:  (batch, seq_len)
            h0, c0: initial states (zeros if None)

        Returns:
            logits: (batch, seq_len, vocab_size)
            h_T:    (batch, hidden_size)
            c_T:    (batch, hidden_size)
        """
        batch = x.shape[0]
        device = x.device
        if h0 is None:
            h0 = torch.zeros(batch, self.hidden_size, device=device)
        if c0 is None:
            c0 = torch.zeros(batch, self.hidden_size, device=device)

        emb = self.embedding(x)
        h_seq, h_T, c_T = lstm_forward(emb, h0, c0, self.W_x, self.W_h, self.b)
        logits = self.output_head(h_seq)
        return logits, h_T, c_T


# ============================================================================
# Training
# ============================================================================

def train_lstm(model, inputs, targets, n_epochs=10, batch_size=32,
               lr=1e-3, verbose=True):
    """Train an LSTMLanguageModel.

    Args:
        model:    LSTMLanguageModel instance
        inputs:   LongTensor (n, seq_len)
        targets:  LongTensor (n, seq_len)
        n_epochs: Number of passes
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
        perm = torch.randperm(n)
        inputs  = inputs[perm]
        targets = targets[perm]

        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, n, batch_size):
            xb = inputs[i : i + batch_size]
            yb = targets[i : i + batch_size]

            logits, _, _ = model(xb)
            batch_sz, seq_len, vocab = logits.shape
            loss = criterion(logits.view(batch_sz * seq_len, vocab),
                             yb.view(batch_sz * seq_len))

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
