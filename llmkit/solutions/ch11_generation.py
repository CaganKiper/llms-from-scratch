"""Chapter 11: Autoregressive Generation

Core concepts:
- Autoregressive loop: predict one token, append to input, repeat
- Greedy decoding: always pick the highest-probability token
- Temperature: scale logits before softmax to control randomness
- Top-k sampling: restrict sampling to the k most probable tokens
- Nucleus (top-p) sampling: restrict to the smallest set covering p probability mass

The model is the TransformerLanguageModel from Chapter 10.
"""

import torch
import torch.nn.functional as F


# ============================================================================
# Sampling Helpers
# ============================================================================

def sample_with_temperature(logits, temperature=1.0):
    """Sample a single token using temperature scaling.

    Divides logits by temperature before softmax:
    - temperature < 1: sharper distribution (more confident, less diverse)
    - temperature > 1: flatter distribution (more random, more diverse)
    - temperature = 1: unchanged softmax

    Args:
        logits:      (vocab_size,) — raw unnormalized scores
        temperature: float > 0

    Returns:
        token_id: int
    """
    scaled = logits / temperature
    probs  = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def top_k_sample(logits, k, temperature=1.0):
    """Sample from the top-k most probable tokens.

    All tokens outside the top-k are masked to -inf before sampling.

    Args:
        logits:      (vocab_size,)
        k:           Number of tokens to keep
        temperature: Temperature for scaling

    Returns:
        token_id: int
    """
    k = min(k, logits.shape[-1])
    # Find the k-th largest logit value
    top_k_values, _ = torch.topk(logits, k)
    threshold = top_k_values[-1]
    # Zero out everything below threshold
    filtered = logits.clone()
    filtered[filtered < threshold] = float('-inf')
    return sample_with_temperature(filtered, temperature)


def nucleus_sample(logits, p, temperature=1.0):
    """Nucleus (top-p) sampling.

    Keep the smallest set of tokens whose cumulative probability exceeds p.
    This adapts dynamically: sometimes 5 tokens cover 90%, sometimes 50 do.

    Args:
        logits:      (vocab_size,)
        p:           Cumulative probability threshold (0 < p <= 1)
        temperature: Temperature for scaling

    Returns:
        token_id: int
    """
    scaled = logits / temperature
    probs  = F.softmax(scaled, dim=-1)
    # Sort descending
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=0)
    # Keep tokens where cumulative probability has not yet exceeded p
    # (shift by 1 so we always keep at least the top token)
    remove_mask = cumulative - sorted_probs > p
    sorted_probs[remove_mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()   # renormalize
    sampled_idx = torch.multinomial(sorted_probs, num_samples=1).item()
    return sorted_indices[sampled_idx].item()


def greedy_decode(model, prompt_ids, max_new_tokens=50):
    """Generate tokens greedily (always pick argmax).

    Args:
        model:          TransformerLanguageModel (or any model with .forward(x))
        prompt_ids:     LongTensor (seq_len,) — initial context
        max_new_tokens: Number of new tokens to generate

    Returns:
        token_ids: LongTensor (seq_len + max_new_tokens,)
    """
    model.eval()
    ids = prompt_ids.clone().unsqueeze(0)   # (1, seq_len)
    max_ctx = getattr(model, 'max_seq_len', None)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Trim to model's maximum context window (Transformer only)
            ctx = ids[:, -max_ctx:] if max_ctx is not None else ids
            logits = model(ctx)[0]
            next_logits = logits[0, -1, :]          # last position
            next_id = next_logits.argmax(dim=-1)
            ids = torch.cat([ids, next_id.unsqueeze(0).unsqueeze(0)], dim=1)

    return ids[0]


# ============================================================================
# General-Purpose Generate
# ============================================================================

def generate(model, tokenizer, prompt, max_new_tokens=100,
             strategy='top_k', temperature=1.0, k=40, p=0.9):
    """Generate text from a prompt string.

    Args:
        model:          TransformerLanguageModel
        tokenizer:      Object with .encode(str) → list[int]
                        and .decode(list[int]) → str
        prompt:         Input string
        max_new_tokens: How many new tokens to produce
        strategy:       'greedy' | 'temperature' | 'top_k' | 'nucleus'
        temperature:    Sampling temperature (ignored for greedy)
        k:              Top-k value (used when strategy='top_k')
        p:              Nucleus probability mass (used when strategy='nucleus')

    Returns:
        generated_text: str — the full text including the prompt
    """
    model.eval()
    # Use encode_ids if available (returns ints), else fall back to encode
    encode_fn = getattr(tokenizer, 'encode_ids', tokenizer.encode)
    decode_fn = getattr(tokenizer, 'decode_ids', tokenizer.decode)

    prompt_ids = torch.tensor(encode_fn(prompt), dtype=torch.long)
    ids = prompt_ids.clone().unsqueeze(0)
    max_ctx = getattr(model, 'max_seq_len', None)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            ctx = ids[:, -max_ctx:] if max_ctx is not None else ids
            logits = model(ctx)[0]
            next_logits = logits[0, -1, :]

            if strategy == 'greedy':
                next_id = next_logits.argmax().item()
            elif strategy == 'temperature':
                next_id = sample_with_temperature(next_logits, temperature)
            elif strategy == 'top_k':
                next_id = top_k_sample(next_logits, k=k, temperature=temperature)
            elif strategy == 'nucleus':
                next_id = nucleus_sample(next_logits, p=p, temperature=temperature)
            else:
                raise ValueError(f"Unknown strategy: {strategy!r}")

            ids = torch.cat(
                [ids, torch.tensor([[next_id]], dtype=torch.long)], dim=1
            )

    token_ids = ids[0].tolist()
    return decode_fn(token_ids)
